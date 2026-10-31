"""Shot-based observable evaluation for simulators and quantum hardware."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Number
from typing import Any, Literal, cast

import numpy as np

from quspect.backends import (
    Backend,
    CloudBackend,
    ExecutionBatch,
    ExecutionJob,
    ExecutionRequest,
    JobReference,
    JobStatus,
)
from quspect.backends._exceptions import JobError
from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.ir.compilation import decompose_deep
from quspect.ir.qsal import dumps as dumps_qsal
from quspect.operators.qubit_operator import TermKey
from quspect.simulation.result import SimulationResult
from quspect.variational.adjoint import ParameterPoint, _coerce_point
from quspect.variational.objective import ObservableObjective
from quspect.variational.result import (
    FloatArray,
    ObjectiveEvaluation,
    ObjectiveSweep,
)

GroupingMode = Literal["qwc", "none"]
_SAMPLING_WORKLOAD_VERSION = 1


@dataclass(frozen=True)
class _ShiftOccurrence:
    instruction_index: int
    parameter_index: int
    gate_name: str
    chain: tuple[float, ...]
    shift: float = math.pi / 2
    multiplier: float = 0.5


@dataclass(frozen=True)
class _MeasurementGroup:
    basis: tuple[tuple[int, str], ...]
    term_indices: tuple[int, ...]
    measured_qubits: tuple[int, ...]


@dataclass(frozen=True)
class _SamplingPlan:
    circuit: QuantumCircuit
    parameter_order: tuple[Parameter, ...]
    observable_order: tuple[str, ...]
    terms: tuple[TermKey, ...]
    coefficients: np.ndarray
    identity_offsets: np.ndarray
    occurrences: tuple[_ShiftOccurrence, ...]
    chain_matrix: np.ndarray
    groups: tuple[_MeasurementGroup, ...]

    @property
    def variant_count(self) -> int:
        return 1 + 2 * len(self.occurrences)

    @property
    def experiment_count(self) -> int:
        return self.variant_count * len(self.groups)


class SamplingEngine:
    """Compile Pauli objectives into labelled hardware sampling requests.

    Unlike :class:`AdjointEngine`, this strategy obtains expectation values
    from measurement statistics and differentiates symbolic RX/RY/RZ gates
    with occurrence-level parameter shifts. It therefore works on both local
    simulators and batch-capable cloud QPUs.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        shots: int = 4096,
        grouping: GroupingMode = "qwc",
        execution_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(backend, Backend):
            raise TypeError(
                f"SamplingEngine.backend must be Backend, got {type(backend).__name__}."
            )
        if isinstance(shots, bool) or not isinstance(shots, int):
            raise TypeError("SamplingEngine.shots must be an integer.")
        if shots <= 0:
            raise ValueError("SamplingEngine.shots must be positive.")
        if grouping not in {"qwc", "none"}:
            raise ValueError("grouping must be 'qwc' or 'none'.")
        if not backend.capabilities.supports_sampling:
            raise ValueError(f"Backend {backend.name!r} does not support sampling.")

        options = dict(execution_options or {})
        forbidden = {"shots", "result_format"} & set(options)
        if forbidden:
            raise ValueError(
                "shots and result_format are controlled by SamplingEngine, "
                f"not execution_options: {sorted(forbidden)}."
            )
        self._backend = backend
        self._shots = shots
        self._grouping = grouping
        self._execution_options = options

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def shots(self) -> int:
        return self._shots

    def compile(self, objective: ObservableObjective) -> PreparedSamplingObjective:
        """Validate and snapshot an observable measurement plan."""
        if not isinstance(objective, ObservableObjective):
            raise TypeError(
                "SamplingEngine.compile expects ObservableObjective, "
                f"got {type(objective).__name__}."
            )
        plan = _compile_plan(objective, grouping=self._grouping)
        return PreparedSamplingObjective(
            plan,
            backend=self._backend,
            shots=self._shots,
            execution_options=self._execution_options,
        )


class PreparedSamplingObjective:
    """An immutable measurement plan reusable across optimizer iterations."""

    def __init__(
        self,
        plan: _SamplingPlan,
        *,
        backend: Backend,
        shots: int,
        execution_options: Mapping[str, Any],
    ) -> None:
        self._plan = plan
        self._backend = backend
        self._shots = shots
        self._execution_options = dict(execution_options)
        self._workload_id = _plan_workload_id(plan)

    @property
    def parameter_order(self) -> tuple[Parameter, ...]:
        return self._plan.parameter_order

    @property
    def observable_order(self) -> tuple[str, ...]:
        return self._plan.observable_order

    @property
    def num_measurement_groups(self) -> int:
        return len(self._plan.groups)

    @property
    def num_experiments_per_point(self) -> int:
        return self._plan.experiment_count

    @property
    def workload_id(self) -> str:
        """Stable fingerprint used to validate recovered execution jobs."""
        return self._workload_id

    @property
    def metadata(self) -> Mapping[str, str]:
        return {
            "backend": self._backend.name,
            "differentiation": "parameter-shift",
            "shots": str(self._shots),
            "measurement_groups": str(len(self._plan.groups)),
            "workload_id": self._workload_id,
        }

    def submit(self, point: ParameterPoint) -> ObjectiveJob:
        """Build all base/shift measurement circuits and submit one logical job."""
        bindings = _coerce_point(point, self._plan.parameter_order)
        if not self._plan.groups:
            evaluation = self._identity_evaluation()
            return ObjectiveJob(self._plan, immediate=evaluation, shots=self._shots)

        circuits, labels = _build_experiments(self._plan, bindings)
        request = ExecutionRequest.batch(
            circuits,
            labels=labels,
            shots=self._shots,
            result_format="auto",
            options=self._execution_options,
            metadata={
                "workload": "observable-objective",
                "differentiation": "parameter-shift",
                "workload_id": self._workload_id,
            },
        )
        execution_job = self._backend.submit_request(request)
        return ObjectiveJob(self._plan, execution_job=execution_job, shots=self._shots)

    def attach(self, job: ExecutionJob) -> ObjectiveJob:
        """Attach a recovered execution job to this compiled decoder."""
        if not isinstance(job, ExecutionJob):
            raise TypeError(f"attach expects ExecutionJob, got {type(job).__name__}.")
        self._validate_job_manifest(job)
        return ObjectiveJob(self._plan, execution_job=job, shots=self._shots)

    def evaluate(
        self,
        point: ParameterPoint,
        *,
        timeout: float | None = None,
    ) -> ObjectiveEvaluation:
        return self.submit(point).result(timeout=timeout)

    def evaluate_many(
        self,
        points: Sequence[ParameterPoint],
        *,
        timeout: float | None = None,
    ) -> ObjectiveSweep:
        """Evaluate several points through one flattened logical task."""
        return self.submit_many(points).result(timeout=timeout)

    def submit_many(
        self,
        points: Sequence[ParameterPoint],
    ) -> ObjectiveSweepJob:
        """Submit several optimizer points as one recoverable execution job."""
        point_items = tuple(points)
        bindings = tuple(
            _coerce_point(point, self._plan.parameter_order) for point in point_items
        )
        if not bindings:
            return ObjectiveSweepJob(
                self._plan,
                num_points=0,
                immediate=_objective_sweep((), self._plan, metadata=self.metadata),
                shots=self._shots,
                metadata=self.metadata,
            )
        elif not self._plan.groups:
            evaluations = tuple(self._identity_evaluation() for _ in bindings)
            return ObjectiveSweepJob(
                self._plan,
                num_points=len(bindings),
                immediate=_objective_sweep(
                    evaluations,
                    self._plan,
                    metadata=self.metadata,
                ),
                shots=self._shots,
                metadata=self.metadata,
            )

        circuits: list[QuantumCircuit] = []
        labels: list[str] = []
        for point_index, point_bindings in enumerate(bindings):
            point_circuits, point_labels = _build_experiments(
                self._plan,
                point_bindings,
            )
            for circuit, label in zip(point_circuits, point_labels, strict=True):
                qualified_label = f"point-{point_index}/{label}"
                circuits.append(circuit.copy(name=qualified_label))
                labels.append(qualified_label)

        request = ExecutionRequest.batch(
            circuits,
            labels=labels,
            shots=self._shots,
            result_format="auto",
            options=self._execution_options,
            metadata={
                "workload": "observable-objective-sweep",
                "differentiation": "parameter-shift",
                "points": len(bindings),
                "workload_id": self._workload_id,
            },
        )
        execution_job = self._backend.submit_request(request)
        return ObjectiveSweepJob(
            self._plan,
            num_points=len(bindings),
            execution_job=execution_job,
            shots=self._shots,
            metadata=self.metadata,
        )

    def attach_many(self, job: ExecutionJob) -> ObjectiveSweepJob:
        """Attach a recovered flattened sweep job and validate its manifest."""
        if not isinstance(job, ExecutionJob):
            raise TypeError(
                f"attach_many expects ExecutionJob, got {type(job).__name__}."
            )
        if self._plan.experiment_count <= 0:
            raise JobError("An identity-only sampling plan has no remote sweep job.")
        batch_size = job.reference.batch_size
        if batch_size is None or batch_size % self._plan.experiment_count:
            raise JobError(
                "Recovered job batch size is not a whole sampling sweep.",
                job_id=job.job_id,
            )
        num_points = batch_size // self._plan.experiment_count
        labels = _sweep_labels(self._plan, num_points)
        shapes = _experiment_shapes(self._plan) * num_points
        self._validate_job_manifest(job, labels=labels, shapes=shapes)
        return ObjectiveSweepJob(
            self._plan,
            num_points=num_points,
            execution_job=job,
            shots=self._shots,
            metadata=self.metadata,
        )

    def _validate_job_manifest(
        self,
        job: ExecutionJob,
        *,
        labels: tuple[str, ...] | None = None,
        shapes: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        reference = job.reference
        expected_labels = labels or _experiment_labels(self._plan)
        expected_shapes = shapes or _experiment_shapes(self._plan)
        mismatches: list[str] = []
        if reference.workload_id != self._workload_id:
            mismatches.append("workload fingerprint")
        if reference.backend_name != self._backend.name:
            mismatches.append("backend")
        expected_platform = (
            self._backend.platform_name
            if isinstance(self._backend, CloudBackend)
            else "local"
        )
        if reference.platform != expected_platform:
            mismatches.append("platform")
        if reference.batch_size != len(expected_labels):
            mismatches.append("batch size")
        if tuple(reference.labels) != expected_labels:
            mismatches.append("labels")
        if tuple(reference.experiment_shapes) != expected_shapes:
            mismatches.append("experiment shapes")
        if reference.shots != self._shots:
            mismatches.append("shots")
        if reference.result_format != "auto":
            mismatches.append("result format")
        if mismatches:
            raise JobError(
                "Recovered job does not match this sampling plan: "
                f"{', '.join(mismatches)}.",
                job_id=job.job_id,
            )

    def _identity_evaluation(self) -> ObjectiveEvaluation:
        return ObjectiveEvaluation(
            values=self._plan.identity_offsets,
            jacobian=np.zeros(
                (len(self.observable_order), len(self.parameter_order)),
                dtype=float,
            ),
            parameter_order=self.parameter_order,
            observable_order=self.observable_order,
            metadata=self.metadata,
        )


class ObjectiveJob:
    """Decode an execution job into values and a parameter-shift Jacobian."""

    def __init__(
        self,
        plan: _SamplingPlan,
        *,
        execution_job: ExecutionJob | None = None,
        immediate: ObjectiveEvaluation | None = None,
        shots: int,
    ) -> None:
        if (execution_job is None) == (immediate is None):
            raise ValueError("ObjectiveJob requires exactly one result source.")
        self._plan = plan
        self._execution_job = execution_job
        self._cached = immediate
        self._shots = shots

    def result(self, timeout: float | None = None) -> ObjectiveEvaluation:
        if self._cached is not None:
            return self._cached
        if self._execution_job is None:
            raise AssertionError("ObjectiveJob has no execution source.")
        raw = self._execution_job.result(timeout=timeout)
        raw = _coerce_batch_result(
            raw,
            expected_labels=_experiment_labels(self._plan),
            job_id=self.job_id,
        )
        self._cached = _decode_evaluation(
            self._plan,
            raw,
            shots=self._shots,
            job_id=self.job_id,
        )
        return self._cached

    def status(self) -> JobStatus:
        if self._execution_job is None:
            return JobStatus.COMPLETED
        return self._execution_job.status()

    def cancel(self) -> bool:
        if self._execution_job is None:
            return False
        return self._execution_job.cancel()

    @property
    def reference(self) -> JobReference | None:
        return (
            self._execution_job.reference if self._execution_job is not None else None
        )

    @property
    def job_id(self) -> str | None:
        return self._execution_job.job_id if self._execution_job is not None else None

    @property
    def job_ids(self) -> tuple[str, ...]:
        return self._execution_job.job_ids if self._execution_job is not None else ()


class ObjectiveSweepJob:
    """Decode one flattened execution job into a multi-point objective sweep."""

    def __init__(
        self,
        plan: _SamplingPlan,
        *,
        num_points: int,
        execution_job: ExecutionJob | None = None,
        immediate: ObjectiveSweep | None = None,
        shots: int,
        metadata: Mapping[str, str],
    ) -> None:
        if (execution_job is None) == (immediate is None):
            raise ValueError("ObjectiveSweepJob requires exactly one result source.")
        if num_points < 0:
            raise ValueError("ObjectiveSweepJob.num_points cannot be negative.")
        self._plan = plan
        self._num_points = num_points
        self._execution_job = execution_job
        self._cached = immediate
        self._shots = shots
        self._metadata = dict(metadata)

    @property
    def num_points(self) -> int:
        return self._num_points

    def result(self, timeout: float | None = None) -> ObjectiveSweep:
        if self._cached is not None:
            return self._cached
        if self._execution_job is None:
            raise AssertionError("ObjectiveSweepJob has no execution source.")
        labels = _sweep_labels(self._plan, self._num_points)
        raw = self._execution_job.result(timeout=timeout)
        batch = _coerce_batch_result(
            raw,
            expected_labels=labels,
            job_id=self.job_id,
        )
        evaluations = _decode_sweep_evaluations(
            self._plan,
            batch,
            num_points=self._num_points,
            shots=self._shots,
            job_id=self.job_id,
        )
        metadata = dict(self._metadata)
        if self.job_id:
            metadata["job_id"] = self.job_id
        self._cached = _objective_sweep(
            evaluations,
            self._plan,
            metadata=metadata,
        )
        return self._cached

    def status(self) -> JobStatus:
        if self._execution_job is None:
            return JobStatus.COMPLETED
        return self._execution_job.status()

    def cancel(self) -> bool:
        if self._execution_job is None:
            return False
        return self._execution_job.cancel()

    @property
    def reference(self) -> JobReference | None:
        return (
            self._execution_job.reference if self._execution_job is not None else None
        )

    @property
    def job_id(self) -> str | None:
        return self._execution_job.job_id if self._execution_job is not None else None

    @property
    def job_ids(self) -> tuple[str, ...]:
        return self._execution_job.job_ids if self._execution_job is not None else ()


def _compile_plan(
    objective: ObservableObjective,
    *,
    grouping: GroupingMode,
) -> _SamplingPlan:
    circuit = decompose_deep(
        objective.circuit.copy(),
        supported_gates={
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "cx",
            "cz",
            "swap",
            "ccx",
            "rx",
            "ry",
            "rz",
        },
    )
    if circuit.num_qubits == 0:
        raise ValueError("Sampling objective circuit must contain a qubit.")
    for instruction in circuit.instructions:
        if instruction.is_measure:
            raise ValueError("Sampling objective state preparation cannot measure.")
        if instruction.is_channel:
            raise ValueError(
                "Sampling objective state preparation cannot contain noise."
            )

    parameter_order = _parameter_order(circuit)
    occurrences = _shift_occurrences(circuit, parameter_order)
    observable_order = tuple(objective.observables)
    term_order: list[TermKey] = []
    term_index: dict[TermKey, int] = {}
    rows: list[dict[TermKey, float]] = []
    identity_offsets = np.zeros(len(observable_order), dtype=float)

    for observable_index, (name, operator) in enumerate(objective.observables.items()):
        if operator.is_zero:
            raise ValueError(f"Observable {name!r} cannot be the zero operator.")
        row: dict[TermKey, float] = {}
        for term, coefficient in operator:
            numeric = _real_coefficient(coefficient, observable_name=name)
            if not term:
                identity_offsets[observable_index] += numeric
                continue
            _validate_term(term, circuit.num_qubits, observable_name=name)
            if term not in term_index:
                term_index[term] = len(term_order)
                term_order.append(term)
            row[term] = row.get(term, 0.0) + numeric
        rows.append(row)

    coefficients = np.zeros((len(observable_order), len(term_order)), dtype=float)
    for observable_index, row in enumerate(rows):
        for term, coefficient in row.items():
            coefficients[observable_index, term_index[term]] = coefficient

    groups = _measurement_groups(tuple(term_order), grouping=grouping)
    chain_matrix = np.asarray(
        [occurrence.chain for occurrence in occurrences],
        dtype=float,
    ).reshape((len(occurrences), len(parameter_order)))
    for array in (coefficients, identity_offsets, chain_matrix):
        array.setflags(write=False)
    return _SamplingPlan(
        circuit=circuit,
        parameter_order=parameter_order,
        observable_order=observable_order,
        terms=tuple(term_order),
        coefficients=coefficients,
        identity_offsets=identity_offsets,
        occurrences=occurrences,
        chain_matrix=chain_matrix,
        groups=groups,
    )


def _plan_workload_id(plan: _SamplingPlan) -> str:
    """Hash every semantic part of a compiled sampling decoder."""
    circuit_document = json.loads(dumps_qsal(plan.circuit))
    payload = {
        "version": _SAMPLING_WORKLOAD_VERSION,
        "circuit": circuit_document["circuit"],
        "parameter_order": [parameter.name for parameter in plan.parameter_order],
        "observable_order": list(plan.observable_order),
        "terms": [[[qubit, pauli] for qubit, pauli in term] for term in plan.terms],
        "coefficients": plan.coefficients.tolist(),
        "identity_offsets": plan.identity_offsets.tolist(),
        "occurrences": [
            {
                "instruction_index": occurrence.instruction_index,
                "parameter_index": occurrence.parameter_index,
                "gate_name": occurrence.gate_name,
                "chain": list(occurrence.chain),
                "shift": occurrence.shift,
                "multiplier": occurrence.multiplier,
            }
            for occurrence in plan.occurrences
        ],
        "groups": [
            {
                "basis": [[qubit, pauli] for qubit, pauli in group.basis],
                "term_indices": list(group.term_indices),
                "measured_qubits": list(group.measured_qubits),
            }
            for group in plan.groups
        ],
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sampling-v{_SAMPLING_WORKLOAD_VERSION}:{digest}"


def _parameter_order(circuit: QuantumCircuit) -> tuple[Parameter, ...]:
    ordered: list[Parameter] = []
    seen: set[Parameter] = set()
    for instruction in circuit.instructions:
        if not instruction.is_gate or instruction.gate is None:
            continue
        for value in instruction.gate.params:
            parameters: Sequence[Parameter]
            if isinstance(value, Parameter):
                parameters = (value,)
            elif isinstance(value, ParameterExpression):
                parameters = tuple(value.terms)
            else:
                parameters = ()
            for parameter in parameters:
                if parameter not in seen:
                    seen.add(parameter)
                    ordered.append(parameter)
    return tuple(ordered)


def _shift_occurrences(
    circuit: QuantumCircuit,
    parameter_order: tuple[Parameter, ...],
) -> tuple[_ShiftOccurrence, ...]:
    parameter_index = {
        parameter: index for index, parameter in enumerate(parameter_order)
    }
    occurrences: list[_ShiftOccurrence] = []
    for instruction_index, instruction in enumerate(circuit.instructions):
        if not instruction.is_gate or instruction.gate is None:
            continue
        gate = instruction.gate
        for gate_parameter_index, value in enumerate(gate.params):
            if isinstance(value, Parameter):
                terms = {value: 1.0}
                constant = 0.0
            elif isinstance(value, ParameterExpression) and not value.is_constant:
                terms = dict(value.terms)
                constant = value.constant
            else:
                if isinstance(value, ParameterExpression) and not math.isfinite(
                    value.constant
                ):
                    raise ValueError(f"Gate {gate.name!r} has a non-finite constant.")
                continue

            if gate.name.lower() not in {"rx", "ry", "rz"} or gate_parameter_index != 0:
                raise ValueError(
                    "Sampling parameter-shift currently supports symbolic RX, RY, "
                    f"and RZ only; got {gate.name!r}."
                )
            if not math.isfinite(constant):
                raise ValueError(f"Gate {gate.name!r} has a non-finite offset.")
            chain = [0.0] * len(parameter_order)
            for parameter, raw_coefficient in terms.items():
                coefficient = float(raw_coefficient)
                if not math.isfinite(coefficient):
                    raise ValueError(
                        f"Gate {gate.name!r} has a non-finite affine coefficient."
                    )
                chain[parameter_index[parameter]] = coefficient
            occurrences.append(
                _ShiftOccurrence(
                    instruction_index=instruction_index,
                    parameter_index=gate_parameter_index,
                    gate_name=gate.name.lower(),
                    chain=tuple(chain),
                )
            )
    return tuple(occurrences)


def _real_coefficient(value: object, *, observable_name: str) -> float:
    if isinstance(value, (Parameter, ParameterExpression)):
        raise ValueError(
            f"Observable {observable_name!r} has a symbolic coefficient; "
            "sampled objectives currently require fixed coefficients."
        )
    if not isinstance(value, Number):
        raise TypeError(f"Observable {observable_name!r} coefficient is not numeric.")
    numeric = complex(value)  # type: ignore[call-overload]
    if not math.isfinite(numeric.real) or not math.isfinite(numeric.imag):
        raise ValueError(f"Observable {observable_name!r} coefficient must be finite.")
    if abs(numeric.imag) > 1e-14:
        raise ValueError(f"Observable {observable_name!r} coefficient must be real.")
    return float(numeric.real)


def _validate_term(term: TermKey, num_qubits: int, *, observable_name: str) -> None:
    # Defensive: ``QubitOperator`` already rejects non-Pauli letters and
    # reduces repeated qubits with the Pauli algebra at construction, so the
    # "repeats" and "invalid Pauli" branches are unreachable through the
    # public API.  They are kept to guard the measurement-basis decoding
    # against terms injected by other means.
    seen: set[int] = set()
    for qubit, pauli in term:
        if qubit in seen:
            raise ValueError(
                f"Observable {observable_name!r} repeats qubit {qubit} in one term."
            )
        seen.add(qubit)
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError(
                f"Observable {observable_name!r} uses qubit {qubit} "
                "outside the circuit."
            )
        if pauli.upper() not in {"X", "Y", "Z"}:
            raise ValueError(
                f"Observable {observable_name!r} contains invalid Pauli {pauli!r}."
            )


def _measurement_groups(
    terms: tuple[TermKey, ...],
    *,
    grouping: GroupingMode,
) -> tuple[_MeasurementGroup, ...]:
    if grouping == "none":
        return tuple(
            _group_from_indices(terms, (index,)) for index in range(len(terms))
        )

    grouped: list[tuple[dict[int, str], list[int]]] = []
    ordered_indices = sorted(
        range(len(terms)),
        key=lambda index: (-len(terms[index]), terms[index]),
    )
    for index in ordered_indices:
        term_basis = {qubit: pauli.upper() for qubit, pauli in terms[index]}
        for basis, indices in grouped:
            if all(
                basis.get(qubit, pauli) == pauli for qubit, pauli in term_basis.items()
            ):
                basis.update(term_basis)
                indices.append(index)
                break
        else:
            grouped.append((dict(term_basis), [index]))
    return tuple(
        _MeasurementGroup(
            basis=tuple(sorted(basis.items())),
            term_indices=tuple(indices),
            measured_qubits=tuple(sorted(basis)),
        )
        for basis, indices in grouped
    )


def _group_from_indices(
    terms: tuple[TermKey, ...], indices: tuple[int, ...]
) -> _MeasurementGroup:
    basis = {qubit: pauli.upper() for index in indices for qubit, pauli in terms[index]}
    return _MeasurementGroup(
        basis=tuple(sorted(basis.items())),
        term_indices=indices,
        measured_qubits=tuple(sorted(basis)),
    )


def _build_experiments(
    plan: _SamplingPlan,
    bindings: Mapping[str, float],
) -> tuple[tuple[QuantumCircuit, ...], tuple[str, ...]]:
    parameter_bindings = {
        parameter: bindings[parameter.name] for parameter in plan.parameter_order
    }
    numeric_base = plan.circuit.bind_parameters(parameter_bindings)
    variants: list[tuple[str, QuantumCircuit]] = [("base", numeric_base)]
    for index, occurrence in enumerate(plan.occurrences):
        variants.append(
            (
                f"occ-{index}/plus",
                _shift_circuit(numeric_base, occurrence, +occurrence.shift),
            )
        )
        variants.append(
            (
                f"occ-{index}/minus",
                _shift_circuit(numeric_base, occurrence, -occurrence.shift),
            )
        )

    circuits: list[QuantumCircuit] = []
    labels: list[str] = []
    for variant_label, variant in variants:
        for group_index, group in enumerate(plan.groups):
            label = f"{variant_label}/group-{group_index}"
            circuits.append(_measurement_circuit(variant, group, label=label))
            labels.append(label)
    return tuple(circuits), tuple(labels)


def _shift_circuit(
    numeric_base: QuantumCircuit,
    occurrence: _ShiftOccurrence,
    delta: float,
) -> QuantumCircuit:
    instructions = list(numeric_base.instructions)
    instruction = instructions[occurrence.instruction_index]
    if instruction.gate is None:
        raise AssertionError("Shift occurrence no longer points to a gate.")
    parameters = list(instruction.gate.params)
    parameters[occurrence.parameter_index] = (
        float(parameters[occurrence.parameter_index]) + delta
    )
    shifted_gate = replace(instruction.gate, params=tuple(parameters))
    instructions[occurrence.instruction_index] = replace(
        instruction,
        gate=shifted_gate,
    )
    return QuantumCircuit._from_instruction_snapshot(
        instructions,
        name=numeric_base.name,
    )


def _measurement_circuit(
    state_preparation: QuantumCircuit,
    group: _MeasurementGroup,
    *,
    label: str,
) -> QuantumCircuit:
    circuit = state_preparation.copy(name=label)
    for qubit, basis in group.basis:
        if basis == "X":
            circuit.h(qubit)
        elif basis == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
    circuit.barrier(*group.measured_qubits)
    circuit.measure(group.measured_qubits, tuple(range(len(group.measured_qubits))))
    return circuit


def _experiment_labels(plan: _SamplingPlan) -> tuple[str, ...]:
    labels = [f"base/group-{index}" for index in range(len(plan.groups))]
    for occurrence_index in range(len(plan.occurrences)):
        for sign in ("plus", "minus"):
            labels.extend(
                f"occ-{occurrence_index}/{sign}/group-{group_index}"
                for group_index in range(len(plan.groups))
            )
    return tuple(labels)


def _experiment_shapes(plan: _SamplingPlan) -> tuple[tuple[int, int], ...]:
    return tuple(
        (plan.circuit.num_qubits, len(group.measured_qubits))
        for _ in range(plan.variant_count)
        for group in plan.groups
    )


def _sweep_labels(plan: _SamplingPlan, num_points: int) -> tuple[str, ...]:
    point_labels = _experiment_labels(plan)
    return tuple(
        f"point-{point_index}/{label}"
        for point_index in range(num_points)
        for label in point_labels
    )


def _decode_evaluation(
    plan: _SamplingPlan,
    batch: ExecutionBatch,
    *,
    shots: int,
    job_id: str | None,
) -> ObjectiveEvaluation:
    expected_labels = _experiment_labels(plan)
    if len(batch) != len(expected_labels):
        raise JobError(
            f"Sampling objective expected {len(expected_labels)} results, "
            f"got {len(batch)}.",
            job_id=job_id,
        )
    actual_labels = tuple(batch.labels)
    if actual_labels != expected_labels:
        expected_set = set(expected_labels)
        actual_set = set(actual_labels)
        if actual_set == expected_set:
            detail = "labels are out of order"
        else:
            missing = sorted(expected_set - actual_set)
            unexpected = sorted(actual_set - expected_set)
            detail = f"missing={missing}, unexpected={unexpected}"
        raise JobError(
            "Sampling objective result labels do not match the submitted "
            f"manifest ({detail}).",
            job_id=job_id,
        )

    for result_index, (label, result) in enumerate(
        zip(expected_labels, batch.results, strict=True)
    ):
        group = plan.groups[result_index % len(plan.groups)]
        expected_shape = (plan.circuit.num_qubits, len(group.measured_qubits))
        actual_shape = (result.num_qubits, result.num_clbits)
        if actual_shape != expected_shape:
            raise JobError(
                f"Sampling experiment {label!r} returned shape {actual_shape}, "
                f"expected {expected_shape}.",
                job_id=job_id,
            )

    failed = [
        label
        for label, result in zip(expected_labels, batch.results, strict=True)
        if not result.success
    ]
    if failed:
        raise JobError(
            f"Sampling objective contains failed experiment(s): {failed}.",
            job_id=job_id,
        )

    term_values = np.empty((plan.variant_count, len(plan.terms)), dtype=float)
    term_values.fill(np.nan)
    for variant_index in range(plan.variant_count):
        variant_prefix = (
            "base" if variant_index == 0 else _variant_prefix(variant_index)
        )
        for group_index, group in enumerate(plan.groups):
            label = f"{variant_prefix}/group-{group_index}"
            result_index = variant_index * len(plan.groups) + group_index
            result = batch.results[result_index]
            try:
                decoded = _decode_group(plan, group, result)
            except JobError as exc:
                raise JobError(
                    f"Sampling experiment {label!r} could not be decoded: {exc}",
                    error_type=exc.error_type,
                    job_id=job_id,
                ) from exc
            for term_index, value in decoded.items():
                term_values[variant_index, term_index] = value
    if np.isnan(term_values).any():
        raise JobError("Sampling result did not cover every Pauli term.", job_id=job_id)

    values = plan.identity_offsets + plan.coefficients @ term_values[0]
    local = np.zeros(
        (len(plan.observable_order), len(plan.occurrences)),
        dtype=float,
    )
    for occurrence_index, occurrence in enumerate(plan.occurrences):
        plus_index = 1 + 2 * occurrence_index
        minus_index = plus_index + 1
        local[:, occurrence_index] = occurrence.multiplier * (
            plan.coefficients @ (term_values[plus_index] - term_values[minus_index])
        )
    jacobian = local @ plan.chain_matrix
    metadata = {
        "backend": batch.backend_name,
        "differentiation": "parameter-shift",
        "shots": str(shots),
        "measurement_groups": str(len(plan.groups)),
        "experiments": str(len(expected_labels)),
    }
    if job_id:
        metadata["job_id"] = job_id
    return ObjectiveEvaluation(
        values=values,
        jacobian=jacobian,
        parameter_order=plan.parameter_order,
        observable_order=plan.observable_order,
        metadata=metadata,
    )


def _coerce_batch_result(
    raw: SimulationResult | ExecutionBatch,
    *,
    expected_labels: tuple[str, ...],
    job_id: str | None,
) -> ExecutionBatch:
    if isinstance(raw, ExecutionBatch):
        return raw
    if len(expected_labels) != 1:
        raise JobError(
            "Sampling objective expected a batch but received one result.",
            job_id=job_id,
        )
    return ExecutionBatch(
        results=(raw,),
        labels=expected_labels,
        backend_name=str(raw.metadata.get("backend", "")),
        job_id=job_id,
    )


def _decode_sweep_evaluations(
    plan: _SamplingPlan,
    batch: ExecutionBatch,
    *,
    num_points: int,
    shots: int,
    job_id: str | None,
) -> list[ObjectiveEvaluation]:
    point_labels = _experiment_labels(plan)
    expected_labels = _sweep_labels(plan, num_points)
    if len(batch) != len(expected_labels):
        raise JobError(
            f"Sampling sweep expected {len(expected_labels)} results, "
            f"got {len(batch)}.",
            job_id=job_id,
        )
    if tuple(batch.labels) != expected_labels:
        expected_set = set(expected_labels)
        actual_set = set(batch.labels)
        if actual_set == expected_set:
            detail = "labels are out of order"
        else:
            missing = sorted(expected_set - actual_set)
            unexpected = sorted(actual_set - expected_set)
            detail = f"missing={missing}, unexpected={unexpected}"
        raise JobError(
            f"Sampling sweep result labels do not match its manifest ({detail}).",
            job_id=job_id,
        )

    point_size = plan.experiment_count
    evaluations: list[ObjectiveEvaluation] = []
    for point_index in range(num_points):
        start = point_index * point_size
        stop = start + point_size
        point_batch = ExecutionBatch(
            results=batch.results[start:stop],
            labels=point_labels,
            backend_name=batch.backend_name,
            job_id=batch.job_id or job_id,
            metadata=batch.metadata,
        )
        evaluations.append(
            _decode_evaluation(
                plan,
                point_batch,
                shots=shots,
                job_id=job_id,
            )
        )
    return evaluations


def _objective_sweep(
    evaluations: Sequence[ObjectiveEvaluation],
    plan: _SamplingPlan,
    *,
    metadata: Mapping[str, str],
) -> ObjectiveSweep:
    if evaluations:
        values = np.stack([evaluation.values for evaluation in evaluations])
        jacobian = np.stack([evaluation.jacobian for evaluation in evaluations])
    else:
        values = np.empty((0, len(plan.observable_order)), dtype=float)
        jacobian = np.empty(
            (0, len(plan.observable_order), len(plan.parameter_order)),
            dtype=float,
        )
    return ObjectiveSweep(
        values=cast(FloatArray, values),
        jacobian=cast(FloatArray, jacobian),
        parameter_order=plan.parameter_order,
        observable_order=plan.observable_order,
        metadata=metadata,
    )


def _variant_prefix(variant_index: int) -> str:
    occurrence_index = (variant_index - 1) // 2
    sign = "plus" if (variant_index - 1) % 2 == 0 else "minus"
    return f"occ-{occurrence_index}/{sign}"


def _decode_group(
    plan: _SamplingPlan,
    group: _MeasurementGroup,
    result: SimulationResult,
) -> dict[int, float]:
    width = len(group.measured_qubits)
    distribution: dict[str, float]
    if result.probabilities:
        distribution = {
            str(bitstring): float(probability)
            for bitstring, probability in result.probabilities.items()
        }
    else:
        total_counts = sum(result.counts.values())
        if total_counts <= 0:
            raise JobError("Sampling experiment returned no measurement shots.")
        distribution = {
            str(bitstring): count / total_counts
            for bitstring, count in result.counts.items()
        }

    total = 0.0
    for bitstring, probability in distribution.items():
        if len(bitstring) != width or set(bitstring) - {"0", "1"}:
            raise JobError(f"Invalid {width}-bit sampling outcome {bitstring!r}.")
        if not math.isfinite(probability):
            raise JobError(f"Outcome {bitstring!r} has non-finite probability.")
        total += probability
    if not math.isfinite(total) or abs(total) < 1e-15:
        raise JobError("Sampling probabilities have zero total weight.")

    qubit_to_clbit = {qubit: clbit for clbit, qubit in enumerate(group.measured_qubits)}
    decoded: dict[int, float] = {}
    for term_index in group.term_indices:
        term = plan.terms[term_index]
        expectation = 0.0
        for bitstring, probability in distribution.items():
            parity = sum(bitstring[qubit_to_clbit[qubit]] == "1" for qubit, _ in term)
            eigenvalue = -1.0 if parity % 2 else 1.0
            expectation += eigenvalue * probability / total
        decoded[term_index] = expectation
    return decoded
