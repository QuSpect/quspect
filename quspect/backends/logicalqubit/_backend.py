"""LogicalQubit cloud backend with native batch sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quspect.backends._base import CloudBackend, CloudJob, CompositeJob
from quspect.backends._exceptions import (
    BackendError,
    PartialSubmissionError,
    TranslationError,
)
from quspect.backends._execution import (
    BackendCapabilities,
    ExecutionBatch,
    ExecutionRequest,
    JobReference,
)
from quspect.backends.logicalqubit._translator import LQCircuitTranslator
from quspect.ir.qsal import dumps

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.simulation.result import SimulationResult

_MAX_BATCH_CIRCUITS = 64
_MAX_SHOTS = 50_000
_SUPPORTED_PROVIDER_OPTIONS = {
    "initial_layout",
    "physical_qubits",
    "readout_correction",
    "dynamic_decoupling",
    "provider_batching",
}

# SABRE models pairwise coupling edges, so higher-arity gates must be lowered
# before topology routing even when provider metadata claims they are native
# or omits a native gate set altogether.  Keep this list deliberately narrow:
# the LogicalQubit translator can still accept CCX when transpilation is
# explicitly disabled.
_PAIRWISE_ROUTING_DECOMPOSITION_GATES = frozenset({"ccx"})


def _workload_id(request: ExecutionRequest) -> str | None:
    value = request.metadata.get("workload_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Execution metadata workload_id must be a non-empty string.")
    return value.strip()


def _decompose_for_pairwise_routing(circuit: QuantumCircuit) -> QuantumCircuit:
    """Lower gates that cannot be represented by pairwise topology routing.

    This is intentionally a single, targeted decomposition pass.  Provider
    metadata may omit ``native_gates`` (as QZ01-repetition_code currently
    does), but CCX must still be expanded before SABRE sees the circuit.
    Other gates remain available for the provider-specific decomposition pass
    or the LogicalQubit translator.
    """
    from quspect.ir.compilation import decompose_circuit

    gate_names = {
        instruction.gate.name
        for instruction in circuit
        if instruction.is_gate and instruction.gate is not None
    }
    selected = gate_names & _PAIRWISE_ROUTING_DECOMPOSITION_GATES
    if not selected:
        return circuit
    return decompose_circuit(circuit, supported_gates=gate_names - selected)


@dataclass(frozen=True)
class _PreparedExperiment:
    circuit: Any
    label: str
    shape: tuple[int, int]
    layout: tuple[int, ...] | None


class LogicalQubitBackend(CloudBackend):
    """A LogicalQubit QPU exposed as a batch-capable sampling backend."""

    def __init__(self, lq_backend: Any, *, provider: Any = None) -> None:
        self._lq_backend = lq_backend
        self._provider = provider
        self._translator = LQCircuitTranslator()
        self._config_cache: dict[str, Any] | None = None

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        transpile: bool = True,
        initial_layout: Sequence[int] | None = None,
        physical_qubits: Sequence[int] | None = None,
        timeout: float | None = None,
        poll_interval: float = 2.0,
        **options: Any,
    ) -> SimulationResult:
        """Submit one circuit and block until its provider result is ready.

        Args:
            circuit: Bound circuit containing terminal measurements.
            shots: Number of samples.
            transpile: Whether QuSpect should decompose, optimize, and route.
            initial_layout: Optional starting logical-to-physical mapping.
                With transpilation disabled this is passed through directly.
            physical_qubits: Optional physical-qubit pool that constrains
                mapping and routing. Requires ``transpile=True``.
            timeout: Maximum result wait in seconds.
            poll_interval: Provider polling interval in seconds.
            **options: Additional LogicalQubit provider options.
        """
        job = self.submit(
            circuit,
            shots=shots,
            transpile=transpile,
            initial_layout=initial_layout,
            physical_qubits=physical_qubits,
            **options,
        )
        result = job.result(timeout=timeout, poll_interval=poll_interval)
        if isinstance(result, ExecutionBatch):
            return result.single()
        return result

    @property
    def name(self) -> str:
        return str(self._lq_backend.name)

    def submit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        transpile: bool = True,
        initial_layout: Sequence[int] | None = None,
        physical_qubits: Sequence[int] | None = None,
        **options: Any,
    ) -> CloudJob:
        """Submit one circuit and return a cloud job.

        Args:
            circuit: Bound circuit containing terminal measurements.
            shots: Number of samples.
            transpile: Whether QuSpect should decompose, optimize, and route.
            initial_layout: Optional starting mapping where
                ``initial_layout[logical] = physical``. With transpilation
                disabled it is passed directly to the provider.
            physical_qubits: Optional ordered pool of physical device qubits.
                Compilation and inserted SWAPs are restricted to the induced
                coupling subgraph. Requires ``transpile=True``.
            **options: Additional LogicalQubit provider options.
        """
        from quspect.circuit.circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                "LogicalQubitBackend.submit expects QuantumCircuit; use "
                "submit_request() for labelled batches."
            )
        if circuit.parameters:
            names = sorted(parameter.name for parameter in circuit.parameters)
            raise TranslationError(
                f"Circuit contains unbound parameter(s): {names}. "
                "Bind them or use ExecutionRequest.sweep()."
            )

        result_format = options.pop("result_format", "auto")
        compile_options: dict[str, Any] = {"transpile": transpile, **options}
        if initial_layout is not None:
            compile_options["initial_layout"] = list(initial_layout)
        if physical_qubits is not None:
            compile_options["physical_qubits"] = list(physical_qubits)

        request = ExecutionRequest.single(
            circuit,
            label=circuit.name or "circuit",
            shots=shots,
            result_format=result_format,
            options=compile_options,
        )
        prepared, provider_options = self._prepare(request)
        provider_options.pop("provider_batching", None)
        if len(prepared) != 1:
            raise AssertionError("Single submission prepared more than one experiment.")
        item = prepared[0]
        provider_options = dict(provider_options)
        if item.layout is not None and "initial_layout" not in provider_options:
            provider_options["initial_layout"] = list(item.layout)
        raw_job = self._run_sdk(
            item.circuit, request.shots, request.result_format, provider_options
        )
        reference = self._make_reference(
            raw_job,
            batch_size=None,
            labels=(item.label,),
            shapes=(item.shape,),
            shots=request.shots,
            result_format=request.result_format,
            workload_id=_workload_id(request),
        )
        from quspect.backends.logicalqubit._job import LogicalQubitJob

        return LogicalQubitJob(raw_job, reference=reference, return_batch=False)

    def submit_request(self, request: ExecutionRequest) -> CloudJob:
        """Submit an ordered request, using native batches of at most 64 circuits.

        Request validation (result format, ``metadata["workload_id"]``,
        translation) happens before any provider job is created.

        Raises:
            TypeError: If ``request`` is not an ``ExecutionRequest``.
            ValueError: If the result format is unsupported or
                ``metadata["workload_id"]`` is not a non-empty string.
        """
        if not isinstance(request, ExecutionRequest):
            raise TypeError(
                "LogicalQubitBackend.submit_request expects ExecutionRequest, "
                f"got {type(request).__name__}."
            )
        if request.result_format not in self.capabilities.result_formats:
            raise ValueError(
                f"LogicalQubit does not support result format "
                f"{request.result_format!r}."
            )
        # Validate request metadata before any provider job is created, so an
        # invalid workload_id can never leave an orphaned remote job behind.
        workload_id = _workload_id(request)
        prepared, provider_options = self._prepare(request)
        provider_batching = provider_options.pop("provider_batching", "native")
        groups = self._partition(
            prepared,
            provider_options.get("initial_layout"),
            max_group_size=(1 if provider_batching == "individual" else None),
        )

        jobs: list[CloudJob] = []
        try:
            offset = 0
            for group, effective_layout in groups:
                group_options = dict(provider_options)
                if effective_layout is not None:
                    group_options["initial_layout"] = list(effective_layout)
                circuits = [item.circuit for item in group]
                program = circuits[0] if provider_batching == "individual" else circuits
                raw_job = self._run_sdk(
                    program,
                    request.shots,
                    request.result_format,
                    group_options,
                )
                labels = tuple(item.label for item in group)
                shapes = tuple(item.shape for item in group)
                reference = self._make_reference(
                    raw_job,
                    batch_size=len(group),
                    labels=labels,
                    shapes=shapes,
                    shots=request.shots,
                    result_format=request.result_format,
                    workload_id=workload_id,
                )
                from quspect.backends.logicalqubit._job import LogicalQubitJob

                jobs.append(
                    LogicalQubitJob(raw_job, reference=reference, return_batch=True)
                )
                offset += len(group)
        except Exception as exc:
            cancellation_failures: list[str] = []
            for submitted in jobs:
                try:
                    if not submitted.cancel():
                        cancellation_failures.append(submitted.job_id)
                except Exception:
                    cancellation_failures.append(submitted.job_id)
            if jobs:
                raise PartialSubmissionError(
                    "LogicalQubit batch submission failed after "
                    f"{offset} experiment(s); submitted job IDs are available "
                    f"on the exception: {exc}",
                    submitted_references=tuple(job.reference for job in jobs),
                    cancellation_failures=cancellation_failures,
                ) from exc
            if isinstance(exc, (BackendError, TranslationError)):
                raise
            raise BackendError(
                "Failed to submit LogicalQubit batch after "
                f"{offset} experiment(s): {exc}"
            ) from exc

        if len(jobs) == 1:
            return jobs[0]
        return CompositeJob(jobs)

    def _prepare(
        self, request: ExecutionRequest
    ) -> tuple[list[_PreparedExperiment], dict[str, Any]]:
        """Bind before transpilation, then translate each experiment."""
        self._validate_shots(request.shots)
        options = dict(request.options)
        transpile = options.pop("transpile", True)
        if not isinstance(transpile, bool):
            raise TypeError("LogicalQubit option 'transpile' must be bool.")
        unknown = sorted(set(options) - _SUPPORTED_PROVIDER_OPTIONS)
        if unknown:
            raise ValueError(f"Unsupported LogicalQubit option(s): {unknown}.")

        explicit_layout = options.get("initial_layout")
        if explicit_layout is not None:
            if not isinstance(explicit_layout, (list, tuple)) or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in explicit_layout
            ):
                raise TypeError("initial_layout must be a sequence of integer qubits.")
            options["initial_layout"] = list(explicit_layout)

        physical_qubits = options.pop("physical_qubits", None)
        if physical_qubits is not None:
            if not isinstance(physical_qubits, (list, tuple)) or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in physical_qubits
            ):
                raise TypeError("physical_qubits must be a sequence of integer qubits.")
            physical_qubits = list(physical_qubits)
            if not transpile:
                raise ValueError("physical_qubits requires transpile=True.")

        # With transpilation enabled, initial_layout seeds SABRE rather than
        # being forwarded directly to the provider.  The routed circuit may
        # acquire workspace wires, so its effective layout is produced by
        # _transpile() and attached to the prepared experiment below.
        if transpile:
            options.pop("initial_layout", None)

        provider_batching = options.get("provider_batching", "native")
        if provider_batching not in {"native", "individual"}:
            raise ValueError("provider_batching must be 'native' or 'individual'.")

        prepared: list[_PreparedExperiment] = []
        for circuit, label, shape in zip(
            request.bound_circuits(),
            request.labels,
            request.experiment_shapes,
            strict=True,
        ):
            if not any(instruction.is_measure for instruction in circuit.instructions):
                raise TranslationError(
                    f"Experiment {label!r} has no measurement. "
                    "A sampling request sent to quantum hardware must measure."
                )

            layout: list[int] | None = None
            circuit_to_run = circuit
            if transpile:
                try:
                    circuit_to_run, layout = self._transpile(
                        circuit,
                        initial_layout=explicit_layout,
                        physical_qubits=physical_qubits,
                    )
                except Exception as exc:
                    raise TranslationError(
                        f"Failed to transpile experiment {label!r}: {exc}"
                    ) from exc

            try:
                lq_circuit = self._translator.translate(dumps(circuit_to_run))
            except TranslationError:
                raise
            except Exception as exc:
                raise TranslationError(
                    f"Failed to translate experiment {label!r}: {exc}"
                ) from exc

            normalized_layout = None
            if layout and len(layout) == circuit_to_run.num_qubits:
                normalized_layout = tuple(int(value) for value in layout)
            prepared.append(
                _PreparedExperiment(
                    circuit=lq_circuit,
                    label=label,
                    shape=shape,
                    layout=normalized_layout,
                )
            )
        return prepared, options

    @staticmethod
    def _partition(
        prepared: Sequence[_PreparedExperiment],
        explicit_layout: Sequence[int] | None,
        *,
        max_group_size: int | None = None,
    ) -> list[tuple[tuple[_PreparedExperiment, ...], tuple[int, ...] | None]]:
        """Partition by provider batch limit and shared-layout constraint."""
        groups: list[
            tuple[tuple[_PreparedExperiment, ...], tuple[int, ...] | None]
        ] = []
        current: list[_PreparedExperiment] = []
        current_layout: tuple[int, ...] | None = None
        forced_layout = tuple(explicit_layout) if explicit_layout is not None else None
        group_limit = max_group_size or _MAX_BATCH_CIRCUITS

        for item in prepared:
            item_layout = forced_layout if forced_layout is not None else item.layout
            if current and (
                len(current) >= group_limit or item_layout != current_layout
            ):
                groups.append((tuple(current), current_layout))
                current = []
            if not current:
                current_layout = item_layout
            current.append(item)
        if current:
            groups.append((tuple(current), current_layout))
        return groups

    def _run_sdk(
        self,
        program: Any,
        shots: int,
        result_format: str,
        options: dict[str, Any],
    ) -> Any:
        try:
            submission_options = {"shots": shots, **options}
            if result_format != "auto":
                submission_options["result_format"] = result_format
            return self._lq_backend.run(program, **submission_options)
        except Exception as exc:
            raise BackendError(
                f"Failed to submit to LogicalQubit backend {self.name!r}: {exc}"
            ) from exc

    def _make_reference(
        self,
        raw_job: Any,
        *,
        batch_size: int | None,
        labels: tuple[str, ...],
        shapes: tuple[tuple[int, int], ...],
        shots: int,
        result_format: str,
        workload_id: str | None,
    ) -> JobReference:
        children = tuple(getattr(raw_job, "_children", ()))
        child_job_ids = tuple(
            str(child.job_id) for child in children if getattr(child, "job_id", None)
        )
        return JobReference(
            platform="logicalqubit",
            backend_name=self.name,
            job_id=str(raw_job.job_id),
            base_url=getattr(raw_job, "base_url", None)
            or getattr(self._lq_backend, "base_url", None),
            transport=_job_transport(raw_job, children),
            batch_size=batch_size,
            labels=labels,
            experiment_shapes=shapes,
            shots=shots,
            result_format=result_format,
            workload_id=workload_id,
            child_job_ids=child_job_ids,
            child_batch_sizes=(1,) * len(child_job_ids),
        )

    @staticmethod
    def _validate_shots(shots: int) -> None:
        if isinstance(shots, bool) or not isinstance(shots, int):
            raise TypeError("shots must be an integer.")
        if shots < 1 or shots > _MAX_SHOTS:
            raise ValueError(f"shots must be in [1, {_MAX_SHOTS}], got {shots}.")

    def _transpile(
        self,
        circuit: QuantumCircuit,
        *,
        initial_layout: Sequence[int] | None = None,
        physical_qubits: Sequence[int] | None = None,
    ) -> tuple[QuantumCircuit, list[int] | None]:
        """Decompose, optimize, then deterministically route a bound circuit.

        ``physical_qubits`` restricts routing to an induced device subgraph.
        ``initial_layout`` fixes the starting logical-to-physical mapping.
        """
        from quspect.transpiler._passes import (
            DecomposePass,
            LayoutRoutingPass,
            OptimizePass,
        )
        from quspect.transpiler._target import CompilationTarget

        device_target = CompilationTarget.from_backend(self)
        # SABRE supports only one- and two-qubit operations.  This fallback is
        # required even when native gate metadata is absent or includes CCX:
        # pairwise coupling data cannot describe a three-qubit interaction.
        circuit = _decompose_for_pairwise_routing(circuit)

        # If the provider advertises a basis, recursively lower the fallback
        # output as well (for example, CCX-generated CX gates on a CZ-native
        # device).  With missing metadata, preserve translator-supported gates.
        if device_target.native_gates:
            circuit = DecomposePass(device_target).run(circuit)

        # Optimize after decomposition so redundant primitive gates are
        # removed before routing. Running optimization after SABRE would be
        # unsafe because the generic DAG does not carry its evolving mapping.
        # With a known basis, single-qubit runs are resynthesized directly
        # into native gates instead of being fused into ``u3``.
        circuit = OptimizePass.from_target(device_target).run(circuit)

        target = device_target
        dense_to_physical = list(range(device_target.num_qubits))
        dense_initial_layout: list[int] | None = None

        if physical_qubits is not None:
            target, dense_to_physical = device_target.induced_subgraph(physical_qubits)
            if not target.coupling_graph and circuit.num_qubits > 1:
                raise ValueError(
                    "The selected physical_qubits do not contain a usable "
                    "two-qubit coupling edge."
                )
            if target.num_qubits > 1 and circuit.num_qubits > 1:
                reachable = {0}
                frontier = [0]
                while frontier:
                    current = frontier.pop()
                    for neighbor in target.adjacency[current]:
                        if neighbor not in reachable:
                            reachable.add(neighbor)
                            frontier.append(neighbor)
                if len(reachable) != target.num_qubits:
                    raise ValueError(
                        "physical_qubits must induce a connected coupling "
                        "subgraph on the backend."
                    )

        if initial_layout is not None:
            if not isinstance(initial_layout, (list, tuple)):
                raise TypeError("initial_layout must be a list or tuple of integers.")
            requested_layout = list(initial_layout)
            if physical_qubits is None:
                dense_initial_layout = requested_layout
            else:
                physical_to_dense = {
                    physical: dense for dense, physical in enumerate(dense_to_physical)
                }
                outside = [
                    physical
                    for physical in requested_layout
                    if physical not in physical_to_dense
                ]
                if outside:
                    raise ValueError(
                        "initial_layout contains qubits outside physical_qubits: "
                        f"{outside}."
                    )
                dense_initial_layout = [
                    physical_to_dense[physical] for physical in requested_layout
                ]

        if circuit.num_qubits > target.num_qubits:
            raise ValueError(
                f"Circuit uses {circuit.num_qubits} qubits but the selected "
                f"target provides only {target.num_qubits} physical qubits."
            )

        layout: list[int] | None = None
        if target.coupling_graph:
            routed, candidate_layout, _ = LayoutRoutingPass(
                target,
                iter_num=3,
                seed=0,
                initial_mapping=dense_initial_layout,
            ).run(circuit)
            # SABRE may add workspace wires when routing through otherwise
            # unused physical qubits.  Keep every wire that occurs in the
            # routed circuit and trim only unused trailing search workspace.
            if len(candidate_layout) >= routed.num_qubits:
                circuit = routed
                layout = [
                    dense_to_physical[dense]
                    for dense in candidate_layout[: routed.num_qubits]
                ]
        elif initial_layout is not None:
            if len(initial_layout) != circuit.num_qubits:
                raise ValueError(
                    f"initial_layout length ({len(initial_layout)}) must equal "
                    f"the circuit qubit count ({circuit.num_qubits})."
                )
            layout = list(initial_layout)

        return circuit, layout

    @property
    def platform_name(self) -> str:
        return "logicalqubit"

    @property
    def num_qubits(self) -> int:
        cfg = self._get_config()
        for key in ("num_qubits", "n_qubits", "qubit_num", "qubits"):
            value = cfg.get(key)
            if isinstance(value, (list, tuple)):
                return len(value)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
        topology = cfg.get("topology", {})
        if isinstance(topology, dict):
            coordinates = topology.get("qubit_coordinates")
            if isinstance(coordinates, list):
                return len(coordinates)
            coupling = topology.get("coupling_map")
            if isinstance(coupling, list) and coupling:
                return max(int(value) for edge in coupling for value in edge) + 1
        return 0

    @property
    def native_gate_set(self) -> set[str]:
        cfg = self._get_config()
        raw = cfg.get("native_gate_set", cfg.get("native_gates", []))
        if isinstance(raw, (list, tuple, set)):
            return {str(gate).lower() for gate in raw}
        return set()

    @property
    def capabilities(self) -> BackendCapabilities:
        cfg = self._get_config()
        raw_modes = cfg.get("supported_modes", ["circuit"])
        modes = (
            frozenset(str(mode).lower() for mode in raw_modes)
            if isinstance(raw_modes, list)
            else frozenset({"circuit"})
        )
        raw_measurements = cfg.get("supported_measurement_types", [])
        measurement_types = (
            frozenset(str(item) for item in raw_measurements)
            if isinstance(raw_measurements, list)
            else frozenset()
        )
        return BackendCapabilities(
            supports_sampling="circuit" in modes,
            supports_batch="circuit" in modes,
            supports_estimator="script" in modes,
            supports_statevector=False,
            supports_job_recovery=True,
            supports_cancel=True,
            max_circuits_per_job=_MAX_BATCH_CIRCUITS,
            max_shots=_MAX_SHOTS,
            supported_modes=modes,
            result_formats=frozenset({"auto", "counts", "memory"}),
            measurement_types=measurement_types,
        )

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._get_config())

    def _get_config(self) -> dict[str, Any]:
        if self._config_cache is None:
            try:
                self._config_cache = dict(self._lq_backend.config)
            except Exception:
                self._config_cache = {}
        return self._config_cache

    def __repr__(self) -> str:
        return f"LogicalQubitBackend(name={self.name!r}, qubits={self.num_qubits})"


def _job_transport(raw_job: Any, children: Sequence[Any]) -> str:
    """Infer SDK fallback transport from its real children, not its wrapper."""
    child_modes = {
        bool(child._use_async)
        for child in children
        if getattr(child, "_use_async", None) is not None
    }
    if len(child_modes) > 1:
        raise BackendError("LogicalQubit fallback children use mixed transports.")
    if child_modes:
        return "async" if child_modes.pop() else "sync"
    return "async" if bool(getattr(raw_job, "_use_async", True)) else "sync"
