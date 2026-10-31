"""Cross-validate ``quspect.variational.sampling`` against qiskit references.

Two kinds of backend are used:

* ``ExactBackend`` -- a local :class:`~quspect.backends.Backend` whose ``run``
  returns the *exact* measurement distribution computed by qiskit
  ``Statevector`` (``SimulationResult.probabilities``).  The sampling decoder
  then has no shot noise, so values must match ``StatevectorEstimator`` to
  1e-9 and parameter-shift Jacobians must match a qiskit finite difference to
  1e-7.  This checks the measurement-basis rotations, QWC grouping, bit
  decoding, identity offsets and occurrence shift/chain rule deterministically.
* ``quspect.simulation.Simulator`` -- real shot sampling, judged statistically.

Sampling tolerance
------------------
For ``H = c_0 I + sum_k c_k P_k`` every Pauli estimator ``P_hat_k`` is a mean of
``shots`` i.i.d. +-1 outcomes with single-shot std ``s_k = sqrt(1 - <P_k>^2)``.
Terms in one QWC group share shots (correlated); different groups use
independent shots.  Regardless of correlations ``std(sum X_k) <= sum std(X_k)``
(triangle inequality / Cauchy-Schwarz), so the conservative single-shot std is
``sigma_H = sum_k |c_k| s_k`` and the check is
``|H_hat - <H>| <= N_SIGMA * sigma_H / sqrt(shots)`` with exact ``<P_k>`` from
qiskit.  For the parameter-shift derivative
``dH/dtheta = sum_occ a_occ * (H(+pi/2) - H(-pi/2)) / 2`` every shifted
estimate has std ``<= sum_k |c_k| / sqrt(shots)`` (``s_k <= 1``), giving the
conservative single-shot std ``sum_occ |a_occ| * sum_k |c_k|``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qiskit_tests.helpers import assert_mean_close, qiskit_probabilities
from qiskit_tests.test_variational_adjoint import (
    BUILDERS,
    QiskitObjective,
    _hamiltonians,
    qubit_operator_to_sparse_pauli_op,
)
from quspect import QuantumCircuit
from quspect.backends import (
    Backend,
    ExecutionBatch,
    ExecutionJob,
    ExecutionRequest,
    JobReference,
    JobStatus,
)
from quspect.backends._exceptions import JobError
from quspect.backends._execution import BackendCapabilities
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.operators.qubit_operator import QubitOperator
from quspect.simulation import Simulator
from quspect.simulation.result import SimulationResult
from quspect.variational import (
    AdjointEngine,
    ObjectiveEvaluation,
    ObjectiveJob,
    ObjectiveSweep,
    ObjectiveSweepJob,
    ObservableObjective,
    PreparedSamplingObjective,
    SamplingEngine,
)

A, B, C = Parameter("a"), Parameter("b"), Parameter("c")
SHOTS = 20000


# =============================================================================
# Local backends and job doubles
# =============================================================================


class ExactBackend(Backend):
    """Local backend returning exact qiskit measurement probabilities."""

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        save_statevector: bool = False,
        **options: Any,
    ) -> SimulationResult:
        """Return the exact distribution of ``circuit`` (quspect-keyed)."""
        return SimulationResult(
            counts={},
            probabilities=qiskit_probabilities(circuit),
            num_shots=shots,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
        )

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "exact-qiskit"


class NoSamplingBackend(ExactBackend):
    """A backend that advertises no sampling support."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Capabilities with sampling disabled."""
        return BackendCapabilities(supports_sampling=False)


class RecordingSimulator(Simulator):
    """Simulator that remembers every submitted execution job."""

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.jobs: list[ExecutionJob] = []
        self.requests: list[ExecutionRequest] = []

    def submit_request(self, request: ExecutionRequest) -> ExecutionJob:
        """Submit and record the job."""
        job = super().submit_request(request)
        self.requests.append(request)
        self.jobs.append(job)
        return job


class ReplayJob(ExecutionJob):
    """An execution job 'recovered' from a reference with a canned result."""

    def __init__(
        self, reference: JobReference, raw: SimulationResult | ExecutionBatch
    ) -> None:
        self._reference = reference
        self._raw = raw

    def result(
        self, timeout: float | None = None, *, poll_interval: float = 2.0
    ) -> SimulationResult | ExecutionBatch:
        """Return the canned result."""
        return self._raw

    def status(self) -> JobStatus:
        """Always completed."""
        return JobStatus.COMPLETED

    def cancel(self) -> bool:
        """Cancellation is never accepted."""
        return False

    @property
    def job_id(self) -> str:
        """Job identifier from the reference."""
        return self._reference.job_id

    @property
    def reference(self) -> JobReference:
        """The recovery reference."""
        return self._reference


# =============================================================================
# Reference helpers
# =============================================================================


def _objective(name: str) -> tuple[ObservableObjective, dict[str, float]]:
    circ, point = BUILDERS[name]()
    return ObservableObjective(circ, _hamiltonians(circ.num_qubits)), point


def _rotation_objective() -> tuple[ObservableObjective, dict[str, float]]:
    """RX/RY/RZ-only circuit with shared and affine parameters (3 qubits)."""
    circ = QuantumCircuit().h(0).h(2)
    circ.rx(A, 0).ry(1.5 * A - 0.4, 1).rz(B, 2).cx(0, 1).cx(1, 2)
    circ.ry(B - 0.5 * C, 0).rx(C, 2).h(1).ry(A, 2)
    observables = {
        "H": QubitOperator(
            {
                ((0, "X"), (1, "X")): 0.6,
                ((0, "X"),): -0.4,
                ((0, "Z"), (1, "Z")): 0.5,
                ((1, "Z"),): 0.3,
                ((2, "Y"),): -0.7,
                (): 1.25,
            }
        ),
        "Z2": QubitOperator({((2, "Z"),): 1.0}),
    }
    return ObservableObjective(circ, observables), {"a": 0.42, "b": -0.9, "c": 1.7}


def _point_list(
    point: Mapping[str, float], order: tuple[Parameter, ...]
) -> list[float]:
    return [point[p.name] for p in order]


def _bound_statevector(
    objective: ObservableObjective, point: Mapping[str, float]
) -> Statevector:
    bound = objective.circuit.bind_parameters(
        {Parameter(k): v for k, v in point.items()}
    )
    return Statevector(np.asarray(Simulator(seed=0).statevector(bound)))


def _value_std(
    objective: ObservableObjective, point: Mapping[str, float]
) -> np.ndarray:
    """Conservative single-shot std ``sum_k |c_k| sqrt(1 - <P_k>^2)`` per observable."""
    sv = _bound_statevector(objective, point)
    n = objective.circuit.num_qubits
    out = []
    for op in objective.observables.values():
        total = 0.0
        for term, coeff in op:
            if not term:
                continue
            p = qubit_operator_to_sparse_pauli_op(QubitOperator({term: 1.0}), n)
            e = float(np.real(sv.expectation_value(p)))
            total += abs(complex(coeff)) * math.sqrt(max(0.0, 1.0 - e * e))
        out.append(total)
    return np.asarray(out)


def _gradient_std(objective: ObservableObjective, names: list[str]) -> np.ndarray:
    """Conservative single-shot std of each parameter-shift Jacobian entry."""
    chain = dict.fromkeys(names, 0.0)
    for instr in objective.circuit.instructions:
        if instr.gate is None:
            continue
        for value in instr.gate.params:
            if isinstance(value, Parameter):
                chain[value.name] += 1.0
            elif isinstance(value, ParameterExpression):
                for param, coef in value.terms.items():
                    chain[param.name] += abs(float(coef))
    coeff_l1 = [
        sum(abs(complex(c)) for t, c in op if t)
        for op in objective.observables.values()
    ]
    return np.outer(coeff_l1, [chain[n] for n in names])


def _assert_statistically_close(
    evaluation: ObjectiveEvaluation,
    objective: ObservableObjective,
    point: Mapping[str, float],
    shots: int,
) -> None:
    ref = QiskitObjective.from_quspect(objective)
    names = [p.name for p in evaluation.parameter_order]
    exact_values = ref.values(point)
    exact_jac = ref.finite_difference(point, names)
    value_std = _value_std(objective, point)
    grad_std = _gradient_std(objective, names)
    for i in range(len(exact_values)):
        assert_mean_close(
            evaluation.values[i], exact_values[i], std=value_std[i], n=shots
        )
        for j in range(len(names)):
            assert_mean_close(
                evaluation.jacobian[i, j], exact_jac[i, j], std=grad_std[i, j], n=shots
            )


# =============================================================================
# Exact (noise-free) decoding against qiskit
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("grouping", ["qwc", "none"])
@pytest.mark.parametrize("case", sorted(BUILDERS))
def test_exact_backend_values_and_jacobian(case: str, grouping: str) -> None:
    """With exact probabilities the sampling decoder equals qiskit to 1e-9/1e-7."""
    objective, point = _objective(case)
    prepared = SamplingEngine(ExactBackend(), grouping=grouping).compile(objective)  # type: ignore[arg-type]
    ev = prepared.evaluate(point)
    ref = QiskitObjective.from_quspect(objective)
    names = [p.name for p in prepared.parameter_order]
    assert sorted(names) == sorted(point)
    np.testing.assert_allclose(ev.values, ref.values(point), atol=1e-9)
    np.testing.assert_allclose(
        ev.jacobian, ref.finite_difference(point, names), atol=1e-7
    )
    adjoint = AdjointEngine().compile(objective).evaluate(point)
    order = [p.name for p in AdjointEngine().compile(objective).parameter_order]
    perm = [order.index(n) for n in names]
    np.testing.assert_allclose(ev.jacobian, adjoint.jacobian[:, perm], atol=1e-9)


@pytest.mark.qiskit_ref
def test_exact_backend_rotation_objective_with_identity_offset() -> None:
    """Identity offsets are added exactly and Y/X/Z bases decode correctly."""
    objective, point = _rotation_objective()
    prepared = SamplingEngine(ExactBackend()).compile(objective)
    ev = prepared.evaluate(point)
    ref = QiskitObjective.from_quspect(objective)
    np.testing.assert_allclose(ev.values, ref.values(point), atol=1e-9)
    names = [p.name for p in prepared.parameter_order]
    np.testing.assert_allclose(
        ev.jacobian, ref.parameter_shift(point, names), atol=1e-9
    )


def test_qwc_grouping_counts_and_experiment_manifest() -> None:
    """QWC merges commuting terms; 'none' measures every term separately."""
    objective, _ = _rotation_objective()
    qwc = SamplingEngine(ExactBackend()).compile(objective)
    none = SamplingEngine(ExactBackend(), grouping="none").compile(objective)
    # Terms X0X1, X0, Y2 | Z0Z1, Z1, Z2 -> 2 QWC groups; 6 single-term groups.
    assert qwc.num_measurement_groups == 2
    assert none.num_measurement_groups == 6
    # Occurrences: rx(a), ry(1.5a-0.4), rz(b), ry(b-0.5c), rx(c), ry(a) = 6.
    assert qwc.num_experiments_per_point == (1 + 2 * 6) * 2
    assert none.num_experiments_per_point == (1 + 2 * 6) * 6
    assert qwc.workload_id != none.workload_id
    assert qwc.workload_id.startswith("sampling-v1:")
    assert qwc.parameter_order == (A, B, C)
    assert qwc.observable_order == ("H", "Z2")


def test_workload_id_is_stable_and_semantic() -> None:
    """Identical objectives share a fingerprint; a changed coefficient does not."""
    objective, _ = _rotation_objective()
    first = SamplingEngine(ExactBackend()).compile(objective)
    second = SamplingEngine(ExactBackend(), shots=17).compile(objective)
    assert first.workload_id == second.workload_id
    changed = dict(objective.observables)
    changed["Z2"] = QubitOperator({((2, "Z"),): 2.0})
    third = SamplingEngine(ExactBackend()).compile(
        ObservableObjective(objective.circuit, changed)
    )
    assert third.workload_id != first.workload_id


# =============================================================================
# Statistical checks with the shot-based simulator
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize("grouping", ["qwc", "none"])
def test_simulator_values_and_gradient_statistics(grouping: str) -> None:
    """Shot estimates of values and parameter-shift gradients are within 5 sigma."""
    objective, point = _rotation_objective()
    engine = SamplingEngine(Simulator(seed=2024), shots=SHOTS, grouping=grouping)  # type: ignore[arg-type]
    prepared = engine.compile(objective)
    ev = prepared.evaluate(point)
    _assert_statistically_close(ev, objective, point, SHOTS)
    assert ev.metadata["shots"] == str(SHOTS)
    assert ev.metadata["backend"] == Simulator().name
    assert ev.metadata["experiments"] == str(prepared.num_experiments_per_point)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_simulator_values_for_controlled_rotations() -> None:
    """Decomposed controlled rotations sample to the qiskit expectation."""
    objective, point = _objective("controlled")
    prepared = SamplingEngine(Simulator(seed=99), shots=SHOTS).compile(objective)
    ev = prepared.evaluate(point)
    exact = QiskitObjective.from_quspect(objective).values(point)
    std = _value_std(objective, point)
    for i in range(len(exact)):
        assert_mean_close(ev.values[i], exact[i], std=std[i], n=SHOTS)


@pytest.mark.statistical
def test_eigenstate_has_zero_shot_noise() -> None:
    """For an eigenstate of every term, sampled values are exact."""
    circ = QuantumCircuit().x(0).h(1).rx(A, 2)
    op = QubitOperator({((0, "Z"),): 0.5, ((1, "X"),): -2.0, (): 0.1})
    objective = ObservableObjective(circ, {"E": op})
    ev = SamplingEngine(Simulator(seed=3), shots=500).compile(objective).evaluate([0.0])
    assert ev.values[0] == pytest.approx(0.5 * -1 - 2.0 + 0.1, abs=1e-12)
    np.testing.assert_allclose(ev.jacobian, [[0.0]], atol=1e-12)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_sweep_matches_points_statistically() -> None:
    """``evaluate_many`` decodes each point of a flattened sweep correctly."""
    objective, _ = _rotation_objective()
    backend = RecordingSimulator(seed=77)
    prepared = SamplingEngine(backend, shots=SHOTS).compile(objective)
    points = [
        {"a": 0.42, "b": -0.9, "c": 1.7},
        {"a": -1.3, "b": 0.25, "c": 0.0},
        {"a": 2.2, "b": 1.1, "c": -0.6},
    ]
    sweep = prepared.evaluate_many(points)
    assert isinstance(sweep, ObjectiveSweep)
    assert len(sweep) == 3
    assert sweep.values.shape == (3, 2)
    assert sweep.jacobian.shape == (3, 2, 3)
    assert len(backend.requests) == 1
    assert len(backend.requests[0].labels) == 3 * prepared.num_experiments_per_point
    assert backend.requests[0].labels[0] == "point-0/base/group-0"
    assert sweep.metadata["job_id"] == backend.jobs[0].job_id
    for i, point in enumerate(points):
        _assert_statistically_close(sweep.at(i), objective, point, SHOTS)


@pytest.mark.qiskit_ref
def test_exact_sweep_matches_single_point_evaluation() -> None:
    """Exact-backend sweeps equal point-by-point evaluation (sequence points)."""
    objective, _ = _rotation_objective()
    prepared = SamplingEngine(ExactBackend()).compile(objective)
    points = [[0.1, 0.2, 0.3], [-0.7, 1.4, 2.1]]
    sweep = prepared.evaluate_many(points)
    for i, point in enumerate(points):
        single = prepared.evaluate(point)
        np.testing.assert_allclose(sweep.values[i], single.values, atol=1e-12)
        np.testing.assert_allclose(sweep.jacobian[i], single.jacobian, atol=1e-12)


# =============================================================================
# Identity-only plans and empty sweeps
# =============================================================================


def test_identity_only_plan_is_immediate() -> None:
    """Identity-only objectives never touch the backend."""
    backend = RecordingSimulator(seed=1)
    objective = ObservableObjective(
        QuantumCircuit().rx(A, 0), {"I": QubitOperator.identity(0.8)}
    )
    prepared = SamplingEngine(backend).compile(objective)
    assert prepared.num_measurement_groups == 0
    assert prepared.num_experiments_per_point == 0
    job = prepared.submit([0.3])
    assert isinstance(job, ObjectiveJob)
    assert job.status() is JobStatus.COMPLETED
    assert job.cancel() is False
    assert job.reference is None
    assert job.job_id is None
    assert job.job_ids == ()
    ev = job.result()
    assert ev.values.tolist() == [0.8]
    assert ev.jacobian.tolist() == [[0.0]]
    sweep_job = prepared.submit_many([[0.1], [0.2]])
    assert sweep_job.num_points == 2
    assert sweep_job.status() is JobStatus.COMPLETED
    assert sweep_job.cancel() is False
    assert sweep_job.reference is None
    assert sweep_job.job_id is None
    assert sweep_job.job_ids == ()
    assert sweep_job.result().values.tolist() == [[0.8], [0.8]]
    assert backend.requests == []
    with pytest.raises(JobError, match="identity-only"):
        prepared.attach_many(_dummy_job())


def _dummy_job() -> ReplayJob:
    ref = JobReference(platform="local", backend_name="x", job_id="j1")
    return ReplayJob(ref, ExecutionBatch(results=()))


def test_empty_sweep() -> None:
    """Submitting no points yields an empty, correctly shaped sweep."""
    objective, _ = _rotation_objective()
    prepared = SamplingEngine(ExactBackend()).compile(objective)
    job = prepared.submit_many([])
    assert job.num_points == 0
    sweep = job.result()
    assert sweep.values.shape == (0, 2)
    assert sweep.jacobian.shape == (0, 2, 3)


# =============================================================================
# Job lifecycle and recovery
# =============================================================================


def _submitted(
    seed: int = 5,
) -> tuple[PreparedSamplingObjective, RecordingSimulator, ObjectiveJob]:
    objective, point = _rotation_objective()
    backend = RecordingSimulator(seed=seed)
    prepared = SamplingEngine(backend, shots=1000).compile(objective)
    return prepared, backend, prepared.submit(point)


def test_job_properties_and_recovery_roundtrip() -> None:
    """A JSON round-tripped reference re-attaches and decodes identically."""
    prepared, backend, job = _submitted()
    assert job.status() is JobStatus.COMPLETED
    assert job.cancel() is False
    assert job.job_id is not None and job.job_id.startswith("local-")
    assert job.job_ids == (job.job_id,)
    ref = job.reference
    assert ref is not None
    assert ref.workload_id == prepared.workload_id
    assert ref.batch_size == prepared.num_experiments_per_point
    original = job.result()
    assert job.result() is original  # cached
    restored_ref = JobReference.from_dict(ref.to_dict())
    raw = backend.jobs[0].result()
    # A freshly compiled plan for the same objective accepts the job.
    objective, _ = _rotation_objective()
    fresh = SamplingEngine(Simulator(seed=0), shots=1000).compile(objective)
    recovered = fresh.attach(ReplayJob(restored_ref, raw))
    ev = recovered.result()
    np.testing.assert_array_equal(ev.values, original.values)
    np.testing.assert_array_equal(ev.jacobian, original.jacobian)
    assert ev.metadata["job_id"] == ref.job_id
    assert prepared.metadata["workload_id"] == prepared.workload_id
    assert prepared.metadata["differentiation"] == "parameter-shift"
    assert prepared.metadata["shots"] == "1000"


def test_sweep_job_recovery_roundtrip() -> None:
    """``attach_many`` infers the point count and reproduces the sweep."""
    objective, _ = _rotation_objective()
    backend = RecordingSimulator(seed=11)
    prepared = SamplingEngine(backend, shots=500).compile(objective)
    job = prepared.submit_many([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert isinstance(job, ObjectiveSweepJob)
    assert job.status() is JobStatus.COMPLETED
    assert job.cancel() is False
    assert job.job_ids == (job.job_id,)
    original = job.result()
    assert job.result() is original
    ref = job.reference
    assert ref is not None
    recovered = prepared.attach_many(ReplayJob(ref, backend.jobs[0].result()))
    assert recovered.num_points == 2
    np.testing.assert_array_equal(recovered.result().values, original.values)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"workload_id": "sampling-v1:deadbeef"}, "workload fingerprint"),
        ({"backend_name": "other"}, "backend"),
        ({"platform": "cloud"}, "platform"),
        ({"shots": 999}, "shots"),
        ({"result_format": "counts"}, "result format"),
        ({"batch_size": 1, "labels": (), "experiment_shapes": ()}, "batch size"),
    ],
)
def test_attach_rejects_mismatched_manifest(change: dict[str, Any], match: str) -> None:
    """Any manifest mismatch on recovery raises JobError."""
    prepared, backend, job = _submitted()
    assert job.reference is not None
    ref = dataclasses.replace(job.reference, **change)
    with pytest.raises(JobError, match=match):
        prepared.attach(ReplayJob(ref, backend.jobs[0].result()))


def test_attach_rejects_reordered_labels_and_shapes() -> None:
    """Reordered labels or altered experiment shapes are detected."""
    prepared, backend, job = _submitted()
    ref = job.reference
    assert ref is not None
    reordered = dataclasses.replace(ref, labels=tuple(reversed(ref.labels)))
    with pytest.raises(JobError, match="labels"):
        prepared.attach(ReplayJob(reordered, backend.jobs[0].result()))
    shapes = [(nq + 1, nc) for nq, nc in ref.experiment_shapes]
    reshaped = dataclasses.replace(ref, experiment_shapes=shapes)
    with pytest.raises(JobError, match="experiment shapes"):
        prepared.attach(ReplayJob(reshaped, backend.jobs[0].result()))


def test_attach_type_errors() -> None:
    """``attach``/``attach_many`` require ExecutionJob instances."""
    prepared, _, _ = _submitted()
    with pytest.raises(TypeError):
        prepared.attach("job")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        prepared.attach_many("job")  # type: ignore[arg-type]


def test_attach_many_rejects_partial_batches() -> None:
    """A batch size that is not a whole number of points is rejected."""
    prepared, backend, job = _submitted()
    ref = job.reference
    assert ref is not None
    raw = backend.jobs[0].result()
    no_size = dataclasses.replace(ref, batch_size=None, labels=(), experiment_shapes=())
    with pytest.raises(JobError, match="whole sampling sweep"):
        prepared.attach_many(ReplayJob(no_size, raw))
    partial = dataclasses.replace(
        ref,
        batch_size=ref.batch_size - 1,  # type: ignore[operator]
        labels=ref.labels[:-1],
        experiment_shapes=ref.experiment_shapes[:-1],
    )
    with pytest.raises(JobError, match="whole sampling sweep"):
        prepared.attach_many(ReplayJob(partial, raw))
    # A single-point (non-sweep) reference has unprefixed labels.
    with pytest.raises(JobError, match="labels"):
        prepared.attach_many(ReplayJob(ref, raw))


# =============================================================================
# Corrupted results
# =============================================================================


def _tampered(
    mutate: Any,
) -> ObjectiveJob:
    prepared, backend, job = _submitted()
    assert job.reference is not None
    raw = backend.jobs[0].result()
    assert isinstance(raw, ExecutionBatch)
    return prepared.attach(ReplayJob(job.reference, mutate(raw)))


def _replace_result(raw: ExecutionBatch, index: int, **kw: Any) -> ExecutionBatch:
    results = list(raw.results)
    results[index] = dataclasses.replace(results[index], **kw)
    return ExecutionBatch(results=results, labels=raw.labels, backend_name="sim")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r[:-1], "expected 26 results"),
        (
            lambda r: ExecutionBatch(
                results=tuple(reversed(r.results)), labels=tuple(reversed(r.labels))
            ),
            "out of order",
        ),
        (
            lambda r: ExecutionBatch(
                results=r.results, labels=("bogus", *r.labels[1:])
            ),
            "missing",
        ),
        (lambda r: _replace_result(r, 3, success=False), "failed experiment"),
        (lambda r: _replace_result(r, 0, num_clbits=7), "shape"),
        (
            lambda r: _replace_result(r, 0, counts={"2": 5}, probabilities={}),
            "Invalid",
        ),
        (
            lambda r: _replace_result(r, 0, counts={}, probabilities={}),
            "no measurement",
        ),
        (
            lambda r: _replace_result(r, 0, probabilities={"000": math.nan}),
            "non-finite",
        ),
        (
            lambda r: _replace_result(r, 0, probabilities={"000": 0.0}),
            "zero total",
        ),
        (lambda r: r.results[0], "expected a batch"),
    ],
)
def test_corrupted_results_raise_job_error(mutate: Any, match: str) -> None:
    """Truncated, reordered, failed or malformed results raise JobError."""
    with pytest.raises(JobError, match=match):
        _tampered(mutate).result()


def test_single_result_is_accepted_for_one_experiment() -> None:
    """A one-experiment plan accepts a bare SimulationResult from the job."""
    objective = ObservableObjective(
        QuantumCircuit().h(0), {"x": QubitOperator({((0, "X"),): 1.0})}
    )
    backend = RecordingSimulator(seed=2)
    prepared = SamplingEngine(backend, shots=100).compile(objective)
    assert prepared.num_experiments_per_point == 1
    job = prepared.submit([])
    assert job.reference is not None
    single = backend.jobs[0].result()
    assert isinstance(single, ExecutionBatch)
    ev = prepared.attach(ReplayJob(job.reference, single.single())).result()
    assert ev.values[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r[:-1], "expected"),
        (
            lambda r: ExecutionBatch(
                results=tuple(reversed(r.results)), labels=tuple(reversed(r.labels))
            ),
            "out of order",
        ),
        (
            lambda r: ExecutionBatch(
                results=r.results, labels=("bogus", *r.labels[1:])
            ),
            "missing",
        ),
    ],
)
def test_corrupted_sweep_results_raise_job_error(mutate: Any, match: str) -> None:
    """Sweep decoding validates the flattened label manifest."""
    objective, _ = _rotation_objective()
    backend = RecordingSimulator(seed=3)
    prepared = SamplingEngine(backend, shots=200).compile(objective)
    job = prepared.submit_many([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    assert job.reference is not None
    raw = backend.jobs[0].result()
    recovered = prepared.attach_many(ReplayJob(job.reference, mutate(raw)))
    with pytest.raises(JobError, match=match):
        recovered.result()


def test_job_constructors_validate_sources() -> None:
    """Jobs need exactly one result source and a non-negative point count."""
    objective, _ = _rotation_objective()
    prepared = SamplingEngine(ExactBackend()).compile(objective)
    plan = prepared._plan
    with pytest.raises(ValueError, match="exactly one"):
        ObjectiveJob(plan, shots=10)
    with pytest.raises(ValueError, match="exactly one"):
        ObjectiveSweepJob(plan, num_points=1, shots=10, metadata={})
    empty = prepared.submit_many([]).result()
    with pytest.raises(ValueError, match="negative"):
        ObjectiveSweepJob(plan, num_points=-1, immediate=empty, shots=10, metadata={})


# =============================================================================
# Engine / compile validation
# =============================================================================


def test_engine_properties_and_execution_options() -> None:
    """Engine exposes backend/shots and forwards execution options."""
    backend = RecordingSimulator(seed=4)
    engine = SamplingEngine(
        backend, shots=64, execution_options={"save_statevector": False}
    )
    assert engine.backend is backend
    assert engine.shots == 64
    objective, point = _rotation_objective()
    engine.compile(objective).evaluate(point)
    request = backend.requests[0]
    assert request.shots == 64
    assert dict(request.options) == {"save_statevector": False}
    assert request.metadata["workload"] == "observable-objective"


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"shots": 0}, ValueError),
        ({"shots": -5}, ValueError),
        ({"shots": 10.0}, TypeError),
        ({"shots": True}, TypeError),
        ({"grouping": "greedy"}, ValueError),
        ({"execution_options": {"shots": 10}}, ValueError),
        ({"execution_options": {"result_format": "counts"}}, ValueError),
    ],
)
def test_engine_rejects_bad_arguments(
    kwargs: dict[str, Any], exc: type[Exception]
) -> None:
    """Non-positive or non-integer shots and bad options raise."""
    with pytest.raises(exc):
        SamplingEngine(ExactBackend(), **kwargs)


def test_engine_rejects_bad_backends() -> None:
    """Non-Backend objects and non-sampling backends are rejected."""
    with pytest.raises(TypeError, match="Backend"):
        SamplingEngine("simulator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sampling"):
        SamplingEngine(NoSamplingBackend())
    with pytest.raises(TypeError, match="ObservableObjective"):
        SamplingEngine(ExactBackend()).compile(QuantumCircuit().h(0))  # type: ignore[arg-type]


def _z0() -> dict[str, QubitOperator]:
    return {"z": QubitOperator({((0, "Z"),): 1.0})}


@pytest.mark.parametrize(
    ("circuit", "observables", "exc", "match"),
    [
        (QuantumCircuit().h(0).measure([0], [0]), _z0(), ValueError, "measure"),
        (QuantumCircuit().h(0).bit_flip(0.1, 0), _z0(), ValueError, "noise"),
        (QuantumCircuit(), _z0(), ValueError, "qubit"),
        (QuantumCircuit().h(0), {"z": QubitOperator.zero()}, ValueError, "zero"),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "Z"),): A})},
            ValueError,
            "symbolic",
        ),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "Z"),): 0.5j})},
            ValueError,
            "real",
        ),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "Z"),): math.nan})},
            ValueError,
            "finite",
        ),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((2, "Z"),): 1.0})},
            ValueError,
            "outside",
        ),
        # A repeated qubit is reduced at QubitOperator construction
        # (X0 Z0 = -1j Y0); the resulting non-real coefficient is then
        # rejected by SamplingEngine.
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "X"), (0, "Z")): 1.0})},
            ValueError,
            "real",
        ),
    ],
)
def test_compile_rejects_invalid_objectives(
    circuit: QuantumCircuit,
    observables: dict[str, QubitOperator],
    exc: type[Exception],
    match: str,
) -> None:
    """Measurements, noise, empty circuits and bad observables are rejected."""
    with pytest.raises(exc, match=match):
        SamplingEngine(ExactBackend()).compile(
            ObservableObjective(circuit, observables)
        )


def test_compile_accepts_repeated_qubit_reduced_with_real_phase() -> None:
    """X0 X0 reduces to the identity at construction (real phase +1).

    The identity term becomes an exact offset with no measurement, and the
    remaining ``Z0 Y1 Y1 = Z0`` term is measured normally.
    """
    reduced = QubitOperator(
        {((0, "X"), (0, "X")): 2.0, ((0, "Z"), (1, "Y"), (1, "Y")): 1.0}
    )
    assert reduced == QubitOperator({(): 2.0, ((0, "Z"),): 1.0})
    circuit = QuantumCircuit().x(0).h(1)

    identity_only = SamplingEngine(ExactBackend()).compile(
        ObservableObjective(circuit, {"c": QubitOperator({((0, "X"), (0, "X")): 1.5})})
    )
    assert identity_only.num_measurement_groups == 0
    np.testing.assert_allclose(identity_only.evaluate({}).values, [1.5])

    prepared = SamplingEngine(ExactBackend()).compile(
        ObservableObjective(circuit, {"z": reduced})
    )
    assert prepared.num_measurement_groups == 1
    np.testing.assert_allclose(prepared.evaluate({}).values, [2.0 - 1.0], atol=1e-12)


@pytest.mark.parametrize(
    ("point", "match"),
    [
        ({"a": 0.1, "b": 0.2}, "missing"),
        ({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.0}, "unknown"),
        ([0.1, 0.2], "expected 3"),
        ([0.1, 0.2, math.inf], "finite"),
    ],
)
def test_submit_rejects_bad_points(point: Any, match: str) -> None:
    """Unbound, unknown, mis-sized or non-finite parameters raise ValueError."""
    objective, _ = _rotation_objective()
    prepared = SamplingEngine(ExactBackend()).compile(objective)
    with pytest.raises(ValueError, match=match):
        prepared.submit(point)
    with pytest.raises(ValueError, match=match):
        prepared.submit_many([point])


# =============================================================================
# Extra objectives checked with both engines
# =============================================================================


def _special_names() -> tuple[ObservableObjective, dict[str, float]]:
    theta, z = Parameter("θ[0]"), Parameter("z")
    circ = QuantumCircuit().h(0).barrier().rx(z, 0).ry(theta, 1)
    circ.barrier(0, 1).cx(0, 1).rz(A, 1)
    observables = {
        "能量 E/1": QubitOperator({((0, "X"), (1, "Y")): 1.0}),
        "identity-only": QubitOperator.identity(0.3),
        "z": QubitOperator({((1, "Z"),): 1.0, (): -0.5}),
    }
    return ObservableObjective(circ, observables), {"z": 0.3, "θ[0]": -0.7, "a": 1.1}


def _full_cu3() -> tuple[ObservableObjective, dict[str, float]]:
    circ = QuantumCircuit().h(0).h(1).cu3(A, B, C, 0, 1).cp(2 * A, 1, 0).h(0)
    op = QubitOperator({((0, "X"), (1, "Y")): 1.0, ((1, "X"),): 0.5, ((0, "Y"),): 0.3})
    return ObservableObjective(circ, {"m": op}), {"a": 0.4, "b": -0.8, "c": 1.3}


def _cancelling() -> tuple[ObservableObjective, dict[str, float]]:
    circ = QuantumCircuit().rx(A - A + 0.3, 0).ry(A, 0)
    return ObservableObjective(circ, {"z": QubitOperator({((0, "Z"),): 1.0})}), {
        "a": 0.2
    }


def _shared_terms() -> tuple[ObservableObjective, dict[str, float]]:
    circ = QuantumCircuit().ry(A, 0).rx(B, 1).cx(0, 1).ry(0.5 * B - A, 1)
    observables = {
        "P": QubitOperator({((0, "Z"),): 1.0, ((1, "X"),): 0.5}),
        "Q": QubitOperator({((0, "Z"),): 2.0, ((1, "Y"),): -1.0, (): 0.2}),
    }
    return ObservableObjective(circ, observables), {"a": 0.9, "b": -0.35}


EXTRA = {
    "special_names": _special_names,
    "full_cu3": _full_cu3,
    "cancel": _cancelling,
    "shared_terms": _shared_terms,
}


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("engine_kind", ["adjoint", "sampling-exact"])
@pytest.mark.parametrize("case", sorted(EXTRA))
def test_extra_objectives_match_qiskit(case: str, engine_kind: str) -> None:
    """Barriers, unicode names, identity-only rows, symbolic cu3/cp all agree."""
    objective, point = EXTRA[case]()
    engine: AdjointEngine | SamplingEngine = (
        AdjointEngine() if engine_kind == "adjoint" else SamplingEngine(ExactBackend())
    )
    prepared = engine.compile(objective)
    assert prepared.observable_order == tuple(objective.observables)
    names = [p.name for p in prepared.parameter_order]
    assert sorted(names) == sorted(point)
    ev = prepared.evaluate(point)
    ref = QiskitObjective.from_quspect(objective)
    np.testing.assert_allclose(ev.values, ref.values(point), atol=1e-9)
    np.testing.assert_allclose(
        ev.jacobian, ref.finite_difference(point, names), atol=1e-7
    )


@pytest.mark.parametrize(
    ("gate_angle", "match"),
    [
        (A + math.inf, "non-finite offset"),
        (ParameterExpression({A: math.inf}), "non-finite affine"),
        (ParameterExpression({}, constant=math.inf), "non-finite constant"),
    ],
)
def test_sampling_rejects_non_finite_gate_angles(gate_angle: Any, match: str) -> None:
    """Non-finite offsets, coefficients or constants are rejected at compile."""
    circ = QuantumCircuit().ry(B, 0).rx(gate_angle, 0)
    with pytest.raises(ValueError, match=match):
        SamplingEngine(ExactBackend()).compile(ObservableObjective(circ, _z0()))


@pytest.mark.parametrize("kind", ["adjoint", "sampling"])
def test_non_numeric_coefficient_rejected(kind: str) -> None:
    """A non-numeric observable coefficient raises TypeError in both engines."""
    objective = ObservableObjective(
        QuantumCircuit().h(0), {"z": QubitOperator({((0, "Z"),): "x"})}
    )
    engine: AdjointEngine | SamplingEngine = (
        AdjointEngine() if kind == "adjoint" else SamplingEngine(ExactBackend())
    )
    with pytest.raises(TypeError, match="numeric"):
        engine.compile(objective)
