"""Tests for the provider-neutral backend layer (``quspect.backends``).

Covers ``_base.py`` (Backend / ExecutionJob / CompletedJob / CompositeJob /
CloudBackend), ``_execution.py`` (CircuitExperiment / ExecutionRequest /
ExecutionBatch / JobReference / BackendCapabilities) and ``_exceptions.py``.
The local ``Simulator`` and ``DensityMatrixSimulator`` are ``Backend``
subclasses and serve as the concrete local backends; their results are
cross-checked against direct ``Simulator.run`` calls and against qiskit's
exact distributions.  Cloud abstractions are exercised with in-memory fakes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest

import quspect
from qiskit_tests.helpers import (
    assert_counts_match,
    qiskit_probabilities,
    random_circuit_pair,
)
from quspect.backends import (
    Backend,
    BackendCapabilities,
    CircuitExperiment,
    CloudBackend,
    CloudJob,
    CompletedJob,
    CompositeJob,
    ExecutionBatch,
    ExecutionJob,
    ExecutionRequest,
    JobReference,
    JobStatus,
    PartialSubmissionError,
)
from quspect.backends._exceptions import (
    AuthenticationError,
    BackendError,
    BackendNotFoundError,
    JobError,
    JobTimeoutError,
    OptionalDependencyError,
    TranslationError,
)
from quspect.circuit.parameter import Parameter
from quspect.simulation import DensityMatrixSimulator, Simulator
from quspect.simulation.result import SimulationResult

SHOTS = 8000


# =============================================================================
# Helpers / fakes
# =============================================================================


def _result(counts: dict[str, int] | None = None) -> SimulationResult:
    """Build a small SimulationResult with the given counts."""
    counts = counts if counts is not None else {"0": 1}
    total = sum(counts.values())
    return SimulationResult(
        counts=dict(counts),
        probabilities={k: v / total for k, v in counts.items()} if total else {},
        num_shots=total,
        num_qubits=len(next(iter(counts), "")),
        num_clbits=len(next(iter(counts), "")),
    )


def _bell(name: str = "bell") -> quspect.QuantumCircuit:
    """Return a measured Bell-state circuit."""
    return quspect.QuantumCircuit(name=name).h(0).cx(0, 1).measure_all()


def _ref(job_id: str = "j1", **kwargs: Any) -> JobReference:
    """Build a JobReference with sane defaults for fake cloud jobs."""
    data: dict[str, Any] = {
        "platform": "fake",
        "backend_name": "qpu",
        "job_id": job_id,
        "base_url": "https://fake.invalid",
    }
    data.update(kwargs)
    return JobReference(**data)


class FakeCloudJob(CloudJob):
    """In-memory CloudJob with scripted status / result / cancel behaviour."""

    def __init__(
        self,
        reference: JobReference,
        *,
        status: JobStatus = JobStatus.COMPLETED,
        result: SimulationResult | ExecutionBatch | None = None,
        cancel: bool | Exception = True,
    ) -> None:
        self._reference_value = reference
        self._status = status
        self._result_value = result if result is not None else _result()
        self._cancel = cancel
        self.result_calls: list[tuple[float | None, float]] = []

    def result(
        self, timeout: float | None = None, *, poll_interval: float = 2.0
    ) -> SimulationResult | ExecutionBatch:
        """Record the call and return the scripted result."""
        self.result_calls.append((timeout, poll_interval))
        return self._result_value

    def status(self) -> JobStatus:
        """Return the scripted status."""
        return self._status

    def cancel(self) -> bool:
        """Return or raise the scripted cancellation outcome."""
        if isinstance(self._cancel, Exception):
            raise self._cancel
        return self._cancel

    @property
    def job_id(self) -> str:
        """Return the reference's job id."""
        return self._reference_value.job_id

    @property
    def reference(self) -> JobReference:
        """Return the scripted reference."""
        return self._reference_value


class RecordingBackend(Backend):
    """Local backend that records ``run`` calls and returns fixed results."""

    def __init__(self, formats: frozenset[str] = frozenset({"counts"})) -> None:
        self.calls: list[tuple[quspect.QuantumCircuit, int, dict[str, Any]]] = []
        self._formats = formats

    def run(
        self,
        circuit: quspect.QuantumCircuit,
        *,
        shots: int = 1024,
        save_statevector: bool = False,
        **options: Any,
    ) -> SimulationResult:
        """Record the call and return a dummy result."""
        self.calls.append((circuit, shots, dict(options)))
        return _result({"0": shots})

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "recording"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Advertise the configured result formats."""
        return BackendCapabilities(result_formats=self._formats)


class FakeCloudBackend(CloudBackend):
    """CloudBackend whose submit_request returns a scripted job."""

    def __init__(self, job: CloudJob) -> None:
        self._job = job

    def run(
        self,
        circuit: quspect.QuantumCircuit,
        *,
        shots: int = 1024,
        save_statevector: bool = False,
        **options: Any,
    ) -> SimulationResult:
        """Not used by these tests."""
        raise NotImplementedError

    def submit(
        self, circuit: quspect.QuantumCircuit, *, shots: int = 1024, **options: Any
    ) -> CloudJob:
        """Return the scripted job."""
        return self._job

    def submit_request(self, request: ExecutionRequest) -> CloudJob:
        """Return the scripted job."""
        return self._job

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "fake-cloud"

    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "fake"

    @property
    def num_qubits(self) -> int:
        """Return a fixed qubit count."""
        return 5

    @property
    def native_gate_set(self) -> set[str]:
        """Return a fixed native gate set."""
        return {"cx", "rz"}


# =============================================================================
# Local simulators as Backends
# =============================================================================


def test_simulators_are_backends() -> None:
    """Simulator and DensityMatrixSimulator implement the Backend interface."""
    assert isinstance(Simulator(seed=0), Backend)
    assert isinstance(DensityMatrixSimulator(seed=0), Backend)
    assert Simulator(seed=0).name == "simulator"
    assert DensityMatrixSimulator(seed=0).name == "density_matrix_simulator"
    caps = Simulator(seed=0).capabilities
    assert caps.supports_batch and caps.max_circuits_per_job is None
    assert caps.result_formats == frozenset({"counts"})


@pytest.mark.parametrize("seed", [1, 7, 99])
def test_simulator_execute_matches_sequential_run_exactly(seed: int) -> None:
    """execute() on a seeded Simulator equals sequential run() calls exactly."""
    c1, _ = random_circuit_pair(3, 12, seed)
    c1.measure_all()
    c2 = _bell("b2")
    request = ExecutionRequest.batch([c1, c2], shots=500)
    batch = Simulator(seed=seed).execute(request)
    sim = Simulator(seed=seed)
    expected = [sim.run(c1, shots=500).counts, sim.run(c2, shots=500).counts]
    assert batch.get_counts() == expected
    assert batch.backend_name == "simulator"
    assert batch.metadata["execution"] == "synchronous"
    assert len(batch) == 2


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize("seed", [3, 11])
def test_simulator_execute_matches_qiskit_distribution(seed: int) -> None:
    """Batch results from Simulator.execute follow qiskit's exact distribution."""
    circuits = []
    for k in range(3):
        circ, _ = random_circuit_pair(3, 15, seed * 10 + k)
        circ.measure_all()
        circuits.append(circ)
    batch = Simulator(seed=seed).execute(ExecutionRequest.batch(circuits, shots=SHOTS))
    for circ, res in zip(circuits, batch, strict=True):
        assert res.num_shots == SHOTS
        assert_counts_match(res.counts, qiskit_probabilities(circ), SHOTS)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_sweep_execution_matches_qiskit_per_point() -> None:
    """A parameter sweep binds each point correctly (checked against qiskit)."""
    theta = Parameter("theta")
    template = quspect.QuantumCircuit().ry(theta, 0).cx(0, 1).rx(theta, 1)
    template.measure_all()
    points = [0.0, 0.4, 1.3, math.pi]
    request = ExecutionRequest.sweep(
        template, [{"theta": v} for v in points], shots=SHOTS
    )
    batch = Simulator(seed=5).execute(request)
    assert batch.labels == tuple(f"point-{i}" for i in range(len(points)))
    for value, res in zip(points, batch, strict=True):
        bound = template.bind_parameters({theta: value})
        assert_counts_match(res.counts, qiskit_probabilities(bound), SHOTS)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_partial_measurement_execute_matches_qiskit() -> None:
    """Measuring a subset of qubits into permuted clbits uses quspect bit order."""
    circ = quspect.QuantumCircuit().x(0).h(1).ry(0.9, 2)
    circ.measure([2, 0], [0, 1])
    batch = Simulator(seed=8).execute(ExecutionRequest.single(circ, shots=SHOTS))
    res = batch.single()
    assert all(len(k) == 2 for k in res.counts)
    assert_counts_match(res.counts, qiskit_probabilities(circ), SHOTS)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_density_matrix_backend_execute_matches_qiskit_full_measure() -> None:
    """DensityMatrixSimulator.execute with measure_all matches qiskit."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).ry(0.6, 2).measure_all()
    batch = DensityMatrixSimulator(seed=4).execute(
        ExecutionRequest.single(circ, shots=SHOTS)
    )
    counts = {k: v for k, v in batch.single().counts.items() if v}
    assert_counts_match(counts, qiskit_probabilities(circ), SHOTS)


@pytest.mark.qiskit_ref
def test_density_matrix_backend_partial_measure_uses_clbits() -> None:
    """DM backend counts must be keyed by clbits like the Simulator backend."""
    circ = quspect.QuantumCircuit().x(0).id(1)
    circ.measure([0], [0])
    dm = DensityMatrixSimulator(seed=1).execute(ExecutionRequest.single(circ, shots=20))
    sv = Simulator(seed=1).execute(ExecutionRequest.single(circ, shots=20))
    assert sv.single().counts == {"1": 20}
    assert {k: v for k, v in dm.single().counts.items() if v} == {"1": 20}


def test_simulator_submit_request_returns_completed_job() -> None:
    """Simulator.submit_request yields an immediately-complete local job."""
    request = ExecutionRequest.batch(
        [_bell("a"), _bell("b")],
        shots=64,
        metadata={"workload_id": "  wl-1 "},
    )
    job = Simulator(seed=2).submit_request(request)
    assert isinstance(job, CompletedJob)
    assert isinstance(job, ExecutionJob)
    assert job.status() is JobStatus.COMPLETED
    assert job.done() is True
    assert job.cancel() is False
    assert job.job_id.startswith("local-")
    assert job.job_ids == (job.job_id,)
    ref = job.reference
    assert ref.platform == "local"
    assert ref.transport == "in-process"
    assert ref.batch_size == 2
    assert ref.labels == ("a", "b")
    assert ref.experiment_shapes == ((2, 2), (2, 2))
    assert ref.shots == 64
    assert ref.workload_id == "wl-1"
    result = job.result(timeout=0.0)
    assert result is job.result()
    assert result.get_counts() == Simulator(seed=2).execute(request).get_counts()


def test_completed_job_rejects_invalid_workload_id() -> None:
    """A non-string workload_id in metadata raises ValueError."""
    request = ExecutionRequest.single(_bell(), shots=4, metadata={"workload_id": 3})
    with pytest.raises(ValueError, match="workload_id"):
        Simulator(seed=0).submit_request(request)


# =============================================================================
# Backend.execute default implementation
# =============================================================================


def test_backend_execute_requires_execution_request() -> None:
    """execute() rejects anything that is not an ExecutionRequest."""
    with pytest.raises(TypeError, match="ExecutionRequest"):
        Simulator(seed=0).execute(_bell())  # type: ignore[arg-type]


def test_backend_execute_rejects_unsupported_result_format() -> None:
    """execute() rejects result formats the backend does not advertise."""
    request = ExecutionRequest.single(_bell(), shots=4, result_format="memory")
    with pytest.raises(ValueError, match="does not support result format"):
        Simulator(seed=0).execute(request)


def test_backend_execute_auto_maps_to_counts_and_forwards_options() -> None:
    """'auto' maps to 'counts'; options and shots reach run() unchanged."""
    backend = RecordingBackend()
    request = ExecutionRequest.batch(
        [_bell("x"), _bell("y")],
        shots=33,
        result_format="AUTO",
        options={"foo": 1},
        metadata={"tag": "t"},
    )
    batch = backend.execute(request)
    assert [c.name for c, _, _ in backend.calls] == ["x", "y"]
    assert all(shots == 33 for _, shots, _ in backend.calls)
    assert backend.calls[0][2] == {"foo": 1, "result_format": "counts"}
    assert batch.metadata == {"execution": "synchronous", "tag": "t"}
    assert batch.labels == ("x", "y")


def test_backend_execute_memory_format_forwarded_when_supported() -> None:
    """A backend advertising 'memory' receives result_format='memory'."""
    backend = RecordingBackend(frozenset({"counts", "memory"}))
    backend.execute(ExecutionRequest.single(_bell(), result_format="memory"))
    assert backend.calls[0][2]["result_format"] == "memory"


def test_backend_execute_binds_circuit_snapshots() -> None:
    """Circuits passed to run() are bound snapshots, not the template object."""
    theta = Parameter("t")
    template = quspect.QuantumCircuit().rx(theta, 0).measure_all()
    backend = RecordingBackend()
    backend.execute(ExecutionRequest.single(template, bindings={theta: 0.5}))
    ran = backend.calls[0][0]
    assert ran is not template
    assert not ran.parameters


def test_execution_job_done_default_uses_status() -> None:
    """ExecutionJob.done() is True exactly for terminal statuses."""
    for status in JobStatus:
        job = FakeCloudJob(_ref(), status=status)
        assert job.done() is (
            status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        )
        assert job.job_ids == ("j1",)


# =============================================================================
# CloudBackend.execute
# =============================================================================


def test_cloud_backend_execute_returns_batch_unchanged() -> None:
    """A job returning an ExecutionBatch is passed through."""
    batch = ExecutionBatch(results=(_result(),), labels=("a",))
    backend = FakeCloudBackend(FakeCloudJob(_ref(), result=batch))
    assert backend.execute(ExecutionRequest.single(_bell())) is batch
    assert backend.platform_name == "fake"
    assert backend.num_qubits == 5
    assert backend.native_gate_set == {"cx", "rz"}


def test_cloud_backend_execute_wraps_single_result() -> None:
    """A job returning one SimulationResult is wrapped with request labels."""
    res = _result({"11": 3})
    backend = FakeCloudBackend(FakeCloudJob(_ref(), result=res))
    out = backend.execute(ExecutionRequest.single(_bell(), label="lbl"))
    assert out.labels == ("lbl",)
    assert out.single() is res
    assert out.backend_name == "fake-cloud"


def test_cloud_backend_execute_rejects_unknown_result_type() -> None:
    """A job returning a non-result object raises TypeError."""
    job = FakeCloudJob(_ref())
    job._result_value = "nope"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="unsupported result type"):
        FakeCloudBackend(job).execute(ExecutionRequest.single(_bell()))


# =============================================================================
# CompositeJob
# =============================================================================


def test_composite_requires_children() -> None:
    """An empty CompositeJob is rejected."""
    with pytest.raises(ValueError, match="at least one"):
        CompositeJob([])


def test_composite_requires_cloud_children() -> None:
    """Non-CloudJob children are rejected."""
    local = Simulator(seed=0).submit_request(ExecutionRequest.single(_bell()))
    with pytest.raises(TypeError, match="CloudJob"):
        CompositeJob([local])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "override",
    [
        {"platform": "other"},
        {"backend_name": "other"},
        {"base_url": "https://other.invalid"},
        {"transport": "sync"},
    ],
)
def test_composite_rejects_mixed_routes(override: dict[str, str]) -> None:
    """Children with different platform/backend/url/transport are rejected."""
    a = FakeCloudJob(_ref("a"))
    b = FakeCloudJob(_ref("b", **override))
    with pytest.raises(ValueError, match="one platform"):
        CompositeJob([a, b])


def test_composite_rejects_mixed_workload_ids() -> None:
    """Children with different workload IDs are rejected."""
    a = FakeCloudJob(_ref("a", workload_id="w1"))
    b = FakeCloudJob(_ref("b", workload_id="w2"))
    with pytest.raises(ValueError, match="workload ID"):
        CompositeJob([a, b])


def test_composite_rejects_duplicate_child_labels() -> None:
    """Explicit labels repeated across children are rejected."""
    a = FakeCloudJob(_ref("a", batch_size=1, labels=("x",)))
    b = FakeCloudJob(_ref("b", batch_size=1, labels=("x",)))
    with pytest.raises(ValueError, match="globally unique"):
        CompositeJob([a, b])


def test_composite_reference_aggregation() -> None:
    """The composite reference concatenates labels, shapes and child ids."""
    a = FakeCloudJob(
        _ref(
            "a",
            batch_size=2,
            labels=("p", "q"),
            experiment_shapes=((1, 1), (2, 2)),
            shots=10,
            result_format="counts",
            workload_id="w",
        )
    )
    b = FakeCloudJob(
        _ref(
            "b",
            batch_size=1,
            labels=("r",),
            experiment_shapes=((3, 3),),
            shots=10,
            result_format="counts",
            workload_id="w",
        )
    )
    job = CompositeJob([a, b])
    ref = job.reference
    assert job.job_id == "a+2"
    assert job.job_ids == ("a", "b")
    assert ref.job_id == "a"
    assert ref.batch_size == 3
    assert ref.labels == ("p", "q", "r")
    assert ref.experiment_shapes == ((1, 1), (2, 2), (3, 3))
    assert ref.child_job_ids == ("a", "b")
    assert ref.child_batch_sizes == (2, 1)
    assert ref.shots == 10
    assert ref.result_format == "counts"
    assert ref.workload_id == "w"
    restored = JobReference.from_dict(ref.to_dict())
    assert restored == ref


def test_composite_mixed_shots_and_formats_become_none() -> None:
    """Differing shots / result formats across children aggregate to None."""
    a = FakeCloudJob(_ref("a", shots=10, result_format="counts"))
    b = FakeCloudJob(_ref("b", shots=20, result_format="memory"))
    ref = CompositeJob([a, b]).reference
    assert ref.shots is None
    assert ref.result_format is None


def test_composite_generates_unique_labels_avoiding_reserved() -> None:
    """Generated labels for unlabelled children skip labels used elsewhere."""
    a = FakeCloudJob(_ref("a"))  # unlabelled, batch_size None -> 1 label
    b = FakeCloudJob(_ref("b", batch_size=1, labels=("experiment-0",)))
    c = FakeCloudJob(_ref("c", batch_size=2))
    ref = CompositeJob([a, b, c]).reference
    assert ref.labels == (
        "experiment-0-1",
        "experiment-0",
        "experiment-2",
        "experiment-3",
    )
    assert len(set(ref.labels)) == 4
    assert ref.batch_size == 4
    assert ref.child_batch_sizes == (1, 1, 2)


def test_composite_flattens_nested_child_job_ids() -> None:
    """A child whose reference lists child ids contributes all of them."""
    nested = FakeCloudJob(
        _ref("n", batch_size=3, child_job_ids=("n1", "n2"), child_batch_sizes=(1, 2))
    )
    nested_default = FakeCloudJob(_ref("m", batch_size=2, child_job_ids=("m1", "m2")))
    plain = FakeCloudJob(_ref("p"))
    job = CompositeJob([nested, nested_default, plain])
    assert job.job_ids == ("n1", "n2", "m1", "m2", "p")
    assert job.reference.child_batch_sizes == (1, 2, 1, 1, 1)
    assert job.job_id == "n1+5"


def test_composite_result_concatenates_in_order() -> None:
    """result() concatenates batch and single child results in child order."""
    r1, r2, r3 = _result({"0": 1}), _result({"1": 2}), _result({"01": 3})
    a = FakeCloudJob(
        _ref("a", batch_size=2, labels=("x", "y")),
        result=ExecutionBatch(results=(r1, r2), labels=("x", "y")),
    )
    b = FakeCloudJob(_ref("b", batch_size=1, labels=("z",)), result=r3)
    job = CompositeJob([a, b])
    batch = job.result(poll_interval=0.5)
    assert batch.results == (r1, r2, r3)
    assert batch.labels == ("x", "y", "z")
    assert batch.by_label("z") is r3
    assert batch.job_id == "a+2"
    assert batch.backend_name == "qpu"
    assert batch.metadata["platform"] == "fake"
    assert batch.metadata["child_job_ids"] == ("a", "b")
    assert a.result_calls == [(None, 0.5)]


def test_composite_result_splits_remaining_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each child receives the remaining part of the overall timeout."""
    clock = iter([100.0, 103.0, 120.0])
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    a, b = FakeCloudJob(_ref("a")), FakeCloudJob(_ref("b"))
    CompositeJob([a, b]).result(timeout=10.0, poll_interval=1.0)
    assert a.result_calls == [(7.0, 1.0)]
    assert b.result_calls == [(0.0, 1.0)]


@pytest.mark.parametrize(
    ("timeout", "poll", "match"),
    [
        (-1.0, 1.0, "timeout"),
        (math.inf, 1.0, "timeout"),
        (math.nan, 1.0, "timeout"),
        (None, 0.0, "poll_interval"),
        (None, -2.0, "poll_interval"),
        (None, math.inf, "poll_interval"),
    ],
)
def test_composite_result_validates_arguments(
    timeout: float | None, poll: float, match: str
) -> None:
    """Invalid timeout / poll_interval values raise ValueError."""
    job = CompositeJob([FakeCloudJob(_ref())])
    with pytest.raises(ValueError, match=match):
        job.result(timeout=timeout, poll_interval=poll)


_S = JobStatus


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((_S.COMPLETED, _S.RUNNING), _S.RUNNING),
        ((_S.FAILED, _S.RUNNING, _S.QUEUED), _S.RUNNING),
        ((_S.FAILED, _S.QUEUED), _S.QUEUED),
        ((_S.QUEUED, _S.PENDING), _S.QUEUED),
        ((_S.CANCELLED, _S.PENDING), _S.PENDING),
        ((_S.COMPLETED, _S.UNKNOWN), _S.UNKNOWN),
        ((_S.COMPLETED, _S.FAILED, _S.CANCELLED), _S.FAILED),
        ((_S.COMPLETED, _S.CANCELLED), _S.CANCELLED),
        ((_S.COMPLETED, _S.COMPLETED), _S.COMPLETED),
    ],
)
def test_composite_status_precedence(
    statuses: tuple[JobStatus, ...], expected: JobStatus
) -> None:
    """Active children dominate; terminal precedence is FAILED > CANCELLED."""
    jobs = [FakeCloudJob(_ref(f"j{i}"), status=s) for i, s in enumerate(statuses)]
    composite = CompositeJob(jobs)
    assert composite.status() is expected
    terminal = {_S.COMPLETED, _S.FAILED, _S.CANCELLED}
    assert composite.done() is all(s in terminal for s in statuses)


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ((True, True), True),
        ((True, False), False),
        ((RuntimeError("x"), True), False),
        ((False, RuntimeError("y")), False),
    ],
)
def test_composite_cancel_aggregates_and_tries_every_child(
    outcomes: tuple[bool | Exception, ...], expected: bool
) -> None:
    """cancel() tries every child and is True only if all accepted."""
    calls: list[str] = []

    class Tracking(FakeCloudJob):
        def cancel(self) -> bool:
            """Record the call before delegating."""
            calls.append(self.job_id)
            return super().cancel()

    jobs = [Tracking(_ref(f"j{i}"), cancel=o) for i, o in enumerate(outcomes)]
    assert CompositeJob(jobs).cancel() is expected
    assert calls == [j.job_id for j in jobs]


# =============================================================================
# CircuitExperiment
# =============================================================================


def test_experiment_snapshots_circuit() -> None:
    """Mutating the source circuit after construction does not affect it."""
    circ = quspect.QuantumCircuit().h(0)
    exp = CircuitExperiment(circ)
    circ.x(0)
    assert len(exp.circuit.instructions) == 1
    assert exp.shape == (1, 0)
    assert exp.bound_circuit() is not exp.circuit


def test_experiment_bindings_by_name_and_parameter() -> None:
    """Bindings keyed by name or Parameter normalize to template parameters."""
    a, b = Parameter("a"), Parameter("b")
    circ = quspect.QuantumCircuit().rx(a, 0).ry(b, 0)
    exp = CircuitExperiment(circ, bindings={"b": 2, a: 1.5}, label="  L  ")
    assert exp.label == "L"
    assert {p.name: v for p, v in exp.bindings.items()} == {"a": 1.5, "b": 2.0}
    assert all(isinstance(p, Parameter) for p in exp.bindings)
    assert not exp.bound_circuit().parameters
    with pytest.raises(TypeError):
        exp.bindings["a"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    ("bindings", "exc", "match"),
    [
        ({1: 0.1}, TypeError, "keys"),
        ({"": 0.1}, TypeError, "keys"),
        ({"a": "x"}, TypeError, "real scalar"),
        ({"a": None}, TypeError, "real scalar"),
        ({"a": math.inf}, ValueError, "finite"),
        ({}, ValueError, "missing"),
        ({"a": 1.0, "zz": 2.0}, ValueError, "unknown"),
    ],
)
def test_experiment_binding_validation(
    bindings: dict[Any, Any], exc: type[Exception], match: str
) -> None:
    """Invalid bindings raise descriptive errors."""
    circ = quspect.QuantumCircuit().rx(Parameter("a"), 0)
    with pytest.raises(exc, match=match):
        CircuitExperiment(circ, bindings=bindings)


def test_experiment_duplicate_binding_rejected() -> None:
    """Binding the same parameter by name and by object is rejected."""
    a = Parameter("a")
    circ = quspect.QuantumCircuit().rx(a, 0)
    with pytest.raises(ValueError, match="Duplicate"):
        CircuitExperiment(circ, bindings={a: 1.0, "a": 2.0})


@pytest.mark.parametrize("label", ["", "   "])
def test_experiment_invalid_label(label: str) -> None:
    """Blank labels are rejected."""
    with pytest.raises(ValueError, match="label"):
        CircuitExperiment(quspect.QuantumCircuit().h(0), label=label)


def test_experiment_non_circuit_and_bad_metadata() -> None:
    """Non-circuit inputs and bad metadata keys are rejected."""
    with pytest.raises(TypeError, match="QuantumCircuit"):
        CircuitExperiment("circ")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        CircuitExperiment(quspect.QuantumCircuit().h(0), metadata={"": 1})
    exp = CircuitExperiment(quspect.QuantumCircuit().h(0), metadata={"k": 1})
    assert dict(exp.metadata) == {"k": 1}


# =============================================================================
# ExecutionRequest
# =============================================================================


def test_request_label_deduplication() -> None:
    """Labels fall back to circuit name / index and are de-duplicated."""
    named = quspect.QuantumCircuit(name="c").h(0)
    unnamed = quspect.QuantumCircuit().h(0)
    req = ExecutionRequest(
        experiments=[
            CircuitExperiment(named),
            CircuitExperiment(named),
            CircuitExperiment(unnamed),
            CircuitExperiment(named, label="c-1"),
        ]
    )
    assert req.labels == ("c", "c-1", "experiment-2", "c-1-1")
    assert len(req) == 4
    assert req.experiment_shapes == ((1, 0),) * 4
    assert len(req.bound_circuits()) == 4


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"experiments": []}, ValueError, "at least one"),
        ({"experiments": ["x"]}, TypeError, "CircuitExperiment"),
        ({"shots": 1.5}, TypeError, "integer"),
        ({"shots": True}, TypeError, "integer"),
        ({"shots": 0}, ValueError, "positive"),
        ({"result_format": "json"}, ValueError, "result_format"),
        ({"options": {"": 1}}, ValueError, "non-empty"),
    ],
)
def test_request_validation(
    kwargs: dict[str, Any], exc: type[Exception], match: str
) -> None:
    """Invalid request fields raise descriptive errors."""
    data: dict[str, Any] = {
        "experiments": [CircuitExperiment(quspect.QuantumCircuit().h(0))]
    }
    data.update(kwargs)
    with pytest.raises(exc, match=match):
        ExecutionRequest(**data)


def test_request_normalizes_result_format_and_freezes_mappings() -> None:
    """result_format is normalized and options/metadata are read-only."""
    req = ExecutionRequest.single(
        _bell(), result_format=" Memory ", options={"o": 1}, metadata={"m": 2}
    )
    assert req.result_format == "memory"
    with pytest.raises(TypeError):
        req.options["o"] = 5  # type: ignore[index]
    assert dict(req.metadata) == {"m": 2}


def test_request_constructors_label_length_mismatch() -> None:
    """batch() and sweep() reject label lists of the wrong length."""
    with pytest.raises(ValueError, match="labels"):
        ExecutionRequest.batch([_bell()], labels=["a", "b"])
    circ = quspect.QuantumCircuit().rx(Parameter("t"), 0).measure_all()
    with pytest.raises(ValueError, match="labels"):
        ExecutionRequest.sweep(circ, [{"t": 0.0}], labels=["a", "b"])
    req = ExecutionRequest.sweep(circ, [{"t": 0.0}, {"t": 1.0}], labels=["u", "v"])
    assert req.labels == ("u", "v")


# =============================================================================
# ExecutionBatch
# =============================================================================


def test_batch_sequence_api() -> None:
    """ExecutionBatch supports len, indexing, slicing, iteration and lookup."""
    r = [_result({"0": 1}), _result({"1": 2}), _result({"0": 3})]
    batch = ExecutionBatch(
        results=r, labels=("a", "b", "c"), job_id="J", metadata={"k": 1}
    )
    assert len(batch) == 3
    assert batch[1] is r[1]
    sub = batch[1:]
    assert isinstance(sub, ExecutionBatch)
    assert sub.labels == ("b", "c")
    assert sub.job_id == "J"
    assert dict(sub.metadata) == {"k": 1}
    assert list(batch) == r
    assert batch.by_label("c") is r[2]
    with pytest.raises(KeyError, match="zz"):
        batch.by_label("zz")
    with pytest.raises(ValueError, match="one result"):
        batch.single()
    assert batch.get_counts() == [{"0": 1}, {"1": 2}, {"0": 3}]
    assert batch.success is True
    assert batch.failed_indices == ()


def test_batch_default_labels_and_failures() -> None:
    """Default labels are generated; failed results are reported."""
    bad = _result()
    bad.success = False
    batch = ExecutionBatch(results=[_result(), bad])
    assert batch.labels == ("experiment-0", "experiment-1")
    assert batch.success is False
    assert batch.failed_indices == (1,)
    assert ExecutionBatch(results=()).success is False


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"results": ["x"]}, TypeError, "SimulationResult"),
        ({"results": [_result()], "labels": ("a", "b")}, ValueError, "equal length"),
        (
            {"results": [_result(), _result()], "labels": ("a", "a")},
            ValueError,
            "unique",
        ),
    ],
)
def test_batch_validation(
    kwargs: dict[str, Any], exc: type[Exception], match: str
) -> None:
    """Invalid batches raise descriptive errors."""
    with pytest.raises(exc, match=match):
        ExecutionBatch(**kwargs)


# =============================================================================
# JobReference
# =============================================================================


def test_job_reference_roundtrip_and_normalization() -> None:
    """to_dict/from_dict round-trips and fields are normalized."""
    ref = JobReference(
        platform=" p ",
        backend_name="b",
        job_id="j",
        base_url="https://host.invalid/api?page=2",
        transport="sync",
        batch_size=2,
        labels=(" a ", "b"),
        experiment_shapes=[[1, 1], (2, 0)],  # type: ignore[list-item]
        shots=5,
        result_format=" COUNTS ",
        workload_id=" w ",
        child_job_ids=("c1", "c2"),
        child_batch_sizes=(1, 1),
    )
    assert ref.platform == "p"
    assert ref.labels == ("a", "b")
    assert ref.experiment_shapes == ((1, 1), (2, 0))
    assert ref.result_format == "counts"
    assert ref.workload_id == "w"
    assert ref.is_batch is True
    data = ref.to_dict()
    assert data["version"] == 1
    assert JobReference.from_dict(data) == ref
    minimal = JobReference.from_dict(
        {"platform": "p", "backend_name": "b", "job_id": "j"}
    )
    assert minimal.transport == "async"
    assert minimal.is_batch is False


def test_job_reference_rejects_unknown_version() -> None:
    """from_dict rejects unsupported versions."""
    with pytest.raises(ValueError, match="version"):
        JobReference.from_dict(
            {"version": 2, "platform": "p", "backend_name": "b", "job_id": "j"}
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"platform": " "}, "platform"),
        ({"job_id": 3}, "job_id"),
        ({"transport": "carrier-pigeon"}, "transport"),
        ({"batch_size": 0}, "batch_size"),
        ({"shots": 0}, "shots"),
        ({"result_format": "json"}, "result_format"),
        ({"workload_id": " "}, "workload_id"),
        ({"base_url": "https://u:p@host.invalid"}, "credentials"),
        ({"base_url": "https://host.invalid/?api-key=1"}, "credentials"),
        ({"base_url": "https://host.invalid/?Token=1"}, "credentials"),
        ({"labels": ("a", " ")}, "labels"),
        ({"labels": ("a", "a ")}, "unique"),
        ({"batch_size": 2, "labels": ("a",)}, "batch_size"),
        ({"batch_size": 2, "experiment_shapes": ((1, 1),)}, "experiment_shapes"),
        ({"experiment_shapes": ((-1, 0),)}, "negative"),
        ({"child_job_ids": ("",)}, "child job IDs"),
        ({"child_job_ids": ("a",), "child_batch_sizes": (1, 1)}, "align"),
        ({"child_job_ids": ("a",), "child_batch_sizes": (0,)}, "positive"),
        (
            {"batch_size": 3, "child_job_ids": ("a",), "child_batch_sizes": (2,)},
            "add up",
        ),
    ],
)
def test_job_reference_validation(kwargs: dict[str, Any], match: str) -> None:
    """Invalid JobReference fields raise ValueError."""
    data: dict[str, Any] = {"platform": "p", "backend_name": "b", "job_id": "j"}
    data.update(kwargs)
    with pytest.raises(ValueError, match=match):
        JobReference(**data)


# =============================================================================
# BackendCapabilities / exceptions
# =============================================================================


def test_backend_capabilities_validation_and_freezing() -> None:
    """Capabilities validate limits and freeze set-valued fields."""
    caps = BackendCapabilities(
        supported_modes={"circuit", "wave"},  # type: ignore[arg-type]
        result_formats=["counts"],  # type: ignore[arg-type]
        measurement_types={"m"},  # type: ignore[arg-type]
    )
    assert caps.supported_modes == frozenset({"circuit", "wave"})
    assert isinstance(caps.result_formats, frozenset)
    assert isinstance(caps.measurement_types, frozenset)
    assert BackendCapabilities(max_circuits_per_job=None).max_circuits_per_job is None
    with pytest.raises(ValueError, match="max_circuits_per_job"):
        BackendCapabilities(max_circuits_per_job=0)
    with pytest.raises(ValueError, match="max_shots"):
        BackendCapabilities(max_shots=-1)


def test_partial_submission_error_fields() -> None:
    """PartialSubmissionError flattens child job ids and stringifies failures."""
    refs = [
        _ref("a"),
        _ref("b", batch_size=2, child_job_ids=("b1", "b2")),
    ]
    err = PartialSubmissionError(
        "boom", submitted_references=refs, cancellation_failures=[1, "x"]
    )
    assert str(err) == "boom"
    assert err.submitted_references == tuple(refs)
    assert err.submitted_job_ids == ("a", "b1", "b2")
    assert err.cancellation_failures == ("1", "x")
    assert isinstance(err, BackendError)


def test_exception_hierarchy() -> None:
    """All backend errors derive from BackendError; JobError keeps metadata."""
    for cls in (
        AuthenticationError,
        TranslationError,
        JobError,
        BackendNotFoundError,
        OptionalDependencyError,
    ):
        assert issubclass(cls, BackendError)
    err = JobTimeoutError("t", error_type="timeout", job_id="j")
    assert isinstance(err, JobError)
    assert (err.error_type, err.job_id) == ("timeout", "j")
    plain = JobError("x")
    assert plain.error_type is None and plain.job_id is None


def test_public_exports() -> None:
    """quspect.backends exports exactly the documented public names."""
    import quspect.backends as backends

    expected: Sequence[str] = (
        "Backend",
        "CloudBackend",
        "CloudJob",
        "CompositeJob",
        "ExecutionJob",
        "CompletedJob",
        "JobStatus",
        "CircuitExperiment",
        "ExecutionRequest",
        "ExecutionBatch",
        "JobReference",
        "BackendCapabilities",
        "PartialSubmissionError",
    )
    assert set(backends.__all__) == set(expected)
    getter: Callable[[str], Any] = lambda n: getattr(backends, n)  # noqa: E731
    assert all(getter(n) is not None for n in expected)
