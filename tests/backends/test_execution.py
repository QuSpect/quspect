"""Offline tests for provider-neutral execution contracts."""

from __future__ import annotations

import math

import pytest

from quspect import Parameter, QuantumCircuit, Simulator
from quspect.backends import (
    CircuitExperiment,
    CloudJob,
    CompositeJob,
    ExecutionBatch,
    ExecutionRequest,
    JobReference,
    JobStatus,
    PartialSubmissionError,
)
from quspect.simulation.result import SimulationResult


def _result(bitstring: str = "0", *, success: bool = True) -> SimulationResult:
    return SimulationResult(
        counts={bitstring: 8} if success else {},
        probabilities={bitstring: 1.0} if success else {},
        num_shots=8,
        num_qubits=len(bitstring),
        num_clbits=len(bitstring),
        success=success,
        error=None if success else "failed",
    )


class _StubCloudJob(CloudJob):
    def __init__(
        self,
        reference: JobReference,
        *,
        status: JobStatus = JobStatus.COMPLETED,
        result: SimulationResult | ExecutionBatch | None = None,
        cancel_result: bool = True,
        cancel_error: Exception | None = None,
    ) -> None:
        self._reference = reference
        self._status = status
        self._result = _result() if result is None else result
        self._cancel_result = cancel_result
        self._cancel_error = cancel_error
        self.cancel_calls = 0
        self.result_calls: list[tuple[float | None, float]] = []

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> SimulationResult | ExecutionBatch:
        self.result_calls.append((timeout, poll_interval))
        return self._result

    def status(self) -> JobStatus:
        return self._status

    def cancel(self) -> bool:
        self.cancel_calls += 1
        if self._cancel_error is not None:
            raise self._cancel_error
        return self._cancel_result

    @property
    def job_id(self) -> str:
        return self._reference.job_id

    @property
    def reference(self) -> JobReference:
        return self._reference


def _cloud_reference(
    job_id: str,
    *,
    base_url: str = "https://jobs.example",
    transport: str = "async",
    batch_size: int = 1,
    labels: tuple[str, ...] | None = None,
) -> JobReference:
    return JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id=job_id,
        base_url=base_url,
        transport=transport,
        batch_size=batch_size,
        labels=(job_id,) if labels is None else labels,
        shots=8,
        result_format="counts",
    )


def test_experiment_snapshots_and_normalizes_bindings() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit(name="ansatz").ry(theta, 0)
    metadata = {"kind": "base"}

    experiment = CircuitExperiment(
        circuit,
        bindings={"theta": 0.25},
        metadata=metadata,
    )
    circuit.x(1)
    metadata["kind"] = "changed"

    assert experiment.circuit.num_qubits == 1
    assert next(iter(experiment.bindings)).name == "theta"
    assert experiment.metadata["kind"] == "base"
    with pytest.raises(TypeError):
        experiment.metadata["new"] = True  # type: ignore[index]
    bound = experiment.bound_circuit()
    assert bound.instructions[0].gate is not None
    assert bound.instructions[0].gate.params == (0.25,)


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ({}, "missing"),
        ({"theta": 0.1, "extra": 0.2}, "unknown"),
        ({"theta": math.inf}, "finite"),
    ],
)
def test_experiment_rejects_invalid_bindings(bindings, message: str) -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().rx(theta, 0)
    with pytest.raises(ValueError, match=message):
        CircuitExperiment(circuit, bindings=bindings)


def test_execution_request_sweep_preserves_binding_and_label_order() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit(name="template").ry(theta, 0).measure_all()
    request = ExecutionRequest.sweep(
        circuit,
        [{theta: 0.0}, {"theta": 1.5}],
        labels=("zero", "shift"),
        shots=200,
        options={"readout_correction": True},
    )

    assert request.labels == ("zero", "shift")
    assert request.experiment_shapes == ((1, 1), (1, 1))
    assert [
        item.instructions[0].gate.params[0]  # type: ignore[union-attr]
        for item in request.bound_circuits()
    ] == [0.0, 1.5]
    assert request.options["readout_correction"] is True


@pytest.mark.parametrize("shots", [0, -1])
def test_execution_request_requires_positive_shots(shots: int) -> None:
    circuit = QuantumCircuit().measure([0], [0])
    with pytest.raises(ValueError, match="positive"):
        ExecutionRequest.single(circuit, shots=shots)


def test_local_execute_and_submit_share_the_batch_contract() -> None:
    zero = QuantumCircuit(name="zero").measure([0], [0])
    one = QuantumCircuit(name="one").x(0).measure_all()
    request = ExecutionRequest.batch(
        (zero, one),
        labels=("zero", "one"),
        shots=32,
        metadata={"workload_id": "test-workload"},
    )
    simulator = Simulator(seed=7)

    direct = simulator.execute(request)
    assert isinstance(direct, ExecutionBatch)
    assert direct.labels == ("zero", "one")
    assert direct.by_label("zero").counts == {"0": 32}
    assert direct.by_label("one").counts == {"1": 32}

    job = simulator.submit_request(request)
    assert job.status() is JobStatus.COMPLETED
    assert job.result().labels == direct.labels
    assert job.reference.platform == "local"
    assert job.reference.batch_size == 2
    assert job.reference.workload_id == "test-workload"
    assert job.cancel() is False


def test_execution_batch_never_silently_truncates() -> None:
    batch = ExecutionBatch(
        results=(_result("0"), _result("1", success=False)),
        labels=("ok", "bad"),
        backend_name="fake",
    )
    assert len(batch) == 2
    assert batch.by_label("ok").counts == {"0": 8}
    assert not batch.success
    assert batch.failed_indices == (1,)
    with pytest.raises(ValueError, match="requires one"):
        batch.single()
    assert batch[:1].single().success


def test_job_reference_round_trip_contains_no_credentials() -> None:
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_123",
        base_url="https://example.invalid",
        transport="async",
        batch_size=2,
        labels=("a", "b"),
        experiment_shapes=((1, 1), (2, 2)),
        shots=64,
        result_format="counts",
        workload_id="sampling-v1:abc123",
        child_job_ids=("task_a", "task_b"),
    )
    payload = reference.to_dict()

    assert JobReference.from_dict(payload) == reference
    assert not {"api_key", "token", "authorization"} & set(payload)

    partial = PartialSubmissionError(
        "partial",
        submitted_references=(reference,),
    )
    assert partial.submitted_job_ids == ("task_a", "task_b")


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.invalid",
        "https://example.invalid?api_key=secret",
        "https://example.invalid?token=secret",
    ],
)
def test_job_reference_rejects_credentials_in_endpoint(url: str) -> None:
    with pytest.raises(ValueError, match="credentials"):
        JobReference(
            platform="logicalqubit",
            backend_name="QZ02",
            job_id="task",
            base_url=url,
        )


def test_job_reference_requires_unique_non_empty_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        _cloud_reference("task", batch_size=2, labels=("same", "same"))
    with pytest.raises(ValueError, match="non-empty"):
        _cloud_reference("task", labels=(" ",))


@pytest.mark.parametrize(
    ("statuses", "expected", "done"),
    [
        ((JobStatus.FAILED, JobStatus.RUNNING), JobStatus.RUNNING, False),
        ((JobStatus.FAILED, JobStatus.QUEUED), JobStatus.QUEUED, False),
        ((JobStatus.COMPLETED, JobStatus.UNKNOWN), JobStatus.UNKNOWN, False),
        ((JobStatus.COMPLETED, JobStatus.CANCELLED), JobStatus.CANCELLED, True),
        ((JobStatus.FAILED, JobStatus.COMPLETED), JobStatus.FAILED, True),
        ((JobStatus.CANCELLED, JobStatus.CANCELLED), JobStatus.CANCELLED, True),
        ((JobStatus.COMPLETED, JobStatus.COMPLETED), JobStatus.COMPLETED, True),
    ],
)
def test_composite_job_status_waits_for_every_child_terminal(
    statuses: tuple[JobStatus, JobStatus],
    expected: JobStatus,
    done: bool,
) -> None:
    jobs = tuple(
        _StubCloudJob(_cloud_reference(f"task-{index}"), status=status)
        for index, status in enumerate(statuses)
    )
    composite = CompositeJob(jobs)

    assert composite.status() is expected
    assert composite.done() is done


def test_composite_job_cancel_is_best_effort_across_every_child() -> None:
    jobs = (
        _StubCloudJob(
            _cloud_reference("task-error"),
            cancel_error=RuntimeError("provider unavailable"),
        ),
        _StubCloudJob(_cloud_reference("task-rejected"), cancel_result=False),
        _StubCloudJob(_cloud_reference("task-accepted"), cancel_result=True),
    )
    composite = CompositeJob(jobs)

    assert composite.cancel() is False
    assert [job.cancel_calls for job in jobs] == [1, 1, 1]


@pytest.mark.parametrize(
    ("base_url", "transport"),
    [
        ("https://other.example", "async"),
        ("https://jobs.example", "sync"),
    ],
)
def test_composite_job_requires_one_endpoint_and_transport(
    base_url: str,
    transport: str,
) -> None:
    first = _StubCloudJob(_cloud_reference("task-first"))
    second = _StubCloudJob(
        _cloud_reference(
            "task-second",
            base_url=base_url,
            transport=transport,
        )
    )

    with pytest.raises(ValueError, match="base URL, and transport"):
        CompositeJob((first, second))


@pytest.mark.parametrize(
    "result_options",
    [
        {"timeout": -0.1},
        {"timeout": math.inf},
        {"poll_interval": 0.0},
        {"poll_interval": math.nan},
    ],
)
def test_composite_job_rejects_invalid_wait_values(
    result_options: dict[str, float],
) -> None:
    child = _StubCloudJob(_cloud_reference("task"))
    composite = CompositeJob((child,))

    with pytest.raises(ValueError):
        composite.result(**result_options)
    assert child.result_calls == []


def test_composite_job_generates_global_labels_for_unlabelled_chunks() -> None:
    first_batch = ExecutionBatch(results=(_result("0"), _result("1")))
    second_batch = ExecutionBatch(results=(_result("1"), _result("0")))
    first = _StubCloudJob(
        _cloud_reference("task-first", batch_size=2, labels=()),
        result=first_batch,
    )
    second = _StubCloudJob(
        _cloud_reference("task-second", batch_size=2, labels=()),
        result=second_batch,
    )

    composite = CompositeJob((first, second))
    expected = tuple(f"experiment-{index}" for index in range(4))

    assert composite.reference.labels == expected
    assert composite.result().labels == expected


def test_composite_job_rejects_duplicate_labels_across_children() -> None:
    first = _StubCloudJob(_cloud_reference("task-first", labels=("shared",)))
    second = _StubCloudJob(_cloud_reference("task-second", labels=("shared",)))

    with pytest.raises(ValueError, match="globally unique"):
        CompositeJob((first, second))
