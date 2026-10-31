"""Offline tests for the LogicalQubit job adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from quspect.backends import ExecutionBatch, JobReference, JobStatus
from quspect.backends._exceptions import JobError, JobTimeoutError
from quspect.backends.logicalqubit._job import LogicalQubitJob


@dataclass
class _RawStatus:
    value: str


class _RawResult:
    def __init__(
        self,
        counts: dict[str, int] | None = None,
        *,
        success: bool = True,
        shots: int = 8,
        job_id: str = "task_child",
    ) -> None:
        self.data = {
            "counts": counts or {},
            "shots": shots,
            "result_format": "counts",
            "success": success,
        }
        if not success:
            self.data.update({"error": "child failed", "error_type": "hardware_error"})
        self.status = "completed"
        self.job_id = job_id
        self.backend_name = "QZ02"
        self.metadata = {"execution_time": 0.25, "result": "omitted"}

    def get_counts(self):
        return dict(self.data["counts"])


class _RawBatch:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.backend_name = "QZ02"
        self.job_id = "task_batch"
        self.metadata = {"cost": 1.0, "result": {"large": "payload"}}


class _RawJob:
    def __init__(self, result, *, status: object = _RawStatus("completed")) -> None:
        self.job_id = "task_batch"
        self.backend_name = "QZ02"
        self.base_url = "https://example.invalid"
        self._use_async = True
        self._result = result
        self._status = status
        self.cancelled = False
        self.result_calls: list[dict] = []

    def result(self, **kwargs):
        self.result_calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def status(self):
        if isinstance(self._status, Exception):
            raise self._status
        return self._status

    def cancel(self):
        self.cancelled = True
        return True


def _reference(*, batch_size: int | None = 2) -> JobReference:
    return JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_batch",
        base_url="https://example.invalid",
        transport="async",
        batch_size=batch_size,
        labels=("zero", "one") if batch_size else ("single",),
        experiment_shapes=((1, 1), (2, 1)) if batch_size else ((3, 1),),
        shots=8,
        result_format="counts",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("queued", JobStatus.QUEUED),
        ("pending", JobStatus.PENDING),
        ("retrying", JobStatus.RUNNING),
        ("running", JobStatus.RUNNING),
        ("completed", JobStatus.COMPLETED),
        ("failed", JobStatus.FAILED),
        ("cancelled", JobStatus.CANCELLED),
        ("new-provider-state", JobStatus.UNKNOWN),
    ],
)
def test_status_uses_sdk_enum_value(raw: str, expected: JobStatus) -> None:
    job = LogicalQubitJob(_RawJob(_RawResult(), status=_RawStatus(raw)))
    assert job.status() is expected


def test_status_failure_is_not_disguised_as_pending() -> None:
    job = LogicalQubitJob(_RawJob(_RawResult(), status=RuntimeError("offline")))
    with pytest.raises(JobError, match="offline"):
        job.status()


def test_batch_conversion_preserves_order_shape_and_partial_failure() -> None:
    raw = _RawBatch(
        (
            _RawResult({"0": 8}, job_id="task_batch#0"),
            _RawResult({"1": 7, "0": 1}, success=False, job_id="task_batch#1"),
        )
    )
    job = LogicalQubitJob(_RawJob(raw), reference=_reference(), return_batch=True)

    result = job.result(timeout=12, poll_interval=0.5)
    assert isinstance(result, ExecutionBatch)
    assert result.labels == ("zero", "one")
    assert result[0].counts == {"0": 8}
    assert result[0].num_qubits == 1
    assert result[1].num_qubits == 2
    assert not result[1].success
    assert result.failed_indices == (1,)
    assert "result" not in result.metadata
    assert job._lq_job.result_calls == [{"timeout": 12, "wait": 0.5, "verbose": False}]


def test_single_empty_counts_does_not_invent_a_shot() -> None:
    raw = _RawResult({}, shots=0)
    job = LogicalQubitJob(
        _RawJob(raw),
        reference=_reference(batch_size=None),
        return_batch=False,
    )
    result = job.result()
    assert result.num_shots == 0
    assert result.counts == {}


def test_asymmetric_bitstring_is_not_reversed_by_the_adapter() -> None:
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_order",
        batch_size=None,
        labels=("q0-one",),
        experiment_shapes=((2, 2),),
        shots=8,
        result_format="counts",
    )
    job = LogicalQubitJob(
        _RawJob(_RawResult({"10": 8}, shots=8)),
        reference=reference,
    )
    assert job.result().counts == {"10": 8}


def test_non_integral_corrected_counts_remain_available_without_truncation() -> None:
    raw = _RawResult({"0": 3, "1": 2}, shots=5)
    raw.data["counts"] = {"0": 3.9, "1": 1.1}
    raw.data["readout_correction_applied"] = True
    job = LogicalQubitJob(
        _RawJob(raw),
        reference=_reference(batch_size=None),
    )

    result = job.result()
    assert result.counts == {"0": 4, "1": 1}
    assert result.probabilities == pytest.approx({"0": 0.78, "1": 0.22})
    assert result.metadata["quasi_counts"] == {"0": 3.9, "1": 1.1}


def test_memory_result_preserves_each_shot_and_aggregates_counts() -> None:
    raw = _RawResult({}, shots=4)
    raw.data = {
        "memory": ["10", "01", "10", "10"],
        "shots": 4,
        "result_format": "memory",
        "success": True,
    }
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_memory",
        labels=("wide",),
        experiment_shapes=((12, 2),),
        shots=4,
        result_format="auto",
    )
    job = LogicalQubitJob(_RawJob(raw), reference=reference)

    result = job.result()
    assert result.memory == ["10", "01", "10", "10"]
    assert result.counts == {"01": 1, "10": 3}
    assert result.probabilities == {"01": 0.25, "10": 0.75}
    assert result.metadata["result_format"] == "memory"


def test_timeout_zero_is_non_blocking() -> None:
    raw_job = _RawJob(_RawResult(), status=_RawStatus("queued"))
    job = LogicalQubitJob(raw_job)
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0)
    assert raw_job.result_calls == []


def test_sdk_timeout_keeps_specific_exception_type() -> None:
    exceptions = pytest.importorskip("lqcloud.exceptions")
    raw_job = _RawJob(exceptions.JobTimeoutError("too slow"))
    job = LogicalQubitJob(raw_job)
    with pytest.raises(JobTimeoutError, match="too slow"):
        job.result(timeout=1)


def test_cancel_and_reference_are_preserved() -> None:
    raw_job = _RawJob(_RawResult())
    reference = _reference()
    job = LogicalQubitJob(raw_job, reference=reference, return_batch=True)
    assert job.cancel() is True
    assert raw_job.cancelled
    assert job.reference is reference


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -1.0])
def test_result_rejects_non_finite_or_negative_timeout(timeout: float) -> None:
    raw_job = _RawJob(_RawResult({"0": 8}))
    with pytest.raises(ValueError, match="timeout"):
        LogicalQubitJob(raw_job).result(timeout=timeout)
    assert raw_job.result_calls == []


def test_return_batch_without_labels_infers_sdk_batch_size() -> None:
    raw_job = _RawJob(_RawBatch([_RawResult({"0": 8}), _RawResult({"1": 8})]))
    raw_job._is_batch = True
    raw_job._batch_size = 2
    job = LogicalQubitJob(raw_job, return_batch=True)
    assert job.reference.batch_size == 2
    result = job.result()
    assert isinstance(result, ExecutionBatch)
    assert result.get_counts() == [{"0": 8}, {"1": 8}]


def test_return_batch_without_labels_single_sdk_job_is_batch_of_one() -> None:
    job = LogicalQubitJob(_RawJob(_RawResult({"0": 8})), return_batch=True)
    assert job.reference.batch_size == 1
    assert isinstance(job.result(), ExecutionBatch)


def test_return_batch_without_labels_unknown_sdk_batch_size_is_clear() -> None:
    raw_job = _RawJob(_RawBatch([_RawResult({"0": 8})]))
    raw_job._is_batch = True
    raw_job._batch_size = None
    with pytest.raises(ValueError, match="batch size"):
        LogicalQubitJob(raw_job, return_batch=True)
