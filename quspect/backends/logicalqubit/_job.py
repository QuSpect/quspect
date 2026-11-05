"""LogicalQubit job lifecycle and provider-result conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from quspect.backends._base import CloudJob, JobStatus
from quspect.backends._exceptions import JobError, JobTimeoutError
from quspect.backends._execution import ExecutionBatch, JobReference
from quspect.simulation.result import SimulationResult

_STATUS_MAP: dict[str, JobStatus] = {
    "queued": JobStatus.QUEUED,
    "pending": JobStatus.PENDING,
    "retrying": JobStatus.RUNNING,
    "running": JobStatus.RUNNING,
    "completed": JobStatus.COMPLETED,
    "failed": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "canceled": JobStatus.CANCELLED,
}


class LogicalQubitJob(CloudJob):
    """Adapt an ``lqcloud.Job`` to QuSpect's provider-neutral job API.

    When ``reference`` is omitted it is built from the SDK job.  With
    ``return_batch=True`` the batch size is taken from ``labels`` when given,
    otherwise inferred from the SDK job: its declared batch size for an SDK
    batch job (falling back to ``experiment_shapes``), or ``1`` for a
    single-circuit job.  A ``ValueError`` is raised when an SDK batch job does
    not expose a usable batch size.

    Readout-corrected results follow lqcloud 0.6: ``counts`` (and ``memory``)
    hold the raw measured outcomes, ``probabilities`` the server-corrected
    distribution, and ``metadata["quasi_counts"]`` its unrounded
    ``probability * shots`` values.
    """

    def __init__(
        self,
        lq_job: Any,
        *,
        reference: JobReference | None = None,
        labels: Sequence[str] = (),
        experiment_shapes: Sequence[tuple[int, int]] = (),
        requested_shots: int | None = None,
        result_format: str | None = None,
        return_batch: bool = False,
    ) -> None:
        self._lq_job = lq_job
        if reference is None:
            is_sdk_batch = bool(getattr(lq_job, "_is_batch", False))
            batch_size = getattr(lq_job, "_batch_size", None) if is_sdk_batch else None
            if return_batch:
                batch_size = _infer_batch_size(
                    lq_job,
                    labels=labels,
                    experiment_shapes=experiment_shapes,
                    is_sdk_batch=is_sdk_batch,
                    sdk_batch_size=batch_size,
                )
            reference = JobReference(
                platform="logicalqubit",
                backend_name=str(getattr(lq_job, "backend_name", "") or "unknown"),
                job_id=str(lq_job.job_id),
                base_url=getattr(lq_job, "base_url", None),
                # lqcloud 0.6 only has the asynchronous task route.
                transport="async",
                batch_size=batch_size,
                labels=tuple(labels),
                experiment_shapes=tuple(experiment_shapes),
                shots=requested_shots,
                result_format=result_format,
            )
        self._reference = reference
        self._return_batch = return_batch or reference.is_batch

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> SimulationResult | ExecutionBatch:
        """Wait silently and convert every provider child result in order."""
        if timeout is not None and (timeout < 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be non-negative and finite, or None.")
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be a positive finite number.")

        # lqcloud treats timeout=0 as unlimited. Give QuSpect the
        # conventional non-blocking meaning instead.
        if timeout == 0 and not self.done():
            raise JobTimeoutError(
                f"LogicalQubit job {self.job_id} is not complete.",
                job_id=self.job_id,
            )

        try:
            raw_result = self._lq_job.result(
                timeout=None if timeout == 0 else timeout,
                wait=poll_interval,
                verbose=False,
            )
        except Exception as exc:
            raise _map_sdk_exception(exc, self.job_id) from exc

        batch = _convert_result(
            raw_result,
            labels=self._reference.labels,
            experiment_shapes=self._reference.experiment_shapes,
            requested_shots=self._reference.shots,
            backend_name=self._reference.backend_name,
            job_id=self.job_id,
        )
        if self._return_batch:
            return batch
        result = batch.single()
        if not result.success:
            raise JobError(
                result.error or f"LogicalQubit job {self.job_id} failed.",
                error_type=str(result.metadata.get("error_type") or "") or None,
                job_id=self.job_id,
            )
        return result

    def status(self) -> JobStatus:
        """Return the real provider status without masking query failures."""
        try:
            raw = self._lq_job.status()
        except Exception as exc:
            raise _map_sdk_exception(exc, self.job_id) from exc
        value = getattr(raw, "value", raw)
        return _STATUS_MAP.get(str(value).strip().lower(), JobStatus.UNKNOWN)

    def cancel(self) -> bool:
        """Request cancellation and preserve the provider's acceptance flag."""
        try:
            return bool(self._lq_job.cancel())
        except Exception as exc:
            raise _map_sdk_exception(exc, self.job_id) from exc

    @property
    def job_id(self) -> str:
        return str(self._lq_job.job_id)

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(self._reference.child_job_ids) or (self.job_id,)

    @property
    def reference(self) -> JobReference:
        return self._reference


def _infer_batch_size(
    lq_job: Any,
    *,
    labels: Sequence[str],
    experiment_shapes: Sequence[tuple[int, int]],
    is_sdk_batch: bool,
    sdk_batch_size: object,
) -> int:
    """Resolve the batch size of a ``return_batch=True`` job reference."""
    if labels:
        return len(labels)
    if is_sdk_batch:
        if (
            isinstance(sdk_batch_size, int)
            and not isinstance(sdk_batch_size, bool)
            and sdk_batch_size > 0
        ):
            return sdk_batch_size
        if experiment_shapes:
            return len(experiment_shapes)
        raise ValueError(
            f"Cannot infer the batch size of LogicalQubit batch job "
            f"{lq_job.job_id}: the SDK job reports no positive batch size; "
            "pass labels explicitly."
        )
    if experiment_shapes:
        return len(experiment_shapes)
    return 1


def _map_sdk_exception(exc: Exception, job_id: str) -> JobError:
    """Map lqcloud's non-uniform hierarchy without importing it eagerly."""
    try:
        from lqcloud.exceptions import JobError as LQJobError
        from lqcloud.exceptions import JobTimeoutError as LQJobTimeoutError
    except ImportError:
        return JobError(str(exc), job_id=job_id)

    if isinstance(exc, LQJobTimeoutError):
        return JobTimeoutError(str(exc), job_id=job_id)
    if isinstance(exc, LQJobError):
        return JobError(
            str(exc),
            error_type=getattr(exc, "error_type", None),
            job_id=getattr(exc, "job_id", None) or job_id,
        )
    if isinstance(exc, JobError):
        return exc
    return JobError(str(exc), job_id=job_id)


def _convert_result(
    lq_result: Any,
    *,
    labels: Sequence[str] = (),
    experiment_shapes: Sequence[tuple[int, int]] = (),
    requested_shots: int | None = None,
    backend_name: str = "",
    job_id: str | None = None,
) -> ExecutionBatch:
    """Convert one SDK Result or an entire BatchResult without truncation."""
    if lq_result is None or isinstance(lq_result, str):
        raise JobError(
            f"LogicalQubit job {job_id or '<unknown>'} returned no circuit result.",
            job_id=job_id,
        )

    raw_children = getattr(lq_result, "results", None)
    children = tuple(raw_children) if isinstance(raw_children, list) else (lq_result,)
    if not children:
        raise JobError(
            f"LogicalQubit job {job_id or '<unknown>'} returned an empty batch.",
            job_id=job_id,
        )

    normalized_labels = (
        tuple(labels)
        if labels
        else tuple(f"experiment-{index}" for index in range(len(children)))
    )
    if len(normalized_labels) != len(children):
        raise JobError(
            "LogicalQubit result count does not match the submitted experiments: "
            f"{len(children)} != {len(normalized_labels)}.",
            job_id=job_id,
        )
    shapes = tuple(experiment_shapes)
    if shapes and len(shapes) != len(children):
        raise JobError(
            "LogicalQubit result shapes do not match the returned batch size.",
            job_id=job_id,
        )

    converted = tuple(
        _convert_child(
            child,
            shape=shapes[index] if shapes else None,
            requested_shots=requested_shots,
        )
        for index, child in enumerate(children)
    )
    metadata_source = getattr(lq_result, "metadata", {})
    batch_metadata = {
        "platform": "logicalqubit",
        **_compact_metadata(metadata_source),
    }
    return ExecutionBatch(
        results=converted,
        labels=normalized_labels,
        backend_name=backend_name or str(getattr(lq_result, "backend_name", "")),
        job_id=job_id or str(getattr(lq_result, "job_id", "")) or None,
        metadata=batch_metadata,
    )


def _convert_child(
    child: Any,
    *,
    shape: tuple[int, int] | None,
    requested_shots: int | None,
) -> SimulationResult:
    data = getattr(child, "data", {})
    data = dict(data) if isinstance(data, Mapping) else {}
    status = str(getattr(child, "status", "")).strip().lower()
    child_success = data.get("success", True) is not False
    success = status in {"", "completed"} and child_success
    error = data.get("error") or data.get("message")
    error_type = data.get("error_type")
    raw_memory = data.get("memory")
    memory = (
        [str(bitstring) for bitstring in raw_memory]
        if isinstance(raw_memory, list)
        else None
    )

    measured: dict[str, float] = {}
    corrected: dict[str, float] | None = None
    if success:
        try:
            raw_counts = _measured_counts(child, data)
            raw_corrected = _corrected_distribution(data)
        except Exception as exc:
            success = False
            error = f"Unable to decode provider counts: {exc}"
        else:
            try:
                measured = _numeric_distribution(raw_counts)
                if raw_corrected is not None:
                    corrected = _numeric_distribution(raw_corrected)
            except _InvalidCountError as exc:
                success = False
                error = str(exc)
                measured, corrected = {}, None
    counts = {key: max(0, int(round(value))) for key, value in measured.items()}

    raw_shots = data.get("shots", requested_shots)
    if isinstance(raw_shots, bool):
        raw_shots = None
    try:
        num_shots = (
            int(raw_shots) if raw_shots is not None else int(sum(counts.values()))
        )
    except (TypeError, ValueError):
        num_shots = int(sum(counts.values()))
    num_shots = max(0, num_shots)

    # With readout correction the corrected distribution is what the
    # probabilities (and therefore QuSpect's estimators) report.
    distribution = corrected if corrected is not None else measured
    total = sum(distribution.values())
    probabilities = (
        {key: value / total for key, value in distribution.items()} if total > 0 else {}
    )

    inferred_width = len(next(iter(counts), ""))
    num_qubits, num_clbits = shape or (inferred_width, inferred_width)
    metadata: dict[str, Any] = {
        "platform": "logicalqubit",
        "provider_job_id": str(getattr(child, "job_id", "")),
        "result_format": data.get(
            "result_format", "memory" if memory is not None else "counts"
        ),
        "error_type": error_type,
        "readout_correction_requested": data.get("readout_correction_requested"),
        "readout_correction_applied": data.get("readout_correction_applied"),
        **_compact_metadata(getattr(child, "metadata", {})),
    }
    if corrected is not None:
        metadata["quasi_counts"] = dict(corrected)
    elif any(not value.is_integer() or value < 0 for value in measured.values()):
        metadata["quasi_counts"] = dict(measured)

    return SimulationResult(
        counts=dict(sorted(counts.items())) if success else {},
        probabilities=dict(sorted(probabilities.items())) if success else {},
        num_shots=num_shots,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        statevector=None,
        success=success,
        error=str(error) if error else None,
        metadata=metadata,
        memory=memory if success else None,
    )


class _InvalidCountError(ValueError):
    """A provider count or corrected value that is not a finite number."""


def _numeric_distribution(raw: Mapping[str, Any]) -> dict[str, float]:
    """Read a provider distribution without lqcloud's float-to-int coercion."""
    values: dict[str, float] = {}
    for bitstring, raw_count in raw.items():
        try:
            numeric = float(raw_count)
        except (TypeError, ValueError):
            raise _InvalidCountError(
                f"Invalid count for outcome {bitstring!r}: {raw_count!r}."
            ) from None
        if not math.isfinite(numeric):
            raise _InvalidCountError(f"Non-finite count for outcome {bitstring!r}.")
        values[str(bitstring)] = numeric
    return values


def _is_memory_payload(data: Mapping[str, Any]) -> bool:
    return data.get("result_format") == "memory" or "memory" in data


def _measured_counts(child: Any, data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the raw measured counts, as lqcloud 0.6 ``Result.get_counts()``.

    With readout correction the server stores the corrected distribution
    under ``counts`` and the measured one under ``raw_counts`` (counts mode),
    or attaches ``corrected_counts`` next to the raw ``memory`` (memory mode).
    A corrected payload without ``raw_counts`` has nothing better to offer
    than its ``counts``.
    """
    if _is_memory_payload(data):
        memory = data.get("memory", [])
        if isinstance(memory, list):
            aggregated: dict[str, int] = {}
            for bitstring in memory:
                key = str(bitstring)
                aggregated[key] = aggregated.get(key, 0) + 1
            return aggregated
    raw_counts = data.get("raw_counts")
    if data.get("readout_correction_applied") and isinstance(raw_counts, Mapping):
        return raw_counts
    counts = data.get("counts")
    if isinstance(counts, Mapping):
        return counts
    fallback = child.get_counts()
    if not isinstance(fallback, Mapping):
        raise TypeError("Provider result did not contain a counts mapping.")
    return fallback


def _corrected_distribution(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the server's corrected ``probability * shots`` values, if any."""
    if not data.get("readout_correction_applied"):
        return None
    source = data.get("corrected_counts" if _is_memory_payload(data) else "counts")
    return source if isinstance(source, Mapping) else None


def _compact_metadata(value: Any) -> dict[str, Any]:
    """Keep useful job metadata while avoiding duplicated raw result payloads."""
    if not isinstance(value, Mapping):
        return {}
    excluded = {"result", "command_data", "stdout", "stderr", "artifacts"}
    return {str(key): item for key, item in value.items() if key not in excluded}
