"""Origin Quantum job lifecycle and result conversion."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from quspect.backends._base import CloudJob, JobStatus
from quspect.backends._exceptions import JobError, JobTimeoutError, TranslationError
from quspect.backends._execution import ExecutionBatch, JobReference
from quspect.backends.originq._client import OriginQClient
from quspect.backends.originq._translator import decode_distribution
from quspect.simulation.result import SimulationResult

PLATFORM = "originq"

# ``taskState`` values (pyqpanda3 JobStatus: WAITING=1, COMPUTING=2,
# FINISHED=3, FAILED=4, QUEUING=5).  The SDK has no cancelled member, and
# which ``taskState`` the cloud reports for a cancelled task has not been
# observed, so none is mapped to ``JobStatus.CANCELLED``: any other value
# stays ``UNKNOWN``.  (pyqpanda3 0.5.1 itself reports every value outside
# 1-5 as FAILED with "returned unsupported taskState".)
_STATUS_MAP: dict[str, JobStatus] = {
    "1": JobStatus.PENDING,
    "2": JobStatus.RUNNING,
    "3": JobStatus.COMPLETED,
    "4": JobStatus.FAILED,
    "5": JobStatus.QUEUED,
}


def map_status(task_state: Any) -> JobStatus:
    """Map a provider ``taskState`` to :class:`JobStatus`."""
    return _STATUS_MAP.get(str(task_state).strip(), JobStatus.UNKNOWN)


class OriginQJob(CloudJob):
    """One Origin Quantum task (a single circuit or a native QPU batch).

    The reference must carry one ``measurement_maps`` entry per experiment:
    Origin Quantum reports results by measured qubit, and only the
    submission-time map can turn them back into QuSpect classical bits.

    Origin Quantum returns probabilities rather than shot counts (also on
    real QPUs, where readout correction is applied server-side by default).
    ``probabilities`` therefore holds the provider distribution, with
    negative (readout-corrected) values clipped to zero and the rest
    renormalized.  ``counts`` allocates the requested shots to it by the
    largest-remainder rule, so the counts sum to the shots; the result
    metadata records ``counts_source="probabilities"`` and, when the
    provider reported a negative value, its unmodified values as
    ``quasi_probabilities``.

    A completed task whose result cannot be decoded raises
    :class:`~quspect.backends.JobError` chained to the decoding error.
    """

    def __init__(
        self,
        client: OriginQClient,
        task_id: str,
        *,
        reference: JobReference,
        return_batch: bool = False,
    ) -> None:
        if not reference.measurement_maps:
            raise ValueError("OriginQJob requires reference.measurement_maps.")
        self._client = client
        self._task_id = task_id
        self._reference = reference
        self._return_batch = return_batch or reference.is_batch

    def _detail(self) -> dict[str, Any]:
        try:
            return self._client.task_detail(self._task_id)
        except Exception as exc:
            raise JobError(str(exc), job_id=self._task_id) from exc

    def status(self) -> JobStatus:
        return map_status(self._detail().get("taskState"))

    def cancel(self) -> bool:
        """Ask Origin Quantum to cancel the task (a batch is one task).

        Sends a single request, without querying the status first and
        without retrying.  Returns ``True`` when the cloud accepted the
        request and ``False`` when it answered that the task cannot be
        cancelled.  ``True`` does not mean the task has stopped.

        Which ``taskState`` a cancelled task reports is not known: afterwards
        :meth:`status` returns whatever the cloud reports, ``UNKNOWN`` for a
        value outside pyqpanda3's five states, and :meth:`result` keeps
        polling while it does, so pass a ``timeout``.

        Raises:
            JobError: The request failed (network, HTTP error, rejected key
                or an unreadable answer); whether the task was cancelled is
                then unknown.
        """
        try:
            return self._client.cancel_task(self._task_id)
        except Exception as exc:
            raise JobError(str(exc), job_id=self._task_id) from exc

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> SimulationResult | ExecutionBatch:
        if timeout is not None and (timeout < 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be non-negative and finite, or None.")
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be a positive finite number.")

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            detail = self._detail()
            status = map_status(detail.get("taskState"))
            if status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise JobTimeoutError(
                    f"Origin Quantum job {self._task_id} is not complete "
                    f"(status: {status.value}).",
                    job_id=self._task_id,
                )
            sleep_for = poll_interval
            if deadline is not None:
                sleep_for = max(0.0, min(poll_interval, deadline - time.monotonic()))
            time.sleep(sleep_for)

        if status == JobStatus.FAILED:
            message = detail.get("errInfo") or detail.get("errCode") or "job failed"
            raise JobError(
                f"Origin Quantum job {self._task_id} failed: {message}",
                job_id=self._task_id,
            )
        batch = convert_result(detail, reference=self._reference, job_id=self._task_id)
        if self._return_batch:
            return batch
        return batch.single()

    @property
    def job_id(self) -> str:
        return self._task_id

    @property
    def reference(self) -> JobReference:
        return self._reference


def parse_task_result(
    detail: Mapping[str, Any], *, job_id: str | None = None
) -> list[dict[str, float]]:
    """Return one ``{hex key: value}`` distribution per submitted program.

    Raises:
        JobError: If ``taskResult`` is missing or malformed.  A decoding
            failure (invalid JSON, a value that is not a number) is chained
            as the error's ``__cause__``.
    """
    raw = detail.get("taskResult")
    if not isinstance(raw, list):
        raise JobError("Origin Quantum task has no taskResult list.", job_id=job_id)
    distributions: list[dict[str, float]] = []
    for item in raw:
        try:
            entry = json.loads(item) if isinstance(item, str) else item
        except ValueError as exc:
            raise JobError(
                f"Origin Quantum taskResult entry is not valid JSON: {exc}",
                job_id=job_id,
            ) from exc
        if not isinstance(entry, Mapping):
            raise JobError(
                "Origin Quantum taskResult entry is not an object.", job_id=job_id
            )
        keys, values = entry.get("key"), entry.get("value")
        if (
            not isinstance(keys, list)
            or not isinstance(values, list)
            or len(keys) != len(values)
        ):
            raise JobError(
                "Origin Quantum taskResult entry has mismatched keys.", job_id=job_id
            )
        distribution: dict[str, float] = {}
        for key, value in zip(keys, values, strict=True):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise JobError(
                    f"Origin Quantum taskResult value for {key!r} is not a "
                    f"number: {value!r}.",
                    job_id=job_id,
                ) from exc
            if not math.isfinite(numeric):
                raise JobError(
                    f"Non-finite Origin Quantum value for {key!r}.", job_id=job_id
                )
            distribution[str(key)] = distribution.get(str(key), 0.0) + numeric
        distributions.append(distribution)
    return distributions


def convert_result(
    detail: Mapping[str, Any],
    *,
    reference: JobReference,
    job_id: str,
) -> ExecutionBatch:
    """Convert a finished ``taskDetail`` into an ordered :class:`ExecutionBatch`."""
    maps = reference.measurement_maps
    distributions = parse_task_result(detail, job_id=job_id)
    if len(distributions) != len(maps):
        raise JobError(
            "Origin Quantum result count does not match the submitted "
            f"experiments: {len(distributions)} != {len(maps)}.",
            job_id=job_id,
        )
    labels = tuple(reference.labels) or tuple(
        f"experiment-{index}" for index in range(len(maps))
    )
    shapes = tuple(reference.experiment_shapes)
    results = tuple(
        _convert_one(
            distribution,
            clbit_map=maps[index],
            shape=shapes[index] if shapes else None,
            shots=reference.shots,
            job_id=job_id,
        )
        for index, distribution in enumerate(distributions)
    )
    metadata = {
        "platform": PLATFORM,
        "chip_id": detail.get("chipId"),
        "qpu_run_time": detail.get("qpuRunTime"),
        "total_time": detail.get("totalTime"),
    }
    return ExecutionBatch(
        results=results,
        labels=labels,
        backend_name=reference.backend_name,
        job_id=job_id,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def _convert_one(
    raw: Mapping[str, float],
    *,
    clbit_map: Sequence[Sequence[int]],
    shape: tuple[int, int] | None,
    shots: int | None,
    job_id: str,
) -> SimulationResult:
    num_clbits = (
        shape[1]
        if shape
        else 1 + max((c for clbits in clbit_map for c in clbits), default=-1)
    )
    num_qubits = shape[0] if shape else len(clbit_map)
    try:
        values = decode_distribution(raw, clbit_map, num_clbits)
    except TranslationError as exc:
        raise JobError(str(exc), job_id=job_id) from exc

    metadata: dict[str, Any] = {
        "platform": PLATFORM,
        "provider_job_id": job_id,
        "counts_source": "probabilities",
    }
    if any(value < 0 for value in values.values()):
        # Readout correction can return slightly negative quasi-probabilities.
        # Keep the provider's values; clip and renormalize the distribution.
        metadata["quasi_probabilities"] = dict(sorted(values.items()))
    clipped = {key: max(0.0, value) for key, value in values.items()}
    total = sum(clipped.values())
    probabilities = (
        {key: value / total for key, value in clipped.items()} if total > 0 else {}
    )
    num_shots = shots or 0
    counts = _allocate_shots(probabilities, num_shots)
    return SimulationResult(
        counts=dict(sorted(counts.items())),
        probabilities=dict(sorted(probabilities.items())),
        num_shots=num_shots,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        metadata=metadata,
    )


def _allocate_shots(probabilities: Mapping[str, float], shots: int) -> dict[str, int]:
    """Turn a distribution into counts by the largest-remainder rule.

    Every outcome first gets ``floor(probability * shots)``; the shots that
    are left go one each to the outcomes with the largest fractional parts
    (ties: bitstring order).  Unlike rounding each outcome on its own, the
    counts sum to ``shots`` for any normalized distribution.
    """
    if shots <= 0 or not probabilities:
        return {}
    quotas = {key: value * shots for key, value in probabilities.items()}
    counts = {key: math.floor(quota) for key, quota in quotas.items()}
    remaining = shots - sum(counts.values())
    by_remainder = sorted(quotas, key=lambda key: (counts[key] - quotas[key], key))
    for key in by_remainder[: max(0, remaining)]:
        counts[key] += 1
    return {key: count for key, count in counts.items() if count > 0}
