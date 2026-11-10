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
    ``probabilities`` therefore holds the provider distribution and
    ``counts`` is derived as ``round(probability * shots)``; the result
    metadata records ``counts_source="probabilities"``.
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


def parse_task_result(detail: Mapping[str, Any]) -> list[dict[str, float]]:
    """Return one ``{hex key: value}`` distribution per submitted program."""
    raw = detail.get("taskResult")
    if not isinstance(raw, list):
        raise JobError("Origin Quantum task has no taskResult list.")
    distributions: list[dict[str, float]] = []
    for item in raw:
        entry = json.loads(item) if isinstance(item, str) else item
        if not isinstance(entry, Mapping):
            raise JobError("Origin Quantum taskResult entry is not an object.")
        keys, values = entry.get("key"), entry.get("value")
        if (
            not isinstance(keys, list)
            or not isinstance(values, list)
            or len(keys) != len(values)
        ):
            raise JobError("Origin Quantum taskResult entry has mismatched keys.")
        distribution: dict[str, float] = {}
        for key, value in zip(keys, values, strict=True):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise JobError(f"Non-finite Origin Quantum value for {key!r}.")
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
    distributions = parse_task_result(detail)
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
        metadata["quasi_probabilities"] = dict(sorted(values.items()))
    total = sum(values.values())
    probabilities = (
        {key: value / total for key, value in values.items()} if total > 0 else {}
    )
    num_shots = shots or 0
    counts: dict[str, int] = {}
    for key, value in probabilities.items():
        count = int(round(value * num_shots))
        if count > 0:
            counts[key] = count
    return SimulationResult(
        counts=dict(sorted(counts.items())),
        probabilities=dict(sorted(probabilities.items())),
        num_shots=num_shots,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        metadata=metadata,
    )
