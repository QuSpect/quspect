"""Tianyan job lifecycle and result conversion."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from quspect.backends._base import CloudJob, JobStatus
from quspect.backends._exceptions import JobError, JobTimeoutError, TranslationError
from quspect.backends._execution import ExecutionBatch, JobReference
from quspect.backends.tianyan._client import TianyanClient
from quspect.backends.tianyan._translator import decode_result
from quspect.simulation.result import SimulationResult

PLATFORM = "tianyan"
#: The result endpoint accepts at most this many query IDs per call.
QUERY_LIMIT = 50


class TianyanJob(CloudJob):
    """One or more Tianyan tasks (one query ID per circuit), in request order.

    ``counts`` are the measured shots (``resultStatus``).  ``probabilities``
    is the provider's ``probability`` field — the readout-corrected
    distribution on QPUs — when present, otherwise ``counts / shots``; the
    result metadata records which in ``probabilities_source``.

    The service exposes no task status: a task is "running" until its result
    appears.  Tasks cannot be cancelled.
    """

    def __init__(
        self,
        client: TianyanClient,
        query_ids: Sequence[str],
        *,
        reference: JobReference,
        return_batch: bool = False,
    ) -> None:
        ids = tuple(str(item) for item in query_ids)
        if not ids:
            raise ValueError("TianyanJob requires at least one query ID.")
        if len(reference.measurement_maps) != len(ids):
            raise ValueError("TianyanJob needs one measurement map per query ID.")
        self._client = client
        self._ids = ids
        self._reference = reference
        self._return_batch = return_batch or reference.is_batch
        self._results: list[dict[str, Any]] | None = None

    def _fetch(self) -> list[dict[str, Any]] | None:
        if self._results is not None:
            return self._results
        collected: dict[str, dict[str, Any]] = {}
        for start in range(0, len(self._ids), QUERY_LIMIT):
            chunk = self._ids[start : start + QUERY_LIMIT]
            try:
                items = self._client.results(chunk)
            except Exception as exc:
                raise JobError(str(exc), job_id=self.job_id) from exc
            if items is None:
                return None
            for position, item in enumerate(items):
                task_id = str(item.get("experimentTaskId") or chunk[position])
                collected[task_id] = item
        missing = [task_id for task_id in self._ids if task_id not in collected]
        if missing:
            raise JobError(
                f"Tianyan returned no result for task(s) {missing}.", job_id=self.job_id
            )
        self._results = [collected[task_id] for task_id in self._ids]
        return self._results

    def status(self) -> JobStatus:
        return JobStatus.COMPLETED if self._fetch() is not None else JobStatus.RUNNING

    def cancel(self) -> bool:
        """Tianyan exposes no cancellation for these tasks; returns ``False``."""
        return False

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 5.0,
    ) -> SimulationResult | ExecutionBatch:
        if timeout is not None and (timeout < 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be non-negative and finite, or None.")
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be a positive finite number.")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            items = self._fetch()
            if items is not None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise JobTimeoutError(
                    f"Tianyan job {self.job_id} is not complete.", job_id=self.job_id
                )
            sleep_for = poll_interval
            if deadline is not None:
                sleep_for = max(0.0, min(poll_interval, deadline - time.monotonic()))
            time.sleep(sleep_for)

        batch = convert_results(items, reference=self._reference, job_id=self.job_id)
        if self._return_batch:
            return batch
        return batch.single()

    @property
    def job_id(self) -> str:
        return self._ids[0]

    @property
    def job_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def reference(self) -> JobReference:
        return self._reference


def convert_results(
    items: Sequence[dict[str, Any]],
    *,
    reference: JobReference,
    job_id: str,
) -> ExecutionBatch:
    """Convert provider results (request order) into an :class:`ExecutionBatch`."""
    maps = reference.measurement_maps
    if len(items) != len(maps):
        raise JobError(
            f"Tianyan returned {len(items)} results for {len(maps)} experiments.",
            job_id=job_id,
        )
    labels = tuple(reference.labels) or tuple(
        f"experiment-{i}" for i in range(len(maps))
    )
    shapes = tuple(reference.experiment_shapes)
    results = []
    for index, item in enumerate(items):
        shape = shapes[index] if shapes else None
        num_clbits = (
            shape[1]
            if shape
            else 1 + max((c for clbits in maps[index] for c in clbits), default=-1)
        )
        try:
            counts, provider_probabilities = decode_result(
                item, maps[index], num_clbits=num_clbits
            )
        except (TranslationError, ValueError, TypeError) as exc:
            raise JobError(
                f"Cannot decode Tianyan task {item.get('experimentTaskId')}: {exc}",
                job_id=job_id,
            ) from exc
        shots = sum(counts.values())
        metadata: dict[str, Any] = {
            "platform": PLATFORM,
            "provider_job_id": str(item.get("experimentTaskId") or ""),
        }
        if provider_probabilities:
            total = sum(provider_probabilities.values())
            if any(value < 0 for value in provider_probabilities.values()):
                metadata["quasi_probabilities"] = dict(
                    sorted(provider_probabilities.items())
                )
            probabilities = (
                {k: v / total for k, v in provider_probabilities.items()}
                if total
                else {}
            )
            metadata["probabilities_source"] = "provider"
        else:
            probabilities = {k: v / shots for k, v in counts.items()} if shots else {}
            metadata["probabilities_source"] = "counts"
        results.append(
            SimulationResult(
                counts=dict(sorted(counts.items())),
                probabilities=dict(sorted(probabilities.items())),
                num_shots=shots,
                num_qubits=shape[0] if shape else len(maps[index]),
                num_clbits=num_clbits,
                metadata=metadata,
            )
        )
    return ExecutionBatch(
        results=tuple(results),
        labels=labels,
        backend_name=reference.backend_name,
        job_id=job_id,
        metadata={"platform": PLATFORM},
    )
