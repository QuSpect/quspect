"""Quafu task lifecycle and provider-result conversion."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from quspect.backends._base import CloudJob, JobStatus
from quspect.backends._exceptions import JobError, JobTimeoutError
from quspect.backends._execution import ExecutionBatch, JobReference
from quspect.backends.quafu._client import QuafuClient
from quspect.simulation.result import SimulationResult

_STATUS_MAP: dict[str, JobStatus] = {
    "finished": JobStatus.COMPLETED,
    "completed": JobStatus.COMPLETED,
    "failed": JobStatus.FAILED,
    "canceled": JobStatus.CANCELLED,
    "cancelled": JobStatus.CANCELLED,
    "running": JobStatus.RUNNING,
    "transpiled": JobStatus.QUEUED,
    "compiled": JobStatus.QUEUED,
    "queued": JobStatus.QUEUED,
    "waiting": JobStatus.QUEUED,
    "pending": JobStatus.PENDING,
}

_MEASURE_RE = re.compile(r"measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\]\s*;")
_CREG_RE = re.compile(r"creg\s+\w+\[(\d+)\]\s*;")
_QREG_RE = re.compile(r"qreg\s+\w+\[(\d+)\]\s*;")


def parse_status(raw: Any) -> JobStatus:
    return _STATUS_MAP.get(str(raw).strip().lower(), JobStatus.UNKNOWN)


def measured_clbits(qasm: str) -> tuple[int, ...]:
    """Return the sorted, distinct classical bits written by *qasm*."""
    return tuple(sorted({int(clbit) for _, clbit in _MEASURE_RE.findall(qasm)}))


def decode_bitstring(key: str, clbits: Sequence[int], num_clbits: int) -> str:
    """Convert a Quafu outcome key to QuSpect's bit order.

    Quafu keys contain only the measured classical bits, little-endian: the
    rightmost character is the lowest measured clbit. QuSpect bitstrings have
    one character per clbit with character ``i`` holding clbit ``i``;
    unmeasured clbits read ``0``.
    """
    key = str(key).strip()
    if len(key) != len(clbits) or any(char not in "01" for char in key):
        raise ValueError(
            f"Outcome {key!r} does not match {len(clbits)} measured clbit(s)."
        )
    bits = ["0"] * num_clbits
    for position, clbit in enumerate(clbits):
        bits[clbit] = key[len(key) - 1 - position]
    return "".join(bits)


def convert_result(
    payload: Mapping[str, Any],
    *,
    shape: tuple[int, int] | None = None,
    requested_shots: int | None = None,
) -> SimulationResult:
    """Convert one ``/task/result`` payload into a :class:`SimulationResult`.

    Raw counts populate ``counts``. When readout correction was requested the
    server's corrected quasi-probabilities (which may be slightly negative)
    are clipped and renormalised into ``probabilities``, and kept unmodified
    as ``metadata["quasi_probabilities"]`` / ``metadata["quasi_counts"]``.
    """
    submitted = str(payload.get("circuit") or "")
    clbits = measured_clbits(submitted)
    if shape is None:
        creg = _CREG_RE.search(submitted)
        qreg = _QREG_RE.search(submitted)
        num_clbits = int(creg.group(1)) if creg else (max(clbits) + 1 if clbits else 0)
        shape = (int(qreg.group(1)) if qreg else 0, num_clbits)
    num_qubits, num_clbits = shape
    num_clbits = max(num_clbits, (max(clbits) + 1) if clbits else 0)

    status = parse_status(payload.get("status"))
    error = str(payload.get("error") or "").strip() or None
    success = status == JobStatus.COMPLETED and error is None
    if not success and error is None:
        error = f"Quafu task ended with status {payload.get('status')!r}."

    counts: dict[str, int] = {}
    probabilities: dict[str, float] = {}
    quasi: dict[str, float] | None = None
    if success:
        try:
            for key, value in dict(payload.get("count") or {}).items():
                number = float(value)
                if not math.isfinite(number) or number < 0 or not number.is_integer():
                    raise ValueError(f"invalid count {value!r} for {key!r}")
                bitstring = decode_bitstring(key, clbits, num_clbits)
                counts[bitstring] = counts.get(bitstring, 0) + int(number)
            corrected = payload.get("corrected")
            if isinstance(corrected, Mapping) and corrected:
                quasi = {}
                for key, value in corrected.items():
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"non-finite corrected value for {key!r}")
                    bitstring = decode_bitstring(key, clbits, num_clbits)
                    quasi[bitstring] = quasi.get(bitstring, 0.0) + number
        except (TypeError, ValueError) as exc:
            success = False
            error = f"Unable to decode Quafu counts: {exc}"
            counts, quasi = {}, None

    total_counts = sum(counts.values())
    if success:
        if quasi is not None:
            clipped = {key: max(0.0, value) for key, value in quasi.items()}
            norm = sum(clipped.values())
            probabilities = (
                {key: value / norm for key, value in clipped.items() if value > 0}
                if norm > 0
                else {}
            )
        elif total_counts:
            probabilities = {key: value / total_counts for key, value in counts.items()}

    metadata: dict[str, Any] = {
        "platform": "quafu",
        "provider_job_id": str(payload.get("tid", "")),
        "chip": payload.get("chip"),
        "provider_status": payload.get("status"),
        "created": payload.get("created"),
        "finished": payload.get("finished"),
        "submitted_qasm": submitted or None,
        "executed_qasm": payload.get("transpiled") or None,
        "measured_clbits": list(clbits),
        "submitted_shots": requested_shots,
        "readout_correction_applied": quasi is not None,
    }
    if quasi is not None:
        metadata["quasi_probabilities"] = dict(sorted(quasi.items()))
        metadata["quasi_counts"] = {
            key: value * total_counts for key, value in sorted(quasi.items())
        }

    return SimulationResult(
        counts=dict(sorted(counts.items())),
        probabilities=dict(sorted(probabilities.items())),
        num_shots=total_counts if success else 0,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        statevector=None,
        success=success,
        error=error,
        metadata=metadata,
    )


class QuafuJob(CloudJob):
    """Handle to one Quafu task (one circuit)."""

    def __init__(
        self,
        client: QuafuClient,
        reference: JobReference,
        *,
        return_batch: bool = False,
    ) -> None:
        if reference.platform != "quafu":
            raise ValueError("QuafuJob requires a 'quafu' job reference.")
        self._client = client
        self._reference = reference
        self._return_batch = return_batch

    def _fetch(self) -> dict[str, Any]:
        payload = self._client.get(f"/task/result/{self.job_id}")
        return dict(payload) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _is_terminal(payload: Mapping[str, Any]) -> bool:
        if not payload:
            return False
        if str(payload.get("error") or "").strip():
            return True
        return parse_status(payload.get("status")) in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> SimulationResult | ExecutionBatch:
        """Poll until the task finishes and return its converted result."""
        if timeout is not None and (timeout < 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be non-negative and finite, or None.")
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be a positive finite number.")

        start = time.monotonic()
        while True:
            payload = self._fetch()
            if self._is_terminal(payload):
                break
            if self.status() == JobStatus.CANCELLED:
                payload = {"status": "Canceled", "tid": self.job_id}
                break
            elapsed = time.monotonic() - start
            if timeout is not None and elapsed >= timeout:
                raise JobTimeoutError(
                    f"Quafu task {self.job_id} did not finish within {timeout} s.",
                    job_id=self.job_id,
                )
            delay = poll_interval
            if timeout is not None:
                delay = min(delay, max(0.0, timeout - elapsed))
            time.sleep(delay)

        shapes = tuple(self._reference.experiment_shapes)
        result = convert_result(
            payload,
            shape=shapes[0] if shapes else None,
            requested_shots=self._reference.shots,
        )
        if self._return_batch:
            labels = tuple(self._reference.labels) or ("experiment-0",)
            return ExecutionBatch(
                results=(result,),
                labels=labels,
                backend_name=self._reference.backend_name,
                job_id=self.job_id,
                metadata={"platform": "quafu"},
            )
        if not result.success:
            raise JobError(
                result.error or f"Quafu task {self.job_id} failed.",
                job_id=self.job_id,
            )
        return result

    def status(self) -> JobStatus:
        return parse_status(self._client.get(f"/task/status/{self.job_id}"))

    def cancel(self) -> bool:
        """Request cancellation of an unfinished task.

        Returns ``False`` for a task that already reached a terminal state.
        Quafu acknowledges a cancel request without a body, so ``True`` means
        the request was sent, not that the task stopped.
        """
        if self.done():
            return False
        self._client.get(f"/task/cancel/{self.job_id}")
        return True

    @property
    def job_id(self) -> str:
        return self._reference.job_id

    @property
    def reference(self) -> JobReference:
        return self._reference

    def __repr__(self) -> str:
        backend = self._reference.backend_name
        return f"QuafuJob(job_id={self.job_id!r}, backend={backend!r})"
