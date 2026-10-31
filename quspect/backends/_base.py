"""Abstract base classes for the Backend subsystem.

Defines the unified :class:`Backend` interface that both local simulators
and cloud QPU backends implement, plus the async :class:`CloudJob` handle
for remote execution.
"""

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quspect.backends._execution import (
        BackendCapabilities,
        ExecutionBatch,
        ExecutionRequest,
        JobReference,
    )
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.simulation.result import SimulationResult


class JobStatus(enum.Enum):
    """Status of a cloud job."""

    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def _request_workload_id(request: ExecutionRequest) -> str | None:
    value = request.metadata.get("workload_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Execution metadata workload_id must be a non-empty string.")
    return value.strip()


class Backend(ABC):
    """Unified interface for all computation backends.

    Every backend — local simulator or remote QPU — implements ``run()``
    so that user code can switch between them without changing the
    calling convention.

    Example:
        >>> sim = Simulator(seed=42)
        >>> sim.run(circuit, shots=1024)
        SimulationResult(...)

        >>> provider = LogicalQubitProvider()
        >>> backend = provider.get_backend("QZ02")
        >>> backend.run(circuit, shots=2000)
        SimulationResult(...)
    """

    @abstractmethod
    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        save_statevector: bool = False,
        **options: Any,
    ) -> SimulationResult:
        """Execute *circuit* and return results synchronously.

        For cloud backends this blocks until the remote job completes.

        Args:
            circuit: The circuit to execute. All parameters must be bound.
            shots: Number of measurement repetitions.
            save_statevector: If True and supported, include the final
                state vector in the result. Cloud backends ignore this.
            **options: Backend-specific options (e.g. ``initial_layout``).

        Returns:
            A :class:`~quspect.simulation.SimulationResult` with counts
            and probabilities.
        """

    def execute(self, request: ExecutionRequest) -> ExecutionBatch:
        """Execute a labelled circuit request and preserve batch order.

        Local backends use this default implementation, which binds and runs
        every experiment synchronously. Cloud backends override it to use a
        provider-native batch job.
        """
        from quspect.backends._execution import ExecutionBatch, ExecutionRequest

        if not isinstance(request, ExecutionRequest):
            raise TypeError(
                "Backend.execute expects ExecutionRequest, "
                f"got {type(request).__name__}."
            )
        effective_format = (
            "counts" if request.result_format == "auto" else request.result_format
        )
        if effective_format not in self.capabilities.result_formats:
            raise ValueError(
                f"Backend {self.name!r} does not support result format "
                f"{request.result_format!r}."
            )

        run_options = dict(request.options)
        run_options.setdefault("result_format", effective_format)
        results = tuple(
            self.run(circuit, shots=request.shots, **run_options)
            for circuit in request.bound_circuits()
        )
        return ExecutionBatch(
            results=results,
            labels=request.labels,
            backend_name=self.name,
            metadata={
                "execution": "synchronous",
                **dict(request.metadata),
            },
        )

    def submit_request(self, request: ExecutionRequest) -> ExecutionJob:
        """Submit a request through the common job interface.

        Simulator jobs complete immediately. Remote backends override this
        method and return a live :class:`CloudJob`.
        """
        return CompletedJob(self, request)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the execution features supported by this backend."""
        from quspect.backends._execution import BackendCapabilities

        return BackendCapabilities(
            supports_sampling=True,
            supports_batch=True,
            supports_statevector=False,
            max_circuits_per_job=None,
            max_shots=None,
            supported_modes=frozenset({"circuit"}),
            result_formats=frozenset({"counts"}),
        )


class ExecutionJob(ABC):
    """Provider-neutral handle for a submitted execution request.

    Both an immediate simulator execution and a remote QPU submission expose
    this lifecycle. Call :meth:`result` to wait and retrieve the result.
    """

    @abstractmethod
    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> SimulationResult | ExecutionBatch:
        """Block until the job completes and return results.

        Args:
            timeout: Maximum seconds to wait. ``None`` means no limit.

        Returns:
            A :class:`~quspect.simulation.SimulationResult`.

        Raises:
            JobTimeoutError: If *timeout* expires before completion.
            JobError: If the job failed on the remote platform.
        """

    @abstractmethod
    def status(self) -> JobStatus:
        """Return the current job status."""

    @abstractmethod
    def cancel(self) -> bool:
        """Request cancellation and report whether it was accepted."""

    @property
    @abstractmethod
    def job_id(self) -> str:
        """Return the platform-assigned job identifier."""

    @property
    @abstractmethod
    def reference(self) -> JobReference:
        """Return a credential-free, serializable reference to this job."""

    @property
    def job_ids(self) -> tuple[str, ...]:
        """Return all platform IDs represented by this handle."""
        return (self.job_id,)

    def done(self) -> bool:
        """Return whether the job has reached a terminal state."""
        return self.status() in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }


class CompletedJob(ExecutionJob):
    """An in-process job that has already completed on a local backend."""

    def __init__(self, backend: Backend, request: ExecutionRequest) -> None:
        from quspect.backends._execution import JobReference

        self._job_id = f"local-{uuid.uuid4().hex}"
        self._result = backend.execute(request)
        self._reference = JobReference(
            platform="local",
            backend_name=backend.name,
            job_id=self._job_id,
            transport="in-process",
            batch_size=len(request.experiments),
            labels=request.labels,
            experiment_shapes=request.experiment_shapes,
            shots=request.shots,
            result_format=request.result_format,
            workload_id=_request_workload_id(request),
        )

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> ExecutionBatch:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.COMPLETED

    def cancel(self) -> bool:
        return False

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def reference(self) -> JobReference:
        return self._reference


class CloudJob(ExecutionJob):
    """Handle to a job submitted to a cloud quantum platform."""


class CompositeJob(CloudJob):
    """One logical job composed of several provider jobs in stable order."""

    def __init__(self, jobs: Sequence[CloudJob]) -> None:
        from quspect.backends._execution import JobReference

        self._jobs = tuple(jobs)
        if not self._jobs:
            raise ValueError("CompositeJob requires at least one child job.")
        if not all(isinstance(job, CloudJob) for job in self._jobs):
            raise TypeError("CompositeJob children must be CloudJob instances.")

        references = tuple(job.reference for job in self._jobs)
        first = references[0]
        if any(
            reference.platform != first.platform
            or reference.backend_name != first.backend_name
            or reference.base_url != first.base_url
            or reference.transport != first.transport
            for reference in references[1:]
        ):
            raise ValueError(
                "CompositeJob children must use one platform, backend, "
                "base URL, and transport."
            )
        if any(
            reference.workload_id != first.workload_id for reference in references[1:]
        ):
            raise ValueError("CompositeJob children must use one workload ID.")

        reserved_labels = {
            label for reference in references for label in reference.labels
        }
        labels_list: list[str] = []
        used_labels: set[str] = set()
        global_index = 0
        for reference in references:
            reference_size = reference.batch_size or 1
            if reference.labels:
                reference_labels = tuple(reference.labels)
            else:
                generated: list[str] = []
                for local_index in range(reference_size):
                    base = f"experiment-{global_index + local_index}"
                    label = base
                    suffix = 1
                    while label in used_labels or label in reserved_labels:
                        label = f"{base}-{suffix}"
                        suffix += 1
                    generated.append(label)
                    used_labels.add(label)
                reference_labels = tuple(generated)
            labels_list.extend(reference_labels)
            used_labels.update(reference_labels)
            global_index += reference_size
        labels = tuple(labels_list)
        if len(set(labels)) != len(labels):
            raise ValueError("CompositeJob child labels must be globally unique.")
        shapes = tuple(
            shape for reference in references for shape in reference.experiment_shapes
        )
        batch_size = sum(reference.batch_size or 1 for reference in references)
        all_job_ids: list[str] = []
        child_batch_sizes: list[int] = []
        for job, reference in zip(self._jobs, references, strict=True):
            if reference.child_job_ids:
                all_job_ids.extend(reference.child_job_ids)
                child_batch_sizes.extend(
                    reference.child_batch_sizes or (1,) * len(reference.child_job_ids)
                )
            else:
                all_job_ids.append(job.job_id)
                child_batch_sizes.append(reference.batch_size or 1)
        shots = (
            first.shots
            if all(reference.shots == first.shots for reference in references)
            else None
        )
        result_format = (
            first.result_format
            if all(
                reference.result_format == first.result_format
                for reference in references
            )
            else None
        )
        self._job_id = f"{all_job_ids[0]}+{len(all_job_ids)}"
        self._reference = JobReference(
            platform=first.platform,
            backend_name=first.backend_name,
            job_id=all_job_ids[0],
            base_url=first.base_url,
            transport=first.transport,
            batch_size=batch_size,
            labels=labels,
            experiment_shapes=shapes,
            shots=shots,
            result_format=result_format,
            workload_id=first.workload_id,
            child_job_ids=tuple(all_job_ids),
            child_batch_sizes=tuple(child_batch_sizes),
        )

    def result(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> ExecutionBatch:
        """Wait for every child and concatenate their result batches."""
        import math
        import time

        from quspect.backends._execution import ExecutionBatch

        if timeout is not None and (timeout < 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be non-negative and finite, or None.")
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be a positive finite number.")

        start = time.monotonic()
        results: list[SimulationResult] = []
        for job in self._jobs:
            remaining = None
            if timeout is not None:
                remaining = max(0.0, timeout - (time.monotonic() - start))
            child = job.result(timeout=remaining, poll_interval=poll_interval)
            if isinstance(child, ExecutionBatch):
                results.extend(child.results)
            else:
                results.append(child)
        return ExecutionBatch(
            results=tuple(results),
            labels=self._reference.labels,
            backend_name=self._reference.backend_name,
            job_id=self.job_id,
            metadata={
                "platform": self._reference.platform,
                "child_job_ids": self.job_ids,
            },
        )

    def status(self) -> JobStatus:
        statuses = tuple(job.status() for job in self._jobs)

        # An active child keeps the logical job active even when another child
        # has already failed or been cancelled.  Final-state precedence only
        # applies once every child has reached a terminal state.
        if any(status == JobStatus.RUNNING for status in statuses):
            return JobStatus.RUNNING
        if any(status == JobStatus.QUEUED for status in statuses):
            return JobStatus.QUEUED
        if any(status == JobStatus.PENDING for status in statuses):
            return JobStatus.PENDING

        terminal = {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
        if not all(status in terminal for status in statuses):
            return JobStatus.UNKNOWN
        if any(status == JobStatus.FAILED for status in statuses):
            return JobStatus.FAILED
        if any(status == JobStatus.CANCELLED for status in statuses):
            return JobStatus.CANCELLED
        return JobStatus.COMPLETED

    def done(self) -> bool:
        """Return True only after every child has reached a terminal state."""
        return all(job.done() for job in self._jobs)

    def cancel(self) -> bool:
        accepted = True
        for job in self._jobs:
            try:
                child_accepted = job.cancel()
            except Exception:
                child_accepted = False
            accepted = child_accepted and accepted
        return accepted

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(self._reference.child_job_ids)

    @property
    def reference(self) -> JobReference:
        return self._reference


class CloudBackend(Backend):
    """A quantum backend hosted on a cloud platform.

    Extends :class:`Backend` with async submission and platform metadata
    that is specific to remote QPUs.
    """

    @abstractmethod
    def submit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        **options: Any,
    ) -> CloudJob:
        """Submit *circuit* asynchronously and return a job handle.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement repetitions.
            **options: Backend-specific options.

        Returns:
            A :class:`CloudJob` that can be polled for status and results.
        """

    @abstractmethod
    def submit_request(self, request: ExecutionRequest) -> CloudJob:
        """Submit a labelled single- or multi-circuit request."""

    def execute(self, request: ExecutionRequest) -> ExecutionBatch:
        """Submit a cloud request and block until its full batch completes."""
        from quspect.backends._execution import ExecutionBatch
        from quspect.simulation.result import SimulationResult

        result = self.submit_request(request).result()
        if isinstance(result, ExecutionBatch):
            return result
        if isinstance(result, SimulationResult):
            return ExecutionBatch(
                results=(result,),
                labels=request.labels,
                backend_name=self.name,
            )
        raise TypeError(
            f"Cloud job returned unsupported result type {type(result).__name__}."
        )

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Identifier for the cloud platform (e.g. ``"logicalqubit"``)."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of physical qubits on this QPU."""

    @property
    @abstractmethod
    def native_gate_set(self) -> set[str]:
        """Set of gate names natively supported by this QPU."""
