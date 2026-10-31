"""LogicalQubit authentication, backend discovery, and job recovery."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from quspect.backends._base import CloudJob, CompositeJob
from quspect.backends._exceptions import OptionalDependencyError
from quspect.backends._execution import JobReference

if TYPE_CHECKING:
    from quspect.backends.logicalqubit._backend import LogicalQubitBackend

try:
    import lqcloud

    _HAS_LQCLOUD = True
except ImportError:
    _HAS_LQCLOUD = False
    lqcloud = None


def _require_lqcloud() -> None:
    if not _HAS_LQCLOUD:
        raise OptionalDependencyError(
            "The lqcloud SDK is required to use LogicalQubit backends. "
            "Install it with: pip install 'quspect[logicalqubit]'"
        )


class LogicalQubitProvider:
    """Authenticate with LogicalQubit and create recoverable backend jobs.

    Explicit arguments override SDK configuration, but omitted values are not
    replaced. This lets ``lqcloud`` honor ``~/.lqcloud/config.json`` (including
    a custom endpoint) while QuSpect always disables interactive prompts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
    ) -> None:
        _require_lqcloud()

        resolved_key = api_key or os.environ.get("LQCLOUD_API_KEY")
        resolved_url = url or os.environ.get("LQCLOUD_URL")
        provider_options: dict[str, Any] = {"interactive": False}
        if resolved_key:
            provider_options["api_key"] = resolved_key
        if resolved_url:
            provider_options["url"] = resolved_url
        self._provider = lqcloud.LQCloudProvider(**provider_options)

    @property
    def url(self) -> str:
        return str(self._provider.url)

    def get_backend(
        self,
        name: str,
        *,
        use_async: bool = True,
        verify: bool = True,
    ) -> LogicalQubitBackend:
        """Return a named QPU with an explicit SDK transport route."""
        from quspect.backends.logicalqubit._backend import LogicalQubitBackend

        lq_backend = self._provider.get_backend(
            name,
            use_async=use_async,
            verify=verify,
        )
        return LogicalQubitBackend(lq_backend, provider=self)

    def list_backends(self) -> list[dict[str, Any]]:
        """Return metadata for every backend visible to this account."""
        return list(self._provider.get_backends())

    def retrieve_job(
        self,
        reference: JobReference | str,
        *,
        backend_name: str | None = None,
        use_async: bool = True,
        batch_size: int | None = None,
        labels: Sequence[str] = (),
        experiment_shapes: Sequence[tuple[int, int]] = (),
        shots: int | None = None,
        result_format: str | None = None,
    ) -> CloudJob:
        """Recreate a live job handle without persisting any credentials.

        Passing a :class:`JobReference` is preferred. A raw job ID is accepted
        for interoperability, but the backend name and any batch context then
        have to be supplied separately.
        """
        _require_lqcloud()
        if isinstance(reference, JobReference):
            if reference.platform != "logicalqubit":
                raise ValueError(
                    "LogicalQubitProvider cannot recover a job from platform "
                    f"{reference.platform!r}."
                )
            descriptor = reference
        elif isinstance(reference, str) and reference.strip():
            if not backend_name:
                raise ValueError("backend_name is required when retrieving by job ID.")
            descriptor = JobReference(
                platform="logicalqubit",
                backend_name=backend_name,
                job_id=reference,
                base_url=self.url,
                transport="async" if use_async else "sync",
                batch_size=batch_size,
                labels=tuple(labels),
                experiment_shapes=tuple(experiment_shapes),
                shots=shots,
                result_format=result_format,
            )
        else:
            raise TypeError("reference must be JobReference or a non-empty job ID.")

        if descriptor.transport not in {"async", "sync"}:
            raise ValueError(
                "LogicalQubit job references require 'async' or 'sync' transport."
            )
        if descriptor.base_url and not _same_origin(descriptor.base_url, self.url):
            raise ValueError(
                "LogicalQubit job reference endpoint must use the provider's "
                "configured origin."
            )

        if descriptor.child_job_ids:
            sizes = descriptor.child_batch_sizes or (1,) * len(descriptor.child_job_ids)
            return self._retrieve_composite(descriptor, sizes)
        return self._retrieve_one(descriptor)

    def _retrieve_one(self, reference: JobReference) -> CloudJob:
        from lqcloud.job import Job

        from quspect.backends.logicalqubit._job import LogicalQubitJob

        raw_job = Job(
            reference.job_id,
            self._provider,
            backend_name=reference.backend_name,
            base_url=reference.base_url or self.url,
            use_async=reference.transport != "sync",
        )
        if reference.is_batch:
            raw_job._is_batch = True
            raw_job._batch_size = reference.batch_size
        return LogicalQubitJob(
            raw_job,
            reference=reference,
            return_batch=reference.is_batch,
        )

    def _retrieve_composite(
        self,
        reference: JobReference,
        child_sizes: Sequence[int],
    ) -> CloudJob:
        """Recover native chunks or SDK fallback children in original order."""
        children: list[CloudJob] = []
        offset = 0
        for child_id, child_size in zip(
            reference.child_job_ids,
            child_sizes,
            strict=True,
        ):
            next_offset = offset + child_size
            child_reference = JobReference(
                platform=reference.platform,
                backend_name=reference.backend_name,
                job_id=child_id,
                base_url=reference.base_url,
                transport=reference.transport,
                batch_size=child_size,
                labels=reference.labels[offset:next_offset],
                experiment_shapes=reference.experiment_shapes[offset:next_offset],
                shots=reference.shots,
                result_format=reference.result_format,
                workload_id=reference.workload_id,
            )
            children.append(self._retrieve_one(child_reference))
            offset = next_offset
        if offset != reference.batch_size:
            raise ValueError(
                "Recovered child batch sizes do not match the parent reference."
            )
        return CompositeJob(children)

    def __repr__(self) -> str:
        return f"LogicalQubitProvider(url={self.url!r})"


def _same_origin(candidate: str, configured: str) -> bool:
    """Compare URL origins before an authenticated SDK session is reused."""

    def origin(value: str) -> tuple[str, str, int] | None:
        try:
            parsed = urlsplit(value)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                return None
            default_port = 443 if parsed.scheme.lower() == "https" else 80
            return (
                parsed.scheme.lower(),
                parsed.hostname.lower(),
                parsed.port or default_port,
            )
        except ValueError:
            return None

    candidate_origin = origin(candidate)
    return candidate_origin is not None and candidate_origin == origin(configured)
