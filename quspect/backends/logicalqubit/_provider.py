"""LogicalQubit authentication, backend discovery, and job recovery."""

from __future__ import annotations

import os
import warnings
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
            "Install it with: pip install 'quspect[logicalqubit]' "
            "(lqcloud 0.6 is published for Python 3.11 and 3.12 only)."
        )


def _check_use_async(use_async: bool | None) -> None:
    """Reject the removed synchronous route; warn that the flag is obsolete.

    The LogicalQubit cloud removed its synchronous submission endpoint and
    lqcloud 0.6 submits and polls every job through the asynchronous task
    route. ``use_async`` is kept only so existing calls keep working.
    """
    if use_async is None:
        return
    if not use_async:
        raise ValueError(
            "use_async=False is no longer supported: the LogicalQubit cloud "
            "removed its synchronous submission endpoint. Drop the argument."
        )
    warnings.warn(
        "use_async has no effect (every LogicalQubit job is asynchronous) and "
        "will be removed; drop the argument.",
        DeprecationWarning,
        stacklevel=3,
    )


def _warn_if_environment_overrides(api_key: str | None, url: str | None) -> None:
    """Report explicit arguments that lqcloud >= 0.6 will silently ignore.

    lqcloud 0.6 gives ``LQCLOUD_API_KEY`` / ``LQCLOUD_TOKEN`` and
    ``LQCLOUD_URL`` precedence over constructor arguments. Submitting under a
    different account or endpoint than the one written in code must not go
    unnoticed.
    """
    env_key = os.environ.get("LQCLOUD_API_KEY")
    env_token = os.environ.get("LQCLOUD_TOKEN")
    if api_key and (env_key or env_token) and env_key != api_key:
        warnings.warn(
            "LogicalQubitProvider(api_key=...) is overridden by the "
            "LQCLOUD_API_KEY / LQCLOUD_TOKEN environment variable "
            "(lqcloud >= 0.6 gives environment credentials precedence). "
            "Unset the variable to use the explicit key.",
            UserWarning,
            stacklevel=3,
        )
    env_url = os.environ.get("LQCLOUD_URL")
    if url and env_url and env_url.rstrip("/") != url.rstrip("/"):
        warnings.warn(
            f"LogicalQubitProvider(url={url!r}) is overridden by "
            f"LQCLOUD_URL={env_url!r} (lqcloud >= 0.6 gives the environment "
            "precedence). Unset the variable to use the explicit URL.",
            UserWarning,
            stacklevel=3,
        )


class LogicalQubitProvider:
    """Authenticate with LogicalQubit and create recoverable backend jobs.

    Only explicitly supplied arguments are forwarded to ``lqcloud``, which
    resolves the rest from ``LQCLOUD_API_KEY`` / ``LQCLOUD_TOKEN`` /
    ``LQCLOUD_URL`` and then ``~/.lqcloud/config.json``. Since lqcloud 0.6
    those environment variables take precedence over explicit arguments; a
    ``UserWarning`` is emitted when that discards an argument. QuSpect always
    disables interactive prompts.

    Args:
        api_key: LogicalQubit API key.
        url: Platform URL.
        store: lqcloud's local job store, which records submitted circuits
            and fetched results under ``~/.lqcloud/store`` and is enabled by
            default since lqcloud 0.6. Pass ``False`` to disable it, ``True``
            or a directory (or an ``lqcloud.JobStore``) to enable it
            explicitly. ``None`` leaves the choice to lqcloud
            (``LQCLOUD_STORE``, then ``~/.lqcloud/config.json``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        *,
        store: Any = None,
    ) -> None:
        _require_lqcloud()

        _warn_if_environment_overrides(api_key, url)
        provider_options: dict[str, Any] = {"interactive": False}
        if api_key:
            provider_options["api_key"] = api_key
        if url:
            provider_options["url"] = url
        if store is not None:
            provider_options["store"] = store
        self._provider = lqcloud.LQCloudProvider(**provider_options)

    @property
    def url(self) -> str:
        return str(self._provider.url)

    def get_backend(
        self,
        name: str,
        *,
        use_async: bool | None = None,
        verify: bool = True,
    ) -> LogicalQubitBackend:
        """Return a named QPU.

        Args:
            name: Backend name, e.g. ``"QZ02"``.
            use_async: Deprecated and ignored: every job is asynchronous.
                ``False`` raises ``ValueError`` because the synchronous route
                no longer exists.
            verify: Check the name against :meth:`list_backends` first.
        """
        from quspect.backends.logicalqubit._backend import LogicalQubitBackend

        _check_use_async(use_async)
        lq_backend = self._provider.get_backend(name, verify=verify)
        return LogicalQubitBackend(lq_backend, provider=self)

    def list_backends(self) -> list[dict[str, Any]]:
        """Return metadata for every backend visible to this account."""
        return list(self._provider.get_backends())

    def retrieve_job(
        self,
        reference: JobReference | str,
        *,
        backend_name: str | None = None,
        use_async: bool | None = None,
        batch_size: int | None = None,
        labels: Sequence[str] = (),
        experiment_shapes: Sequence[tuple[int, int]] = (),
        shots: int | None = None,
        result_format: str | None = None,
    ) -> CloudJob:
        """Recreate a live job handle without persisting any credentials.

        Passing a :class:`JobReference` is preferred. A raw job ID is accepted
        for interoperability, but the backend name and any batch context then
        have to be supplied separately. ``use_async`` is deprecated and
        ignored, as in :meth:`get_backend`.

        References saved with ``transport="sync"`` by older QuSpect versions
        are still accepted; lqcloud polls every job through the asynchronous
        task endpoint.
        """
        _require_lqcloud()
        _check_use_async(use_async)
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
                transport="async",
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
        """Recover QuSpect's chunk jobs (or older SDK fallback children) in order."""
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
