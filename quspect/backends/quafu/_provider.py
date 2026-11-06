"""Quafu-SQC authentication, chip discovery, and job recovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from quspect.backends._base import CloudJob, CompositeJob
from quspect.backends._exceptions import (
    AuthenticationError,
    BackendError,
    BackendNotFoundError,
)
from quspect.backends._execution import JobReference
from quspect.backends.quafu._backend import QuafuBackend
from quspect.backends.quafu._client import QuafuClient, Transport
from quspect.backends.quafu._job import QuafuJob


class QuafuProvider:
    """Entry point to the Quafu-SQC superconducting quantum cloud.

    No Quafu SDK is needed; requests go straight to the HTTP API.

    Args:
        token: API token from https://quafu-sqc.baqis.ac.cn (valid 30 days).
            Falls back to ``QUAFU_API_TOKEN``, ``QPU_API_TOKEN``, then the
            JSON file ``~/.quafu/api`` (``{"token": "..."}``).
        url: Platform URL (default ``https://quafu-sqc.baqis.ac.cn``, or
            ``QUAFU_URL``).
        timeout: Per-request HTTP timeout in seconds.
        retries: Extra attempts for GET requests after a network error or
            5xx answer (task submission is never retried).
        verify: Check the token immediately.

    Example:
        >>> provider = QuafuProvider()
        >>> provider.list_backends()
        {'Baihua': 12, 'Haituo': 0, 'Hongluo': 'Calibrating', ...}
        >>> backend = provider.get_backend("Haituo")
        >>> backend.run(circuit, shots=1024).counts
    """

    def __init__(
        self,
        token: str | None = None,
        url: str | None = None,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        verify: bool = True,
        transport: Transport | None = None,
    ) -> None:
        self._client = QuafuClient(
            token, url, timeout=timeout, retries=retries, transport=transport
        )
        self._expiry: str | None = None
        if verify:
            self.verify()

    @property
    def url(self) -> str:
        return self._client.url

    @property
    def client(self) -> QuafuClient:
        return self._client

    def verify(self) -> str:
        """Validate the token and return its expiry time as reported by Quafu."""
        payload = self._client.get("/task/verify")
        if isinstance(payload, (list, tuple)) and payload:
            self._expiry = str(payload[0])
            return self._expiry
        raise AuthenticationError(f"Quafu rejected the API token: {payload}")

    @property
    def token_expiry(self) -> str | None:
        """Token expiry from the last :meth:`verify` call."""
        return self._expiry

    def list_backends(self) -> dict[str, int | str]:
        """Return ``{chip: queue length or state}`` (e.g. ``"Calibrating"``)."""
        payload = self._client.get("/task/status/0")
        if not isinstance(payload, Mapping):
            raise BackendError(
                f"Unexpected Quafu backend listing: {str(payload)[:200]}"
            )
        return dict(payload)

    def get_backend(
        self,
        name: str,
        *,
        verify: bool = True,
        min_coupler_fidelity: float = 0.0,
    ) -> QuafuBackend:
        """Return chip *name* (e.g. ``"Haituo"``, ``"Baihua"``, ``"Dongling"``)."""
        if verify:
            available = self.list_backends()
            if name not in available:
                raise BackendNotFoundError(
                    f"Quafu has no chip {name!r}; available: {sorted(available)}."
                )
        return QuafuBackend(
            name,
            self._client,
            provider=self,
            min_coupler_fidelity=min_coupler_fidelity,
        )

    def retrieve_job(
        self,
        reference: JobReference | str | int,
        *,
        backend_name: str | None = None,
        label: str | None = None,
        experiment_shape: tuple[int, int] | None = None,
    ) -> CloudJob:
        """Recreate a job handle from a saved reference or a task id."""
        if isinstance(reference, JobReference):
            if reference.platform != "quafu":
                raise ValueError(
                    f"QuafuProvider cannot recover a {reference.platform!r} job."
                )
            if reference.base_url and not _same_origin(reference.base_url, self.url):
                raise ValueError("Job reference belongs to a different Quafu endpoint.")
            if reference.child_job_ids:
                return self._retrieve_composite(reference)
            return QuafuJob(self._client, reference, return_batch=reference.is_batch)

        if isinstance(reference, bool) or not isinstance(reference, (str, int)):
            raise TypeError("reference must be a JobReference or a Quafu task id.")
        tid = str(reference).strip()
        if not tid.isdigit():
            raise ValueError(f"Quafu task ids are integers, got {reference!r}.")
        descriptor = JobReference(
            platform="quafu",
            backend_name=backend_name or "unknown",
            job_id=tid,
            base_url=self.url,
            transport="async",
            labels=(label,) if label else (),
            experiment_shapes=(experiment_shape,) if experiment_shape else (),
        )
        return QuafuJob(self._client, descriptor, return_batch=False)

    def _retrieve_composite(self, reference: JobReference) -> CloudJob:
        sizes: Sequence[int] = reference.child_batch_sizes or (1,) * len(
            reference.child_job_ids
        )
        if any(size != 1 for size in sizes):
            raise ValueError("Quafu tasks always hold exactly one experiment.")
        children: list[CloudJob] = []
        for index, child_id in enumerate(reference.child_job_ids):
            children.append(
                QuafuJob(
                    self._client,
                    JobReference(
                        platform="quafu",
                        backend_name=reference.backend_name,
                        job_id=child_id,
                        base_url=reference.base_url,
                        transport=reference.transport,
                        batch_size=1,
                        labels=tuple(reference.labels[index : index + 1]),
                        experiment_shapes=tuple(
                            reference.experiment_shapes[index : index + 1]
                        ),
                        shots=reference.shots,
                        result_format=reference.result_format,
                        workload_id=reference.workload_id,
                    ),
                    return_batch=True,
                )
            )
        return CompositeJob(children)

    def __repr__(self) -> str:
        return f"QuafuProvider(url={self.url!r})"


def _same_origin(a: str, b: str) -> bool:
    def origin(value: str) -> tuple[str, str, int | None]:
        parts = urlsplit(value)
        return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port

    return origin(a) == origin(b)
