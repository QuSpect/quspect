"""Tianyan authentication, backend discovery and job recovery."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from quspect.backends._base import CloudJob
from quspect.backends._exceptions import AuthenticationError, BackendNotFoundError
from quspect.backends._execution import JobReference
from quspect.backends.tianyan._backend import NON_SAMPLING, TianyanBackend, status_name
from quspect.backends.tianyan._client import DEFAULT_URL, TianyanClient, Transport
from quspect.backends.tianyan._job import PLATFORM, TianyanJob

LOGIN_KEY_ENV = "TIANYAN_LOGIN_KEY"


def _origin(value: str | None) -> tuple[str, str, int] | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname.lower(), parsed.port or default_port


class TianyanProvider:
    """Authenticate with the Tianyan quantum cloud (中电信天衍量子计算云平台).

    Talks to the cloud's HTTPS API directly; cqlib is not required.

    Args:
        login_key: The "API Token" from the personal center of
            https://qc.zdxlz.com.  Defaults to ``TIANYAN_LOGIN_KEY``.
        url: Optional service URL override.
        transport: Optional HTTP transport (testing).
        submit_retries: Retries for a submission that hit a gateway timeout
            (see :class:`TianyanClient`); ``0`` never risks a duplicate task.
    """

    def __init__(
        self,
        login_key: str | None = None,
        *,
        url: str | None = None,
        transport: Transport | None = None,
        submit_retries: int = 3,
    ) -> None:
        key = login_key or os.environ.get(LOGIN_KEY_ENV)
        if not key:
            raise AuthenticationError(
                f"No Tianyan login key: pass login_key or set {LOGIN_KEY_ENV}."
            )
        self._client = TianyanClient(
            key,
            url or DEFAULT_URL,
            transport=transport,
            submit_retries=submit_retries,
        )

    @property
    def url(self) -> str:
        return self._client.base_url

    def _machines(self) -> dict[str, dict[str, Any]]:
        machines: dict[str, dict[str, Any]] = {}
        for entry in self._client.list_machines():
            name = entry.get("code") or entry.get("machineName")
            if name:
                machines.setdefault(str(name), entry)
        return machines

    def list_backends(self) -> dict[str, str]:
        """Return ``{machine name: status}`` (``running``, ``calibrating``, ...)."""
        return {
            name: status_name(entry.get("status"))
            for name, entry in self._machines().items()
            if name not in NON_SAMPLING
        }

    def get_backend(self, name: str) -> TianyanBackend:
        """Return a QPU (e.g. ``"tianyan176"``) or simulator (``"tianyan_sw"``)."""
        if name in NON_SAMPLING:
            raise ValueError(f"{name!r} computes single amplitudes, not samples.")
        machine = self._machines().get(name)
        if machine is None:
            raise BackendNotFoundError(f"Unknown Tianyan backend {name!r}.")
        return TianyanBackend(self._client, machine, provider=self)

    def retrieve_job(self, reference: JobReference) -> CloudJob:
        """Recreate a job handle from a :class:`JobReference`.

        The reference carries every query ID and the measurement maps needed
        to decode the results; no other state is required.
        """
        if not isinstance(reference, JobReference):
            raise TypeError("reference must be a JobReference.")
        if reference.platform != PLATFORM:
            raise ValueError(
                f"TianyanProvider cannot recover a job from {reference.platform!r}."
            )
        if not reference.measurement_maps:
            raise ValueError("The job reference carries no measurement_maps.")
        if reference.base_url and _origin(reference.base_url) != _origin(self.url):
            raise ValueError(
                "Job reference endpoint does not match this provider's service URL."
            )
        query_ids = tuple(reference.child_job_ids) or (reference.job_id,)
        return TianyanJob(
            self._client,
            query_ids,
            reference=reference,
            return_batch=reference.is_batch,
        )

    def __repr__(self) -> str:
        return f"TianyanProvider(url={self.url!r})"
