"""Origin Quantum authentication, backend discovery and job recovery."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from quspect.backends._base import CloudJob, CompositeJob
from quspect.backends._exceptions import AuthenticationError, BackendNotFoundError
from quspect.backends._execution import JobReference
from quspect.backends.originq._backend import (
    NON_SAMPLING_BACKENDS,
    QPU_MACHINE_TYPE,
    SIMULATOR_MACHINE_TYPE,
    OriginQBackend,
)
from quspect.backends.originq._client import OriginQClient, Transport
from quspect.backends.originq._job import PLATFORM, OriginQJob

API_KEY_ENV = "QPANDA_QCLOUD_API_KEY"
#: Service used by pyqpanda3 when a key does not embed its own URL.
DEFAULT_URL = "http://pyqanda-admin.qpanda.cn"


def url_from_api_key(api_key: str) -> str | None:
    """Return the service URL embedded in a new-format Origin Quantum key.

    New keys are hex: a 4-digit length ``n``, then ``n`` ASCII bytes of the
    service URL, then the secret.  Returns ``None`` for any other format.
    Only the URL is returned; the secret part is never exposed.
    """
    text = api_key.strip()
    try:
        length = int(text[:4], 16)
        url = bytes.fromhex(text[4 : 4 + 2 * length]).decode("ascii")
    except (ValueError, UnicodeDecodeError):
        return None
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return url


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


class OriginQProvider:
    """Authenticate with the Origin Quantum cloud (本源量子云).

    Talks to the cloud's HTTP API directly; pyqpanda3 is not required.

    Args:
        api_key: API key from the Origin Quantum cloud console.  Defaults to
            the ``QPANDA_QCLOUD_API_KEY`` environment variable.
        url: Optional service URL override.  New-format keys embed the URL,
            so it is normally not needed.
        transport: Optional HTTP transport (testing).

    Warning:
        The service URL embedded in current keys is plain ``http``; the key
        travels unencrypted, exactly as with pyqpanda3.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        url: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise AuthenticationError(
                f"No Origin Quantum API key: pass api_key or set {API_KEY_ENV}."
            )
        self._url = url or url_from_api_key(key) or DEFAULT_URL
        self._client = OriginQClient(key, self._url, transport=transport)

    @property
    def url(self) -> str:
        """Service URL (credential-free)."""
        return self._url

    def _resources(self) -> dict[str, dict[str, Any]]:
        """Sampling-capable resources by name (full-amplitude simulator, QPUs)."""
        by_name: dict[str, dict[str, Any]] = {}
        for entry in self._client.list_resources():
            name = entry.get("pyqpandaChipId")
            machine_type = entry.get("machineType")
            if not name or machine_type not in (
                SIMULATOR_MACHINE_TYPE,
                QPU_MACHINE_TYPE,
            ):
                continue
            by_name.setdefault(str(name), entry)
        return by_name

    def list_backends(self) -> dict[str, bool]:
        """Return ``{backend name: currently online}`` for sampling backends."""
        return {
            name: bool(entry.get("onlineFlag")) and bool(entry.get("available", True))
            for name, entry in self._resources().items()
        }

    def get_backend(self, name: str) -> OriginQBackend:
        """Return a QPU (e.g. ``"WK_C180_2"``) or ``"full_amplitude"``."""
        if name in NON_SAMPLING_BACKENDS:
            raise ValueError(
                f"{name!r} returns amplitudes, not samples; it is not supported."
            )
        resource = self._resources().get(name)
        if resource is None:
            raise BackendNotFoundError(f"Unknown Origin Quantum backend {name!r}.")
        return OriginQBackend(self._client, resource, provider=self)

    def retrieve_job(
        self,
        reference: JobReference | str,
        *,
        backend_name: str | None = None,
        measurement_maps: Sequence[Sequence[Sequence[int]]] = (),
        labels: Sequence[str] = (),
        experiment_shapes: Sequence[tuple[int, int]] = (),
        shots: int | None = None,
        batch_size: int | None = None,
    ) -> CloudJob:
        """Recreate a job handle from a :class:`JobReference` (preferred).

        A raw task ID is accepted only together with ``backend_name`` and the
        ``measurement_maps`` recorded at submission: Origin Quantum results
        cannot be decoded into classical bits without them.
        """
        if isinstance(reference, JobReference):
            if reference.platform != PLATFORM:
                raise ValueError(
                    "OriginQProvider cannot recover a job from platform "
                    f"{reference.platform!r}."
                )
            descriptor = reference
        elif isinstance(reference, str) and reference.strip():
            if not backend_name or not measurement_maps:
                raise ValueError(
                    "Retrieving by task ID requires backend_name and "
                    "measurement_maps; prefer the JobReference."
                )
            descriptor = JobReference(
                platform=PLATFORM,
                backend_name=backend_name,
                job_id=reference,
                base_url=self.url,
                batch_size=batch_size,
                labels=tuple(labels),
                experiment_shapes=tuple(experiment_shapes),
                shots=shots,
                measurement_maps=measurement_maps,
            )
        else:
            raise TypeError("reference must be JobReference or a non-empty task ID.")

        if not descriptor.measurement_maps:
            raise ValueError("The job reference carries no measurement_maps.")
        if descriptor.base_url and _origin(descriptor.base_url) != _origin(self.url):
            raise ValueError(
                "Job reference endpoint does not match this provider's service URL."
            )
        if descriptor.child_job_ids:
            return self._retrieve_composite(descriptor)
        return self._retrieve_one(descriptor)

    def _retrieve_one(self, reference: JobReference) -> CloudJob:
        return OriginQJob(
            self._client,
            reference.job_id,
            reference=reference,
            return_batch=reference.is_batch,
        )

    def _retrieve_composite(self, reference: JobReference) -> CloudJob:
        sizes = reference.child_batch_sizes or (1,) * len(reference.child_job_ids)
        children: list[CloudJob] = []
        offset = 0
        for child_id, size in zip(reference.child_job_ids, sizes, strict=True):
            end = offset + size
            children.append(
                self._retrieve_one(
                    JobReference(
                        platform=reference.platform,
                        backend_name=reference.backend_name,
                        job_id=child_id,
                        base_url=reference.base_url,
                        transport=reference.transport,
                        batch_size=size,
                        labels=reference.labels[offset:end],
                        experiment_shapes=reference.experiment_shapes[offset:end],
                        shots=reference.shots,
                        result_format=reference.result_format,
                        workload_id=reference.workload_id,
                        measurement_maps=reference.measurement_maps[offset:end],
                    )
                )
            )
            offset = end
        if offset != reference.batch_size:
            raise ValueError(
                "Recovered child batch sizes do not match the parent reference."
            )
        return CompositeJob(children)

    def __repr__(self) -> str:
        return f"OriginQProvider(url={self.url!r})"
