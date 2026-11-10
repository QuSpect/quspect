"""Minimal Origin Quantum cloud HTTP client (standard library only).

The cloud has no public API documentation; this client speaks the protocol
used by pyqpanda3 0.5.1, recorded from its traffic in November 2026:

``GET  /oqcs/task/getResourceList.json``
    Backends: ``pyqpandaChipId``, ``onlineFlag``, ``machineType``, ...
``POST /oqcs/task/origin/chipInfo.json``  ``{"ChipID": name}``
    Chip data: ``AvailableQubits``, ``adjMatrix``, ``BasicQGate``, ...
``POST /oqcs/task/origin/submit.json``
    ``{"QProg": [OriginIR, ...], "Configuration": {...}, ...}`` → ``taskId``.
    Header ``task_mode`` is ``1`` for one program and ``2`` for a batch.
``POST /oqcs/task/origin/taskDetail.json``  ``{"taskId": id}``
    ``taskState`` and ``taskResult``.  Requires a ``task_mode`` header;
    pyqpanda3 sends ``0`` when it does not know how the task was submitted.
``POST /oqcs/task/cancelTask.json``  ``{"taskId": id}``
    Ask the cloud to cancel a task, with header ``task_mode: 0`` however the
    task was submitted.  The answer carries no payload: ``success`` is
    ``true`` when the request was accepted and ``false`` when the task
    cannot be cancelled.  Unlike the endpoints above this one was not
    recorded from live traffic: the request is the one
    ``pyqpanda3.qcloud.QCloudJob.cancel()`` sends to a stand-in server, and
    the answer is read the way the SDK reads it.

Every request is authenticated with ``Authorization: oqcs_auth=<api key>``.
Responses wrap their payload as ``{"success", "code", "message", "obj"}``
with ``code == 10000`` on success.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from quspect.backends._exceptions import AuthenticationError, BackendError

_logger = logging.getLogger(__name__)

#: ``(method, url, headers, body) -> (status, response bytes)``
Transport = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]

_OK = 10000
_RESOURCES = "/oqcs/task/getResourceList.json"
_CHIP_INFO = "/oqcs/task/origin/chipInfo.json"
_SUBMIT = "/oqcs/task/origin/submit.json"
_TASK_DETAIL = "/oqcs/task/origin/taskDetail.json"
_CANCEL = "/oqcs/task/cancelTask.json"


def urllib_transport(timeout: float = 30.0) -> Transport:
    """Return the default transport, built on :mod:`urllib.request`."""

    def send(
        method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url, data=body, headers=dict(headers), method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()

    return send


class OriginQClient:
    """Authenticated access to the five Origin Quantum task endpoints.

    Args:
        api_key: Origin Quantum API key (sent verbatim in ``Authorization``).
        base_url: Service URL, e.g. ``http://pyqanda-admin.qpanda.cn``.
        transport: Optional transport for testing.
        retries: Extra attempts for idempotent reads on network errors or
            HTTP 5xx.  Submissions are never retried (a retry could create a
            duplicate task), and neither are cancellations.
        backoff: Initial delay between retries in seconds (doubles).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        transport: Transport | None = None,
        retries: int = 3,
        backoff: float = 1.0,
    ) -> None:
        if not api_key:
            raise AuthenticationError("An Origin Quantum API key is required.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or urllib_transport()
        self._retries = max(0, int(retries))
        self._backoff = float(backoff)

    @property
    def base_url(self) -> str:
        return self._base_url

    def __repr__(self) -> str:  # never show the key
        return f"OriginQClient(base_url={self._base_url!r})"

    # ------------------------------------------------------------------

    def list_resources(self) -> list[dict[str, Any]]:
        result = self._call("GET", _RESOURCES, None, idempotent=True, wrapped=False)
        if not isinstance(result, list):
            raise BackendError("Origin Quantum resource list has an unexpected shape.")
        return [dict(item) for item in result if isinstance(item, Mapping)]

    def chip_info(self, chip_id: str) -> dict[str, Any]:
        obj = self._call("POST", _CHIP_INFO, {"ChipID": chip_id}, idempotent=True)
        if not isinstance(obj, Mapping):
            raise BackendError(f"Origin Quantum returned no chip info for {chip_id!r}.")
        if obj.get("errCode") not in (None, 0):
            raise BackendError(
                f"Origin Quantum chip info error for {chip_id!r}: "
                f"{obj.get('errInfo') or obj.get('errCode')}"
            )
        return dict(obj)

    def submit(
        self,
        programs: Sequence[str],
        *,
        machine_type: str,
        configuration: Mapping[str, Any],
        chip_id: str | None = None,
        qubit_num: int | None = None,
        clbit_num: int | None = None,
        batch: bool = False,
    ) -> str:
        """Submit OriginIR programs and return the provider task ID."""
        body: dict[str, Any] = {
            "QProg": list(programs),
            "batchNo": "",
            "taskFrom": 4,
            "QMachineType": str(machine_type),
            "Configuration": dict(configuration),
        }
        if chip_id is not None:
            body["ChipID"] = chip_id
        if not batch:
            body["qubitNum"] = int(qubit_num or 0)
            body["classicalbitNum"] = int(clbit_num or 0)
        obj = self._call(
            "POST",
            _SUBMIT,
            body,
            idempotent=False,
            task_mode="2" if batch else "1",
        )
        task_id = obj.get("taskId") if isinstance(obj, Mapping) else None
        if not isinstance(task_id, str) or not task_id:
            raise BackendError("Origin Quantum accepted the task but returned no ID.")
        return task_id

    def task_detail(self, task_id: str) -> dict[str, Any]:
        obj = self._call(
            "POST", _TASK_DETAIL, {"taskId": task_id}, idempotent=True, task_mode="0"
        )
        if not isinstance(obj, Mapping):
            raise BackendError(f"Origin Quantum returned no detail for task {task_id}.")
        return dict(obj)

    def cancel_task(self, task_id: str) -> bool:
        """Ask the cloud to cancel a task and return whether it accepted.

        One request, never retried.  ``True`` means Origin Quantum accepted
        the request, not that the task has already stopped; ``False`` means it
        answered ``success: false``, which pyqpanda3 documents as "the task
        cannot be cancelled".  As in the SDK, ``success`` alone decides.

        Raises:
            AuthenticationError: The API key was rejected (HTTP 401/403).
            BackendError: Network failure, any other HTTP error, or an
                answer without a boolean ``success``: whether the task was
                cancelled is then unknown.
        """
        data = self._call(
            "POST",
            _CANCEL,
            {"taskId": task_id},
            idempotent=False,
            task_mode="0",
            wrapped=False,
        )
        success = data.get("success") if isinstance(data, Mapping) else None
        if not isinstance(success, bool):
            raise BackendError(
                f"Origin Quantum {_CANCEL} returned no boolean 'success' "
                f"for task {task_id}."
            )
        if not success:
            _logger.info(
                "Origin Quantum declined to cancel task %s (code %s): %s",
                task_id,
                data.get("code"),
                data.get("message"),
            )
        return success

    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        payload: Any,
        *,
        idempotent: bool,
        task_mode: str | None = None,
        wrapped: bool = True,
    ) -> Any:
        headers = {
            "Accept": "*/*",
            "Authorization": f"oqcs_auth={self._api_key}",
            "Content-Type": "application/json",
            "origin-language": "en",
        }
        if task_mode is not None:
            headers["task_mode"] = task_mode
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        url = self._base_url + path

        attempts = 1 + (self._retries if idempotent else 0)
        delay = self._backoff
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                status, raw = self._transport(method, url, headers, body)
            except (OSError, urllib.error.URLError) as exc:  # network level
                last_error = exc
            else:
                if status >= 500:
                    last_error = BackendError(f"HTTP {status} from {path}")
                else:
                    return self._unwrap(path, status, raw, wrapped=wrapped)
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay *= 2
        raise BackendError(
            f"Origin Quantum request {path} failed: {last_error}"
        ) from last_error

    @staticmethod
    def _unwrap(path: str, status: int, raw: bytes, *, wrapped: bool) -> Any:
        if status in (401, 403):
            raise AuthenticationError(
                f"Origin Quantum rejected the API key ({status})."
            )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BackendError(
                f"Origin Quantum {path} returned non-JSON (HTTP {status})."
            ) from exc
        if status >= 400:
            message = data.get("message") if isinstance(data, Mapping) else None
            raise BackendError(f"Origin Quantum {path} HTTP {status}: {message}")
        if not wrapped:
            return data
        if not isinstance(data, Mapping):
            raise BackendError(f"Origin Quantum {path} returned an unexpected shape.")
        if not data.get("success") or data.get("code") != _OK:
            raise BackendError(
                f"Origin Quantum {path} failed (code {data.get('code')}): "
                f"{data.get('message')}"
            )
        return data.get("obj")
