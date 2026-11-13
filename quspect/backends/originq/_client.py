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
    ``true`` when the request was accepted and ``false`` when the cloud
    declines, with its reason in ``message``.  The request is the one
    ``pyqpanda3.qcloud.QCloudJob.cancel()`` sends to a stand-in server, and
    the answer is read the way the SDK reads it.

    Checked against the live service on 12 November 2026, on WK_C180:

    * Sent 1 to 9 s after submitting a QPU task (three tasks: 0.96 s,
      1.05 s and 8.9 s): accepted, HTTP 200 with ``success: true``,
      ``code: 10000``, ``message: "success"``.  Each task reported
      ``taskState`` ``"35"`` at the next read; the first still did
      17 minutes later, with no result and no run time.  The third was read
      before the cancel and reported ``"2"`` until then, so ``"35"`` follows
      from the cancel.
    * Sent again to that cancelled task: declined, HTTP 200 with
      ``success: false``, ``code: 10096`` and the message "Unable to cancel
      this task. Please refresh and try again.".
    * The same request sent by another client 3.4 s after submitting a QPU
      task was declined with that message (its HTTP status was not
      recorded), and the task completed normally 209 s later.

    So a refused task keeps running, and a task submitted seconds earlier
    can already be refused.  Not known: which tasks the cloud accepts a
    cancel for (state ``"2"`` alone does not decide: see above), and whether
    ``"35"`` is reported in any other situation.  pyqpanda3's documentation
    (0.5.1) says only that ``cancel()`` returns whether the service accepted
    the request; it names no cancelled state and no rule for when a cancel
    is allowed.

Every request is authenticated with ``Authorization: oqcs_auth=<api key>``.
Responses wrap their payload as ``{"success", "code", "message", "obj"}``
with ``code == 10000`` on success.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

_MAX_PROVIDER_MESSAGE_CHARS = 500
# Control characters and bidirectional overrides: a provider message is shown
# to people as one line of plain text.
_UNSAFE_TEXT = re.compile("[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


@dataclass(frozen=True)
class CancelResponse:
    """Origin Quantum's answer to one cancel request.

    Attributes:
        accepted: The answer's ``success``.  ``True`` means the cloud
            accepted the request, not that the task has stopped; ``False``
            means it declined, and the task keeps running.
        http_status: HTTP status of the answer.
        code: The answer's ``code`` field as sent, ``None`` when absent.
        provider_message: The answer's ``message`` field: on a refusal, the
            cloud's own reason (for example "Unable to cancel this task.
            Please refresh and try again.").  One line of at most 500
            characters with the API key removed; ``None`` when the cloud
            sent no text.
    """

    accepted: bool
    http_status: int
    code: Any = None
    provider_message: str | None = None


def _provider_message(value: object, api_key: str) -> str | None:
    """Reduce a provider-supplied message to one bounded line without the key."""
    if not isinstance(value, str):
        return None
    text = value.replace(api_key, "[REDACTED]")
    text = " ".join(_UNSAFE_TEXT.sub(" ", text).split())
    return text[:_MAX_PROVIDER_MESSAGE_CHARS] or None


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

    def request_cancel(self, task_id: str) -> CancelResponse:
        """Ask the cloud to cancel a task and return its answer.

        One request, never retried.  The answer's boolean ``success``
        decides, as in pyqpanda3: ``accepted`` is ``True`` when Origin
        Quantum accepted the request (not when the task has stopped) and
        ``False`` when it declined, with the cloud's reason in
        ``provider_message``.  A declined task keeps running.

        The live service declines with HTTP 200; an HTTP 4xx whose body
        says ``success: false`` is read as a refusal too, and
        ``http_status`` records what was sent.

        Raises:
            AuthenticationError: The API key was rejected (HTTP 401/403).
            BackendError: Network failure, HTTP 5xx, any other HTTP error
                whose body is not a refusal, or an answer without a boolean
                ``success``: whether the task was cancelled is then unknown.
        """
        status, raw = self._send(
            "POST", _CANCEL, {"taskId": task_id}, idempotent=False, task_mode="0"
        )
        data = self._decode(_CANCEL, status, raw)
        success = data.get("success") if isinstance(data, Mapping) else None
        if status >= 400 and success is not False:
            message = data.get("message") if isinstance(data, Mapping) else None
            raise BackendError(f"Origin Quantum {_CANCEL} HTTP {status}: {message}")
        if not isinstance(success, bool):
            raise BackendError(
                f"Origin Quantum {_CANCEL} returned no boolean 'success' "
                f"for task {task_id}."
            )
        response = CancelResponse(
            accepted=success,
            http_status=status,
            code=data.get("code"),
            provider_message=_provider_message(data.get("message"), self._api_key),
        )
        if not success:
            _logger.info(
                "Origin Quantum declined to cancel task %s (HTTP %s, code %s): %s",
                task_id,
                status,
                response.code,
                response.provider_message,
            )
        return response

    def cancel_task(self, task_id: str) -> bool:
        """Ask the cloud to cancel a task and return whether it accepted.

        ``request_cancel(task_id).accepted``: see :meth:`request_cancel`
        for the answer's meaning, the cloud's reason for a refusal, and the
        errors raised.
        """
        return self.request_cancel(task_id).accepted

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
        status, raw = self._send(
            method, path, payload, idempotent=idempotent, task_mode=task_mode
        )
        return self._unwrap(path, status, raw, wrapped=wrapped)

    def _send(
        self,
        method: str,
        path: str,
        payload: Any,
        *,
        idempotent: bool,
        task_mode: str | None = None,
    ) -> tuple[int, bytes]:
        """Send one request (retried when idempotent); HTTP 5xx is an error."""
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
                    return status, raw
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay *= 2
        raise BackendError(
            f"Origin Quantum request {path} failed: {last_error}"
        ) from last_error

    @staticmethod
    def _decode(path: str, status: int, raw: bytes) -> Any:
        """Return the JSON body; a rejected key or a non-JSON body raises."""
        if status in (401, 403):
            raise AuthenticationError(
                f"Origin Quantum rejected the API key ({status})."
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BackendError(
                f"Origin Quantum {path} returned non-JSON (HTTP {status})."
            ) from exc

    @classmethod
    def _unwrap(cls, path: str, status: int, raw: bytes, *, wrapped: bool) -> Any:
        data = cls._decode(path, status, raw)
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
