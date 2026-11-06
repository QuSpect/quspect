"""Minimal Tianyan (天衍) cloud HTTP client (standard library only).

The cloud has no public API documentation; this client speaks the protocol
used by cqlib 1.3.11 (``cqlib.quantum_platform``), checked against the live
service in November 2026:

``POST /qccp-auth/oauth2/sdk/opnId`` (form)
    ``grant_type=openId, openId=<login key>, account_type=member`` →
    ``data.access_token``.
``GET  /qccp-quantum/sdk/quantumComputer/list``
    Machines: ``code`` (name), ``status`` (0 running, 1 calibrating,
    2 maintenance, 3 offline), ``isToll`` (1 free, 2 paid), ``label``.
``GET  /qccp-quantum/sdk/experiment/download/config/<machine>``
    Chip configuration and calibration (real QPUs only).
``POST /qccp-quantum/sdk/experiment/temporary/save``
    Submit QCIS programs → ``data.query_ids`` (one per program).
``POST /qccp-quantum/sdk/experiment/result/find``  ``{"query_ids": [...]}``
    ``data.experimentResultModelList`` once every task has finished.

Authenticated calls send ``Authorization: Bearer <token>`` and
``basicToken: <token>``.  Responses are ``{"code", "msg", "data"}`` with
``code == 0`` on success.  An expired or missing token comes back as HTTP 200
with ``code == 401``; the gateway also returns intermittent HTTP 504s, so
login and every read are retried; submissions are retried a few times as well
(a timed-out submission may still have created its task, see
``submit_retries``).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from quspect.backends._exceptions import AuthenticationError, BackendError

#: ``(method, url, headers, body) -> (status, response bytes)``
Transport = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]

DEFAULT_URL = "https://qc.zdxlz.com"
_LOGIN = "/qccp-auth/oauth2/sdk/opnId"
_MACHINES = "/qccp-quantum/sdk/quantumComputer/list"
_CONFIG = "/qccp-quantum/sdk/experiment/download/config/"
_SUBMIT = "/qccp-quantum/sdk/experiment/temporary/save"
_RESULTS = "/qccp-quantum/sdk/experiment/result/find"
_CIRCUITS = "/qccp-quantum/sdk/experiment/getQcis/by/taskIds"
_TOKEN_EXPIRED = 401
_logger = logging.getLogger(__name__)


def urllib_transport(timeout: float = 60.0) -> Transport:
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


class _RetryableError(Exception):
    """A transient failure (network error, HTTP 5xx)."""


class TianyanClient:
    """Authenticated access to the Tianyan task endpoints.

    Args:
        login_key: Tianyan "API Token" from the personal center of the web
            console (sent only to the login endpoint).
        base_url: Service URL (default ``https://qc.zdxlz.com``).
        transport: Optional transport for testing.
        retries: Extra attempts for login and reads on network errors or
            HTTP 5xx.
        submit_retries: Extra attempts for a submission on network errors or
            HTTP 5xx.  The Tianyan gateway answers a large share of requests
            with HTTP 504, and a timed-out submission may or may not have
            created its task, so a retry can duplicate a task.  Each retry is
            logged as a warning.  Pass ``0`` to never retry (e.g. on paid
            machines).
        backoff: Initial delay between retries in seconds (doubles, capped).
    """

    def __init__(
        self,
        login_key: str,
        base_url: str = DEFAULT_URL,
        *,
        transport: Transport | None = None,
        retries: int = 6,
        submit_retries: int = 3,
        backoff: float = 1.0,
    ) -> None:
        if not login_key:
            raise AuthenticationError("A Tianyan login key is required.")
        self._login_key = login_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or urllib_transport()
        self._retries = max(0, int(retries))
        self._submit_retries = max(0, int(submit_retries))
        self._backoff = float(backoff)
        self._token: str | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    def __repr__(self) -> str:  # never show the key or token
        return f"TianyanClient(base_url={self._base_url!r})"

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Exchange the login key for an access token."""
        body = urllib.parse.urlencode(
            {
                "grant_type": "openId",
                "openId": self._login_key,
                "account_type": "member",
            }
        ).encode("ascii")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        def attempt() -> Any:
            status, raw = self._send("POST", _LOGIN, headers, body)
            data = self._decode(_LOGIN, status, raw)
            if not isinstance(data, Mapping) or data.get("code") != 0:
                raise AuthenticationError(
                    "Tianyan rejected the login key: "
                    f"{_message(data) if isinstance(data, Mapping) else data}"
                )
            token = (data.get("data") or {}).get("access_token")
            if not token:
                raise AuthenticationError("Tianyan login returned no access token.")
            return token

        self._token = str(self._with_retries(_LOGIN, attempt))

    def list_machines(self) -> list[dict[str, Any]]:
        data = self._call("GET", _MACHINES, None, idempotent=True)
        if not isinstance(data, list):
            raise BackendError("Tianyan machine list has an unexpected shape.")
        return [dict(item) for item in data if isinstance(item, Mapping)]

    def machine_config(self, machine: str) -> dict[str, Any]:
        if not machine or not all(ch.isalnum() or ch in "_-" for ch in machine):
            raise ValueError(f"Invalid Tianyan machine name {machine!r}.")
        data = self._call("GET", _CONFIG + machine, None, idempotent=True)
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, Mapping):
            raise BackendError(f"Tianyan returned no configuration for {machine!r}.")
        return dict(data)

    def submit(self, programs: Sequence[str], *, machine: str, shots: int) -> list[str]:
        """Submit QCIS programs and return one query ID per program."""
        payload = {
            "exp_id": None,
            "lab_id": None,
            "inputCode": list(programs),
            "languageCode": "qcis",
            "name": "",
            "shots": int(shots),
            "source": "SDK",
            "computerCode": machine,
            "experimentDetailName": "1" if len(programs) == 1 else None,
            "is_verify": True,
        }
        data = self._call("POST", _SUBMIT, payload, idempotent=False)
        ids = (data or {}).get("query_ids") if isinstance(data, Mapping) else None
        if not isinstance(ids, list) or len(ids) != len(programs):
            raise BackendError(
                "Tianyan accepted the submission but returned "
                f"{0 if not isinstance(ids, list) else len(ids)} task ID(s) for "
                f"{len(programs)} program(s)."
            )
        return [str(item) for item in ids]

    def results(self, query_ids: Sequence[str]) -> list[dict[str, Any]] | None:
        """Return results for all IDs, or ``None`` while any is unfinished."""
        data = self._call(
            "POST", _RESULTS, {"query_ids": list(query_ids)}, idempotent=True
        )
        items = (
            (data or {}).get("experimentResultModelList")
            if isinstance(data, Mapping)
            else None
        )
        if not items or len(items) != len(query_ids):
            return None
        return [dict(item) for item in items if isinstance(item, Mapping)]

    def executed_circuits(self, query_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Submitted, compiled (``mapQcis``) and executed (``computerQcis``) QCIS."""
        data = self._call(
            "POST", _CIRCUITS, {"query_ids": list(query_ids)}, idempotent=True
        )
        return [dict(item) for item in data or [] if isinstance(item, Mapping)]

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _call(self, method: str, path: str, payload: Any, *, idempotent: bool) -> Any:
        if self._token is None:
            self.login()
        body = None if payload is None else json.dumps(payload).encode("utf-8")

        def attempt() -> Any:
            for relogin in (False, True):
                headers = {
                    "Authorization": f"Bearer {self._token}",
                    "basicToken": str(self._token),
                    "Content-Type": "application/json",
                }
                status, raw = self._send(method, path, headers, body)
                data = self._decode(path, status, raw)
                code = data.get("code") if isinstance(data, Mapping) else None
                if (status == 401 or code == _TOKEN_EXPIRED) and not relogin:
                    self.login()
                    continue
                if code != 0:
                    raise BackendError(f"Tianyan {path} failed: {_message(data)}")
                return data.get("data")
            raise AuthenticationError("Tianyan rejected the refreshed access token.")

        if idempotent:
            return self._with_retries(path, attempt)
        return self._with_retries(
            path, attempt, retries=self._submit_retries, warn_duplicates=True
        )

    def _with_retries(
        self,
        path: str,
        attempt: Callable[[], Any],
        *,
        retries: int | None = None,
        warn_duplicates: bool = False,
    ) -> Any:
        limit = self._retries if retries is None else retries
        delay = self._backoff
        last: Exception | None = None
        for index in range(1 + limit):
            try:
                return attempt()
            except _RetryableError as exc:
                last = exc
            if index < limit:
                if warn_duplicates:
                    _logger.warning(
                        "Tianyan %s failed (%s); retrying (%d/%d). The failed "
                        "attempt may still have created a task.",
                        path,
                        last,
                        index + 1,
                        limit,
                    )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise BackendError(f"Tianyan request {path} failed: {last}") from last

    def _send(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> tuple[int, bytes]:
        try:
            status, raw = self._transport(method, self._base_url + path, headers, body)
        except (OSError, urllib.error.URLError) as exc:
            raise _RetryableError(str(exc)) from exc
        if status >= 500:
            raise _RetryableError(f"HTTP {status}")
        return status, raw

    @staticmethod
    def _decode(path: str, status: int, raw: bytes) -> Any:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BackendError(
                f"Tianyan {path} returned non-JSON (HTTP {status})."
            ) from exc
        if status >= 400 and status != 401:
            raise BackendError(f"Tianyan {path} HTTP {status}: {_message(data)}")
        return data


def _message(data: Any) -> str:
    """Extract the English (or any) message from ``{"msg": "{CN, EN}"}``."""
    if not isinstance(data, Mapping):
        return str(data)
    message = data.get("msg") or data.get("message") or ""
    if isinstance(message, str) and message.startswith("{"):
        try:
            parsed = json.loads(message)
        except ValueError:
            return message
        if isinstance(parsed, Mapping):
            return str(parsed.get("EN") or parsed.get("CN") or message)
    return str(message or data.get("code"))
