"""Minimal HTTP client for the Quafu-SQC cloud (https://quafu-sqc.baqis.ac.cn).

Only the Python standard library is used, so the Quafu backend has no
third-party dependency. The wire protocol mirrors the official ``quafusqc``
SDK (MIT licensed) and was verified against the live service:

* every authenticated request carries the API token in a ``token`` header;
* ``GET  /task/verify``              -> ``[expiry, 0]`` (a dict on rejection)
* ``GET  /task/status/0``            -> ``{chip: queue length | state}``
* ``POST /task/run/?name=&chip=&shots=`` with a JSON body
  ``{"circuit", "compile", "options"}`` -> integer task id
* ``GET  /task/status/{tid}``        -> status string (``"Finished"``, ...)
* ``GET  /task/result/{tid}``        -> ``{}`` until done, then a result dict
* ``GET  /task/cancel/{tid}``
* ``GET  /task/backendtest/{chip}1`` -> chip topology and calibration
  (public, no token; the trailing ``1`` is part of the route)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from quspect.backends._exceptions import AuthenticationError, BackendError

DEFAULT_URL = "https://quafu-sqc.baqis.ac.cn"
TOKEN_ENV_VARS = ("QUAFU_API_TOKEN", "QPU_API_TOKEN")
TOKEN_FILE = Path("~/.quafu/api")

# (method, url, headers, body, timeout) -> (status, body bytes)
Transport = Callable[
    [str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]
]


def resolve_token(token: str | None = None) -> str:
    """Return the API token from the argument, environment, or token file.

    Precedence: explicit ``token``, then ``QUAFU_API_TOKEN``, then
    ``QPU_API_TOKEN`` (the variable the official SDK reads), then the
    ``token`` field of the JSON file ``~/.quafu/api``.
    """
    if token and token.strip():
        return token.strip()
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    path = TOKEN_FILE.expanduser()
    if path.is_file():
        try:
            value = str(json.loads(path.read_text(encoding="utf-8"))["token"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise AuthenticationError(
                f"Cannot read a Quafu token from {path}: expected JSON "
                '{"token": "..."}.'
            ) from exc
        if value:
            return value
    raise AuthenticationError(
        "No Quafu API token found. Pass token=..., set QUAFU_API_TOKEN, or "
        f'write {{"token": "..."}} to {TOKEN_FILE}. Tokens are issued at '
        f"{DEFAULT_URL} and expire after 30 days."
    )


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=body, method=method, headers=dict(headers)
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


class QuafuClient:
    """Thin JSON-over-HTTP wrapper around the Quafu-SQC task API.

    The token is kept private and never appears in ``repr`` or error
    messages.
    """

    def __init__(
        self,
        token: str | None = None,
        url: str | None = None,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
        transport: Transport | None = None,
    ) -> None:
        if retries < 0 or backoff < 0:
            raise ValueError("retries and backoff must be non-negative.")
        self._retries = int(retries)
        self._backoff = float(backoff)
        self._token = resolve_token(token)
        self._url = (url or os.environ.get("QUAFU_URL") or DEFAULT_URL).rstrip("/")
        parsed = urllib.parse.urlsplit(self._url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Invalid Quafu URL: {self._url!r}.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self._timeout = float(timeout)
        self._transport = transport or _urllib_transport

    @property
    def url(self) -> str:
        return self._url

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Any = None,
        auth: bool = True,
    ) -> Any:
        """Send one request and return the decoded JSON payload."""
        url = self._url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers: dict[str, str] = {"Accept": "application/json"}
        if auth:
            headers["token"] = self._token
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        # GETs are idempotent and retried on network errors and 5xx answers;
        # a POST could create a duplicate task, so it is sent exactly once.
        attempts = self._retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                status, raw = self._transport(method, url, headers, data, self._timeout)
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                if attempt + 1 < attempts:
                    time.sleep(self._backoff * 2**attempt)
                    continue
                raise BackendError(f"Cannot reach Quafu at {self._url}: {exc}") from exc
            if status >= 500 and attempt + 1 < attempts:
                time.sleep(self._backoff * 2**attempt)
                continue
            break

        text = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text) if text else None
        except ValueError:
            payload = text
        if status in (401, 403):
            raise AuthenticationError(
                f"Quafu rejected the API token ({status}): {_detail(payload)}"
            )
        if status >= 400:
            raise BackendError(
                f"Quafu {method} {path} failed with HTTP {status}: {_detail(payload)}"
            )
        return payload

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def __repr__(self) -> str:
        return f"QuafuClient(url={self._url!r})"


def _detail(payload: Any) -> str:
    if isinstance(payload, Mapping) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:300]
