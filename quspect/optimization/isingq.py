"""iSing cloud QUBO compilation, recoverable submission and result decoding.

The wire contract follows plectrum-core 0.1.2 and isingq-toolkit 0.5.0, plus
the ``dir`` field that the upload signature carries since November 2026: the
provider's storage policy only admits object keys under that directory.
Device discovery, quotas and approval policy belong to the calling application.
This module does not use the circuit Backend interface or the native extension.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, overload
from urllib.parse import quote, urlsplit

import numpy as np

from quspect.optimization.conversion import (
    bqm_to_ising,
    ising_to_binary,
    qubo_matrix_from_bqm,
)
from quspect.optimization.model import (
    BinaryQuadraticModel,
    IsingProblem,
    OptimizationSense,
    VariableType,
)

if TYPE_CHECKING:
    import httpx

__all__ = ["IsingQClient", "IsingQError", "IsingQProgram", "compile_isingq"]

logger = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_ERROR_RESPONSE_BYTES = 16 * 1024
_MAX_PROVIDER_MESSAGE_CHARS = 512
_OBJECT_NAME = "quspect-qubo.csv"
# Object-key limit of the provider's storage service, in UTF-8 bytes.
_MAX_OBJECT_KEY_BYTES = 1023
# HTTP statuses of a create request that leave the outcome open: the request
# may have been processed (timeouts, conflicts, throttling, server errors).
_AMBIGUOUS_CREATE_STATUSES = frozenset({408, 409, 425, 429})


@dataclass(frozen=True)
class IsingQProgram:
    """Frozen QUBO upload plus the original objective and variable ordering.

    Instances are produced by :func:`compile_isingq`. The service minimizes
    ``x.T @ Q @ x`` over binary ``x``. Constants stay local; maximizing
    objectives are negated before upload. No coefficient rounding is applied.
    Keep the program (or the model it was compiled from) alongside a saved
    task ID so that results can be decoded after a restart.

    Args:
        model: Snapshot of the original logical model, in its own variable
            type and optimization sense.
        csv: ASCII CSV encoding of the upper-triangular minimizing QUBO
            matrix that is uploaded to the provider.
        binary_offset: Constant term of the binary (unsigned) form of the
            model. It is not uploaded.
        objective_sign: ``1`` for minimization, ``-1`` for maximization; the
            uploaded matrix equals ``objective_sign`` times the binary QUBO.

    Example:
        >>> from quspect.optimization import bqm_from_qubo_matrix, compile_isingq
        >>> program = compile_isingq(bqm_from_qubo_matrix([[-1, 2], [0, -3]]))
        >>> program.objective_sign
        1
        >>> program.csv.splitlines()[0]
        b'-1,2'
    """

    model: BinaryQuadraticModel
    csv: bytes
    binary_offset: float
    objective_sign: int

    @property
    def sha256(self) -> str:
        """Hex SHA-256 digest of the uploaded CSV payload.

        Returns:
            The 64-character lowercase hexadecimal digest of :attr:`csv`.

        Example:
            >>> from quspect.optimization import bqm_from_qubo_matrix, compile_isingq
            >>> len(compile_isingq(bqm_from_qubo_matrix([[1.0]])).sha256)
            64
        """
        return hashlib.sha256(self.csv).hexdigest()

    def decode(
        self, bits: object, *, provider_energy: float | None = None
    ) -> dict[str, Any]:
        """Recompute the original objective; keep vendor energy diagnostic only.

        The returned objective is evaluated on the original model (including
        its offset, variable type and sense), independently of whatever energy
        the provider reports.

        Args:
            bits: One binary vector (values ``0``/``1``) in the model's
                variable order. For spin models, ``s = 2x - 1`` is applied.
            provider_energy: Optional energy reported by the provider. It is
                validated and returned unchanged for diagnostics only.

        Returns:
            A dict with the keys ``"variables"``, ``"variable_type"``,
            ``"sense"``, ``"bits"``, ``"sample"`` (in the model's own domain),
            ``"objective_value"`` (original objective), ``"submitted_objective_value"``
            (value of the uploaded QUBO, ``x.T @ Q @ x``) and
            ``"provider_energy"``.

        Raises:
            ValueError: If ``bits`` is not one binary vector of the right
                length, or if the objective or provider energy is not finite.

        Example:
            >>> from quspect.optimization import bqm_from_qubo_matrix, compile_isingq
            >>> model = bqm_from_qubo_matrix([[-1, 2], [0, -3]], offset=5)
            >>> compile_isingq(model).decode([0, 1])["objective_value"]
            2.0
        """
        vector = np.asarray(bits)
        if vector.shape != (self.model.num_variables,) or not np.all(
            (vector == 0) | (vector == 1)
        ):
            raise ValueError("The result must be one binary vector in variable order")
        vector = vector.astype(np.int8)
        sample = (
            vector
            if self.model.variable_type is VariableType.BINARY
            else 2 * vector - 1
        )
        objective = self.model.evaluate(sample)
        if not math.isfinite(objective):
            raise ValueError("The decoded objective is not finite")
        if provider_energy is not None and not math.isfinite(provider_energy):
            raise ValueError("The provider energy is not finite")
        return {
            "variables": list(self.model.variables),
            "variable_type": self.model.variable_type.value,
            "sense": self.model.sense.value,
            "bits": vector.tolist(),
            "sample": sample.tolist(),
            "objective_value": objective,
            "submitted_objective_value": self.objective_sign
            * (objective - self.binary_offset),
            "provider_energy": provider_energy,
        }


def compile_isingq(model: BinaryQuadraticModel | IsingProblem) -> IsingQProgram:
    """Compile a logical binary/spin objective to a minimizing upper QUBO.

    Spin models are converted with ``s = 2x - 1``; maximizing models are
    negated so that the uploaded matrix is always minimized. The model, offset
    and variable names are snapshotted into the returned program.
    ``HardwareIsingProgram.matrix`` uses a different sign/factor convention and
    must not be passed here.

    Args:
        model: A nonempty logical :class:`BinaryQuadraticModel` or
            :class:`IsingProblem`.

    Returns:
        An :class:`IsingQProgram` holding the CSV upload and the original
        model snapshot.

    Raises:
        ValueError: If ``model`` is not a nonempty supported model, or if the
            QUBO conversion produces non-finite coefficients.

    Example:
        >>> from quspect.optimization import bqm_from_qubo_matrix, compile_isingq
        >>> program = compile_isingq(bqm_from_qubo_matrix([[-1, 2], [0, -3]]))
        >>> program.csv.decode().splitlines()
        ['-1,2', '0,-3']
    """
    if isinstance(model, IsingProblem):
        model = model.as_bqm()
    if not isinstance(model, BinaryQuadraticModel) or model.num_variables < 1:
        raise ValueError("A nonempty BinaryQuadraticModel or IsingProblem is required")
    original = BinaryQuadraticModel(
        model.variables,
        model.linear,
        model.quadratic,
        model.offset,
        model.variable_type,
        model.sense,
    )
    binary = (
        original
        if original.variable_type is VariableType.BINARY
        else ising_to_binary(bqm_to_ising(original))
    )
    sign = -1 if original.sense is OptimizationSense.MAXIMIZE else 1
    matrix = sign * qubo_matrix_from_bqm(binary)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("QUBO conversion overflowed")
    output = io.StringIO()
    np.savetxt(output, matrix, delimiter=",", fmt="%.17g")
    return IsingQProgram(
        original, output.getvalue().encode("ascii"), binary.offset, sign
    )


class IsingQError(RuntimeError):
    """Sanitized transport failure with explicit submission uncertainty.

    ``unknown`` means creation may have reached the vendor. Never retry it
    automatically. ``not_created`` means failure preceded task creation.
    ``rejected`` means the provider answered the create request with a
    definite refusal (an HTTP client error such as 400, 401, 403 or 404): no
    task exists. ``not_applicable`` is used for polling failures, which say
    nothing about whether the task exists.

    The exception text holds the phase, the outcome and the HTTP status. It
    never contains credentials or response bodies. The provider's own
    explanation, when it sent one, is kept separately in
    :attr:`provider_message`.

    Args:
        phase: Request phase that failed, such as ``"upload_signature"``,
            ``"upload"``, ``"create_task"`` or ``"poll"``.
        outcome: Submission outcome. Defaults to ``"not_applicable"`` for
            ``"poll"`` and ``"not_created"`` otherwise.
        status_code: HTTP status of the provider's answer, if one arrived.
        provider_message: Sanitized provider message, if one was returned.

    Attributes:
        phase: The failed request phase.
        submission_outcome: One of ``"not_created"``, ``"rejected"``,
            ``"unknown"`` or ``"not_applicable"``.
        status_code: HTTP status of the answer, or ``None`` when no answer
            arrived (timeout, connection failure) or the failure was local.
        provider_message: The ``message`` field of the provider's JSON
            answer with credentials, URLs and control characters removed and
            at most 512 characters, or ``None``. It is provider-supplied
            text: display it, do not interpret it.

    Example:
        >>> error = IsingQError("upload")
        >>> error.submission_outcome
        'not_created'
        >>> IsingQError("poll").submission_outcome
        'not_applicable'
        >>> refused = IsingQError("create_task", "rejected", status_code=400)
        >>> str(refused)
        'iSing request failed during create_task; submission rejected (HTTP 400)'
    """

    def __init__(
        self,
        phase: str,
        outcome: str | None = None,
        *,
        status_code: int | None = None,
        provider_message: str | None = None,
    ) -> None:
        if outcome is None:
            outcome = "not_applicable" if phase == "poll" else "not_created"
        text = f"iSing request failed during {phase}; submission {outcome}"
        if status_code is not None:
            text += f" (HTTP {status_code})"
        super().__init__(text)
        self.phase = phase
        self.submission_outcome = outcome
        self.status_code = status_code
        self.provider_message = provider_message


def _https_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "An HTTPS URL without credentials, query or fragment is required"
        )
    return value.rstrip("/")


def _task_id(value: object) -> str:
    # Some deployments return numeric task IDs; bool is an int subclass.
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value
    ):
        raise ValueError("Invalid iSing task identifier")
    return value


def _upload_prefix(value: object) -> str:
    """Return the object-key prefix the upload signature requires.

    The result ends with ``/``; it is empty when the signature names no
    directory. Only what is dangerous is refused (an absolute path, a ``.`` or
    ``..`` segment, a control character, an oversized key), not a set of
    allowed characters: the provider chooses the directory.
    """
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError("Invalid upload directory")
    prefix = value if value.endswith("/") else value + "/"
    key_bytes = len(prefix.encode("utf-8")) + 32 + 1 + len(_OBJECT_NAME)
    if (
        prefix.startswith(("/", "\\"))
        or key_bytes > _MAX_OBJECT_KEY_BYTES
        or any(segment in (".", "..") for segment in re.split(r"[/\\]", prefix))
        or any(ord(c) < 0x20 or 0x7F <= ord(c) < 0xA0 for c in prefix)
    ):
        raise ValueError("Invalid upload directory")
    return prefix


_MARKUP = re.compile(r"<!doctype|</?[a-z][^>]*>", re.IGNORECASE)
# A Python traceback, an exception class name before a colon, or a Java frame.
_STACK_TRACE = re.compile(
    r"Traceback \(most recent call last\)|[\w.]*\w(?:Error|Exception):"
    r"|^\s*at\s+[\w.$<>]+\(",
    re.MULTILINE,
)
_CREDENTIAL_PAIR = re.compile(
    r"\b(authorization|cookie|set-cookie|password|secret|signature|policy|token"
    r"|(?:api|access|private|secret)[ _-]?key"
    r"|(?:access|security|refresh)[ _-]?token"
    r"|x[_-]oss[_-](?:credential|security[_-]token))\b"
    r"[\"']?\s*[=:\uff1a]\s*(?:bearer\s+)?(?:\"[^\"]*\"?|'[^']*'?|[^\s;,]+)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+[^\s;,]+", re.IGNORECASE)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_UNSAFE_TEXT = re.compile("[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def _provider_message(value: object, secrets: Sequence[str] = ()) -> str | None:
    """Reduce a provider-supplied message to bounded, credential-free text.

    Markup and stack traces are dropped whole: they are proxy or server
    internals, not the provider's explanation.
    """
    if (
        not isinstance(value, str)
        or _MARKUP.search(value)
        or _STACK_TRACE.search(value)
    ):
        return None
    text = value
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        for form in {secret, quote(secret, safe="")}:
            text = text.replace(form, "[REDACTED]")
    text = _CREDENTIAL_PAIR.sub(lambda match: match.group(1) + "=[REDACTED]", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _URL.sub("[REDACTED_URL]", text)
    text = " ".join(_UNSAFE_TEXT.sub(" ", text).split())
    return text[:_MAX_PROVIDER_MESSAGE_CHARS] or None


def _provider_status(value: object) -> int:
    """Read a task status sent as an integer, an integral float or digits."""
    if isinstance(value, bool):
        raise ValueError("Invalid iSing task status")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]{1,9}", value.strip()):
        return int(value.strip())
    raise ValueError("Invalid iSing task status")


def _provider_energy(value: object) -> int | float | None:
    """Read the diagnostic energy; a number sent as text is accepted.

    Anything else unusable (other text, a non-finite number, another type) is
    dropped instead of failing a finished task: the solution vector is the
    result, and the objective is recomputed locally from it.
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


class IsingQClient:
    """Explicit-key cloud transport; submit returns before polling.

    Install ``quspect[isingq]``. Calls are bounded, redirects and retries are
    disabled, and the account key is sent only to the configured API origin.
    OSS uploads use the temporary form credentials. No credential is persisted.
    Use the client as a context manager, or call :meth:`close` explicitly.

    Args:
        api_key: iSing account API key. Must be nonempty and contain no
            whitespace.
        base_url: HTTPS API origin without credentials, query or fragment.
        timeout: Per-request timeout in seconds; positive and finite.
        transport: Optional ``httpx`` transport, mainly for testing with
            ``httpx.MockTransport``.

    Raises:
        OptionalDependencyError: If ``httpx`` is not installed.
        ValueError: If the key, URL or timeout is invalid.

    Example:
        .. code-block:: python

           import os
           from quspect.optimization import IsingQClient, compile_isingq

           with IsingQClient(os.environ["ISINGQ_API_KEY"]) as client:
               task_id = client.submit(compile_isingq(model))
               snapshot = client.poll(task_id, compile_isingq(model))
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.isingq.com",
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            import httpx
        except ImportError:
            from quspect.backends._exceptions import OptionalDependencyError

            raise OptionalDependencyError(
                "The httpx package is required to use the iSing cloud client. "
                "Install it with: pip install 'quspect[isingq]'"
            ) from None

        if (
            not isinstance(api_key, str)
            or not api_key
            or any(c.isspace() for c in api_key)
        ):
            raise ValueError("An iSing API key is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._base_url = _https_url(base_url)
        self._headers = {"Authorization": api_key, "channel": "sdk"}
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool.

        Example:
            >>> client = IsingQClient("test-key")
            >>> client.close()
        """
        self._http.close()

    def __enter__(self) -> IsingQClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @overload
    def _request(
        self,
        method: str,
        url: str,
        *,
        phase: str,
        json_response: Literal[True] = ...,
        secrets: Sequence[str] = ...,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @overload
    def _request(
        self,
        method: str,
        url: str,
        *,
        phase: str,
        json_response: Literal[False],
        secrets: Sequence[str] = ...,
        **kwargs: Any,
    ) -> None: ...

    def _request(
        self,
        method: str,
        url: str,
        *,
        phase: str,
        json_response: bool = True,
        secrets: Sequence[str] = (),
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        _, body = self._exchange(
            method,
            url,
            phase=phase,
            json_response=json_response,
            secrets=secrets,
            **kwargs,
        )
        return body if json_response else None

    def _exchange(
        self,
        method: str,
        url: str,
        *,
        phase: str,
        json_response: bool = True,
        secrets: Sequence[str] = (),
        **kwargs: Any,
    ) -> tuple[int, dict[str, Any]]:
        """Send one request; return the HTTP status and the JSON object.

        The object is empty when ``json_response`` is false. ``secrets`` are
        redacted from a provider message, in addition to the account key.
        """
        status: int | None = None
        message: str | None = None
        try:
            with self._http.stream(method, url, **kwargs) as response:
                status = response.status_code
                if not response.is_success:
                    message = self._failure_message(response, secrets)
                    raise ValueError("Unsuccessful response")
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > _MAX_RESPONSE_BYTES:
                        raise ValueError("Response exceeds limit")
                if not json_response:
                    return status, {}
                body = json.loads(content)
                if not isinstance(body, dict):
                    raise ValueError("Invalid response")
                return status, body
        except Exception as error:
            # Log only the exception type: messages may echo secrets or bodies.
            logger.debug("iSing %s request failed: %s", phase, type(error).__name__)
            outcome = None
            if phase == "create_task":
                # Only a client error that is not about timing or state says
                # that the provider refused the task. No answer, a redirect, a
                # server error or an unreadable 2xx body may hide a created task.
                refused = (
                    status is not None
                    and 400 <= status < 500
                    and status not in _AMBIGUOUS_CREATE_STATUSES
                )
                outcome = "rejected" if refused else "unknown"
            raise IsingQError(
                phase, outcome, status_code=status, provider_message=message
            ) from None

    def _failure_message(
        self, response: httpx.Response, secrets: Sequence[str]
    ) -> str | None:
        """Read the provider's ``message`` from a bounded error body, if any.

        Reading or parsing a failed response must never replace its HTTP
        outcome, so every problem here means "no message".
        """
        try:
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > _MAX_ERROR_RESPONSE_BYTES:
                    return None
            body = json.loads(content)
            if not isinstance(body, dict):
                return None
            return self._message(body, secrets)
        except Exception:
            return None

    def _message(self, body: dict[str, Any], secrets: Sequence[str]) -> str | None:
        """Sanitize a body's ``message``; the account key is always redacted."""
        return _provider_message(
            body.get("message"), (self._headers["Authorization"], *secrets)
        )

    def submit(
        self,
        program: IsingQProgram,
        *,
        computer_type_id: int = 1,
        calculate_count: int = 1,
        post_process: bool = False,
        use_credit: bool = False,
    ) -> str:
        """Upload and create exactly once; persist the returned ID before polling.

        The CSV is uploaded to the provider's OSS host with temporary form
        credentials, then one task is created. Nothing is retried. The object
        key is ``<dir><random>/quspect-qubo.csv``, where ``dir`` is the
        directory named by the upload signature (empty if it names none).
        ``calculate_count`` is a vendor iteration count, not circuit shots.
        ``gear`` is intentionally absent: the published cloud wire does not send it.

        Args:
            program: Compiled program from :func:`compile_isingq`.
            computer_type_id: Vendor routing value; positive integer.
            calculate_count: Vendor calculation count; positive integer.
            post_process: Whether to request vendor post-processing.
            use_credit: Whether the task may consume account credit.

        Returns:
            The provider task ID. Persist it before polling.

        Raises:
            ValueError: If an option has the wrong type or value.
            IsingQError: If a request fails. ``submission_outcome`` is
                ``"not_created"`` before task creation, ``"rejected"`` when
                the provider refused the create request with an HTTP client
                error, and ``"unknown"`` once the create request may have
                been processed (no answer, a server error, or an answer
                without a task ID). ``status_code`` and ``provider_message``
                carry the provider's answer when there was one.

        Example:
            .. code-block:: python

               with IsingQClient(api_key) as client:
                   task_id = client.submit(program, calculate_count=1)
        """
        for value in (computer_type_id, calculate_count):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Computer type and calculate count must be positive integers"
                )
        if type(post_process) is not bool or type(use_credit) is not bool:
            raise ValueError("Post processing and credit options must be boolean")
        signature = self._request(
            "GET",
            self._base_url + "/files/getPostSignatureForOssUpload",
            phase="upload_signature",
            headers=self._headers,
        ).get("data")
        try:
            fields = (
                "host",
                "policy",
                "signature",
                "x_oss_credential",
                "x_oss_date",
                "security_token",
            )
            if not isinstance(signature, dict) or any(
                not isinstance(signature.get(k), str) or not signature[k]
                for k in fields
            ):
                raise ValueError("Invalid signature")
            host = _https_url(signature["host"])
            prefix = _upload_prefix(signature.get("dir"))
        except Exception:
            raise IsingQError("upload_signature") from None
        # The directory may be the same for two signatures, so the key keeps a
        # segment of its own: a later upload must never replace this file.
        key = f"{prefix}{uuid.uuid4().hex}/{_OBJECT_NAME}"
        secrets = tuple(
            signature[k]
            for k in ("policy", "signature", "x_oss_credential", "security_token")
        )
        form = {
            "success_action_status": "200",
            "policy": signature["policy"],
            "x-oss-signature": signature["signature"],
            "x-oss-signature-version": "OSS4-HMAC-SHA256",
            "x-oss-credential": signature["x_oss_credential"],
            "x-oss-date": signature["x_oss_date"],
            "key": key,
            "x-oss-security-token": signature["security_token"],
        }
        self._request(
            "POST",
            host,
            phase="upload",
            data=form,
            files={"file": ("qubo.csv", program.csv, "text/csv")},
            json_response=False,
            secrets=secrets,
        )
        status, body = self._exchange(
            "POST",
            self._base_url + "/tasks/create-general",
            phase="create_task",
            headers=self._headers,
            secrets=secrets,
            json={
                "name": "quspect-" + program.sha256[:12],
                "computerTypeId": computer_type_id,
                "inputJFile": host + "/" + key,
                "inputHFile": None,
                "questionType": 1,
                "caculateCount": calculate_count,
                "postProcess": int(post_process),
                "useCoupon": False,
                "useCredit": use_credit,
            },
        )
        for candidate in (body, body.get("data")):
            if isinstance(candidate, dict):
                for field in ("id", "taskId", "task_id"):
                    try:
                        return _task_id(candidate.get(field))
                    except ValueError:
                        continue
        # A 2xx answer without a usable ID does not say that no task exists.
        logger.debug("iSing create_task response did not contain a valid task ID")
        raise IsingQError(
            "create_task",
            "unknown",
            status_code=status,
            provider_message=self._message(body, secrets),
        )

    def poll(self, task_id: str, program: IsingQProgram) -> dict[str, Any]:
        """Read one status snapshot. Cancellation and device discovery are unavailable.

        Published clients treat statuses 1/2/3 as terminal. Preserve that raw
        status; do not invent individual meanings or a distribution of samples.
        Polling can be resumed from a new client using only the saved task ID
        and the compiled program.

        Args:
            task_id: Task ID returned by :meth:`submit`. Integer IDs are
                accepted and normalized to strings.
            program: The program that was submitted, used to decode results.

        Returns:
            A dict with ``"status"`` (``"running"``, ``"succeeded"`` or
            ``"failed"``) and the integer ``"provider_status"`` (a status
            sent as text such as ``"2"`` is accepted). Succeeded snapshots
            include ``"result"`` as returned by :meth:`IsingQProgram.decode`;
            its ``"provider_energy"`` is ``None`` when the provider sent no
            usable number. Malformed terminal results include ``"error"``,
            and failed snapshots include ``"provider_message"`` when the
            provider explained the failure.

        Raises:
            ValueError: If ``task_id`` is not a valid task identifier.
            IsingQError: If the status request fails or returns an invalid
                payload; ``submission_outcome`` is ``"not_applicable"``.

        Example:
            .. code-block:: python

               with IsingQClient(api_key) as client:
                   snapshot = client.poll(task_id, program)
               if snapshot["status"] == "succeeded":
                   print(snapshot["result"]["objective_value"])
        """
        http_status, body = self._exchange(
            "GET",
            self._base_url + "/tasks/" + _task_id(task_id),
            phase="poll",
            headers=self._headers,
        )
        payload = body.get("data", body)
        try:
            if not isinstance(payload, dict):
                raise ValueError("Invalid task payload")
            status = _provider_status(payload.get("status"))
        except ValueError:
            raise IsingQError(
                "poll",
                status_code=http_status,
                provider_message=self._message(body, ()),
            ) from None
        if status not in (1, 2, 3):
            return {"status": "running", "provider_status": status}
        failed: dict[str, Any] = {"status": "failed", "provider_status": status}
        # Keep the provider's own explanation of a task that ended without a result.
        message = self._message(payload, ()) or self._message(body, ())
        if message is not None:
            failed["provider_message"] = message
        raw = payload.get("result")
        if not isinstance(raw, dict):
            return failed
        try:
            bits = raw.get("bits")
            if bits is None:
                bits = raw.get("spin_config")
            if not isinstance(bits, list) or any(type(v) is not int for v in bits):
                raise ValueError("Invalid solution vector")
            if bits and all(v in (-1, 1) for v in bits):
                bits = [(v + 1) // 2 for v in bits]
            decoded = program.decode(
                bits, provider_energy=_provider_energy(raw.get("energy"))
            )
        except (TypeError, ValueError, OverflowError):
            # Terminal malformed results must not produce an endless polling loop.
            logger.debug("iSing task returned a malformed terminal result")
            return {**failed, "error": "Invalid iSing solution"}
        return {"status": "succeeded", "provider_status": status, "result": decoded}
