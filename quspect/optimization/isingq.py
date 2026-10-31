"""iSing cloud QUBO compilation, recoverable submission and result decoding.

The wire contract follows plectrum-core 0.1.2 and isingq-toolkit 0.5.0.
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
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, overload
from urllib.parse import urlsplit

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
    ``not_applicable`` is used for polling failures, which say nothing about
    whether the task exists. Messages never contain credentials or response
    bodies.

    Args:
        phase: Request phase that failed, such as ``"upload_signature"``,
            ``"upload"``, ``"create_task"`` or ``"poll"``.
        outcome: Submission outcome. Defaults to ``"not_applicable"`` for
            ``"poll"`` and ``"not_created"`` otherwise.

    Attributes:
        phase: The failed request phase.
        submission_outcome: One of ``"not_created"``, ``"unknown"`` or
            ``"not_applicable"``.

    Example:
        >>> error = IsingQError("upload")
        >>> error.submission_outcome
        'not_created'
        >>> IsingQError("poll").submission_outcome
        'not_applicable'
    """

    def __init__(self, phase: str, outcome: str | None = None) -> None:
        if outcome is None:
            outcome = "not_applicable" if phase == "poll" else "not_created"
        super().__init__(f"iSing request failed during {phase}; submission {outcome}")
        self.phase = phase
        self.submission_outcome = outcome


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
        **kwargs: Any,
    ) -> None: ...

    def _request(
        self,
        method: str,
        url: str,
        *,
        phase: str,
        json_response: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        try:
            with self._http.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > _MAX_RESPONSE_BYTES:
                        raise ValueError("Response exceeds limit")
                if not json_response:
                    return None
                body = json.loads(content)
                if not isinstance(body, dict):
                    raise ValueError("Invalid response")
                return body
        except Exception as error:
            # Log only the exception type: messages may echo secrets or bodies.
            logger.debug("iSing %s request failed: %s", phase, type(error).__name__)
            raise IsingQError(
                phase, "unknown" if phase == "create_task" else None
            ) from None

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
        credentials, then one task is created. Nothing is retried.
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
                ``"not_created"`` before task creation and ``"unknown"`` once
                the create request may have reached the provider.

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
        except Exception:
            raise IsingQError("upload_signature") from None
        key = f"{uuid.uuid4().hex}/quspect-qubo.csv"
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
        )
        body = self._request(
            "POST",
            self._base_url + "/tasks/create-general",
            phase="create_task",
            headers=self._headers,
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
        logger.debug("iSing create_task response did not contain a valid task ID")
        raise IsingQError("create_task", "unknown")

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
            ``"failed"``) and the raw integer ``"provider_status"``. Succeeded
            snapshots include ``"result"`` as returned by
            :meth:`IsingQProgram.decode`; malformed terminal results include
            ``"error"``.

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
        body = self._request(
            "GET",
            self._base_url + "/tasks/" + _task_id(task_id),
            phase="poll",
            headers=self._headers,
        )
        payload = body.get("data", body)
        if not isinstance(payload, dict) or type(payload.get("status")) is not int:
            raise IsingQError("poll")
        status = payload["status"]
        if status not in (1, 2, 3):
            return {"status": "running", "provider_status": status}
        raw = payload.get("result")
        if not isinstance(raw, dict):
            return {"status": "failed", "provider_status": status}
        try:
            bits = raw.get("bits")
            if bits is None:
                bits = raw.get("spin_config")
            if not isinstance(bits, list) or any(type(v) is not int for v in bits):
                raise ValueError("Invalid solution vector")
            if bits and all(v in (-1, 1) for v in bits):
                bits = [(v + 1) // 2 for v in bits]
            energy = raw.get("energy")
            if energy is not None and (
                type(energy) not in (int, float) or not math.isfinite(energy)
            ):
                raise ValueError("Invalid energy")
            decoded = program.decode(bits, provider_energy=energy)
        except (TypeError, ValueError, OverflowError):
            # Terminal malformed results must not produce an endless polling loop.
            logger.debug("iSing task returned a malformed terminal result")
            return {
                "status": "failed",
                "provider_status": status,
                "error": "Invalid iSing solution",
            }
        return {"status": "succeeded", "provider_status": status, "result": decoded}
