"""Offline contract tests against the installed ``lqcloud`` SDK.

The other LogicalQubit suites replace the SDK with fakes, so they cannot see a
signature or payload change in a new lqcloud release (lqcloud 0.6 removed
``Job(use_async=...)``, for example).  These tests drive the real SDK objects
QuSpect relies on: the provider, a backend handle, circuit submission, job
polling and result objects.  HTTP is stubbed at ``request_with_retry`` -- the
patch point lqcloud's own test-suite uses -- and every socket is blocked.
"""

from __future__ import annotations

import json
import socket
import warnings
from typing import Any

import pytest

pytest.importorskip("lqcloud")

import lqcloud.auth  # noqa: E402
import lqcloud.job.job as sdk_job_module  # noqa: E402
import lqcloud.provider.cloud_provider as sdk_provider_module  # noqa: E402
from lqcloud.job import BatchResult, Job, Result  # noqa: E402

from quspect import QuantumCircuit  # noqa: E402
from quspect.backends import (  # noqa: E402
    ExecutionBatch,
    ExecutionRequest,
    JobReference,
)
from quspect.backends.logicalqubit import (  # noqa: E402
    LogicalQubitJob,
    LogicalQubitProvider,
)
from quspect.backends.logicalqubit._job import _convert_result  # noqa: E402

_URL = "https://lqcloud.invalid"
_CONFIG = {
    "name": "QZ02",
    "qubits": 4,
    "supported_modes": ["circuit"],
    "topology": {"coupling_map": [[0, 1], [1, 2], [2, 3]]},
}
_PLAIN = {
    "counts": {"00": 5, "11": 3},
    "shots": 8,
    "result_format": "counts",
    "success": True,
}
# lqcloud 0.6 readout-corrected payloads: counts mode keeps the corrected
# distribution under ``counts`` and the measurement under ``raw_counts``;
# memory mode attaches ``corrected_counts`` next to the raw shots.
_CORRECTED_COUNTS = {
    "counts": {"00": 4.6, "01": 0.0, "10": -0.1, "11": 3.5},
    "raw_counts": {"00": 5, "10": 1, "11": 2},
    "shots": 8,
    "result_format": "counts",
    "readout_correction_requested": True,
    "readout_correction_applied": True,
    "success": True,
}
_CORRECTED_MEMORY = {
    "memory": ["11", "00", "11", "11"],
    "corrected_counts": {"00": 0.9, "11": 3.1},
    "shots": 4,
    "result_format": "memory",
    "readout_correction_requested": True,
    "readout_correction_applied": True,
    "success": True,
}


class _Response:
    def __init__(self, payload: dict[str, Any], url: str) -> None:
        self.status_code = 200
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _Cloud:
    """Record submissions and serve scripted task-status payloads."""

    def __init__(self) -> None:
        self.submissions: list[dict[str, Any]] = []
        self.status: dict[str, dict[str, Any]] = {}

    def request(self, session: Any, method: str, url: str, **kwargs: Any) -> _Response:
        if method == "POST" and url == f"{_URL}/api/v1/tasks/async":
            self.submissions.append(kwargs["json"])
            task_id = f"task_{len(self.submissions)}"
            return _Response({"task_id": task_id, "status": "queued"}, url)
        if method == "GET" and url.startswith(f"{_URL}/api/v1/tasks/async/"):
            return _Response(self.status[url.rsplit("/", 1)[-1]], url)
        if method == "GET" and url == f"{_URL}/api/v1/qpus":
            return _Response({"qpus": [_CONFIG]}, url)
        raise AssertionError(f"unexpected request: {method} {url}")


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> _Cloud:
    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network access is forbidden in SDK contract tests")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    for name in ("LQCLOUD_API_KEY", "LQCLOUD_TOKEN", "LQCLOUD_URL", "LQCLOUD_STORE"):
        monkeypatch.delenv(name, raising=False)
    # Keep the developer's ~/.lqcloud (saved account) out of the test.
    monkeypatch.setattr(lqcloud.auth, "DEFAULT_CONFIG_FILE", tmp_path / "config.json")
    fake = _Cloud()
    monkeypatch.setattr(sdk_provider_module, "request_with_retry", fake.request)
    monkeypatch.setattr(sdk_job_module, "request_with_retry", fake.request)
    return fake


@pytest.fixture
def provider(cloud: _Cloud) -> LogicalQubitProvider:
    # store=False: nothing may be written under ~/.lqcloud/store.
    return LogicalQubitProvider(api_key="not-a-real-key", url=_URL, store=False)


def _bell() -> QuantumCircuit:
    circuit = QuantumCircuit(name="bell")
    circuit.h(0).cx(0, 1)
    return circuit.measure([0, 1], [0, 1])


def test_provider_and_backend_handle_without_deprecation_warnings(
    provider: LogicalQubitProvider,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend = provider.get_backend("QZ02")

    assert provider.url == _URL
    assert provider._provider.store is None
    assert backend.name == "QZ02"
    assert backend.num_qubits == 4
    assert provider.list_backends() == [_CONFIG]


def test_single_submission_payload_and_result(
    provider: LogicalQubitProvider, cloud: _Cloud
) -> None:
    backend = provider.get_backend("QZ02", verify=False)
    job = backend.submit(
        _bell(),
        shots=8,
        transpile=False,
        initial_layout=[1, 2],
        readout_correction=True,
    )

    assert isinstance(job._lq_job, Job)
    assert job.job_id == "task_1"
    assert job.reference.transport == "async"
    command = cloud.submissions[0]["command"]
    assert (command["action"], command["batch"], command["shots"]) == (
        "run_circuit",
        False,
        8,
    )
    (section,) = command["circuits"]
    assert section["initial_layout"] == [1, 2]
    assert section["readout_correction"] is True
    assert section["result_format"] == "counts"

    cloud.status["task_1"] = {
        "status": "completed",
        "qpu_name": "QZ02",
        "result": _CORRECTED_COUNTS,
    }
    result = job.result(timeout=5, poll_interval=0.01)
    assert result.counts == {"00": 5, "10": 1, "11": 2}
    assert result.probabilities == pytest.approx(
        {"00": 4.6 / 8, "01": 0.0, "10": -0.1 / 8, "11": 3.5 / 8}
    )


def test_native_batch_is_recoverable_from_a_saved_reference(
    provider: LogicalQubitProvider, cloud: _Cloud
) -> None:
    backend = provider.get_backend("QZ02", verify=False)
    request = ExecutionRequest.batch(
        (_bell(), _bell()),
        labels=("plain", "memory"),
        shots=8,
        options={"transpile": False, "initial_layout": [0, 1]},
    )
    job = backend.submit_request(request)

    command = cloud.submissions[0]["command"]
    assert command["batch"] is True
    assert len(command["circuits"]) == 2
    saved = job.reference.to_dict()
    assert saved["transport"] == "async"
    assert saved["child_job_ids"] == []

    cloud.status["task_1"] = {
        "status": "completed",
        "result": {"result_format": "batch", "results": [_PLAIN, _CORRECTED_MEMORY]},
    }
    recovered = provider.retrieve_job(JobReference.from_dict(saved))
    assert isinstance(recovered, LogicalQubitJob)
    assert isinstance(recovered._lq_job, Job)
    assert recovered._lq_job._batch_size == 2

    batch = recovered.result(timeout=5, poll_interval=0.01)
    assert isinstance(batch, ExecutionBatch)
    assert batch.labels == ("plain", "memory")
    assert batch.get_counts() == [{"00": 5, "11": 3}, {"00": 1, "11": 3}]
    assert batch.by_label("memory").memory == ["11", "00", "11", "11"]


def test_conversion_matches_the_sdk_result_accessors() -> None:
    payloads = (_PLAIN, _CORRECTED_COUNTS, _CORRECTED_MEMORY)
    children = [
        Result("QZ02", f"task_x#{index}", "COMPLETED", dict(data))
        for index, data in enumerate(payloads)
    ]
    converted = _convert_result(
        BatchResult("QZ02", "task_x", "COMPLETED", children),
        labels=("plain", "rc-counts", "rc-memory"),
        backend_name="QZ02",
        job_id="task_x",
    )

    with warnings.catch_warnings():
        # get_counts() announces lqcloud 0.6's readout-correction semantics.
        warnings.simplefilter("ignore", UserWarning)
        for sdk_result, result in zip(children, converted.results, strict=True):
            assert result.success
            assert result.counts == sdk_result.get_counts()
            assert result.probabilities == pytest.approx(sdk_result.get_probabilities())
