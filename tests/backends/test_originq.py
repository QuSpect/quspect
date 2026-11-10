"""Origin Quantum backend: OriginIR, bit order, HTTP client, jobs (pure Python).

Runs offline without pyqpanda3.  End-to-end and protocol checks against
pyqpanda3 as an independent reference are in ``test_originq_reference.py``.
"""

from __future__ import annotations

import json
import re

import pytest

from quspect import QuantumCircuit
from quspect.backends import ExecutionRequest, JobReference, JobStatus
from quspect.backends._exceptions import (
    AuthenticationError,
    BackendError,
    JobError,
    JobTimeoutError,
    PartialSubmissionError,
    TranslationError,
)
from quspect.backends.originq import OriginQClient, OriginQJob
from quspect.backends.originq import _provider as provider_module
from quspect.backends.originq._backend import OriginQBackend
from quspect.backends.originq._job import map_status, parse_task_result
from quspect.backends.originq._translator import decode_distribution, translate
from tests.backends._originq_fakes import (
    KEY,
    WK_C180_2_AVAILABLE,
    FakeCloud,
    chip_info_payload,
)

URL = "http://cloud.invalid"

# ===========================================================================
# Pure Python
# ===========================================================================


def test_decode_reverses_little_endian_into_clbit_order() -> None:
    assert decode_distribution({"0x1": 1.0}, [[0], [1]], 2) == {"10": 1.0}
    assert decode_distribution({"01": 1.0}, [[0], [1]], 2) == {"10": 1.0}


def test_decode_permuted_sparse_and_fanout_clbits() -> None:
    # bit 0 feeds clbits 2 and 0 (fan-out), bit 1 feeds clbit 3, clbit 1 unused.
    decoded = decode_distribution({"0x1": 0.25, "0x2": 0.75}, [[2, 0], [3]], 4)
    assert decoded == {"1010": 0.25, "0001": 0.75}


def test_decode_rejects_outcome_wider_than_measurement() -> None:
    with pytest.raises(TranslationError):
        decode_distribution({"0x4": 1.0}, [[0], [1]], 2)


# ------------------------------------------------------------- OriginIR

#: pyqpanda3 0.5.1 ``convert_qprog_to_originir`` spelling of each gate.
_SDK_SPELLING = {
    "id": (lambda c: c.id(1), ["I q[1]"]),
    "h": (lambda c: c.h(1), ["H q[1]"]),
    "sdg": (lambda c: c.sdg(1), ["DAGGER", "S q[1]", "ENDDAGGER"]),
    "tdg": (lambda c: c.tdg(1), ["DAGGER", "T q[1]", "ENDDAGGER"]),
    "sx": (lambda c: c.sx(1), ["X1 q[1]"]),
    "sxdg": (lambda c: c.sxdg(1), ["DAGGER", "X1 q[1]", "ENDDAGGER"]),
    "rx": (lambda c: c.rx(0.5, 1), ["RX q[1],(0.500000000000000)"]),
    "u3": (
        lambda c: c.u3(0.5, -1.25, 2.0, 1),
        ["U3 q[1],(0.500000000000000,-1.250000000000000,2.000000000000000)"],
    ),
    "cx": (lambda c: c.cx(2, 0), ["CNOT q[2],q[0]"]),
    "crz": (
        lambda c: c.crz(0.5, 2, 0),
        ["CONTROL q[2]", "RZ q[0],(0.500000000000000)", "ENDCONTROL"],
    ),
    "cp": (lambda c: c.cp(0.5, 2, 0), ["CR q[2],q[0],(0.500000000000000)"]),
    "ccx": (lambda c: c.ccx(2, 0, 1), ["CONTROL q[2],q[0]", "X q[1]", "ENDCONTROL"]),
}


@pytest.mark.parametrize("name", sorted(_SDK_SPELLING))
def test_gate_spelling_matches_pyqpanda3(name: str) -> None:
    build, expected = _SDK_SPELLING[name]
    circuit = QuantumCircuit()
    build(circuit)
    circuit.measure([1], [0])
    lines = translate(circuit).text.splitlines()
    assert lines[2:-1] == expected


def test_angles_keep_full_precision_without_exponents() -> None:
    circuit = QuantumCircuit()
    circuit.rx(1e-9, 0).rz(-3.0e5, 0).measure([0], [0])
    text = translate(circuit).text
    assert "RX q[0],(0.000000001000000)" in text
    assert "RZ q[0],(-300000.000000000000000)" in text
    assert "e" not in re.sub(r"[A-Z]+|q\[\d+\]|c\[\d+\]", "", text)


def test_measurements_are_emitted_once_ascending_into_dense_rank() -> None:
    circuit = QuantumCircuit()
    circuit.x(4)
    circuit.measure([4, 1, 4], [0, 2, 3])
    program = translate(circuit)
    assert program.text.splitlines()[:2] == ["QINIT 5", "CREG 2"]
    assert program.text.splitlines()[-2:] == ["MEASURE q[1],c[0]", "MEASURE q[4],c[1]"]
    assert program.measurement.measured_qubits == (1, 4)
    assert program.measurement.clbits == ((2,), (0, 3))


def test_qubit_map_places_wires_before_canonical_ordering() -> None:
    circuit = QuantumCircuit()
    circuit.cx(0, 1).measure([0, 1], [0, 1])
    program = translate(circuit, qubit_map=[58, 49])
    assert "CNOT q[58],q[49]" in program.text
    assert program.num_qubits == 59
    # physical 49 (wire 1 -> clbit 1) becomes result bit 0.
    assert program.measurement.clbits == ((1,), (0,))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda c: c.h(0).measure([0], [0]).x(0), "after being measured"),
        (lambda c: c.x(0).measure([0, 1], [0, 0]), "more than once"),
        (lambda c: c.h(0), "no measurement"),
    ],
)
def test_translation_rejects_what_the_provider_cannot_express(build, message) -> None:
    circuit = QuantumCircuit()
    build(circuit)
    with pytest.raises(TranslationError, match=message):
        translate(circuit)


def test_unbound_parameter_is_rejected() -> None:
    from quspect.circuit.parameter import Parameter

    circuit = QuantumCircuit()
    circuit.rx(Parameter("theta"), 0).measure([0], [0])
    with pytest.raises(TranslationError, match="unbound"):
        translate(circuit)


# ------------------------------------------------------------- HTTP client


class _Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict, bytes | None]] = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, dict(headers), body))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(obj) -> tuple[int, bytes]:
    return 200, json.dumps({"success": True, "code": 10000, "obj": obj}).encode()


def test_client_sends_auth_and_task_mode_headers() -> None:
    transport = _Recorder(_ok({"taskId": "T1"}), _ok({"taskState": "1"}))
    client = OriginQClient(KEY, URL + "/", transport=transport)
    task = client.submit(
        ["QINIT 1\nCREG 1\nH q[0]\nMEASURE q[0],c[0]\n"],
        machine_type="0",
        configuration={"shot": 10, "measure_type": 1},
        qubit_num=1,
        clbit_num=1,
    )
    client.task_detail(task)
    (m1, u1, h1, b1), (m2, u2, h2, b2) = transport.calls
    assert (m1, u1) == ("POST", URL + "/oqcs/task/origin/submit.json")
    assert h1["Authorization"] == f"oqcs_auth={KEY}" and h1["task_mode"] == "1"
    assert json.loads(b1) == {
        "QProg": ["QINIT 1\nCREG 1\nH q[0]\nMEASURE q[0],c[0]\n"],
        "batchNo": "",
        "taskFrom": 4,
        "QMachineType": "0",
        "Configuration": {"shot": 10, "measure_type": 1},
        "qubitNum": 1,
        "classicalbitNum": 1,
    }
    assert u2.endswith("/oqcs/task/origin/taskDetail.json") and h2["task_mode"] == "0"
    assert json.loads(b2) == {"taskId": "T1"}
    assert KEY not in repr(client)


def test_client_batch_submission_uses_task_mode_2_without_sizes() -> None:
    transport = _Recorder(_ok({"taskId": "T2"}))
    OriginQClient(KEY, URL, transport=transport).submit(
        ["a", "b"], machine_type="5", configuration={}, chip_id="WK_C180_2", batch=True
    )
    _, _, headers, body = transport.calls[0]
    payload = json.loads(body)
    assert headers["task_mode"] == "2"
    assert payload["ChipID"] == "WK_C180_2"
    assert "qubitNum" not in payload and "classicalbitNum" not in payload


def test_client_retries_reads_but_never_submissions() -> None:
    transport = _Recorder(ConnectionResetError("reset"), (502, b""), _ok({"x": 1}))
    client = OriginQClient(KEY, URL, transport=transport, backoff=0)
    assert client.task_detail("T") == {"x": 1}
    assert len(transport.calls) == 3

    transport = _Recorder(ConnectionResetError("reset"), _ok({"taskId": "dup"}))
    client = OriginQClient(KEY, URL, transport=transport, backoff=0)
    with pytest.raises(BackendError, match="submit"):
        client.submit(["p"], machine_type="0", configuration={})
    assert len(transport.calls) == 1


def test_client_surfaces_provider_errors() -> None:
    failure = (200, json.dumps({"success": False, "code": 10001, "message": "nope"}))
    client = OriginQClient(
        KEY, URL, transport=_Recorder((failure[0], failure[1].encode()))
    )
    with pytest.raises(BackendError, match="nope"):
        client.task_detail("T")
    client = OriginQClient(KEY, URL, transport=_Recorder((401, b"{}")))
    with pytest.raises(AuthenticationError):
        client.list_resources()


# ------------------------------------------------------------- results


def _reference(job_id: str, **kwargs) -> JobReference:
    return JobReference(
        platform="originq",
        backend_name="full_amplitude",
        job_id=job_id,
        shots=kwargs.pop("shots", 100),
        experiment_shapes=((1, 1),),
        measurement_maps=(((0,),),),
        **kwargs,
    )


class _StubClient:
    def __init__(self, details):
        self.details = list(details)

    def task_detail(self, task_id):
        return self.details.pop(0) if len(self.details) > 1 else self.details[0]


def test_parse_task_result_reads_json_string_entries() -> None:
    detail = {"taskResult": ['{"value":[0.758,0.242],"key":["0x2","0x3"]}']}
    assert parse_task_result(detail) == [{"0x2": 0.758, "0x3": 0.242}]


def test_status_mapping_polling_and_counts() -> None:
    done = {"taskState": "3", "taskResult": ['{"key":["0x1"],"value":[1.0]}']}
    client = _StubClient(
        [{"taskState": "5"}, {"taskState": "1"}, {"taskState": "2"}, done]
    )
    job = OriginQJob(client, "T", reference=_reference("T"))
    assert job.status() == JobStatus.QUEUED
    assert job.status() == JobStatus.PENDING
    result = job.result(poll_interval=1e-3)
    assert result.probabilities == {"1": 1.0}
    assert result.counts == {"1": 100}
    assert result.metadata["counts_source"] == "probabilities"


def test_timeout_and_failure() -> None:
    job = OriginQJob(_StubClient([{"taskState": "2"}]), "T", reference=_reference("T"))
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01, poll_interval=1e-3)
    failed = {"taskState": "4", "errInfo": "chip offline"}
    job = OriginQJob(_StubClient([failed]), "T", reference=_reference("T"))
    with pytest.raises(JobError, match="chip offline"):
        job.result(poll_interval=1e-3)


# ------------------------------------------------------------- cancellation
#
# The request is the one pyqpanda3 0.5.1 sends (compared with the SDK in
# ``test_originq_reference.py``).  The answers follow the SDK's handling:
# ``success`` alone decides, any HTTP error is an error.  Nothing here was
# recorded on the real cloud.

_ACCEPTED = {"success": True, "code": 10000, "message": "success", "obj": None}
_REFUSED = {"success": False, "code": 10001, "message": "task finished", "obj": None}


def _answer(payload, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode()


def _fake_job(cloud: FakeCloud, task_id: str = "T1", states=("1", "2")) -> OriginQJob:
    """A job for a task registered directly on the fake (no pyqpanda3 needed)."""
    cloud.tasks[task_id] = {"results": [{"0x1": 1.0}], "states": list(states)}
    client = OriginQClient(cloud.key, URL, transport=cloud, backoff=0)
    return OriginQJob(client, task_id, reference=_reference(task_id))


def test_client_cancel_sends_the_request_pyqpanda3_sends() -> None:
    transport = _Recorder(_answer(_ACCEPTED))
    assert OriginQClient(KEY, URL, transport=transport).cancel_task("T1") is True
    ((method, url, headers, body),) = transport.calls
    assert (method, url) == ("POST", URL + "/oqcs/task/cancelTask.json")
    assert headers == {
        "Accept": "*/*",
        "Authorization": f"oqcs_auth={KEY}",
        "Content-Type": "application/json",
        "origin-language": "en",
        "task_mode": "0",
    }
    assert json.loads(body) == {"taskId": "T1"}


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        (_ACCEPTED, True),
        ({"success": True}, True),  # pyqpanda3 reads ``success`` only
        ({"success": True, "code": 10001, "message": "odd"}, True),
        (_REFUSED, False),
        ({"success": False}, False),
    ],
)
def test_client_cancel_returns_the_clouds_decision(answer, accepted) -> None:
    transport = _Recorder(_answer(answer))
    client = OriginQClient(KEY, URL, transport=transport, backoff=0)
    assert client.cancel_task("T1") is accepted
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "failure",
    [ConnectionResetError("reset"), (502, b""), _answer(_ACCEPTED, status=500)],
)
def test_client_cancel_is_sent_once_and_never_retried(failure) -> None:
    transport = _Recorder(failure, _answer(_ACCEPTED))
    client = OriginQClient(KEY, URL, transport=transport, backoff=0)
    with pytest.raises(BackendError, match="cancelTask"):
        client.cancel_task("T1")
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        ((401, b"{}"), AuthenticationError, "API key"),
        ((403, b"{}"), AuthenticationError, "API key"),
        (_answer({"message": "no such task"}, 404), BackendError, "404: no such task"),
        (_answer({"message": "bad request"}, 400), BackendError, "400: bad request"),
        ((200, b"<html>"), BackendError, "non-JSON"),
        (_answer({"code": 10000, "message": "x"}), BackendError, "success"),
        (_answer({"success": "true", "code": 10000}), BackendError, "success"),
        (_answer([]), BackendError, "success"),
    ],
)
def test_client_cancel_raises_unless_the_cloud_decided(
    response, error, message
) -> None:
    client = OriginQClient(KEY, URL, transport=_Recorder(response), backoff=0)
    with pytest.raises(error, match=message):
        client.cancel_task("T1")


def test_backends_declare_cancellation() -> None:
    cloud = FakeCloud()
    assert _backend(cloud, "WK_C180_2").capabilities.supports_cancel is True
    assert _backend(cloud, "full_amplitude").capabilities.supports_cancel is True


def test_job_cancel_accepted_with_a_single_request() -> None:
    cloud = FakeCloud()
    job = _fake_job(cloud)
    assert job.cancel() is True
    assert cloud.tasks["T1"]["cancelled"] is True
    ((method, path, headers),) = cloud.requests  # no status query first
    assert (method, path) == ("POST", "/oqcs/task/cancelTask.json")
    assert headers["task_mode"] == "0"


def test_job_cancel_refused_for_a_finished_task() -> None:
    cloud = FakeCloud()
    job = _fake_job(cloud, states=())
    assert job.cancel() is False
    assert "cancelled" not in cloud.tasks["T1"]
    assert job.status() == JobStatus.COMPLETED
    assert job.result(poll_interval=1e-3).probabilities == {"1": 1.0}


def test_job_cancel_refused_for_an_unknown_task() -> None:
    cloud = FakeCloud()
    client = OriginQClient(cloud.key, URL, transport=cloud, backoff=0)
    job = OriginQJob(client, "NOPE", reference=_reference("NOPE"))
    assert job.cancel() is False
    assert len(cloud.requests) == 1


def test_job_cancel_network_failure_is_a_job_error_and_not_retried() -> None:
    cloud = FakeCloud()
    job = _fake_job(cloud)
    cloud.fail_next_cancel = 1
    with pytest.raises(JobError, match="cancelTask") as info:
        job.cancel()
    assert info.value.job_id == "T1"
    assert len(cloud.requests) == 1
    assert "cancelled" not in cloud.tasks["T1"]


@pytest.mark.parametrize(
    ("response", "cause"),
    [
        ((500, b""), BackendError),
        (_answer({"message": "bad request"}, 400), BackendError),
        ((401, b"{}"), AuthenticationError),
    ],
)
def test_job_cancel_http_refusal_is_a_job_error(response, cause) -> None:
    transport = _Recorder(response)
    client = OriginQClient(KEY, URL, transport=transport, backoff=0)
    job = OriginQJob(client, "T", reference=_reference("T"))
    with pytest.raises(JobError) as info:
        job.cancel()
    assert info.value.job_id == "T"
    assert type(info.value.__cause__) is cause
    assert len(transport.calls) == 1


@pytest.mark.parametrize("state", ["0", "6", "7", "-1", "cancelled", "", None])
def test_task_states_outside_the_sdk_enum_stay_unknown(state) -> None:
    """pyqpanda3's JobStatus stops at 5 and names no cancelled state."""
    assert map_status(state) == JobStatus.UNKNOWN


def test_state_after_cancel_is_whatever_the_cloud_reports() -> None:
    """No cancelled ``taskState`` is known, so nothing maps to CANCELLED."""
    cloud = FakeCloud()
    job = _fake_job(cloud)
    assert job.cancel() is True
    assert job.status() == JobStatus.PENDING  # the fake invents no state

    cloud.cancelled_state = "6"  # stand-in for a value outside the SDK enum
    assert job.status() == JobStatus.UNKNOWN
    assert not job.done()
    with pytest.raises(JobTimeoutError, match="status: unknown"):
        job.result(timeout=0.01, poll_interval=1e-3)

    cloud.cancelled_state = "4"
    assert job.status() == JobStatus.FAILED
    with pytest.raises(JobError, match="failed"):
        job.result(poll_interval=1e-3)


def test_partial_submission_cancels_the_tasks_already_created() -> None:
    transport = _Recorder(
        _ok({"taskId": "T1"}),
        _ok({"taskId": "T2"}),
        _ok({"taskId": "T3"}),
        ConnectionResetError("reset"),  # fourth submission
        _answer(_ACCEPTED),
        _answer(_REFUSED),
        ConnectionResetError("reset"),  # third cancel
    )
    resource = next(
        r for r in FakeCloud().resources() if r["pyqpandaChipId"] == "full_amplitude"
    )
    backend = OriginQBackend(
        OriginQClient(KEY, URL, transport=transport, backoff=0), resource
    )
    circuit = QuantumCircuit()
    circuit.h(0).measure([0], [0])
    request = ExecutionRequest.batch([circuit] * 4, labels=list("abcd"), shots=10)
    with pytest.raises(PartialSubmissionError, match="after 3 experiment") as info:
        backend.submit_request(request)
    assert info.value.submitted_job_ids == ("T1", "T2", "T3")
    assert info.value.cancellation_failures == ("T2", "T3")
    cancels = [(url, json.loads(body)) for _, url, _, body in transport.calls[4:]]
    assert cancels == [
        (URL + "/oqcs/task/cancelTask.json", {"taskId": task})
        for task in ("T1", "T2", "T3")
    ]


def test_api_key_url_extraction_never_returns_the_secret() -> None:
    url = "http://example.invalid"
    key = f"{len(url):04x}" + url.encode().hex() + "deadbeef" * 4
    assert provider_module.url_from_api_key(key) == url
    assert provider_module.url_from_api_key("not-a-key") is None


def test_provider_lists_sampling_backends_only() -> None:
    cloud = FakeCloud()
    provider = provider_module.OriginQProvider(api_key=KEY, url=URL, transport=cloud)
    assert provider.list_backends() == {"WK_C180_2": True, "full_amplitude": True}
    with pytest.raises(ValueError, match="amplitudes"):
        provider.get_backend("partial_amplitude")
    simulator = provider.get_backend("full_amplitude")
    assert simulator.is_simulator and simulator.num_qubits == 35
    qpu = provider.get_backend("WK_C180_2")
    assert qpu.available_qubits == WK_C180_2_AVAILABLE
    assert qpu.capabilities.max_shots == 49600
    assert len(qpu.config["topology"]["coupling_map"]) == 20


def test_qpu_option_validation() -> None:
    backend = _backend(FakeCloud(), "WK_C180_2")
    circuit = QuantumCircuit()
    circuit.h(0).cx(0, 1).measure([0, 1], [0, 1])
    with pytest.raises(ValueError, match="conflicts"):
        backend.submit(circuit, provider_mapping=True)
    with pytest.raises(ValueError, match="require transpile=True"):
        backend.submit(circuit, transpile=False, physical_qubits=[39, 48])
    with pytest.raises(ValueError, match="unavailable"):
        backend.submit(circuit, physical_qubits=[0, 1])
    with pytest.raises(ValueError, match="at most 49600"):
        backend.submit(circuit, shots=50_000)
    with pytest.raises(ValueError, match="Unsupported"):
        backend.submit(circuit, bogus=1)
    with pytest.raises(ValueError, match="cloud simulator"):
        _backend(FakeCloud(), "full_amplitude").submit(
            circuit, readout_correction=False
        )


def test_default_pool_is_largest_connected_component() -> None:
    chips = {"X": chip_info_payload(6, [0, 1, 2, 4, 5], [(0, 1), (1, 2), (4, 5)])}
    assert _backend(FakeCloud(chips=chips), "X").default_physical_qubits() == [0, 1, 2]


def _backend(cloud: FakeCloud, name: str) -> OriginQBackend:
    client = OriginQClient(cloud.key, URL, transport=cloud, backoff=0)
    resource = next(
        r
        for r in cloud.resources()
        if r["pyqpandaChipId"] == name and r["machineType"] in (0, 5)
    )
    return OriginQBackend(client, resource)


def test_real_chip_info_parses_to_the_sdk_reported_topology() -> None:
    """Recorded WK_C180_2 chipInfo vs. what pyqpanda3's chip_info() returned."""
    from quspect.backends.originq._backend import edges_from_adjacency
    from tests.backends._originq_fakes import WK_C180_2_EDGES, real_chip_info

    info = real_chip_info()
    assert edges_from_adjacency(info["adjMatrix"]) == [list(e) for e in WK_C180_2_EDGES]
    backend = _backend(FakeCloud(), "WK_C180_2")
    assert backend.default_physical_qubits() == WK_C180_2_AVAILABLE


def _wk_c180_backend() -> OriginQBackend:
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "originq_chip_WK_C180.json"
    return _backend(
        FakeCloud(chips={"WK_C180": json.loads(path.read_text())}), "WK_C180"
    )


def test_placement_avoids_badly_calibrated_qubits_on_wk_c180() -> None:
    """Regression: routing onto WK_C180 qubits 0-2 returned wrong results.

    Recorded calibration: qubit 0 has single-gate fidelity 0 (failed), qubits
    1 and 2 read out at 0.88 / 0.83.  The provider's own mapping avoided
    them and was correct; placement must avoid them too.
    """
    backend = _wk_c180_backend()
    fidelity = backend._chip()["qubit_fidelity"]
    assert fidelity[0] == 0.0
    edges = {tuple(e) for e in backend.config["topology"]["coupling_map"]}
    median = sorted(fidelity.values())[len(fidelity) // 2]
    for size in (1, 2, 3, 5, 8):
        region = backend.placement_qubits(size)
        assert len(region) == size
        assert not set(region) & {0, 1, 2}
        assert all(fidelity[q] >= median for q in region), (size, region)
        # connected: every qubit but one touches another region qubit
        if size > 1:
            assert all(
                any((min(a, b), max(a, b)) in edges for b in region if b != a)
                for a in region
            )


def test_placement_without_calibration_falls_back_to_largest_component() -> None:
    chips = {"X": chip_info_payload(6, [0, 1, 2, 4, 5], [(0, 1), (1, 2), (4, 5)])}
    backend = _backend(FakeCloud(chips=chips), "X")
    assert backend.placement_qubits(2) == [0, 1]
    assert backend.placement_qubits(4) == [0, 1, 2]  # nothing fits: whole component
