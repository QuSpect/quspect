"""Origin Quantum backend checked against pyqpanda3 as an independent reference.

pyqpanda3 is a test-only dependency (``pip install -e '.[originq-tests]'``):
its OriginIR parser and CPUQVM evaluate what QuSpect emits, and the SDK's own
HTTP traffic is compared with QuSpect's client to catch protocol drift.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

pq = pytest.importorskip("pyqpanda3.core", reason="pyqpanda3 reference not installed")

from pyqpanda3.intermediate_compiler import (  # noqa: E402
    convert_originir_string_to_qprog,
)

from quspect import QuantumCircuit  # noqa: E402
from quspect.backends import (  # noqa: E402
    CompositeJob,
    ExecutionRequest,
    JobReference,
    JobStatus,
)
from quspect.backends._exceptions import JobError  # noqa: E402
from quspect.backends.originq import _provider as provider_module  # noqa: E402
from quspect.backends.originq._job import _STATUS_MAP  # noqa: E402
from quspect.backends.originq._translator import translate  # noqa: E402
from quspect.simulation import Simulator  # noqa: E402
from tests.backends._originq_fakes import (  # noqa: E402
    KEY,
    WK_C180_2_AVAILABLE,
    FakeCloud,
    assert_same_distribution,
    chip_info_payload,
    quspect_exact_distribution,
)
from tests.backends.test_originq import URL, _backend  # noqa: E402

# ===========================================================================
# pyqpanda3 as an independent reference
# ===========================================================================

_PREP = [(0.3, 1.1), (1.7, -0.4), (2.2, 0.9)]
_T, _F, _L = 0.37, -1.21, 2.05
_GATE_CASES = {
    "id": lambda c: c.id(1),
    "h": lambda c: c.h(1),
    "x": lambda c: c.x(1),
    "y": lambda c: c.y(1),
    "z": lambda c: c.z(1),
    "s": lambda c: c.s(1),
    "sdg": lambda c: c.sdg(1),
    "t": lambda c: c.t(1),
    "tdg": lambda c: c.tdg(1),
    "sx": lambda c: c.sx(1),
    "sxdg": lambda c: c.sxdg(1),
    "rx": lambda c: c.rx(_T, 1),
    "ry": lambda c: c.ry(_T, 1),
    "rz": lambda c: c.rz(_T, 1),
    "p": lambda c: c.p(_T, 1),
    "u1": lambda c: c.u1(_T, 1),
    "u2": lambda c: c.u2(_F, _L, 1),
    "u3": lambda c: c.u3(_T, _F, _L, 1),
    "cx": lambda c: c.cx(2, 0),
    "cz": lambda c: c.cz(2, 0),
    "swap": lambda c: c.swap(2, 0),
    "crx": lambda c: c.crx(_T, 2, 0),
    "cry": lambda c: c.cry(_T, 2, 0),
    "crz": lambda c: c.crz(_T, 2, 0),
    "cp": lambda c: c.cp(_T, 2, 0),
    "cu3": lambda c: c.cu3(_T, _F, _L, 2, 0),
    "rxx": lambda c: c.rxx(_T, 2, 0),
    "ryy": lambda c: c.ryy(_F, 0, 2),
    "rzz": lambda c: c.rzz(_L, 1, 2),
    "ccx": lambda c: c.ccx(2, 0, 1),
}


@pytest.mark.parametrize("name", sorted(_GATE_CASES))
def test_every_gate_matches_quspect_through_pyqpanda3_parser(name: str) -> None:
    circuit = QuantumCircuit()
    for qubit, (ry, rz) in enumerate(_PREP):
        circuit.ry(ry, qubit).rz(rz, qubit)
    _GATE_CASES[name](circuit)
    expected = np.asarray(
        Simulator().run(circuit, shots=1, save_statevector=True).statevector
    )
    measured = circuit.copy()
    measured.measure([0], [0])
    ir = "\n".join(
        line
        for line in translate(measured).text.splitlines()
        if not line.startswith("MEASURE")
    )
    vm = pq.CPUQVM()
    vm.run(convert_originir_string_to_qprog(ir + "\n"), 1)
    actual = np.asarray(vm.result().get_state_vector())
    # Both frameworks index the state little-endian (bit q = qubit q).
    assert abs(abs(np.vdot(expected, actual)) - 1.0) < 1e-12


def _cases() -> dict[str, QuantumCircuit]:
    cases: dict[str, QuantumCircuit] = {}
    c = QuantumCircuit()
    c.ry(1.0, 0).x(1).measure([0, 1], [0, 1])
    cases["asymmetric"] = c
    c = QuantumCircuit()
    c.ry(1.0, 0).x(1).measure([0, 1], [1, 0])
    cases["permuted_clbits"] = c
    c = QuantumCircuit()
    c.ry(0.7, 0).x(1).measure([1], [1]).measure([0], [0])
    cases["reverse_measure_order"] = c
    c = QuantumCircuit()
    c.ry(0.4, 0).ry(2.1, 1).measure([0, 1], [3, 0])
    cases["sparse_clbits"] = c
    c = QuantumCircuit()
    c.ry(1.3, 0).measure([0, 0], [0, 2])
    cases["fanout"] = c
    c = QuantumCircuit()
    c.h(0).cx(0, 1).ry(0.9, 2).cx(1, 2).measure([2], [0])
    cases["partial_measurement"] = c
    c = QuantumCircuit()
    c.h(0).ry(0.3, 1).ry(1.2, 2).ry(2.0, 3)
    c.cx(0, 3).ccx(0, 1, 2).cu3(0.4, 0.5, 0.6, 3, 1).swap(0, 2).crz(0.8, 2, 3)
    c.barrier()
    c.measure([0, 1, 2, 3], [2, 0, 3, 1])
    cases["four_qubit_entangled"] = c
    c = QuantumCircuit()
    c.h(0).cx(0, 1).measure([0], [0]).barrier().ry(0.5, 1).measure([1], [1])
    cases["barrier_after_measure"] = c
    return cases


CASES = _cases()


def test_fake_reproduces_measured_cloud_key_order() -> None:
    """Regression guard for the fake itself (values recorded on the cloud)."""
    from tests.backends._originq_fakes import exact_cloud_probabilities

    ir = "QINIT 2\nCREG 2\nX q[0]\nMEASURE q[0],c[1]\nMEASURE q[1],c[0]\n"
    assert exact_cloud_probabilities(ir) == pytest.approx({"0x1": 1.0})
    ir = "QINIT 2\nCREG 2\nX q[0]\nMEASURE q[1],c[0]\nMEASURE q[0],c[1]\n"
    with pytest.raises(RuntimeError, match="MEASURE order"):
        exact_cloud_probabilities(ir)


@pytest.mark.parametrize("name", sorted(CASES))
def test_cloud_simulator_end_to_end_matches_quspect(name: str) -> None:
    circuit = CASES[name]
    result = _backend(FakeCloud(), "full_amplitude").run(
        circuit, shots=1000, poll_interval=1e-3
    )
    assert result.num_clbits == circuit.num_clbits
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))


@pytest.mark.parametrize("name", sorted(CASES))
def test_qpu_quspect_routing_end_to_end_matches_quspect(name: str) -> None:
    """transpile=True on WK_C180_2: the fake rejects any off-graph gate."""
    circuit = CASES[name]
    cloud = FakeCloud()
    result = _backend(cloud, "WK_C180_2").run(circuit, shots=2000, poll_interval=1e-3)
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))
    payload = cloud.submissions[0]
    assert payload["Configuration"]["mappingFlag"] is False
    used = {int(q) for q in re.findall(r"q\[(\d+)\]", payload["QProg"][0])}
    assert used <= set(WK_C180_2_AVAILABLE)


def test_qpu_routing_on_a_line_forces_swaps_and_still_decodes() -> None:
    chips = {"LINE4": chip_info_payload(4, [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])}
    circuit = QuantumCircuit()
    circuit.ry(0.6, 0).ry(1.9, 3).cx(0, 3).cx(3, 1).ry(0.4, 2).cx(2, 0)
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])
    result = _backend(FakeCloud(chips=chips), "LINE4").run(
        circuit, shots=100, poll_interval=1e-3
    )
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))


def test_qpu_routing_keeps_per_qubit_measurements_terminal() -> None:
    """One measure instruction per qubit was refused after routing a SWAP."""
    chips = {
        "LINE5": chip_info_payload(5, [0, 1, 2, 3, 4], [(i, i + 1) for i in range(4)])
    }
    circuit = QuantumCircuit()
    circuit.h(0).ry(0.4, 1).ry(1.1, 2)
    circuit.cx(2, 1).cx(0, 2).cx(0, 1).cx(2, 1)
    for qubit in range(3):
        circuit.measure([qubit], [2 - qubit])
    result = _backend(FakeCloud(chips=chips), "LINE5").run(
        circuit, shots=100, poll_interval=1e-3
    )
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))


def test_qpu_provider_mapping_mode_submits_logical_circuit() -> None:
    cloud = FakeCloud()
    circuit = CASES["permuted_clbits"]
    result = _backend(cloud, "WK_C180_2").run(
        circuit, shots=100, transpile=False, poll_interval=1e-3
    )
    config = cloud.submissions[0]["Configuration"]
    assert config["mappingFlag"] is True and config["amendFlag"] == 1
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))


def test_physical_qubits_pin_the_placement() -> None:
    cloud = FakeCloud()
    _backend(cloud, "WK_C180_2").run(
        CASES["asymmetric"], physical_qubits=[57, 66], poll_interval=1e-3
    )
    used = set(re.findall(r"q\[(\d+)\]", cloud.submissions[0]["QProg"][0]))
    assert used <= {"57", "66"}


def test_qpu_request_uses_one_native_batch_in_order() -> None:
    cloud = FakeCloud()
    names = ["asymmetric", "sparse_clbits", "fanout"]
    request = ExecutionRequest.batch([CASES[n] for n in names], labels=names, shots=500)
    batch = _backend(cloud, "WK_C180_2").execute(request)
    assert len(cloud.submissions) == 1 and len(cloud.submissions[0]["QProg"]) == 3
    assert batch.labels == tuple(names)
    for name in names:
        assert_same_distribution(
            batch.by_label(name).probabilities, quspect_exact_distribution(CASES[name])
        )


def test_simulator_request_becomes_one_task_per_circuit() -> None:
    cloud = FakeCloud()
    names = ["asymmetric", "permuted_clbits"]
    request = ExecutionRequest.batch([CASES[n] for n in names], labels=names, shots=10)
    job = _backend(cloud, "full_amplitude").submit_request(request)
    assert isinstance(job, CompositeJob) and len(cloud.submissions) == 2
    batch = job.result(poll_interval=1e-3)
    for name in names:
        assert_same_distribution(
            batch.by_label(name).probabilities, quspect_exact_distribution(CASES[name])
        )


def test_parameter_sweep_binds_before_translation() -> None:
    from quspect.circuit.parameter import Parameter

    theta = Parameter("theta")
    template = QuantumCircuit()
    template.ry(theta, 0).cx(0, 1).measure([0, 1], [1, 0])
    values = [0.0, 0.8, math.pi]
    request = ExecutionRequest.sweep(template, [{"theta": v} for v in values], shots=10)
    batch = _backend(FakeCloud(), "WK_C180_2").execute(request)
    for value, result in zip(values, batch, strict=True):
        p1 = math.sin(value / 2) ** 2
        expected = {k: v for k, v in {"00": 1 - p1, "11": p1}.items() if v > 1e-12}
        assert_same_distribution(result.probabilities, expected)


def test_polling_survives_dropped_connections() -> None:
    cloud = FakeCloud()
    cloud.fail_next_detail = 2
    result = _backend(cloud, "full_amplitude").run(
        CASES["asymmetric"], shots=10, poll_interval=1e-3
    )
    assert_same_distribution(
        result.probabilities, quspect_exact_distribution(CASES["asymmetric"])
    )


def test_reference_round_trip_recovers_and_decodes() -> None:
    """A serialized reference alone is enough to decode a recovered batch."""
    cloud = FakeCloud()
    provider = provider_module.OriginQProvider(api_key=KEY, url=URL, transport=cloud)
    names = ["permuted_clbits", "sparse_clbits"]
    job = provider.get_backend("WK_C180_2").submit_request(
        ExecutionRequest.batch([CASES[n] for n in names], labels=names, shots=100)
    )
    restored = JobReference.from_dict(json.loads(json.dumps(job.reference.to_dict())))
    batch = provider.retrieve_job(restored).result(poll_interval=1e-3)
    for name in names:
        assert_same_distribution(
            batch.by_label(name).probabilities, quspect_exact_distribution(CASES[name])
        )
    other = provider_module.OriginQProvider(api_key=KEY, url="http://other.invalid")
    with pytest.raises(ValueError, match="endpoint"):
        other.retrieve_job(restored)


def test_cancel_end_to_end_on_the_fake_cloud() -> None:
    cloud = FakeCloud()
    job = _backend(cloud, "WK_C180_2").submit(CASES["asymmetric"], shots=100)
    assert job.cancel() is True
    assert cloud.tasks[job.job_id]["cancelled"] is True

    names = ["asymmetric", "permuted_clbits"]
    request = ExecutionRequest.batch([CASES[n] for n in names], labels=names, shots=10)
    composite = _backend(cloud, "full_amplitude").submit_request(request)
    assert isinstance(composite, CompositeJob)
    assert composite.cancel() is True
    assert all(cloud.tasks[task]["cancelled"] for task in composite.job_ids)
    assert composite.status() == JobStatus.CANCELLED

    # The cloud refuses a task it has already cancelled, so the logical job
    # reports False.
    assert composite.cancel() is False

    # One child of a fresh job finishes first: refused, so False again.
    composite = _backend(cloud, "full_amplitude").submit_request(request)
    cloud.tasks[composite.job_ids[0]]["states"].clear()
    assert composite.cancel() is False
    assert cloud.tasks[composite.job_ids[1]]["cancelled"] is True


# --------------------------------------------- protocol contract vs the SDK


@pytest.fixture
def sdk_capture_server(tmp_path):
    """A stand-in server (separate process) recording what pyqpanda3 sends."""
    from tests.backends._originq_fakes import real_chip_info, start_capture_server

    pytest.importorskip("pyqpanda3.qcloud")
    url, recorded, process = start_capture_server(
        tmp_path, real_chip_info(), FakeCloud().resources()
    )
    try:
        yield url, recorded
    finally:
        process.kill()
        process.wait(timeout=10)


def _normalize_ir(text: str) -> str:
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.6f}", text)


def test_submission_payloads_match_pyqpanda3(sdk_capture_server) -> None:
    """Detect protocol drift: QuSpect must send what the SDK sends."""
    from pyqpanda3.qcloud import QCloudOptions, QCloudService

    url, recorded = sdk_capture_server
    service = QCloudService(KEY, url)
    prog = pq.QProg()
    prog << pq.RY(57, 0.3) << pq.CZ(57, 66) << pq.measure(57, 0) << pq.measure(66, 1)
    options = QCloudOptions()
    options.set_mapping(False)
    service.backend("WK_C180_2").run(prog, 1000, options)
    service.backend("WK_C180_2").run([prog, prog], 500, options)
    small = pq.QProg()
    small << pq.RY(0, 0.3) << pq.CZ(0, 1) << pq.measure(0, 0) << pq.measure(1, 1)
    service.backend("full_amplitude").run(small, 100)
    sdk = [r for r in recorded() if r["path"].endswith("submit.json")]

    provider = provider_module.OriginQProvider(api_key=KEY, url=url)
    circuit = QuantumCircuit()
    circuit.ry(0.3, 57).cz(57, 66).measure([57, 66], [0, 1])
    small_circuit = QuantumCircuit()
    small_circuit.ry(0.3, 0).cz(0, 1).measure([0, 1], [0, 1])
    start = len(recorded())
    qpu = provider.get_backend("WK_C180_2")
    qpu.submit(circuit, shots=1000, transpile=False, provider_mapping=False)
    qpu.submit_request(
        ExecutionRequest.batch(
            [circuit, circuit],
            shots=500,
            options={"transpile": False, "provider_mapping": False},
        )
    )
    provider.get_backend("full_amplitude").submit(small_circuit, shots=100)
    ours = [r for r in recorded()[start:] if r["path"].endswith("submit.json")]

    assert len(sdk) == len(ours) == 3
    for theirs, mine in zip(sdk, ours, strict=True):
        mine_h = {k.lower(): v for k, v in mine["headers"].items()}
        theirs_h = {k.lower(): v for k, v in theirs["headers"].items()}
        assert mine_h["task_mode"] == theirs_h["task_mode"]
        assert mine_h["authorization"] == theirs_h["authorization"]
        t_body, m_body = dict(theirs["body"]), dict(mine["body"])
        t_prog, m_prog = t_body.pop("QProg"), m_body.pop("QProg")
        assert m_body == t_body
        assert [_normalize_ir(p) for p in m_prog] == [_normalize_ir(p) for p in t_prog]


def test_chip_info_parsing_matches_pyqpanda3(sdk_capture_server) -> None:
    from pyqpanda3.qcloud import QCloudService

    url, _ = sdk_capture_server
    info = QCloudService(KEY, url).backend("WK_C180_2").chip_info()
    backend = provider_module.OriginQProvider(api_key=KEY, url=url).get_backend(
        "WK_C180_2"
    )
    assert backend.available_qubits == sorted(info.available_qubits())
    assert backend.num_qubits == info.qubits_num()
    assert backend.config["topology"]["coupling_map"] == sorted(
        sorted(edge) for edge in info.get_chip_topology()
    )


#: Headers set by the HTTP library rather than by the protocol.
_TRANSPORT_HEADERS = {
    "host",
    "content-length",
    "connection",
    "user-agent",
    "accept-encoding",
}


def _protocol_request(entry: dict) -> dict:
    headers = {
        name.lower(): value
        for name, value in entry["headers"].items()
        if name.lower() not in _TRANSPORT_HEADERS
    }
    return {
        "method": entry["method"],
        "path": entry["path"],
        "headers": headers,
        "body": entry["body"],
    }


def test_cancel_request_and_answer_match_pyqpanda3(sdk_capture_server) -> None:
    """QuSpect cancels with the request ``QCloudJob.cancel()`` sends.

    Compared against the SDK talking to the stand-in server, not against
    the real cloud: path, method, protocol headers and body must be equal,
    and both must read the answer the same way.
    """
    from pyqpanda3.qcloud import QCloudJob, QCloudOptions, QCloudService

    url, recorded = sdk_capture_server
    service = QCloudService(KEY, url)
    prog = pq.QProg()
    prog << pq.RY(57, 0.3) << pq.CZ(57, 66) << pq.measure(57, 0) << pq.measure(66, 1)
    options = QCloudOptions()
    options.set_mapping(False)
    small = pq.QProg()
    small << pq.RY(0, 0.3) << pq.CZ(0, 1) << pq.measure(0, 0) << pq.measure(1, 1)
    sdk_answers = [
        service.backend("WK_C180_2").run(prog, 1000, options).cancel(),
        service.backend("WK_C180_2").run([prog, prog], 500, options).cancel(),
        service.backend("full_amplitude").run(small, 100).cancel(),
        QCloudJob("RECOVERED").cancel(),
        QCloudJob("REFUSE-1").cancel(),
        QCloudJob.cancel_task("REFUSE-2"),
    ]
    sdk = [r for r in recorded() if r["path"].endswith("cancelTask.json")]

    provider = provider_module.OriginQProvider(api_key=KEY, url=url)
    circuit = QuantumCircuit()
    circuit.ry(0.3, 57).cz(57, 66).measure([57, 66], [0, 1])
    small_circuit = QuantumCircuit()
    small_circuit.ry(0.3, 0).cz(0, 1).measure([0, 1], [0, 1])
    qpu = provider.get_backend("WK_C180_2")
    simulator = provider.get_backend("full_amplitude")
    direct = {"transpile": False, "provider_mapping": False}
    start = len(recorded())
    our_answers = [
        qpu.submit(circuit, shots=1000, **direct).cancel(),
        qpu.submit_request(
            ExecutionRequest.batch([circuit, circuit], shots=500, options=direct)
        ).cancel(),
        simulator.submit(small_circuit, shots=100).cancel(),
    ]
    for task_id in ("RECOVERED", "REFUSE-1", "REFUSE-2"):
        recovered = provider.retrieve_job(
            task_id, backend_name="WK_C180_2", measurement_maps=(((0,),),)
        )
        our_answers.append(recovered.cancel())
    ours = [r for r in recorded()[start:] if r["path"].endswith("cancelTask.json")]

    assert sdk_answers == our_answers == [True, True, True, True, False, False]
    assert len(sdk) == len(ours) == 6
    for theirs, mine in zip(sdk, ours, strict=True):
        assert _protocol_request(mine) == _protocol_request(theirs)
    # The comparison is not vacuous: this is what both sent.
    assert _protocol_request(ours[3]) == {
        "method": "POST",
        "path": "/oqcs/task/cancelTask.json",
        "headers": {
            "accept": "*/*",
            "authorization": f"oqcs_auth={KEY}",
            "content-type": "application/json",
            "origin-language": "en",
            "task_mode": "0",
        },
        "body": {"taskId": "RECOVERED"},
    }


def test_status_map_is_exactly_the_sdk_job_status_enum() -> None:
    """Fails when pyqpanda3 gains a state (for example a cancelled one)."""
    from pyqpanda3.qcloud import JobStatus as SdkStatus

    expected = {
        "WAITING": JobStatus.PENDING,
        "COMPUTING": JobStatus.RUNNING,
        "FINISHED": JobStatus.COMPLETED,
        "FAILED": JobStatus.FAILED,
        "QUEUING": JobStatus.QUEUED,
    }
    members = SdkStatus.__members__
    assert set(members) == set(expected)
    sdk_states = {str(member.value): expected[name] for name, member in members.items()}
    assert sdk_states == _STATUS_MAP


def test_task_state_outside_the_enum_sdk_fails_quspect_reports_unknown(
    sdk_capture_server,
) -> None:
    """What a status query after a cancel can show, as far as it is known.

    pyqpanda3 has no cancelled state: it turns every ``taskState`` outside
    its enum into FAILED ("unsupported taskState").  The real cloud reported
    35 after it accepted a cancel, which QuSpect maps to CANCELLED; any
    other value outside the enum stays UNKNOWN.
    """
    from pyqpanda3.qcloud import JobStatus as SdkStatus
    from pyqpanda3.qcloud import QCloudJob, QCloudService

    url, _ = sdk_capture_server
    QCloudService(KEY, url)
    provider = provider_module.OriginQProvider(api_key=KEY, url=url)

    def ours(task_id: str):
        return provider.retrieve_job(
            task_id, backend_name="WK_C180_2", measurement_maps=(((0,),),)
        )

    for state, mapped in _STATUS_MAP.items():
        sdk_status = QCloudJob(f"STATE-{state}").status()
        assert str(sdk_status.value) == state
        assert ours(f"STATE-{state}").status() == mapped

    result = QCloudJob("STATE-6").query()
    assert result.job_status() == SdkStatus.FAILED
    assert "unsupported taskState 6" in result.error_message()
    assert ours("STATE-6").status() == JobStatus.UNKNOWN

    result = QCloudJob("STATE-35").query()
    assert result.job_status() == SdkStatus.FAILED
    assert "unsupported taskState 35" in result.error_message()
    assert ours("STATE-35").status() == JobStatus.CANCELLED
    with pytest.raises(JobError, match="was cancelled") as info:
        ours("STATE-35").result(poll_interval=1e-3)
    assert info.value.error_type == "cancelled"

    assert ours("STATE-4").status() == JobStatus.FAILED
    with pytest.raises(JobError, match="failed"):
        ours("STATE-4").result(poll_interval=1e-3)
