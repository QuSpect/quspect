"""Tianyan backend: QCIS, result decoding, HTTP client, placement (pure Python).

Runs offline without cqlib.  End-to-end and protocol checks against cqlib as
an independent reference are in ``test_tianyan_reference.py``.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.parse

import pytest

from quspect import QuantumCircuit
from quspect.backends import ExecutionRequest, JobReference, JobStatus
from quspect.backends._exceptions import (
    AuthenticationError,
    BackendError,
    JobTimeoutError,
    TranslationError,
)
from quspect.backends.tianyan import TianyanClient, TianyanJob, TianyanProvider
from quspect.backends.tianyan._backend import TianyanBackend, parse_chip
from quspect.backends.tianyan._translator import angle, decode_result, translate
from tests.backends._tianyan_fakes import KEY, FakeTianyan, line_config, real_config

URL = "https://tianyan.invalid"

# ------------------------------------------------------------------ QCIS


@pytest.mark.parametrize(
    ("build", "line"),
    [
        (lambda c: c.h(3), "H Q3"),
        (lambda c: c.sdg(3), "SD Q3"),
        (lambda c: c.tdg(3), "TD Q3"),
        (lambda c: c.sx(3), "X2P Q3"),
        (lambda c: c.sxdg(3), "X2M Q3"),
        (lambda c: c.rx(0.5, 3), "RX Q3 0.5"),
        (lambda c: c.rz(-1.25, 3), "RZ Q3 -1.25"),
        (lambda c: c.cz(3, 1), "CZ Q3 Q1"),
    ],
)
def test_gate_spelling(build, line) -> None:
    circuit = QuantumCircuit()
    build(circuit)
    circuit.measure([3], [0])
    assert translate(circuit).text.splitlines()[0] == line


def test_angles_round_trip_exactly_without_exponents() -> None:
    for value in (1e-9, -3.0e5, math.pi, 0.1 + 0.2, 1e-300):
        text = angle(value)
        assert "e" not in text.lower()
        assert float(text) == value


def test_identity_is_dropped_and_barriers_kept() -> None:
    circuit = QuantumCircuit()
    circuit.id(0).h(1).barrier().measure([0, 1], [0, 1])
    assert translate(circuit).text.splitlines() == ["H Q1", "B Q0 Q1", "M Q0", "M Q1"]


def test_measurements_once_ascending_with_fanout() -> None:
    circuit = QuantumCircuit()
    circuit.x(4).measure([4, 1, 4], [0, 2, 3])
    program = translate(circuit)
    assert program.text.splitlines()[-2:] == ["M Q1", "M Q4"]
    assert program.measurement.measured_qubits == (1, 4)
    assert program.measurement.clbits == ((2,), (0, 3))


def test_qubit_map_relabels_wires() -> None:
    circuit = QuantumCircuit()
    circuit.cz(0, 1).measure([0, 1], [0, 1])
    program = translate(circuit, qubit_map=[62, 57])
    assert program.text.splitlines() == ["CZ Q62 Q57", "M Q57", "M Q62"]
    assert program.measurement.clbits == ((1,), (0,))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda c: c.h(0).measure([0], [0]).x(0), "after being measured"),
        (lambda c: c.x(0).measure([0], [0]).measure([1], [0]), "more than once"),
        (lambda c: c.h(0), "no measurement"),
        (lambda c: c.cx(0, 1).measure([0], [0]), "not a QCIS instruction"),
    ],
)
def test_translation_errors(build, message) -> None:
    circuit = QuantumCircuit()
    build(circuit)
    with pytest.raises(TranslationError, match=message):
        translate(circuit)


# ------------------------------------------------------------ decoding


def test_decode_uses_reported_order_and_reversed_probability_keys() -> None:
    # Recorded on tianyan_sw: X Q1; M Q1; M Q0 -> order [1, 0], rows [1, 0],
    # probability {"01": 1.0} (first measured qubit rightmost).
    result = {"resultStatus": [[1, 0], [1, 0], [1, 0]], "probability": {"01": 1.0}}
    counts, probs = decode_result(result, ((0,), (1,)), num_clbits=2)
    assert counts == {"01": 2}  # clbit 1 (qubit 1) set
    assert probs == {"01": 1.0}


def test_decode_fanout_and_unmeasured_clbits() -> None:
    result = {
        "resultStatus": [[0, 2], [1, 0], [1, 1]],
        "probability": {"01": 0.5, "11": 0.5},
    }
    counts, probs = decode_result(result, ((3, 1), (0,)), num_clbits=4)
    assert counts == {"0101": 1, "1101": 1}
    assert probs == {"0101": 0.5, "1101": 0.5}


def test_decode_rejects_a_different_measured_set() -> None:
    from quspect.backends.tianyan._translator import MeasurementMap

    measurement = MeasurementMap((0, 1), ((0,), (1,)), 2)
    with pytest.raises(TranslationError, match="submitted"):
        decode_result({"resultStatus": [[0, 2], [1, 0]]}, measurement)


def test_probability_absent_falls_back_to_counts() -> None:
    counts, probs = decode_result(
        {"resultStatus": [[0], [1], [0]]}, ((0,),), num_clbits=1
    )
    assert counts == {"0": 1, "1": 1} and probs is None


# ----------------------------------------------------------- HTTP client


def test_login_form_and_bearer_headers() -> None:
    cloud = FakeTianyan(evaluate=False)
    seen = []

    def spy(method, url, headers, body):
        seen.append((method, url, dict(headers), body))
        return cloud(method, url, headers, body)

    client = TianyanClient(KEY, URL, transport=spy)
    client.list_machines()
    (m1, u1, h1, b1), (m2, u2, h2, _) = seen
    assert (m1, u1) == ("POST", URL + "/qccp-auth/oauth2/sdk/opnId")
    assert dict(urllib.parse.parse_qsl(b1.decode())) == {
        "grant_type": "openId",
        "openId": KEY,
        "account_type": "member",
    }
    assert (m2, u2) == ("GET", URL + "/qccp-quantum/sdk/quantumComputer/list")
    assert h2["Authorization"] == "Bearer token-1" and h2["basicToken"] == "token-1"
    assert KEY not in repr(client) and "token" not in repr(client)


def test_expired_token_in_json_body_triggers_relogin() -> None:
    cloud = FakeTianyan(evaluate=False)
    client = TianyanClient(KEY, URL, transport=cloud)
    client.list_machines()
    cloud.expire_token_once = True
    client.list_machines()
    assert cloud.logins == 2


def test_gateway_timeouts_are_retried(caplog) -> None:
    cloud = FakeTianyan(evaluate=False)
    cloud.fail_next = {"opnId": 2, "list": 2, "temporary/save": 1}
    client = TianyanClient(KEY, URL, transport=cloud, backoff=0)
    assert client.list_machines()
    with caplog.at_level(logging.WARNING):
        ids = client.submit(["X Q0\nM Q0"], machine="tianyan_sw", shots=10)
    assert len(ids) == 1
    assert "may still have created a task" in caplog.text


def test_submit_retries_can_be_disabled() -> None:
    cloud = FakeTianyan(evaluate=False)
    cloud.fail_next = {"temporary/save": 1}
    client = TianyanClient(KEY, URL, transport=cloud, backoff=0, submit_retries=0)
    with pytest.raises(BackendError, match="504"):
        client.submit(["X Q0\nM Q0"], machine="tianyan_sw", shots=10)
    assert cloud.submissions == []


def test_provider_errors_are_readable() -> None:
    client = TianyanClient(KEY, URL, transport=FakeTianyan(evaluate=False), backoff=0)
    with pytest.raises(BackendError, match="I gate"):
        client.submit(["I Q0 0\nM Q0"], machine="tianyan_sw", shots=10)
    with pytest.raises(AuthenticationError):
        TianyanClient("wrong", URL, transport=FakeTianyan(evaluate=False)).login()


# ------------------------------------------------------------- chip config


def test_real_config_excludes_disabled_qubits_and_couplers() -> None:
    chip = parse_chip(real_config())
    assert chip["num_qubits"] == 66
    assert {22, 24}.isdisjoint(chip["available_qubits"])
    assert len(chip["available_qubits"]) == 64
    assert len(chip["coupling_map"]) == 95  # 110 couplers, 15 disabled
    assert [57, 62] in chip["coupling_map"]
    # values are matched by qubit_used name, not list position
    assert chip["edge_fidelity"][(57, 62)] == pytest.approx(1 - 0.0095)
    assert all(q not in chip["qubit_fidelity"] for q in (22, 24))


def _backend(cloud: FakeTianyan, name: str) -> TianyanBackend:
    client = TianyanClient(KEY, URL, transport=cloud, backoff=0)
    machine = next(m for m in cloud.machines() if m["code"] == name)
    return TianyanBackend(client, machine)


def test_placement_prefers_calibrated_enabled_region() -> None:
    backend = _backend(FakeTianyan(evaluate=False), "tianyan176")
    chip = backend._chip()
    median = sorted(chip["qubit_fidelity"].values())[len(chip["qubit_fidelity"]) // 2]
    for size in (1, 2, 4, 6):
        region = backend.placement_qubits(size)
        assert len(region) == size and {22, 24}.isdisjoint(region)
        assert all(chip["qubit_fidelity"][q] >= median for q in region)


def test_transpile_false_rejects_uncoupled_cz_and_disabled_qubits() -> None:
    backend = _backend(FakeTianyan(evaluate=False), "tianyan176")
    c = QuantumCircuit()
    c.cz(4, 5).measure([4, 5], [0, 1])  # Q4-Q5 is not a coupler
    with pytest.raises(TranslationError, match="uncoupled"):
        backend.submit(c, transpile=False)
    c = QuantumCircuit()
    c.x(22).measure([22], [0])
    with pytest.raises(TranslationError, match="disabled"):
        backend.submit(c, transpile=False)


def test_option_validation() -> None:
    backend = _backend(FakeTianyan(evaluate=False), "tianyan176")
    c = QuantumCircuit()
    c.h(0).measure([0], [0])
    with pytest.raises(ValueError, match="require transpile=True"):
        backend.submit(c, transpile=False, physical_qubits=[57])
    with pytest.raises(ValueError, match="disabled or unknown"):
        backend.submit(c, physical_qubits=[22])
    with pytest.raises(ValueError, match="shots"):
        backend.submit(c, shots=100_001)
    with pytest.raises(ValueError, match="QPUs only"):
        _backend(FakeTianyan(evaluate=False), "tianyan_sw").submit(
            c, physical_qubits=[0]
        )


def test_provider_lists_machines_and_rejects_single_amplitude() -> None:
    provider = TianyanProvider(KEY, url=URL, transport=FakeTianyan(evaluate=False))
    backends = provider.list_backends()
    assert (
        backends["tianyan176"] == "running" and backends["tianyan24"] == "calibrating"
    )
    assert "tianyan_sa" not in backends
    with pytest.raises(ValueError, match="single amplitudes"):
        provider.get_backend("tianyan_sa")


# ------------------------------------------------------------- batching


def test_qpu_batches_group_by_measured_qubits() -> None:
    cloud = FakeTianyan(evaluate=False, chips={"line": line_config(4)})
    backend = _backend(cloud, "line")

    def circuit(measured):
        c = QuantumCircuit()
        c.h(0)
        for clbit, qubit in enumerate(measured):
            c.measure([qubit], [clbit])
        return c

    circuits = [circuit([0, 1]), circuit([0, 1]), circuit([0]), circuit([0, 1])]
    request = ExecutionRequest.batch(
        circuits, shots=10, options={"transpile": False}, labels=["a", "b", "c", "d"]
    )
    job = backend.submit_request(request)
    sizes = [len(s["inputCode"]) for s in cloud.submissions]
    assert sizes == [2, 1, 1]  # [a, b] share M Q0 M Q1; c and d break the run
    batch = job.result(poll_interval=1e-3)
    assert batch.labels == ("a", "b", "c", "d")


# ------------------------------------------------------------- job lifecycle


def test_job_polls_until_results_and_recovers_from_reference() -> None:
    cloud = FakeTianyan(evaluate=False, chips={"line": line_config(3)})
    cloud.pending_polls = 2
    backend = _backend(cloud, "line")
    c = QuantumCircuit()
    c.h(0).measure([0, 1], [1, 0])
    request = ExecutionRequest.batch([c, c], shots=8, options={"transpile": False})
    job = backend.submit_request(request)
    assert job.status() == JobStatus.RUNNING
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0)
    batch = job.result(poll_interval=1e-3)
    assert [r.num_shots for r in batch] == [8, 8]
    assert job.cancel() is False

    restored = JobReference.from_dict(json.loads(json.dumps(job.reference.to_dict())))
    provider = TianyanProvider(KEY, url=URL, transport=cloud)
    again = provider.retrieve_job(restored)
    assert isinstance(again, TianyanJob)
    assert again.result(poll_interval=1e-3).get_counts() == batch.get_counts()
    other = TianyanProvider(KEY, url="https://other.invalid", transport=cloud)
    with pytest.raises(ValueError, match="endpoint"):
        other.retrieve_job(restored)
