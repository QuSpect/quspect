"""Tianyan backend checked against cqlib as an independent reference.

cqlib is a test-only dependency (``pip install -e '.[tianyan-tests]'``): its
QCIS parser and statevector simulator evaluate what QuSpect emits, and its
HTTP traffic is compared with QuSpect's client to catch protocol drift.
"""

from __future__ import annotations

import http.server
import json
import math
import re
import threading
import urllib.parse

import numpy as np
import pytest

pytest.importorskip("cqlib", reason="cqlib reference not installed")

from cqlib.circuits import Circuit as CqCircuit  # noqa: E402
from cqlib.simulator import StatevectorSimulator  # noqa: E402

from quspect import QuantumCircuit  # noqa: E402
from quspect.backends import ExecutionRequest  # noqa: E402
from quspect.backends.tianyan import TianyanClient, TianyanProvider  # noqa: E402
from quspect.backends.tianyan._backend import TianyanBackend  # noqa: E402
from quspect.backends.tianyan._translator import QCIS_BASIS, translate  # noqa: E402
from quspect.simulation import Simulator  # noqa: E402
from tests.backends._originq_fakes import (  # noqa: E402
    assert_same_distribution,
    quspect_exact_distribution,
)
from tests.backends._tianyan_fakes import (  # noqa: E402
    KEY,
    FakeTianyan,
    line_config,
    real_config,
)

URL = "https://tianyan.invalid"
_PREP = [(0.3, 1.1), (1.7, -0.4), (2.2, 0.9)]
_T = 0.37
_GATES = {
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
    "cz": lambda c: c.cz(2, 0),
}


def test_every_basis_gate_is_covered() -> None:
    assert set(_GATES) == set(QCIS_BASIS)


@pytest.mark.parametrize("name", sorted(_GATES))
def test_gate_semantics_match_quspect_through_cqlib(name: str) -> None:
    circuit = QuantumCircuit()
    for qubit, (ry, rz) in enumerate(_PREP):
        circuit.ry(ry, qubit).rz(rz, qubit)
    _GATES[name](circuit)
    expected = np.asarray(
        Simulator().run(circuit, shots=1, save_statevector=True).statevector
    )
    measured = circuit.copy()
    measured.measure([0], [0])
    qcis = "\n".join(
        line
        for line in translate(measured).text.splitlines()
        if not line.startswith("M ")
    )
    state = StatevectorSimulator(CqCircuit.load(qcis)).statevector()
    actual = np.zeros(8, complex)
    for key, value in state.items():
        actual[int(key, 2)] = complex(value)  # cqlib: Q0 is the rightmost character
    assert abs(abs(np.vdot(expected, actual)) - 1.0) < 1e-12


def _cases() -> dict[str, QuantumCircuit]:
    cases: dict[str, QuantumCircuit] = {}

    def add(name, build):
        c = QuantumCircuit(name=name)
        build(c)
        cases[name] = c

    add("asymmetric", lambda c: c.ry(1.0, 0).x(1).measure([0, 1], [0, 1]))
    add("permuted_clbits", lambda c: c.ry(1.0, 0).x(1).measure([0, 1], [1, 0]))
    add(
        "reverse_order", lambda c: c.ry(0.7, 0).x(1).measure([1], [1]).measure([0], [0])
    )
    add("sparse_fanout", lambda c: c.ry(0.6, 0).x(2).measure([0, 2, 0], [3, 0, 1]))
    add("partial", lambda c: c.h(0).cx(0, 1).ry(0.9, 2).cx(1, 2).measure([2], [0]))
    add(
        "four_qubit",
        lambda c: (
            c.h(0)
            .ry(0.3, 1)
            .ry(1.2, 2)
            .ry(2.0, 3)
            .cx(0, 3)
            .ccx(0, 1, 2)
            .cu3(0.4, 0.5, 0.6, 3, 1)
            .swap(0, 2)
            .crz(0.8, 2, 3)
            .barrier()
            .measure([0, 1, 2, 3], [2, 0, 3, 1])
        ),
    )
    add(
        "pauli_rotations",
        lambda c: (
            c.h(0)
            .ry(0.9, 1)
            .rxx(0.4, 0, 1)
            .ryy(-0.7, 1, 2)
            .rzz(1.1, 2, 0)
            .measure([0, 1, 2], [0, 1, 2])
        ),
    )
    return cases


CASES = _cases()


def _backend(cloud: FakeTianyan, name: str) -> TianyanBackend:
    client = TianyanClient(KEY, URL, transport=cloud, backoff=0)
    machine = next(m for m in cloud.machines() if m["code"] == name)
    return TianyanBackend(client, machine)


@pytest.mark.parametrize("machine", ["tianyan_sw", "tianyan176"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_end_to_end_matches_quspect(machine: str, name: str) -> None:
    """Routing on tianyan176 is checked against its real topology and
    disabled qubits by the fake, which rejects anything the server would."""
    circuit = CASES[name]
    cloud = FakeTianyan()
    result = _backend(cloud, machine).run(circuit, shots=2000, poll_interval=1e-3)
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))
    assert sum(result.counts.values()) == 2000


def test_routing_on_a_line_forces_swaps_and_still_decodes() -> None:
    circuit = QuantumCircuit()
    circuit.ry(0.6, 0).ry(1.9, 3).cx(0, 3).cx(3, 1).ry(0.4, 2).cx(2, 0)
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])
    cloud = FakeTianyan(chips={"line": line_config(4)})
    result = _backend(cloud, "line").run(circuit, shots=100, poll_interval=1e-3)
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))


def test_routing_keeps_per_qubit_measurements_terminal() -> None:
    """One measure instruction per qubit was refused after routing a SWAP."""
    circuit = QuantumCircuit()
    circuit.h(0).ry(0.4, 1).ry(1.1, 2)
    circuit.cx(2, 1).cx(0, 2).cx(0, 1).cx(2, 1)
    for qubit in range(3):
        circuit.measure([qubit], [2 - qubit])
    cloud = FakeTianyan(chips={"line": line_config(5)})
    result = _backend(cloud, "line").run(circuit, shots=100, poll_interval=1e-3)
    assert_same_distribution(result.probabilities, quspect_exact_distribution(circuit))


def test_batch_keeps_request_order_across_groups() -> None:
    cloud = FakeTianyan()
    names = sorted(CASES)
    request = ExecutionRequest.batch([CASES[n] for n in names], labels=names, shots=500)
    batch = _backend(cloud, "tianyan176").execute(request)
    assert batch.labels == tuple(names)
    for name in names:
        assert_same_distribution(
            batch.by_label(name).probabilities, quspect_exact_distribution(CASES[name])
        )


def test_parameter_sweep() -> None:
    from quspect.circuit.parameter import Parameter

    theta = Parameter("theta")
    template = QuantumCircuit()
    template.ry(theta, 0).cx(0, 1).measure([0, 1], [1, 0])
    values = [0.0, 0.8, math.pi]
    request = ExecutionRequest.sweep(template, [{"theta": v} for v in values], shots=10)
    batch = _backend(FakeTianyan(), "tianyan176").execute(request)
    for value, result in zip(values, batch, strict=True):
        p1 = math.sin(value / 2) ** 2
        expected = {k: v for k, v in {"00": 1 - p1, "11": p1}.items() if v > 1e-12}
        assert_same_distribution(result.probabilities, expected, tol=1e-9)


# --------------------------------------------- protocol contract vs cqlib


class _Recorder(http.server.BaseHTTPRequestHandler):
    log: list[dict] = []
    config: dict = {}

    def _reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode() if length else ""
        type_ = self.headers.get("Content-Type", "")
        body = (
            dict(urllib.parse.parse_qsl(raw))
            if "form" in type_
            else (json.loads(raw) if raw else None)
        )
        entry = {
            "method": self.command,
            "path": self.path,
            "body": body,
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }
        _Recorder.log.append(entry)
        return entry

    def do_GET(self):  # noqa: N802
        entry = self._record()
        if "download/config" in entry["path"]:
            self._reply({"code": 0, "data": _Recorder.config})
        else:
            self._reply(
                {
                    "code": 0,
                    "data": [
                        {
                            "code": "tianyan176",
                            "id": "1",
                            "label": "1",
                            "isToll": 1,
                            "status": 0,
                        }
                    ],
                }
            )

    def do_POST(self):  # noqa: N802
        entry = self._record()
        if entry["path"].endswith("opnId"):
            self._reply({"code": 0, "data": {"access_token": "T0K"}})
        else:
            n = len((entry["body"] or {}).get("inputCode") or [1])
            self._reply({"code": 0, "data": {"query_ids": [str(i) for i in range(n)]}})

    def log_message(self, *args):
        pass


@pytest.fixture
def capture_server(monkeypatch):
    _Recorder.log = []
    _Recorder.config = real_config()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _cqlib_platform(url: str):
    from cqlib import TianYanPlatform

    scheme, host = url.split("://")

    class Local(TianYanPlatform):
        SCHEME = scheme
        DOMAIN = host

    return Local(login_key=KEY, machine_name="tianyan176")


def test_requests_match_cqlib(capture_server) -> None:
    """Detect protocol drift: QuSpect must send what cqlib sends."""
    qcis = "X Q57\nM Q57\nM Q62"
    platform = _cqlib_platform(capture_server)
    platform.query_quantum_computer_list()
    platform.submit_job(circuit=qcis, num_shots=1000)
    platform.submit_job(circuit=[qcis, qcis], num_shots=500)
    theirs = list(_Recorder.log)

    _Recorder.log = []
    client = TianyanClient(KEY, capture_server)
    client.list_machines()
    client.submit([qcis], machine="tianyan176", shots=1000)
    client.submit([qcis, qcis], machine="tianyan176", shots=500)
    ours = list(_Recorder.log)

    assert [(e["method"], e["path"]) for e in ours] == [
        (e["method"], e["path"]) for e in theirs
    ]
    for mine, ref in zip(ours, theirs, strict=True):
        assert mine["body"] == ref["body"], mine["path"]
        for header in ("authorization", "basictoken"):
            assert mine["headers"].get(header) == ref["headers"].get(header)


def test_config_parsing_matches_cqlib_topology(capture_server) -> None:
    platform = _cqlib_platform(capture_server)
    cfg = platform.download_config()
    backend = TianyanProvider(KEY, url=capture_server).get_backend("tianyan176")
    disabled = set(cfg["disabledCouplers"].split(","))
    expected = sorted(
        sorted(int(q[1:]) for q in pair)
        for g, pair in cfg["overview"]["coupler_map"].items()
        if g not in disabled and not {"Q22", "Q24"} & set(pair)
    )
    assert backend.config["topology"]["coupling_map"] == expected
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", backend.config["calibration_time"]
    )
