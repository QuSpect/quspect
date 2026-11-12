"""Offline tests for the Quafu-SQC backend.

A fake HTTP transport stands in for https://quafu-sqc.baqis.ac.cn. It
executes submitted QASM on the QuSpect simulator and answers in Quafu's
wire format (little-endian keys over measured clbits only), so these tests
check the full compile -> submit -> decode round trip.
"""

from __future__ import annotations

import json
import random
import urllib.parse

import pytest

from quspect import QuantumCircuit
from quspect.backends import CompositeJob, ExecutionBatch, ExecutionRequest, JobStatus
from quspect.backends._compile import estimated_log_fidelity
from quspect.backends._exceptions import (
    AuthenticationError,
    BackendNotFoundError,
    JobError,
    JobTimeoutError,
    TranslationError,
)
from quspect.backends._execution import CircuitExperiment, JobReference
from quspect.backends.quafu import QuafuProvider
from quspect.backends.quafu._backend import QuafuBackend, round_shots
from quspect.backends.quafu._client import resolve_token
from quspect.backends.quafu._job import (
    convert_result,
    decode_bitstring,
    measured_clbits,
)
from quspect.ir import qasm2_loads
from quspect.simulation import Simulator
from tests.backends._measurement_cases import (
    FULL_SPELLINGS,
    measurements_are_last,
    random_measured_circuits,
    used_after_measurement,
)
from tests.backends._originq_fakes import (
    assert_same_distribution,
    quspect_exact_distribution,
)

TOKEN = "secret-token"
URL = "https://quafu.example"

# 0-1-2   2-3 disabled (fidelity 0), 6 has an uncalibrated qubit.
#   |     |
#   4-----5-3
#         |
#         6
CHIP = {
    "size": [3, 3],
    "basis_gates": ["h", "rx", "ry", "rz", "cz"],
    "priority_qubits": [[1], [1, 4], [1, 4, 0], [0, 2, 5], [0, 1, 2, 4, 5]],
    "calibration_time": "2026-11-06 10:00:00",
    "qubits_info": {
        f"Q{i}": {"label": f"Q{i}", "index": i, "fidelity": 0.0 if i == 6 else 0.999}
        for i in range(7)
    },
    "couplers_info": {
        f"C{i}": {
            "label": f"C{i}",
            "index": i,
            "qubits_index": list(pair),
            "fidelity": f,
        }
        for i, (pair, f) in enumerate(
            [
                ((0, 1), 0.98),
                ((1, 2), 0.97),
                ((2, 3), 0.0),
                ((1, 4), 0.99),
                ((4, 5), 0.96),
                ((3, 5), 0.95),
                ((5, 6), 0.90),
            ]
        )
    },
    "global_info": {},
}


def _to_quafu_key(bitstring: str, clbits: tuple[int, ...]) -> str:
    return "".join(bitstring[c] for c in reversed(clbits))


class FakeQuafu:
    """In-memory Quafu-SQC service."""

    def __init__(self, *, pending_polls: int = 0, fail_run_after: int | None = None):
        self.pending_polls = pending_polls
        self.fail_run_after = fail_run_after
        self.tasks: dict[int, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.next_tid = 1000

    def __call__(self, method, url, headers, body, timeout):
        parts = urllib.parse.urlsplit(url)
        path, query = parts.path, dict(urllib.parse.parse_qsl(parts.query))
        self.calls.append((method, path))
        if path.startswith("/task/backendtest/"):
            assert "token" not in headers
            if path == "/task/backendtest/Fake1":
                return 200, json.dumps(CHIP).encode()
            return 200, b'{"detail": "File not found"}'
        if headers.get("token") != TOKEN:
            return 401, b'{"detail": "invalid token"}'
        if path == "/task/verify":
            return 200, b'["2026-12-06 11:38:31", 0]'
        if path == "/task/status/0":
            return 200, b'{"Fake": 3, "Down": "Offline"}'
        if method == "POST" and path == "/task/run/":
            if (
                self.fail_run_after is not None
                and len(self.tasks) >= self.fail_run_after
            ):
                return 500, b'{"detail": "boom"}'
            assert headers["Content-Type"] == "application/json"
            payload = json.loads(body)
            tid = self.next_tid
            self.next_tid += 1
            self.tasks[tid] = {
                "query": query,
                "body": payload,
                "polls": 0,
                "cancelled": False,
            }
            return 200, str(tid).encode()
        kind, _, tid_text = path.removeprefix("/task/").partition("/")
        tid = int(tid_text)
        task = self.tasks.get(tid)
        if kind == "cancel":
            if task:
                task["cancelled"] = True
            return 200, b"null"
        if task is None:
            return 200, (
                b"{}" if kind == "result" else f'"task {tid} not found!"'.encode()
            )
        done = task["polls"] >= self.pending_polls
        if kind == "status":
            if task["cancelled"]:
                return 200, b'"Canceled"'
            return 200, (b'"Finished"' if done else b'"Transpiled"')
        if kind == "result":
            task["polls"] += 1
            if not done or task["cancelled"]:
                return 200, b"{}"
            return 200, json.dumps(self._execute(tid, task)).encode()
        return 404, b'{"detail": "Not Found"}'

    def _execute(self, tid: int, task: dict) -> dict:
        qasm = task["body"]["circuit"]
        options = task["body"]["options"]
        base = {
            "tid": tid,
            "chip": task["query"]["chip"],
            "circuit": qasm,
            "transpiled": qasm,
            "qlisp": "[]",
            "created": "2026-11-06 12:00:00",
            "finished": "2026-11-06 12:00:01",
        }
        if "fail" in task["query"]["name"]:
            return {
                **base,
                "status": None,
                "error": "Transpiler: TranspileError!!",
                "count": None,
            }
        circuit = qasm2_loads(qasm)
        shots = int(task["query"]["shots"])
        result = Simulator(seed=7).run(circuit, shots=shots)
        clbits = measured_clbits(qasm)
        counts: dict[str, int] = {}
        for bitstring, count in result.counts.items():
            key = _to_quafu_key(bitstring, clbits)
            counts[key] = counts.get(key, 0) + count
        corrected = {}
        if options.get("correct"):
            corrected = {k: v / shots for k, v in counts.items()}
            corrected[next(iter(corrected))] -= 0.01
        return {
            **base,
            "status": "Finished",
            "error": "",
            "count": counts,
            "corrected": corrected,
        }


@pytest.fixture
def fake() -> FakeQuafu:
    return FakeQuafu()


@pytest.fixture
def provider(fake: FakeQuafu) -> QuafuProvider:
    return QuafuProvider(TOKEN, URL, transport=fake)


@pytest.fixture
def backend(provider: QuafuProvider):
    return provider.get_backend("Fake")


def _ghz(n: int, name: str = "ghz") -> QuantumCircuit:
    circuit = QuantumCircuit(name=name)
    circuit.h(0)
    for i in range(n - 1):
        circuit.cx(i, i + 1)
    circuit.measure(list(range(n)), list(range(n)))
    return circuit


# ---------------------------------------------------------------------------
# Token and client
# ---------------------------------------------------------------------------


def test_token_precedence(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("quspect.backends.quafu._client.TOKEN_FILE", tmp_path / "api")
    monkeypatch.delenv("QUAFU_API_TOKEN", raising=False)
    monkeypatch.delenv("QPU_API_TOKEN", raising=False)
    with pytest.raises(AuthenticationError):
        resolve_token()
    (tmp_path / "api").write_text('{"token": "from-file"}')
    assert resolve_token() == "from-file"
    monkeypatch.setenv("QPU_API_TOKEN", "sdk-env")
    assert resolve_token() == "sdk-env"
    monkeypatch.setenv("QUAFU_API_TOKEN", "env")
    assert resolve_token() == "env"
    assert resolve_token("explicit") == "explicit"


def test_bad_token_raises_authentication_error(fake: FakeQuafu) -> None:
    with pytest.raises(AuthenticationError):
        QuafuProvider("wrong", URL, transport=fake)


def test_token_is_not_exposed(provider: QuafuProvider) -> None:
    assert TOKEN not in repr(provider)
    assert TOKEN not in repr(provider.client)


def test_listing_and_unknown_backend(provider: QuafuProvider) -> None:
    assert provider.token_expiry == "2026-12-06 11:38:31"
    assert provider.list_backends() == {"Fake": 3, "Down": "Offline"}
    with pytest.raises(BackendNotFoundError):
        provider.get_backend("Nope")


# ---------------------------------------------------------------------------
# Result decoding
# ---------------------------------------------------------------------------


def test_decode_bitstring_reverses_and_expands() -> None:
    # Quafu: rightmost char = lowest measured clbit; QuSpect: char i = clbit i.
    assert decode_bitstring("001", (0, 1, 2), 3) == "100"
    assert decode_bitstring("10", (1, 3), 4) == "0001"
    with pytest.raises(ValueError):
        decode_bitstring("1", (0, 1), 2)


def test_measured_clbits_from_qasm() -> None:
    qasm = "creg c[4];\nmeasure q[25] -> c[3];\nmeasure q[19] -> c[1];\n"
    assert measured_clbits(qasm) == (1, 3)


def test_convert_failed_payload() -> None:
    result = convert_result(
        {
            "tid": 1,
            "status": None,
            "error": "Coupler error",
            "count": None,
            "circuit": "",
        }
    )
    assert not result.success
    assert result.error == "Coupler error"
    assert result.counts == {}


def test_convert_corrected_payload_clips_quasi_probabilities() -> None:
    payload = {
        "tid": 1,
        "status": "Finished",
        "error": "",
        "circuit": "creg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        "count": {"00": 500, "01": 24, "11": 500},
        "corrected": {"00": 0.5, "01": -0.01, "10": 0.0, "11": 0.51},
    }
    result = convert_result(payload, shape=(2, 2))
    assert result.counts == {"00": 500, "10": 24, "11": 500}
    assert result.metadata["readout_correction_applied"]
    assert result.metadata["quasi_probabilities"]["10"] == pytest.approx(-0.01)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in result.probabilities.values())


def test_round_shots() -> None:
    assert round_shots(1) == 1024
    assert round_shots(1024) == 1024
    assert round_shots(1025) == 2048
    with pytest.raises(ValueError):
        round_shots(0)
    with pytest.raises(TypeError):
        round_shots(True)


# ---------------------------------------------------------------------------
# Device model and compilation
# ---------------------------------------------------------------------------


def test_config_filters_uncalibrated_couplers_and_qubits(backend) -> None:
    assert backend.num_qubits == 7
    assert backend.native_gate_set == {"h", "rx", "ry", "rz", "cz"}
    assert backend.coupling_map == [(0, 1), (1, 2), (1, 4), (3, 5), (4, 5)]
    assert 6 not in backend.config["usable_qubits"]
    assert backend.calibration_time == "2026-11-06 10:00:00"


def test_default_pool_prefers_connected_priority_subsets(backend) -> None:
    assert backend.default_physical_qubits(1) == [1]
    assert backend.default_physical_qubits(3) == [1, 4, 0]
    # [0, 2, 5] is disconnected, so the 5-qubit priority subset is chosen.
    assert backend.default_physical_qubits(4) == [0, 1, 2, 4, 5]
    assert backend.default_physical_qubits(6) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(TranslationError):
        backend.default_physical_qubits(7)


# ---------------------------------------------------------------------------
# Placement by calibration when no suggested group fits
# ---------------------------------------------------------------------------


def _chip(
    couplers: dict[tuple[int, int], float],
    *,
    priority: list[list[int]] | None = None,
    readout: dict[int, float] | None = None,
) -> dict:
    """A chip with the given CZ fidelities and (optionally) readout fidelities."""
    count = max(max(pair) for pair in couplers) + 1
    qubits = {}
    for i in range(count):
        qubits[f"Q{i}"] = {"label": f"Q{i}", "index": i, "fidelity": 0.999}
        if readout is not None:
            qubits[f"Q{i}"]["readout g_fidelity"] = readout.get(i, 0.0)
            qubits[f"Q{i}"]["readout e_fidelity"] = readout.get(i, 0.0)
    return {
        **CHIP,
        "size": [count, 1],
        "priority_qubits": priority or [],
        "qubits_info": qubits,
        "couplers_info": {
            f"C{i}": {
                "label": f"C{i}",
                "index": i,
                "qubits_index": list(pair),
                "fidelity": fidelity,
            }
            for i, (pair, fidelity) in enumerate(couplers.items())
        },
    }


def _line(fidelities: list[float]) -> dict[tuple[int, int], float]:
    return {(i, i + 1): fidelity for i, fidelity in enumerate(fidelities)}


def _grid(rows: int, cols: int, fidelity) -> dict[tuple[int, int], float]:
    couplers = {}
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if col + 1 < cols:
                couplers[(index, index + 1)] = fidelity(index, index + 1)
            if row + 1 < rows:
                couplers[(index, index + cols)] = fidelity(index, index + cols)
    return couplers


def _random_circuits(count: int, seed: int) -> list[QuantumCircuit]:
    rng = random.Random(seed)
    circuits = []
    for index in range(count):
        n = rng.randint(3, 6)
        circuit = QuantumCircuit(name=f"random{index}")
        for qubit in range(n):
            circuit.h(qubit)
        for _ in range(rng.randint(4, 20)):
            if rng.random() < 0.4:
                circuit.ry(round(rng.uniform(-3, 3), 3), rng.randrange(n))
            else:
                circuit.cx(*rng.sample(range(n), 2))
        circuit.measure(list(range(n)), list(range(n)))
        circuits.append(circuit)
    return circuits


def _used_qubits(circuit: QuantumCircuit) -> set[int]:
    return {q for instruction in circuit.instructions for q in instruction.qubits}


def _cz_count(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit.instructions if i.is_gate and i.gate.name == "cz")


def _estimate(backend, circuit: QuantumCircuit) -> float:
    cfg = backend.config
    return estimated_log_fidelity(
        circuit,
        edge_fidelity=cfg["coupler_fidelity"],
        readout_fidelity=cfg["readout_fidelity"],
        gate_fidelity=cfg["qubit_fidelity"],
    )


def test_placement_past_the_suggested_groups_follows_calibration(provider, fake):
    """Regression: with no fitting group, circuits landed on the lowest indices.

    Qubits 0-3 are joined by poor (but calibrated) couplers, 4-7 by good
    ones; Quafu only suggests a two-qubit group.
    """
    chip = _chip(
        _line([0.60, 0.05, 0.60, 0.90, 0.995, 0.995, 0.995]), priority=[[5, 6]]
    )
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)

    # The pool methods keep their meaning: Quafu's group, else every usable qubit.
    assert backend.default_physical_qubits(2) == [5, 6]
    assert backend.default_physical_qubits(3) == list(range(8))
    with pytest.raises(TranslationError, match="no connected group"):
        backend.default_physical_qubits(9)

    backend.submit(_ghz(2))
    backend.submit(_ghz(3))
    backend.submit(_ghz(4))
    sent = [qasm2_loads(task["body"]["circuit"]) for task in fake.tasks.values()]
    assert _used_qubits(sent[0]) == {5, 6}  # Quafu's suggestion
    assert _used_qubits(sent[1]) == {4, 5, 6}
    assert _used_qubits(sent[2]) == {4, 5, 6, 7}


def test_placement_follows_readout_fidelity(provider):
    """Couplers alike, qubits 0-2 read poorly: the circuit moves off them."""
    readout = {i: (0.60 if i < 3 else 0.97) for i in range(8)}
    chip = _chip(_line([0.98] * 7), readout=readout)
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)
    assert backend.config["readout_fidelity"] == readout

    physical, layout = backend._transpile(
        _ghz(3), initial_layout=None, physical_qubits=None
    )
    assert _used_qubits(physical) == set(layout) == {3, 4, 5}


def test_unmeasured_readout_counts_as_the_worst_published(provider):
    """Quafu writes 0 for a readout it did not measure; that is not "perfect"."""
    readout = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.90, 4: 0.99, 5: 0.99, 6: 0.99}
    chip = _chip(_line([0.98] * 6), readout=readout)
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)
    assert backend.config["readout_fidelity"] == {3: 0.90, 4: 0.99, 5: 0.99, 6: 0.99}
    assert set(backend.config["usable_qubits"]) == set(range(7))

    physical, _ = backend._transpile(_ghz(3), initial_layout=None, physical_qubits=None)
    assert _used_qubits(physical) == {4, 5, 6}


def test_flat_calibration_never_costs_more_two_qubit_gates(provider):
    """A region of exactly the circuit's size has no routing room; on a chip
    whose calibration is flat it must not replace the whole-chip placement."""
    chip = _chip(_grid(4, 4, lambda a, b: 0.98))
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)
    whole = list(range(16))
    for circuit in _random_circuits(25, seed=3):
        placed, _ = backend._transpile(
            circuit, initial_layout=None, physical_qubits=None
        )
        baseline, _ = backend._transpile(
            circuit, initial_layout=None, physical_qubits=whole
        )
        assert _cz_count(placed) <= _cz_count(baseline), circuit.name


def test_placement_is_never_estimated_below_the_whole_chip(provider):
    rng = random.Random(3)
    values: dict[tuple[int, int], float] = {}
    chip = _chip(
        _grid(5, 5, lambda a, b: values.setdefault((a, b), rng.uniform(0.9, 0.995))),
        readout={i: rng.uniform(0.9, 0.99) for i in range(25)},
    )
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)
    whole = list(range(25))
    better = 0
    for circuit in _random_circuits(25, seed=4):
        placed, _ = backend._transpile(
            circuit, initial_layout=None, physical_qubits=None
        )
        baseline, _ = backend._transpile(
            circuit, initial_layout=None, physical_qubits=whole
        )
        assert _estimate(backend, placed) >= _estimate(backend, baseline) - 1e-9
        better += _estimate(backend, placed) > _estimate(backend, baseline) + 1e-9
    assert better >= 20


def test_suggested_groups_and_explicit_pools_skip_calibration_placement(
    provider, monkeypatch
):
    """Only the no-group fallback compiles on calibration-chosen regions."""
    import quspect.backends.quafu._backend as module

    def unexpected(*args, **kwargs):
        raise AssertionError("placement_regions must not be consulted here")

    monkeypatch.setattr(module, "placement_regions", unexpected)
    chip = _chip(
        _line([0.60, 0.60, 0.60, 0.90, 0.995, 0.995, 0.995]), priority=[[0, 1, 2]]
    )
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)
    circuit = _ghz(3)

    suggested, layout = backend._transpile(
        circuit, initial_layout=None, physical_qubits=None
    )
    assert set(layout) == {0, 1, 2}  # Quafu's group, poor as its couplers are
    explicit, _ = backend._transpile(
        circuit, initial_layout=None, physical_qubits=[0, 1, 2]
    )
    assert suggested.to_json() == explicit.to_json()

    _, layout = backend._transpile(
        circuit, initial_layout=[2, 1, 0], physical_qubits=None
    )
    assert layout == [2, 1, 0]
    _, layout = backend._transpile(
        _ghz(4), initial_layout=None, physical_qubits=[0, 1, 2, 3]
    )
    assert set(layout) == {0, 1, 2, 3}
    with pytest.raises(AssertionError, match="must not be consulted"):
        backend._transpile(_ghz(4), initial_layout=None, physical_qubits=None)


def test_calibration_placed_circuits_keep_their_result(provider, fake):
    """X/CX circuits have one outcome; it must survive placement and routing."""
    chip = _chip(
        _grid(4, 4, lambda a, b: 0.60 if max(a, b) < 8 else 0.99),
        readout={i: (0.70 if i < 8 else 0.98) for i in range(16)},
    )
    backend = QuafuBackend("Fake", provider.client, chip_info=chip)
    rng = random.Random(9)
    for index in range(12):
        n = rng.randint(3, 6)
        circuit = QuantumCircuit(name=f"basis{index}")
        for qubit in range(n):
            if rng.random() < 0.5:
                circuit.x(qubit)
        circuit.x(0)
        for _ in range(rng.randint(4, 12)):
            circuit.cx(*rng.sample(range(n), 2))
        circuit.measure(list(range(n)), list(range(n)))
        expected = Simulator(seed=1).run(circuit, shots=16)
        result = backend.run(circuit, shots=1024, poll_interval=0.001)
        assert set(result.counts) == set(expected.counts), circuit.name
        sent = qasm2_loads(fake.tasks[max(fake.tasks)]["body"]["circuit"])
        assert min(_used_qubits(sent)) >= 8, circuit.name


def test_transpiled_qasm_is_native_and_on_usable_couplers(backend, fake) -> None:
    circuit = _ghz(5)
    circuit.ccx(0, 2, 4)
    backend.run(circuit)
    body = next(iter(fake.tasks.values()))["body"]
    assert body["options"] == {"compiler": None}
    physical = qasm2_loads(body["circuit"])
    edges = set(backend.coupling_map)
    for instruction in physical:
        if instruction.is_gate:
            assert instruction.gate.name in {"h", "rx", "ry", "rz", "cz"}
            if len(instruction.qubits) == 2:
                assert tuple(sorted(instruction.qubits)) in edges


@pytest.mark.parametrize("pending", [0, 2])
def test_round_trip_matches_simulator(provider, fake, pending) -> None:
    fake.pending_polls = pending
    backend = provider.get_backend("Fake")
    circuit = QuantumCircuit(name="asym")
    circuit.x(0)
    circuit.cx(0, 3)  # not adjacent in any default pool: forces routing
    circuit.h(2)
    circuit.measure([0, 1, 3], [2, 0, 3])  # clbit 1 never written
    result = backend.run(circuit, shots=2000, poll_interval=0.001)
    expected = Simulator(seed=1).run(circuit, shots=2048)
    assert result.num_shots == 2048
    assert result.num_clbits == 4
    assert set(result.counts) == set(expected.counts) == {"0011"}
    assert result.metadata["submitted_shots"] == 2048


def test_bell_distribution_survives_compilation(backend) -> None:
    result = backend.run(_ghz(3))
    assert set(result.counts) <= {"000", "111"}
    assert 400 < result.counts.get("000", 0) < 624


def test_per_qubit_measurements_stay_terminal_in_submitted_qasm(backend, fake) -> None:
    """Regression: a SWAP was routed through an already measured qubit."""
    circuit = QuantumCircuit(name="per_qubit")
    circuit.h(0).ry(0.4, 1).ry(1.1, 2)
    circuit.cx(2, 1).cx(0, 2).cx(0, 1).cx(2, 1)
    for qubit in range(3):
        circuit.measure([qubit], [qubit])

    # A line of three physical qubits: the three-way interaction needs a SWAP.
    backend.submit(circuit, physical_qubits=[0, 1, 2])

    sent = qasm2_loads(next(iter(fake.tasks.values()))["body"]["circuit"])
    assert used_after_measurement(sent) == []
    assert measurements_are_last(sent)
    assert_same_distribution(
        quspect_exact_distribution(sent), quspect_exact_distribution(circuit)
    )


@pytest.mark.parametrize("spelling", FULL_SPELLINGS)
def test_random_circuits_keep_terminal_measurements(backend, fake, spelling) -> None:
    for circuit in random_measured_circuits(spelling, count=25):
        backend.submit(circuit)
        sent = qasm2_loads(fake.tasks[max(fake.tasks)]["body"]["circuit"])
        assert used_after_measurement(sent) == []
        assert measurements_are_last(sent)
        assert_same_distribution(
            quspect_exact_distribution(sent), quspect_exact_distribution(circuit)
        )


def test_mid_circuit_measurement_is_submitted_in_the_written_order(
    backend, fake
) -> None:
    circuit = QuantumCircuit(name="mid")
    circuit.x(0).measure([0], [0]).x(0).h(1).measure([1], [1]).measure([0], [2])

    backend.submit(circuit)

    sent = qasm2_loads(next(iter(fake.tasks.values()))["body"]["circuit"])
    kinds = ["m" if i.is_measure else "g" for i in sent.instructions]
    # measure(0 -> c0) stays between the two X gates of its qubit ...
    assert kinds.index("m") < len(kinds) - 1 - kinds[::-1].index("g")
    # ... and the two terminal measurements are issued last.
    assert kinds[-2:] == ["m", "m"]
    assert [i.clbits for i in sent.instructions if i.is_measure][0] == (0,)


def test_transpile_false_requires_physical_couplers(backend) -> None:
    bad = QuantumCircuit()
    bad.cx(0, 2)
    bad.measure([0, 2], [0, 1])
    with pytest.raises(TranslationError, match="not coupled"):
        backend.submit(bad, transpile=False)
    uncalibrated = QuantumCircuit()
    uncalibrated.cz(2, 3)
    uncalibrated.measure([2, 3], [0, 1])
    with pytest.raises(TranslationError):
        backend.submit(uncalibrated, transpile=False)


def test_transpile_false_with_initial_layout(backend, fake) -> None:
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.cz(0, 1)
    circuit.measure([0, 1], [0, 1])
    result = backend.run(circuit, transpile=False, initial_layout=[5, 3])
    qasm = next(iter(fake.tasks.values()))["body"]["circuit"]
    assert "cz q[5],q[3];" in qasm
    assert "measure q[5] -> c[0];" in qasm
    assert set(result.counts) == {"10"}


def test_server_compiler_passes_logical_circuit(backend, fake) -> None:
    backend.run(
        _ghz(2),
        transpile=False,
        server_compiler="qiskit",
        physical_qubits=[1, 4],
        readout_correction=True,
        dynamic_decoupling="XY4",
    )
    body = next(iter(fake.tasks.values()))["body"]
    assert body["options"] == {
        "compiler": "qiskit",
        "target_qubits": [1, 4],
        "correct": True,
        "open_dd": "XY4",
    }
    assert "cx q[0],q[1];" in body["circuit"]


@pytest.mark.parametrize(
    "options",
    [
        {"server_compiler": "qiskit"},  # needs transpile=False
        {"transpile": False, "server_compiler": "nope"},
        {"transpile": False, "physical_qubits": [0, 1]},
        {"dynamic_decoupling": "UDD"},
        {"bogus": 1},
        {"physical_qubits": [0, 6]},  # uncalibrated qubit
    ],
)
def test_invalid_options_never_submit(backend, fake, options) -> None:
    with pytest.raises((ValueError, TypeError, TranslationError)):
        backend.submit(_ghz(2), **options)
    assert not fake.tasks


def test_rejects_unmeasured_and_duplicate_clbits(backend) -> None:
    unmeasured = QuantumCircuit()
    unmeasured.h(0)
    with pytest.raises(TranslationError, match="no measurement"):
        backend.submit(unmeasured)
    duplicate = QuantumCircuit()
    duplicate.h(0)
    duplicate.measure([0], [0]).measure([1], [0])
    with pytest.raises(TranslationError, match="more than"):
        backend.submit(duplicate)


# ---------------------------------------------------------------------------
# Jobs, batches and recovery
# ---------------------------------------------------------------------------


def test_batch_request_preserves_order_and_recovers(provider, backend) -> None:
    a = QuantumCircuit()
    a.x(1)
    a.measure([0, 1], [0, 1])
    b = QuantumCircuit()
    b.x(0)
    b.measure([0], [0])
    request = ExecutionRequest(
        experiments=(CircuitExperiment(a, label="a"), CircuitExperiment(b, label="b")),
        shots=1024,
        metadata={"workload_id": "w1"},
    )
    job = backend.submit_request(request)
    assert isinstance(job, CompositeJob)
    batch = job.result(poll_interval=0.001)
    assert isinstance(batch, ExecutionBatch)
    assert batch.labels == ("a", "b")
    assert batch.by_label("a").counts == {"01": 1024}
    assert batch.by_label("b").counts == {"1": 1024}

    recovered = provider.retrieve_job(job.reference).result(poll_interval=0.001)
    assert [r.counts for r in recovered.results] == [r.counts for r in batch.results]


def test_retrieve_by_task_id(provider, backend) -> None:
    job = backend.submit(_ghz(2))
    recovered = provider.retrieve_job(int(job.job_id), backend_name="Fake")
    result = recovered.result(poll_interval=0.001)
    assert result.num_clbits == 2
    assert set(result.counts) <= {"00", "11"}
    with pytest.raises(ValueError):
        provider.retrieve_job("abc")
    with pytest.raises(ValueError):
        provider.retrieve_job(
            JobReference(platform="logicalqubit", backend_name="x", job_id="1")
        )


def test_failed_task_raises_job_error(backend) -> None:
    circuit = _ghz(2, name="will-fail")
    with pytest.raises(JobError, match="TranspileError"):
        backend.run(circuit)


def test_timeout_and_cancel(provider, fake) -> None:
    fake.pending_polls = 10**6
    backend = provider.get_backend("Fake")
    job = backend.submit(_ghz(2))
    assert job.status() == JobStatus.QUEUED
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01, poll_interval=0.001)
    assert job.cancel() is True
    assert job.status() == JobStatus.CANCELLED
    assert job.cancel() is False
    with pytest.raises(JobError, match="Canceled"):
        job.result(poll_interval=0.001)


def test_partial_submission_failure_cancels_created_tasks(provider, fake) -> None:
    from quspect.backends import PartialSubmissionError

    fake.fail_run_after = 1
    fake.pending_polls = 10**6
    backend = provider.get_backend("Fake")
    request = ExecutionRequest(
        experiments=(
            CircuitExperiment(_ghz(2), label="x"),
            CircuitExperiment(_ghz(2), label="y"),
        ),
        shots=1024,
    )
    with pytest.raises(PartialSubmissionError) as info:
        backend.submit_request(request)
    assert info.value.submitted_job_ids == ("1000",)
    assert fake.tasks[1000]["cancelled"]
    assert info.value.cancellation_failures == ()


def test_partial_submission_reports_uncancellable_tasks(provider, fake) -> None:
    from quspect.backends import PartialSubmissionError

    fake.fail_run_after = 1  # first task finishes instantly, so cancel() is refused
    backend = provider.get_backend("Fake")
    request = ExecutionRequest(
        experiments=(
            CircuitExperiment(_ghz(2), label="x"),
            CircuitExperiment(_ghz(2), label="y"),
        ),
        shots=1024,
    )
    with pytest.raises(PartialSubmissionError) as info:
        backend.submit_request(request)
    assert info.value.cancellation_failures == ("1000",)


def test_get_requests_retry_but_submissions_do_not(fake) -> None:
    import urllib.error

    from quspect.backends._exceptions import BackendError
    from quspect.backends.quafu._client import QuafuClient

    failures = {"left": 2}

    def flaky(method, url, headers, body, timeout):
        if failures["left"]:
            failures["left"] -= 1
            raise urllib.error.URLError("ssl eof")
        return fake(method, url, headers, body, timeout)

    client = QuafuClient(TOKEN, URL, retries=2, backoff=0, transport=flaky)
    assert client.get("/task/verify")[1] == 0

    failures["left"] = 1
    with pytest.raises(BackendError, match="Cannot reach"):
        client.post("/task/run/", params={"name": "x", "chip": "Fake"}, body={})
    assert not fake.tasks


def test_network_error_is_not_reported_as_translation_error(fake) -> None:
    import urllib.error

    from quspect.backends._exceptions import BackendError

    def chip_down(method, url, headers, body, timeout):
        if "backendtest" in url:
            raise urllib.error.URLError("down")
        return fake(method, url, headers, body, timeout)

    backend = QuafuProvider(TOKEN, URL, retries=0, transport=chip_down).get_backend(
        "Fake"
    )
    with pytest.raises(BackendError) as info:
        backend.submit(_ghz(2))
    assert not isinstance(info.value, TranslationError)


# ---------------------------------------------------------------------------
# Reading a job: not ready, unknown to Quafu, undecodable
# ---------------------------------------------------------------------------

_TWO_BIT_QASM = "creg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"


class _Scripted:
    """Answer selected routes with canned bodies; the fake serves the rest."""

    def __init__(self, fake: FakeQuafu, routes: dict[str, tuple[int, bytes]]) -> None:
        self.fake = fake
        self.routes = dict(routes)
        self.calls: list[str] = []

    def __call__(self, method, url, headers, body, timeout):
        path = urllib.parse.urlsplit(url).path
        for prefix, answer in self.routes.items():
            if path.startswith(prefix):
                self.calls.append(path)
                return answer
        return self.fake(method, url, headers, body, timeout)


def _scripted_job(fake: FakeQuafu, routes: dict[str, tuple[int, bytes]]):
    transport = _Scripted(fake, routes)
    provider = QuafuProvider(TOKEN, URL, transport=transport)
    return provider.retrieve_job(4000, backend_name="Fake"), transport


def _result_body(**fields) -> tuple[int, bytes]:
    payload = {"tid": 4000, "error": "", "circuit": _TWO_BIT_QASM, **fields}
    return 200, json.dumps(payload).encode()


def test_job_errors_say_whether_the_result_is_still_pending() -> None:
    assert JobError("failed").result_pending is False
    assert JobTimeoutError("not ready").result_pending is True
    assert issubclass(JobTimeoutError, JobError)  # existing handlers keep working


def test_result_not_ready_is_a_pending_timeout(fake) -> None:
    """Quafu says Finished but has not published the result yet."""
    job, _ = _scripted_job(
        fake,
        {"/task/status/4000": (200, b'"Finished"'), "/task/result/4000": (200, b"{}")},
    )
    with pytest.raises(JobTimeoutError, match="Finished") as caught:
        job.result(timeout=0.01, poll_interval=0.001)
    assert caught.value.result_pending is True
    assert caught.value.job_id == "4000"


def test_undecodable_result_is_a_job_error_that_is_not_pending(fake) -> None:
    job, _ = _scripted_job(
        fake,
        {"/task/result/4000": _result_body(status="Finished", count={"000": 1024})},
    )
    with pytest.raises(JobError, match="Unable to decode") as caught:
        job.result(timeout=5, poll_interval=0.001)
    assert not isinstance(caught.value, JobTimeoutError)
    assert caught.value.result_pending is False


def test_task_unknown_to_quafu_is_reported_instead_of_polled(provider, fake) -> None:
    job = provider.retrieve_job(4242)
    with pytest.raises(JobError, match="not found") as caught:
        job.status()
    assert caught.value.error_type == "not_found"
    assert caught.value.job_id == "4242"
    with pytest.raises(JobError, match="not found") as caught:
        job.result(timeout=0.05, poll_interval=0.001)
    assert not isinstance(caught.value, JobTimeoutError)
    assert caught.value.result_pending is False
    with pytest.raises(JobError, match="not found"):
        job.cancel()
    assert ("GET", "/task/cancel/4242") not in fake.calls


@pytest.mark.parametrize("status", ["Timeout", "Error", "Archived"])
def test_unmapped_status_in_a_result_ends_the_wait_as_failed(fake, status) -> None:
    """A result with a status QuSpect has no mapping for is not "running"."""
    job, _ = _scripted_job(
        fake,
        {
            "/task/status/4000": (200, json.dumps(status).encode()),
            "/task/result/4000": _result_body(status=status, count=None),
        },
    )
    with pytest.raises(JobError, match=f"status '{status}'") as caught:
        job.result(timeout=0.05, poll_interval=0.001)
    assert not isinstance(caught.value, JobTimeoutError)
    assert job.status() == JobStatus.UNKNOWN
    assert job.provider_status() == status


def test_unmapped_task_status_is_unknown_and_named_by_the_timeout(fake) -> None:
    job, _ = _scripted_job(
        fake,
        {"/task/status/4000": (200, b'"Archived"'), "/task/result/4000": (200, b"{}")},
    )
    assert job.status() == JobStatus.UNKNOWN
    assert job.provider_status() == "Archived"
    assert not job.done()
    with pytest.raises(JobTimeoutError, match="Archived"):
        job.result(timeout=0.01, poll_interval=0.001)


def test_in_progress_status_in_a_result_keeps_waiting(fake) -> None:
    job, _ = _scripted_job(
        fake,
        {
            "/task/status/4000": (200, b'"Running"'),
            "/task/result/4000": _result_body(status="Running"),
        },
    )
    with pytest.raises(JobTimeoutError, match="Running"):
        job.result(timeout=0.01, poll_interval=0.001)


# ---------------------------------------------------------------------------
# Cancel, 403 answers, task ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        (b"null", True),
        (b"", True),
        (b'"Canceled"', True),
        (b'{"detail": "task is running, cannot cancel"}', False),
        (b'"Failed"', False),
        (b'"task 4000 not found!"', False),
        (b"false", False),
    ],
)
def test_cancel_reports_whether_quafu_accepted_it(fake, answer, accepted) -> None:
    job, transport = _scripted_job(
        fake,
        {"/task/status/4000": (200, b'"Running"'), "/task/cancel/4000": (200, answer)},
    )
    assert job.cancel() is accepted
    assert transport.calls.count("/task/cancel/4000") == 1


def test_partial_submission_lists_tasks_whose_cancel_was_refused(fake) -> None:
    from quspect.backends import PartialSubmissionError

    fake.fail_run_after = 1
    fake.pending_polls = 10**6
    refusing = _Scripted(fake, {"/task/cancel/": (200, b'{"detail": "cannot cancel"}')})
    backend = QuafuProvider(TOKEN, URL, transport=refusing).get_backend("Fake")
    request = ExecutionRequest(
        experiments=(
            CircuitExperiment(_ghz(2), label="x"),
            CircuitExperiment(_ghz(2), label="y"),
        ),
        shots=1024,
    )
    with pytest.raises(PartialSubmissionError) as info:
        backend.submit_request(request)
    assert info.value.cancellation_failures == ("1000",)


@pytest.mark.parametrize(
    ("status", "body", "token_rejected"),
    [
        (401, b'{"detail": "invalid token"}', True),
        (401, b'{"detail": "program not in the run"}', True),
        (403, b'{"detail": "Token expired"}', True),
        (403, b'{"detail": "Not authenticated"}', True),
        (403, b'"Forbidden"', True),
        (403, b"", True),
        (403, b"{}", True),
        (403, b'{"detail": "program not in the run"}', False),
        (403, b'"chip Baihua is under maintenance"', False),
    ],
)
def test_only_credential_answers_read_as_a_rejected_token(
    status, body, token_rejected
) -> None:
    """A 403 whose body gives another reason is not a rejected token."""
    from quspect.backends._exceptions import BackendError
    from quspect.backends.quafu._client import QuafuClient

    client = QuafuClient(TOKEN, URL, transport=lambda *args: (status, body))
    with pytest.raises(BackendError) as caught:
        client.get("/task/status/7")
    message = str(caught.value)
    assert isinstance(caught.value, AuthenticationError) is token_rejected
    assert ("rejected the API token" in message) is token_rejected
    if not token_rejected:
        reason = json.loads(body)
        reason = reason["detail"] if isinstance(reason, dict) else reason
        assert message == f"Quafu refused GET /task/status/7 (HTTP 403): {reason}"
    assert TOKEN not in message


@pytest.mark.parametrize("answer", [b"0", b"-1"])
def test_submit_rejects_a_non_positive_task_id(fake, answer) -> None:
    from quspect.backends._exceptions import BackendError

    transport = _Scripted(fake, {"/task/run/": (200, answer)})
    backend = QuafuProvider(TOKEN, URL, transport=transport).get_backend("Fake")
    with pytest.raises(BackendError, match="task id"):
        backend.submit(_ghz(2))


@pytest.mark.parametrize("task_id", [0, "0", "00", -1, "-1"])
def test_retrieve_rejects_non_positive_task_ids(provider, fake, task_id) -> None:
    """``/task/status/0`` is the chip listing, not a task."""
    with pytest.raises(ValueError, match="positive"):
        provider.retrieve_job(task_id)
    reference = JobReference(platform="quafu", backend_name="Fake", job_id=str(task_id))
    with pytest.raises(ValueError, match="positive"):
        provider.retrieve_job(reference)
    composite = JobReference(
        platform="quafu",
        backend_name="Fake",
        job_id="composite",
        child_job_ids=("1000", str(task_id)),
        batch_size=2,
        child_batch_sizes=(1, 1),
    )
    with pytest.raises(ValueError, match="positive"):
        provider.retrieve_job(composite)
    assert not any("/task/status/0" in path for _, path in fake.calls[2:])
