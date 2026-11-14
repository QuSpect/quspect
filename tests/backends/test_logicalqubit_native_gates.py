"""Native gates of a LogicalQubit chip: names read from the chip, names sent.

lqcloud names its native gates ``x, y, z, xhalf, mxhalf, yhalf, rz, cz, su2``
(see ``lqcloud.circuit.gates``).  A chip that declares ``native_gates`` in
those names must still compile, and √X must travel as lqcloud's native
``sx`` (``xhalf``), not as ``H · S · H``.

The first half runs without lqcloud.  The second half builds real lqcloud
circuits and serializes them offline; it never constructs a provider.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from quspect import QuantumCircuit, Simulator
from quspect.backends.logicalqubit._backend import LogicalQubitBackend
from quspect.ir.qsal import dumps, loads
from tests.backends.test_logicalqubit_backend import (
    _FakeSDKBackend,
    _grover_target_101,
    _IdentityTranslator,
)

#: The set the LogicalQubit device catalog's own test chip declares.
LQCLOUD_NATIVE = ["x", "y", "z", "xhalf", "yhalf", "rz", "cz", "su2"]


def _chip(native_gates: object, *, qubits: int = 5) -> _FakeSDKBackend:
    raw = _FakeSDKBackend()
    raw.config["qubits"] = qubits
    raw.config["native_gates"] = native_gates
    raw.config["topology"] = {
        "coupling_map": [[index, index + 1] for index in range(qubits - 1)]
    }
    return raw


def _sent_gate_names(document: object) -> list[str]:
    assert isinstance(document, str)
    instructions = json.loads(document)["circuit"]["instructions"]
    return [item["gate"]["name"] for item in instructions if item["type"] == "GATE"]


# ---------------------------------------------------------------------------
# Reading the chip's native gates (no lqcloud needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (LQCLOUD_NATIVE, {"x", "y", "z", "sx", "yhalf", "rz", "cz", "u3"}),
        (["XHalf", "MXHalf", "RZ", "CZ"], {"sx", "sxdg", "rz", "cz"}),
        (["xhalf", "rz", "cz"], {"sx", "rz", "cz"}),
        # QuSpect's own names, and names it does not know, pass through.
        (["sx", "rz", "cz"], {"sx", "rz", "cz"}),
        (["CZ", "Rx"], {"cz", "rx"}),
        (["cz", "h", "u", "id"], {"cz", "h", "u3", "id"}),
        (None, set()),
        ("cx", set()),
    ],
)
def test_native_gate_set_is_in_quspect_gate_names(
    declared: object, expected: set[str]
) -> None:
    assert LogicalQubitBackend(_chip(declared)).native_gate_set == expected


def test_native_gate_set_reads_the_native_gate_set_key_too() -> None:
    raw = _chip(None)
    raw.config["native_gate_set"] = ["su2", "cz"]
    assert LogicalQubitBackend(raw).native_gate_set == {"u3", "cz"}


@pytest.mark.parametrize("declared", [LQCLOUD_NATIVE, ["xhalf", "rz", "cz"]])
def test_chip_declaring_lqcloud_native_names_compiles(declared: list[str]) -> None:
    """Known issue: every compile failed with "Cannot express gate(s) ['h']"."""
    backend = LogicalQubitBackend(_chip(declared))
    backend._translator = _IdentityTranslator()
    source = _grover_target_101()

    backend.submit(source, shots=8)

    document, _ = backend._lq_backend.calls[0]
    names = set(_sent_gate_names(document))
    assert names <= backend.native_gate_set
    assert "cz" in names
    routed = loads(document).circuit
    assert routed is not None
    assert Simulator(seed=7).run(routed, shots=32).counts == {"101": 32}


def _routed_with_a_fence(fence: str) -> QuantumCircuit:
    """A classical triangle of CX gates, which a line holds only with SWAPs.

    The fence (a barrier between gates, or a mid-circuit measurement) sends
    the circuit through the device pipeline that routes after lowering.
    """
    circuit = QuantumCircuit().x(0).cx(0, 1).cx(1, 2)
    if fence == "barrier":
        circuit.barrier()
    else:
        circuit.measure([0], [3])
    circuit.cx(0, 2).t(2).cx(2, 1).x(1)
    return circuit.measure([0, 1, 2], [0, 1, 2])


@pytest.mark.parametrize("fence", ["barrier", "mid-circuit measurement"])
def test_routing_swaps_are_sent_in_the_declared_native_gates(fence: str) -> None:
    """Known issue: routing SWAPs went out as ``swap``, which lqcloud lowered
    to ``yhalf`` / ``myhalf``, gates the chip did not declare."""
    backend = LogicalQubitBackend(_chip(["xhalf", "rz", "cz"], qubits=6))
    backend._translator = _IdentityTranslator()
    circuit = _routed_with_a_fence(fence)

    backend.submit(circuit, shots=8)

    document, options = backend._lq_backend.calls[0]
    assert set(_sent_gate_names(document)) <= {"sx", "rz", "cz"}
    layout = options["initial_layout"]
    routed = loads(document).circuit
    assert routed is not None
    for instruction in routed:
        if instruction.is_gate and len(instruction.qubits) == 2:
            a, b = (layout[wire] for wire in instruction.qubits)
            assert abs(a - b) == 1
    expected = Simulator(seed=3).run(circuit, shots=16).counts
    assert len(expected) == 1
    assert Simulator(seed=3).run(routed, shots=16).counts == expected


def test_chip_with_quspect_native_names_keeps_sx_as_sx() -> None:
    backend = LogicalQubitBackend(_chip(["sx", "rz", "cz"]))
    backend._translator = _IdentityTranslator()

    backend.submit(QuantumCircuit().h(0).cx(0, 1).measure_all(), shots=8)

    names = set(_sent_gate_names(backend._lq_backend.calls[0][0]))
    assert names <= {"sx", "rz", "cz"}
    assert "sx" in names


# ---------------------------------------------------------------------------
# What reaches lqcloud (real circuit builder and serializer, offline)
# ---------------------------------------------------------------------------

#: Serialized instructions that are not gates (lqcloud starts every payload
#: with a reset of each qubit and a barrier).
_NOT_GATES = {"reset", "barrier", "measure"}
_SQRT_HALF = math.sqrt(0.5)
_NATIVE_MATRICES: dict[str, np.ndarray] = {
    "id": np.eye(2, dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) * _SQRT_HALF,
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.diag([1, -1]).astype(complex),
    "s": np.diag([1, 1j]).astype(complex),
    "sdg": np.diag([1, -1j]).astype(complex),
    "t": np.diag([1, np.exp(0.25j * math.pi)]).astype(complex),
    # X/2 and −X/2: rotations by ±π/2 about X.
    "xhalf": np.array([[1, -1j], [-1j, 1]], dtype=complex) * _SQRT_HALF,
    "mxhalf": np.array([[1, 1j], [1j, 1]], dtype=complex) * _SQRT_HALF,
}


def _native_unitary(payload: dict[str, Any], num_qubits: int) -> np.ndarray:
    """Unitary of a serialized lqcloud circuit (qubit 0 is the lowest bit)."""
    total = np.eye(2**num_qubits, dtype=complex)

    def embed(matrix: np.ndarray, qubit: int) -> np.ndarray:
        out = np.ones((1, 1), dtype=complex)
        for index in range(num_qubits - 1, -1, -1):
            out = np.kron(out, matrix if index == qubit else np.eye(2))
        return out

    for instruction in payload["instructions"]:
        name = instruction["name"]
        qubits = [int(q) for q in instruction["qubits"]]
        params = [float(p) for p in instruction.get("params", [])]
        if name in _NOT_GATES:
            continue
        if name == "cz":
            gate = np.eye(2**num_qubits, dtype=complex)
            for state in range(2**num_qubits):
                if (state >> qubits[0]) & 1 and (state >> qubits[1]) & 1:
                    gate[state, state] = -1
        elif name == "rz":
            gate = embed(
                np.diag([np.exp(-0.5j * params[0]), np.exp(0.5j * params[0])]),
                qubits[0],
            )
        elif name == "su2":
            matrix = np.array(
                [
                    [complex(params[0], params[1]), complex(params[2], params[3])],
                    [complex(params[4], params[5]), complex(params[6], params[7])],
                ]
            )
            gate = embed(matrix, qubits[0])
        else:
            gate = embed(_NATIVE_MATRICES[name], qubits[0])
        total = gate @ total
    return total


def _quspect_unitary(circuit: QuantumCircuit, num_qubits: int) -> np.ndarray:
    columns = []
    for state in range(2**num_qubits):
        prepared = QuantumCircuit()
        for qubit in range(num_qubits):
            if (state >> qubit) & 1:
                prepared.x(qubit)
            else:
                prepared.id(qubit)
        for instruction in circuit:
            prepared.append(instruction.gate, instruction.qubits)
        columns.append(np.asarray(Simulator().statevector(prepared)))
    return np.stack(columns, axis=1)


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    index = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    phase = a[index] / b[index]
    return bool(abs(abs(phase) - 1) < 1e-9 and np.allclose(a, phase * b, atol=1e-9))


class TestAgainstLqcloud:
    """Needs the lqcloud package (Python 3.11–3.12); skipped without it."""

    @pytest.fixture(autouse=True)
    def _lqcloud(self) -> None:
        pytest.importorskip("lqcloud")

    @staticmethod
    def _serialize(circuit: Any) -> dict[str, Any]:
        from lqcloud.backend.serialization import serialize_circuit

        return dict(serialize_circuit(circuit))

    @staticmethod
    def _translate(circuit: QuantumCircuit) -> Any:
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator

        return LQCircuitTranslator().translate(dumps(circuit))

    @pytest.mark.parametrize(
        ("method", "builder_name", "native_name"),
        [("sx", "sx", "xhalf"), ("sxdg", "sxdg", "mxhalf")],
    )
    def test_sx_is_sent_as_the_native_gate(
        self, method: str, builder_name: str, native_name: str
    ) -> None:
        """Known issue: √X went out as H · S · H (three pulses for one)."""
        circuit = QuantumCircuit()
        getattr(circuit, method)(0)
        circuit.barrier().measure([0], [0])

        translated = self._translate(circuit)

        gates = [gate.name for gate, _, _ in translated.data]
        assert gates == [builder_name, "barrier", "measure"]
        sent = [item["name"] for item in self._serialize(translated)["instructions"]]
        assert [name for name in sent if name not in _NOT_GATES] == [native_name]

    @pytest.mark.parametrize("method", ["sx", "sxdg"])
    def test_native_sx_is_the_same_gate_as_quspect_sx(self, method: str) -> None:
        def build() -> QuantumCircuit:
            circuit = QuantumCircuit().h(0)
            getattr(circuit, method)(0)
            return circuit.t(0)

        measured = build().barrier().measure([0], [0])
        payload = self._serialize(self._translate(measured))

        assert _equal_up_to_phase(
            _native_unitary(payload, 1), _quspect_unitary(build(), 1)
        )

    @pytest.mark.parametrize(
        "declared",
        [
            LQCLOUD_NATIVE,
            ["xhalf", "rz", "cz"],
            ["sx", "rz", "cz"],
            ["xhalf", "mxhalf", "rz", "cz"],
        ],
    )
    def test_compiled_circuit_stays_in_the_declared_native_gates(
        self, declared: list[str]
    ) -> None:
        """What lqcloud would send holds only gates the chip declared."""
        rng = np.random.default_rng(8)
        allowed = {"xhalf" if name == "sx" else name for name in declared}
        for _ in range(12):
            circuit = QuantumCircuit()
            for _ in range(10):
                kind = int(rng.integers(0, 7))
                a, b = (int(q) for q in rng.permutation(2)[:2])
                if kind == 0:
                    circuit.h(a)
                elif kind == 1:
                    circuit.sx(a)
                elif kind == 2:
                    circuit.sxdg(a)
                elif kind == 3:
                    circuit.rx(float(rng.uniform(-3, 3)), a)
                elif kind == 4:
                    circuit.t(a)
                elif kind == 5:
                    circuit.cx(min(a, b), min(a, b) + 1)
                else:
                    circuit.u3(*(float(v) for v in rng.uniform(-3, 3, 3)), a)
            circuit.cz(0, 1).cz(1, 2)
            unitary_part = QuantumCircuit()
            for instruction in circuit:
                unitary_part.append(instruction.gate, instruction.qubits)
            circuit.measure([0, 1, 2], [0, 1, 2])

            # Real translator, real lqcloud circuit; the line 0-1-2 needs no SWAP
            # with the identity layout.
            backend = LogicalQubitBackend(_chip(declared, qubits=3))
            backend.submit(circuit, shots=8, initial_layout=[0, 1, 2])
            program, options = backend._lq_backend.calls[0]
            payload = self._serialize(program)

            sent = {item["name"] for item in payload["instructions"]}
            assert sent - _NOT_GATES <= allowed
            assert options["initial_layout"] == [0, 1, 2]
            assert _equal_up_to_phase(
                _native_unitary(payload, 3), _quspect_unitary(unitary_part, 3)
            )

    @pytest.mark.parametrize("fence", ["barrier", "mid-circuit measurement"])
    def test_routed_circuit_stays_in_the_declared_native_gates(
        self, fence: str
    ) -> None:
        """Known issue: lqcloud lowered routing SWAPs to ``yhalf``/``myhalf``."""
        rng = np.random.default_rng(54)
        for _ in range(20):
            n = int(rng.integers(3, 7))
            a, b, c = (int(q) for q in rng.choice(n, 3, replace=False))
            circuit = QuantumCircuit()
            for qubit in range(n):
                circuit.h(qubit)
            # A triangle of two-qubit gates: the line needs routing SWAPs.
            circuit.cx(a, b).cx(b, c)
            if fence == "barrier":
                circuit.barrier()
            else:
                circuit.measure([a], [n])
            circuit.cx(a, c)
            for _ in range(6):
                circuit.ry(float(rng.uniform(-3, 3)), int(rng.integers(0, n)))
                x, y = (int(q) for q in rng.choice(n, 2, replace=False))
                circuit.cz(x, y)
            circuit.measure(list(range(n)), list(range(n)))

            backend = LogicalQubitBackend(_chip(["xhalf", "rz", "cz"], qubits=6))
            backend.submit(circuit, shots=8)
            program, _ = backend._lq_backend.calls[0]
            payload = self._serialize(program)

            sent = {item["name"] for item in payload["instructions"]}
            assert sent - _NOT_GATES <= {"xhalf", "rz", "cz"}
