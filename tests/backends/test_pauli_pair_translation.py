"""Cloud translators emit ``rxx``/``ryy``/``rzz`` as exact CX/CNOT forms.

The providers are only known to accept the gates the translators already
emit, so the Pauli rotations go out as ``CX · RZ · CX`` with ``H`` / ``RX(±π/2)``
basis changes.  Every emission is replayed into a QuSpect circuit and
compared with the original gate on a generic state.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from quspect import QuantumCircuit, Simulator

_PREP = [(0.3, 1.1), (1.7, -0.4), (2.2, 0.9)]
_CASES = {
    "rxx": lambda c: c.rxx(0.37, 2, 0),
    "ryy": lambda c: c.ryy(-1.21, 0, 2),
    "rzz": lambda c: c.rzz(2.05, 1, 2),
}


def _prepared() -> QuantumCircuit:
    circuit = QuantumCircuit()
    for qubit, (ry, rz) in enumerate(_PREP):
        circuit.ry(ry, qubit).rz(rz, qubit)
    return circuit


def _assert_same_state(a: QuantumCircuit, b: QuantumCircuit) -> None:
    sa = Simulator().statevector(a)
    sb = Simulator().statevector(b)
    assert abs(abs(np.vdot(sa, sb)) - 1.0) < 1e-12


@pytest.mark.parametrize("name", sorted(_CASES))
def test_originq_emits_an_exact_cnot_form(name: str) -> None:
    from quspect.backends.originq._translator import translate

    circuit = _prepared()
    _CASES[name](circuit)
    measured = circuit.copy()
    measured.measure([0], [0])
    lines = translate(measured).text.splitlines()
    body = [line for line in lines[2:] if not line.startswith("MEASURE")]
    gates = body[2 * len(_PREP) :]  # after the RY/RZ preparation
    assert {line.split()[0] for line in gates} <= {"H", "RX", "CNOT", "RZ"}
    assert sum(line.startswith("CNOT") for line in gates) == 2

    replay = _prepared()
    methods = {"H": "h", "CNOT": "cx", "RZ": "rz", "RX": "rx"}
    for line in gates:
        mnemonic, rest = line.split(" ", 1)
        qubits = [int(q) for q in re.findall(r"q\[(\d+)\]", rest)]
        args = re.findall(r"\(([^)]*)\)", rest)
        params = [float(x) for x in args[0].split(",")] if args else []
        getattr(replay, methods[mnemonic])(*params, *qubits)
    _assert_same_state(circuit, replay)


class _RecordingLQCircuit:
    """Stands in for an lqcloud circuit; replays calls into QuSpect."""

    def __init__(self) -> None:
        self.circuit = _prepared()
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any) -> None:
            self.calls.append(name)
            getattr(self.circuit, name)(*args)

        return call


@pytest.mark.parametrize("name", sorted(_CASES))
def test_logicalqubit_emits_an_exact_cx_form(name: str) -> None:
    from quspect.backends.logicalqubit._translator import _GATE_MAP

    circuit = _prepared()
    _CASES[name](circuit)
    instruction = circuit.instructions[-1]
    assert instruction.gate is not None

    recorder = _RecordingLQCircuit()
    _GATE_MAP[name](recorder, list(instruction.qubits), list(instruction.gate.params))
    assert set(recorder.calls) <= {"h", "rx", "cx", "rz"}
    assert recorder.calls.count("cx") == 2
    _assert_same_state(circuit, recorder.circuit)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_tianyan_lowers_to_qcis(name: str) -> None:
    from quspect.backends.tianyan._translator import QCIS_BASIS, translate
    from quspect.transpiler._passes import DecomposePass
    from quspect.transpiler._target import CompilationTarget

    circuit = _prepared()
    _CASES[name](circuit)
    lowered = DecomposePass(
        CompilationTarget(num_qubits=0, native_gates=set(QCIS_BASIS))
    ).run(circuit)
    assert {i.gate.name for i in lowered if i.gate} <= QCIS_BASIS
    _assert_same_state(circuit, lowered)
    measured = lowered.copy()
    measured.measure([0, 1, 2], [0, 1, 2])
    assert "CZ" in translate(measured).text
