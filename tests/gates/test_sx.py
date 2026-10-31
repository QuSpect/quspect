"""Tests for the √X gates (``sx``/``sxdg``) and the ZSX single-qubit synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates import gate_matrix
from quspect.gates.parameterized import RX, U3
from quspect.gates.standard import SX, SXDG, H, X
from quspect.gates.synthesis import _sequence_matrix, synthesize_one_qubit
from quspect.ir import QASM2Exporter, qasm2_loads

_IBM_1Q = {"rz", "sx", "x"}


def _phase_dist(a: np.ndarray, b: np.ndarray) -> float:
    k = int(np.argmax(np.abs(a)))
    phase = a.flat[k] / b.flat[k]
    return float(np.max(np.abs(a - phase * b)))


def test_sx_matrices() -> None:
    sx = gate_matrix(SX)
    np.testing.assert_allclose(sx @ sx, gate_matrix(X), atol=1e-15)
    np.testing.assert_allclose(gate_matrix(SXDG), sx.conj().T, atol=1e-15)
    np.testing.assert_allclose(
        sx, np.exp(1j * np.pi / 4) * gate_matrix(RX.with_params(np.pi / 2)), atol=1e-15
    )
    h = gate_matrix(H)
    np.testing.assert_allclose(h @ np.diag([1, 1j]) @ h, sx, atol=1e-15)


def test_circuit_methods_and_inverse() -> None:
    qc = QuantumCircuit().sx(0).sxdg(1)
    assert [i.gate.name for i in qc] == ["sx", "sxdg"]
    assert [i.gate.name for i in qc.dagger()] == ["sx", "sxdg"]


def test_qasm_round_trip() -> None:
    qc = QuantumCircuit().sx(0).sxdg(1).cx(0, 1).sx(1)
    text = QASM2Exporter().export(qc)
    assert "sx q[0];" in text and "sxdg q[1];" in text
    back = qasm2_loads(text)
    assert [(i.gate.name, i.qubits) for i in back] == [
        (i.gate.name, i.qubits) for i in qc
    ]


def test_qasm_import_sx_is_native() -> None:
    qc = qasm2_loads('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nsx q[0];\n')
    assert [i.gate.name for i in qc] == ["sx"]


@pytest.mark.parametrize("seed", range(5))
def test_zsx_synthesis_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(100):
        m = gate_matrix(U3.with_params(*rng.uniform(-4, 4, 3)))
        seq = synthesize_one_qubit(m, _IBM_1Q)
        assert seq is not None and len(seq) <= 5
        assert {g.name for g in seq} <= _IBM_1Q
        assert _phase_dist(m, _sequence_matrix(seq)) < 1e-9


@pytest.mark.parametrize(
    ("theta", "max_len"), [(np.pi / 2, 3), (-np.pi / 2, 3), (np.pi, 2), (0.0, 1)]
)
def test_zsx_special_angle_shortcuts(theta: float, max_len: int) -> None:
    m = gate_matrix(U3.with_params(theta, 0.3, 0.7))
    seq = synthesize_one_qubit(m, _IBM_1Q)
    assert seq is not None and len(seq) <= max_len
    assert _phase_dist(m, _sequence_matrix(seq)) < 1e-9


def test_named_gates_in_ibm_basis() -> None:
    seq = synthesize_one_qubit(gate_matrix(H), _IBM_1Q)
    assert seq is not None and [g.name for g in seq] == ["rz", "sx", "rz"]
    seq = synthesize_one_qubit(gate_matrix(SX), {"rz", "sx"})
    assert seq is not None and [g.name for g in seq] == ["sx"]
    # without x, anti-diagonal unitaries use sx·sx
    seq = synthesize_one_qubit(gate_matrix(X), {"rz", "sx"})
    assert seq is not None and [g.name for g in seq] == ["sx", "sx"]
