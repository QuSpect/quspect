"""Tests for :class:`quspect.transpiler.PhaseFoldingPass`."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import PhaseFoldingPass


def _unitary(circuit: QuantumCircuit, n: int) -> np.ndarray:
    """Big-endian unitary of a gate-only circuit on ``n`` qubits."""
    from quspect.gates._matrix_utils import gate_matrix

    psi = np.eye(2**n, dtype=complex).reshape([2] * n + [2**n])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = np.asarray(gate_matrix(ins.gate), dtype=complex).reshape([2] * (2 * k))
        psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(ins.qubits))
    return psi.reshape(2**n, 2**n)


def _assert_equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> None:
    """Phase-aligned element-wise comparison (linear in the error)."""
    overlap = complex(np.vdot(b, a))
    phase = overlap / abs(overlap)
    assert float(np.max(np.abs(a - phase * b))) < 1e-9


def _random_circuit(n: int, size: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    angles = [math.pi / 4 * k for k in range(-8, 9)] + [0.3, -1.1, 2.0]
    qc = QuantumCircuit()
    for _ in range(size):
        r = rng.random()
        q = rng.randrange(n)
        a, b, c = rng.sample(range(n), 3)
        if r < 0.25:
            qc.cx(a, b)
        elif r < 0.45:
            getattr(qc, rng.choice(["t", "tdg", "s", "sdg", "z"]))(q)
        elif r < 0.6:
            qc.rz(rng.choice(angles), q)
        elif r < 0.65:
            qc.p(rng.choice(angles), q)
        elif r < 0.7:
            qc.h(q)
        elif r < 0.74:
            qc.x(q)
        elif r < 0.77:
            qc.y(q)
        elif r < 0.81:
            qc.cz(a, b)
        elif r < 0.85:
            qc.cp(rng.choice(angles), a, b)
        elif r < 0.88:
            qc.swap(a, b)
        elif r < 0.92:
            qc.ccx(a, b, c)
        elif r < 0.95:
            qc.ry(0.7, q)
        else:
            qc.crz(rng.choice(angles), a, b)
    return qc


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_equivalent(seed: int) -> None:
    n = 4
    qc = _random_circuit(n, 60, seed)
    out = PhaseFoldingPass().run(qc)
    assert len(out) <= len(qc)
    _assert_equal_up_to_phase(_unitary(out, n), _unitary(qc, n))


def test_merges_over_cnot_network() -> None:
    # q1 holds q0 ⊕ q1 at both rotations; they cancel.
    qc = QuantumCircuit().cx(0, 1).t(1).cx(0, 1).h(0).cx(0, 1).tdg(1)
    out = PhaseFoldingPass().run(qc)
    names = [i.gate.name for i in out]
    # after h(0) the second rotation acts on a fresh parity: nothing merges
    assert names == ["cx", "t", "cx", "h", "cx", "tdg"]
    qc = QuantumCircuit().cx(0, 1).t(1).cx(2, 0).cx(0, 2).tdg(1).cx(0, 1)
    out = PhaseFoldingPass().run(qc)
    assert [i.gate.name for i in out] == ["cx", "cx", "cx", "cx"]
    _assert_equal_up_to_phase(_unitary(out, 3), _unitary(qc, 3))


def test_constant_flip_negates_angle() -> None:
    # x flips the parity's constant: rz(θ) on ¬q equals rz(−θ) on q.
    qc = QuantumCircuit().rz(0.4, 0).x(0).rz(0.4, 0).x(0)
    out = PhaseFoldingPass().run(qc)
    assert [i.gate.name for i in out] == ["x", "x"]
    _assert_equal_up_to_phase(_unitary(out, 1), _unitary(qc, 1))


def test_multiple_of_two_pi_removed() -> None:
    qc = QuantumCircuit().cx(1, 0).rz(-4 * math.pi, 0).cx(2, 0).rz(4 * math.pi, 0)
    qc = qc.cx(1, 0).rz(2 * math.pi, 0).cx(2, 0).rz(-2 * math.pi, 0)
    out = PhaseFoldingPass().run(qc)
    assert [i.gate.name for i in out] == ["cx", "cx", "cx", "cx"]
    stats = PhaseFoldingPass()
    stats.run(qc)
    assert stats.stats.rotations_in == 4
    assert stats.stats.rotations_out == 0


def test_diagonal_two_qubit_gates_are_transparent() -> None:
    qc = QuantumCircuit().t(0).cz(0, 1).cp(0.3, 1, 0).t(0)
    out = PhaseFoldingPass().run(qc)
    assert [i.gate.name for i in out] == ["rz", "cz", "cp"]
    _assert_equal_up_to_phase(_unitary(out, 2), _unitary(qc, 2))


def test_swap_moves_parity() -> None:
    qc = QuantumCircuit().s(0).swap(0, 1).s(1)
    out = PhaseFoldingPass().run(qc)
    assert [i.gate.name for i in out] == ["rz", "swap"]
    _assert_equal_up_to_phase(_unitary(out, 2), _unitary(qc, 2))


def test_toffoli_target_gets_fresh_parity() -> None:
    qc = QuantumCircuit().t(2).ccx(0, 1, 2).tdg(2).t(0).ccx(0, 1, 2).tdg(0)
    out = PhaseFoldingPass().run(qc)
    # the controls keep their parity (t/tdg on q0 cancel), the target does not
    assert [i.gate.name for i in out] == ["t", "ccx", "tdg", "ccx"]
    _assert_equal_up_to_phase(_unitary(out, 3), _unitary(qc, 3))


def test_measurement_is_a_hard_boundary() -> None:
    qc = QuantumCircuit().t(0).measure([0], [0]).t(0)
    out = PhaseFoldingPass().run(qc)
    assert [i.gate.name if i.gate else "measure" for i in out] == [
        "t",
        "measure",
        "t",
    ]


def test_symbolic_parameters_untouched() -> None:
    from quspect.circuit.parameter import Parameter

    theta = Parameter("theta")
    qc = QuantumCircuit().rz(theta, 0).t(0).rz(theta, 0).t(0)
    out = PhaseFoldingPass().run(qc)
    names = [i.gate.name for i in out]
    assert names.count("rz") >= 2
    assert len(out) == 3  # only the two t merge


def test_unmerged_rotations_kept_verbatim() -> None:
    qc = QuantumCircuit().t(0).h(0).t(0)
    out = PhaseFoldingPass().run(qc)
    assert [(i.gate, i.qubits) for i in out] == [(i.gate, i.qubits) for i in qc]
