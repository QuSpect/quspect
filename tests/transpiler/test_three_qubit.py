"""Tests for :class:`quspect.transpiler.ThreeQubitBlockPass`."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import ThreeQubitBlockPass
from quspect.transpiler._three_qubit import (
    _local_perm,
    _ops_unitary,
    synthesize_three_qubit,
)


def _haar(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _unitary(circuit: QuantumCircuit, n: int) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    psi = np.eye(2**n, dtype=complex).reshape([2] * n + [2**n])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = np.asarray(gate_matrix(ins.gate), dtype=complex).reshape([2] * (2 * k))
        psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(ins.qubits))
    return psi.reshape(2**n, 2**n)


def _phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    overlap = complex(np.vdot(b, a))
    return float(np.max(np.abs(a - overlap / abs(overlap) * b)))


def _two_qubit_count(ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]]) -> int:
    return sum(1 for _, q, _ in ops if len(q) == 2)


@pytest.mark.parametrize("seed", range(8))
def test_general_unitary_at_most_24(seed: int) -> None:
    u = _haar(8, seed)
    ops = synthesize_three_qubit(u)
    assert ops is not None
    assert _two_qubit_count(ops) <= 24
    assert _phase_distance(_ops_unitary(ops), u) < 1e-9


@pytest.mark.parametrize("control", [0, 1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_multiplexed_unitary_at_most_10(control: int, seed: int) -> None:
    mux = np.zeros((8, 8), dtype=complex)
    mux[:4, :4] = _haar(4, seed)
    mux[4:, 4:] = _haar(4, 100 + seed)
    others = [q for q in range(3) if q != control]
    perm = _local_perm(control, *others)
    u = perm.T @ mux @ perm  # select qubit moved to ``control``
    ops = synthesize_three_qubit(u)
    assert ops is not None
    assert _two_qubit_count(ops) <= 10
    assert _phase_distance(_ops_unitary(ops), u) < 1e-9


def test_degenerate_multiplexor() -> None:
    # A = B: the select qubit is idle (degenerate eigenvalues of A B†).
    a = _haar(4, 7)
    u = np.kron(np.eye(2), a)
    ops = synthesize_three_qubit(u)
    assert ops is not None
    assert _phase_distance(_ops_unitary(ops), u) < 1e-9


def _controlled_power_circuit(reps: int, seed: int) -> QuantumCircuit:
    """``reps`` copies of a controlled two-qubit gadget (like QPE's c-U^k)."""
    rng = random.Random(seed)
    angles = [rng.uniform(-3, 3) for _ in range(6)]
    qc = QuantumCircuit().h(0)
    for _ in range(reps):
        qc.cx(0, 1).ry(angles[0], 2).ccx(0, 1, 2).rz(angles[1], 1)
        qc.cp(angles[2], 0, 2).cx(2, 1).ccx(0, 2, 1).t(0)
    return qc


@pytest.mark.parametrize("reps", [2, 5, 16])
def test_pass_collapses_repeated_controlled_blocks(reps: int) -> None:
    qc = _controlled_power_circuit(reps, reps)
    pass_ = ThreeQubitBlockPass()
    out = pass_.run(qc)
    assert pass_.stats.blocks_replaced == 1
    assert sum(1 for i in out if len(i.qubits) >= 2) <= 10
    assert _phase_distance(_unitary(out, 3), _unitary(qc, 3)) < 1e-9


def test_cheap_blocks_kept() -> None:
    qc = QuantumCircuit().h(0).cx(0, 1).cx(1, 2).t(2).cx(0, 2)
    out = ThreeQubitBlockPass().run(qc)
    assert [(i.gate, i.qubits) for i in out] == [(i.gate, i.qubits) for i in qc]


@pytest.mark.parametrize("seed", range(10))
def test_random_multi_block_circuits_equivalent(seed: int) -> None:
    rng = random.Random(seed)
    n = 5
    qc = QuantumCircuit()
    for _ in range(80):
        r = rng.random()
        a, b, c = rng.sample(range(n), 3)
        # mostly stay within a sliding window of three qubits
        if rng.random() < 0.8:
            base = rng.randrange(n - 2)
            a, b, c = rng.sample(range(base, base + 3), 3)
        if r < 0.3:
            qc.cx(a, b)
        elif r < 0.45:
            qc.ccx(a, b, c)
        elif r < 0.6:
            qc.cp(rng.uniform(-3, 3), a, b)
        elif r < 0.8:
            qc.ry(rng.uniform(-3, 3), a)
        else:
            qc.t(a)
    out = ThreeQubitBlockPass().run(qc)
    assert _phase_distance(_unitary(out, n), _unitary(qc, n)) < 1e-9


def test_measurement_and_barrier_are_boundaries() -> None:
    qc = _controlled_power_circuit(4, 1)
    qc.measure([0], [0])
    qc = qc.compose(_controlled_power_circuit(4, 2), qubits=[0, 1, 2])
    pass_ = ThreeQubitBlockPass()
    out = pass_.run(qc)
    assert pass_.stats.blocks_replaced == 2
    names = [i.gate.name if i.gate else "measure" for i in out]
    assert names.count("measure") == 1
