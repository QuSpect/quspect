"""Tests for permutation-aware block synthesis and commutation-aware SABRE."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget
from quspect.transpiler._mapping import _build_routing_dag, _extract_gates

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _line(n: int, native: set[str] | None = None) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=native or set()
    )


def _grid(rows: int, cols: int) -> CompilationTarget:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return CompilationTarget.from_coupling_map(rows * cols, edges)


def _simulate(circuit: QuantumCircuit, state: np.ndarray, n: int) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    psi = state.reshape([2] * n)
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = np.asarray(gate_matrix(ins.gate), dtype=complex).reshape([2] * (2 * k))
        psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(ins.qubits))
    return psi.reshape(-1)


def _embed(state: np.ndarray, layout: list[int], num_physical: int) -> np.ndarray:
    n = len(layout)
    free = [p for p in range(num_physical) if p not in layout]
    zeros = np.zeros(2 ** (num_physical - n), dtype=complex)
    zeros[0] = 1.0
    tensor = np.kron(state, zeros).reshape([2] * num_physical)
    order = [
        layout.index(p) if p in layout else n + free.index(p)
        for p in range(num_physical)
    ]
    return np.transpose(tensor, order).reshape(-1)


def _assert_routed_equivalent(
    circ: QuantumCircuit,
    routed: QuantumCircuit,
    pi0: list[int],
    pi_final: list[int],
    target: CompilationTarget,
) -> None:
    """The routed wire-space circuit implements ``circ`` with the mappings."""
    n, big_n = circ.num_qubits, target.num_qubits
    physical = QuantumCircuit()
    for ins in routed:
        assert ins.gate is not None
        qs = tuple(pi0[w] for w in ins.qubits)
        if len(qs) == 2:
            assert tuple(sorted(qs)) in target.coupling_graph
        physical.append(ins.gate, qs)
    rng = np.random.default_rng(1)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    expected = _embed(_simulate(circ, psi, n), list(pi_final[:n]), big_n)
    got = _simulate(physical, _embed(psi, list(pi0[:n]), big_n), big_n)
    assert abs(abs(np.vdot(expected, got)) - 1.0) < 1e-9


def _commuting_circuit(n: int, size: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for _ in range(size):
        a, b, c = rng.sample(range(n), 3)
        r = rng.random()
        if r < 0.3:
            qc.cp(rng.uniform(-3, 3), a, b)
        elif r < 0.55:
            qc.cx(a, b)
        elif r < 0.65:
            qc.cz(a, b)
        elif r < 0.75:
            qc.rz(rng.uniform(-3, 3), a)
        elif r < 0.8:
            qc.x(a)
        elif r < 0.85:
            qc.h(a)
        elif r < 0.9:
            qc.ccx(a, b, c)
        else:
            qc.t(a)
    return qc


# ---------------------------------------------------------------------------
# Permutation-aware block synthesis
# ---------------------------------------------------------------------------


def _dcnot_block() -> QuantumCircuit:
    # CZ · (XH ⊗ H) · CZ: Weyl (π/4, π/4, 0), i.e. SWAP times a single CZ.
    qc = QuantumCircuit().h(0).x(0).h(1).cz(0, 1).x(0).h(0).h(1).cz(0, 1)
    return qc.h(0).h(1)


def test_permutation_synthesis_halves_dcnot_block() -> None:
    target = _line(2, BASIS)
    circ = _dcnot_block()
    plain, _, _ = SABRE(target).solve(circ, seed=0)
    routed, pi0, pi_final = SABRE(target, permutation_synthesis=True).solve(
        circ, seed=0
    )
    count = sum(1 for i in routed if len(i.qubits) == 2)
    assert sum(1 for i in plain if len(i.qubits) == 2) == 2
    assert count == 1
    assert pi_final[:2] == [pi0[1], pi0[0]]
    _assert_routed_equivalent(circ, routed, pi0, pi_final, target)


def test_permutation_synthesis_skipped_without_basis() -> None:
    target = _line(2)  # no native gates: no {cz, h, rz, x} synthesis
    routed, _, _ = SABRE(target, permutation_synthesis=True).solve(
        _dcnot_block(), seed=0
    )
    assert sum(1 for i in routed if len(i.qubits) == 2) == 2


def test_permutation_synthesis_relabels_later_gates() -> None:
    from quspect.ir.compilation import decompose_deep

    target = _line(4, BASIS)
    circ = _dcnot_block().cx(0, 2).cp(0.4, 1, 3).h(0).cx(1, 2)
    circ = decompose_deep(circ, supported_gates=BASIS)
    for seed in range(3):
        routed, pi0, pi_final = SABRE(target, permutation_synthesis=True).solve(
            circ, seed=seed
        )
        _assert_routed_equivalent(circ, routed, pi0, pi_final, target)


def test_cheaper_with_swap_rejects_non_improving_blocks() -> None:
    from quspect.transpiler._mapping import _cheaper_with_swap

    cz = np.diag([1, 1, 1, -1]).astype(complex)
    assert _cheaper_with_swap(cz @ cz, 2) is None  # identity: 0 CZ already
    swap = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
    ops = _cheaper_with_swap(swap @ cz, 2)  # SWAP · CZ (2 CZ) → CZ
    assert ops is not None
    assert sum(1 for name, _, _ in ops if name == "cz") == 1


# ---------------------------------------------------------------------------
# Commutation-aware routing DAG
# ---------------------------------------------------------------------------


def _dag(circ: QuantumCircuit, commutation: bool):  # noqa: ANN202
    gates, n = _extract_gates(circ)
    return _build_routing_dag(gates, n, circ, commutation=commutation)


def test_commuting_diagonal_gates_unordered() -> None:
    circ = QuantumCircuit().cp(0.3, 0, 1).rz(0.2, 1).cz(1, 2).cp(0.1, 0, 2)
    plain = _dag(circ, False)
    comm = _dag(circ, True)
    assert plain.pred == [[], [0], [0, 1]]
    assert comm.pred == [[], [], []]
    # the rz between the diagonal gates stays attached to the next node
    assert [g.name for g in comm.prefix[1]] == ["rz"]


def test_cx_sharing_target_or_control_commute() -> None:
    circ = QuantumCircuit().cx(0, 2).cx(1, 2).cx(0, 3).cx(1, 3)
    assert _dag(circ, True).pred == [[], [], [], []]
    # control of one gate on the other's target: no commutation
    circ = QuantumCircuit().cx(0, 1).cx(1, 2)
    assert _dag(circ, True).pred == [[], [0]]


def test_non_diagonal_single_qubit_gate_breaks_group() -> None:
    circ = QuantumCircuit().cz(0, 1).h(1).cz(1, 2).cz(0, 2)
    comm = _dag(circ, True)
    assert comm.pred[1] == [0]  # h(1) separates cz(1, 2) from cz(0, 1)
    # cz(0, 2) is diagonal on both of its qubits like everything before it
    assert comm.pred[2] == []


def test_group_member_after_group_depends_on_all_members() -> None:
    circ = QuantumCircuit().cz(0, 1).cz(0, 2).cz(0, 3).h(0).cz(0, 4)
    comm = _dag(circ, True)
    assert comm.pred[:3] == [[], [], []]
    assert comm.pred[3] == [0, 1, 2]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("bonus", [0.0, 0.3])
def test_commutation_routing_equivalent(seed: int, bonus: float) -> None:
    target = _grid(2, 3) if seed % 2 else _line(6)
    circ = _commuting_circuit(6, 40, seed)
    sabre = SABRE(target, commutation_aware=True, swap_fusion_bonus=bonus)
    routed, pi0, pi_final = sabre.solve(circ, seed=seed, layout_trials=3)
    _assert_routed_equivalent(circ, routed, pi0, pi_final, target)


@pytest.mark.parametrize("seed", range(4))
def test_route_candidates_commutation_equivalent(seed: int) -> None:
    target = _line(6)
    circ = _commuting_circuit(6, 30, 100 + seed)
    sabre = SABRE(target, commutation_aware=True, swap_fusion_bonus=0.3)
    for cand in sabre.route_candidates(circ, seed=seed, layout_trials=3, swap_trials=2):
        _assert_routed_equivalent(
            circ, cand.circuit, cand.initial_mapping, cand.final_mapping, target
        )


def test_commutation_shortens_qaoa_routing() -> None:
    rng = random.Random(3)
    n = 8
    circ = QuantumCircuit()
    for q in range(n):
        circ.h(q)
    for a in range(n):
        for b in range(a + 1, n):
            circ.cp(rng.uniform(-math.pi, math.pi), a, b)
    target = _line(n)
    plain, _, _ = SABRE(target).solve(circ, seed=1)
    comm, pi0, pi_final = SABRE(target, commutation_aware=True).solve(circ, seed=1)
    swaps = [sum(1 for i in c if i.gate.name == "swap") for c in (plain, comm)]
    assert swaps[1] <= swaps[0]
    _assert_routed_equivalent(circ, comm, pi0, pi_final, target)


def test_invalid_fusion_bonus_rejected() -> None:
    with pytest.raises(ValueError, match="swap_fusion_bonus"):
        SABRE(_line(3), swap_fusion_bonus=1.0)


def test_commutation_gain() -> None:
    from quspect.transpiler._transpile import _commutation_gain

    chain = QuantumCircuit().h(0).cx(0, 1).h(1).cx(1, 2).h(2).cx(2, 3)
    assert _commutation_gain(chain) == 1.0
    star = QuantumCircuit().cp(0.1, 0, 1).cp(0.2, 0, 2).cp(0.3, 0, 3)
    assert _commutation_gain(star) == pytest.approx(1 / 3)
    assert _commutation_gain(QuantumCircuit()) == 1.0
