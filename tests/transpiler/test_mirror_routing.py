"""Mirror-gate-aware routing (MIRAGE) and block routing nodes.

* :func:`_annotate_mirror_costs` gives every pair block its exact mirror cost
  ``cz(U · SWAP) - cz(U)`` on its last (forward) / first (backward) node;
* ``block_nodes`` merges consecutive same-pair gates into one routing node
  without changing the emitted gate sequence;
* mirror swaps merge with their block after routing, the routed circuits stay
  exact (phase-aligned element-wise error ``<= 1e-9`` on the emitted gates)
  and the option off reproduces plain SABRE.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget, transpile
from quspect.transpiler import _transpile as transpile_module
from quspect.transpiler._mapping import (
    _annotate_mirror_costs,
    _build_routing_dag,
    _extract_gates,
)
from quspect.transpiler._routing import route_dag

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _line(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASIS)
    )


def _su4(qc: QuantumCircuit, a: int, b: int, rng: random.Random) -> None:
    """A generic two-qubit block: three CZ layers with random local rotations."""
    for _ in range(3):
        for q in (a, b):
            qc.rz(rng.uniform(-math.pi, math.pi), q)
            qc.h(q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        qc.cz(a, b)
    for q in (a, b):
        qc.rz(rng.uniform(-math.pi, math.pi), q)
        qc.h(q)


def _su4_circuit(n: int, blocks: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(blocks):
        a, b = rng.sample(range(n), 2)
        _su4(qc, a, b, rng)
    return qc


def _apply(circuit: QuantumCircuit, state: np.ndarray, n: int) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    psi = state.reshape([2] * n)
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
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


def _assert_exact(circ: QuantumCircuit, result: Any, target: CompilationTarget) -> None:
    """Basis, coupling and phase-aligned element-wise equivalence (linear 1e-9)."""
    for instr in result.circuit:
        assert instr.gate is not None and instr.gate.name in BASIS
        if len(instr.qubits) == 2:
            assert tuple(sorted(instr.qubits)) in target.coupling_graph
    n, big_n = circ.num_qubits, target.num_qubits
    rng = np.random.default_rng(1)
    for _ in range(2):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi /= np.linalg.norm(psi)
        expected = _embed(_apply(circ, psi, n), result.final_layout, big_n)
        got = _apply(result.circuit, _embed(psi, result.initial_layout, big_n), big_n)
        overlap = np.vdot(got, expected)
        aligned = got * (overlap / abs(overlap))
        assert np.max(np.abs(aligned - expected)) < 1e-9


# =============================================================================
# Mirror costs and block nodes
# =============================================================================


@pytest.mark.parametrize("merge", [False, True])
def test_mirror_costs_from_weyl_coordinates(merge: bool) -> None:
    qc = QuantumCircuit()
    _su4(qc, 0, 1, random.Random(3))  # generic SU(4): U·SWAP also 3 CZ
    qc.cz(1, 2)  # CZ·SWAP ~ iSWAP: 2 CZ instead of 1
    qc.cp(0.3, 2, 3)  # CP(θ)·SWAP: 3 CZ instead of 2
    qc.h(3).cz(2, 3)
    gates, n = _extract_gates(qc)
    dag = _build_routing_dag(gates, n, qc, merge_pairs=merge)
    _annotate_mirror_costs(dag, gates, qc)
    pairs = [(a, b) for a, b in zip(dag.nq1, dag.nq2, strict=True)]
    if merge:
        assert pairs == [(0, 1), (1, 2), (2, 3)]
        assert dag.mirror_fwd == [0, 1, 1]
        assert dag.mirror_bwd == [0, 1, 1]
    else:
        assert pairs == [(0, 1)] * 3 + [(1, 2), (2, 3), (2, 3)]
        assert dag.mirror_fwd == [-1, -1, 0, 1, -1, 1]
        assert dag.mirror_bwd == [0, -1, -1, 1, 1, -1]


def test_block_nodes_keep_emitted_gates() -> None:
    qc = _su4_circuit(6, 25, 7).measure_all()
    full = CompilationTarget.from_coupling_map(
        6, [(i, j) for i in range(6) for j in range(i + 1, 6)], native_gates=BASIS
    )
    plain = SABRE(full).solve(qc, seed=1)
    merged = SABRE(full, block_nodes=True).solve(qc, seed=1)
    assert [(i.gate, i.qubits, i.clbits) for i in plain[0]] == [
        (i.gate, i.qubits, i.clbits) for i in merged[0]
    ]
    assert plain[1:] == merged[1:]
    gates, n = _extract_gates(qc)
    assert len(_build_routing_dag(gates, n, qc, merge_pairs=True).nodes) < len(
        _build_routing_dag(gates, n, qc).nodes
    )


def test_block_nodes_ignored_on_commutation_dag() -> None:
    qc = QuantumCircuit().cz(0, 1).cz(0, 1).cp(0.2, 1, 2)
    gates, n = _extract_gates(qc)
    a = _build_routing_dag(gates, n, qc, commutation=True)
    b = _build_routing_dag(gates, n, qc, commutation=True, merge_pairs=True)
    assert (a.nq1, a.nq2, a.pred) == (b.nq1, b.nq2, b.pred)


# =============================================================================
# Routing with mirrors
# =============================================================================


def test_mirror_routing_uses_free_swaps() -> None:
    target = _line(6)
    qc = _su4_circuit(6, 40, 11)
    sabre = SABRE(
        target, permutation_synthesis=True, block_nodes=True, mirror_gates=True
    )
    prep = sabre._prepare(qc, True)
    layout = list(range(6))
    res = route_dag(sabre, layout, prep.dag, forward=True, rng=random.Random(0))
    assert res.num_mirrors > 0
    # mirrors are swap events whose CZ cost is that of the merged block (0–3)
    assert res.swap_cost < 3 * res.num_swaps
    plain = SABRE(target, permutation_synthesis=True)
    ref = route_dag(
        plain, layout, plain._prepare(qc, True).dag, forward=True, rng=random.Random(0)
    )
    assert ref.num_mirrors == 0
    assert ref.swap_cost == 3 * ref.num_swaps


def test_mirror_option_off_is_plain_sabre() -> None:
    target = _line(5)
    qc = _su4_circuit(5, 20, 2)
    ref = SABRE(target).solve(qc, seed=4, layout_trials=2)
    # mirrors explicitly off; mirrors on a commutation DAG are ignored
    off = SABRE(target, mirror_gates=False).solve(qc, seed=4, layout_trials=2)
    assert [(i.gate, i.qubits) for i in ref[0]] == [(i.gate, i.qubits) for i in off[0]]
    comm = SABRE(target, commutation_aware=True).solve(qc, seed=4, layout_trials=2)
    comm_m = SABRE(target, commutation_aware=True, mirror_gates=True).solve(
        qc, seed=4, layout_trials=2
    )
    assert [(i.gate, i.qubits) for i in comm[0]] == [
        (i.gate, i.qubits) for i in comm_m[0]
    ]
    assert comm[1:] == comm_m[1:]


def test_mirror_parameters_validated() -> None:
    with pytest.raises(ValueError, match="mirror_threshold"):
        SABRE(_line(3), mirror_threshold=-0.1)
    with pytest.raises(ValueError, match="mirror_cz_penalty"):
        SABRE(_line(3), mirror_cz_penalty=-1.0)


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mirrored_blocks_resynthesized_exactly(level: int, seed: int) -> None:
    """Mirror SWAPs merge into ``U · SWAP`` (3 CZ) and stay exact."""
    target = _line(7)
    qc = _su4_circuit(7, 30, 100 + seed)
    kwargs: dict[str, Any] = {"optimization_level": level, "seed": seed}
    if level == 3:
        kwargs["work_limit"] = 20_000
    res = transpile(qc, target, **kwargs)
    _assert_exact(qc, res, target)
    # every SU(4) block costs 3 CZ; each routing SWAP at most 3 more
    stats = res.stats
    assert stats.two_qubit_gates <= 3 * 30 + 3 * stats.num_swaps
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(transpile_module, "_MIRROR_LEVEL", 9)
        old = transpile(qc, target, **kwargs)
    _assert_exact(qc, old, target)
