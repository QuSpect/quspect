"""Tests for connectivity-aware Toffoli synthesis and three-qubit SABRE routing."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates.gate import Gate
from quspect.gates.standard import CCX
from quspect.transpiler import (
    SABRE,
    CompilationTarget,
    ConsolidateBlocksPass,
    HighLevelOptimizePass,
)
from quspect.transpiler._passes import DecomposePass, OptimizePass
from quspect.transpiler._toffoli import (
    ccx_candidates,
    ccx_path_decomposition,
    ccx_triangle_decomposition,
    synthesize_ccx_for_coupling,
)
from tests.synthesis._unitary import circuit_unitary

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _ccx_unitary() -> np.ndarray:
    qc = QuantumCircuit().append(CCX, (0, 1, 2))
    return circuit_unitary(qc, 3)


def _seq_unitary(ops: list[tuple[Gate, tuple[int, ...]]], n: int) -> np.ndarray:
    qc = QuantumCircuit()
    for gate, qs in ops:
        qc.append(gate, qs)
    return circuit_unitary(qc, n)


def _aligned_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Element-wise distance after aligning the phase on the largest entry."""
    k = int(np.argmax(np.abs(v)))
    phase = u.flat[k] / v.flat[k]
    return float(np.max(np.abs(u - phase * v)))


def _line(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _grid(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


def _permutation(sigma: list[int]) -> np.ndarray:
    """Big-endian matrix moving the state of wire ``l`` to wire ``sigma[l]``."""
    m = len(sigma)
    dim = 1 << m
    mat = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = 0
        for bit in range(m):
            if (x >> (m - 1 - bit)) & 1:
                y |= 1 << (m - 1 - sigma[bit])
        mat[y, x] = 1.0
    return mat


def _assert_routed_exact(
    circ: QuantumCircuit,
    routed: QuantumCircuit,
    initial: list[int],
    final: list[int],
    target: CompilationTarget,
) -> None:
    for ins in routed:
        assert len(ins.qubits) <= 2
        if len(ins.qubits) == 2:
            a, b = (initial[w] for w in ins.qubits)
            assert (min(a, b), max(a, b)) in target.coupling_graph
    m = len(initial)
    phys_to_wire = {p: w for w, p in enumerate(initial)}
    sigma = [phys_to_wire[final[w]] for w in range(m)]
    u = circuit_unitary(circ, m)
    r = circuit_unitary(routed, m)
    assert _aligned_distance(r, _permutation(sigma) @ u) < 1e-9


def _random_ccx_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(depth):
        qs = rng.sample(range(n), 3)
        r = rng.random()
        if r < 0.4:
            qc.ccx(*qs)
        elif r < 0.7:
            qc.cx(qs[0], qs[1])
        else:
            qc.h(qs[0]).t(qs[1]).rz(rng.uniform(-3, 3), qs[2])
    return qc


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("middle_pos", [0, 1, 2])
def test_path_decomposition_is_exact(perm: tuple[int, ...], middle_pos: int) -> None:
    c1, c2, t = perm
    ops = ccx_path_decomposition(c1, c2, t, perm[middle_pos])
    ref = circuit_unitary(QuantumCircuit().ccx(c1, c2, t), 3)
    # 无全局相位：逐元素直接比较。
    assert np.max(np.abs(_seq_unitary(ops, 3) - ref)) < 1e-12
    cx_pairs = {frozenset(qs) for g, qs in ops if g.name == "cx"}
    middle = perm[middle_pos]
    assert all(middle in pair for pair in cx_pairs)
    assert sum(1 for g, _ in ops if g.name == "cx") == 8


def test_path_decomposition_rejects_bad_middle() -> None:
    with pytest.raises(ValueError, match="middle"):
        ccx_path_decomposition(0, 1, 2, middle=5)
    with pytest.raises(ValueError, match="distinct"):
        ccx_path_decomposition(0, 0, 2, middle=0)


def test_triangle_decomposition_is_exact() -> None:
    ops = ccx_triangle_decomposition(0, 1, 2)
    assert _aligned_distance(_seq_unitary(ops, 3), _ccx_unitary()) < 1e-12


@pytest.mark.parametrize(
    "edges",
    [
        [(0, 1), (1, 2), (0, 2)],
        [(0, 1), (1, 2)],
        [(0, 2), (1, 2)],
        [(0, 1), (0, 2)],
    ],
)
def test_all_candidates_exact_and_on_edges(edges: list[tuple[int, int]]) -> None:
    edge_set = {frozenset(e) for e in edges}

    def adjacent(x: int, y: int) -> bool:
        return frozenset((x, y)) in edge_set

    ref = _ccx_unitary()
    for ops in ccx_candidates(0, 1, 2, adjacent):
        assert _aligned_distance(_seq_unitary(ops, 3), ref) < 1e-12
        for _gate, qs in ops:
            if len(qs) == 2:
                assert frozenset(qs) in edge_set


def test_synthesis_rejects_disconnected_qubits() -> None:
    with pytest.raises(ValueError, match="not connected"):
        synthesize_ccx_for_coupling(0, 1, 2, lambda x, y: {x, y} == {0, 1})


def test_context_picks_cheaper_variant() -> None:
    """MAJ (cx z,y · cx z,x · ccx) needs 7 two-qubit gates on a triangle."""

    def adjacent(x: int, y: int) -> bool:
        return True

    before = [(g, qs) for g, qs in _maj_prefix()]
    ops = synthesize_ccx_for_coupling(0, 1, 2, adjacent, before=before)
    qc = QuantumCircuit()
    for gate, qs in (*before, *ops):
        qc.append(gate, qs)
    out = ConsolidateBlocksPass().run(qc)
    assert sum(1 for i in out if len(i.qubits) == 2) == 7
    ref = QuantumCircuit().cx(2, 1).cx(2, 0).ccx(0, 1, 2)
    assert _aligned_distance(circuit_unitary(out, 3), circuit_unitary(ref, 3)) < 1e-9


def _maj_prefix() -> list[tuple[Gate, tuple[int, ...]]]:
    from quspect.gates.standard import CX

    return [(CX, (2, 1)), (CX, (2, 0))]


# ---------------------------------------------------------------------------
# SABRE with three-qubit Toffoli nodes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "edges",
    [
        _line(5),
        [*_line(5), (4, 0)],
        _grid(2, 3),
        list(itertools.combinations(range(5), 2)),
    ],
    ids=["line5", "ring5", "grid2x3", "full5"],
)
def test_sabre_routes_ccx_exactly(seed: int, edges: list[tuple[int, int]]) -> None:
    n_phys = max(max(e) for e in edges) + 1
    target = CompilationTarget.from_coupling_map(n_phys, edges)
    circ = _random_ccx_circuit(5, 14, seed)
    routed, initial, final = SABRE(target).solve(circ, seed=seed)
    _assert_routed_exact(circ, routed, initial, final, target)


def test_single_ccx_on_line_needs_no_swap() -> None:
    target = CompilationTarget.from_coupling_map(3, _line(3))
    circ = QuantumCircuit().ccx(0, 1, 2)
    routed, initial, final = SABRE(target).solve(circ, seed=0)
    assert not any(i.gate is not None and i.gate.name == "swap" for i in routed)
    assert sum(1 for i in routed if len(i.qubits) == 2) == 8
    _assert_routed_exact(circ, routed, initial, final, target)


def test_ccx_on_triangle_uses_six_cx() -> None:
    target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2), (0, 2)])
    circ = QuantumCircuit().ccx(0, 1, 2)
    routed, initial, final = SABRE(target).solve(circ, seed=0)
    assert sum(1 for i in routed if len(i.qubits) == 2) == 6
    _assert_routed_exact(circ, routed, initial, final, target)


def test_ccx_with_fixed_layout_and_release_valve() -> None:
    """Far-apart Toffoli qubits with a tiny release valve still route exactly."""
    target = CompilationTarget.from_coupling_map(6, _line(6))
    circ = QuantumCircuit().h(0).ccx(0, 5, 2).cx(1, 4).ccx(4, 1, 3)
    sabre = SABRE(target, release_valve_threshold=1)
    routed, initial, final = sabre.solve(circ, initial_mapping=list(range(6)))
    _assert_routed_exact(circ, routed, initial, final, target)


def test_ccx_with_measurements_and_swaps() -> None:
    target = CompilationTarget.from_coupling_map(4, _line(4))
    circ = QuantumCircuit().h(0).swap(0, 3).ccx(0, 1, 3).cx(3, 2).ccx(2, 0, 1)
    circ.measure([0, 1, 2, 3], [0, 1, 2, 3])
    routed, initial, final = SABRE(target).solve(circ, seed=3)
    n_meas = sum(1 for i in routed if i.is_measure)
    assert n_meas == 4
    gates_only = QuantumCircuit()
    for ins in circ:
        if ins.is_gate and ins.gate is not None:
            gates_only.append(ins.gate, ins.qubits)
    routed_gates = QuantumCircuit()
    for ins in routed:
        if ins.is_gate and ins.gate is not None:
            routed_gates.append(ins.gate, ins.qubits)
    _assert_routed_exact(gates_only, routed_gates, initial, final, target)


def test_other_three_qubit_gates_still_rejected() -> None:
    target = CompilationTarget.from_coupling_map(3, _line(3))
    circ = QuantumCircuit().append(Gate(name="cswap", num_qubits=3), (0, 1, 2))
    with pytest.raises(ValueError, match=r"cannot route 3-qubit gate 'cswap'"):
        SABRE(target).solve(circ, seed=0)


def test_circuit_without_ccx_takes_two_qubit_router() -> None:
    """No ccx: results identical to routing the same circuit twice (smoke)."""
    target = CompilationTarget.from_coupling_map(5, _line(5))
    circ = QuantumCircuit().cx(0, 4).cx(1, 3).cx(2, 0)
    a = SABRE(target).solve(circ, seed=5)
    b = SABRE(target).solve(circ, seed=5)
    assert [i.qubits for i in a[0]] == [i.qubits for i in b[0]]
    assert a[1:] == b[1:]


# ---------------------------------------------------------------------------
# Pipeline: Cuccaro adder
# ---------------------------------------------------------------------------


def _cuccaro(n: int) -> QuantumCircuit:
    k = (n - 2) // 2
    a = [2 * i + 1 for i in range(k)]
    b = [2 * i + 2 for i in range(k)]
    cin, cout = 0, 2 * k + 1
    qc = QuantumCircuit()
    for q in a[::2] + b[1::2]:
        qc.x(q)

    def maj(x: int, y: int, z: int) -> None:
        qc.cx(z, y).cx(z, x).ccx(x, y, z)

    def uma(x: int, y: int, z: int) -> None:
        qc.ccx(x, y, z).cx(z, x).cx(x, y)

    maj(cin, b[0], a[0])
    for i in range(1, k):
        maj(a[i - 1], b[i], a[i])
    qc.cx(a[k - 1], cout)
    for i in range(k - 1, 0, -1):
        uma(a[i - 1], b[i], a[i])
    uma(cin, b[0], a[0])
    return qc


def _pipeline(
    circ: QuantumCircuit, target: CompilationTarget, seed: int
) -> tuple[QuantumCircuit, list[int], list[int]]:
    decompose = DecomposePass(target)
    consolidate = ConsolidateBlocksPass(target)
    c = HighLevelOptimizePass().run(circ)
    c = consolidate.run(DecomposePass(target, keep={"ccx"}).run(c))
    routed, initial, final = SABRE(target).solve(c, seed=seed)
    c = consolidate.run(decompose.run(routed))
    c = decompose.run(OptimizePass(basis_gates=BASIS).run(c))
    c = decompose.run(consolidate.run(c))
    return c, initial, final


@pytest.mark.parametrize(
    ("edges", "max_2q"),
    [
        (_line(6), 60),
        ([(i, j) for i in range(6) for j in range(i + 1, 6)], 30),
        (_grid(2, 3), 60),
    ],
    ids=["line6", "full6", "grid2x3"],
)
def test_cuccaro_adder_pipeline(edges: list[tuple[int, int]], max_2q: int) -> None:
    target = CompilationTarget.from_coupling_map(6, edges, native_gates=BASIS)
    circ = _cuccaro(6)
    out, initial, final = _pipeline(circ, target, seed=1)
    assert {i.gate.name for i in out if i.gate is not None} <= BASIS
    assert sum(1 for i in out if len(i.qubits) == 2) <= max_2q
    _assert_routed_exact(circ, out, initial, final, target)


def test_decompose_pass_keep() -> None:
    target = CompilationTarget.from_coupling_map(3, _line(3), native_gates=BASIS)
    circ = QuantumCircuit().ccx(0, 1, 2).cx(0, 1)
    kept = DecomposePass(target, keep={"ccx"}).run(circ)
    assert [i.gate.name for i in kept if i.gate is not None].count("ccx") == 1
    assert "cx" not in {i.gate.name for i in kept if i.gate is not None}
    assert "ccx" not in {
        i.gate.name for i in DecomposePass(target).run(circ) if i.gate is not None
    }


def _random_mapping_gates(rng: random.Random, n: int, count: int) -> list:
    from quspect.gates.parameterized import RZ
    from quspect.gates.standard import CX, H, T
    from quspect.transpiler._mapping import _MappingGate

    gates = []
    for k in range(count):
        r = rng.random()
        if r < 0.15:
            a, b, c = rng.sample(range(n), 3)
            gates.append(_MappingGate(False, a, b, k, "ccx", None, None, c))
        elif r < 0.5:
            a, b = rng.sample(range(n), 2)
            gates.append(_MappingGate(True, a, b, k, "cx", None, CX))
        elif r < 0.55:
            gates.append(_MappingGate(False, rng.randrange(n), 0, k, "measure", 0))
        else:
            q = rng.randrange(n)
            gate = rng.choice([H, T, RZ.with_params(rng.choice([0.5, -0.0, 0.0]))])
            gates.append(_MappingGate(False, q, q, k, gate.name, None, gate))
    return gates


@pytest.mark.parametrize("seed", range(12))
def test_indexed_context_windows_match_the_plain_walk(seed: int) -> None:
    """The per-qubit position walk collects the same windows as the full walk."""
    from quspect.transpiler import _toffoli

    rng = random.Random(seed)
    n = 7
    adj = [set(range(n)) - {q} for q in range(n)]  # complete graph: any triple
    gates = _random_mapping_gates(rng, n, 160)
    seen: list[tuple[str, tuple, list]] = []
    real = _toffoli.synthesize_ccx_for_coupling

    def spy(c1, c2, t, adjacent, *, before=(), after=()):  # type: ignore[no-untyped-def]
        seen.append(("call", (c1, c2, t), [list(before), list(after)]))
        return real(c1, c2, t, adjacent, before=before, after=after)

    gate_of = {g.index: g.gate for g in gates}

    class _Circ:
        def __getitem__(self, i: int) -> object:
            return type("I", (), {"gate": gate_of[i]})()

    try:
        _toffoli.synthesize_ccx_for_coupling = spy
        indexed = _toffoli.expand_toffolis(list(gates), adj, _Circ())  # type: ignore[arg-type]
    finally:
        _toffoli.synthesize_ccx_for_coupling = real
    # 参考：在已输出的门上逐门向前、在输入上逐门向后遍历（原实现）
    from quspect.transpiler._mapping import _MappingGate

    def lookup(g: _MappingGate) -> object:
        return g.gate if g.gate is not None else gate_of[g.index]

    out: list[_MappingGate] = []
    expected = []
    for k, g in enumerate(gates):
        if g.q3 < 0:
            out.append(g)
            continue
        triple = (g.q1, g.q2, g.q3)
        before = _toffoli._context_window(
            out, range(len(out) - 1, -1, -1), triple, lookup
        )[::-1]
        after = _toffoli._context_window(
            gates, range(k + 1, len(gates)), triple, lookup
        )
        expected.append(("call", triple, [before, after]))
        for gate, qs in real(*triple, lambda x, y: True, before=before, after=after):
            q2 = qs[1] if len(qs) == 2 else qs[0]
            out.append(
                _MappingGate(len(qs) == 2, qs[0], q2, g.index, gate.name, None, gate)
            )
    assert seen == expected
    assert [(g.q1, g.q2, g.name, g.gate, g.index) for g in indexed] == [
        (g.q1, g.q2, g.name, g.gate, g.index) for g in out
    ]


@pytest.mark.parametrize("seed", range(6))
def test_variant_choice_cache_matches_uncached_choice(seed: int) -> None:
    """Relabelled contexts reuse the cached variant; it equals the full search."""
    from quspect.gates.parameterized import RZ
    from quspect.gates.standard import CX, H, T
    from quspect.transpiler import _toffoli

    rng = random.Random(100 + seed)
    line = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}

    def adjacent(x: int, y: int) -> bool:
        return (min(x, y), max(x, y)) in line or x == y

    pool = [H, T, CX, RZ.with_params(0.25), RZ.with_params(-0.0)]
    for _ in range(40):
        c1, c2, t = rng.choice([(0, 2, 1), (1, 3, 2), (2, 0, 1), (4, 1, 0)])
        triple = (c1, c2, t)

        def side(triple: tuple[int, int, int] = triple) -> list:
            out = []
            for _ in range(rng.randrange(4)):
                g = rng.choice(pool)
                qs = tuple(rng.sample(triple, 2)) if g is CX else (rng.choice(triple),)
                out.append((g, qs))
            return out

        before, after = side(), side()
        got = _toffoli.synthesize_ccx_for_coupling(
            c1, c2, t, adjacent, before=before, after=after
        )
        cands = ccx_candidates(c1, c2, t, adjacent)
        if before or after:
            keys = [
                (*_toffoli._consolidated_cost((*before, *ops, *after)), r)
                for r, ops in enumerate(cands)
            ]
            assert got == cands[min(keys)[2]]
        else:
            assert got == cands[0]
