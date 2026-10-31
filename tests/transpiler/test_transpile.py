"""Tests for :func:`quspect.transpiler.transpile` and SABRE candidate routing."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import (
    SABRE,
    CompilationTarget,
    RoutingCandidate,
    TranspileResult,
    transpile,
)
from quspect.transpiler._mapping import _CandidatePlan, estimate_two_qubit_cost

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _grid(rows: int, cols: int) -> CompilationTarget:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return CompilationTarget.from_coupling_map(
        rows * cols, edges, native_gates=set(BASIS)
    )


def _random_circuit(n: int, num_2q: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for _ in range(num_2q):
        a, b = rng.sample(range(n), 2)
        kind = rng.randrange(3)
        if kind == 0:
            qc.cx(a, b)
        elif kind == 1:
            qc.cp(rng.uniform(0, math.pi), a, b)
        else:
            qc.cz(a, b)
        qc.rz(rng.uniform(0, math.pi), a)
    return qc


def _qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit()
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(math.pi / 2 ** (j - i), j, i)
    return qc


def _ccx_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    """Random circuit mixing Toffolis (some cancelling pairs), CX and 1q gates."""
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for _ in range(depth):
        qs = rng.sample(range(n), 3)
        r = rng.random()
        if r < 0.35:
            qc.ccx(*qs)
        elif r < 0.45:
            qc.ccx(*qs).t(qs[0]).ccx(qs[1], qs[0], qs[2])
        elif r < 0.75:
            qc.cx(qs[0], qs[1])
        else:
            qc.t(qs[0]).rz(rng.uniform(-3, 3), qs[2])
    return qc


def _cuccaro(n: int) -> QuantumCircuit:
    k = (n - 2) // 2
    a = [2 * i + 1 for i in range(k)]
    b = [2 * i + 2 for i in range(k)]
    cin, cout = 0, 2 * k + 1
    qc = QuantumCircuit()
    for q in a[::2] + b[1::2]:
        qc.x(q)
    qc.cx(b[0], a[0]).cx(a[0], cin).ccx(cin, b[0], a[0])
    for i in range(1, k):
        qc.cx(a[i], b[i]).cx(a[i], a[i - 1]).ccx(a[i - 1], b[i], a[i])
    qc.cx(a[k - 1], cout)
    for i in range(k - 1, 0, -1):
        qc.ccx(a[i - 1], b[i], a[i]).cx(a[i], a[i - 1]).cx(a[i - 1], b[i])
    qc.ccx(cin, b[0], a[0]).cx(a[0], cin).cx(cin, b[0])
    return qc


def _embed(state: np.ndarray, layout: list[int], num_physical: int) -> np.ndarray:
    """Put logical qubit ``q`` of ``state`` on physical ``layout[q]``, others |0>."""
    n = len(layout)
    free = [p for p in range(num_physical) if p not in layout]
    zeros = np.zeros(2 ** (num_physical - n), dtype=complex)
    zeros[0] = 1.0
    tensor = np.kron(state, zeros).reshape([2] * num_physical)
    # axis k of ``tensor``: logical k (k < n), else free physical free[k - n]
    order = [
        layout.index(p) if p in layout else n + free.index(p)
        for p in range(num_physical)
    ]
    return np.transpose(tensor, order).reshape(-1)


def _simulate(circuit: QuantumCircuit, state: np.ndarray, n: int) -> np.ndarray:
    """Apply ``circuit`` to a big-endian ``n``-qubit state vector."""
    from quspect.gates._matrix_utils import gate_matrix

    psi = state.reshape([2] * n)
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
        psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(ins.qubits))
    return psi.reshape(-1)


def _assert_equivalent(
    circ: QuantumCircuit, result: TranspileResult, target: CompilationTarget
) -> None:
    """Basis, coupling and state-vector equivalence of a transpile result."""
    for instr in result.circuit:
        assert instr.gate is not None
        assert instr.gate.name in BASIS
        if len(instr.qubits) == 2:
            assert tuple(sorted(instr.qubits)) in target.coupling_graph
    n, big_n = circ.num_qubits, target.num_qubits
    rng = np.random.default_rng(0)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    expected = _embed(_simulate(circ, psi, n), result.final_layout, big_n)
    got = _simulate(result.circuit, _embed(psi, result.initial_layout, big_n), big_n)
    assert abs(abs(np.vdot(expected, got)) - 1.0) < 1e-9


def _two_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if i.is_gate and len(i.qubits) == 2)


# ---------------------------------------------------------------------------
# transpile()
# ---------------------------------------------------------------------------


def test_invalid_level_rejected() -> None:
    with pytest.raises(ValueError, match="optimization_level"):
        transpile(QuantumCircuit().h(0), _grid(2, 2), optimization_level=4)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_output_in_basis_and_on_edges(level: int) -> None:
    target = _grid(3, 4)
    result = transpile(_random_circuit(10, 40, 1), target, optimization_level=level)
    for instr in result.circuit:
        assert instr.gate is not None
        assert instr.gate.name in BASIS
        if len(instr.qubits) == 2:
            assert tuple(sorted(instr.qubits)) in target.coupling_graph
    assert result.stats.two_qubit_gates == _two_qubit(result.circuit)
    assert result.stats.optimization_level == level


@pytest.mark.parametrize("level", [2, 3])
def test_deterministic_with_seed(level: int) -> None:
    target = _grid(3, 4)
    circ = _random_circuit(11, 50, 4)
    a = transpile(circ, target, optimization_level=level, seed=9)
    b = transpile(circ, target, optimization_level=level, seed=9)
    assert [(i.gate, i.qubits) for i in a.circuit] == [
        (i.gate, i.qubits) for i in b.circuit
    ]
    assert a.initial_layout == b.initial_layout
    assert a.final_layout == b.final_layout
    assert a.stats.selected == b.stats.selected


@pytest.mark.parametrize(
    ("circ", "rows", "cols"),
    [
        (_random_circuit(9, 45, 0), 3, 3),
        (_random_circuit(12, 60, 1), 3, 4),
        (_random_circuit(12, 60, 2), 3, 4),
        (_qft(8), 3, 3),
        (_qft(10), 3, 4),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_level3_never_worse_than_level2(
    circ: QuantumCircuit, rows: int, cols: int, seed: int
) -> None:
    target = _grid(rows, cols)
    o2 = transpile(circ, target, optimization_level=2, seed=seed)
    o3 = transpile(circ, target, optimization_level=3, seed=seed)
    assert o3.stats.two_qubit_gates <= o2.stats.two_qubit_gates
    # The level-2 result is among the fully evaluated level-3 candidates.
    o2_score = o2.stats.candidate_scores[0][1:]
    assert any(score[1:] == o2_score for score in o3.stats.candidate_scores)
    assert o3.stats.num_candidates >= 1
    assert o3.stats.num_evaluated >= 1


@pytest.mark.parametrize("kind", ["random", "toffoli"])
def test_level2_matches_manual_pipeline(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quspect.transpiler._transpile as transpile_module
    from quspect.ir.compilation import decompose_deep
    from quspect.transpiler import (
        ConsolidateBlocksPass,
        HighLevelOptimizePass,
        PhaseFoldingPass,
    )
    from quspect.transpiler._gauge import minimize_single_qubit_gates
    from quspect.transpiler._passes import DecomposePass, OptimizePass
    from quspect.transpiler._schedule import schedule_for_depth
    from quspect.transpiler._transpile import _depth, _depth_key

    # the SWAP-face alternatives of the final stage are tested in test_swap_face
    monkeypatch.setattr(transpile_module, "_SWAP_POST", False)
    monkeypatch.setattr(transpile_module, "_SWAP_LATE", False)
    # the LNS refinement of the routing is tested in test_route_lns
    monkeypatch.setattr(transpile_module, "_LNS_LEVEL", 4)
    target = _grid(3, 3)
    circ = _random_circuit(9, 40, 3) if kind == "random" else _ccx_circuit(9, 30, 3)
    dec = DecomposePass(target)
    con = ConsolidateBlocksPass(target)
    c = HighLevelOptimizePass().run(circ)
    # phase folding on the CX level, then CX cancellation
    folder = PhaseFoldingPass()
    folded = folder.run(decompose_deep(c, supported_gates=BASIS | {"cx", "ccx"}))
    assert folder.stats.rotations_out < folder.stats.rotations_in
    c = HighLevelOptimizePass().run(folded)
    c = con.run(DecomposePass(target, keep={"ccx"}).run(c))
    c = DecomposePass(target, keep={"ccx"}).run(c)
    # level 2 routes on whole two-qubit blocks with mirror gates (MIRAGE)
    routed, pi0, pi_final = SABRE(
        target,
        permutation_synthesis=True,
        block_nodes=True,
        mirror_gates=True,
        mirror_threshold=0.25,
        mirror_cz_penalty=0.5,
        swap_fusion_bonus=0.2,
        dilation_layouts=1,
    ).solve(c, iter_num=3, seed=5)
    c = con.run(dec.run(routed))
    c = dec.run(OptimizePass(basis_gates=BASIS).run(c))
    c = dec.run(con.run(c))
    # final single-qubit minimization of the selected result
    # and the depth-oriented reordering of commuting gates, also tried before
    # the minimization when that lowers the two-qubit depth
    one = frozenset(BASIS - {"cz"})
    out = schedule_for_depth(minimize_single_qubit_gates(c, one))
    early = schedule_for_depth(c)
    if early is not c and _depth(early, two_qubit_only=True) < _depth(
        out, two_qubit_only=True
    ):
        alt = schedule_for_depth(minimize_single_qubit_gates(early, one))
        if _depth_key(alt) < _depth_key(out):
            out = alt
    c = out
    result = transpile(circ, target, optimization_level=2, seed=5)
    assert [(i.gate, i.qubits) for i in result.wire_circuit] == [
        (i.gate, i.qubits) for i in c
    ]
    assert result.wire_layout == pi0
    assert result.final_layout == pi_final[: circ.num_qubits]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("circ", "rows", "cols"),
    [
        (_ccx_circuit(6, 20, 1), 2, 3),
        (_ccx_circuit(7, 25, 2), 2, 4),
        (_cuccaro(6), 2, 3),
        (_cuccaro(8), 2, 4),
    ],
)
def test_toffoli_circuits_equivalent(
    circ: QuantumCircuit, rows: int, cols: int, level: int
) -> None:
    target = _grid(rows, cols)
    result = transpile(circ, target, optimization_level=level, seed=0)
    _assert_equivalent(circ, result, target)
    assert result.stats.two_qubit_gates == _two_qubit(result.circuit)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("circ", [_ccx_circuit(9, 40, 4), _cuccaro(10)])
def test_toffoli_level3_never_worse_than_level2(
    circ: QuantumCircuit, seed: int
) -> None:
    target = _grid(3, 4)
    o2 = transpile(circ, target, optimization_level=2, seed=seed)
    o3 = transpile(circ, target, optimization_level=3, seed=seed)
    assert o3.stats.two_qubit_gates <= o2.stats.two_qubit_gates
    _assert_equivalent(circ, o3, target)


def test_toffoli_routed_as_unit() -> None:
    # A Toffoli on a line costs 8 CX when routed whole; its 6-CX textbook
    # decomposition needs a SWAP (6 + 3 = 9) on the line.
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (1, 2)], native_gates=set(BASIS)
    )
    circ = QuantumCircuit().ccx(0, 1, 2)
    for level in (0, 1, 2, 3):
        result = transpile(circ, target, optimization_level=level, seed=0)
        assert result.stats.num_swaps == 0
        assert result.stats.two_qubit_gates <= 8
        _assert_equivalent(circ, result, target)


def test_cancelling_toffolis_removed() -> None:
    target = _grid(2, 2)
    circ = QuantumCircuit().ccx(0, 1, 2).t(0).ccx(1, 0, 2).cx(2, 3)
    result = transpile(circ, target, optimization_level=2, seed=0)
    assert result.stats.two_qubit_gates == 1
    _assert_equivalent(circ, result, target)


def test_exact_layout_short_circuits_search() -> None:
    target = _grid(3, 3)
    ghz = QuantumCircuit().h(0)
    for q in range(8):
        ghz.cx(q, q + 1)
    result = transpile(ghz, target, optimization_level=3, seed=0)
    assert result.stats.selected == "exact"
    assert result.stats.num_swaps == 0
    assert result.stats.two_qubit_gates == 8


def test_level3_budget_respected() -> None:
    target = _grid(3, 4)
    circ = _random_circuit(12, 60, 5)
    result = transpile(
        circ,
        target,
        optimization_level=3,
        seed=0,
        layout_trials=3,
        swap_trials=2,
        max_evaluations=1,
    )
    stats = result.stats
    assert (stats.layout_trials, stats.swap_trials) == (3, 2)
    # trivial and low-dilation (unrefined + refined) + 3 random layouts,
    # 2 routings each, for the mirror, the block-only and the
    # commutation-aware candidate family
    assert stats.num_candidates <= 3 * (3 + 2 + 2) * 2
    # best candidate + level-2 default + best commutation-aware candidate,
    # each possibly with its LNS refinement (plus the light refinement of the
    # default when the large-window sweep improved it further)
    assert stats.num_evaluated <= 2 * 3 + 1
    assert len(stats.candidate_scores) == stats.num_evaluated


def test_work_limit_shrinks_search() -> None:
    target = _grid(3, 4)
    circ = _random_circuit(12, 80, 6)
    big = transpile(circ, target, optimization_level=3, seed=0)
    small = transpile(circ, target, optimization_level=3, seed=0, work_limit=1)
    # The level-2 reference trials (trivial x2 + low-dilation x2 + 6 random)
    # always run, and the block-only and commutation-aware searches their
    # first trial (trivial x2 + low-dilation x2 + 1 random); nothing beyond
    # them once the budget is spent.
    assert small.stats.num_candidates <= 6 + 4 + 5 + 5
    assert big.stats.num_candidates > small.stats.num_candidates
    o2 = transpile(circ, target, optimization_level=2, seed=0)
    assert small.stats.two_qubit_gates <= o2.stats.two_qubit_gates
    again = transpile(circ, target, optimization_level=3, seed=0, work_limit=1)
    assert [(i.gate, i.qubits) for i in again.circuit] == [
        (i.gate, i.qubits) for i in small.circuit
    ]


def test_time_limit_stops_search() -> None:
    target = _grid(3, 4)
    circ = _random_circuit(12, 80, 6)
    result = transpile(
        circ, target, optimization_level=3, seed=0, layout_trials=20, time_limit=0.0
    )
    # Only the level-2 reference trials (and the first block-only and
    # commutation-aware trials) run once the (zero) budget is spent.
    assert result.stats.num_candidates <= 6 + 4 + 5 + 5
    o2 = transpile(circ, target, optimization_level=2, seed=0)
    assert result.stats.two_qubit_gates <= o2.stats.two_qubit_gates


def test_no_native_gates_target() -> None:
    target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
    circ = QuantumCircuit().h(0).cx(0, 2).rz(0.3, 2)
    result = transpile(circ, target, optimization_level=3, seed=0)
    for instr in result.circuit:
        if len(instr.qubits) == 2:
            assert tuple(sorted(instr.qubits)) in target.coupling_graph


def test_empty_circuit() -> None:
    target = _grid(2, 2)
    result = transpile(QuantumCircuit(), target, optimization_level=3)
    assert len(result.circuit) == 0
    assert result.initial_layout == []


# ---------------------------------------------------------------------------
# SABRE.route_candidates
# ---------------------------------------------------------------------------


def test_route_candidates_default_matches_solve() -> None:
    target = _grid(3, 4)
    circ = _random_circuit(12, 60, 7)
    sabre = SABRE(target)
    routed, pi0, pi_final = sabre.solve(circ, seed=3)
    cands = sabre.route_candidates(circ, seed=3, swap_trials=3)
    defaults = [c for c in cands if c.is_default]
    assert len(defaults) == 1
    d = defaults[0]
    assert [(i.gate, i.qubits) for i in d.circuit] == [
        (i.gate, i.qubits) for i in routed
    ]
    assert d.initial_mapping == pi0
    assert d.final_mapping == pi_final


def test_route_candidates_fixed_layout_default_matches_solve() -> None:
    target = _grid(3, 3)
    circ = _random_circuit(9, 40, 8)
    layout = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    sabre = SABRE(target)
    routed, pi0, _ = sabre.solve(circ, seed=2, initial_mapping=layout)
    cands = sabre.route_candidates(circ, seed=2, initial_mapping=layout, swap_trials=4)
    d = next(c for c in cands if c.is_default)
    assert d.initial_mapping == pi0
    assert [(i.gate, i.qubits) for i in d.circuit] == [
        (i.gate, i.qubits) for i in routed
    ]
    assert all(c.initial_mapping[:9] == layout for c in cands)


def test_route_candidates_sorted_and_top_k() -> None:
    target = _grid(3, 4)
    circ = _random_circuit(12, 60, 9)
    sabre = SABRE(target)
    cands = sabre.route_candidates(circ, seed=1, layout_trials=6, swap_trials=3)
    keys = [(c.estimated_two_qubit_gates, c.num_swaps, c.depth) for c in cands]
    assert keys == sorted(keys)
    assert all(isinstance(c, RoutingCandidate) for c in cands)
    top = sabre.route_candidates(circ, seed=1, layout_trials=6, swap_trials=3, top_k=2)
    assert 2 <= len(top) <= 3
    assert any(c.is_default for c in top)
    assert top[0].estimated_two_qubit_gates == cands[0].estimated_two_qubit_gates


def test_route_candidates_argument_validation() -> None:
    sabre = SABRE(_grid(2, 2))
    circ = QuantumCircuit().cx(0, 3)
    with pytest.raises(ValueError, match="swap_trials"):
        sabre.route_candidates(circ, swap_trials=0)
    with pytest.raises(ValueError, match="top_k"):
        sabre.route_candidates(circ, top_k=0)
    with pytest.raises(ValueError, match="layout_trials"):
        sabre.route_candidates(circ, layout_trials=0)


def test_route_candidates_with_toffolis() -> None:
    target = _grid(3, 3)
    circ = _ccx_circuit(9, 30, 5)
    sabre = SABRE(target)
    routed, pi0, pi_final = sabre.solve(circ, seed=2)
    cands = sabre.route_candidates(circ, seed=2, swap_trials=2)
    d = next(c for c in cands if c.is_default)
    assert [(i.gate, i.qubits) for i in d.circuit] == [
        (i.gate, i.qubits) for i in routed
    ]
    assert (d.initial_mapping, d.final_mapping) == (pi0, pi_final)
    for cand in cands:
        assert all(len(i.qubits) <= 2 for i in cand.circuit)
        # scored after Toffoli synthesis: at most the raw two-qubit count
        assert cand.estimated_two_qubit_gates <= _two_qubit(cand.circuit) + 2 * sum(
            1 for i in cand.circuit if i.gate is not None and i.gate.name == "swap"
        )


def test_estimate_two_qubit_cost_blocks() -> None:
    from quspect.transpiler._mapping import _MappingGate

    def g2(a: int, b: int, name: str = "cz") -> _MappingGate:
        return _MappingGate(True, a, b, 0, name)

    # SWAP · CZ on one pair -> 2; lone SWAP -> 3; CZ on other pair -> 1
    gates = [g2(0, 1), g2(0, 1, "swap"), g2(1, 2, "swap"), g2(2, 3)]
    assert estimate_two_qubit_cost(gates, 4) == 2 + 3 + 1
    # a measurement closes the block
    meas = _MappingGate(False, 0, 0, 0, "measure", clbit=0)
    assert estimate_two_qubit_cost([g2(0, 1), meas, g2(0, 1, "swap")], 2) == 1 + 3
    # long runs saturate at 3
    assert estimate_two_qubit_cost([g2(0, 1)] * 5, 2) == 3
    # an unexpanded ccx closes the runs on its qubits and costs 6
    ccx = _MappingGate(False, 0, 1, 0, "ccx", q3=2)
    assert estimate_two_qubit_cost([g2(0, 1), ccx, g2(0, 1)], 3) == 1 + 6 + 1


@pytest.mark.parametrize("seed", range(6))
def test_route_candidates_reference_trials_prefix(seed: int) -> None:
    target = _grid(3, 4)
    circ = _random_circuit(12, 60, 10 + seed)
    sabre = SABRE(target)
    for ref in (1, 3, 6):
        routed, pi0, pi_final = sabre.solve(circ, seed=seed, layout_trials=ref)
        cands = sabre.route_candidates(
            circ, seed=seed, layout_trials=15, reference_trials=ref, work_limit=1
        )
        d = next(c for c in cands if c.is_default)
        assert [(i.gate, i.qubits) for i in d.circuit] == [
            (i.gate, i.qubits) for i in routed
        ]
        assert (d.initial_mapping, d.final_mapping) == (pi0, pi_final)
    with pytest.raises(ValueError, match="reference_trials"):
        sabre.route_candidates(circ, layout_trials=3, reference_trials=4)


# ---------------------------------------------------------------------------
# approximation_degree
# ---------------------------------------------------------------------------


def _weak_qft(n: int) -> QuantumCircuit:
    """QFT-like circuit whose angles go down to ~1e-7 on few qubits."""
    qc = QuantumCircuit()
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(math.pi / 2 ** (4 * (j - i) + 2), j, i)
    qc.crz(1e-6, 0, n - 1).cry(3e-7, 1, 2).cx(0, 1)
    return qc


def test_approximation_degree_validation() -> None:
    from quspect.transpiler import approximation_tolerance

    target = _grid(1, 3)
    for bad in (-0.1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="approximation_degree"):
            transpile(QuantumCircuit().h(0), target, approximation_degree=bad)  # type: ignore[arg-type]
    assert approximation_tolerance(None) == 0.0
    assert approximation_tolerance(1.0) == 1e-12
    assert approximation_tolerance(1) == 1e-12
    assert approximation_tolerance(0.99) == pytest.approx(0.01)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_approximation_degree_error_bound(level: int) -> None:
    n = 5
    circ = _weak_qft(n)
    target = CompilationTarget.from_coupling_map(
        n, [(i, j) for i in range(n) for j in range(i + 1, n)], native_gates=BASIS
    )
    exact = transpile(
        circ, target, optimization_level=level, seed=1, approximation_degree=None
    )
    assert exact.stats.approximated_operations == 0
    assert exact.stats.approximation_error == 0.0
    _assert_equivalent(circ, exact, target)
    for degree, fewer in ((1.0, True), (1 - 1e-6, True)):
        res = transpile(
            circ, target, optimization_level=level, seed=1, approximation_degree=degree
        )
        st = res.stats
        assert st.approximation_tolerance == max(1 - degree, 1e-12)
        assert st.approximated_operations > 0
        assert (_two_qubit(res.circuit) < _two_qubit(exact.circuit)) == fewer
        # 输出与输入的差（态矢层面）受 stats.approximation_error 约束
        rng = np.random.default_rng(level)
        for _ in range(3):
            psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
            psi /= np.linalg.norm(psi)
            expected = _embed(_simulate(circ, psi, n), res.final_layout, n)
            got = _simulate(res.circuit, _embed(psi, res.initial_layout, n), n)
            overlap = abs(np.vdot(expected, got))
            assert 1 - overlap <= st.approximation_error**2 / 2 + 1e-9
        assert st.approximation_error < 1e-4 if degree == 1.0 else True


def test_default_approximation_leaves_strong_circuits_alone() -> None:
    target = _grid(3, 3)
    circ = _random_circuit(9, 40, 3)
    a = transpile(circ, target, optimization_level=2, seed=5)
    b = transpile(circ, target, optimization_level=2, seed=5, approximation_degree=None)
    assert a.stats.approximated_operations == 0
    assert [(i.gate, i.qubits) for i in a.circuit] == [
        (i.gate, i.qubits) for i in b.circuit
    ]


def test_near_two_cz_block_reduced_after_routing() -> None:
    from quspect.synthesis import synthesize_two_qubit
    from quspect.synthesis.two_qubit import canonical_gate
    from tests.synthesis._unitary import random_unitary

    rng = np.random.default_rng(11)

    def local() -> np.ndarray:
        return np.kron(random_unitary(2, rng), random_unitary(2, rng))

    block = synthesize_two_qubit(local() @ canonical_gate(0.5, 0.3, 2e-7) @ local())
    assert _two_qubit(block) == 3
    circ = QuantumCircuit()
    circ.h(0).cx(1, 2)
    circ.compose(block, qubits=[0, 1], inplace=True)
    circ.cx(1, 2)
    target = _grid(1, 3)
    for level in (2, 3):
        exact = transpile(
            circ, target, optimization_level=level, seed=0, approximation_degree=None
        )
        res = transpile(circ, target, optimization_level=level, seed=0)
        assert _two_qubit(res.circuit) == _two_qubit(exact.circuit) - 1
        assert res.stats.approximated_operations == 1
        assert 0 < res.stats.approximation_error < 1e-6


def _plan(
    label: str, est: int, *, default: bool = False, embedded: bool = False
) -> _CandidatePlan:
    return _CandidatePlan(
        source=QuantumCircuit(),
        num_logical=0,
        gates=[],
        initial_mapping=[],
        final_mapping=[],
        num_swaps=0,
        depth=0,
        estimated_two_qubit_gates=est,
        label=label,
        is_default=default,
        embedded=embedded,
    )


def test_pick_evaluations_keeps_non_embedded_candidates() -> None:
    from quspect.transpiler._transpile import _pick_evaluations

    cands = [
        _plan("e0", 100, default=True, embedded=True),
        _plan("e1", 100, embedded=True),
        _plan("r0", 101),
        _plan("r1", 102),
        _plan("r2", 103),
    ]
    picked = [c.label for c in _pick_evaluations(cands, None)]
    # the low-dilation starts rank first; the best other starts are added
    assert picked == ["e0", "e1", "r0", "r1"]
    # mixed selections are unchanged
    mixed = [cands[0], cands[2], cands[1], cands[3]]
    assert [c.label for c in _pick_evaluations(mixed, None)] == ["e0", "r0"]
    # the estimated-gate budget still applies
    big = [_plan("e0", 10, embedded=True), _plan("e1", 2000, embedded=True)]
    big += [_plan("r0", 2000)]
    assert [c.label for c in _pick_evaluations(big, None)] == ["e0", "e1"]


def test_depth_weight_selection() -> None:
    """``depth_weight`` picks the candidate minimizing 2q + w·(2q depth)."""
    circ = _qft(10)
    target = _grid(3, 4)
    base = transpile(circ, target, optimization_level=3, seed=0)
    zero = transpile(circ, target, optimization_level=3, seed=0, depth_weight=0.0)
    assert zero.stats.selected == base.stats.selected
    for w in (0.5, 5.0):
        res = transpile(circ, target, optimization_level=3, seed=0, depth_weight=w)
        scores = res.stats.candidate_scores
        best = min(s[1] + w * s[2] for s in scores)
        chosen = next(s for s in scores if s[0] == res.stats.selected)
        assert chosen[1] + w * chosen[2] == best
    with pytest.raises(ValueError, match="depth_weight"):
        transpile(circ, target, optimization_level=3, depth_weight=-1.0)
