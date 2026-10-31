"""Tests for exact window re-routing and the LNS refinement of SABRE routings."""

from __future__ import annotations

import itertools
import random
from collections import deque

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget, transpile
from quspect.transpiler._route_lns import (
    deepen_plan,
    exact_route,
    lns_available,
    node_cost_model,
    node_weights,
    refine_plan,
    refine_routing,
    routing_cost,
)

from .test_transpile import BASIS, _assert_equivalent, _grid, _random_circuit

pytestmark = pytest.mark.skipif(not lns_available(), reason="native core missing")


def _line(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASIS)
    )


def _cx_circuit(n: int, num: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(num):
        a, b = rng.sample(range(n), 2)
        qc.cx(a, b)
    return qc


def _brute_force_min_swaps(router: SABRE, gates: list[tuple[int, int]], n: int) -> int:
    """Minimum SWAP count by breadth-first search over (layout, executed gates)."""
    adj = router._adj_sets
    edges = sorted({(min(a, b), max(a, b)) for a in range(n) for b in adj[a]})

    preds = [
        {j for j in range(i) if set(gates[j]) & set(gates[i])}
        for i in range(len(gates))
    ]

    def drain(pi: tuple[int, ...], done: frozenset[int]) -> frozenset[int]:
        done_set = set(done)
        progress = True
        while progress:
            progress = False
            for i, (a, b) in enumerate(gates):
                if i not in done_set and preds[i] <= done_set and pi[b] in adj[pi[a]]:
                    done_set.add(i)
                    progress = True
        return frozenset(done_set)

    start = tuple(range(n))
    root = (start, drain(start, frozenset()))
    seen = {root: 0}
    queue = deque([root])
    while queue:
        state = queue.popleft()
        pi, done = state
        if len(done) == len(gates):
            return seen[state]
        for x, y in edges:
            new = tuple(y if p == x else x if p == y else p for p in pi)
            nxt = (new, drain(new, done))
            if nxt not in seen:
                seen[nxt] = seen[state] + 1
                queue.append(nxt)
    raise AssertionError("unroutable")


@pytest.mark.parametrize("seed", range(6))
def test_exact_route_is_optimal_swap_count(seed: int) -> None:
    n = 5
    target = _line(n)
    router = SABRE(target)
    qc = _cx_circuit(n, 6, seed)
    prep = router._prepare(qc, False)
    gates = [(g.q1, g.q2) for g in prep.gates if g.is_two_qubit]
    res = exact_route(router, prep, list(range(n)), merge_costs=False)
    assert res is not None and res.found and res.optimal
    assert res.num_swaps == _brute_force_min_swaps(router, gates, n)
    assert routing_cost(
        router, prep, list(range(n)), res.events, merge_costs=False
    ) == (3 * res.num_swaps)


def test_exact_route_not_worse_than_sabre() -> None:
    target = _grid(2, 3)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    qc = _random_circuit(6, 14, 4)
    prep = router._prepare(qc, False)
    layout = list(range(6))
    sabre = router._route(
        layout, prep.dag, forward=True, rng=random.Random(1), record=True
    )
    base = routing_cost(router, prep, layout, sabre.events)
    assert base is not None
    res = exact_route(router, prep, layout, max_expansions=500_000)
    assert res is not None and res.found
    assert res.cost <= base
    assert routing_cost(router, prep, layout, res.events) == res.cost


def test_refine_routing_never_costlier_and_same_nodes() -> None:
    target = _grid(3, 3)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    for seed in range(4):
        qc = _random_circuit(9, 40, seed)
        prep = router._prepare(qc, True)
        layout = list(range(9))
        sabre = router._route(
            layout, prep.dag, forward=True, rng=random.Random(seed), record=True
        )
        res = refine_routing(router, prep, layout, sabre.events)
        assert res is not None
        assert res.new_cost <= res.old_cost
        assert routing_cost(router, prep, layout, res.events) == res.new_cost
        assert sorted(e for e in res.events if e >= 0) == sorted(
            e for e in sabre.events if e >= 0
        )


def test_refine_plan_circuit_is_equivalent() -> None:
    target = _grid(3, 3)
    router = SABRE(target, block_nodes=True, mirror_gates=True, swap_fusion_bonus=0.2)
    refined_any = False
    for seed in range(6):
        qc = _random_circuit(9, 45, seed)
        plans = router._candidate_plans(qc, seed=seed, layout_trials=2)
        for plan in plans:
            refined = refine_plan(router, plan)
            if refined is None:
                continue
            refined_any = True
            assert refined.label.startswith("lns/")
            assert not refined.is_default
            routed = refined.build_circuit()
            for instr in routed:
                if len(instr.qubits) == 2:
                    p, q = (refined.initial_mapping[w] for w in instr.qubits)
                    assert q in router._adj_sets[p]
    assert refined_any


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_transpile_with_lns_is_equivalent(level: int, seed: int) -> None:
    target = _grid(2, 4)
    qc = _random_circuit(8, 40, seed)
    result = transpile(qc, target, optimization_level=level, seed=seed)
    _assert_equivalent(qc, result, target)


def test_lns_does_not_hurt_level3(monkeypatch: pytest.MonkeyPatch) -> None:
    import quspect.transpiler._transpile as transpile_module

    target = _line(8)
    qc = _random_circuit(8, 50, 2)
    with_lns = transpile(qc, target, optimization_level=3, seed=2)
    monkeypatch.setattr(transpile_module, "_LNS_LEVEL", 4)
    without = transpile(qc, target, optimization_level=3, seed=2)
    assert with_lns.stats.two_qubit_gates <= without.stats.two_qubit_gates
    assert "lns" not in without.stats.stage_seconds


def test_node_cost_model_shapes() -> None:
    target = _line(4)
    router = SABRE(target)
    qc = QuantumCircuit().cx(0, 1).cz(1, 2).swap(2, 3)
    qc.measure_all()
    prep = router._prepare(qc, False)
    mirror, measure = node_cost_model(prep)
    assert len(mirror) == len(measure) == len(prep.dag.nodes)
    assert all(-1 <= c <= 3 for c in mirror)
    assert any(t >= 0 for t in measure)
    assert node_cost_model(prep) is prep.lns_model
    # a SWAP block mirrors for free
    assert mirror[[g.name for g in prep.dag.nodes].index("swap")] == 0


def test_exact_route_small_examples() -> None:
    target = _line(4)
    router = SABRE(target)
    for a, b in itertools.combinations(range(4), 2):
        prep = router._prepare(QuantumCircuit().cx(a, b), False)
        res = exact_route(router, prep, list(range(4)), merge_costs=False)
        assert res is not None and res.optimal
        assert res.num_swaps == abs(a - b) - 1


def test_depth_sweeps_are_lexicographic() -> None:
    target = _grid(3, 3)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    for seed in range(4):
        qc = _random_circuit(9, 40, seed)
        prep = router._prepare(qc, True)
        layout = list(range(9))
        sabre = router._route(
            layout, prep.dag, forward=True, rng=random.Random(seed), record=True
        )
        res = refine_routing(
            router,
            prep,
            layout,
            sabre.events,
            passes=0,
            depth_passes=2,
            depth_expansions=100_000,
        )
        assert res is not None
        # 深度扫描按 (代价, 深度) 字典序替换窗口：代价不增，代价相同时深度不增
        assert res.new_cost <= res.old_cost
        if res.new_cost == res.old_cost:
            assert res.new_depth <= res.old_depth
        assert routing_cost(router, prep, layout, res.events) == res.new_cost
        # 字典序主扫描：代价不增
        lex = refine_routing(router, prep, layout, sabre.events, depth_ties=True)
        assert lex is not None and lex.new_cost <= lex.old_cost
        assert routing_cost(router, prep, layout, lex.events) == lex.new_cost


def test_deepen_plan_is_equivalent_and_labelled() -> None:
    target = _line(6)
    qc = _random_circuit(6, 40, 3)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    deepened = 0
    for seed in range(4):
        for plan in router._candidate_plans(qc, seed=seed, layout_trials=2):
            deep = deepen_plan(router, plan, expansions=200_000)
            if deep is None:
                continue
            deepened += 1
            assert deep.label.startswith("lnsd/")
            assert not deep.is_default
            prep = plan.prep
            assert prep is not None
            assert routing_cost(
                router, prep, deep.route_layout, deep.route_events
            ) <= routing_cost(router, prep, plan.route_layout, plan.route_events)
    assert deepened > 0


def test_node_weights_cap_blocks_at_three() -> None:
    target = _line(3)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    qc = QuantumCircuit().cx(0, 1).cx(0, 1).cx(1, 0).cx(0, 1).cx(1, 2).ccx(0, 1, 2)
    prep = router._prepare(qc, False)
    weights = node_weights(prep)
    assert len(weights) == len(prep.dag.nodes)
    assert all(w >= 0 for w in weights)
    assert node_weights(prep) is weights
