"""Tests for layered routing of SU(4)-block circuits on a path."""

from __future__ import annotations

import random

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget, transpile
from quspect.transpiler._layer_route import (
    block_layers,
    cycle_order,
    layer_route_events,
    layer_route_plan,
    path_order,
    su4_route_events,
)
from quspect.transpiler._route_lns import lns_available, routing_cost

from .test_transpile import BASIS, _assert_equivalent, _grid, _two_qubit

pytestmark = pytest.mark.skipif(not lns_available(), reason="native core missing")


def _line(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASIS)
    )


def _ring(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, (i + 1) % n) for i in range(n)], native_gates=set(BASIS)
    )


def _heavy_square() -> CompilationTarget:
    """A small heavy-hex-like graph: two 6-cycles of degree-2 bridges, 10 qubits."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (2, 6), (4, 7), (5, 8)]
    edges += [(6, 8), (8, 9), (9, 7)]
    return CompilationTarget.from_coupling_map(10, edges, native_gates=set(BASIS))


def _qv(n: int, depth: int, seed: int) -> QuantumCircuit:
    """Quantum-volume-like circuit: random pairings of generic SU(4) blocks."""
    rng = random.Random(seed)
    qc = QuantumCircuit()

    def u3(q: int) -> None:
        qc.u3(rng.uniform(0.1, 3.0), rng.uniform(-3, 3), rng.uniform(-3, 3), q)

    for _ in range(depth):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(0, n - 1, 2):
            a, b = perm[i], perm[i + 1]
            for _ in range(3):
                u3(a)
                u3(b)
                qc.cx(a, b)
            u3(a)
            u3(b)
    return qc


def _prepared(target: CompilationTarget, qc: QuantumCircuit) -> tuple[SABRE, object]:
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    return router, router._prepare(qc, False)


def test_path_order() -> None:
    assert path_order([{1}, {0, 2}, {1}]) == [0, 1, 2]
    assert path_order([{2}, {2}, {0, 1}]) == [0, 2, 1]
    assert path_order([set()]) == [0]
    assert path_order([{1, 2}, {0, 2}, {0, 1}]) is None  # ring
    assert path_order([{1}, {0}, {3}, {2}]) is None  # two components
    assert path_order([{1, 3}, {0, 2, 4}, {1, 5}, {0, 4}, {1, 3, 5}, {2, 4}]) is None


def test_cycle_order() -> None:
    assert cycle_order([{1, 3}, {0, 2}, {1, 3}, {2, 0}]) == [0, 1, 2, 3]
    assert cycle_order([{2, 1}, {0, 2}, {1, 0}]) == [0, 1, 2]
    assert cycle_order([{1}, {0, 2}, {1}]) is None  # path
    two_triangles = [{1, 2}, {0, 2}, {0, 1}, {4, 5}, {3, 5}, {3, 4}]
    assert cycle_order(two_triangles) is None
    assert cycle_order([{1}, {0}]) is None


def test_block_layers_detects_su4_circuits() -> None:
    router, prep = _prepared(_line(6), _qv(6, 4, 0))
    layers = block_layers(prep)  # type: ignore[arg-type]
    assert layers is not None
    dag = prep.dag  # type: ignore[attr-defined]
    # 每层是一个匹配；同一比特对上相继的块合并成一个节点，所以层数 / 块数可能更少
    assert 1 <= len(layers) <= 4
    for layer in layers:
        qubits = [q for v in layer for q in (dag.nq1[v], dag.nq2[v])]
        assert len(qubits) == len(set(qubits))
    assert sum(len(layer) for layer in layers) <= 12
    # 单个 CX 不是一般 SU(4)
    qc = QuantumCircuit().cx(0, 1).cx(1, 2)
    _, prep2 = _prepared(_line(3), qc)
    assert block_layers(prep2) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", range(4))
def test_layer_route_events_are_legal_and_cost_matches(seed: int) -> None:
    n = 8
    router, prep = _prepared(_line(n), _qv(n, 6, seed))
    out = layer_route_events(router, prep, seed=seed)  # type: ignore[arg-type]
    assert out is not None
    layout, events, paid = out
    # 合法（routing_cost 校验事件序列），且合并模型下的代价恰为 3 × 付费 SWAP
    cost = routing_cost(router, prep, layout, events)  # type: ignore[arg-type]
    assert cost == 3 * paid


@pytest.mark.parametrize("seed", range(3))
def test_ring_layer_route_events_are_legal_and_cost_matches(seed: int) -> None:
    n = 8
    router, prep = _prepared(_ring(n), _qv(n, 6, seed))
    out = layer_route_events(router, prep, seed=seed)  # type: ignore[arg-type]
    assert out is not None
    layout, events, paid = out
    cost = routing_cost(router, prep, layout, events)  # type: ignore[arg-type]
    assert cost == 3 * paid


@pytest.mark.parametrize(
    ("target", "width"),
    [(_grid(3, 3), 16), (_grid(2, 4), 1), (_heavy_square(), 16), (_ring(7), 4)],
)
def test_su4_route_events_are_legal_and_cost_matches(
    target: CompilationTarget, width: int
) -> None:
    n = min(target.num_qubits, 9)
    router, prep = _prepared(target, _qv(n, 6, width))
    layout = list(range(target.num_qubits))
    out = su4_route_events(router, prep, layout, seed=3, width=width)  # type: ignore[arg-type]
    assert out is not None
    start, events, paid = out
    assert sorted(start) == layout
    cost = routing_cost(router, prep, start, events)  # type: ignore[arg-type]
    # 免费翻转都被块吸收；个别付费 SWAP 在合并模型下也能并入同一对比特上的块
    assert cost is not None
    assert cost <= 3 * paid
    # 非 SU(4) 块线路不适用
    _, prep2 = _prepared(target, QuantumCircuit().cx(0, 1))
    assert su4_route_events(router, prep2, layout) is None  # type: ignore[arg-type]


def test_layer_route_plan_rollout_on_grid() -> None:
    target = _grid(3, 4)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    base = router._solve_plan(_qv(12, 8, 2), seed=1)
    plan = layer_route_plan(router, base)
    assert plan is not None
    assert plan.label.startswith("rollout/")
    assert not plan.is_default


def test_layer_route_not_for_other_topologies() -> None:
    router, prep = _prepared(_grid(2, 4), _qv(8, 3, 0))
    assert layer_route_events(router, prep) is None  # type: ignore[arg-type]


def test_layer_route_plan_beats_sabre_on_long_lines() -> None:
    n = 16
    target = _line(n)
    qc = _qv(n, 12, 1)
    router = SABRE(target, block_nodes=True, mirror_gates=True)
    base = router._solve_plan(qc, seed=1)
    plan = layer_route_plan(router, base)
    assert plan is not None
    assert plan.label.startswith("layers/")
    assert plan.estimated_two_qubit_gates < base.estimated_two_qubit_gates


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("shape", ["grid", "ring", "heavy"])
def test_transpile_qv_on_2d_and_ring_is_equivalent(level: int, shape: str) -> None:
    if shape == "grid":
        target, n = _grid(2, 4), 8
    elif shape == "ring":
        target, n = _ring(8), 8
    else:
        target, n = _heavy_square(), 9
    qc = _qv(n, 5, 4)
    res = transpile(qc, target, optimization_level=level, seed=0)
    _assert_equivalent(qc, res, target)


@pytest.mark.parametrize("level", [2, 3])
def test_transpile_qv_on_line_is_equivalent(level: int) -> None:
    n = 6
    target = _line(n)
    qc = _qv(n, 5, 3)
    res = transpile(qc, target, optimization_level=level, seed=0)
    _assert_equivalent(qc, res, target)
    assert _two_qubit(res.circuit) <= 3 * 15 + 3 * 20


def test_su4_lookahead_covers_two_layers() -> None:
    from quspect.transpiler._transpile import _su4_extended_set

    router = SABRE(_line(24), block_nodes=True, mirror_gates=True)
    assert _su4_extended_set(router, _qv(24, 2, 0), False) == 24
    assert _su4_extended_set(router, _qv(8, 2, 0), False) is None  # 默认扩展集已覆盖
    plain = SABRE(_line(24))
    assert _su4_extended_set(plain, _qv(24, 2, 0), False) is None
    cx_only = QuantumCircuit()
    for i in range(0, 23, 2):
        cx_only.cx(i, i + 1)
    assert _su4_extended_set(router, cx_only, False) is None


def test_refinement_top_keeps_sabre_candidates() -> None:
    from dataclasses import replace

    from quspect.transpiler._transpile import _is_layered, _top_with_layered

    router = SABRE(_grid(2, 3), block_nodes=True, mirror_gates=True)
    base = router._solve_plan(_qv(6, 3, 0), seed=1)
    plans = [
        replace(base, label="rollout/x"),
        replace(base, label="trial1"),
        replace(base, label="layers/y"),
        replace(base, label="trial2"),
    ]
    assert _is_layered(plans[0]) and _is_layered(plans[2])
    assert not _is_layered(plans[1])
    # 分层 / rollout 候选不占 SABRE 候选的名额
    assert _top_with_layered([0, 1, 2, 3], plans, 1) == [0, 1]
    assert _top_with_layered([0, 2, 1, 3], plans, 2) == [0, 2, 1, 3]
    assert _top_with_layered([1, 0, 3], plans, 1) == [1]
