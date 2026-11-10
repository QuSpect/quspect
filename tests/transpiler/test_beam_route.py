"""Tests for the global beam-search routing candidate (``_beam_route``)."""

from __future__ import annotations

import random

import pytest

import quspect.transpiler._transpile as transpile_module
from quspect import QuantumCircuit
from quspect.transpiler import SABRE, transpile
from quspect.transpiler._beam_route import (
    BEAM_MIN_WIDTH,
    BEAM_WIDTH,
    beam_applicable,
    beam_config,
    beam_route_plan,
)
from quspect.transpiler._mapping import _build_routing_dag, _extract_gates
from quspect.transpiler._route_lns import lns_available

from .test_transpile import _assert_equivalent, _grid, _two_qubit

pytestmark = pytest.mark.skipif(not lns_available(), reason="native core missing")


def _random_cx(n: int, count: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for _ in range(count):
        a, b = rng.sample(range(n), 2)
        qc.cx(a, b)
        qc.rz(rng.uniform(0, 3), b)
    return qc


def _dag(qc: QuantumCircuit):  # noqa: ANN202 - internal helper
    gates, n = _extract_gates(qc)
    return _build_routing_dag(gates, n, qc)


def test_applicability() -> None:
    assert beam_applicable(_dag(QuantumCircuit().cx(0, 1).cx(1, 2)))
    # 末端测量：只排在其他测量之前
    assert beam_applicable(_dag(QuantumCircuit().cx(0, 2).measure([0, 1], [0, 1])))
    # 测量之后还有两比特门：块的链顺序不再表达测量栅栏
    assert not beam_applicable(
        _dag(QuantumCircuit().cx(0, 1).measure([0], [0]).cx(1, 2))
    )
    # 按三比特单元路由的 Toffoli
    assert not beam_applicable(_dag(QuantumCircuit().ccx(0, 1, 2)))
    # 没有两比特门
    assert not beam_applicable(_dag(QuantumCircuit().h(0)))


def test_work_budget() -> None:
    assert beam_config(100, 20, 16, 30) == (BEAM_WIDTH, 2)
    # 大线路：先缩窄束，再去掉布局迭代，最后放弃
    width, iters = beam_config(2_000, 133, 60, 150)  # type: ignore[misc]
    assert BEAM_MIN_WIDTH <= width < BEAM_WIDTH
    assert beam_config(10**7, 100, 100, 180) is None


def test_plan_is_a_valid_routing_with_fewer_swaps() -> None:
    target = _grid(4, 4)
    sabre = SABRE(target)
    qc = _random_cx(12, 160, seed=5)
    plans = sabre._candidate_plans(qc, seed=1, layout_trials=4, exact_layout=False)
    plan = beam_route_plan(sabre, plans)
    assert plan is not None
    assert plan.label.startswith("beam/")
    assert not plan.is_default
    assert plan.num_swaps == sum(1 for e in plan.route_events if e < 0)
    best = min(p.num_swaps for p in plans)
    assert plan.num_swaps < best
    # 重建的线路：每个两比特门落在耦合边上
    routed = plan.build_circuit()
    for instr in routed:
        if instr.is_gate and len(instr.qubits) == 2:
            pa, pb = (plan.initial_mapping[w] for w in instr.qubits)
            assert tuple(sorted((pa, pb))) in target.coupling_graph
    # 确定性
    again = beam_route_plan(sabre, plans)
    assert again is not None
    assert again.route_events == plan.route_events
    assert again.route_layout == plan.route_layout


def test_not_applied_to_toffoli_routing() -> None:
    target = _grid(3, 3)
    sabre = SABRE(target)
    qc = QuantumCircuit().ccx(0, 4, 8).cx(0, 8).ccx(2, 6, 4)
    plans = sabre._candidate_plans(qc, seed=1, exact_layout=False)
    assert beam_route_plan(sabre, plans) is None


@pytest.mark.parametrize("seed", [3, 4])
def test_level3_with_beam_is_valid_and_not_worse(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _grid(3, 4)
    qc = _random_cx(10, 90, seed=seed)
    # 只比较路由候选：CNOT-相位重综合（第 3 级）改变候选打分，而最后阶段（SWAP 面
    # 改写）不随打分单调，会让“选中束搜索候选”与最终门数的先后关系翻转
    monkeypatch.setattr(transpile_module, "_PHASE_POLY_LEVEL", 4)
    with_beam = transpile(qc, target, optimization_level=3, seed=seed)
    _assert_equivalent(qc, with_beam, target)
    assert any(
        label.startswith("beam/") for label, *_ in with_beam.stats.candidate_scores
    )
    monkeypatch.setattr(transpile_module, "_BEAM_ROUTE", False)
    without = transpile(qc, target, optimization_level=3, seed=seed)
    assert not any(
        label.startswith("beam/") for label, *_ in without.stats.candidate_scores
    )
    assert _two_qubit(with_beam.circuit) <= _two_qubit(without.circuit)
