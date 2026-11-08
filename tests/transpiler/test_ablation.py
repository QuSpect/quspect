"""Ablation switches (research only): default off, each switch takes effect."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect._ablation import ABLATION_ENV, COMPONENTS, ablated
from quspect.transpiler import CompilationTarget, transpile

_NATIVE = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


_R21_POST = "pp_small,pp_large,pp_pre,num3q,numblk,repass"
_R21 = f"{_R21_POST},pp_twolevel,pp_raw,plain,beam"


def _qft_like(n: int) -> QuantumCircuit:
    qc = QuantumCircuit()
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(0.7 / (j - i), j, i)
    return qc


def test_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ABLATION_ENV, raising=False)
    assert not any(ablated(c) for c in COMPONENTS)


def test_unknown_component_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ABLATION_ENV, "lns,nope")
    with pytest.raises(ValueError, match="nope"):
        ablated("lns")


def test_select_reduces_level3_to_level2(monkeypatch: pytest.MonkeyPatch) -> None:
    target = CompilationTarget.from_coupling_map(
        5, [(i, i + 1) for i in range(4)], native_gates=_NATIVE
    )
    qc = _qft_like(5)
    monkeypatch.delenv(ABLATION_ENV, raising=False)
    level2 = transpile(qc, target, optimization_level=2, seed=0)
    # 第 21 轮的第 3 级组件（选中之后的再编译、数值实例化、路由前第二起点与
    # 路由后的区域重综合）不属于候选搜索，一起关掉才等于第 2 级
    monkeypatch.setenv(ABLATION_ENV, f"select,{_R21_POST}")
    level3 = transpile(qc, target, optimization_level=3, seed=0)
    assert level3.stats.num_candidates == 1
    assert [(i.gate.name, i.qubits) for i in level3.circuit] == [
        (i.gate.name, i.qubits) for i in level2.circuit
    ]


def test_sched_keeps_gate_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from quspect.transpiler._schedule import schedule_for_depth

    qc = QuantumCircuit().cx(1, 2).cx(0, 1).rz(0.3, 0).cx(0, 3)
    monkeypatch.setenv(ABLATION_ENV, "sched")
    assert schedule_for_depth(qc) is qc


def test_memo_is_separate_per_ablation_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from quspect.synthesis._memo import BoundedCache

    cache: BoundedCache[str, int] = BoundedCache()
    monkeypatch.setenv(ABLATION_ENV, "stab")
    cache.put("k", 1)
    assert cache.get("k") == 1
    monkeypatch.delenv(ABLATION_ENV)
    assert cache.get("k") is None
    cache.put("k", 2)
    monkeypatch.setenv(ABLATION_ENV, "stab")
    assert cache.get("k") == 1


_ALL_EXT = (
    "vf2,qap,mirror,comm,lns,kendall,rollout,select,stab,qsd,phase,approx,sched,orient"
)


@pytest.mark.parametrize(
    "switches",
    ["kak", "sabre2", "gauge", _ALL_EXT, f"{_ALL_EXT},kak,sabre2,gauge"],
)
def test_base_switches_stay_equivalent(
    monkeypatch: pytest.MonkeyPatch, switches: str
) -> None:
    from tests.transpiler.test_transpile import _assert_equivalent

    target = CompilationTarget.from_coupling_map(
        5, [(i, i + 1) for i in range(4)], native_gates=_NATIVE
    )
    qc = _qft_like(5).ccx(0, 2, 4).swap(1, 3)
    monkeypatch.delenv(ABLATION_ENV, raising=False)
    full = transpile(qc, target, optimization_level=3, seed=0)
    monkeypatch.setenv(ABLATION_ENV, switches)
    out = transpile(qc, target, optimization_level=3, seed=0)
    _assert_equivalent(qc, out, target)
    if switches != "gauge":
        assert out.stats.two_qubit_gates >= full.stats.two_qubit_gates


@pytest.mark.parametrize(
    "switches",
    [
        "pp_small",
        "pp_large",
        "pp_twolevel",
        "pp_raw",
        "pp_pre",
        "num3q",
        "numblk",
        "plain",
        "repass",
        "beam",
        _R21,
        f"{_ALL_EXT},{_R21}",
    ],
)
def test_round21_switches_stay_equivalent(
    monkeypatch: pytest.MonkeyPatch, switches: str
) -> None:
    from tests.transpiler.test_transpile import _assert_equivalent

    target = CompilationTarget.from_coupling_map(
        6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], native_gates=_NATIVE
    )
    qc = _qft_like(5).ccx(0, 2, 4).cx(1, 3).t(3).cx(3, 1).swap(0, 4)
    monkeypatch.setenv(ABLATION_ENV, switches)
    out = transpile(qc, target, optimization_level=3, seed=0)
    _assert_equivalent(qc, out, target)
    small = QuantumCircuit().ccx(0, 1, 2).cx(0, 1).h(2).cx(2, 0)
    _assert_equivalent(
        small, transpile(small, target, optimization_level=3, seed=0), target
    )


def test_num3q_switch_takes_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (1, 2)], native_gates=_NATIVE
    )
    qc = QuantumCircuit().ccx(0, 1, 2).cx(0, 1)
    monkeypatch.delenv(ABLATION_ENV, raising=False)
    full = transpile(qc, target, optimization_level=3, seed=0)
    monkeypatch.setenv(ABLATION_ENV, "num3q,repass")
    off = transpile(qc, target, optimization_level=3, seed=0)
    assert "numeric3q" not in off.stats.selected
    assert full.stats.two_qubit_gates <= off.stats.two_qubit_gates
