"""Ablation switches (research only): default off, each switch takes effect."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect._ablation import ABLATION_ENV, COMPONENTS, ablated
from quspect.transpiler import CompilationTarget, transpile

_NATIVE = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


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
    monkeypatch.setenv(ABLATION_ENV, "select")
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
