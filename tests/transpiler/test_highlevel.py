"""Tests for HighLevelOptimizePass (pre-decomposition gate-pair cancellation)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import HighLevelOptimizePass
from tests.synthesis._unitary import circuit_unitary


def _names(circ: QuantumCircuit) -> list[str]:
    return [i.gate.name for i in circ if i.gate is not None]


def test_ccx_pair_cancels() -> None:
    p = HighLevelOptimizePass()
    assert len(p.run(QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2))) == 0
    assert p.stats.pairs_cancelled == {"ccx": 1}


def test_ccx_controls_in_either_order() -> None:
    out = HighLevelOptimizePass().run(QuantumCircuit().ccx(0, 1, 2).ccx(1, 0, 2))
    assert len(out) == 0


def test_ccx_different_target_kept() -> None:
    circ = QuantumCircuit().ccx(0, 1, 2).ccx(0, 2, 1)
    assert len(HighLevelOptimizePass().run(circ)) == 2


def test_commuting_gates_in_between() -> None:
    circ = QuantumCircuit().ccx(0, 1, 2).t(0).rz(0.3, 1).x(2).cx(3, 2).cz(0, 4)
    circ.cx(1, 5).ccx(0, 1, 2)
    out = HighLevelOptimizePass().run(circ)
    assert _names(out) == ["t", "rz", "x", "cx", "cz", "cx"]


@pytest.mark.parametrize(
    "middle",
    [
        lambda c: c.h(0),  # H on a control
        lambda c: c.t(2),  # Z-type on the target
        lambda c: c.cx(2, 3),  # target used as a control
        lambda c: c.cx(3, 0),  # control used as a target
    ],
)
def test_non_commuting_gate_blocks(middle: object) -> None:
    circ = QuantumCircuit().ccx(0, 1, 2)
    middle(circ)  # type: ignore[operator]
    circ.ccx(0, 1, 2)
    assert _names(HighLevelOptimizePass().run(circ)).count("ccx") == 2


def test_measurement_blocks() -> None:
    circ = QuantumCircuit().cx(0, 1)
    circ.measure([0], [0])
    circ.cx(0, 1)
    assert _names(HighLevelOptimizePass().run(circ)).count("cx") == 2


def test_cascading_cancellation() -> None:
    circ = QuantumCircuit().ccx(0, 1, 2).cx(3, 4).swap(0, 5).swap(5, 0)
    circ.cx(3, 4).ccx(0, 1, 2)
    assert len(HighLevelOptimizePass().run(circ)) == 0


def test_window_limit() -> None:
    circ = QuantumCircuit().cx(0, 1)
    for _ in range(5):
        circ.t(0)
    circ.cx(0, 1)
    assert _names(HighLevelOptimizePass(window=3).run(circ)).count("cx") == 2
    assert _names(HighLevelOptimizePass(window=6).run(circ)).count("cx") == 0
    with pytest.raises(ValueError, match="window"):
        HighLevelOptimizePass(window=0)


def test_input_not_modified() -> None:
    circ = QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2)
    HighLevelOptimizePass().run(circ)
    assert len(circ) == 2


@pytest.mark.parametrize("seed", range(20))
def test_random_circuits_keep_unitary(seed: int) -> None:
    rng = random.Random(seed)
    circ = QuantumCircuit()
    for _ in range(30):
        qs = rng.sample(range(4), 3)
        kind = rng.choice(["ccx", "ccx", "cx", "cz", "swap", "t", "x", "h", "rz", "rx"])
        if kind == "ccx":
            circ.ccx(*qs)
        elif kind in ("cx", "cz", "swap"):
            getattr(circ, kind)(qs[0], qs[1])
        elif kind in ("rz", "rx"):
            getattr(circ, kind)(rng.uniform(-3, 3), qs[0])
        else:
            getattr(circ, kind)(qs[0])
    out = HighLevelOptimizePass().run(circ)
    assert len(out) <= len(circ)
    u = circuit_unitary(circ, 4)
    v = circuit_unitary(out, 4)
    # 精确等价（含全局相位）：逐元素比较。
    assert np.max(np.abs(u - v)) < 1e-9
