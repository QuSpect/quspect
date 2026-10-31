"""Regression tests for ``DensityMatrixSimulator.run`` key semantics."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect.simulation.density_simulator import DensityMatrixSimulator


def _nonzero(counts: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in counts.items() if v}


def test_counts_follow_clbit_map() -> None:
    """c0 <- q2 (=0), c1 <- q0 (=1): key '01' (clbit 0 leftmost)."""
    circ = QuantumCircuit().x(0).id(2)
    circ.measure([0, 2], [1, 0])
    result = DensityMatrixSimulator(seed=1).run(circ, shots=50)
    assert _nonzero(result.counts) == {"01": 50}
    assert result.num_clbits == 2
    assert list(result.counts) == ["00", "01", "10", "11"]


def test_partial_measure_has_num_clbits_bits() -> None:
    """Only q1 measured into c0: one-bit keys; unmeasured q0 is ignored."""
    circ = QuantumCircuit().x(1).h(0).measure([1], [0])
    result = DensityMatrixSimulator(seed=1).run(circ, shots=40)
    assert _nonzero(result.counts) == {"1": 40}
    assert result.num_clbits == 1


def test_unwritten_clbits_read_zero() -> None:
    """A clbit that is never written stays 0."""
    circ = QuantumCircuit().x(0).measure([0], [2])
    result = DensityMatrixSimulator(seed=1).run(circ, shots=10)
    assert _nonzero(result.counts) == {"001": 10}


def test_no_measurement_samples_all_qubits() -> None:
    """Without measurements every qubit q is reported in position q."""
    result = DensityMatrixSimulator(seed=1).run(QuantumCircuit().x(1).id(2), shots=7)
    assert _nonzero(result.counts) == {"010": 7}
    assert result.num_clbits == 3


def test_mid_circuit_measurement_distribution() -> None:
    """x, AD(0.3), measure q0, cx(0,1), measure q1: P(11)=0.7, P(00)=0.3."""
    circ = QuantumCircuit().x(0).amplitude_damp(0.3, 0)
    circ.measure([0], [0]).cx(0, 1).measure([1], [1])
    shots = 20000
    counts = _nonzero(DensityMatrixSimulator(seed=4).run(circ, shots=shots).counts)
    assert set(counts) == {"00", "11"}
    assert counts["11"] / shots == pytest.approx(0.7, abs=0.02)


def test_mid_circuit_measurement_branches_per_shot() -> None:
    """h, measure, then x: each shot gets an independent mid-circuit outcome."""
    circ = QuantumCircuit().h(0).measure([0], [0]).x(0).measure([0], [1])
    counts = _nonzero(DensityMatrixSimulator(seed=2).run(circ, shots=4000).counts)
    assert set(counts) == {"01", "10"}
    assert counts["01"] / 4000 == pytest.approx(0.5, abs=0.05)


def test_mid_circuit_branch_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Too many distinct classical branches raise a clear ValueError."""
    import quspect.simulation.density_simulator as mod

    monkeypatch.setattr(mod, "_MAX_BRANCHES", 2)
    circ = QuantumCircuit().h(0).h(1).measure([0, 1], [0, 1]).x(0)
    with pytest.raises(ValueError, match="branches"):
        DensityMatrixSimulator(seed=0).run(circ, shots=10)


def test_negative_shots_rejected() -> None:
    """shots < 0 raises ValueError."""
    with pytest.raises(ValueError):
        DensityMatrixSimulator().run(QuantumCircuit().x(0), shots=-1)
