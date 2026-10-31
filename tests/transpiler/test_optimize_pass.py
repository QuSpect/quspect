"""Tests for the transpiler OptimizePass."""

from __future__ import annotations

from quspect import QuantumCircuit
from quspect.transpiler._passes import OptimizePass


def test_optimize_pass_commutation_enables_cancellation() -> None:
    """cx(0,1) rz(0) cx(0,1) reduces to a single rz (B77)."""
    circ = QuantumCircuit().cx(0, 1).rz(0.3, 0).cx(0, 1)
    out = OptimizePass().run(circ)
    names = [i.gate.name for i in out.instructions if i.gate is not None]
    assert names == ["rz"]


def test_optimize_pass_max_iterations_attribute() -> None:
    """max_iterations is configurable."""
    assert OptimizePass(max_iterations=7).max_iterations == 7
