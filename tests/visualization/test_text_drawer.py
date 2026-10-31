"""Tests for the ASCII circuit drawer."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect.visualization.text_drawer import draw_text_circuit


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        ("crx", (0.3,), "[RX(0.3)]"),
        ("cry", (0.3,), "[RY(0.3)]"),
        ("crz", (0.3,), "[RZ(0.3)]"),
        ("cp", (0.3,), "[P(0.3)]"),
        ("cu3", (0.3, 0.4, 0.5), "[U3(0.3, 0.4, 0.5)]"),
    ],
)
def test_controlled_rotation_target_shows_parameters(
    name: str, params: tuple[float, ...], expected: str
) -> None:
    """Controlled-rotation targets display their angles (B83)."""
    circuit = QuantumCircuit()
    getattr(circuit, name)(*params, 0, 1)
    lines = draw_text_circuit(circuit).splitlines()
    target_line = next(line for line in lines if line.lstrip().startswith("q1"))
    control_line = next(line for line in lines if line.lstrip().startswith("q0"))
    assert expected in target_line
    # The control dot stays centred on the (wider) target box.
    center = target_line.index(expected) + len(expected) // 2
    assert control_line[center] == "o"


def test_controlled_rotations_with_different_angles_differ() -> None:
    """``crz(0.3)`` and ``crz(1.2)`` no longer render identically."""
    a = QuantumCircuit()
    a.crz(0.3, 0, 1)
    b = QuantumCircuit()
    b.crz(1.2, 0, 1)
    assert draw_text_circuit(a) != draw_text_circuit(b)
