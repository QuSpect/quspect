"""End-to-end connectivity tests for the complete built-in gate set."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from quspect import Parameter, QuantumCircuit, QubitOperator, Simulator
from quspect.ir.compilation import decompose_deep
from quspect.variational import AdjointEngine, ObservableObjective, SamplingEngine

_ADJOINT_BASIS = {
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "cx",
    "cz",
    "swap",
    "ccx",
    "rx",
    "ry",
    "rz",
}


@pytest.mark.parametrize(
    "builder",
    [
        lambda circuit: circuit.u1(0.31, 0),
        lambda circuit: circuit.u2(0.2, -0.4, 0),
        lambda circuit: circuit.cry(0.3, 0, 1),
        lambda circuit: circuit.cu3(0.3, 0.2, -0.4, 0, 1),
        lambda circuit: circuit.cp(0.3, 0, 1),
    ],
)
def test_newly_connected_gates_match_basis_decomposition(
    builder: Callable[[QuantumCircuit], QuantumCircuit],
) -> None:
    circuit = QuantumCircuit().h(0).h(1)
    builder(circuit)
    decomposed = decompose_deep(circuit, supported_gates=_ADJOINT_BASIS)

    simulator = Simulator()
    actual = np.asarray(simulator.statevector(circuit))
    expected = np.asarray(simulator.statevector(decomposed))
    pivot = int(np.argmax(np.abs(expected)))
    phase = actual[pivot] / expected[pivot]
    np.testing.assert_allclose(actual, phase * expected, atol=1e-12)


def test_cp_has_expected_statevector() -> None:
    theta = 0.73
    circuit = QuantumCircuit().h(0).h(1).cp(theta, 0, 1)
    expected = np.array([1, 1, 1, np.exp(1j * theta)]) / 2
    np.testing.assert_allclose(Simulator().statevector(circuit), expected, atol=1e-12)


def test_symbolic_cp_is_supported_by_variational_engines() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().h(0).h(1).cp(theta, 0, 1)
    objective = ObservableObjective(
        circuit,
        {"x1": QubitOperator({((1, "X"),): 1.0})},
    )

    point = 0.4
    result = AdjointEngine().compile(objective).evaluate({theta: point})
    assert result.values[0] == pytest.approx((1 + math.cos(point)) / 2)
    assert result.jacobian[0, 0] == pytest.approx(-math.sin(point) / 2)

    sampled = SamplingEngine(Simulator(), shots=100).compile(objective)
    assert sampled.parameter_order == (theta,)
