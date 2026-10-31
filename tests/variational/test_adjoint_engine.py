"""Validation tests for :class:`quspect.variational.AdjointEngine`."""

from __future__ import annotations

import math

import pytest

from quspect import Parameter, ParameterExpression, QuantumCircuit, QubitOperator
from quspect.variational import AdjointEngine, ObservableObjective

_A = Parameter("a")
_B = Parameter("b")


@pytest.mark.parametrize(
    ("angle", "message"),
    [
        (math.inf, "non-finite angle"),
        (math.nan, "non-finite angle"),
        (ParameterExpression({}, constant=math.inf), "non-finite constant"),
        (_A + math.inf, "non-finite offset"),
        (ParameterExpression({_A: math.nan}), "non-finite affine coefficient"),
    ],
)
def test_compile_rejects_non_finite_gate_parameters(
    angle: object, message: str
) -> None:
    circuit = QuantumCircuit().ry(_B, 0).rx(angle, 0)  # type: ignore[arg-type]
    objective = ObservableObjective(circuit, {"Z": QubitOperator({((0, "Z"),): 1.0})})
    with pytest.raises(ValueError, match=message):
        AdjointEngine().compile(objective)
