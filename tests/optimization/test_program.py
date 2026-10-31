from __future__ import annotations

import pytest

from quspect.optimization.model import OptimizationSense
from quspect.optimization.program import (
    LinearEqualityConstraint,
    NonnegativeQuadraticZeroConstraint,
    QuadraticProgram,
)


def test_program_tracks_stable_variable_order_and_objective() -> None:
    program = QuadraticProgram("selection")
    x = program.binary_array("x", 3)

    program.maximize(2 * x[2] + x[0])

    assert tuple(variable.name for variable in program.variables) == (
        "x[0]",
        "x[1]",
        "x[2]",
    )
    assert program.objective is not None
    assert program.objective.sense is OptimizationSense.MAXIMIZE


def test_program_adds_named_supported_constraints() -> None:
    program = QuadraticProgram()
    x = program.binary_variable("x")
    y = program.binary_variable("y")
    linear = program.add_linear_equality(
        x + y - 1,
        name="one",
        penalty=4,
    )
    quadratic = program.add_nonnegative_quadratic_zero(
        x * y,
        name="exclusive",
        penalty=3,
    )

    assert isinstance(linear, LinearEqualityConstraint)
    assert isinstance(quadratic, NonnegativeQuadraticZeroConstraint)
    assert program.constraints == (linear, quadratic)


def test_program_rejects_invalid_constraints() -> None:
    program = QuadraticProgram()
    x = program.binary_variable("x")
    y = program.binary_variable("y")

    with pytest.raises(ValueError, match="degree"):
        program.add_linear_equality(x * y, name="bad", penalty=1)
    with pytest.raises(ValueError, match="strictly positive"):
        program.add_nonnegative_quadratic_zero(x, name="bad", penalty=0)

    program.add_linear_equality(x - y, name="unique", penalty=1)
    with pytest.raises(ValueError, match="already exists"):
        program.add_linear_equality(x, name="unique", penalty=1)
