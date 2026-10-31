from __future__ import annotations

import numpy as np
import pytest

from quspect.optimization.compiler import compile
from quspect.optimization.model import OptimizationSense
from quspect.optimization.program import QuadraticProgram


def test_compile_minimization_adds_squared_and_direct_penalties() -> None:
    program = QuadraticProgram()
    x = program.binary_variable("x")
    y = program.binary_variable("y")
    program.minimize(2 * x - y + 0.5)
    program.add_linear_equality(x + y - 1, name="one", penalty=5)
    program.add_nonnegative_quadratic_zero(x * y, name="pair", penalty=3)

    result = compile(program)

    assert result.variables == ("x", "y")
    assert result.model.sense is OptimizationSense.MINIMIZE
    np.testing.assert_allclose(result.model.linear, [-3, -6])
    np.testing.assert_allclose(result.model.quadratic, [[0, 13], [13, 0]])
    assert result.model.offset == 5.5
    assert result.evaluate_objective([1, 0]) == 2.5
    assert result.is_feasible([1, 0])
    assert not result.is_feasible([1, 1])
    assert result.decode([1, 0]) == {"x": 1, "y": 0}


def test_compile_maximization_reverses_penalty_sign() -> None:
    program = QuadraticProgram()
    x = program.binary_variable("x")
    program.maximize(2 * x)
    program.add_linear_equality(x - 1, name="required", penalty=4)

    result = compile(program)

    assert result.model.sense is OptimizationSense.MAXIMIZE
    assert result.model.linear.tolist() == [6.0]
    assert result.model.offset == -4.0
    assert result.model.evaluate([1]) == 2.0
    assert result.model.evaluate([0]) == -4.0


def test_compilation_result_validates_assignments() -> None:
    program = QuadraticProgram()
    x = program.binary_variable("x")
    program.minimize(x)
    result = compile(program)

    with pytest.raises(ValueError, match="Expected 1 bits"):
        result.decode([])
    with pytest.raises(ValueError, match="binary domain"):
        result.decode([2])
    with pytest.raises(ValueError, match="unknown variables"):
        result.evaluate_objective({"x": 1, "z": 0})


def test_compile_requires_an_objective() -> None:
    with pytest.raises(ValueError, match="no objective"):
        compile(QuadraticProgram())
