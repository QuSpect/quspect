from __future__ import annotations

import pytest

from quspect.optimization.expression import (
    BinaryVariable,
    QuadraticExpression,
    binary_array,
)


def test_binary_arithmetic_reduces_squares_and_evaluates() -> None:
    x = BinaryVariable("x")
    y = BinaryVariable("y")

    expression = 2 + 3 * x - y + x * y + x * x

    assert expression.constant == 2.0
    assert expression.linear == {x: 4.0, y: -1.0}
    assert expression.quadratic == {(x, y): 1.0}
    assert expression.evaluate({"x": 1, "y": 0}) == 6.0


def test_expression_is_immutable_and_rejects_cubic_terms() -> None:
    x = BinaryVariable("x")
    y = BinaryVariable("y")
    z = BinaryVariable("z")
    expression = x + y

    with pytest.raises(AttributeError):
        expression.constant = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="exceed quadratic"):
        _ = (x * y) * z


def test_linear_square_is_quadratic() -> None:
    x = BinaryVariable("x")
    y = BinaryVariable("y")

    expression = (x + y - 1) ** 2

    assert expression == QuadraticExpression(
        1,
        linear={x: -1, y: -1},
        quadratic={(x, y): 2},
    )


def test_binary_array_uses_row_major_names() -> None:
    variables = binary_array("x", (2, 3))

    assert variables.shape == (2, 3)
    assert [variable.name for variable in variables.flat] == [
        "x[0,0]",
        "x[0,1]",
        "x[0,2]",
        "x[1,0]",
        "x[1,1]",
        "x[1,2]",
    ]
    assert not variables.flags.writeable


@pytest.mark.parametrize(
    "operation",
    [
        lambda: BinaryVariable(" "),
        lambda: QuadraticExpression(float("inf")),
        lambda: binary_array("x", (2, 0)),
    ],
)
def test_strict_input_validation(operation: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        operation()  # type: ignore[operator]
