"""Tests for ``quspect.optimization.expression`` and ``program``.

Covers the symbolic binary-quadratic expression algebra (idempotent ``x*x=x``
reduction, degree checks, evaluation) and the ``QuadraticProgram`` builder,
including every documented error path.  Expressions are verified by brute
force against an independent Python polynomial evaluator.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from quspect.optimization import (
    BinaryVariable,
    LinearEqualityConstraint,
    NonnegativeQuadraticZeroConstraint,
    OptimizationSense,
    QuadraticExpression,
    QuadraticObjective,
    QuadraticProgram,
    as_expression,
    binary_array,
)


def _all_bits(n: int) -> list[tuple[int, ...]]:
    """Return every binary assignment of ``n`` variables."""
    return list(itertools.product((0, 1), repeat=n))


# ---------------------------------------------------------------------------
# BinaryVariable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", " x", "x ", 3, None])
def test_binary_variable_rejects_invalid_names(bad: object) -> None:
    """Empty, whitespace-padded and non-string names are rejected."""
    with pytest.raises(ValueError):
        BinaryVariable(bad)  # type: ignore[arg-type]


def test_binary_variable_operators_delegate_to_expression() -> None:
    """Every arithmetic dunder on a variable yields the expected expression."""
    x = BinaryVariable("x")
    y = BinaryVariable("y")
    assert (x + 1).evaluate({"x": 1}) == 2.0
    assert (1 + x).evaluate({"x": 0}) == 1.0
    assert (x - y).evaluate({"x": 0, "y": 1}) == -1.0
    assert (3 - x).evaluate({"x": 1}) == 2.0
    assert (x * 2).evaluate({"x": 1}) == 2.0
    assert (2 * x).evaluate({"x": 1}) == 2.0
    assert (-x).evaluate({"x": 1}) == -1.0
    assert (x**2) == x.as_expression()
    assert x < y  # dataclass(order=True)
    assert x.as_expression().linear == {x: 1.0}


def test_binary_variable_numpy_scalar_multiplication() -> None:
    """NumPy real scalars combine with variables like Python numbers."""
    x = BinaryVariable("x")
    expr = np.float64(2.5) * x
    assert isinstance(expr, QuadraticExpression)
    assert expr.linear == {x: 2.5}


# ---------------------------------------------------------------------------
# QuadraticExpression construction
# ---------------------------------------------------------------------------


def test_expression_constructor_merges_and_canonicalizes_terms() -> None:
    """Duplicate keys add, pair order is canonical and ``x*x`` becomes ``x``."""
    x = BinaryVariable("x")
    y = BinaryVariable("y")
    expr = QuadraticExpression(
        constant=1,
        linear=[("x", 1.0), (x, 2.0)],
        quadratic={("y", "x"): 1.5, (x, y): 0.5, ("x", "x"): 4.0},
    )
    assert expr.constant == 1.0
    assert expr.linear == {x: 7.0}
    assert expr.quadratic == {(x, y): 2.0}
    assert expr.variables == (x, y)
    assert expr.degree == 2


def test_expression_zero_coefficients_are_dropped() -> None:
    """Cancelling and zero terms disappear; degree and ``is_zero`` follow."""
    x = BinaryVariable("x")
    zero = QuadraticExpression(linear={"x": 1.0}) - x
    assert zero.is_zero
    assert zero.degree == 0
    assert zero.variables == ()
    expr = QuadraticExpression(linear={"x": 0.0}, quadratic={("x", "y"): 0.0})
    assert expr.is_zero
    assert not QuadraticExpression(2.0).is_zero
    assert QuadraticExpression(linear={"x": 1}).degree == 1


def test_expression_linear_and_quadratic_views_are_read_only() -> None:
    """``linear``/``quadratic`` return immutable mapping proxies."""
    expr = QuadraticExpression(linear={"x": 1.0}, quadratic={("x", "y"): 2.0})
    with pytest.raises(TypeError):
        expr.linear[BinaryVariable("z")] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        expr.quadratic[(BinaryVariable("a"), BinaryVariable("b"))] = 1.0  # type: ignore[index]


def test_expression_variable_order_is_first_appearance() -> None:
    """Variables keep first-appearance order, not alphabetical order."""
    expr = QuadraticExpression(linear={"z": 1.0, "a": 1.0})
    assert [v.name for v in expr.variables] == ["z", "a"]
    combined = BinaryVariable("b") + expr
    assert [v.name for v in combined.variables] == ["b", "z", "a"]


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"constant": True}, TypeError),
        ({"constant": "1"}, TypeError),
        ({"constant": math.inf}, ValueError),
        ({"linear": {"x": math.nan}}, ValueError),
        ({"linear": {3: 1.0}}, TypeError),
        ({"linear": {" x": 1.0}}, ValueError),
        ({"quadratic": {"x": 1.0}}, TypeError),
        ({"quadratic": {("x", "y", "z"): 1.0}}, TypeError),
        ({"quadratic": {("x", 1): 1.0}}, TypeError),
        ({"quadratic": {("x", "y"): False}}, TypeError),
    ],
)
def test_expression_constructor_errors(kwargs: dict[str, object], exc: type) -> None:
    """Invalid constants, coefficients and keys raise the documented errors."""
    with pytest.raises(exc):
        QuadraticExpression(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# QuadraticExpression arithmetic
# ---------------------------------------------------------------------------


def _random_expression(
    rng: np.random.Generator, names: list[str]
) -> QuadraticExpression:
    """Build a random quadratic expression over ``names``."""
    linear = {name: float(rng.normal()) for name in names}
    quadratic = {
        (a, b): float(rng.normal()) for a, b in itertools.combinations(names, 2)
    }
    return QuadraticExpression(float(rng.normal()), linear, quadratic)


def test_expression_arithmetic_matches_pointwise(rng: np.random.Generator) -> None:
    """Sum, difference, negation and scaling agree pointwise on all bits."""
    names = ["a", "b", "c", "d"]
    e1 = _random_expression(rng, names)
    e2 = _random_expression(rng, names[1:])
    a = BinaryVariable("a")
    for bits in _all_bits(4):
        env = dict(zip(names, bits, strict=True))
        v1 = e1.evaluate(env)
        v2 = e2.evaluate(env)
        assert math.isclose((e1 + e2).evaluate(env), v1 + v2, abs_tol=1e-12)
        assert math.isclose((e1 - e2).evaluate(env), v1 - v2, abs_tol=1e-12)
        assert math.isclose((-e1).evaluate(env), -v1, abs_tol=1e-12)
        assert math.isclose((2.5 * e1).evaluate(env), 2.5 * v1, abs_tol=1e-12)
        assert math.isclose((e1 + 1).evaluate(env), v1 + 1, abs_tol=1e-12)
        assert math.isclose((1 - e1).evaluate(env), 1 - v1, abs_tol=1e-12)
        lin = QuadraticExpression(e1.constant, e1.linear)
        assert math.isclose(
            (lin * a).evaluate(env), lin.evaluate(env) * env["a"], abs_tol=1e-12
        )


def test_expression_square_of_linear_matches_pointwise(
    rng: np.random.Generator,
) -> None:
    """``(linear)**2`` and ``linear*linear`` reproduce the squared values."""
    names = ["p", "q", "r"]
    lin = QuadraticExpression(
        float(rng.normal()), {n: float(rng.normal()) for n in names}
    )
    for bits in _all_bits(3):
        env = dict(zip(names, bits, strict=True))
        expected = lin.evaluate(env) ** 2
        assert math.isclose((lin**2).evaluate(env), expected, abs_tol=1e-12)
        assert math.isclose((lin * lin).evaluate(env), expected, abs_tol=1e-12)


def test_expression_powers() -> None:
    """``**0``, ``**1`` and constant powers behave; higher powers are rejected."""
    x = BinaryVariable("x")
    y = BinaryVariable("y")
    expr = x + y
    assert expr**0 == QuadraticExpression(1.0)
    assert expr**1 is expr
    assert QuadraticExpression(3.0) ** 3 == QuadraticExpression(27.0)
    # x**3 is still x by idempotence through repeated multiplication,
    # but ``__pow__`` only supports exponents up to two for non-constants.
    with pytest.raises(ValueError):
        expr**3
    with pytest.raises(ValueError):
        expr**-1
    with pytest.raises(TypeError):
        expr**1.5  # type: ignore[operator]
    with pytest.raises(TypeError):
        expr**True  # type: ignore[operator]


def test_expression_cubic_product_is_rejected() -> None:
    """A product introducing a genuine cubic monomial raises ``ValueError``."""
    x, y, z = (BinaryVariable(n) for n in "xyz")
    with pytest.raises(ValueError, match="quadratic degree"):
        (x * y) * z
    with pytest.raises(ValueError):
        (x + y) ** 2 * z
    # Idempotent reduction keeps x*y*x quadratic.
    assert ((x * y) * x).quadratic == {(x, y): 1.0}


def test_expression_unsupported_operands_return_not_implemented() -> None:
    """Non-numeric operands raise ``TypeError`` through ``NotImplemented``."""
    expr = BinaryVariable("x") + 1
    for op in (
        lambda: expr + "a",
        lambda: expr - "a",
        lambda: "a" - expr,
        lambda: expr * "a",
        lambda: expr + True,
    ):
        with pytest.raises(TypeError):
            op()


def test_expression_equality_and_hash() -> None:
    """Equal polynomials compare equal regardless of construction order."""
    e1 = QuadraticExpression(1.0, {"a": 1.0, "b": 2.0}, {("b", "a"): 3.0})
    e2 = QuadraticExpression(1.0, {"b": 2.0, "a": 1.0}, {("a", "b"): 3.0})
    assert e1 == e2
    assert hash(e1) == hash(e2)
    assert e1 != e1 + 1


# ---------------------------------------------------------------------------
# QuadraticExpression.evaluate
# ---------------------------------------------------------------------------


def test_evaluate_accepts_mapping_sequence_and_bool_like_values() -> None:
    """Assignments may be mappings by name/variable or ordered sequences."""
    x = BinaryVariable("x")
    expr = QuadraticExpression(0.5, {"x": 2.0, "y": -1.0}, {("x", "y"): 4.0})
    assert expr.evaluate({"x": 1, "y": 1}) == 5.5
    assert expr.evaluate({x: True, "y": np.bool_(True)}) == 5.5
    assert expr.evaluate([1.0, 0.0]) == 2.5
    assert expr.evaluate((0, 1)) == -0.5
    assert expr.evaluate(np.array([1, 1])) == 5.5
    # Extra variables in a mapping are ignored.
    assert expr.evaluate({"x": 0, "y": 0, "z": 1}) == 0.5


@pytest.mark.parametrize(
    ("assignment", "exc"),
    [
        ({"x": 1}, ValueError),  # missing y
        ({"x": 2, "y": 0}, ValueError),  # non-binary
        ({"x": math.nan, "y": 0}, ValueError),
        ({"x": "1", "y": 0}, ValueError),
        ({BinaryVariable("x"): 1, "x": 0, "y": 0}, ValueError),  # duplicate
        ({1: 0, "x": 0, "y": 0}, ValueError),  # invalid key
        ([1], ValueError),  # wrong length
        ("10", TypeError),
        (b"10", TypeError),
    ],
)
def test_evaluate_errors(assignment: object, exc: type) -> None:
    """Invalid assignments raise the documented exceptions."""
    expr = QuadraticExpression(linear={"x": 1.0, "y": 1.0})
    with pytest.raises(exc):
        expr.evaluate(assignment)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# as_expression / binary_array
# ---------------------------------------------------------------------------


def test_as_expression_coercions_and_errors() -> None:
    """Constants, variables and expressions coerce; others raise TypeError."""
    expr = QuadraticExpression(1.0)
    assert as_expression(expr) is expr
    assert as_expression(BinaryVariable("x")).linear == {BinaryVariable("x"): 1.0}
    assert as_expression(3) == QuadraticExpression(3.0)
    assert as_expression(np.int64(2)) == QuadraticExpression(2.0)
    for bad in (True, "x", None, [1]):
        with pytest.raises(TypeError):
            as_expression(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        as_expression(math.inf)


def test_binary_array_row_major_names_and_read_only() -> None:
    """Names follow ``name[i,j]`` in C order and the array is read-only."""
    arr = binary_array("x", (2, 3))
    assert arr.shape == (2, 3)
    assert [v.name for v in arr.flat] == [
        f"x[{i},{j}]" for i in range(2) for j in range(3)
    ]
    assert not arr.flags.writeable
    vec = binary_array("y", 3)
    assert [v.name for v in vec] == ["y[0]", "y[1]", "y[2]"]


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("", 2),
        (" x", 2),
        ("x", 0),
        ("x", -1),
        ("x", True),
        ("x", ()),
        ("x", [2, 3]),
        ("x", (2, 0)),
        ("x", (2, 1.5)),
        ("x", (2, True)),
    ],
)
def test_binary_array_errors(name: str, shape: object) -> None:
    """Invalid names or shapes raise ``ValueError``."""
    with pytest.raises(ValueError):
        binary_array(name, shape)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# QuadraticProgram
# ---------------------------------------------------------------------------


def test_program_construction_and_properties() -> None:
    """Variables, objective and constraints are tracked in stable order."""
    program = QuadraticProgram("demo")
    assert program.name == "demo"
    assert program.objective is None
    assert program.constraints == ()
    a = program.binary_variable("a")
    b = program.binary_var("b")
    xs = program.binary_array("x", 2)
    z = BinaryVariable("z")  # undeclared, registered on use
    program.minimize(a + 2 * b + xs[0] * z)
    assert [v.name for v in program.variables] == ["a", "b", "x[0]", "x[1]", "z"]
    objective = program.objective
    assert isinstance(objective, QuadraticObjective)
    assert objective.sense is OptimizationSense.MINIMIZE
    program.maximize(a)
    assert program.objective is not None
    assert program.objective.sense is OptimizationSense.MAXIMIZE
    program.maximize(3)
    assert program.objective is not None
    assert program.objective.expression == QuadraticExpression(3.0)

    eq = program.add_linear_equality(a + b - 1, name="one", penalty=2)
    assert isinstance(eq, LinearEqualityConstraint)
    assert eq.expression is eq.residual
    assert eq.penalty == 2.0
    w = BinaryVariable("w")
    quad = program.add_nonnegative_quadratic_zero(a * w, name="q", penalty=1.5)
    assert isinstance(quad, NonnegativeQuadraticZeroConstraint)
    eq2 = program.add_linear_equality_constraint(a, name="eq2", penalty=1)
    quad2 = program.add_quadratic_zero_constraint(b * a, name="q2", penalty=1)
    assert program.constraints == (eq, quad, eq2, quad2)
    assert program.variables[-1] == w
    assert QuadraticProgram().name is None


@pytest.mark.parametrize("bad", ["", "  ", " p", 5])
def test_program_name_validation(bad: object) -> None:
    """Program names must be non-empty and whitespace-trimmed strings."""
    with pytest.raises(ValueError):
        QuadraticProgram(bad)  # type: ignore[arg-type]


def test_program_duplicate_declarations_are_atomic() -> None:
    """Duplicate variables raise; a conflicting array registers nothing."""
    program = QuadraticProgram()
    program.binary_variable("x[1]")
    with pytest.raises(ValueError, match="already declared"):
        program.binary_variable("x[1]")
    with pytest.raises(ValueError, match="already declared"):
        program.binary_array("x", 3)
    assert [v.name for v in program.variables] == ["x[1]"]


def test_program_constraint_errors() -> None:
    """Constraint names, penalties and degrees are validated."""
    program = QuadraticProgram()
    x = program.binary_variable("x")
    y = program.binary_variable("y")
    program.add_linear_equality(x - 1, name="c", penalty=1.0)
    with pytest.raises(ValueError, match="already exists"):
        program.add_linear_equality(x, name="c", penalty=1.0)
    with pytest.raises(ValueError, match="already exists"):
        program.add_nonnegative_quadratic_zero(x * y, name="c", penalty=1.0)
    with pytest.raises(ValueError, match="degree"):
        program.add_linear_equality(x * y, name="d", penalty=1.0)
    with pytest.raises(ValueError):
        program.add_linear_equality(x, name=" d", penalty=1.0)
    for bad_penalty, exc in [
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
        (True, TypeError),
        ("1", TypeError),
    ]:
        with pytest.raises(exc):
            program.add_linear_equality(x, name="p", penalty=bad_penalty)  # type: ignore[arg-type]
        with pytest.raises(exc):
            program.add_nonnegative_quadratic_zero(x, name="p", penalty=bad_penalty)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        program.minimize("x")  # type: ignore[arg-type]
    # Failed additions must not have registered constraints.
    assert [c.name for c in program.constraints] == ["c"]
