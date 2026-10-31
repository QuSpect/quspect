"""Immutable symbolic expressions for binary quadratic optimization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

Scalar: TypeAlias = int | float
LinearInput: TypeAlias = (
    Mapping["BinaryVariable | str", Scalar]
    | Iterable[tuple["BinaryVariable | str", Scalar]]
)
QuadraticKey: TypeAlias = tuple["BinaryVariable | str", "BinaryVariable | str"]
QuadraticInput: TypeAlias = (
    Mapping[QuadraticKey, Scalar] | Iterable[tuple[QuadraticKey, Scalar]]
)
ExpressionLike: TypeAlias = "QuadraticExpression | BinaryVariable | int | float"
Monomial: TypeAlias = tuple["BinaryVariable", ...]


def _validate_name(name: object, *, label: str = "Variable name") -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    if name != name.strip():
        raise ValueError(f"{label} {name!r} contains outer whitespace.")
    return name


def _coerce_coefficient(value: object, *, label: str = "Coefficient") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}.")
    coefficient = float(value)
    if not math.isfinite(coefficient):
        raise ValueError(f"{label} must be finite.")
    return coefficient


def _coerce_bit(value: object, *, label: str) -> int:
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number in (0.0, 1.0):
            return int(number)
    raise ValueError(f"{label} must be in the binary domain {{0, 1}}.")


@dataclass(frozen=True, order=True)
class BinaryVariable:
    """A named variable whose value is restricted to ``0`` or ``1``."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))

    def as_expression(self) -> QuadraticExpression:
        """Return this variable as a linear expression."""
        return QuadraticExpression(linear=((self, 1.0),))

    def __add__(self, other: object) -> QuadraticExpression:
        return self.as_expression() + other

    def __radd__(self, other: object) -> QuadraticExpression:
        return self.as_expression().__radd__(other)

    def __sub__(self, other: object) -> QuadraticExpression:
        return self.as_expression() - other

    def __rsub__(self, other: object) -> QuadraticExpression:
        return self.as_expression().__rsub__(other)

    def __mul__(self, other: object) -> QuadraticExpression:
        return self.as_expression() * other

    def __rmul__(self, other: object) -> QuadraticExpression:
        return self.as_expression().__rmul__(other)

    def __neg__(self) -> QuadraticExpression:
        return -self.as_expression()

    def __pow__(self, exponent: object) -> QuadraticExpression:
        return self.as_expression() ** exponent


def _coerce_variable(value: BinaryVariable | str) -> BinaryVariable:
    if isinstance(value, BinaryVariable):
        return value
    if isinstance(value, str):
        return BinaryVariable(value)
    raise TypeError("Expression term keys must be BinaryVariable instances or names.")


def _canonical_monomial(variables: Iterable[BinaryVariable]) -> Monomial:
    # Binary variables are idempotent: x * x == x.
    by_name = {variable.name: variable for variable in variables}
    return tuple(by_name[name] for name in sorted(by_name))


def _iter_items(
    value: Mapping[object, Scalar] | Iterable[tuple[object, Scalar]],
) -> Iterable[tuple[object, Scalar]]:
    return value.items() if isinstance(value, Mapping) else value


@dataclass(frozen=True, init=False)
class QuadraticExpression:
    """An immutable polynomial of degree at most two in binary variables.

    Repeated binary factors are reduced using ``x * x = x``.  Arithmetic is
    closed over constants, variables, and quadratic expressions; a product
    that would introduce a genuine cubic term raises ``ValueError``.
    """

    constant: float
    _linear: tuple[tuple[BinaryVariable, float], ...]
    _quadratic: tuple[tuple[tuple[BinaryVariable, BinaryVariable], float], ...]
    _variables: tuple[BinaryVariable, ...] = field(compare=False, repr=False)

    def __init__(
        self,
        constant: Scalar = 0.0,
        linear: LinearInput = (),
        quadratic: QuadraticInput = (),
    ) -> None:
        polynomial: dict[Monomial, float] = {
            (): _coerce_coefficient(constant, label="Constant")
        }
        variable_order: list[BinaryVariable] = []

        def remember(variable: BinaryVariable) -> None:
            if variable not in variable_order:
                variable_order.append(variable)

        for raw_variable, raw_coefficient in _iter_items(
            cast(
                Mapping[object, Scalar] | Iterable[tuple[object, Scalar]],
                linear,
            )
        ):
            variable = _coerce_variable(cast(BinaryVariable | str, raw_variable))
            coefficient = _coerce_coefficient(raw_coefficient)
            remember(variable)
            monomial: Monomial = (variable,)
            polynomial[monomial] = polynomial.get(monomial, 0.0) + coefficient

        for raw_pair, raw_coefficient in _iter_items(
            cast(
                Mapping[object, Scalar] | Iterable[tuple[object, Scalar]],
                quadratic,
            )
        ):
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise TypeError("Quadratic term keys must be pairs of variables.")
            left = _coerce_variable(
                cast(tuple[BinaryVariable | str, BinaryVariable | str], raw_pair)[0]
            )
            right = _coerce_variable(
                cast(tuple[BinaryVariable | str, BinaryVariable | str], raw_pair)[1]
            )
            coefficient = _coerce_coefficient(raw_coefficient)
            remember(left)
            remember(right)
            monomial = _canonical_monomial((left, right))
            polynomial[monomial] = polynomial.get(monomial, 0.0) + coefficient

        normalized = self._normalize_polynomial(polynomial)
        self._set_from_polynomial(normalized, tuple(variable_order))

    @staticmethod
    def _normalize_polynomial(
        polynomial: Mapping[Monomial, float],
    ) -> dict[Monomial, float]:
        normalized: dict[Monomial, float] = {}
        for raw_monomial, raw_coefficient in polynomial.items():
            monomial = _canonical_monomial(raw_monomial)
            if len(monomial) > 2:
                names = " * ".join(variable.name for variable in monomial)
                raise ValueError(
                    f"Binary quadratic expressions cannot contain term {names}."
                )
            coefficient = _coerce_coefficient(raw_coefficient)
            if coefficient == 0.0:
                continue
            normalized[monomial] = normalized.get(monomial, 0.0) + coefficient
        return {
            monomial: coefficient
            for monomial, coefficient in normalized.items()
            if coefficient != 0.0
        }

    def _set_from_polynomial(
        self,
        polynomial: Mapping[Monomial, float],
        variable_order: tuple[BinaryVariable, ...],
    ) -> None:
        constant = polynomial.get((), 0.0)
        linear = tuple(
            sorted(
                (
                    (monomial[0], coefficient)
                    for monomial, coefficient in polynomial.items()
                    if len(monomial) == 1
                ),
                key=lambda item: item[0].name,
            )
        )
        quadratic = tuple(
            sorted(
                (
                    ((monomial[0], monomial[1]), coefficient)
                    for monomial, coefficient in polynomial.items()
                    if len(monomial) == 2
                ),
                key=lambda item: (item[0][0].name, item[0][1].name),
            )
        )
        used = {variable for monomial in polynomial for variable in monomial}
        ordered = tuple(variable for variable in variable_order if variable in used)
        missing = sorted(used.difference(ordered), key=lambda variable: variable.name)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "_linear", linear)
        object.__setattr__(self, "_quadratic", quadratic)
        object.__setattr__(self, "_variables", ordered + tuple(missing))

    @classmethod
    def _from_polynomial(
        cls,
        polynomial: Mapping[Monomial, float],
        variable_order: Iterable[BinaryVariable],
    ) -> QuadraticExpression:
        expression = object.__new__(cls)
        normalized = cls._normalize_polynomial(polynomial)
        expression._set_from_polynomial(normalized, tuple(variable_order))
        return expression

    @property
    def linear(self) -> Mapping[BinaryVariable, float]:
        """Return a read-only variable-to-linear-coefficient mapping."""
        return MappingProxyType(dict(self._linear))

    @property
    def quadratic(
        self,
    ) -> Mapping[tuple[BinaryVariable, BinaryVariable], float]:
        """Return a read-only pair-to-quadratic-coefficient mapping."""
        return MappingProxyType(dict(self._quadratic))

    @property
    def variables(self) -> tuple[BinaryVariable, ...]:
        """Return variables in stable first-appearance order."""
        return self._variables

    @property
    def degree(self) -> int:
        """Return the polynomial degree, with constants having degree zero."""
        if self._quadratic:
            return 2
        if self._linear:
            return 1
        return 0

    @property
    def is_zero(self) -> bool:
        """Return whether every coefficient is zero."""
        return self.constant == 0.0 and not self._linear and not self._quadratic

    def _polynomial(self) -> dict[Monomial, float]:
        polynomial: dict[Monomial, float] = {}
        if self.constant != 0.0:
            polynomial[()] = self.constant
        polynomial.update(
            {(variable,): coefficient for variable, coefficient in self._linear}
        )
        polynomial.update(dict(self._quadratic))
        return polynomial

    def _coerce_operand(self, other: object) -> QuadraticExpression | None:
        if isinstance(other, QuadraticExpression):
            return other
        if isinstance(other, BinaryVariable):
            return other.as_expression()
        if isinstance(other, Real) and not isinstance(other, bool):
            return QuadraticExpression(float(other))
        return None

    def _merged_order(self, other: QuadraticExpression) -> tuple[BinaryVariable, ...]:
        return self.variables + tuple(
            variable for variable in other.variables if variable not in self.variables
        )

    def __add__(self, other: object) -> QuadraticExpression:
        right = self._coerce_operand(other)
        if right is None:
            return NotImplemented
        polynomial = self._polynomial()
        for monomial, coefficient in right._polynomial().items():
            polynomial[monomial] = polynomial.get(monomial, 0.0) + coefficient
        return self._from_polynomial(polynomial, self._merged_order(right))

    def __radd__(self, other: object) -> QuadraticExpression:
        return self.__add__(other)

    def __neg__(self) -> QuadraticExpression:
        return self._from_polynomial(
            {
                monomial: -coefficient
                for monomial, coefficient in self._polynomial().items()
            },
            self.variables,
        )

    def __sub__(self, other: object) -> QuadraticExpression:
        right = self._coerce_operand(other)
        if right is None:
            return NotImplemented
        return self + (-right)

    def __rsub__(self, other: object) -> QuadraticExpression:
        left = self._coerce_operand(other)
        if left is None:
            return NotImplemented
        return left - self

    def __mul__(self, other: object) -> QuadraticExpression:
        right = self._coerce_operand(other)
        if right is None:
            return NotImplemented
        polynomial: dict[Monomial, float] = {}
        for left_monomial, left_coefficient in self._polynomial().items():
            for right_monomial, right_coefficient in right._polynomial().items():
                monomial = _canonical_monomial(left_monomial + right_monomial)
                if len(monomial) > 2:
                    names = " * ".join(variable.name for variable in monomial)
                    raise ValueError(
                        "Multiplication would exceed quadratic degree "
                        f"with term {names}."
                    )
                coefficient = left_coefficient * right_coefficient
                polynomial[monomial] = polynomial.get(monomial, 0.0) + coefficient
        return self._from_polynomial(polynomial, self._merged_order(right))

    def __rmul__(self, other: object) -> QuadraticExpression:
        return self.__mul__(other)

    def __pow__(self, exponent: object) -> QuadraticExpression:
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("Expression exponent must be a non-negative integer.")
        if exponent < 0:
            raise ValueError("Expression exponent must be non-negative.")
        if exponent == 0:
            return QuadraticExpression(1.0)
        if exponent == 1:
            return self
        if exponent == 2:
            return self * self
        if self.degree == 0:
            return QuadraticExpression(self.constant**exponent)
        raise ValueError("Powers above two exceed quadratic degree.")

    def evaluate(
        self,
        assignment: (
            Mapping[str, object] | Mapping[BinaryVariable, object] | Sequence[object]
        ),
    ) -> float:
        """Evaluate the expression at a complete binary assignment."""
        values: dict[str, int]
        if isinstance(assignment, Mapping):
            values = {}
            for key, value in assignment.items():
                name = (
                    key.name if isinstance(key, BinaryVariable) else _validate_name(key)
                )
                if name in values:
                    raise ValueError(
                        f"Assignment contains duplicate variable {name!r}."
                    )
                values[name] = _coerce_bit(value, label=f"Value for {name!r}")
            missing = [
                variable.name
                for variable in self.variables
                if variable.name not in values
            ]
            if missing:
                raise ValueError(
                    "Assignment is missing variables: " + ", ".join(missing) + "."
                )
        else:
            if isinstance(assignment, str | bytes):
                raise TypeError("Assignment must be a mapping or a sequence of bits.")
            raw_values = list(assignment)
            if len(raw_values) != len(self.variables):
                raise ValueError(
                    f"Expected {len(self.variables)} values, got {len(raw_values)}."
                )
            values = {
                variable.name: _coerce_bit(value, label=f"Value for {variable.name!r}")
                for variable, value in zip(self.variables, raw_values, strict=True)
            }

        result = self.constant
        result += sum(
            coefficient * values[variable.name]
            for variable, coefficient in self._linear
        )
        result += sum(
            coefficient * values[left.name] * values[right.name]
            for (left, right), coefficient in self._quadratic
        )
        return float(result)


def as_expression(value: ExpressionLike) -> QuadraticExpression:
    """Coerce a constant or binary variable to a quadratic expression."""
    if isinstance(value, QuadraticExpression):
        return value
    if isinstance(value, BinaryVariable):
        return value.as_expression()
    if isinstance(value, Real) and not isinstance(value, bool):
        return QuadraticExpression(float(value))
    raise TypeError(
        "Expected a QuadraticExpression, BinaryVariable, or finite real constant."
    )


def binary_array(
    name: str,
    shape: int | tuple[int, ...],
) -> NDArray[np.object_]:
    """Create a read-only NumPy array of variables in row-major name order.

    A two-by-three array named ``x`` contains ``x[0,0]`` through ``x[1,2]``;
    flattening it in the default C order preserves that same sequence.
    """
    prefix = _validate_name(name, label="Array name")
    dimensions = (
        (shape,) if isinstance(shape, int) and not isinstance(shape, bool) else shape
    )
    if not isinstance(dimensions, tuple) or not dimensions:
        raise ValueError("shape must be a positive integer or a non-empty tuple.")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError("Every array dimension must be a positive integer.")

    variables: NDArray[np.object_] = np.empty(dimensions, dtype=object)
    for index in np.ndindex(dimensions):
        suffix = ",".join(str(component) for component in index)
        variables[index] = BinaryVariable(f"{prefix}[{suffix}]")
    variables.setflags(write=False)
    return variables
