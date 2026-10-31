r"""Symbolic parameters and affine parameter expressions.

``Parameter`` is a named placeholder.  Arithmetic on parameters produces an
immutable ``ParameterExpression`` of the form

.. math:: c + \sum_i a_i \theta_i.

Restricting expressions to this affine form is intentional: it gives the C++
simulator a small, deterministic Jacobian that can be applied after computing
the derivative of a gate with respect to its local angle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, order=True)
class Parameter:
    """A named placeholder used by parameterized quantum circuits.

    Parameters compare and hash by name.  Arithmetic with another parameter or
    a real scalar creates a :class:`ParameterExpression` without mutating the
    original object.
    """

    name: str

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"

    def as_expression(self) -> ParameterExpression:
        """Return this parameter as the affine expression ``1 * self``."""
        return ParameterExpression({self: 1.0})

    def __add__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return self.as_expression() + expression

    def __radd__(self, other: object) -> ParameterExpression:
        return self + other

    def __sub__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return self.as_expression() - expression

    def __rsub__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return expression - self.as_expression()

    def __neg__(self) -> ParameterExpression:
        return -self.as_expression()

    def __mul__(self, other: object) -> Parameter | ParameterExpression:
        # Preserve the long-standing ``1 * parameter is parameter`` identity.
        # Operator coefficients rely on this canonical form.
        if isinstance(other, Real) and float(other) == 1.0:
            return self
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return self.as_expression() * expression

    def __rmul__(self, other: object) -> Parameter | ParameterExpression:
        return self * other

    def __truediv__(self, other: object) -> Parameter | ParameterExpression:
        if isinstance(other, Real) and float(other) == 1.0:
            return self
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return self.as_expression() / expression

    def __rtruediv__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return expression / self.as_expression()


TermInput = Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None


@dataclass(frozen=True)
class ParameterExpression:
    """An immutable affine expression over named parameters.

    Args:
        terms: Parameter-to-coefficient mapping (or iterable of pairs).
            Strings are accepted as a convenience and converted to
            :class:`Parameter` objects.
        constant: The real constant offset.

    Multiplication and division are deliberately limited to real scalars (or
    constant expressions).  Multiplying two expressions that both contain
    parameters would create a non-affine expression and therefore raises
    ``TypeError``.

    Example:
        >>> theta = Parameter("theta")
        >>> phi = Parameter("phi")
        >>> expr = 2 * theta - 0.5 * phi + 1
        >>> expr.evaluate({theta: 0.25, phi: 1.0})
        1.0
    """

    constant: float = 0.0
    _term_items: tuple[tuple[Parameter, float], ...] = field(
        default_factory=tuple, repr=False
    )

    def __init__(self, terms: TermInput = None, *, constant: int | float = 0.0) -> None:
        constant_value = _require_real(constant, "constant")
        coefficients: dict[Parameter, float] = {}

        if terms is not None:
            items = terms.items() if isinstance(terms, Mapping) else terms
            for raw_parameter, raw_coefficient in items:
                parameter = (
                    raw_parameter
                    if isinstance(raw_parameter, Parameter)
                    else Parameter(raw_parameter)
                    if isinstance(raw_parameter, str)
                    else None
                )
                if parameter is None:
                    raise TypeError(
                        "Expression term keys must be Parameter or str, "
                        f"got {type(raw_parameter).__name__}."
                    )
                coefficient = _require_real(raw_coefficient, "coefficient")
                coefficients[parameter] = coefficients.get(parameter, 0.0) + coefficient

        # Removing exact zeroes keeps equality, hashing and the parameter set
        # canonical (e.g. ``theta - theta`` has no remaining parameters).
        term_items = tuple(
            sorted(
                (
                    (parameter, coefficient)
                    for parameter, coefficient in coefficients.items()
                    if coefficient != 0.0
                ),
                key=lambda item: item[0].name,
            )
        )
        canonical_constant = 0.0 if constant_value == 0.0 else constant_value
        object.__setattr__(self, "constant", canonical_constant)
        object.__setattr__(self, "_term_items", term_items)

    @property
    def terms(self) -> Mapping[Parameter, float]:
        """A read-only parameter-to-coefficient mapping."""
        return MappingProxyType(dict(self._term_items))

    @property
    def parameters(self) -> frozenset[Parameter]:
        """Parameters with a non-zero coefficient in this expression."""
        return frozenset(parameter for parameter, _ in self._term_items)

    @property
    def is_constant(self) -> bool:
        """Whether this expression contains no symbolic parameters."""
        return not self._term_items

    def coefficient(self, parameter: Parameter) -> float:
        """Return the coefficient of ``parameter`` (zero when absent)."""
        return self.terms.get(parameter, 0.0)

    def evaluate(self, binding: Mapping[Parameter, int | float]) -> float:
        """Evaluate the expression using a complete parameter binding.

        Raises:
            ValueError: If any parameter in the expression is not bound.
            TypeError: If a bound value is not real.
        """
        missing = [
            parameter for parameter, _ in self._term_items if parameter not in binding
        ]
        if missing:
            names = ", ".join(repr(parameter) for parameter in missing)
            raise ValueError(f"Parameter(s) {names} are not bound.")

        result = self.constant
        for parameter, coefficient in self._term_items:
            value = _require_real(binding[parameter], f"binding for {parameter!r}")
            result += coefficient * value
        return float(result)

    def bind(self, binding: Mapping[Parameter, int | float]) -> float:
        """Alias for :meth:`evaluate`."""
        return self.evaluate(binding)

    def __float__(self) -> float:
        if not self.is_constant:
            raise TypeError("Cannot convert a symbolic ParameterExpression to float.")
        return self.constant

    def __repr__(self) -> str:
        terms = (
            "{"
            + ", ".join(
                f"{parameter!r}: {coefficient!r}"
                for parameter, coefficient in self._term_items
            )
            + "}"
        )
        return f"ParameterExpression({terms}, constant={self.constant!r})"

    def __str__(self) -> str:
        pieces: list[str] = []
        for parameter, coefficient in self._term_items:
            if coefficient == 1.0:
                pieces.append(parameter.name)
            elif coefficient == -1.0:
                pieces.append(f"-{parameter.name}")
            else:
                pieces.append(f"{coefficient:g}*{parameter.name}")
        if self.constant != 0.0 or not pieces:
            pieces.append(f"{self.constant:g}")
        return " + ".join(pieces).replace(" + -", " - ")

    def __add__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        coefficients = dict(self._term_items)
        for parameter, coefficient in expression._term_items:
            coefficients[parameter] = coefficients.get(parameter, 0.0) + coefficient
        return ParameterExpression(
            coefficients,
            constant=self.constant + expression.constant,
        )

    def __radd__(self, other: object) -> ParameterExpression:
        return self + other

    def __sub__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return self + (-expression)

    def __rsub__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        return expression - self

    def __neg__(self) -> ParameterExpression:
        return self._scaled(-1.0)

    def __mul__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        if not self.is_constant and not expression.is_constant:
            raise TypeError(
                "Multiplication of two non-constant parameter expressions "
                "is not supported."
            )
        if self.is_constant:
            return expression._scaled(self.constant)
        return self._scaled(expression.constant)

    def __rmul__(self, other: object) -> ParameterExpression:
        return self * other

    def __truediv__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        if not expression.is_constant:
            raise TypeError(
                "Division by a non-constant parameter expression is not supported."
            )
        if expression.constant == 0.0:
            raise ZeroDivisionError("division by zero")
        return self._scaled(1.0 / expression.constant)

    def __rtruediv__(self, other: object) -> ParameterExpression:
        expression = _as_expression(other)
        if expression is None:
            return NotImplemented
        if not self.is_constant:
            raise TypeError(
                "Division by a non-constant parameter expression is not supported."
            )
        if self.constant == 0.0:
            raise ZeroDivisionError("division by zero")
        return expression._scaled(1.0 / self.constant)

    def _scaled(self, scalar: float) -> ParameterExpression:
        return ParameterExpression(
            (
                (parameter, coefficient * scalar)
                for parameter, coefficient in self._term_items
            ),
            constant=self.constant * scalar,
        )


def _require_real(value: object, description: str) -> float:
    """Convert a real scalar to ``float`` with a useful error message."""
    if not isinstance(value, Real):
        raise TypeError(f"Expression {description} must be real, got {value!r}.")
    return float(value)


def _as_expression(value: object) -> ParameterExpression | None:
    if isinstance(value, ParameterExpression):
        return value
    if isinstance(value, Parameter):
        return value.as_expression()
    if isinstance(value, Real):
        return ParameterExpression(constant=float(value))
    if isinstance(value, complex):
        raise TypeError("Parameter expressions only support real scalars.")
    return None


#: A gate parameter: a concrete value, named parameter, or affine expression.
ParamValue = float | Parameter | ParameterExpression
