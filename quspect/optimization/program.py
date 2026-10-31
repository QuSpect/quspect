"""User-facing construction of binary quadratic programs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .expression import (
    BinaryVariable,
    ExpressionLike,
    QuadraticExpression,
    as_expression,
    binary_array,
)
from .model import OptimizationSense


def _validate_program_name(name: object, *, optional: bool = False) -> str | None:
    if optional and name is None:
        return None
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Name must be a non-empty string.")
    if name != name.strip():
        raise ValueError(f"Name {name!r} contains outer whitespace.")
    return name


def _validate_penalty(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Penalty must be a real number.")
    penalty = float(value)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("Penalty must be finite and strictly positive.")
    return penalty


@dataclass(frozen=True)
class QuadraticObjective:
    """The original, unpenalized objective and its optimization direction."""

    expression: QuadraticExpression
    sense: OptimizationSense


@dataclass(frozen=True)
class LinearEqualityConstraint:
    """A constraint requiring a linear residual to equal zero."""

    name: str
    residual: QuadraticExpression
    penalty: float

    @property
    def expression(self) -> QuadraticExpression:
        """Return the zero-valued expression constrained by this object."""
        return self.residual


@dataclass(frozen=True)
class NonnegativeQuadraticZeroConstraint:
    """A declared-nonnegative quadratic expression required to equal zero."""

    name: str
    expression: QuadraticExpression
    penalty: float


ProgramConstraint: TypeAlias = (
    LinearEqualityConstraint | NonnegativeQuadraticZeroConstraint
)


class QuadraticProgram:
    """Build an objective and soft-penalty constraints in stable variable order."""

    def __init__(self, name: str | None = None) -> None:
        self._name = _validate_program_name(name, optional=True)
        self._variables: dict[str, BinaryVariable] = {}
        self._objective: QuadraticObjective | None = None
        self._constraints: list[ProgramConstraint] = []
        self._constraint_names: set[str] = set()

    @property
    def name(self) -> str | None:
        """Return the optional program name."""
        return self._name

    @property
    def variables(self) -> tuple[BinaryVariable, ...]:
        """Return declared and encountered variables in stable order."""
        return tuple(self._variables.values())

    @property
    def objective(self) -> QuadraticObjective | None:
        """Return the objective, or ``None`` before one has been supplied."""
        return self._objective

    @property
    def constraints(self) -> tuple[ProgramConstraint, ...]:
        """Return constraints in insertion order."""
        return tuple(self._constraints)

    def binary_variable(self, name: str) -> BinaryVariable:
        """Declare and return one binary variable."""
        variable = BinaryVariable(name)
        if variable.name in self._variables:
            raise ValueError(f"Variable {variable.name!r} is already declared.")
        self._variables[variable.name] = variable
        return variable

    binary_var = binary_variable

    def binary_array(
        self,
        name: str,
        shape: int | tuple[int, ...],
    ) -> NDArray[np.object_]:
        """Declare an array of binary variables in row-major order."""
        variables = binary_array(name, shape)
        flat_variables = tuple(
            cast(BinaryVariable, variable) for variable in variables.flat
        )
        conflicts = [
            variable.name
            for variable in flat_variables
            if variable.name in self._variables
        ]
        if conflicts:
            raise ValueError(f"Variable {conflicts[0]!r} is already declared.")
        for variable in flat_variables:
            self._variables[variable.name] = variable
        return variables

    def _register_expression(self, expression: QuadraticExpression) -> None:
        for variable in expression.variables:
            self._variables.setdefault(variable.name, variable)

    def _set_objective(
        self,
        expression: ExpressionLike,
        sense: OptimizationSense,
    ) -> None:
        objective_expression = as_expression(expression)
        self._register_expression(objective_expression)
        self._objective = QuadraticObjective(objective_expression, sense)

    def minimize(self, expression: ExpressionLike) -> None:
        """Set the objective to minimize the supplied expression."""
        self._set_objective(expression, OptimizationSense.MINIMIZE)

    def maximize(self, expression: ExpressionLike) -> None:
        """Set the objective to maximize the supplied expression."""
        self._set_objective(expression, OptimizationSense.MAXIMIZE)

    def _claim_constraint_name(self, name: object) -> str:
        normalized = _validate_program_name(name)
        assert normalized is not None
        if normalized in self._constraint_names:
            raise ValueError(f"Constraint {normalized!r} already exists.")
        return normalized

    def add_linear_equality(
        self,
        residual: ExpressionLike,
        *,
        name: str,
        penalty: Real,
    ) -> LinearEqualityConstraint:
        """Add ``residual == 0``, compiling it as ``penalty * residual**2``."""
        expression = as_expression(residual)
        if expression.degree > 1:
            raise ValueError("A linear equality residual must have degree at most one.")
        normalized_name = self._claim_constraint_name(name)
        normalized_penalty = _validate_penalty(penalty)
        constraint = LinearEqualityConstraint(
            name=normalized_name,
            residual=expression,
            penalty=normalized_penalty,
        )
        self._register_expression(expression)
        self._constraints.append(constraint)
        self._constraint_names.add(normalized_name)
        return constraint

    add_linear_equality_constraint = add_linear_equality

    def add_nonnegative_quadratic_zero(
        self,
        expression: ExpressionLike,
        *,
        name: str,
        penalty: Real,
    ) -> NonnegativeQuadraticZeroConstraint:
        """Add a declared-nonnegative ``expression == 0`` direct penalty.

        Calling this purpose-specific method is the explicit assertion that the
        supplied expression is nonnegative over all binary assignments.
        """
        quadratic_expression = as_expression(expression)
        normalized_name = self._claim_constraint_name(name)
        normalized_penalty = _validate_penalty(penalty)
        constraint = NonnegativeQuadraticZeroConstraint(
            name=normalized_name,
            expression=quadratic_expression,
            penalty=normalized_penalty,
        )
        self._register_expression(quadratic_expression)
        self._constraints.append(constraint)
        self._constraint_names.add(normalized_name)
        return constraint

    add_quadratic_zero_constraint = add_nonnegative_quadratic_zero
