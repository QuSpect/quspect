"""Compilation of symbolic quadratic programs to backend-neutral BQMs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .expression import BinaryVariable, QuadraticExpression
from .model import BinaryQuadraticModel, OptimizationSense
from .program import (
    LinearEqualityConstraint,
    NonnegativeQuadraticZeroConstraint,
    ProgramConstraint,
    QuadraticObjective,
    QuadraticProgram,
)


def _coerce_bit(value: object, *, label: str) -> int:
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number in (0.0, 1.0):
            return int(number)
    raise ValueError(f"{label} must be in the binary domain {{0, 1}}.")


@dataclass(frozen=True)
class CompilationResult:
    """A compiled model plus immutable metadata for decoding and validation."""

    binary_quadratic_model: BinaryQuadraticModel
    objective: QuadraticObjective
    constraints: tuple[ProgramConstraint, ...]

    @property
    def model(self) -> BinaryQuadraticModel:
        """Return the compiled binary quadratic model."""
        return self.binary_quadratic_model

    @property
    def bqm(self) -> BinaryQuadraticModel:
        """Return the compiled binary quadratic model."""
        return self.binary_quadratic_model

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the stable compiled variable-name order."""
        return self.binary_quadratic_model.variables

    def decode(self, bits: Sequence[object] | NDArray[np.object_]) -> dict[str, int]:
        """Decode a binary vector to a name-to-bit mapping."""
        if isinstance(bits, str | bytes):
            raise TypeError("bits must be a one-dimensional sequence.")
        array = np.asarray(bits, dtype=object)
        if array.ndim != 1:
            raise ValueError(f"bits must be one-dimensional, got shape {array.shape}.")
        if len(array) != len(self.variables):
            raise ValueError(f"Expected {len(self.variables)} bits, got {len(array)}.")
        return {
            name: _coerce_bit(value, label=f"Bit for {name!r}")
            for name, value in zip(self.variables, array.tolist(), strict=True)
        }

    def _assignment(
        self,
        value: Mapping[str | BinaryVariable, object] | Sequence[object],
    ) -> dict[str, int]:
        if not isinstance(value, Mapping):
            return self.decode(value)

        normalized: dict[str, int] = {}
        for raw_name, raw_bit in value.items():
            if isinstance(raw_name, BinaryVariable):
                name = raw_name.name
            elif isinstance(raw_name, str):
                name = raw_name
            else:
                raise TypeError("Assignment keys must be variable names or variables.")
            if name in normalized:
                raise ValueError(f"Assignment contains duplicate variable {name!r}.")
            normalized[name] = _coerce_bit(raw_bit, label=f"Bit for {name!r}")

        expected = set(self.variables)
        actual = set(normalized)
        missing = [name for name in self.variables if name not in actual]
        extra = sorted(actual.difference(expected))
        if missing:
            raise ValueError(
                "Assignment is missing variables: " + ", ".join(missing) + "."
            )
        if extra:
            raise ValueError(
                "Assignment contains unknown variables: " + ", ".join(extra) + "."
            )
        return {name: normalized[name] for name in self.variables}

    def evaluate_objective(
        self,
        value: Mapping[str | BinaryVariable, object] | Sequence[object],
    ) -> float:
        """Evaluate the original objective without constraint penalties."""
        return self.objective.expression.evaluate(self._assignment(value))

    def is_feasible(
        self,
        value: Mapping[str | BinaryVariable, object] | Sequence[object],
        *,
        tolerance: float = 1e-9,
    ) -> bool:
        """Return whether every original zero constraint is satisfied."""
        if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
            raise TypeError("tolerance must be a real number.")
        normalized_tolerance = float(tolerance)
        if not math.isfinite(normalized_tolerance) or normalized_tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        assignment = self._assignment(value)
        return all(
            abs(constraint.expression.evaluate(assignment)) <= normalized_tolerance
            for constraint in self.constraints
        )


def _compile_expression(
    expression: QuadraticExpression,
    *,
    variable_indices: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    size = len(variable_indices)
    linear = np.zeros(size, dtype=np.float64)
    quadratic = np.zeros((size, size), dtype=np.float64)
    for variable, coefficient in expression.linear.items():
        linear[variable_indices[variable.name]] += coefficient
    for (left, right), coefficient in expression.quadratic.items():
        left_index = variable_indices[left.name]
        right_index = variable_indices[right.name]
        quadratic[left_index, right_index] += coefficient
        quadratic[right_index, left_index] += coefficient
    return linear, quadratic, expression.constant


def compile(program: QuadraticProgram) -> CompilationResult:
    """Compile a quadratic program, including its feasibility penalties."""
    if not isinstance(program, QuadraticProgram):
        raise TypeError("program must be a QuadraticProgram.")
    objective = program.objective
    if objective is None:
        raise ValueError("QuadraticProgram has no objective.")

    variables = program.variables
    variable_names = tuple(variable.name for variable in variables)
    variable_indices = {
        variable.name: index for index, variable in enumerate(variables)
    }
    penalized = objective.expression
    penalty_sign = 1.0 if objective.sense is OptimizationSense.MINIMIZE else -1.0
    constraints = program.constraints
    for constraint in constraints:
        if isinstance(constraint, LinearEqualityConstraint):
            penalty_expression = constraint.residual * constraint.residual
        elif isinstance(constraint, NonnegativeQuadraticZeroConstraint):
            penalty_expression = constraint.expression
        else:
            raise TypeError(f"Unsupported constraint type {type(constraint).__name__}.")
        penalized += penalty_sign * constraint.penalty * penalty_expression

    linear, quadratic, offset = _compile_expression(
        penalized,
        variable_indices=variable_indices,
    )
    model = BinaryQuadraticModel(
        variables=variable_names,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        sense=objective.sense,
    )
    return CompilationResult(
        binary_quadratic_model=model,
        objective=objective,
        constraints=constraints,
    )


CompiledQuadraticProgram = CompilationResult
compile_program = compile
compile_quadratic_program = compile
