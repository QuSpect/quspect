"""Backend-neutral data models for binary quadratic optimization.

The classes in this module form the stable boundary between problem modeling
and solver-specific compilation.  They intentionally retain variable order,
linear terms, quadratic couplings, offsets, and optimization direction instead
of collapsing those semantics into a transport matrix too early.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int8]


class VariableType(str, enum.Enum):
    """Domain used by a binary quadratic model.

    ``BINARY`` variables take values in ``{0, 1}``, while ``SPIN`` variables
    take values in ``{-1, +1}``.
    """

    BINARY = "binary"
    SPIN = "spin"


class OptimizationSense(str, enum.Enum):
    """Direction in which an objective is optimized."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


def _validate_variables(variables: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(variables)
    for name in normalized:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Variable names must be non-empty strings.")
        if name != name.strip():
            raise ValueError(f"Variable name {name!r} contains outer whitespace.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Variable names must be unique.")
    return normalized


def _readonly_vector(value: object, *, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    output = np.array(array, dtype=np.float64, copy=True)
    output.setflags(write=False)
    return output


def _readonly_quadratic(value: object, *, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    expected_shape = (size, size)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    if not np.allclose(np.diag(array), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must have a zero diagonal.")
    # Downstream conversions use row sums under the symmetric-matrix
    # invariant.  Canonicalize values accepted within tolerance so those
    # formulas do not retain a direction-dependent floating-point residue.
    # Scale before adding: ``0.5 * (a + a.T)`` overflows to inf for finite
    # entries above ~8.99e307, whereas the halves always sum finitely.
    output = np.asarray(0.5 * array + 0.5 * array.T, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must contain only finite values.")
    np.fill_diagonal(output, 0.0)
    output.setflags(write=False)
    return output


def _coerce_samples(
    value: object,
    *,
    size: int,
    variable_type: VariableType,
) -> tuple[FloatArray, bool]:
    samples = np.asarray(value, dtype=np.float64)
    was_vector = samples.ndim == 1
    if was_vector:
        samples = samples[np.newaxis, :]
    if samples.ndim != 2 or samples.shape[1] != size:
        raise ValueError(
            f"Samples must have shape ({size},) or (n, {size}), got {samples.shape}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples must contain only finite values.")
    allowed = (
        np.logical_or(samples == 0.0, samples == 1.0)
        if variable_type is VariableType.BINARY
        else np.logical_or(samples == -1.0, samples == 1.0)
    )
    if not bool(np.all(allowed)):
        domain = "{0, 1}" if variable_type is VariableType.BINARY else "{-1, +1}"
        raise ValueError(f"Samples must use the {domain} domain.")
    return samples, was_vector


@dataclass(frozen=True)
class BinaryQuadraticModel:
    """Represent a binary or spin quadratic objective.

    The stored objective is

    ``offset + linear @ v + 0.5 * v.T @ quadratic @ v``.

    ``quadratic`` is symmetric with a zero diagonal, so the factor ``0.5``
    counts each pair exactly once.  The variable order is explicit and remains
    stable through compilation and result decoding.

    Args:
        variables: Unique variable names in vector order.
        linear: One linear coefficient per variable.
        quadratic: Symmetric pairwise-coefficient matrix with zero diagonal.
        offset: Constant objective term.
        variable_type: Binary ``{0, 1}`` or spin ``{-1, +1}`` domain.
        sense: Whether the objective is minimized or maximized.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import BinaryQuadraticModel, VariableType
        >>> model = BinaryQuadraticModel(
        ...     variables=("x", "y"),
        ...     linear=np.array([1.0, -2.0]),
        ...     quadratic=np.array([[0.0, 3.0], [3.0, 0.0]]),
        ...     variable_type=VariableType.BINARY,
        ... )
        >>> model.evaluate([1, 0])
        1.0
    """

    variables: tuple[str, ...]
    linear: FloatArray
    quadratic: FloatArray
    offset: float = 0.0
    variable_type: VariableType = VariableType.BINARY
    sense: OptimizationSense = OptimizationSense.MINIMIZE

    def __post_init__(self) -> None:
        variables = _validate_variables(tuple(self.variables))
        try:
            variable_type = VariableType(self.variable_type)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported variable type {self.variable_type!r}."
            ) from exc
        try:
            sense = OptimizationSense(self.sense)
        except ValueError as exc:
            raise ValueError(f"Unsupported optimization sense {self.sense!r}.") from exc
        offset = float(self.offset)
        if not math.isfinite(offset):
            raise ValueError("offset must be finite.")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(
            self,
            "linear",
            _readonly_vector(self.linear, size=len(variables), name="linear"),
        )
        object.__setattr__(
            self,
            "quadratic",
            _readonly_quadratic(
                self.quadratic,
                size=len(variables),
                name="quadratic",
            ),
        )
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "variable_type", variable_type)
        object.__setattr__(self, "sense", sense)

    @property
    def num_variables(self) -> int:
        """Return the number of model variables."""
        return len(self.variables)

    def evaluate(self, sample: object) -> float:
        """Evaluate one domain-valid assignment.

        Args:
            sample: Vector in the model's declared variable domain.

        Returns:
            Objective value before applying the optimization direction.

        Raises:
            ValueError: If the vector has the wrong shape or domain.
        """
        samples, _ = _coerce_samples(
            sample,
            size=self.num_variables,
            variable_type=self.variable_type,
        )
        vector = samples[0]
        return float(
            self.offset
            + np.dot(self.linear, vector)
            + 0.5 * np.dot(vector, self.quadratic @ vector)
        )

    def evaluate_many(self, samples: object) -> FloatArray:
        """Evaluate a batch of assignments in stable input order.

        Args:
            samples: Array with shape ``(n, num_variables)``.

        Returns:
            Read-only one-dimensional array of objective values.

        Raises:
            ValueError: If any assignment has the wrong shape or domain.
        """
        sample_array, _ = _coerce_samples(
            samples,
            size=self.num_variables,
            variable_type=self.variable_type,
        )
        values = (
            self.offset
            + sample_array @ self.linear
            + 0.5
            * np.einsum(
                "bi,ij,bj->b",
                sample_array,
                self.quadratic,
                sample_array,
            )
        )
        output = np.asarray(values, dtype=np.float64)
        output.setflags(write=False)
        return output


@dataclass(frozen=True)
class IsingProblem:
    """Represent a backend-neutral Ising objective.

    QuSpect fixes the logical convention

    ``offset + linear @ spins + 0.5 * spins.T @ quadratic @ spins``.

    Provider-specific signs, coefficient scaling, and auxiliary gauge spins
    belong in a later hardware compilation stage.

    Args:
        variables: Unique spin names in vector order.
        linear: Local-field coefficient for every spin.
        quadratic: Symmetric coupling matrix with zero diagonal.
        offset: Constant objective term.
        sense: Whether the logical objective is minimized or maximized.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import IsingProblem
        >>> problem = IsingProblem(
        ...     variables=("s0", "s1"),
        ...     linear=np.zeros(2),
        ...     quadratic=np.array([[0.0, 1.0], [1.0, 0.0]]),
        ... )
        >>> problem.evaluate([1, -1])
        -1.0
    """

    variables: tuple[str, ...]
    linear: FloatArray
    quadratic: FloatArray
    offset: float = 0.0
    sense: OptimizationSense = OptimizationSense.MINIMIZE

    def __post_init__(self) -> None:
        model = BinaryQuadraticModel(
            variables=tuple(self.variables),
            linear=self.linear,
            quadratic=self.quadratic,
            offset=self.offset,
            variable_type=VariableType.SPIN,
            sense=self.sense,
        )
        object.__setattr__(self, "variables", model.variables)
        object.__setattr__(self, "linear", model.linear)
        object.__setattr__(self, "quadratic", model.quadratic)
        object.__setattr__(self, "offset", model.offset)
        object.__setattr__(self, "sense", model.sense)

    @property
    def num_spins(self) -> int:
        """Return the number of logical spins."""
        return len(self.variables)

    def as_bqm(self) -> BinaryQuadraticModel:
        """Return the equivalent spin-domain quadratic model."""
        return BinaryQuadraticModel(
            variables=self.variables,
            linear=self.linear,
            quadratic=self.quadratic,
            offset=self.offset,
            variable_type=VariableType.SPIN,
            sense=self.sense,
        )

    def evaluate(self, spins: object) -> float:
        """Evaluate one spin assignment.

        Args:
            spins: Vector containing only ``-1`` and ``+1``.

        Returns:
            Logical Ising objective value.

        Raises:
            ValueError: If the assignment has the wrong shape or domain.
        """
        return self.as_bqm().evaluate(spins)

    def evaluate_many(self, spins: object) -> FloatArray:
        """Evaluate multiple spin assignments.

        Args:
            spins: Array with shape ``(n, num_spins)``.

        Returns:
            Read-only objective values in input order.
        """
        return self.as_bqm().evaluate_many(spins)
