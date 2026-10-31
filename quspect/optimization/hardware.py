"""Hardware-level Ising matrix representation and provider-neutral lowering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from quspect.optimization.model import (
    FloatArray,
    IsingProblem,
    OptimizationSense,
)

SpinArray: TypeAlias = NDArray[np.int8]


def _readonly_hardware_matrix(value: object, *, size: int) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (size, size):
        raise ValueError(f"matrix must have shape {(size, size)}, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("matrix must be symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("matrix must have a zero diagonal.")
    # Averaging preserves ``s.T @ matrix @ s`` while making the exact
    # symmetry assumed by local-field solvers explicit.
    # Scale before adding: ``0.5 * (a + a.T)`` overflows to inf for finite
    # entries above ~8.99e307, whereas the halves always sum finitely.
    output = np.asarray(0.5 * matrix + 0.5 * matrix.T, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("matrix must contain only finite values.")
    np.fill_diagonal(output, 0.0)
    output.setflags(write=False)
    return output


def _coerce_hardware_spins(
    value: object,
    *,
    size: int,
) -> tuple[SpinArray, bool]:
    raw = np.asarray(value)
    was_vector = raw.ndim == 1
    if was_vector:
        raw = raw[np.newaxis, :]
    if raw.ndim != 2 or raw.shape[1] != size:
        raise ValueError(
            f"Hardware spins must have shape ({size},) or (n, {size}), got {raw.shape}."
        )
    if not bool(np.all(np.logical_or(raw == -1, raw == 1))):
        raise ValueError("Hardware spins must contain only -1 and +1.")
    return cast(SpinArray, np.asarray(raw, dtype=np.int8)), was_vector


@dataclass(frozen=True)
class HardwareIsingProgram:
    """Represent the exact dense matrix accepted by an Ising-machine backend.

    The matrix follows the dense full-matrix hardware convention

    ``hardware_energy(t) = -t.T @ matrix @ t``.

    Logical fields are represented by couplings to an optional gauge spin.
    ``objective_scale`` and ``objective_offset`` recover the original logical
    objective from a raw hardware energy.

    Args:
        matrix: Dense symmetric coupling matrix with zero diagonal.
        variable_order: Hardware spin names in matrix order. Logical spins
            always occupy the leading ``logical_variable_count`` positions.
        logical_variable_count: Number of user-visible logical spins.
        gauge_index: Optional auxiliary gauge-spin index.
        objective_scale: Multiplier converting hardware energy to the original
            objective value.
        objective_offset: Constant added after energy conversion.
        format: Versioned target-format identifier.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import HardwareIsingProgram
        >>> program = HardwareIsingProgram(
        ...     matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]),
        ...     variable_order=("s", "__gauge__"),
        ...     logical_variable_count=1,
        ...     gauge_index=1,
        ... )
        >>> program.decode_sample([-1, -1]).tolist()
        [1]
    """

    matrix: FloatArray
    variable_order: tuple[str, ...]
    logical_variable_count: int
    gauge_index: int | None = None
    objective_scale: float = 1.0
    objective_offset: float = 0.0
    format: str = "dense-ising-matrix-v1"

    def __post_init__(self) -> None:
        variable_order = tuple(self.variable_order)
        if any(
            not isinstance(name, str) or not name.strip() for name in variable_order
        ):
            raise ValueError("Hardware variable names must be non-empty strings.")
        if len(set(variable_order)) != len(variable_order):
            raise ValueError("Hardware variable names must be unique.")
        if (
            isinstance(self.logical_variable_count, bool)
            or not isinstance(self.logical_variable_count, int)
            or self.logical_variable_count < 0
            or self.logical_variable_count > len(variable_order)
        ):
            raise ValueError("logical_variable_count is outside the matrix range.")
        if self.gauge_index is not None and (
            isinstance(self.gauge_index, bool)
            or not isinstance(self.gauge_index, int)
            or self.gauge_index < self.logical_variable_count
            or self.gauge_index >= len(variable_order)
        ):
            raise ValueError("gauge_index must identify an auxiliary spin.")
        objective_scale = float(self.objective_scale)
        objective_offset = float(self.objective_offset)
        if not math.isfinite(objective_scale) or objective_scale == 0.0:
            raise ValueError("objective_scale must be finite and non-zero.")
        if not math.isfinite(objective_offset):
            raise ValueError("objective_offset must be finite.")
        if not isinstance(self.format, str) or not self.format.strip():
            raise ValueError("format must be a non-empty string.")
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(
            self,
            "matrix",
            _readonly_hardware_matrix(self.matrix, size=len(variable_order)),
        )
        object.__setattr__(self, "objective_scale", objective_scale)
        object.__setattr__(self, "objective_offset", objective_offset)
        object.__setattr__(self, "format", self.format.strip())

    @property
    def bit_num(self) -> int:
        """Return the matrix dimension submitted as the hardware bit count."""
        return len(self.variable_order)

    @property
    def logical_variables(self) -> tuple[str, ...]:
        """Return user-visible spin names in decoded order."""
        return self.variable_order[: self.logical_variable_count]

    @property
    def auxiliary_variables(self) -> tuple[str, ...]:
        """Return compiler-generated hardware spin names."""
        return self.variable_order[self.logical_variable_count :]

    def decode_sample(self, hardware_spins: object) -> SpinArray:
        """Normalize the gauge and remove auxiliary spins from one sample.

        Args:
            hardware_spins: One vector in hardware matrix order.

        Returns:
            Read-only logical spin vector with the gauge fixed to ``+1``.

        Raises:
            ValueError: If the sample shape or domain is invalid.
        """
        samples, was_vector = _coerce_hardware_spins(
            hardware_spins,
            size=self.bit_num,
        )
        if not was_vector:
            raise ValueError("decode_sample expects one hardware spin vector.")
        decoded = np.asarray(self._decode_array(samples)[0], dtype=np.int8)
        decoded.setflags(write=False)
        return decoded

    def decode_samples(self, hardware_spins: object) -> SpinArray:
        """Decode a batch of hardware spin samples.

        Args:
            hardware_spins: Array with shape ``(n, bit_num)``.

        Returns:
            Read-only array with shape ``(n, logical_variable_count)``.
        """
        samples, _ = _coerce_hardware_spins(
            hardware_spins,
            size=self.bit_num,
        )
        decoded = self._decode_array(samples)
        decoded.setflags(write=False)
        return decoded

    def _decode_array(self, samples: SpinArray) -> SpinArray:
        logical = np.array(
            samples[:, : self.logical_variable_count],
            dtype=np.int8,
            copy=True,
        )
        if self.gauge_index is not None:
            gauge = samples[:, self.gauge_index].reshape((-1, 1))
            logical *= gauge
        return logical

    def hardware_energy(self, hardware_spins: object) -> float:
        """Evaluate one raw hardware spin vector.

        Args:
            hardware_spins: One vector containing ``-1`` and ``+1``.

        Returns:
            Energy under ``-t.T @ matrix @ t``.
        """
        samples, was_vector = _coerce_hardware_spins(
            hardware_spins,
            size=self.bit_num,
        )
        if not was_vector:
            raise ValueError("hardware_energy expects one hardware spin vector.")
        vector = samples[0].astype(np.float64, copy=False)
        return float(-np.dot(vector, self.matrix @ vector))

    def objective_value(self, hardware_spins: object) -> float:
        """Recover the original logical objective for one hardware sample."""
        return (
            self.objective_scale * self.hardware_energy(hardware_spins)
            + self.objective_offset
        )


def compile_hardware_ising(
    problem: IsingProblem,
    *,
    matrix_scale: float = 1.0,
) -> HardwareIsingProgram:
    """Lower a logical Ising problem to a dense hardware matrix.

    Linear fields are coupled to one auxiliary gauge spin.  A maximizing
    logical objective is negated because the hardware always minimizes its
    energy.  ``matrix_scale`` can later be selected by a precision adapter
    without losing the exact objective-energy relationship.

    Args:
        problem: Backend-neutral logical Ising problem.
        matrix_scale: Positive coefficient multiplier applied to the matrix.

    Returns:
        Hardware program containing the exact matrix and local decode metadata.

    Raises:
        ValueError: If ``matrix_scale`` is not finite and positive.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import IsingProblem, compile_hardware_ising
        >>> logical = IsingProblem(
        ...     ("s",), np.array([2.0]), np.zeros((1, 1)), offset=3.0
        ... )
        >>> program = compile_hardware_ising(logical)
        >>> program.matrix.tolist()
        [[0.0, -1.0], [-1.0, 0.0]]
        >>> program.objective_value([1, 1])
        5.0
    """
    scale = float(matrix_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("matrix_scale must be finite and positive.")
    direction = 1.0 if problem.sense is OptimizationSense.MINIMIZE else -1.0
    needs_gauge = bool(np.any(problem.linear != 0.0))
    hardware_size = problem.num_spins + int(needs_gauge)
    matrix = np.zeros((hardware_size, hardware_size), dtype=np.float64)
    matrix[: problem.num_spins, : problem.num_spins] = (
        -0.5 * direction * scale * problem.quadratic
    )

    gauge_index: int | None = None
    variable_order = problem.variables
    if needs_gauge:
        gauge_index = problem.num_spins
        gauge_name = _unique_gauge_name(problem.variables)
        variable_order = (*problem.variables, gauge_name)
        gauge_couplings = -0.5 * direction * scale * problem.linear
        matrix[: problem.num_spins, gauge_index] = gauge_couplings
        matrix[gauge_index, : problem.num_spins] = gauge_couplings

    return HardwareIsingProgram(
        matrix=matrix,
        variable_order=variable_order,
        logical_variable_count=problem.num_spins,
        gauge_index=gauge_index,
        objective_scale=direction / scale,
        objective_offset=problem.offset,
    )


def _unique_gauge_name(variables: tuple[str, ...]) -> str:
    base = "__gauge__"
    if base not in variables:
        return base
    suffix = 1
    while f"{base}{suffix}" in variables:
        suffix += 1
    return f"{base}{suffix}"
