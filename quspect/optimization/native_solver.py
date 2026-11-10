"""Native dense-Ising solver adapter.

This module connects :class:`HardwareIsingProgram` directly to the C++
simulated-annealing implementation.  It intentionally consumes the same dense
hardware matrix that a future remote Ising-machine backend will submit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from quspect._native_loader import BUILD_HINT, load_core
from quspect.optimization.hardware import HardwareIsingProgram

SpinArray: TypeAlias = NDArray[np.int8]
FloatArray: TypeAlias = NDArray[np.float64]

_core: Any | None = load_core()


class NativeIsingSolverError(ImportError):
    """Raised when the native extension does not provide the Ising solver."""


def _require_core() -> Any:
    if _core is None:
        raise NativeIsingSolverError(
            "QuSpect native extension (quspect._core_scalar) is not installed. "
            f"{BUILD_HINT}"
        )
    if not hasattr(_core, "_solve_dense_ising"):
        raise NativeIsingSolverError(
            "The QuSpect native extension is present but does not include the "
            "dense Ising solver. Rebuild the C++ extension."
        )
    return _core


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


#: Default inverse temperatures for a matrix whose largest coefficient has
#: modulus 1 (the fixed defaults of releases up to 0.6.2).
_UNIT_BETA_RANGE = (0.1, 5.0)


def default_beta_range(matrix: object) -> tuple[float, float]:
    """Return the default ``(beta_start, beta_end)`` for a hardware matrix.

    The defaults ``0.1`` and ``5.0`` are measured in units of the largest
    coefficient: with ``scale = max |M_ij|`` the range is
    ``(0.1 / scale, 5.0 / scale)``.  ``matrix`` and ``c * matrix`` are
    therefore annealed alike, and a matrix whose largest coefficient has
    modulus 1 gets exactly ``(0.1, 5.0)``.

    Args:
        matrix: Square hardware matrix (finite, symmetric, zero diagonal).

    Returns:
        ``(beta_start, beta_end)`` with ``0 < beta_start < beta_end``.  A
        matrix without any non-zero coefficient gives ``(0.1, 5.0)``.

    Raises:
        ValueError: If the matrix is not a finite square matrix, or its
            coefficients are so small or so large that the range is not
            finite (pass ``beta_start`` and ``beta_end`` explicitly then).

    Example:
        >>> import numpy as np
        >>> default_beta_range(np.array([[0.0, 0.5], [0.5, 0.0]]))
        (0.2, 10.0)
        >>> default_beta_range(np.array([[0.0, -40.0], [-40.0, 0.0]]))
        (0.0025, 0.125)
    """
    magnitudes = np.abs(np.asarray(matrix, dtype=np.float64))
    if magnitudes.ndim != 2 or magnitudes.shape[0] != magnitudes.shape[1]:
        raise ValueError("matrix must be square.")
    if not bool(np.all(np.isfinite(magnitudes))):
        raise ValueError("matrix must contain only finite values.")
    scale = float(magnitudes.max(initial=0.0))
    if scale == 0.0:
        return _UNIT_BETA_RANGE
    beta_start = _UNIT_BETA_RANGE[0] / scale
    beta_end = _UNIT_BETA_RANGE[1] / scale
    if not (math.isfinite(beta_end) and beta_start > 0.0):
        raise ValueError(
            "The matrix coefficients are outside the range a default temperature "
            "schedule can be derived for; pass beta_start and beta_end explicitly."
        )
    return beta_start, beta_end


def _readonly_spins(value: object, *, columns: int, name: str) -> SpinArray:
    array = np.asarray(value, dtype=np.int8)
    if array.ndim != 2 or array.shape[1] != columns:
        raise RuntimeError(
            f"Native {name} has shape {array.shape}; expected (n, {columns})."
        )
    if not bool(np.all(np.logical_or(array == -1, array == 1))):
        raise RuntimeError(f"Native {name} contains values outside {{-1, +1}}.")
    result = np.array(array, dtype=np.int8, copy=True)
    result.setflags(write=False)
    return result


def _readonly_values(value: object, *, rows: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows,) or not bool(np.all(np.isfinite(array))):
        raise RuntimeError(
            f"Native {name} must be a finite vector with shape {(rows,)}."
        )
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeIsingResult:
    """Store native hardware samples and their decoded logical values.

    Results retain one row per requested read, including duplicates.  Rows are
    ordered by ascending hardware energy; ties preserve their original read
    order.

    Args:
        hardware_samples: Raw spins in hardware matrix order.
        logical_samples: Gauge-normalized, auxiliary-free logical spins.
        energies: Values of ``-s.T @ matrix @ s``.
        objective_values: Values of the logical Ising objective supplied to
            hardware lowering, recovered from the hardware energies.  If that
            Ising problem came from a constrained ``QuadraticProgram``, these
            values include its compiled penalties; use
            ``CompilationResult.evaluate_objective`` for the unpenalized
            business objective.
        logical_variables: Logical variable names in result-column order.

    The first row is always the best retained read, so ``best_energy`` is the
    minimum value in ``energies``.
    """

    hardware_samples: SpinArray
    logical_samples: SpinArray
    energies: FloatArray
    objective_values: FloatArray
    logical_variables: tuple[str, ...]

    @property
    def best_hardware_sample(self) -> SpinArray:
        """Return the lowest-energy raw hardware sample."""
        return cast(SpinArray, self.hardware_samples[0])

    @property
    def best_sample(self) -> SpinArray:
        """Return the decoded sample corresponding to the lowest energy."""
        return cast(SpinArray, self.logical_samples[0])

    @property
    def best_energy(self) -> float:
        """Return the lowest hardware energy found."""
        return float(self.energies[0])

    @property
    def best_objective_value(self) -> float:
        """Return the input logical objective value of the best hardware sample."""
        return float(self.objective_values[0])

    @property
    def num_reads(self) -> int:
        """Return the number of retained independent reads."""
        return int(self.hardware_samples.shape[0])


class NativeIsingSolver:
    """Minimize a hardware Ising matrix with native simulated annealing.

    Every read starts from an independently seeded random spin state.  One
    sweep attempts each spin once in a shuffled order while beta follows a
    linear schedule.  The native implementation parallelizes reads with
    OpenMP when available.

    Args:
        seed: Unsigned 64-bit base seed.  A fixed value reproduces all reads.
        num_reads: Number of independent annealing runs to retain.
        sweeps: Number of full spin-update sweeps per read.
        beta_start: Positive initial inverse temperature, used exactly as
            given.  ``None`` (the default) derives it from the matrix of each
            solved program, see :func:`default_beta_range`.
        beta_end: Positive final inverse temperature, not below
            ``beta_start``, used exactly as given.  ``None`` (the default)
            derives it from the matrix of each solved program.

    Raises:
        ValueError: If an option is outside its supported range.
        NativeIsingSolverError: If the C++ extension is unavailable or
            stale.

    Note:
        Releases up to 0.6.2 used the fixed defaults ``beta_start=0.1`` and
        ``beta_end=5.0`` whatever the size of the coefficients; with
        coefficients far from 1 the anneal froze at once or never cooled.
        The defaults are now divided by the largest coefficient modulus of
        the matrix.  Default results for a given seed therefore differ from
        those releases unless that modulus is 1; pass ``beta_start=0.1,
        beta_end=5.0`` to reproduce them.  If only one end is given, the
        other is derived and, where needed, moved to the given value so the
        schedule never runs from cold to hot.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization.hardware import HardwareIsingProgram
        >>> program = HardwareIsingProgram(
        ...     matrix=np.array([[0.0, 0.5], [0.5, 0.0]]),
        ...     variable_order=("a", "b"),
        ...     logical_variable_count=2,
        ... )
        >>> solver = NativeIsingSolver(seed=7, num_reads=4, sweeps=20)
        >>> result = solver.solve(program)
        >>> result.best_energy
        -1.0
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        num_reads: int = 1,
        sweeps: int = 1000,
        beta_start: float | None = None,
        beta_end: float | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an unsigned 64-bit integer.")
        if seed < 0 or seed > np.iinfo(np.uint64).max:
            raise ValueError("seed must be an unsigned 64-bit integer.")
        self._seed = seed
        self._num_reads = _positive_integer(num_reads, name="num_reads")
        self._sweeps = _positive_integer(sweeps, name="sweeps")
        self._beta_start = (
            None
            if beta_start is None
            else _positive_finite(beta_start, name="beta_start")
        )
        self._beta_end = (
            None if beta_end is None else _positive_finite(beta_end, name="beta_end")
        )
        if (
            self._beta_start is not None
            and self._beta_end is not None
            and self._beta_end < self._beta_start
        ):
            raise ValueError("beta_end must be greater than or equal to beta_start.")
        _require_core()

    def beta_range(self, program: HardwareIsingProgram) -> tuple[float, float]:
        """Return the ``(beta_start, beta_end)`` used to solve ``program``.

        Explicit constructor values are returned unchanged.  An end left at
        ``None`` comes from :func:`default_beta_range` of the program's
        matrix; when only one end was given, the derived end is moved to it
        if the schedule would otherwise run from cold to hot.

        Args:
            program: The hardware program to be solved.

        Returns:
            The inverse temperatures of the first and the last sweep.
        """
        if self._beta_start is not None and self._beta_end is not None:
            return self._beta_start, self._beta_end
        derived_start, derived_end = default_beta_range(program.matrix)
        if self._beta_start is not None:
            return self._beta_start, max(derived_end, self._beta_start)
        if self._beta_end is not None:
            return min(derived_start, self._beta_end), self._beta_end
        return derived_start, derived_end

    def solve(self, program: HardwareIsingProgram) -> NativeIsingResult:
        """Solve one hardware-level dense Ising program.

        Args:
            program: Validated hardware program using
                ``E(s) = -s.T @ matrix @ s``.

        Returns:
            Structured raw and decoded samples sorted by ascending hardware
            energy.

        Raises:
            TypeError: If ``program`` is not a :class:`HardwareIsingProgram`.
            RuntimeError: If the native extension returns malformed data.
        """
        if not isinstance(program, HardwareIsingProgram):
            raise TypeError("program must be a HardwareIsingProgram.")
        core = _require_core()
        beta_start, beta_end = self.beta_range(program)
        core_result = core._solve_dense_ising(
            program.matrix,
            self._seed,
            self._num_reads,
            self._sweeps,
            beta_start,
            beta_end,
        )
        hardware_samples = _readonly_spins(
            core_result.samples,
            columns=program.bit_num,
            name="samples",
        )
        energies = _readonly_values(
            core_result.energies,
            rows=hardware_samples.shape[0],
            name="energies",
        )
        logical_samples = np.array(
            program.decode_samples(hardware_samples),
            dtype=np.int8,
            copy=True,
        )
        logical_samples.setflags(write=False)
        objective_values = np.asarray(
            program.objective_scale * energies + program.objective_offset,
            dtype=np.float64,
        )
        objective_values.setflags(write=False)
        return NativeIsingResult(
            hardware_samples=hardware_samples,
            logical_samples=logical_samples,
            energies=energies,
            objective_values=objective_values,
            logical_variables=program.logical_variables,
        )
