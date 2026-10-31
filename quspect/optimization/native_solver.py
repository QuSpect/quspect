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
        beta_start: Positive initial inverse temperature.
        beta_end: Positive final inverse temperature, not below
            ``beta_start``.

    Raises:
        ValueError: If an option is outside its supported range.
        NativeIsingSolverError: If the C++ extension is unavailable or
            stale.

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
        beta_start: float = 0.1,
        beta_end: float = 5.0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an unsigned 64-bit integer.")
        if seed < 0 or seed > np.iinfo(np.uint64).max:
            raise ValueError("seed must be an unsigned 64-bit integer.")
        self._seed = seed
        self._num_reads = _positive_integer(num_reads, name="num_reads")
        self._sweeps = _positive_integer(sweeps, name="sweeps")
        self._beta_start = _positive_finite(beta_start, name="beta_start")
        self._beta_end = _positive_finite(beta_end, name="beta_end")
        if self._beta_end < self._beta_start:
            raise ValueError("beta_end must be greater than or equal to beta_start.")
        _require_core()

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
        core_result = core._solve_dense_ising(
            program.matrix,
            self._seed,
            self._num_reads,
            self._sweeps,
            self._beta_start,
            self._beta_end,
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
