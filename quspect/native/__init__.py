"""QuSpect Native Executor — Python bridge to C++ simulation backend.

This module provides a high-level Python API that wraps the C++ extension
module selected by :mod:`quspect._native_loader` (``quspect._core_avx2`` on
CPUs with AVX2 + FMA, otherwise ``quspect._core_scalar``).  Users can decode QSAL
JSON and run circuits through the optimized C++ state-vector simulator.

Example::

    import json
    from quspect.native import NativeExecutor

    qsal_json = json.dumps({...})
    executor = NativeExecutor(precision="double", seed=42)
    result = executor.run(qsal_json, shots=1024)
    print(result.counts)

Use :func:`load_core` to detect whether the native core is available without
raising::

    from quspect.native import load_core

    if load_core() is None:
        ...  # native core not built; fall back to pure-Python code paths
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from quspect._native_loader import BUILD_HINT, load_core

__all__ = [
    "ExecutionResult",
    "NativeExecutor",
    "NativeImportError",
    "capabilities",
    "decode_qsal",
    "get_num_threads",
    "load_core",
    "set_num_threads",
    "validate",
]

_core: Any = load_core()


class NativeImportError(ImportError):
    """Raised when the C++ native module is not available."""


def _ensure_core() -> None:
    if _core is None:
        raise NativeImportError(
            f"QuSpect native extension (quspect._core_scalar) not found. {BUILD_HINT}"
        )


# =============================================================================
# NativeExecutor
# =============================================================================


@dataclass
class ExecutionResult:
    """Python-side result mirroring C++ ExecutionResult."""

    statevector_f64: list[complex] = field(default_factory=list)
    statevector_f32: list[complex] = field(default_factory=list)
    packed_samples: list[int] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    expectations_real: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def _from_core(cls, core_result: Any) -> ExecutionResult:
        """Convert a C++ ExecutionResult to a Python ExecutionResult."""
        return cls(
            statevector_f64=list(core_result.statevector_f64),
            statevector_f32=list(core_result.statevector_f32),
            packed_samples=list(core_result.packed_samples),
            counts=dict(core_result.counts),
            expectations_real=dict(core_result.expectations_real),
            metadata=dict(core_result.metadata),
        )


@dataclass(frozen=True)
class _NativeAdjointEvaluation:
    """Internal value-and-Jacobian result from an adjoint objective plan.

    ``jacobian`` is laid out as ``[observable][parameter]``.  The associated
    names are included so callers never have to depend on an implicit map or
    set iteration order.
    """

    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    values: tuple[float, ...]
    jacobian: tuple[tuple[float, ...], ...]

    @classmethod
    def _from_core(cls, core_result: Any) -> _NativeAdjointEvaluation:
        return cls(
            parameter_names=tuple(core_result.parameter_names),
            observable_names=tuple(core_result.observable_names),
            values=tuple(core_result.values),
            jacobian=tuple(tuple(row) for row in core_result.jacobian),
        )


class _NativeAdjointPlan:
    """Internal reusable objective plan prepared by the C++ backend."""

    def __init__(self, qsal_json: str, *, precision: str = "double") -> None:
        _ensure_core()
        self._plan = _core._AdjointPlan(qsal_json, precision)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Global parameters in the stable native Jacobian-column order."""
        return tuple(self._plan.parameter_names)

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Hamiltonian labels in result-row order."""
        return tuple(self._plan.observable_names)

    def evaluate(self, values: Mapping[str, float]) -> _NativeAdjointEvaluation:
        """Evaluate all objective values and their Jacobian at one point."""
        core_result = self._plan.evaluate(
            {str(name): float(value) for name, value in values.items()}
        )
        return _NativeAdjointEvaluation._from_core(core_result)


class NativeExecutor:
    """High-level executor wrapping the C++ native simulation backend.

    Args:
        precision: ``"double"`` (default) or ``"single"``.
        seed: Random seed for reproducible sampling.

    Example::

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal_json, shots=1024)
        print(result.counts)
    """

    def __init__(self, *, precision: str = "double", seed: int = 0) -> None:
        _ensure_core()
        self._precision = precision
        self._seed = seed
        self._exec = _core.SimulationExecutor(precision, seed)

    # ------------------------------------------------------------------
    # Primary API: run from QSAL JSON
    # ------------------------------------------------------------------

    def run(
        self,
        qsal_json: str,
        *,
        shots: int = 0,
        return_statevector: bool = False,
        return_expectations: bool = False,
        return_samples: bool = False,
    ) -> ExecutionResult:
        """Decode QSAL JSON, validate, and execute.

        This is the one-stop entry point that combines decode, validate,
        and execute steps.

        Args:
            qsal_json: A QSAL JSON string.
            shots: Number of measurement shots (0 = no sampling).
            return_statevector: If True, return the final state vector.
            return_expectations: If True, compute and return observable expectations.
            return_samples: If True, return raw packed samples.

        Returns:
            An :class:`ExecutionResult` with the requested outputs.

        Raises:
            RuntimeError: If decoding, validation, or execution fails.
        """
        core_result = _core.decode_qsal_and_run(
            qsal_json,
            self._precision,
            self._seed,
            shots,
            return_statevector,
            return_expectations,
            return_samples,
        )
        return ExecutionResult._from_core(core_result)

    # ------------------------------------------------------------------
    # Low-level API: decode or run pre-built programs
    # ------------------------------------------------------------------

    def decode(self, qsal_json: str) -> Any:
        """Decode a QSAL JSON string into a C++ QSALProgram object.

        Args:
            qsal_json: A QSAL JSON string.

        Returns:
            A C++ ``QSALProgram`` object (pybind11 wrapper).

        Raises:
            RuntimeError: If decoding fails.
        """
        return _core.decode_qsal(qsal_json)

    def run_program(
        self,
        program: Any,
        *,
        shots: int = 0,
        return_statevector: bool = False,
        return_expectations: bool = False,
        return_samples: bool = False,
    ) -> ExecutionResult:
        """Execute a pre-built C++ QSALProgram.

        Args:
            program: A C++ ``QSALProgram`` object from :meth:`decode`.
            shots: Number of measurement shots.
            return_statevector: If True, return the final state vector.
            return_expectations: If True, compute observable expectations.
            return_samples: If True, return raw packed samples.

        Returns:
            An :class:`ExecutionResult`.
        """
        req = _core.ExecutionRequest()
        req.shots = shots
        req.return_statevector = return_statevector
        req.return_expectations = return_expectations
        req.return_samples = return_samples

        core_result = self._exec.run(program, req)
        return ExecutionResult._from_core(core_result)

    def _prepare_adjoint(self, qsal_json: str) -> _NativeAdjointPlan:
        """Prepare a unitary QSAL objective for repeated adjoint evaluation.

        This is an internal hook used by the public simulation objective API.
        The plan owns the decoded C++ program, so subsequent optimizer steps
        pass only a small parameter mapping across pybind.
        """
        return _NativeAdjointPlan(qsal_json, precision=self._precision)


# =============================================================================
# Module-level helpers
# =============================================================================


def decode_qsal(qsal_json: str) -> Any:
    """Decode a QSAL JSON string.

    Convenience function — equivalent to ``NativeExecutor().decode(...)``.

    Args:
        qsal_json: A QSAL JSON string.

    Returns:
        A C++ QSALProgram object.
    """
    _ensure_core()
    return _core.decode_qsal(qsal_json)


def validate(program: Any) -> list[str]:
    """Validate a QSALProgram.

    Args:
        program: A C++ QSALProgram object.

    Returns:
        List of validation error messages (empty if valid).
    """
    _ensure_core()
    return [str(error) for error in _core.Validator.validate(program)]


def capabilities() -> dict[str, Any]:
    """Return runtime capabilities of the native backend."""
    _ensure_core()
    return dict(_core.get_runtime_capabilities())


# =============================================================================
# Threading (OpenMP)
# =============================================================================


def get_num_threads() -> int:
    """Return the number of threads used by the native parallel kernels.

    The C++ state-vector kernels (gate application, expectation values,
    measurement probabilities and collapse), the adjoint-gradient loops and
    the Ising annealer are parallelized with OpenMP.  Unless
    :func:`set_num_threads` has been called, the value follows the standard
    OpenMP defaults, i.e. ``OMP_NUM_THREADS`` when it is set, otherwise the
    number of available cores.  States with fewer than ``2**14`` amplitudes
    always run single-threaded regardless of this setting.

    Returns:
        The current thread count (always ``1`` when the extension was built
        without OpenMP; see ``capabilities()["openmp"]``).

    Raises:
        NativeImportError: If the C++ native extension is not available.

    Example:
        >>> from quspect.native import get_num_threads
        >>> get_num_threads() >= 1
        True
    """
    _ensure_core()
    return int(_core.get_num_threads())


def set_num_threads(n: int) -> None:
    """Set the number of threads used by the native parallel kernels.

    The setting is process-wide for the loaded native extension and applies
    to every subsequent native call, from any Python thread.  It takes
    precedence over ``OMP_NUM_THREADS``.  Results do not depend on the thread
    count: reductions use a fixed blocking, and seeded sampling produces the
    same counts for any ``n``.  When the extension was built without OpenMP
    the value is validated and otherwise ignored.

    Args:
        n: Number of threads; must be at least 1.

    Raises:
        ValueError: If ``n`` is smaller than 1.
        TypeError: If ``n`` is not an integer.
        NativeImportError: If the C++ native extension is not available.

    Example:
        >>> from quspect.native import get_num_threads, set_num_threads
        >>> previous = get_num_threads()
        >>> set_num_threads(1)
        >>> get_num_threads()
        1
        >>> set_num_threads(previous)
    """
    _ensure_core()
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    _core.set_num_threads(n)
