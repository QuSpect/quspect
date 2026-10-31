"""Cross-layer tests for HardwareIsingProgram and the native C++ solver."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pytest

from quspect._native_loader import BUILD_HINT, load_core
from quspect.optimization.hardware import HardwareIsingProgram
from quspect.optimization.native_solver import NativeIsingSolver

_core: Any | None = load_core()

pytestmark = pytest.mark.skipif(
    _core is None,
    reason=f"QuSpect native extension not found. {BUILD_HINT}",
)


def test_native_module_contains_ising_bindings() -> None:
    """A stale extension must fail visibly instead of being skipped."""
    assert hasattr(_core, "_solve_dense_ising")
    assert hasattr(_core, "_solve_dense_ising_exact")


def test_solver_uses_hardware_matrix_and_decodes_gauge() -> None:
    """The adapter preserves raw spins and applies program decode metadata."""
    program = HardwareIsingProgram(
        matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]),
        variable_order=("logical", "__gauge__"),
        logical_variable_count=1,
        gauge_index=1,
        objective_scale=1.0,
        objective_offset=3.0,
    )
    result = NativeIsingSolver(
        seed=17,
        num_reads=8,
        sweeps=30,
        beta_start=0.05,
        beta_end=6.0,
    ).solve(program)

    assert result.hardware_samples.shape == (8, 2)
    assert result.logical_samples.shape == (8, 1)
    assert result.logical_variables == ("logical",)
    assert result.num_reads == 8
    assert result.best_energy == pytest.approx(-2.0)
    assert result.best_sample.tolist() == [-1]
    assert result.best_objective_value == pytest.approx(1.0)
    assert np.all(np.diff(result.energies) >= 0.0)
    for row, energy, objective in zip(
        result.hardware_samples,
        result.energies,
        result.objective_values,
        strict=True,
    ):
        assert energy == pytest.approx(program.hardware_energy(row))
        assert objective == pytest.approx(program.objective_value(row))


def test_solver_is_reproducible_and_results_are_read_only() -> None:
    """Fixed options reproduce samples regardless of native parallelism."""
    matrix = np.array(
        [
            [0.0, 0.7, -1.1, 0.2],
            [0.7, 0.0, 0.4, -0.8],
            [-1.1, 0.4, 0.0, 0.6],
            [0.2, -0.8, 0.6, 0.0],
        ]
    )
    program = HardwareIsingProgram(
        matrix=matrix,
        variable_order=("s0", "s1", "s2", "s3"),
        logical_variable_count=4,
    )
    solver = NativeIsingSolver(seed=1234, num_reads=12, sweeps=40)
    first = solver.solve(program)
    second = solver.solve(program)

    np.testing.assert_array_equal(first.hardware_samples, second.hardware_samples)
    np.testing.assert_array_equal(first.logical_samples, second.logical_samples)
    np.testing.assert_array_equal(first.energies, second.energies)
    assert not first.hardware_samples.flags.writeable
    assert not first.logical_samples.flags.writeable
    assert not first.energies.flags.writeable
    assert not first.objective_values.flags.writeable


def test_exact_binding_matches_python_enumeration() -> None:
    """The Gray-code C++ oracle matches independent Python enumeration."""
    matrix = np.array(
        [
            [0.0, 0.3, -1.2, 0.7, -0.4],
            [0.3, 0.0, 0.8, -0.2, 1.1],
            [-1.2, 0.8, 0.0, 0.9, -0.6],
            [0.7, -0.2, 0.9, 0.0, 0.5],
            [-0.4, 1.1, -0.6, 0.5, 0.0],
        ]
    )
    native_result = _core._solve_dense_ising_exact(matrix, 10)
    native_energy = float(np.asarray(native_result.energies)[0])
    expected = min(
        -np.asarray(spins, dtype=np.float64) @ matrix @ spins
        for spins in product((-1, 1), repeat=5)
    )
    assert native_energy == pytest.approx(expected)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.zeros((2, 3)), "square"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "diagonal"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
    ],
)
def test_pybind_rejects_invalid_matrices(matrix: np.ndarray, message: str) -> None:
    """Native validation remains active across the pybind boundary."""
    with pytest.raises(ValueError, match=message):
        _core._solve_dense_ising(matrix)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"num_reads": 0},
        {"sweeps": 0},
        {"beta_start": 0.0},
        {"beta_end": float("inf")},
        {"beta_start": 2.0, "beta_end": 1.0},
    ],
)
def test_python_adapter_rejects_invalid_options(kwargs: dict[str, object]) -> None:
    """The public adapter reports invalid options before native execution."""
    with pytest.raises(ValueError):
        NativeIsingSolver(**kwargs)  # type: ignore[arg-type]


def test_solver_rejects_non_program_input() -> None:
    """The adapter only accepts the frozen hardware-level data model."""
    solver = NativeIsingSolver()
    with pytest.raises(TypeError, match="HardwareIsingProgram"):
        solver.solve(np.zeros((2, 2)))  # type: ignore[arg-type]
