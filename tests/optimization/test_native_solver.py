"""Cross-layer tests for HardwareIsingProgram and the native C++ solver."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pytest

from quspect._native_loader import BUILD_HINT, load_core
from quspect.optimization.hardware import HardwareIsingProgram
from quspect.optimization.native_solver import NativeIsingSolver, default_beta_range

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


# ---------------------------------------------------------------------------
# Default temperature schedule
# ---------------------------------------------------------------------------


def _program(matrix: np.ndarray) -> HardwareIsingProgram:
    size = matrix.shape[0]
    return HardwareIsingProgram(
        matrix=matrix,
        variable_order=tuple(f"s{i}" for i in range(size)),
        logical_variable_count=size,
    )


def _balanced_cut_matrix(scale: float = 1.0) -> np.ndarray:
    """A 14-variable dense cut with weights in the tens (minimise ``Σ w s s``)."""
    weights = np.random.default_rng(3).integers(10, 60, size=(14, 14)).astype(float)
    weights = np.triu(weights, 1)
    return np.asarray(-(weights + weights.T) * scale / 2.0)


def test_default_beta_range_is_in_units_of_the_largest_coefficient() -> None:
    matrix = np.array(
        [
            [0.0, 3.0, -1.0, 0.0],
            [3.0, 0.0, 0.5, 0.0],
            [-1.0, 0.5, 0.0, -4.0],
            [0.0, 0.0, -4.0, 0.0],
        ]
    )
    assert default_beta_range(matrix) == (0.1 / 4.0, 5.0 / 4.0)
    # A matrix normalised to a largest coefficient of 1 keeps the old defaults.
    assert default_beta_range(matrix / 4.0) == (0.1, 5.0)


def test_default_beta_range_is_inverse_to_the_coefficient_scale() -> None:
    matrix = _balanced_cut_matrix()
    start, end = default_beta_range(matrix)
    scaled_start, scaled_end = default_beta_range(matrix * 1024.0)
    assert scaled_start == start / 1024.0
    assert scaled_end == end / 1024.0
    assert 0.0 < start < end


@pytest.mark.parametrize("size", [0, 1, 3])
def test_default_beta_range_of_a_matrix_without_couplings(size: int) -> None:
    """Nothing to anneal: the constants used up to 0.6.2 remain."""
    assert default_beta_range(np.zeros((size, size))) == (0.1, 5.0)
    result = NativeIsingSolver(seed=1, num_reads=2, sweeps=3).solve(
        _program(np.zeros((size, size)))
    )
    assert result.hardware_samples.shape == (2, size)


def test_default_beta_range_rejects_what_it_cannot_scale() -> None:
    with pytest.raises(ValueError, match="square"):
        default_beta_range(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        default_beta_range(np.array([[0.0, np.inf], [np.inf, 0.0]]))
    with pytest.raises(ValueError, match="explicitly"):
        default_beta_range(np.array([[0.0, 1e-320], [1e-320, 0.0]]))


def test_unit_scale_matrix_gives_the_results_of_the_old_defaults() -> None:
    """Nothing changes for a matrix whose largest coefficient has modulus 1."""
    matrix = _balanced_cut_matrix()
    matrix = matrix / np.abs(matrix).max()
    default = NativeIsingSolver(seed=2, num_reads=8, sweeps=80).solve(_program(matrix))
    old = NativeIsingSolver(
        seed=2, num_reads=8, sweeps=80, beta_start=0.1, beta_end=5.0
    ).solve(_program(matrix))
    np.testing.assert_array_equal(default.hardware_samples, old.hardware_samples)
    np.testing.assert_array_equal(default.energies, old.energies)


def test_default_schedule_solves_large_coefficients() -> None:
    """Known issue: beta 0.1..5.0 froze at once when coefficients were in the tens."""
    matrix = _balanced_cut_matrix()
    optimum = float(np.asarray(_core._solve_dense_ising_exact(matrix, 14).energies)[0])
    result = NativeIsingSolver(seed=0, num_reads=64, sweeps=1000).solve(
        _program(matrix)
    )
    assert result.best_energy == pytest.approx(optimum)
    optimal_reads = int(np.sum(np.isclose(result.energies, optimum)))
    assert optimal_reads >= 56


def test_default_schedule_does_not_depend_on_the_coefficient_scale() -> None:
    """Scaling the matrix by a power of two scales the energies and nothing else."""
    matrix = _balanced_cut_matrix()
    solver = NativeIsingSolver(seed=5, num_reads=16, sweeps=200)
    plain = solver.solve(_program(matrix))
    for scale in (1.0 / 64.0, 1024.0):
        scaled = solver.solve(_program(matrix * scale))
        np.testing.assert_array_equal(scaled.hardware_samples, plain.hardware_samples)
        np.testing.assert_array_equal(scaled.energies, plain.energies * scale)


def test_default_schedule_is_the_documented_range() -> None:
    matrix = _balanced_cut_matrix(0.37)
    beta_start, beta_end = default_beta_range(matrix)
    default = NativeIsingSolver(seed=9, num_reads=8, sweeps=50).solve(_program(matrix))
    explicit = NativeIsingSolver(
        seed=9, num_reads=8, sweeps=50, beta_start=beta_start, beta_end=beta_end
    ).solve(_program(matrix))
    np.testing.assert_array_equal(default.hardware_samples, explicit.hardware_samples)
    np.testing.assert_array_equal(default.energies, explicit.energies)


def test_explicit_betas_are_passed_through_unchanged() -> None:
    """Explicit temperatures are never rescaled, whatever the coefficients."""
    matrix = _balanced_cut_matrix()
    result = NativeIsingSolver(
        seed=3, num_reads=8, sweeps=60, beta_start=0.1, beta_end=5.0
    ).solve(_program(matrix))
    native = _core._solve_dense_ising(matrix, 3, 8, 60, 0.1, 5.0)
    np.testing.assert_array_equal(
        result.hardware_samples, np.asarray(native.samples, dtype=np.int8)
    )
    np.testing.assert_array_equal(result.energies, np.asarray(native.energies))


def test_one_explicit_beta_keeps_its_value_and_the_other_is_derived() -> None:
    matrix = _balanced_cut_matrix()
    derived_start, derived_end = default_beta_range(matrix)
    program = _program(matrix)

    def run(**betas: float) -> Any:
        return NativeIsingSolver(seed=4, num_reads=4, sweeps=40, **betas).solve(program)

    def native(beta_start: float, beta_end: float) -> np.ndarray:
        return np.asarray(
            _core._solve_dense_ising(matrix, 4, 4, 40, beta_start, beta_end).energies
        )

    # Only the cold end given: the hot end comes from the matrix.
    np.testing.assert_array_equal(
        run(beta_end=1.0).energies, native(derived_start, 1.0)
    )
    # Only the hot end given: the cold end comes from the matrix.
    np.testing.assert_array_equal(
        run(beta_start=1e-5).energies, native(1e-5, derived_end)
    )
    # A given end on the far side of the derived one: a constant schedule at
    # the given value, never an error and never a changed explicit value.
    assert derived_start > 1e-9
    np.testing.assert_array_equal(run(beta_end=1e-9).energies, native(1e-9, 1e-9))
    assert derived_end < 50.0
    np.testing.assert_array_equal(run(beta_start=50.0).energies, native(50.0, 50.0))
