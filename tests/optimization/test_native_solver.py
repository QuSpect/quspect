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


def _flat_reads(seed: int, num_reads: int) -> list[tuple[int, ...]]:
    """Reads on a flat landscape, in read order.

    Every proposed flip is accepted when all couplings are zero, so a read is
    a pure function of its random stream.  All energies tie, and ties keep
    their read order.
    """
    program = HardwareIsingProgram(
        matrix=np.zeros((40, 40)),
        variable_order=tuple(f"s{index}" for index in range(40)),
        logical_variable_count=40,
    )
    result = NativeIsingSolver(seed=seed, num_reads=num_reads, sweeps=3).solve(program)
    return [tuple(int(spin) for spin in row) for row in result.hardware_samples]


@pytest.mark.parametrize("base_seed", [0, 7, 20260731, 2**64 - 100])
def test_adjacent_seeds_share_no_more_reads_than_distant_seeds(base_seed: int) -> None:
    """Read k of seed s must not be read 0 of seed s + k.

    Two unrelated 40-spin reads agree with probability 2**-40, so neither
    adjacent nor distant seeds may share a read.
    """
    num_reads = 64
    base = set(_flat_reads(base_seed, num_reads))
    assert len(base) == num_reads
    distant = [2**32 + 5, 12345678901234567]
    shared_with_distant = [
        len(base & set(_flat_reads((base_seed + offset) % 2**64, num_reads)))
        for offset in distant
    ]
    shared_with_adjacent = [
        len(base & set(_flat_reads(base_seed + offset, num_reads)))
        for offset in (1, 2, 3, num_reads - 1, num_reads, num_reads + 1)
    ]
    assert shared_with_distant == [0] * len(distant)
    assert max(shared_with_adjacent) <= max(shared_with_distant)


def test_single_reads_of_consecutive_seeds_differ_from_one_multi_read_run() -> None:
    """``num_reads=1`` at seeds s, s+1, ... must not replay one run of seed s."""
    multi = _flat_reads(0, 6)
    single = [_flat_reads(seed, 1)[0] for seed in range(6)]
    assert single[0] == multi[0]  # a read depends only on (seed, read index)
    assert not set(single[1:]) & set(multi)


def test_reads_of_adjacent_seeds_are_uncorrelated() -> None:
    """Same-index reads of neighbouring seeds agree on about half the spins."""
    first = np.array(_flat_reads(1000, 64))
    second = np.array(_flat_reads(1001, 64))
    agreement = float(np.mean(first == second))
    # 2560 fair comparisons: one standard deviation is 0.01.
    assert 0.44 < agreement < 0.56


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
