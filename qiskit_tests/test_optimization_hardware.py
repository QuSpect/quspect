"""Tests for hardware lowering, the CSV codec and the native Ising solver.

Hardware convention (from ``hardware.py``): ``E_hw(t) = -t.T @ M @ t`` with a
symmetric zero-diagonal ``M``; logical fields couple to an auxiliary gauge
spin appended after the logical spins; decoding multiplies logical spins by
the gauge spin; ``objective = objective_scale * E_hw + objective_offset``.

``NativeIsingSolver`` is a heuristic (simulated annealing).  Its optimality
is judged statistically: on small random instances with a fixed seed the
best of several reads must equal the brute-force optimum.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from quspect.optimization import (
    HardwareIsingProgram,
    IsingProblem,
    NativeIsingResult,
    NativeIsingSolver,
    NativeIsingSolverError,
    OptimizationSense,
    QuadraticProgram,
    binary_to_ising,
    compile_hardware_ising,
    compile_program,
    encode_ising_matrix_csv,
)
from quspect.optimization import native_solver as native_module


def _all_spins(n: int) -> np.ndarray:
    """Return all ``2**n`` spin vectors; row ``k`` has ``s_i = 2*bit_i(k) - 1``."""
    k = np.arange(2**n)
    return (2 * ((k[:, None] >> np.arange(n)[None, :]) & 1) - 1).astype(np.int8)


def _random_problem(
    rng: np.random.Generator,
    n: int,
    *,
    sense: OptimizationSense = OptimizationSense.MINIMIZE,
    with_fields: bool = True,
) -> IsingProblem:
    """Return a random logical Ising problem on ``n`` spins."""
    upper = np.triu(rng.normal(size=(n, n)), k=1)
    linear = rng.normal(size=n) if with_fields else np.zeros(n)
    return IsingProblem(
        tuple(f"s{i}" for i in range(n)),
        linear,
        upper + upper.T,
        offset=float(rng.normal()),
        sense=sense,
    )


def _parse_csv(text: str) -> np.ndarray:
    """Independent minimal parser for the headerless dense-matrix CSV."""
    assert text.endswith("\n")
    rows = text[:-1].split("\n")
    return np.array([[float(cell) for cell in row.split(",")] for row in rows])


# ---------------------------------------------------------------------------
# compile_hardware_ising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sense", list(OptimizationSense))
@pytest.mark.parametrize("scale", [1.0, 0.25, 3.0])
@pytest.mark.parametrize("with_fields", [True, False])
def test_hardware_objective_matches_logical_for_every_sample(
    sense: OptimizationSense,
    scale: float,
    with_fields: bool,
    rng: np.random.Generator,
) -> None:
    """For every raw hardware vector, recovered objective = logical energy."""
    n = 4
    problem = _random_problem(rng, n, sense=sense, with_fields=with_fields)
    program = compile_hardware_ising(problem, matrix_scale=scale)
    direction = 1.0 if sense is OptimizationSense.MINIMIZE else -1.0
    assert program.objective_scale == pytest.approx(direction / scale)
    assert program.objective_offset == problem.offset
    assert program.logical_variables == problem.variables
    assert program.format == "dense-ising-matrix-v1"
    if with_fields:
        assert program.bit_num == n + 1
        assert program.gauge_index == n
        assert program.auxiliary_variables == ("__gauge__",)
    else:
        assert program.bit_num == n
        assert program.gauge_index is None
        assert program.auxiliary_variables == ()
    np.testing.assert_allclose(
        program.matrix[:n, :n], -0.5 * direction * scale * problem.quadratic
    )
    hw_energies = []
    for t in _all_spins(program.bit_num):
        logical = program.decode_sample(t)
        assert program.objective_value(t) == pytest.approx(
            problem.evaluate(logical), abs=1e-10
        )
        hw_energies.append(program.hardware_energy(t))
    # Hardware ground state decodes to the logical optimum.
    t_best = _all_spins(program.bit_num)[int(np.argmin(hw_energies))]
    logical_values = problem.evaluate_many(_all_spins(n))
    target = (
        logical_values.min()
        if sense is OptimizationSense.MINIMIZE
        else logical_values.max()
    )
    assert problem.evaluate(program.decode_sample(t_best)) == pytest.approx(target)


def test_hardware_docstring_example() -> None:
    """A positive coupling becomes ``-1`` and the docstring value is 5.0."""
    problem = IsingProblem(
        ("a", "b"), np.zeros(2), np.array([[0.0, 2.0], [2.0, 0.0]]), offset=3.0
    )
    program = compile_hardware_ising(problem)
    assert program.matrix.tolist() == [[0.0, -1.0], [-1.0, 0.0]]
    assert program.objective_value([1, 1]) == 5.0


def test_gauge_name_avoids_collisions() -> None:
    """The gauge name gets the first free numeric suffix."""
    names = ("__gauge__", "__gauge__1", "x")
    problem = IsingProblem(names, np.ones(3), np.zeros((3, 3)))
    assert compile_hardware_ising(problem).variable_order[-1] == "__gauge__2"
    problem = IsingProblem(("__gauge__",), np.ones(1), np.zeros((1, 1)))
    assert compile_hardware_ising(problem).variable_order[-1] == "__gauge__1"


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_compile_hardware_rejects_bad_scale(scale: float) -> None:
    """``matrix_scale`` must be finite and positive."""
    problem = IsingProblem(("a",), np.ones(1), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="matrix_scale"):
        compile_hardware_ising(problem, matrix_scale=scale)


@pytest.mark.qiskit_ref
def test_hardware_energy_matches_qiskit_diagonal(rng: np.random.Generator) -> None:
    """``-t M t`` equals the diagonal of ``-2 sum_{i<j} M_ij Z_i Z_j``.

    Qubit i holds hardware spin i with ``t_i = 1 - 2 b_i`` (``Z|0>=+|0>``).
    """
    program = compile_hardware_ising(_random_problem(rng, 4))
    n = program.bit_num
    terms = [
        ("ZZ", [i, j], -2.0 * program.matrix[i, j])
        for i, j in itertools.combinations(range(n), 2)
    ]
    diag = np.diag(SparsePauliOp.from_sparse_list(terms, num_qubits=n).to_matrix())
    for k in range(2**n):
        t = np.array([1 - 2 * ((k >> i) & 1) for i in range(n)])
        assert diag[k].real == pytest.approx(program.hardware_energy(t), abs=1e-10)


# ---------------------------------------------------------------------------
# HardwareIsingProgram
# ---------------------------------------------------------------------------


def _hw(**overrides: object) -> HardwareIsingProgram:
    """Build a 3-spin hardware program with the given overrides."""
    kwargs: dict[str, object] = {
        "matrix": np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.5], [2.0, 0.5, 0.0]]),
        "variable_order": ("a", "b", "g"),
        "logical_variable_count": 2,
        "gauge_index": 2,
    }
    kwargs.update(overrides)
    return HardwareIsingProgram(**kwargs)  # type: ignore[arg-type]


def test_hardware_program_decode_and_energy() -> None:
    """Decoding fixes the gauge to +1; energy is ``-t M t``."""
    program = _hw(format="  custom-v2 ", objective_scale=2, objective_offset=1)
    assert program.format == "custom-v2"
    assert program.bit_num == 3
    assert program.logical_variables == ("a", "b")
    assert program.auxiliary_variables == ("g",)
    np.testing.assert_array_equal(program.decode_sample([1, -1, -1]), [-1, 1])
    np.testing.assert_array_equal(program.decode_sample([1, -1, 1]), [1, -1])
    decoded = program.decode_samples([[1, 1, -1], [1, 1, 1]])
    np.testing.assert_array_equal(decoded, [[-1, -1], [1, 1]])
    assert not decoded.flags.writeable
    t = np.array([1, -1, 1])
    expected = -float(t @ program.matrix @ t)
    assert program.hardware_energy(t) == expected
    assert program.objective_value(t) == 2 * expected + 1
    no_gauge = _hw(gauge_index=None)
    np.testing.assert_array_equal(no_gauge.decode_sample([1, -1, -1]), [1, -1])
    assert not no_gauge.decode_sample([1, -1, -1]).flags.writeable


def test_hardware_program_canonicalizes_matrix() -> None:
    """Near-symmetric input is averaged; stored matrix is read-only."""
    m = np.array([[0.0, 1.0 + 4e-13], [1.0, 0.0]])
    program = HardwareIsingProgram(m, ("a", "b"), logical_variable_count=2)
    assert program.matrix[0, 1] == program.matrix[1, 0]
    assert not program.matrix.flags.writeable


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"variable_order": ("a", "", "g")}, "non-empty"),
        ({"variable_order": ("a", 1, "g")}, "non-empty"),
        ({"variable_order": ("a", "a", "g")}, "unique"),
        ({"logical_variable_count": -1}, "logical_variable_count"),
        ({"logical_variable_count": 4}, "logical_variable_count"),
        ({"logical_variable_count": True}, "logical_variable_count"),
        ({"logical_variable_count": 2.0}, "logical_variable_count"),
        ({"gauge_index": 1}, "gauge_index"),
        ({"gauge_index": 3}, "gauge_index"),
        ({"gauge_index": True}, "gauge_index"),
        ({"gauge_index": 2.0}, "gauge_index"),
        ({"objective_scale": 0.0}, "objective_scale"),
        ({"objective_scale": math.inf}, "objective_scale"),
        ({"objective_offset": math.nan}, "objective_offset"),
        ({"format": "  "}, "format"),
        ({"format": 1}, "format"),
        ({"matrix": np.zeros((2, 2))}, "shape"),
        ({"matrix": np.full((3, 3), math.nan)}, "finite"),
        ({"matrix": np.triu(np.ones((3, 3)), k=1)}, "symmetric"),
        ({"matrix": np.eye(3)}, "diagonal"),
    ],
)
def test_hardware_program_validation(overrides: dict[str, object], match: str) -> None:
    """Every constructor invariant raises ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        _hw(**overrides)


@pytest.mark.parametrize(
    "method", ["decode_sample", "decode_samples", "hardware_energy", "objective_value"]
)
@pytest.mark.parametrize("bad", [[1, 1], [1, 0, 1], [[[1, 1, 1]]], 1])
def test_hardware_spin_validation(method: str, bad: object) -> None:
    """Wrong-shape or out-of-domain hardware spins raise ``ValueError``."""
    with pytest.raises(ValueError):
        getattr(_hw(), method)(bad)


@pytest.mark.parametrize("method", ["decode_sample", "hardware_energy"])
def test_single_sample_methods_reject_batches(method: str) -> None:
    """Single-vector APIs reject a 2-D batch."""
    with pytest.raises(ValueError, match="one hardware spin vector"):
        getattr(_hw(), method)([[1, 1, 1]])


# ---------------------------------------------------------------------------
# CSV codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sense", list(OptimizationSense))
def test_csv_round_trip_is_exact(
    sense: OptimizationSense, rng: np.random.Generator
) -> None:
    """Parsing the CSV recovers the matrix bit-for-bit, row by row."""
    problem = _random_problem(rng, 5, sense=sense)
    program = compile_hardware_ising(problem, matrix_scale=1 / 3)
    text = encode_ising_matrix_csv(program)
    assert text.count("\n") == program.bit_num
    assert "\r" not in text
    parsed = _parse_csv(text)
    assert parsed.shape == program.matrix.shape
    np.testing.assert_array_equal(parsed, program.matrix)
    # The stdlib csv reader agrees with the hand-written parser.
    via_csv = np.array(list(csv.reader(io.StringIO(text))), dtype=np.float64)
    np.testing.assert_array_equal(via_csv, program.matrix)
    # Rebuilding a program from the parsed matrix preserves energies.
    rebuilt = HardwareIsingProgram(
        parsed,
        program.variable_order,
        logical_variable_count=program.logical_variable_count,
        gauge_index=program.gauge_index,
        objective_scale=program.objective_scale,
        objective_offset=program.objective_offset,
    )
    for t in _all_spins(program.bit_num):
        assert rebuilt.objective_value(t) == program.objective_value(t)


def test_csv_extreme_values_and_docstring() -> None:
    """Tiny/huge magnitudes survive; the docstring output is reproduced."""
    upper = np.array(
        [[0.0, 1e-300, -1.2345678901234567e300], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]
    )
    m = upper + upper.T
    program = HardwareIsingProgram(m, ("a", "b", "c"), logical_variable_count=3)
    np.testing.assert_array_equal(_parse_csv(encode_ising_matrix_csv(program)), m)
    doc = HardwareIsingProgram(
        np.array([[0.0, 1.0], [1.0, 0.0]]), ("s0", "s1"), logical_variable_count=2
    )
    assert encode_ising_matrix_csv(doc) == "0.0,1.0\n1.0,0.0\n"


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_hardware_program_large_finite_matrix_stays_finite() -> None:
    """An accepted finite symmetric matrix must be stored unchanged (finite)."""
    m = np.array([[0.0, 1e308], [1e308, 0.0]])
    program = HardwareIsingProgram(m, ("a", "b"), logical_variable_count=2)
    np.testing.assert_array_equal(program.matrix, m)


def test_csv_rejects_non_program() -> None:
    """Non-``HardwareIsingProgram`` input raises ``TypeError``."""
    with pytest.raises(TypeError, match="HardwareIsingProgram"):
        encode_ising_matrix_csv(np.zeros((2, 2)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# NativeIsingSolver
# ---------------------------------------------------------------------------


def _check_result_consistency(
    result: NativeIsingResult, program: HardwareIsingProgram, problem: IsingProblem
) -> None:
    """Assert internal consistency of a native result."""
    assert result.logical_variables == program.logical_variables
    assert result.hardware_samples.shape == (result.num_reads, program.bit_num)
    assert np.all(np.diff(result.energies) >= 0)
    for name in ("hardware_samples", "logical_samples", "energies", "objective_values"):
        assert not getattr(result, name).flags.writeable
    for i in range(result.num_reads):
        t = result.hardware_samples[i]
        assert result.energies[i] == pytest.approx(program.hardware_energy(t))
        np.testing.assert_array_equal(
            result.logical_samples[i], program.decode_sample(t)
        )
        assert result.objective_values[i] == pytest.approx(
            problem.evaluate(result.logical_samples[i])
        )
    assert result.best_energy == result.energies.min()
    np.testing.assert_array_equal(result.best_sample, result.logical_samples[0])
    np.testing.assert_array_equal(
        result.best_hardware_sample, result.hardware_samples[0]
    )
    assert result.best_objective_value == result.objective_values[0]


@pytest.mark.statistical
@pytest.mark.parametrize("sense", list(OptimizationSense))
def test_native_solver_finds_brute_force_optimum(sense: OptimizationSense) -> None:
    """Best of 16 annealing reads equals the exhaustive optimum on 20 instances."""
    rng = np.random.default_rng(2024)
    solver = NativeIsingSolver(seed=7, num_reads=16, sweeps=500)
    hits = 0
    total_reads = 0
    for trial in range(20):
        n = 3 + trial % 6  # 3..8 logical spins
        problem = _random_problem(rng, n, sense=sense)
        program = compile_hardware_ising(problem)
        result = solver.solve(program)
        _check_result_consistency(result, program, problem)
        values = problem.evaluate_many(_all_spins(n))
        target = values.min() if sense is OptimizationSense.MINIMIZE else values.max()
        assert result.best_objective_value == pytest.approx(target, abs=1e-9)
        hits += int(np.sum(np.isclose(result.objective_values, target, atol=1e-9)))
        total_reads += result.num_reads
    # Individual reads should overwhelmingly reach the optimum on tiny problems.
    assert hits / total_reads > 0.5


@pytest.mark.statistical
def test_native_solver_on_compiled_constrained_program() -> None:
    """End-to-end: program -> BQM -> Ising -> hardware -> solver is feasible-optimal."""
    program = QuadraticProgram()
    x = program.binary_array("x", 4)
    weights = [3.0, -1.0, 2.0, -2.5]
    program.maximize(sum(w * v for w, v in zip(weights, x, strict=True)))
    program.add_linear_equality(x[0] + x[1] + x[2] + x[3] - 2, name="k", penalty=20)
    compiled = compile_program(program)
    problem = binary_to_ising(compiled.bqm)
    hardware = compile_hardware_ising(problem)
    result = NativeIsingSolver(seed=11, num_reads=8, sweeps=300).solve(hardware)
    bits = ((result.best_sample + 1) // 2).tolist()
    assert compiled.is_feasible(bits)
    assert compiled.evaluate_objective(bits) == pytest.approx(5.0)  # x0 + x2
    assert result.best_objective_value == pytest.approx(5.0)


def test_native_solver_is_reproducible_for_fixed_seed() -> None:
    """Identical seeds give identical results; max uint64 seed is accepted."""
    problem = _random_problem(np.random.default_rng(3), 6)
    program = compile_hardware_ising(problem)
    r1 = NativeIsingSolver(seed=5, num_reads=6, sweeps=50).solve(program)
    r2 = NativeIsingSolver(seed=5, num_reads=6, sweeps=50).solve(program)
    np.testing.assert_array_equal(r1.hardware_samples, r2.hardware_samples)
    np.testing.assert_array_equal(r1.energies, r2.energies)
    big = NativeIsingSolver(
        seed=int(np.iinfo(np.uint64).max), num_reads=2, sweeps=10, beta_end=0.1
    ).solve(program)
    assert big.num_reads == 2
    _check_result_consistency(big, program, problem)


def test_native_solver_edge_sizes() -> None:
    """Zero-spin and field-free single-spin programs are handled."""
    empty = compile_hardware_ising(IsingProblem((), np.zeros(0), np.zeros((0, 0))))
    result = NativeIsingSolver(seed=1, num_reads=2).solve(empty)
    assert result.hardware_samples.shape == (2, 0)
    assert result.best_energy == 0.0
    single = compile_hardware_ising(IsingProblem(("a",), [0.0], [[0.0]], offset=2))
    result = NativeIsingSolver(seed=1, num_reads=3, sweeps=5).solve(single)
    assert np.all(result.objective_values == 2.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": 2**64}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.0}, "seed"),
        ({"num_reads": 0}, "num_reads"),
        ({"num_reads": True}, "num_reads"),
        ({"sweeps": 1.5}, "sweeps"),
        ({"beta_start": 0.0}, "beta_start"),
        ({"beta_start": math.nan}, "beta_start"),
        ({"beta_end": math.inf}, "beta_end"),
        ({"beta_start": 2.0, "beta_end": 1.0}, "beta_end"),
    ],
)
def test_native_solver_option_validation(kwargs: dict[str, object], match: str) -> None:
    """Invalid solver options raise ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        NativeIsingSolver(**kwargs)  # type: ignore[arg-type]


def test_native_solver_rejects_non_program() -> None:
    """``solve`` requires a ``HardwareIsingProgram``."""
    with pytest.raises(TypeError):
        NativeIsingSolver().solve(np.zeros((2, 2)))  # type: ignore[arg-type]


def test_native_solver_missing_or_stale_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing or stale native extensions raise ``NativeIsingSolverError``."""
    monkeypatch.setattr(native_module, "_core", None)
    with pytest.raises(NativeIsingSolverError, match="not installed"):
        NativeIsingSolver()
    monkeypatch.setattr(native_module, "_core", SimpleNamespace())
    with pytest.raises(NativeIsingSolverError, match="Rebuild"):
        NativeIsingSolver()
    assert issubclass(NativeIsingSolverError, ImportError)


@pytest.mark.parametrize(
    ("samples", "energies", "match"),
    [
        (np.ones((2, 3)), np.zeros(2), "shape"),
        (np.ones(2), np.zeros(1), "shape"),
        (np.zeros((2, 2)), np.zeros(2), "outside"),
        (np.ones((2, 2)), np.zeros(3), "finite vector"),
        (np.ones((2, 2)), np.array([0.0, math.nan]), "finite vector"),
    ],
)
def test_native_solver_rejects_malformed_core_output(
    monkeypatch: pytest.MonkeyPatch,
    samples: np.ndarray,
    energies: np.ndarray,
    match: str,
) -> None:
    """Malformed native results raise ``RuntimeError``."""
    solver = NativeIsingSolver()
    fake = SimpleNamespace(
        _solve_dense_ising=lambda *args: SimpleNamespace(
            samples=samples, energies=energies
        )
    )
    monkeypatch.setattr(native_module, "_core", fake)
    program = HardwareIsingProgram(np.zeros((2, 2)), ("a", "b"), 2)
    with pytest.raises(RuntimeError, match=match):
        solver.solve(program)
