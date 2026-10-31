"""Tests for ``quspect.optimization`` models, conversions and compilation.

Conventions verified here (from the source and by brute force):

* ``BinaryQuadraticModel`` / ``IsingProblem`` energy is
  ``offset + linear @ v + 0.5 * v.T @ quadratic @ v`` with a symmetric,
  zero-diagonal ``quadratic`` (each pair counted once).
* binary <-> spin mapping is ``s = 2x - 1`` (``x=1 -> s=+1``,
  ``x=0 -> s=-1``).
* Qubit mapping used for the qiskit cross-check: variable ``i`` is qubit
  ``i`` and the computational-basis bit of qubit ``i`` equals ``x_i``.  Since
  ``Z|0> = +|0>`` the Z eigenvalue is ``z_i = 1 - 2 x_i = -s_i``.  Hence the
  diagonal operator is ``offset - sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``
  and its diagonal entry ``k`` (little-endian: ``x_i`` = bit ``i`` of ``k``)
  equals the energy of the binary assignment encoded by ``k``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from quspect.optimization import (
    BinaryQuadraticModel,
    BinaryVariable,
    CompilationResult,
    CompiledQuadraticProgram,
    IsingProblem,
    OptimizationSense,
    QuadraticExpression,
    QuadraticProgram,
    VariableType,
    binary_samples_to_spins,
    binary_to_ising,
    bqm_from_qubo_matrix,
    bqm_to_ising,
    compile_program,
    compile_quadratic_program,
    ising_to_binary,
    qubo_matrix_from_bqm,
    spin_samples_to_binary,
)
from quspect.optimization.compiler import compile as compile_fn


def _bits_matrix(n: int) -> np.ndarray:
    """Return all ``2**n`` binary assignments; row ``k`` has ``x_i`` = bit i."""
    k = np.arange(2**n)
    return ((k[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)


def _random_symmetric(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return a random symmetric matrix with zero diagonal."""
    upper = np.triu(rng.normal(size=(n, n)), k=1)
    return upper + upper.T


def _random_bqm(
    rng: np.random.Generator,
    n: int,
    *,
    sense: OptimizationSense = OptimizationSense.MINIMIZE,
) -> BinaryQuadraticModel:
    """Return a random binary-domain model on ``n`` variables."""
    return BinaryQuadraticModel(
        variables=tuple(f"v{i}" for i in range(n)),
        linear=rng.normal(size=n),
        quadratic=_random_symmetric(rng, n),
        offset=float(rng.normal()),
        variable_type=VariableType.BINARY,
        sense=sense,
    )


def _reference_energy(
    linear: np.ndarray, quadratic: np.ndarray, offset: float, v: np.ndarray
) -> float:
    """Independent loop-based evaluation of the documented energy formula."""
    n = len(linear)
    energy = offset + sum(linear[i] * v[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            energy += quadratic[i, j] * v[i] * v[j]
    return float(energy)


# ---------------------------------------------------------------------------
# BinaryQuadraticModel / IsingProblem
# ---------------------------------------------------------------------------


def test_bqm_evaluate_matches_reference(rng: np.random.Generator) -> None:
    """``evaluate`` and ``evaluate_many`` match the documented formula."""
    model = _random_bqm(rng, 5)
    bits = _bits_matrix(5)
    many = model.evaluate_many(bits)
    assert not many.flags.writeable
    for row, value in zip(bits, many, strict=True):
        ref = _reference_energy(model.linear, model.quadratic, model.offset, row)
        assert math.isclose(model.evaluate(row), ref, abs_tol=1e-12)
        assert math.isclose(float(value), ref, abs_tol=1e-12)
    assert model.num_variables == 5
    assert not model.linear.flags.writeable
    assert not model.quadratic.flags.writeable


def test_bqm_docstring_example_and_string_enums() -> None:
    """The docstring example evaluates to 1.0; enums accept string values."""
    model = BinaryQuadraticModel(
        variables=("x", "y"),
        linear=np.array([1.0, -2.0]),
        quadratic=np.array([[0.0, 3.0], [3.0, 0.0]]),
        variable_type="binary",  # type: ignore[arg-type]
        sense="maximize",  # type: ignore[arg-type]
    )
    assert model.evaluate([1, 1]) == 2.0
    assert model.variable_type is VariableType.BINARY
    assert model.sense is OptimizationSense.MAXIMIZE


def test_bqm_canonicalizes_near_symmetric_quadratic() -> None:
    """Asymmetry within 1e-12 is averaged away, giving exact symmetry."""
    q = np.array([[0.0, 1.0 + 5e-13], [1.0, 0.0]])
    model = BinaryQuadraticModel(("a", "b"), [0.0, 0.0], q)
    assert model.quadratic[0, 1] == model.quadratic[1, 0]


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_bqm_large_finite_quadratic_stays_finite() -> None:
    """An accepted finite symmetric quadratic must be stored unchanged."""
    q = np.array([[0.0, 1e308], [1e308, 0.0]])
    model = BinaryQuadraticModel(("a", "b"), [0.0, 0.0], q)
    np.testing.assert_array_equal(model.quadratic, q)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"variables": ("a", "a")}, "unique"),
        ({"variables": ("a", "")}, "non-empty"),
        ({"variables": ("a", " b")}, "whitespace"),
        ({"variables": ("a", 3)}, "non-empty"),
        ({"variable_type": "ternary"}, "variable type"),
        ({"sense": "sideways"}, "sense"),
        ({"offset": math.nan}, "offset"),
        ({"linear": [1.0]}, "shape"),
        ({"linear": [1.0, math.inf]}, "finite"),
        ({"quadratic": np.zeros((3, 3))}, "shape"),
        ({"quadratic": [[0.0, math.nan], [math.nan, 0.0]]}, "finite"),
        ({"quadratic": [[0.0, 1.0], [2.0, 0.0]]}, "symmetric"),
        ({"quadratic": [[1.0, 0.0], [0.0, 0.0]]}, "diagonal"),
    ],
)
def test_bqm_validation_errors(kwargs: dict[str, object], match: str) -> None:
    """Invalid names, enums, offsets, shapes and matrices raise ValueError."""
    base: dict[str, object] = {
        "variables": ("a", "b"),
        "linear": [0.0, 0.0],
        "quadratic": np.zeros((2, 2)),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        BinaryQuadraticModel(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sample", "vtype"),
    [
        ([0, 1, 1], VariableType.BINARY),
        ([[0, 1]], VariableType.BINARY),
        ([[[0, 1]]], VariableType.BINARY),
        ([0, 2], VariableType.BINARY),
        ([0, math.nan], VariableType.BINARY),
        ([-1, 1], VariableType.BINARY),
        ([0, 1], VariableType.SPIN),
    ],
)
def test_bqm_sample_validation(sample: object, vtype: VariableType) -> None:
    """Wrong-shape, non-finite or out-of-domain samples raise ValueError."""
    model = BinaryQuadraticModel(("a", "b"), [1.0, 1.0], np.zeros((2, 2)), 0, vtype)
    if np.asarray(sample).shape == (1, 2):
        # A (1, n) batch is accepted by evaluate_many but evaluate takes row 0.
        assert model.evaluate(sample) == 1.0
        return
    with pytest.raises(ValueError):
        model.evaluate(sample)
    with pytest.raises(ValueError):
        model.evaluate_many(sample)


def test_ising_problem_properties_and_evaluate(rng: np.random.Generator) -> None:
    """``IsingProblem`` stores spin-domain data and evaluates spins."""
    n = 4
    h = rng.normal(size=n)
    j = _random_symmetric(rng, n)
    problem = IsingProblem(tuple("abcd"), h, j, offset=0.25, sense="maximize")  # type: ignore[arg-type]
    assert problem.num_spins == 4
    assert problem.sense is OptimizationSense.MAXIMIZE
    bqm = problem.as_bqm()
    assert bqm.variable_type is VariableType.SPIN
    spins = 2 * _bits_matrix(n) - 1
    many = problem.evaluate_many(spins)
    for row, value in zip(spins, many, strict=True):
        ref = _reference_energy(h, j, 0.25, row)
        assert math.isclose(problem.evaluate(row), ref, abs_tol=1e-12)
        assert math.isclose(float(value), ref, abs_tol=1e-12)
    with pytest.raises(ValueError):
        problem.evaluate([0, 1, 1, 1])
    with pytest.raises(ValueError):
        IsingProblem(("a",), [1.0], [[1.0]])
    # Docstring example.
    doc = IsingProblem(("s0", "s1"), np.zeros(2), np.array([[0.0, 1.0], [1.0, 0]]))
    assert doc.evaluate([1, -1]) == -1.0


# ---------------------------------------------------------------------------
# binary <-> Ising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 6, 10])
@pytest.mark.parametrize("sense", list(OptimizationSense))
def test_binary_to_ising_preserves_every_energy(
    n: int, sense: OptimizationSense, rng: np.random.Generator
) -> None:
    """Exhaustively, ``E_bin(x) == E_ising(2x-1)`` for all assignments."""
    model = _random_bqm(rng, n, sense=sense)
    problem = binary_to_ising(model)
    assert problem.variables == model.variables
    assert problem.sense is sense
    bits = _bits_matrix(n)
    np.testing.assert_allclose(
        problem.evaluate_many(2 * bits - 1), model.evaluate_many(bits), atol=1e-10
    )
    via_view = bqm_to_ising(model)
    np.testing.assert_allclose(via_view.linear, problem.linear)
    assert via_view.offset == pytest.approx(problem.offset)


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_ising_to_binary_preserves_every_energy(
    n: int, rng: np.random.Generator
) -> None:
    """Exhaustively, ``E_ising(s) == E_bin((s+1)/2)``; round trip is exact."""
    problem = IsingProblem(
        tuple(f"s{i}" for i in range(n)),
        rng.normal(size=n),
        _random_symmetric(rng, n),
        offset=float(rng.normal()),
        sense=OptimizationSense.MAXIMIZE,
    )
    model = ising_to_binary(problem)
    assert model.variable_type is VariableType.BINARY
    assert model.sense is OptimizationSense.MAXIMIZE
    bits = _bits_matrix(n)
    np.testing.assert_allclose(
        model.evaluate_many(bits), problem.evaluate_many(2 * bits - 1), atol=1e-10
    )
    back = binary_to_ising(model)
    np.testing.assert_allclose(back.linear, problem.linear, atol=1e-12)
    np.testing.assert_allclose(back.quadratic, problem.quadratic, atol=1e-12)
    assert back.offset == pytest.approx(problem.offset, abs=1e-12)


def test_spin_convention_is_s_equals_2x_minus_1() -> None:
    """A single positive binary bias maps to ``x=1 <-> s=+1``."""
    model = BinaryQuadraticModel(("x",), np.array([2.0]), np.zeros((1, 1)))
    problem = binary_to_ising(model)
    assert problem.evaluate([1]) == model.evaluate([1]) == 2.0
    assert problem.evaluate([-1]) == model.evaluate([0]) == 0.0
    assert ising_to_binary(
        IsingProblem(("s",), np.array([1.0]), np.zeros((1, 1)))
    ).evaluate([0]) == pytest.approx(-1.0)


def test_bqm_to_ising_on_spin_model_and_binary_to_ising_rejects_spin() -> None:
    """A spin BQM is viewed unchanged; ``binary_to_ising`` refuses spin input."""
    spin = BinaryQuadraticModel(
        ("a", "b"),
        [1.0, -1.0],
        [[0.0, 2.0], [2.0, 0.0]],
        offset=3.0,
        variable_type=VariableType.SPIN,
        sense=OptimizationSense.MAXIMIZE,
    )
    view = bqm_to_ising(spin)
    np.testing.assert_array_equal(view.linear, spin.linear)
    np.testing.assert_array_equal(view.quadratic, spin.quadratic)
    assert view.offset == 3.0
    assert view.sense is OptimizationSense.MAXIMIZE
    with pytest.raises(ValueError, match="binary-domain"):
        binary_to_ising(spin)
    with pytest.raises(ValueError, match="binary-domain"):
        qubo_matrix_from_bqm(spin)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("n", [1, 3, 5])
def test_ising_hamiltonian_diagonal_matches_qiskit(
    n: int, rng: np.random.Generator
) -> None:
    """``SparsePauliOp`` diagonal (little-endian) equals binary energies.

    Variable i is qubit i, qubit bit = x_i, Z eigenvalue = 1 - 2x_i = -s_i.
    """
    model = _random_bqm(rng, n)
    problem = binary_to_ising(model)
    terms: list[tuple[str, list[int], complex]] = [("", [], problem.offset)]
    terms += [("Z", [i], -problem.linear[i]) for i in range(n)]
    terms += [
        ("ZZ", [i, j], problem.quadratic[i, j])
        for i, j in itertools.combinations(range(n), 2)
    ]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    diag = np.diag(op.to_matrix()).real
    np.testing.assert_allclose(diag, model.evaluate_many(_bits_matrix(n)), atol=1e-10)
    # Equivalent statement in spin space: index k <-> s = -z = 2x - 1.
    np.testing.assert_allclose(
        diag, problem.evaluate_many(2 * _bits_matrix(n) - 1), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def test_sample_conversions_round_trip() -> None:
    """Binary/spin sample conversion keeps shape and is read-only."""
    bits = np.array([[0, 1, 1], [1, 0, 0]])
    spins = binary_samples_to_spins(bits)
    np.testing.assert_array_equal(spins, [[-1, 1, 1], [1, -1, -1]])
    assert spins.dtype == np.int8
    assert not spins.flags.writeable
    back = spin_samples_to_binary(spins)
    np.testing.assert_array_equal(back, bits)
    assert not back.flags.writeable
    np.testing.assert_array_equal(binary_samples_to_spins([True, False]), [1, -1])
    np.testing.assert_array_equal(spin_samples_to_binary([-1.0, 1.0]), [0, 1])


@pytest.mark.parametrize("bad", [0, [[[0]]], [0, 2], [0.5, 1]])
def test_binary_samples_errors(bad: object) -> None:
    """Non 1/2-D or non-binary samples raise ValueError."""
    with pytest.raises(ValueError):
        binary_samples_to_spins(bad)


@pytest.mark.parametrize("bad", [1, [[[1]]], [0, 1], [-1, 2]])
def test_spin_samples_errors(bad: object) -> None:
    """Non 1/2-D or non-spin samples raise ValueError."""
    with pytest.raises(ValueError):
        spin_samples_to_binary(bad)


# ---------------------------------------------------------------------------
# QUBO matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 4, 7])
def test_qubo_matrix_round_trip(n: int, rng: np.random.Generator) -> None:
    """Upper-triangular Q -> BQM -> Q is exact; energies match ``x Q x``."""
    q = np.triu(rng.normal(size=(n, n)))
    model = bqm_from_qubo_matrix(q, offset=1.5)
    assert model.variables == tuple(f"x[{i}]" for i in range(n))
    bits = _bits_matrix(n)
    expected = np.einsum("bi,ij,bj->b", bits, q, bits) + 1.5
    np.testing.assert_allclose(model.evaluate_many(bits), expected, atol=1e-10)
    back = qubo_matrix_from_bqm(model)
    assert not back.flags.writeable
    np.testing.assert_allclose(back, q, atol=1e-12)
    rebuilt = bqm_from_qubo_matrix(back, offset=model.offset)
    np.testing.assert_allclose(rebuilt.quadratic, model.quadratic)


def test_qubo_matrix_symmetric_and_lower_inputs(rng: np.random.Generator) -> None:
    """Symmetric/lower/full Q give identical energies to ``x.T Q x``."""
    n = 4
    full = rng.normal(size=(n, n))
    bits = _bits_matrix(n)
    for q in (full, full + full.T, np.tril(full)):
        model = bqm_from_qubo_matrix(
            q, variables=list("abcd"), sense=OptimizationSense.MAXIMIZE
        )
        assert model.sense is OptimizationSense.MAXIMIZE
        expected = np.einsum("bi,ij,bj->b", bits, q, bits)
        np.testing.assert_allclose(model.evaluate_many(bits), expected, atol=1e-10)
        canon = qubo_matrix_from_bqm(model)
        np.testing.assert_array_equal(canon, np.triu(canon))
        np.testing.assert_allclose(
            np.einsum("bi,ij,bj->b", bits, canon, bits), expected, atol=1e-10
        )


@pytest.mark.parametrize(
    ("matrix", "kwargs"),
    [
        (np.zeros(3), {}),
        (np.zeros((2, 3)), {}),
        ([[0.0, math.inf], [0.0, 0.0]], {}),
        (np.zeros((2, 2)), {"variables": ["a"]}),
        (np.zeros((2, 2)), {"variables": ["a", "a"]}),
    ],
)
def test_bqm_from_qubo_matrix_errors(matrix: object, kwargs: dict[str, object]) -> None:
    """Non-square, non-finite matrices and bad names raise ValueError."""
    with pytest.raises(ValueError):
        bqm_from_qubo_matrix(matrix, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Compilation of constrained programs
# ---------------------------------------------------------------------------


def _assignments(names: tuple[str, ...]) -> list[dict[str, int]]:
    """Return every assignment of ``names`` as dicts."""
    return [
        dict(zip(names, bits, strict=True))
        for bits in itertools.product((0, 1), repeat=len(names))
    ]


@pytest.mark.parametrize("sense", list(OptimizationSense))
def test_compiled_penalties_feasible_equal_infeasible_penalized(
    sense: OptimizationSense, rng: np.random.Generator
) -> None:
    """Feasible: compiled = objective; infeasible: strictly worse."""
    program = QuadraticProgram("p")
    x = program.binary_array("x", 4)
    obj = QuadraticExpression(
        0.3,
        {v: float(rng.normal()) for v in x},
        {(a, b): float(rng.normal()) for a, b in itertools.combinations(x, 2)},
    )
    if sense is OptimizationSense.MINIMIZE:
        program.minimize(obj)
    else:
        program.maximize(obj)
    program.add_linear_equality(x[0] + x[1] + x[2] - 1, name="one_hot", penalty=2.5)
    program.add_nonnegative_quadratic_zero(x[2] * x[3], name="excl", penalty=1.25)
    result = compile_program(program)
    assert isinstance(result, CompilationResult)
    assert result.bqm is result.model is result.binary_quadratic_model
    assert result.bqm.sense is sense
    assert result.variables == ("x[0]", "x[1]", "x[2]", "x[3]")
    n_feasible = 0
    for env in _assignments(result.variables):
        compiled = result.bqm.evaluate([env[n] for n in result.variables])
        original = result.evaluate_objective(env)
        assert original == pytest.approx(obj.evaluate(env))
        feasible = env["x[0]"] + env["x[1]"] + env["x[2]"] == 1 and not (
            env["x[2]"] and env["x[3]"]
        )
        assert result.is_feasible(env) is feasible
        if feasible:
            n_feasible += 1
            assert compiled == pytest.approx(original, abs=1e-10)
        elif sense is OptimizationSense.MINIMIZE:
            assert compiled > original + 1e-9
        else:
            assert compiled < original - 1e-9
    assert n_feasible == 5


def test_linear_equality_penalty_magnitude() -> None:
    """Infeasible penalty equals ``penalty * residual**2`` exactly."""
    program = QuadraticProgram()
    a, b, c = (program.binary_variable(n) for n in "abc")
    program.minimize(a - b)
    program.add_linear_equality(a + b + c - 1, name="eq", penalty=3.0)
    result = compile_quadratic_program(program)
    for env in _assignments(result.variables):
        residual = env["a"] + env["b"] + env["c"] - 1
        expected = env["a"] - env["b"] + 3.0 * residual**2
        assert result.bqm.evaluate(list(env.values())) == pytest.approx(expected)


def test_inequality_via_slack_minimised_over_slack(rng: np.random.Generator) -> None:
    """``sum x <= 1`` encoded with a slack bit: min over slack = objective."""
    program = QuadraticProgram()
    x = program.binary_array("x", 3)
    slack = program.binary_variable("slack")
    weights = rng.normal(size=3)
    program.minimize(QuadraticExpression(0.0, dict(zip(x, weights, strict=True))))
    program.add_linear_equality(x[0] + x[1] + x[2] + slack - 1, name="le", penalty=4)
    result = compile_fn(program)
    for bits in itertools.product((0, 1), repeat=3):
        objective = float(np.dot(weights, bits))
        best = min(result.bqm.evaluate([*bits, s]) for s in (0, 1))
        if sum(bits) <= 1:
            assert best == pytest.approx(objective, abs=1e-10)
        else:
            assert best > objective + 1e-9
    # With a large penalty the global minimum is feasible and optimal.
    bits_all = _bits_matrix(4)
    energies = result.bqm.evaluate_many(bits_all)
    best_row = bits_all[int(np.argmin(energies))]
    assert best_row[:3].sum() <= 1
    feasible_opt = min(
        float(np.dot(weights, b))
        for b in itertools.product((0, 1), repeat=3)
        if sum(b) <= 1
    )
    assert float(np.min(energies)) == pytest.approx(feasible_opt)


def test_compiled_program_through_ising_matches_qiskit_diagonal() -> None:
    """Compile -> Ising -> SparsePauliOp diagonal minimum is the feasible optimum."""
    program = QuadraticProgram()
    x = program.binary_array("x", 3)
    program.minimize(-1.0 * x[0] - 2.0 * x[1] - 3.0 * x[2] + 0.5 * x[0] * x[2])
    program.add_linear_equality(x[0] + x[1] + x[2] - 2, name="two", penalty=10)
    result = compile_program(program)
    problem = binary_to_ising(result.bqm)
    n = 3
    terms: list[tuple[str, list[int], complex]] = [("", [], problem.offset)]
    terms += [("Z", [i], -problem.linear[i]) for i in range(n)]
    terms += [
        ("ZZ", [i, j], problem.quadratic[i, j])
        for i, j in itertools.combinations(range(n), 2)
    ]
    diag = np.diag(SparsePauliOp.from_sparse_list(terms, num_qubits=n).to_matrix())
    k = int(np.argmin(diag.real))
    bits = [(k >> i) & 1 for i in range(n)]
    assert bits == [0, 1, 1]  # x1 + x2 == 2, objective -5
    assert diag[k].real == pytest.approx(-5.0)


def test_compile_errors_and_unsupported_constraint() -> None:
    """Non-programs, missing objective and unknown constraints are rejected."""
    with pytest.raises(TypeError):
        compile_program("nope")  # type: ignore[arg-type]
    program = QuadraticProgram()
    program.binary_variable("a")
    with pytest.raises(ValueError, match="no objective"):
        compile_program(program)
    program.minimize(BinaryVariable("a"))
    program._constraints.append(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported constraint"):
        compile_program(program)


def test_compilation_result_decode_and_assignment_errors() -> None:
    """``decode``/``evaluate_objective``/``is_feasible`` validate input."""
    program = QuadraticProgram()
    a = program.binary_variable("a")
    b = program.binary_variable("b")
    program.minimize(a + 2 * b)
    program.add_linear_equality(a - b, name="eq", penalty=1)
    result: CompiledQuadraticProgram = compile_program(program)
    assert result.decode([1, np.bool_(False)]) == {"a": 1, "b": 0}
    assert result.decode(np.array([1.0, 1.0])) == {"a": 1, "b": 1}
    assert result.evaluate_objective([1, 1]) == 3.0
    assert result.evaluate_objective({a: 1, "b": 0}) == 1.0
    assert result.is_feasible([1, 1])
    assert not result.is_feasible({"a": 1, "b": 0})
    assert result.is_feasible({"a": 1, "b": 0}, tolerance=1)
    with pytest.raises(TypeError):
        result.decode("10")
    with pytest.raises(ValueError, match="one-dimensional"):
        result.decode([[1, 0]])
    with pytest.raises(ValueError, match="Expected 2"):
        result.decode([1])
    with pytest.raises(ValueError, match="binary domain"):
        result.decode([1, 2])
    with pytest.raises(TypeError):
        result.evaluate_objective({1: 0, "b": 0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="duplicate"):
        result.evaluate_objective({a: 1, "a": 1, "b": 0})
    with pytest.raises(ValueError, match="missing"):
        result.evaluate_objective({"a": 1})
    with pytest.raises(ValueError, match="unknown"):
        result.evaluate_objective({"a": 1, "b": 0, "c": 1})
    for tol, exc in [(True, TypeError), ("x", TypeError), (-1.0, ValueError)]:
        with pytest.raises(exc):
            result.is_feasible([1, 1], tolerance=tol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        result.is_feasible([1, 1], tolerance=math.inf)
