"""Tests for :class:`quspect.operators.FermionOperator`.

``FermionOperator`` has no operator product or Hermitian conjugate, so the
tests build them from the public term mapping.  Operator *values* are checked
in an independent Fock-space representation: hand-written numpy
Jordan-Wigner ladder matrices (qubit 0 = least-significant bit, matching the
quspect statevector / qiskit convention, occupation ``|1>``).
"""

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
import pytest

import quspect
from quspect import FermionOperator, Parameter
from quspect.circuit.parameter import ParameterExpression
from quspect.operators.fermion_operator import _format_fermion_term

FTerm = tuple[tuple[int, int], ...]

_SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1| = a
_Z = np.diag([1.0, -1.0]).astype(complex)
_I2 = np.eye(2, dtype=complex)


# =============================================================================
# Local helpers
# =============================================================================


def _ladder(mode: int, action: int, n: int) -> np.ndarray:
    """Numpy JW ladder matrix: a_j = Z_0 ... Z_{j-1} sigma^-_j (little-endian)."""
    # kron factors ordered qubit n-1 (leftmost) ... qubit 0 (rightmost).
    factors = []
    for q in range(n - 1, -1, -1):
        if q == mode:
            factors.append(_SIGMA_MINUS)
        elif q < mode:
            factors.append(_Z)
        else:
            factors.append(_I2)
    mat = reduce(np.kron, factors)
    return mat.conj().T if action == 1 else mat


def _fmatrix(op: FermionOperator, n: int) -> np.ndarray:
    """Dense Fock-space matrix of a numeric ``FermionOperator`` on n modes."""
    total = np.zeros((2**n, 2**n), dtype=complex)
    for term, coeff in op:
        mat = np.eye(2**n, dtype=complex)
        for mode, action in term:
            mat = mat @ _ladder(mode, action, n)
        total += complex(coeff) * mat  # type: ignore[arg-type]
    return total


def _fprod(a: FermionOperator, b: FermionOperator) -> FermionOperator:
    """Operator product a * b by term concatenation."""
    out = FermionOperator.zero()
    for ta, ca in a:
        for tb, cb in b:
            out = out + FermionOperator({ta + tb: complex(ca) * complex(cb)})  # type: ignore[arg-type]
    return out


def _fdagger(op: FermionOperator) -> FermionOperator:
    """Hermitian conjugate: reverse each term, flip actions, conjugate."""
    return FermionOperator(
        {
            tuple((m, 1 - a) for m, a in reversed(term)): complex(c).conjugate()  # type: ignore[arg-type]
            for term, c in op
        }
    )


def _random_term(rng: np.random.Generator, n: int, length: int) -> FTerm:
    """Random product of ``length`` ladder operators on ``n`` modes."""
    return tuple((int(rng.integers(n)), int(rng.integers(2))) for _ in range(length))


def _is_normal_ordered(term: FTerm) -> bool:
    """True if no annihilator stands left of a creator."""
    seen_annihilator = False
    for _, action in term:
        if action == 0:
            seen_annihilator = True
        elif seen_annihilator:
            return False
    return True


# =============================================================================
# Reference matrices sanity
# =============================================================================


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_reference_ladder_matrices_are_fermionic(n: int) -> None:
    """The numpy reference satisfies the CAR (guards the reference itself)."""
    eye = np.eye(2**n)
    for i in range(n):
        for j in range(n):
            ai, aj = _ladder(i, 0, n), _ladder(j, 0, n)
            np.testing.assert_allclose(
                ai @ aj.conj().T + aj.conj().T @ ai, eye * (i == j), atol=1e-14
            )
            np.testing.assert_allclose(ai @ aj + aj @ ai, 0, atol=1e-14)


# =============================================================================
# Construction / validation
# =============================================================================


def test_package_export() -> None:
    """FermionOperator is exported from quspect and quspect.operators."""
    assert quspect.FermionOperator is FermionOperator


def test_constructor_merges_and_drops_zeros() -> None:
    """Duplicate keys merge, zero / cancelled coefficients are removed."""
    op = FermionOperator({((0, 1), (1, 0)): 1.0, ((1, 1),): 0.0, (): 0j})
    assert op.terms == {((0, 1), (1, 0)): 1.0}
    assert FermionOperator().is_zero
    assert FermionOperator(None).is_zero
    # Order of ladder operators is significant: no normalization of keys.
    assert len(op + FermionOperator({((1, 0), (0, 1)): 1.0})) == 2
    terms = op.terms
    terms[()] = 3.0
    assert () not in op.terms


@pytest.mark.parametrize(
    ("terms", "exc", "match"),
    [
        ({"a0": 1.0}, TypeError, "tuple"),
        ({((0, 1, 2),): 1.0}, ValueError, "mode, action"),
        ({(0,): 1.0}, ValueError, "mode, action"),
        ({((-1, 1),): 1.0}, ValueError, "non-negative"),
        ({(("0", 1),): 1.0}, ValueError, "non-negative"),
        ({((0.0, 1),): 1.0}, ValueError, "non-negative"),
        ({((0, 2),): 1.0}, ValueError, "Action"),
        ({((0, -1),): 1.0}, ValueError, "Action"),
    ],
)
def test_constructor_rejects_invalid_terms(
    terms: Any, exc: type[Exception], match: str
) -> None:
    """Invalid term structures, modes or actions raise."""
    with pytest.raises(exc, match=match):
        FermionOperator(terms)


def test_named_constructors() -> None:
    """zero/identity/number/hopping/coulomb builders produce the documented terms."""
    assert FermionOperator.zero().is_zero
    assert FermionOperator.identity().terms == {(): 1.0}
    assert FermionOperator.identity(2j).terms == {(): 2j}
    assert FermionOperator.number_operator(2, 0.5).terms == {((2, 1), (2, 0)): 0.5}
    assert FermionOperator.hopping_term(0, 1, 0.5).terms == {
        ((0, 1), (1, 0)): 0.5,
        ((1, 1), (0, 0)): 0.5,
    }
    assert FermionOperator.coulomb_term(0, 1, 2, 3, 0.7).terms == {
        ((0, 1), (1, 1), (2, 0), (3, 0)): 0.7
    }


def test_hopping_term_same_mode_is_twice_number_operator() -> None:
    """hopping_term(i, i, c) = c (a†_i a_i + a†_i a_i) = 2c n_i."""
    op = FermionOperator.hopping_term(1, 1, 0.5)
    np.testing.assert_allclose(
        _fmatrix(op, 2), _fmatrix(FermionOperator.number_operator(1, 1.0), 2)
    )


def test_named_constructor_errors() -> None:
    """Negative mode indices are rejected by every builder."""
    with pytest.raises(ValueError, match="non-negative"):
        FermionOperator.number_operator(-1)
    with pytest.raises(ValueError, match="non-negative"):
        FermionOperator.hopping_term(0, -2)
    with pytest.raises(ValueError, match="non-negative"):
        FermionOperator.hopping_term(-1, 0)
    for args in [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)]:
        with pytest.raises(ValueError, match="must be non-negative"):
            FermionOperator.coulomb_term(*args)


@pytest.mark.parametrize("n", [2, 3])
def test_named_constructors_matrices(n: int) -> None:
    """Number, hopping and Coulomb terms have the correct Fock matrices."""
    a = [_ladder(i, 0, n) for i in range(n)]
    ad = [x.conj().T for x in a]
    num = FermionOperator.number_operator(n - 1, 0.5)
    np.testing.assert_allclose(_fmatrix(num, n), 0.5 * ad[-1] @ a[-1], atol=1e-14)
    hop = FermionOperator.hopping_term(0, n - 1, 0.3)
    ref = 0.3 * (ad[0] @ a[-1] + ad[-1] @ a[0])
    np.testing.assert_allclose(_fmatrix(hop, n), ref, atol=1e-14)
    np.testing.assert_allclose(ref, ref.conj().T, atol=1e-14)
    coul = FermionOperator.coulomb_term(0, 1, 1, 0)
    np.testing.assert_allclose(
        _fmatrix(coul, n), ad[0] @ ad[1] @ a[1] @ a[0], atol=1e-14
    )
    # a†0 a†1 a1 a0 = n0 n1.
    np.testing.assert_allclose(
        _fmatrix(coul, n), ad[0] @ a[0] @ ad[1] @ a[1], atol=1e-14
    )


def test_queries() -> None:
    """is_zero / is_identity / mode_count / len / iteration."""
    assert FermionOperator.zero().is_identity
    assert FermionOperator.zero().mode_count() == 0
    assert FermionOperator.identity(3.0).is_identity
    assert FermionOperator.identity().mode_count() == 0
    op = FermionOperator({((0, 1), (4, 0)): 1.0, (): 1.0})
    assert not op.is_identity and not op.is_zero
    assert op.mode_count() == 5
    assert len(op) == 2
    assert dict(iter(op)) == op.terms


# =============================================================================
# Arithmetic
# =============================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_arithmetic_matches_matrices(seed: int) -> None:
    """+, -, unary -, scalar *, / are linear on the Fock matrices."""
    rng = np.random.default_rng(seed)
    n = 3
    a = FermionOperator(
        {_random_term(rng, n, 2): complex(rng.normal(), rng.normal()) for _ in range(4)}
    )
    b = FermionOperator(
        {_random_term(rng, n, 2): float(rng.normal()) for _ in range(4)}
    )
    ma, mb = _fmatrix(a, n), _fmatrix(b, n)
    c = complex(rng.normal(), rng.normal())
    eye = np.eye(2**n)
    cases = [
        (a + b, ma + mb),
        (a - b, ma - mb),
        (-a, -ma),
        (+a, ma),
        (c * a, c * ma),
        (a * c, c * ma),
        (2 * b, 2 * mb),
        (a / c, ma / c),
        (b / c, mb / c),
        (b / 4, mb / 4),
        (a + 1.5, ma + 1.5 * eye),
        (1.5 + a, ma + 1.5 * eye),
        (a - 2j, ma - 2j * eye),
        (2 - a, 2 * eye - ma),
    ]
    for ours, ref in cases:
        np.testing.assert_allclose(_fmatrix(ours, n), ref, atol=1e-12)
    assert (a - a).is_zero
    assert a + 0 is a
    assert a - 0 is a


def test_inplace_and_zero_scalars() -> None:
    """In-place operators return new objects; zero scalars give zero."""
    op = FermionOperator.number_operator(0)
    alias = op
    op += FermionOperator.number_operator(1)
    op -= 1.0
    op *= 2
    assert alias == FermionOperator.number_operator(0)
    assert op.terms == {((0, 1), (0, 0)): 2.0, ((1, 1), (1, 0)): 2.0, (): -2.0}
    assert (0 * op).is_zero
    assert (op * 0j).is_zero
    with pytest.raises(ZeroDivisionError):
        op / 0
    with pytest.raises(ZeroDivisionError):
        op / 0j


def test_unsupported_operand_types() -> None:
    """Non-numeric operands raise TypeError via NotImplemented."""
    op = FermionOperator.number_operator(0)
    for bad in ("x", [1], None):
        with pytest.raises(TypeError):
            op + bad  # type: ignore[operator]
        with pytest.raises(TypeError):
            op - bad  # type: ignore[operator]
        with pytest.raises(TypeError):
            bad - op  # type: ignore[operator]
        with pytest.raises(TypeError):
            op * bad  # type: ignore[operator]
        with pytest.raises(TypeError):
            op / bad  # type: ignore[operator]
    with pytest.raises(TypeError):
        op * op  # type: ignore[operator]
    assert op.__rsub__("x") is NotImplemented  # type: ignore[arg-type]
    other = FermionOperator.number_operator(1)
    assert op.__rsub__(other) == other - op
    assert (op == 1.0) is False


@pytest.mark.parametrize("seed", [3, 4])
def test_term_product_and_dagger_semantics(seed: int) -> None:
    """Term tuples are ordered products; reversed/flipped terms give the adjoint."""
    rng = np.random.default_rng(seed)
    n = 3
    a = FermionOperator(
        {_random_term(rng, n, 2): complex(rng.normal(), rng.normal()) for _ in range(3)}
    )
    b = FermionOperator(
        {_random_term(rng, n, 3): complex(rng.normal(), rng.normal()) for _ in range(3)}
    )
    np.testing.assert_allclose(
        _fmatrix(_fprod(a, b), n), _fmatrix(a, n) @ _fmatrix(b, n), atol=1e-12
    )
    np.testing.assert_allclose(
        _fmatrix(_fdagger(a), n), _fmatrix(a, n).conj().T, atol=1e-12
    )
    herm = a + _fdagger(a)
    m = _fmatrix(herm, n)
    np.testing.assert_allclose(m, m.conj().T, atol=1e-12)


# =============================================================================
# simplify / compress / normal_order
# =============================================================================


def test_simplify_and_compress() -> None:
    """simplify is idempotent; compress drops small numeric coefficients only."""
    t = Parameter("t")
    op = FermionOperator(
        {((0, 1),): 1.0, ((1, 1),): 1e-14, ((2, 0),): 1e-14j, ((3, 0),): 0.5j}
    )
    assert op.simplify() == op
    assert op.compress().terms == {((0, 1),): 1.0, ((3, 0),): 0.5j}
    assert op.compress(tol=0.6).terms == {((0, 1),): 1.0}
    sym = FermionOperator({((0, 1),): t, ((1, 0),): 1e-20})
    assert sym.compress().terms == {((0, 1),): t}


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        (((0, 0), (1, 1)), {((1, 1), (0, 0)): -1.0}),
        (((0, 0), (0, 1)), {(): 1.0, ((0, 1), (0, 0)): -1.0}),
        (((0, 1), (1, 0)), {((0, 1), (1, 0)): 1.0}),
        ((), {(): 1.0}),
        (
            ((0, 0), (1, 0), (1, 1)),
            {((0, 0),): 1.0, ((1, 1), (0, 0), (1, 0)): 1.0},
        ),
    ],
)
def test_normal_order_simple_cases(term: FTerm, expected: dict[FTerm, float]) -> None:
    """Hand-checked normal ordering of short terms."""
    op = FermionOperator({term: 1.0})
    result = op.normal_order()
    assert result == FermionOperator(expected)
    np.testing.assert_allclose(_fmatrix(result, 2), _fmatrix(op, 2), atol=1e-14)


def test_normal_order_symbolic_coefficient() -> None:
    """Symbolic coefficients pass through normal ordering."""
    t = Parameter("t")
    op = t * FermionOperator({((0, 0), (0, 1)): 1.0})
    result = op.normal_order()
    assert result.parameters == {t}
    assert result.bind_parameters({t: 2.0}) == FermionOperator(
        {(): 2.0, ((0, 1), (0, 0)): -2.0}
    )


@pytest.mark.parametrize("seed", range(6))
def test_normal_order_output_is_normal_ordered(seed: int) -> None:
    """Every term of normal_order() has all creators left of annihilators."""
    rng = np.random.default_rng(seed)
    op = FermionOperator(
        {_random_term(rng, 3, int(rng.integers(1, 5))): 1.0 for _ in range(5)}
    )
    for term, _ in op.normal_order():
        assert _is_normal_ordered(term)


def test_normal_order_delta_sign_after_previous_swap() -> None:
    """Minimal repro: a0 a†1 a†0 = -a†1 + a†1 a†0 a0."""
    op = FermionOperator({((0, 0), (1, 1), (0, 1)): 1.0})
    expected = FermionOperator({((1, 1),): -1.0, ((1, 1), (0, 1), (0, 0)): 1.0})
    np.testing.assert_allclose(_fmatrix(expected, 2), _fmatrix(op, 2), atol=1e-14)
    assert op.normal_order() == expected


def test_normal_order_preserves_operator_value_random() -> None:
    """normal_order() must leave the Fock-space matrix unchanged."""
    rng = np.random.default_rng(7)
    n = 3
    for _ in range(20):
        op = FermionOperator(
            {_random_term(rng, n, int(rng.integers(1, 6))): float(rng.normal())}
        )
        np.testing.assert_allclose(
            _fmatrix(op.normal_order(), n), _fmatrix(op, n), atol=1e-12
        )


# =============================================================================
# Symbolic coefficients / binding
# =============================================================================


def test_symbolic_coefficients() -> None:
    """Parameters, affine expressions, binding and the associated errors."""
    t, s = Parameter("t"), Parameter("s")
    n0 = FermionOperator.number_operator(0)
    op = t * n0 + (2 * s + 1) * FermionOperator.number_operator(1)
    assert op.parameters == {t, s}
    assert (n0 * t).parameters == {t}
    assert ((t * n0) * 2).terms == {((0, 1), (0, 0)): 2 * t}
    assert (n0 + t).terms[()] == t
    assert (n0 - t).parameters == {t}
    assert (t - n0).parameters == {t}
    assert (t * n0 - t * n0).is_zero
    assert ((2 * t * n0) / 2).terms == {((0, 1), (0, 0)): t}
    assert ((2 * t * n0) / (2 + 0j)).terms == {((0, 1), (0, 0)): t}
    assert (n0 / ParameterExpression(constant=4.0)).terms == {((0, 1), (0, 0)): 0.25}
    with pytest.raises(ValueError, match="symbolic"):
        (t * n0) * s
    with pytest.raises(ValueError, match="symbolic"):
        n0 / t
    with pytest.raises(ValueError, match="symbolic"):
        n0 / (t + 1)
    with pytest.raises(ValueError, match="imaginary"):
        t * n0 + 1j * n0
    bound = op.bind_parameters({t: 0.5, s: 1.0})
    assert bound == FermionOperator({((0, 1), (0, 0)): 0.5, ((1, 1), (1, 0)): 3.0})
    assert (t * n0).bind_parameters({t: 0.0}).is_zero
    numeric = FermionOperator({((0, 1),): 1j, ((1, 0),): 2.0})
    assert numeric.bind_parameters({}) == numeric
    with pytest.raises(ValueError, match="not bound"):
        (t * n0).bind_parameters({s: 1.0})


# =============================================================================
# Equality / hash / representation
# =============================================================================


def test_equality_and_hash() -> None:
    """Tolerant numeric equality, symbolic equality, consistent hashing."""
    t = Parameter("t")
    a = FermionOperator({((0, 1), (1, 0)): 1.0, (): 0.5j})
    assert a == FermionOperator({((0, 1), (1, 0)): 1, (): 0.5j + 1e-15})
    assert a != FermionOperator({((0, 1), (1, 0)): 1.0 + 1e-9, (): 0.5j})
    assert a != FermionOperator({((0, 1), (1, 0)): 1.0})
    assert a != FermionOperator({((1, 0), (0, 1)): 1.0, (): 0.5j})
    assert a != FermionOperator({((0, 1), (1, 0)): 1.0, (): 0.6j})
    assert hash(a) == hash(FermionOperator({((0, 1), (1, 0)): 1, (): 0.5j}))
    sym = FermionOperator({((0, 1),): t})
    assert sym == FermionOperator({((0, 1),): t})
    assert sym != FermionOperator({((0, 1),): 1.0})
    assert hash(sym) == hash(FermionOperator({((0, 1),): t}))


def test_repr_and_latex() -> None:
    """repr/str/_repr_latex_ formatting."""
    t = Parameter("t")
    assert repr(FermionOperator.zero()) == "FermionOperator(0)"
    op = FermionOperator({((0, 1), (1, 0)): -1.0, (): 2, ((2, 0),): 0.5j})
    assert repr(op) == "FermionOperator(2 I + -a†0 a1 + 0.5j a2)"
    assert str(op) == repr(op)
    assert repr(FermionOperator.number_operator(0)) == "FermionOperator(a†0 a0)"
    assert repr(t * FermionOperator.number_operator(0)) == (
        "FermionOperator(Parameter('t') a†0 a0)"
    )
    assert FermionOperator.zero()._repr_latex_() == "$0$"
    assert FermionOperator({((0, 1), (1, 0)): 0.25})._repr_latex_() == (
        r"$0.25a^{\dagger}_{0}\, a_{1}$"
    )
    assert FermionOperator({((2, 0),): 0.5j, (): 2 + 0j})._repr_latex_() == (
        r"$2\mathbb{I} + (0+0.5i)a_{2}$"
    )
    assert (t * FermionOperator.number_operator(0))._repr_latex_() == (
        r"$\mathrm{t}a^{\dagger}_{0}\, a_{0}$"
    )
    assert ((t + 1) * FermionOperator.number_operator(0))._repr_latex_() == (
        r"$\mathrm{t + 1}a^{\dagger}_{0}\, a_{0}$"
    )
    assert _format_fermion_term(()) == "I"
    assert _format_fermion_term(((0, 2),)) == "a^{2}_{0}"


def test_latex_omits_unit_coefficients() -> None:
    """Coefficients +-1 should not be printed in LaTeX output."""
    op = FermionOperator({((0, 1), (1, 0)): -1.0, (): 1.0})
    assert op._repr_latex_() == r"$\mathbb{I} - a^{\dagger}_{0}\, a_{1}$"
