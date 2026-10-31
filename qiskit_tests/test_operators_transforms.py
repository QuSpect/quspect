"""Tests for the fermion-to-qubit maps in :mod:`quspect.operators.transforms`.

Matrices of the resulting ``QubitOperator`` are built through qiskit
``SparsePauliOp`` (little-endian: term key ``(q, P)`` -> label position
``n - 1 - q``), so every matrix check here is a qiskit cross-reference.

``bravyi_kitaev`` derives its encoding size from the operator's own maximum
mode index, so transforms of operators with different ``mode_count()`` use
different encodings.  ``_bk(op, n)`` pins the encoding to ``n`` modes by
adding (and subtracting the transform of) ``n_{n-1}`` (equivalent to
``bravyi_kitaev(op, n_qubits=n)``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from math import comb

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from quspect import FermionOperator, Parameter, QubitOperator
from quspect.operators.transforms import (
    _build_bk_columns,
    _operator_mul,
    bravyi_kitaev,
    jordan_wigner,
    reverse_jordan_wigner,
)

pytestmark = pytest.mark.qiskit_ref

FTerm = tuple[tuple[int, int], ...]
Transform = Callable[[FermionOperator, int], QubitOperator]

_SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_I2 = np.eye(2, dtype=complex)


# =============================================================================
# Local helpers
# =============================================================================


def _to_spo(op: QubitOperator, n: int) -> SparsePauliOp:
    """Convert a numeric ``QubitOperator`` to ``SparsePauliOp`` on n qubits."""
    labels: list[str] = []
    coeffs: list[complex] = []
    for term, coeff in op:
        chars = ["I"] * n
        for qubit, pauli in term:
            chars[n - 1 - qubit] = pauli
        labels.append("".join(chars))
        coeffs.append(complex(coeff))  # type: ignore[arg-type]
    if not labels:
        return SparsePauliOp("I" * n, [0.0])
    return SparsePauliOp(labels, np.array(coeffs, dtype=complex))


def _qmat(op: QubitOperator, n: int) -> np.ndarray:
    """Little-endian dense matrix of a ``QubitOperator`` on n qubits."""
    return np.asarray(_to_spo(op, n).to_matrix())


def _jw(op: FermionOperator, n: int) -> QubitOperator:
    """Jordan-Wigner (encoding is independent of n)."""
    del n
    return jordan_wigner(op)


def _bk(op: FermionOperator, n: int) -> QubitOperator:
    """Bravyi-Kitaev with the encoding pinned to ``n`` modes."""
    pad = FermionOperator.number_operator(n - 1)
    return bravyi_kitaev(op + pad) - bravyi_kitaev(pad)


def _np_jw_ladder(mode: int, action: int, n: int) -> np.ndarray:
    """Hand-written JW: a_j = Z_0..Z_{j-1} (X_j + iY_j)/2, qubit 0 rightmost."""
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


def _ladders(transform: Transform, n: int) -> list[np.ndarray]:
    """Matrices of a_0 .. a_{n-1} under ``transform``."""
    return [_qmat(transform(FermionOperator({((j, 0),): 1.0}), n), n) for j in range(n)]


def _car_violations(mats: list[np.ndarray]) -> list[tuple[int, int]]:
    """Pairs (i, j) violating {a_i, a_j^dag} = delta_ij or {a_i, a_j} = 0."""
    dim = mats[0].shape[0]
    bad: list[tuple[int, int]] = []
    for i, ai in enumerate(mats):
        for j, aj in enumerate(mats):
            ajd = aj.conj().T
            ok = np.allclose(ai @ ajd + ajd @ ai, np.eye(dim) * (i == j), atol=1e-12)
            ok = ok and np.allclose(ai @ aj + aj @ ai, 0, atol=1e-12)
            if not ok:
                bad.append((i, j))
    return bad


def _random_term(rng: np.random.Generator, n: int, length: int) -> FTerm:
    """Random ladder-operator product on n modes."""
    return tuple((int(rng.integers(n)), int(rng.integers(2))) for _ in range(length))


def _random_fermion_op(
    rng: np.random.Generator, n: int, num_terms: int
) -> FermionOperator:
    """Random complex FermionOperator with terms of length 1..4."""
    return FermionOperator(
        {
            _random_term(rng, n, int(rng.integers(1, 5))): complex(
                rng.normal(), rng.normal()
            )
            for _ in range(num_terms)
        }
    )


def _random_hermitian_hamiltonian(rng: np.random.Generator, n: int) -> FermionOperator:
    """Random number-conserving Hermitian H = sum h_pq a†p aq + sum V a†a†aa + h.c."""
    h = FermionOperator.zero()
    for p in range(n):
        h = h + FermionOperator.number_operator(p, float(rng.normal()))
        for q in range(p + 1, n):
            c = complex(rng.normal(), rng.normal())
            h = h + FermionOperator(
                {((p, 1), (q, 0)): c, ((q, 1), (p, 0)): c.conjugate()}
            )
    for _ in range(3):
        i, j, k, m = (int(x) for x in rng.integers(n, size=4))
        v = complex(rng.normal(), rng.normal())
        term = ((i, 1), (j, 1), (k, 0), (m, 0))
        dag = ((m, 1), (k, 1), (j, 0), (i, 0))
        h = h + FermionOperator({term: v}) + FermionOperator({dag: v.conjugate()})
    return h


def _fprod(a: FermionOperator, b: FermionOperator) -> FermionOperator:
    """Fermionic operator product by term concatenation."""
    out = FermionOperator.zero()
    for ta, ca in a:
        for tb, cb in b:
            out = out + FermionOperator({ta + tb: complex(ca) * complex(cb)})  # type: ignore[arg-type]
    return out


def _fdagger(op: FermionOperator) -> FermionOperator:
    """Hermitian conjugate of a FermionOperator."""
    return FermionOperator(
        {
            tuple((m, 1 - a) for m, a in reversed(t)): complex(c).conjugate()  # type: ignore[arg-type]
            for t, c in op
        }
    )


# =============================================================================
# Jordan-Wigner
# =============================================================================


def test_jw_docstring_example_number_operator() -> None:
    """n_0 -> (I - Z0)/2 exactly as documented."""
    qop = jordan_wigner(FermionOperator.number_operator(0, coefficient=1.0))
    assert qop == QubitOperator({(): 0.5, ((0, "Z"),): -0.5})
    assert repr(qop) == "QubitOperator(0.5 I + -0.5 Z0)"


def test_jw_ladder_operator_terms() -> None:
    """a_2 -> 0.5 X2 Z1 Z0 + 0.5i Y2 Z1 Z0; a†_2 has -0.5i."""
    a2 = jordan_wigner(FermionOperator({((2, 0),): 1.0}))
    assert a2 == QubitOperator(
        {((0, "Z"), (1, "Z"), (2, "X")): 0.5, ((0, "Z"), (1, "Z"), (2, "Y")): 0.5j}
    )
    a2d = jordan_wigner(FermionOperator({((2, 1),): 1.0}))
    assert a2d == QubitOperator(
        {((0, "Z"), (1, "Z"), (2, "X")): 0.5, ((0, "Z"), (1, "Z"), (2, "Y")): -0.5j}
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_jw_matches_handwritten_numpy(n: int) -> None:
    """Element-wise equality with numpy JW (Z-string on lower qubits)."""
    for j in range(n):
        for action in (0, 1):
            qop = jordan_wigner(FermionOperator({((j, action),): 1.0}))
            np.testing.assert_allclose(
                _qmat(qop, n), _np_jw_ladder(j, action, n), atol=1e-14
            )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jw_matches_handwritten_numpy_for_random_operators(seed: int) -> None:
    """Random multi-term operators agree with numpy products of JW ladders."""
    rng = np.random.default_rng(seed)
    n = 4
    fop = _random_fermion_op(rng, n, 6)
    ref = np.zeros((2**n, 2**n), dtype=complex)
    for term, coeff in fop:
        mat = np.eye(2**n, dtype=complex)
        for mode, action in term:
            mat = mat @ _np_jw_ladder(mode, action, n)
        ref += complex(coeff) * mat  # type: ignore[arg-type]
    np.testing.assert_allclose(_qmat(jordan_wigner(fop), n), ref, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_jw_canonical_anticommutation(n: int) -> None:
    """JW images obey {a_i, a_j†} = δ_ij I and {a_i, a_j} = 0."""
    assert _car_violations(_ladders(_jw, n)) == []


def test_jw_identity_zero_and_nilpotent() -> None:
    """Identity, zero, a_i a_i = 0 and a†a + a a† = 1 at the operator level."""
    assert jordan_wigner(FermionOperator.identity(2.5)) == QubitOperator.identity(2.5)
    assert jordan_wigner(FermionOperator.zero()).is_zero
    assert jordan_wigner(FermionOperator({((1, 0), (1, 0)): 1.0})).is_zero
    assert jordan_wigner(FermionOperator({((1, 1), (1, 1)): 1.0})).is_zero
    total = FermionOperator({((2, 1), (2, 0)): 1.0, ((2, 0), (2, 1)): 1.0})
    assert jordan_wigner(total) == QubitOperator.identity()


# =============================================================================
# Bravyi-Kitaev
# =============================================================================


def test_bk_docstring_example_and_columns() -> None:
    """Documented example and the β-matrix column sets for 4 and 8 modes."""
    assert len(bravyi_kitaev(FermionOperator.number_operator(0))) == 2
    assert bravyi_kitaev(FermionOperator.number_operator(0)) == QubitOperator(
        {(): 0.5, ((0, "Z"),): -0.5}
    )
    assert _build_bk_columns(1) == [{0}]
    # Standard BK β_4 = [[1,0,0,0],[1,1,0,0],[0,0,1,0],[1,1,1,1]].
    assert _build_bk_columns(4) == [{0, 1, 3}, {1, 3}, {2, 3}, {3}]
    assert _build_bk_columns(3) == [{0, 1}, {1}, {2}]
    cols8 = _build_bk_columns(8)
    assert cols8[0] == {0, 1, 3, 7} and cols8[4] == {4, 5, 7} and cols8[7] == {7}


def test_bk_identity_and_zero() -> None:
    """Identity and zero map to themselves."""
    assert bravyi_kitaev(FermionOperator.identity(1.5)) == QubitOperator.identity(1.5)
    assert bravyi_kitaev(FermionOperator.zero()).is_zero


@pytest.mark.parametrize("n", [1, 2])
def test_bk_canonical_anticommutation_small(n: int) -> None:
    """BK images obey the CAR for n <= 2 modes."""
    assert _car_violations(_ladders(_bk, n)) == []


@pytest.mark.parametrize("n", [3, 4])
def test_bk_canonical_anticommutation(n: int) -> None:
    """BK images obey {a_i, a_j†} = δ_ij I and {a_i, a_j} = 0."""
    assert _car_violations(_ladders(_bk, n)) == []


def test_bk_minimal_anticommutator_repro() -> None:
    """Minimal repro: BK({a_0, a_2}) must be the zero operator."""
    op = FermionOperator({((0, 0), (2, 0)): 1.0, ((2, 0), (0, 0)): 1.0})
    assert bravyi_kitaev(op).is_zero


# =============================================================================
# Properties shared by all transforms
# =============================================================================

TRANSFORMS = [
    pytest.param(_jw, id="jw"),
    pytest.param(_bk, id="bk"),
]


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_total_number_operator_spectrum(transform: Transform, n: int) -> None:
    """Σ n_i has eigenvalues 0..n with binomial degeneracies."""
    total = FermionOperator.zero()
    for i in range(n):
        total = total + FermionOperator.number_operator(i)
    mat = _qmat(transform(total, n), n)
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-14)
    eig = np.rint(np.linalg.eigvalsh(mat)).astype(int)
    np.testing.assert_allclose(np.linalg.eigvalsh(mat), np.sort(eig), atol=1e-12)
    values, counts = np.unique(eig, return_counts=True)
    assert values.tolist() == list(range(n + 1))
    assert counts.tolist() == [comb(n, k) for k in range(n + 1)]


def _check_homomorphism(transform: Transform, n: int, seed: int) -> None:
    """transform(AB) = transform(A)transform(B), sums and adjoints map over."""
    rng = np.random.default_rng(seed)
    a = _random_fermion_op(rng, n, 4)
    b = _random_fermion_op(rng, n, 4)
    ta, tb = transform(a, n), transform(b, n)
    ma, mb = _qmat(ta, n), _qmat(tb, n)
    np.testing.assert_allclose(
        _qmat(transform(_fprod(a, b), n), n), ma @ mb, atol=1e-12
    )
    np.testing.assert_allclose(_qmat(_operator_mul(ta, tb), n), ma @ mb, atol=1e-12)
    assert transform(a + b, n) == ta + tb
    assert transform(2.5j * a, n) == 2.5j * ta
    np.testing.assert_allclose(
        _qmat(transform(_fdagger(a), n), n), ma.conj().T, atol=1e-12
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_jw_is_algebra_homomorphism(n: int, seed: int) -> None:
    """Jordan-Wigner preserves products, sums, scalars and adjoints."""
    _check_homomorphism(_jw, n, seed)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_bk_is_free_algebra_homomorphism(n: int, seed: int) -> None:
    """BK maps term products to operator products."""
    _check_homomorphism(_bk, n, seed)


def _spectrum(op: QubitOperator, n: int) -> np.ndarray:
    """Sorted eigenvalues of a Hermitian QubitOperator."""
    mat = _qmat(op, n)
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)
    return np.asarray(np.linalg.eigvalsh(mat))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [1, 2])
def test_jw_bk_spectra_agree_small(n: int, seed: int) -> None:
    """A random Hermitian Hamiltonian has the same spectrum under JW and BK."""
    rng = np.random.default_rng(seed)
    h = _random_hermitian_hamiltonian(rng, n)
    np.testing.assert_allclose(
        _spectrum(_jw(h, n), n), _spectrum(_bk(h, n), n), atol=1e-10
    )


@pytest.mark.parametrize("n", [3, 4])
def test_jw_bk_spectra_agree(n: int) -> None:
    """Same Hamiltonian, same spectrum under JW and BK for n >= 3."""
    rng = np.random.default_rng(11)
    h = _random_hermitian_hamiltonian(rng, n)
    np.testing.assert_allclose(
        _spectrum(_jw(h, n), n), _spectrum(_bk(h, n), n), atol=1e-10
    )


@pytest.mark.parametrize("seed", [3, 4])
def test_jw_spectrum_matches_fock_reference(seed: int) -> None:
    """JW spectrum equals the spectrum of the numpy Fock-space Hamiltonian."""
    rng = np.random.default_rng(seed)
    n = 4
    h = _random_hermitian_hamiltonian(rng, n)
    ref = np.zeros((2**n, 2**n), dtype=complex)
    for term, coeff in h:
        mat = np.eye(2**n, dtype=complex)
        for mode, action in term:
            mat = mat @ _np_jw_ladder(mode, action, n)
        ref += complex(coeff) * mat  # type: ignore[arg-type]
    np.testing.assert_allclose(
        _spectrum(jordan_wigner(h), n), np.linalg.eigvalsh(ref), atol=1e-10
    )


# =============================================================================
# Error paths
# =============================================================================


@pytest.mark.parametrize("transform", [jordan_wigner, bravyi_kitaev])
def test_symbolic_coefficients_rejected(
    transform: Callable[[FermionOperator], QubitOperator],
) -> None:
    """Symbolic coefficients must be bound before transforming."""
    t = Parameter("t")
    with pytest.raises(ValueError, match="bind_parameters"):
        transform(t * FermionOperator.number_operator(0))
    with pytest.raises(ValueError, match="bind_parameters"):
        transform(FermionOperator.identity(t))
    bound = (t * FermionOperator.number_operator(0)).bind_parameters({t: 2.0})
    assert transform(bound) == QubitOperator({(): 1.0, ((0, "Z"),): -1.0})


def test_reverse_jordan_wigner_not_implemented() -> None:
    """reverse_jordan_wigner is a documented placeholder."""
    with pytest.raises(NotImplementedError):
        reverse_jordan_wigner(QubitOperator({((0, "Z"),): 1.0}))
