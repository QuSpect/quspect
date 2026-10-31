"""Cross-validate :class:`quspect.operators.QubitOperator` against qiskit.

Conventions pinned here (see ``test_term_key_maps_to_reversed_qiskit_label``):

* A term key ``((q, P), ...)`` means Pauli ``P`` on qubit ``q``.  In a qiskit
  Pauli label the *rightmost* character is qubit 0, so the key maps to the
  label ``s`` with ``s[n - 1 - q] = P``.
* ``QubitOperator.from_pauli_string(s)`` puts ``s[i]`` on qubit ``i`` (leftmost
  character = qubit 0), i.e. it equals ``SparsePauliOp(s[::-1])``.
* ``QubitOperator`` has **no** matrix export, operator product, adjoint or
  commutator.  Matrices are therefore built here through ``SparsePauliOp``
  (little-endian, same as quspect statevectors); operator products use the
  Pauli-algebra routine ``quspect.operators.transforms._operator_mul`` (the
  only product implementation in the package); adjoint / commutator are
  composed from public arithmetic and checked against qiskit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

import quspect
import quspect.operators as ops_pkg
from qiskit_tests.helpers import random_circuit_pair
from quspect import Parameter, QubitOperator
from quspect.circuit.parameter import ParameterExpression
from quspect.operators.qubit_operator import (
    _format_coeff,
    _format_term,
    _is_pauli_term,
)
from quspect.operators.transforms import _operator_mul
from quspect.variational import AdjointEngine, ObservableObjective

PAULIS = ("X", "Y", "Z")


# =============================================================================
# Local helpers
# =============================================================================


def _to_spo(op: QubitOperator, num_qubits: int) -> SparsePauliOp:
    """Convert a numeric ``QubitOperator`` to a qiskit ``SparsePauliOp``."""
    labels: list[str] = []
    coeffs: list[complex] = []
    for term, coeff in op:
        chars = ["I"] * num_qubits
        for qubit, pauli in term:
            chars[num_qubits - 1 - qubit] = pauli
        labels.append("".join(chars))
        coeffs.append(complex(coeff))  # type: ignore[arg-type]
    if not labels:
        return SparsePauliOp("I" * num_qubits, [0.0])
    return SparsePauliOp(labels, np.array(coeffs, dtype=complex))


def _from_spo(spo: SparsePauliOp) -> QubitOperator:
    """Convert a qiskit ``SparsePauliOp`` to a ``QubitOperator``."""
    terms: dict[tuple[tuple[int, str], ...], Any] = {}
    n = spo.num_qubits
    for label, coeff in spo.to_list():
        key = tuple((q, label[n - 1 - q]) for q in range(n) if label[n - 1 - q] != "I")
        terms[key] = terms.get(key, 0) + complex(coeff)
    return QubitOperator(terms)


def _matrix(op: QubitOperator, num_qubits: int) -> np.ndarray:
    """Little-endian dense matrix of a numeric ``QubitOperator``."""
    return np.asarray(_to_spo(op, num_qubits).to_matrix())


def _random_qubit_op(
    rng: np.random.Generator,
    num_qubits: int,
    num_terms: int,
    *,
    hermitian: bool = False,
) -> QubitOperator:
    """Random operator with (possibly repeated) Pauli strings."""
    terms: list[tuple[tuple[tuple[int, str], ...], complex | float]] = []
    for _ in range(num_terms):
        key = tuple(
            (q, PAULIS[int(rng.integers(3))])
            for q in range(num_qubits)
            if rng.random() < 0.7
        )
        if hermitian:
            coeff: complex | float = float(rng.normal())
        else:
            coeff = complex(rng.normal(), rng.normal())
        terms.append((key, coeff))
    op = QubitOperator.zero()
    for key, coeff in terms:
        op = op + QubitOperator({key: coeff})
    return op


def _dagger(op: QubitOperator) -> QubitOperator:
    """Hermitian conjugate built from the public term mapping."""
    return QubitOperator(
        {term: complex(coeff).conjugate() for term, coeff in op.terms.items()}  # type: ignore[arg-type]
    )


# =============================================================================
# Conventions
# =============================================================================


@pytest.mark.qiskit_ref
def test_term_key_maps_to_reversed_qiskit_label() -> None:
    """Key ((q, P),) is qiskit label with P at position n-1-q (qubit 0 rightmost)."""
    op = QubitOperator({((0, "X"), (1, "Z"), (3, "Y")): 1.0})
    assert _to_spo(op, 4) == SparsePauliOp("YIZX")
    expected = Pauli("YIZX").to_matrix()
    np.testing.assert_allclose(_matrix(op, 4), expected, atol=1e-14)
    # Explicit little-endian kron check: qubit 0 is the rightmost factor.
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    eye = np.eye(2)
    kron = np.kron(np.kron(np.kron(y, eye), z), x)
    np.testing.assert_allclose(_matrix(op, 4), kron, atol=1e-14)


@pytest.mark.qiskit_ref
def test_from_pauli_string_is_reversed_qiskit_label() -> None:
    """from_pauli_string(s) places s[i] on qubit i, i.e. SparsePauliOp(s[::-1])."""
    for label in ("XZ", "IXY", "ZZIY", "y", "xIz"):
        op = QubitOperator.from_pauli_string(label, 0.25)
        spo = SparsePauliOp(label.upper()[::-1], [0.25])
        np.testing.assert_allclose(_matrix(op, len(label)), spo.to_matrix(), atol=1e-14)


@pytest.mark.qiskit_ref
def test_statevector_expectation_uses_same_endianness() -> None:
    """<psi|Z0|psi> for |01> (qubit 0 = 1) is -1 in both libraries."""
    circ = quspect.QuantumCircuit().x(0).id(1)
    sv = np.asarray(quspect.Simulator().statevector(circ))
    z0 = QubitOperator({((0, "Z"),): 1.0})
    value = Statevector(sv).expectation_value(_to_spo(z0, 2))
    assert value == pytest.approx(-1.0)


# =============================================================================
# Construction / normalization
# =============================================================================


def test_package_exports() -> None:
    """quspect.operators re-exports both operator classes."""
    assert ops_pkg.__all__ == ["QubitOperator", "FermionOperator"]
    assert ops_pkg.QubitOperator is QubitOperator
    assert quspect.QubitOperator is QubitOperator


def test_constructor_normalizes_terms() -> None:
    """Terms are sorted, identity letters dropped, lowercase upcased, zeros removed."""
    op = QubitOperator(
        {
            ((2, "z"), (0, "X"), (1, "I")): 1.5,
            ((1, "Y"),): 0.0,
            ((0, "I"),): 2.0,
        }
    )
    assert op.terms == {((0, "X"), (2, "Z")): 1.5, (): 2.0}
    assert QubitOperator().is_zero
    assert QubitOperator(None) == QubitOperator.zero()


def test_constructor_merges_and_cancels_duplicate_keys() -> None:
    """Keys equal after normalization are merged; cancellation removes them."""
    op = QubitOperator({((1, "Z"), (0, "X")): 1.0, ((0, "X"), (1, "Z")): 2.0})
    assert op.terms == {((0, "X"), (1, "Z")): 3.0}
    gone = QubitOperator({((1, "Z"), (0, "X")): 1.0, ((0, "X"), (1, "Z")): -1.0})
    assert gone.is_zero


def test_terms_property_is_a_copy() -> None:
    """Mutating the returned terms dict does not affect the operator."""
    op = QubitOperator({((0, "X"),): 1.0})
    terms = op.terms
    terms[((1, "Z"),)] = 5.0
    assert len(op) == 1


def test_zero_identity_and_queries() -> None:
    """zero/identity constructors and is_zero/is_identity/qubit_count/len."""
    zero = QubitOperator.zero()
    assert zero.is_zero and zero.is_identity and zero.qubit_count() == 0
    ident = QubitOperator.identity(2.5)
    assert ident.is_identity and not ident.is_zero
    assert ident.terms == {(): 2.5}
    assert QubitOperator.identity().terms == {(): 1.0}
    op = QubitOperator({((4, "X"),): 1.0, (): 1.0})
    assert not op.is_identity
    assert op.qubit_count() == 5
    assert len(op) == 2
    assert dict(iter(op)) == op.terms


def test_from_pauli_string_with_qubits_argument() -> None:
    """Explicit qubit labels are honoured and identities dropped."""
    op = QubitOperator.from_pauli_string("XIZ", 2.0, qubits=(5, 1, 3))
    assert op.terms == {((3, "Z"), (5, "X")): 2.0}
    assert QubitOperator.from_pauli_string("III").terms == {(): 1.0}


def test_from_pauli_string_errors() -> None:
    """Invalid letters or mismatched qubit tuples raise ValueError."""
    with pytest.raises(ValueError, match="Invalid Pauli character"):
        QubitOperator.from_pauli_string("XA")
    with pytest.raises(ValueError, match="must match"):
        QubitOperator.from_pauli_string("XY", qubits=(0,))


def test_is_pauli_term_helper() -> None:
    """The private validator accepts only tuples of (int, I/X/Y/Z) pairs."""
    assert _is_pauli_term(((0, "X"), (1, "i")))
    assert _is_pauli_term(())
    assert not _is_pauli_term([(0, "X")])  # type: ignore[arg-type]
    assert not _is_pauli_term(((0, "X", 1),))  # type: ignore[arg-type]
    assert not _is_pauli_term((("0", "X"),))  # type: ignore[arg-type]
    assert not _is_pauli_term(((0, "Q"),))


def test_constructor_rejects_invalid_pauli_letter() -> None:
    """A term with a non-Pauli letter must be rejected."""
    with pytest.raises(ValueError):
        QubitOperator({((0, "Q"),): 1.0})


def test_constructor_repeated_qubit_is_rejected_or_reduced() -> None:
    """A repeated qubit must raise ValueError or be reduced to the Pauli product."""
    try:
        op = QubitOperator({((0, "X"), (0, "Z")): 1.0})
    except ValueError:
        return
    assert op == QubitOperator({((0, "Y"),): -1j})


# =============================================================================
# Linear algebra vs SparsePauliOp
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_add_sub_scale_divide_match_qiskit(seed: int) -> None:
    """+, -, unary -, scalar *, / agree with SparsePauliOp matrices."""
    rng = np.random.default_rng(seed)
    n = 3
    a = _random_qubit_op(rng, n, 6)
    b = _random_qubit_op(rng, n, 5)
    sa, sb = _to_spo(a, n), _to_spo(b, n)
    c = complex(rng.normal(), rng.normal())
    r = float(rng.normal())
    cases = [
        (a + b, sa + sb),
        (a - b, sa - sb),
        (-a, -sa),
        (+a, sa),
        (c * a, c * sa),
        (a * c, c * sa),
        (r * a, r * sa),
        (a / c, sa / c),
        (a / r, sa / r),
        (a / 3, sa / 3),
    ]
    for ours, ref in cases:
        np.testing.assert_allclose(_matrix(ours, n), ref.to_matrix(), atol=1e-12)
        assert ours == _from_spo(ref.simplify(atol=0))


@pytest.mark.qiskit_ref
def test_scalar_add_sub_acts_on_identity() -> None:
    """op + s, s + op, op - s and s - op add/subtract s * I."""
    op = QubitOperator({((0, "X"),): 1.0, (): 0.5})
    spo = _to_spo(op, 1)
    eye = SparsePauliOp("I")
    for ours, ref in [
        (op + 2.0, spo + 2.0 * eye),
        (2.0 + op, spo + 2.0 * eye),
        (op - 1j, spo - 1j * eye),
        (3 - op, 3 * eye - spo),
    ]:
        np.testing.assert_allclose(_matrix(ours, 1), ref.to_matrix(), atol=1e-14)
    assert op + 0 is op
    assert op - 0 is op
    assert (op - 0.5).terms == {((0, "X"),): 1.0}


def test_inplace_operators_return_new_instances() -> None:
    """+=, -=, *= do not mutate the original (immutable semantics)."""
    op = QubitOperator({((0, "X"),): 1.0})
    alias = op
    op += QubitOperator({((1, "Z"),): 1.0})
    op -= 0.5
    op *= 2
    assert alias.terms == {((0, "X"),): 1.0}
    assert op.terms == {((0, "X"),): 2.0, ((1, "Z"),): 2.0, (): -1.0}


def test_multiplication_by_zero_and_division_errors() -> None:
    """0 * op is zero; division by zero raises ZeroDivisionError."""
    op = QubitOperator({((0, "X"),): 1.0})
    assert (0 * op).is_zero
    assert (op * 0j).is_zero
    with pytest.raises(ZeroDivisionError):
        op / 0
    with pytest.raises(ZeroDivisionError):
        op / 0j


def test_unsupported_operand_types() -> None:
    """Non-numeric operands yield TypeError via NotImplemented."""
    op = QubitOperator({((0, "X"),): 1.0})
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
    # Direct dunder calls exercise the remaining branches.
    assert op.__rsub__("x") is NotImplemented  # type: ignore[arg-type]
    other = QubitOperator({((1, "Z"),): 1.0})
    assert op.__rsub__(other) == other - op
    # Operator-operator product is not supported by QubitOperator itself.
    with pytest.raises(TypeError):
        op * other  # type: ignore[operator]
    assert (op == 1.0) is False


# =============================================================================
# Operator product / adjoint / commutator / power (vs qiskit)
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_operator_product_matches_qiskit_compose(seed: int) -> None:
    """_operator_mul(a, b) equals matrix A @ B == SparsePauliOp a.dot(b)."""
    rng = np.random.default_rng(seed)
    n = 3
    a = _random_qubit_op(rng, n, 5)
    b = _random_qubit_op(rng, n, 4)
    sa, sb = _to_spo(a, n), _to_spo(b, n)
    prod = _operator_mul(a, b)
    # qiskit: a.dot(b) == a @ b == b.compose(a) is the matrix product A·B.
    ref = sa.dot(sb).simplify(atol=1e-14)
    np.testing.assert_allclose((sa @ sb).to_matrix(), ref.to_matrix(), atol=1e-12)
    np.testing.assert_allclose(sb.compose(sa).to_matrix(), ref.to_matrix(), atol=1e-12)
    np.testing.assert_allclose(
        _matrix(prod, n), _matrix(a, n) @ _matrix(b, n), atol=1e-12
    )
    assert prod == _from_spo(ref)


@pytest.mark.qiskit_ref
def test_single_qubit_pauli_multiplication_table() -> None:
    """All 16 single-qubit Pauli products agree with qiskit Pauli.dot."""
    for p in "IXYZ":
        for q in "IXYZ":
            a = QubitOperator.from_pauli_string(p)
            b = QubitOperator.from_pauli_string(q)
            ref = SparsePauliOp(Pauli(p).dot(Pauli(q)))
            np.testing.assert_allclose(
                _matrix(_operator_mul(a, b), 1), ref.to_matrix(), atol=1e-14
            )


@pytest.mark.qiskit_ref
def test_operator_product_cancellation_and_real_cast() -> None:
    """Cancelling products vanish and real results are stored as float."""
    x0 = QubitOperator({((0, "X"),): 1.0})
    y0 = QubitOperator({((0, "Y"),): 1.0})
    comm = _operator_mul(x0, y0) - _operator_mul(y0, x0)
    assert comm == QubitOperator({((0, "Z"),): 2j})
    anti = _operator_mul(x0 + y0, x0 - y0) + _operator_mul(x0 - y0, x0 + y0)
    assert anti.is_zero
    assert _operator_mul(x0 + y0, x0 + y0) == QubitOperator({(): 2.0})
    squared = _operator_mul(x0, x0)
    assert isinstance(squared.terms[()], float)
    # (X + iY)(X + iY) = 0: every product cancels.
    sigma = x0 + 1j * y0
    assert _operator_mul(sigma, sigma).is_zero
    # Products with negligible magnitude (< 1e-15) are dropped.
    tiny = QubitOperator({((0, "X"),): 1e-16})
    assert _operator_mul(tiny, x0).is_zero


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", [20, 21])
def test_adjoint_and_commutator_match_qiskit(seed: int) -> None:
    """Adjoint (conjugated coefficients) and [A, B] agree with qiskit."""
    rng = np.random.default_rng(seed)
    n = 3
    a = _random_qubit_op(rng, n, 5)
    b = _random_qubit_op(rng, n, 5)
    sa, sb = _to_spo(a, n), _to_spo(b, n)
    np.testing.assert_allclose(
        _matrix(_dagger(a), n), sa.adjoint().to_matrix(), atol=1e-12
    )
    comm = _operator_mul(a, b) - _operator_mul(b, a)
    ref = (sa.dot(sb) - sb.dot(sa)).simplify(atol=1e-12)
    np.testing.assert_allclose(_matrix(comm, n), ref.to_matrix(), atol=1e-12)
    # Hermitian part is Hermitian.
    herm = (a + _dagger(a)) / 2
    mat = _matrix(herm, n)
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


@pytest.mark.qiskit_ref
def test_operator_power_matches_qiskit() -> None:
    """Repeated _operator_mul equals SparsePauliOp.power."""
    rng = np.random.default_rng(30)
    a = _random_qubit_op(rng, 2, 4)
    power = QubitOperator.identity()
    for _ in range(3):
        power = _operator_mul(power, a)
    ref = _to_spo(a, 2).power(3)
    np.testing.assert_allclose(_matrix(power, 2), ref.to_matrix(), atol=1e-11)


# =============================================================================
# simplify / compress / equality / hash
# =============================================================================


@pytest.mark.qiskit_ref
def test_simplify_and_compress_match_qiskit() -> None:
    """simplify keeps the matrix; compress drops |c| < tol like SparsePauliOp."""
    op = QubitOperator(
        {
            ((0, "X"),): 1.0,
            ((1, "Z"),): 1e-14,
            ((0, "Y"),): 1e-14j,
            ((2, "Y"),): 0.3j,
        }
    )
    assert op.simplify() == op
    compressed = op.compress()
    assert compressed.terms == {((0, "X"),): 1.0, ((2, "Y"),): 0.3j}
    ref = _to_spo(op, 3).simplify(atol=1e-12)
    np.testing.assert_allclose(_matrix(compressed, 3), ref.to_matrix(), atol=1e-14)
    assert op.compress(tol=0.5).terms == {((0, "X"),): 1.0}
    t = Parameter("t")
    symbolic = QubitOperator({((0, "X"),): t, ((1, "Z"),): 1e-20})
    assert symbolic.compress().terms == {((0, "X"),): t}


def test_equality_semantics() -> None:
    """Equality uses tolerances and compares mixed numeric types."""
    a = QubitOperator({((0, "X"),): 1.0})
    assert a == QubitOperator({((0, "X"),): 1})
    assert a == QubitOperator({((0, "X"),): 1 + 0j})
    assert a == QubitOperator({((0, "X"),): 1.0 + 1e-14})
    assert a != QubitOperator({((0, "X"),): 1.0 + 1e-9})
    assert a != QubitOperator({((0, "Y"),): 1.0})
    assert a != QubitOperator({((0, "X"),): 1.0, ((1, "Z"),): 1.0})
    assert a != QubitOperator({((0, "X"),): 1.0 + 1e-3j})
    t = Parameter("t")
    assert QubitOperator({((0, "X"),): t}) == QubitOperator({((0, "X"),): t})
    assert QubitOperator({((0, "X"),): t}) != a


def test_hash_consistent_with_equality() -> None:
    """Equal operators hash equally and can be used as set members."""
    a = QubitOperator({((0, "X"), (1, "Z")): 0.5, (): 1j})
    b = QubitOperator({((1, "Z"), (0, "X")): 0.5, (): 1j})
    assert a == b and hash(a) == hash(b)
    assert len({a, b, QubitOperator.zero()}) == 2
    t = Parameter("t")
    assert hash(QubitOperator({((0, "X"),): t})) == hash(
        QubitOperator({((0, "X"),): t})
    )


# =============================================================================
# Symbolic coefficients
# =============================================================================


def test_symbolic_coefficients_algebra() -> None:
    """Parameters combine affinely; products of two symbols are rejected."""
    t, s = Parameter("t"), Parameter("s")
    x0 = QubitOperator({((0, "X"),): 1.0})
    assert (t * x0).terms == {((0, "X"),): t}
    assert (x0 * t).parameters == {t}
    combined = t * x0 + s * x0 + 2 * x0
    assert combined.parameters == {t, s}
    assert (t * x0 - t * x0).is_zero
    assert (x0 + t).terms[()] == t
    assert (x0 - t).parameters == {t}
    assert (t - x0).parameters == {t}
    # ParameterExpression constants are canonicalized to plain numbers.
    assert (x0 * ParameterExpression(constant=0.0)).is_zero
    assert (x0 / ParameterExpression(constant=2.0)) == 0.5 * x0
    assert QubitOperator({(): ParameterExpression(constant=3.0)}).terms == {(): 3.0}
    with pytest.raises(ValueError, match="symbolic"):
        (t * x0) * s
    with pytest.raises(ValueError, match="symbolic"):
        x0 / t
    with pytest.raises(ValueError, match="symbolic"):
        x0 / (t + 1)
    with pytest.raises(ValueError, match="imaginary"):
        t * x0 + 1j * x0
    # Complex division of symbolic coefficient by a real complex is fine.
    assert ((2 * t * x0) / (2 + 0j)).terms == {((0, "X"),): t}
    with pytest.raises(ValueError, match="imaginary"):
        (t * x0) / 1j
    # Number times an already-symbolic coefficient (symbol on the right).
    assert ((t * x0) * 2).terms == {((0, "X"),): 2 * t}
    # Real coefficients divided by a complex scalar become complex.
    assert (x0 / 2j).terms == {((0, "X"),): -0.5j}
    # Real-valued complex mixes with symbols.
    assert ((t * x0) + (1 + 0j) * x0).parameters == {t}


@pytest.mark.qiskit_ref
def test_bind_parameters_matches_numeric_operator() -> None:
    """bind_parameters evaluates Parameters and expressions, then matches qiskit."""
    t, s = Parameter("t"), Parameter("s")
    op = (
        t * QubitOperator({((0, "X"),): 1.0})
        + (2 * s + 1) * QubitOperator({((1, "Z"),): 1.0})
        + QubitOperator({((0, "Y"), (1, "Y")): 0.25j})
        + (t - t) * QubitOperator({((1, "X"),): 1.0})
    )
    bound = op.bind_parameters({t: 0.3, s: -0.5})
    assert bound.parameters == set()
    ref = SparsePauliOp(["IX", "ZI", "YY"], [0.3, 0.0, 0.25j]).simplify()
    np.testing.assert_allclose(_matrix(bound, 2), ref.to_matrix(), atol=1e-14)
    zeroed = (t * QubitOperator({((0, "X"),): 1.0})).bind_parameters({t: 0.0})
    assert zeroed.is_zero
    with pytest.raises(ValueError, match="not bound"):
        op.bind_parameters({t: 1.0})
    with pytest.raises(ValueError, match="not bound"):
        (t * QubitOperator({((0, "X"),): 1.0})).bind_parameters({})
    numeric = QubitOperator({((0, "Z"),): 0.5, ((1, "X"),): 1j})
    assert numeric.bind_parameters({}) == numeric


# =============================================================================
# Representation
# =============================================================================


def test_repr_and_str() -> None:
    """repr/str formats coefficients and terms deterministically."""
    t = Parameter("t")
    assert repr(QubitOperator.zero()) == "QubitOperator(0)"
    op = QubitOperator({((1, "Z"), (0, "X")): -1.0, (): 2.5, ((3, "Y"),): 1})
    assert repr(op) == "QubitOperator(2.5 I + -X0 Z1 + Y3)"
    assert str(op) == repr(op)
    assert repr(QubitOperator({((0, "X"),): 0.5 + 1j})) == (
        "QubitOperator(0.5+1.0j X0)"
    )
    assert repr(QubitOperator({((0, "X"),): t})) == ("QubitOperator(Parameter('t') X0)")
    assert repr(QubitOperator({((0, "X"),): 2 * t})) == "QubitOperator(2*t X0)"
    assert repr(QubitOperator({((0, "X"),): t + 1})) == "QubitOperator((t + 1) X0)"


@pytest.mark.parametrize(
    ("coeff", "text"),
    [
        (1, ""),
        (-1, "-"),
        (3, "3"),
        (1.0, ""),
        (-1.0, "-"),
        (2.0, "2"),
        (0.5, "0.5"),
        (1 + 0j, ""),
        (-1 + 0j, "-"),
        (4 + 0j, "4"),
        (0.25 + 0j, "0.25"),
        (2j, "2.0j"),
        (-2j, "-2.0j"),
        (1 - 2j, "1.0-2.0j"),
        (0j, "0"),
        ("weird", "weird"),
    ],
)
def test_format_coeff(coeff: Any, text: str) -> None:
    """_format_coeff renders each numeric branch."""
    assert _format_coeff(coeff) == text


def test_format_term_and_latex() -> None:
    """_format_term and _repr_latex_ render terms and coefficients."""
    assert _format_term(()) == "I"
    assert _format_term(((0, "x"), (2, "Z"))) == "X0 Z2"
    assert QubitOperator.zero()._repr_latex_() == "$0$"
    t = Parameter("t")
    assert QubitOperator({((0, "X"),): t})._repr_latex_() == (
        r"$\mathrm{t}\sigma_{0}^{X}$"
    )
    assert QubitOperator({((0, "X"),): t + 1})._repr_latex_() == (
        r"$\mathrm{t + 1}\sigma_{0}^{X}$"
    )
    latex = QubitOperator({((0, "X"), (1, "Z")): -0.5, (): 1j})._repr_latex_()
    assert latex == r"$(0.0+1.0i)\mathbb{I} - 0.5\sigma_{0}^{X} \otimes \sigma_{1}^{Z}$"
    assert QubitOperator({((0, "X"),): 2 + 0j})._repr_latex_() == (
        r"$2.0\sigma_{0}^{X}$"
    )


def test_latex_omits_unit_coefficients() -> None:
    """Unit coefficients +-1 should not be printed in LaTeX output."""
    op = QubitOperator({((0, "X"),): 1.0, ((1, "Z"),): -1.0})
    assert op._repr_latex_() == r"$\sigma_{0}^{X} - \sigma_{1}^{Z}$"


# =============================================================================
# Expectation values (via the variational adjoint engine)
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", [40, 41, 42])
def test_expectation_value_matches_qiskit(seed: int) -> None:
    """AdjointEngine <psi|H|psi> for random H equals qiskit expectation_value."""
    rng = np.random.default_rng(seed)
    n = 3
    circ, qc = random_circuit_pair(n, 12, seed)
    theta = Parameter("theta")
    circ.ry(theta, 1)
    angle = float(rng.uniform(-np.pi, np.pi))
    qc.ry(angle, 1)
    ham = _random_qubit_op(rng, n, 8, hermitian=True) + 0.3
    objective = ObservableObjective(circ, {"h": ham})
    value = AdjointEngine().compile(objective).evaluate({theta: angle}).values[0]
    ref = Statevector(qc).expectation_value(_to_spo(ham, n))
    assert abs(value - ref.real) < 1e-10
    assert abs(ref.imag) < 1e-12
    # Same number from the quspect statevector and the explicit matrix.
    sv = np.asarray(
        quspect.Simulator().statevector(circ.bind_parameters({theta: angle}))
    )
    direct = np.vdot(sv, _matrix(ham, n) @ sv)
    assert abs(direct - ref) < 1e-10
