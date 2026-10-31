"""Tests for QubitOperator."""

from __future__ import annotations

import pytest

from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.operators.qubit_operator import (
    QubitOperator,
    _format_term,
    _normalize_term,
)


class TestNormalizeTerm:
    """Tests for internal term normalization."""

    def test_sorts_by_qubit_index(self) -> None:
        term: tuple[tuple[int, str], ...] = ((2, "X"), (0, "Z"), (1, "Y"))
        result = _normalize_term(term)
        assert result == ((0, "Z"), (1, "Y"), (2, "X"))

    def test_sorts_by_pauli_when_same_qubit(self) -> None:
        # X < Y < Z
        term: tuple[tuple[int, str], ...] = ((0, "Z"), (0, "X"), (0, "Y"))
        result = _normalize_term(term)
        assert result == ((0, "X"), (0, "Y"), (0, "Z"))

    def test_empty_term(self) -> None:
        assert _normalize_term(()) == ()


class TestFormatTerm:
    """Tests for Pauli term formatting."""

    def test_single_qubit(self) -> None:
        assert _format_term(((0, "X"),)) == "X0"

    def test_multi_qubit(self) -> None:
        assert _format_term(((0, "X"), (1, "Z"))) == "X0 Z1"

    def test_identity(self) -> None:
        assert _format_term(()) == "I"


class TestQubitOperatorConstruction:
    """Tests for constructing QubitOperator instances."""

    def test_empty_construction(self) -> None:
        op = QubitOperator()
        assert op.is_zero
        assert len(op) == 0

    def test_single_term(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        assert len(op) == 1
        assert not op.is_zero

    def test_zero_coefficient_removed(self) -> None:
        op = QubitOperator({((0, "X"),): 0.0})
        assert op.is_zero

    def test_identity_term(self) -> None:
        op = QubitOperator({(): 1.0})
        assert op.is_identity
        assert len(op) == 1

    def test_strips_identity_pauli(self) -> None:
        # (0, "I") should be stripped
        op = QubitOperator({((0, "I"), (1, "X")): 1.0})
        assert len(op) == 1
        terms = dict(op)
        assert ((1, "X"),) in terms

    def test_strips_all_identity_paulis(self) -> None:
        op = QubitOperator({((0, "I"), (1, "I")): 1.0})
        assert op.is_identity
        assert () in dict(op)

    def test_merges_duplicate_terms(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 2.0})
        op = a + b
        assert len(op) == 1
        assert dict(op)[((0, "X"),)] == pytest.approx(3.0)

    def test_merges_to_zero(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): -1.0})
        op = a + b
        assert op.is_zero

    def test_lowercase_pauli_accepted(self) -> None:
        op = QubitOperator({((0, "x"),): 1.0})
        assert dict(op)[((0, "X"),)] == pytest.approx(1.0)


class TestQubitOperatorConstructors:
    """Tests for classmethod constructors."""

    def test_zero(self) -> None:
        op = QubitOperator.zero()
        assert op.is_zero
        assert len(op) == 0

    def test_identity_default(self) -> None:
        op = QubitOperator.identity()
        assert op.is_identity
        assert dict(op)[()] == pytest.approx(1.0)

    def test_identity_with_coeff(self) -> None:
        op = QubitOperator.identity(2.5)
        assert op.is_identity
        assert dict(op)[()] == pytest.approx(2.5)

    def test_from_pauli_string_simple(self) -> None:
        op = QubitOperator.from_pauli_string("XZ", coefficient=0.5)
        assert len(op) == 1
        assert dict(op)[((0, "X"), (1, "Z"))] == pytest.approx(0.5)

    def test_from_pauli_string_with_identity(self) -> None:
        op = QubitOperator.from_pauli_string("IX")
        assert len(op) == 1
        assert dict(op)[((1, "X"),)] == pytest.approx(1.0)

    def test_from_pauli_string_all_identity(self) -> None:
        op = QubitOperator.from_pauli_string("II")
        assert op.is_identity

    def test_from_pauli_string_custom_qubits(self) -> None:
        op = QubitOperator.from_pauli_string("XY", qubits=(2, 3))
        assert len(op) == 1
        assert dict(op)[((2, "X"), (3, "Y"))] == pytest.approx(1.0)

    def test_from_pauli_string_invalid_char(self) -> None:
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            QubitOperator.from_pauli_string("XW")

    def test_from_pauli_string_wrong_qubits_length(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            QubitOperator.from_pauli_string("XYZ", qubits=(0, 1))


class TestQubitOperatorQuery:
    """Tests for query methods."""

    def test_is_zero_empty(self) -> None:
        assert QubitOperator().is_zero

    def test_is_zero_non_empty(self) -> None:
        assert not QubitOperator({((0, "X"),): 1.0}).is_zero

    def test_is_identity_empty_tuple(self) -> None:
        assert QubitOperator({(): 1.0}).is_identity

    def test_is_identity_zero(self) -> None:
        assert QubitOperator.zero().is_identity

    def test_is_identity_multiple_terms(self) -> None:
        assert not QubitOperator({((0, "X"),): 1.0, (): 2.0}).is_identity

    def test_qubit_count(self) -> None:
        op = QubitOperator({((0, "X"), (2, "Z")): 1.0})
        assert op.qubit_count() == 3

    def test_qubit_count_empty(self) -> None:
        assert QubitOperator().qubit_count() == 0

    def test_parameters_empty(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        assert op.parameters == set()

    def test_parameters_with_param(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): t})
        assert op.parameters == {t}

    def test_iteration(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0, ((1, "Z"),): 2.0})
        items = list(op)
        assert len(items) == 2

    def test_terms_property(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        t = op.terms
        assert isinstance(t, dict)
        # Mutating the copy should not affect original
        t[((1, "Z"),)] = 3.0
        assert len(op) == 1


class TestQubitOperatorArithmetic:
    """Tests for arithmetic operations."""

    def test_add_operators(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((1, "Z"),): 2.0})
        result = a + b
        assert len(result) == 2
        assert dict(result)[((0, "X"),)] == pytest.approx(1.0)
        assert dict(result)[((1, "Z"),)] == pytest.approx(2.0)

    def test_add_merges_same_term(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 2.0})
        result = a + b
        assert len(result) == 1
        assert dict(result)[((0, "X"),)] == pytest.approx(3.0)

    def test_add_scalar(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = op + 3.0
        assert len(result) == 2
        assert dict(result)[()] == pytest.approx(3.0)

    def test_add_zero_scalar(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = op + 0
        assert result == op

    def test_radd(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = 2.0 + op
        assert dict(result)[()] == pytest.approx(2.0)

    def test_sub_operators(self) -> None:
        a = QubitOperator({((0, "X"),): 3.0})
        b = QubitOperator({((0, "X"),): 1.0})
        result = a - b
        assert dict(result)[((0, "X"),)] == pytest.approx(2.0)

    def test_sub_scalar(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = op - 2.0
        assert dict(result)[()] == pytest.approx(-2.0)

    def test_rsub(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = 3.0 - op
        assert dict(result)[()] == pytest.approx(3.0)
        assert dict(result)[((0, "X"),)] == pytest.approx(-1.0)

    def test_mul_scalar(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0, ((1, "Z"),): 2.0})
        result = op * 3.0
        assert dict(result)[((0, "X"),)] == pytest.approx(3.0)
        assert dict(result)[((1, "Z"),)] == pytest.approx(6.0)

    def test_rmul_scalar(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = 3.0 * op
        assert dict(result)[((0, "X"),)] == pytest.approx(3.0)

    def test_mul_parameter(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): 1.0})
        result = t * op
        assert dict(result)[((0, "X"),)] == t

    def test_mul_zero(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = op * 0
        assert result.is_zero

    def test_truediv(self) -> None:
        op = QubitOperator({((0, "X"),): 2.0})
        result = op / 2.0
        assert dict(result)[((0, "X"),)] == pytest.approx(1.0)

    def test_truediv_zero(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        with pytest.raises(ZeroDivisionError):
            _ = op / 0

    def test_neg(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = -op
        assert dict(result)[((0, "X"),)] == pytest.approx(-1.0)

    def test_pos(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        result = +op
        assert result == op

    def test_iadd(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        a += QubitOperator({((1, "Z"),): 2.0})
        assert len(a) == 2

    def test_complex_coefficient_arithmetic(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0 + 1j})
        result = op * 2
        assert isinstance(dict(result)[((0, "X"),)], complex)
        assert dict(result)[((0, "X"),)] == pytest.approx(2.0 + 2j)

    def test_complex_division(self) -> None:
        op = QubitOperator({((0, "X"),): 2.0 + 2j})
        result = op / 2.0
        assert dict(result)[((0, "X"),)] == pytest.approx(1.0 + 1j)


class TestQubitOperatorSimplifyCompress:
    """Tests for simplify and compress methods."""

    def test_simplify_merges(self) -> None:
        # Construction already merges, but test explicitly
        op = QubitOperator({((0, "X"),): 1.0})
        simplified = op.simplify()
        assert len(simplified) == 1

    def test_compress_removes_small(self) -> None:
        op = QubitOperator(
            {
                ((0, "X"),): 1e-15,
                ((1, "Z"),): 1.0,
            }
        )
        result = op.compress()
        assert len(result) == 1
        assert dict(result)[((1, "Z"),)] == pytest.approx(1.0)

    def test_compress_keeps_large(self) -> None:
        op = QubitOperator({((0, "X"),): 1e-10})
        result = op.compress(tol=1e-15)
        assert len(result) == 1

    def test_compress_custom_tol(self) -> None:
        op = QubitOperator(
            {
                ((0, "X"),): 0.1,
                ((1, "Z"),): 0.01,
                ((2, "Y"),): 0.001,
            }
        )
        result = op.compress(tol=0.05)
        assert len(result) == 1

    def test_compress_keeps_parameters(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): t})
        result = op.compress(tol=1.0)
        assert len(result) == 1  # Parameters never removed


class TestQubitOperatorBindParameters:
    """Tests for parameter binding."""

    def test_bind_single_parameter(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): t})
        bound = op.bind_parameters({t: 0.5})
        assert dict(bound)[((0, "X"),)] == pytest.approx(0.5)

    def test_bind_multiple_parameters(self) -> None:
        t = Parameter("t")
        p = Parameter("p")
        op = QubitOperator(
            {
                ((0, "X"),): t,
                ((1, "Z"),): p,
            }
        )
        bound = op.bind_parameters({t: 0.5, p: 1.0})
        assert dict(bound)[((0, "X"),)] == pytest.approx(0.5)
        assert dict(bound)[((1, "Z"),)] == pytest.approx(1.0)

    def test_bind_unbound_parameter_raises(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): t})
        with pytest.raises(ValueError, match="not bound"):
            op.bind_parameters({})

    def test_bind_removes_zero(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): t})
        bound = op.bind_parameters({t: 0.0})
        assert bound.is_zero

    def test_bind_with_numeric_terms(self) -> None:
        t = Parameter("t")
        op = QubitOperator(
            {
                ((0, "X"),): t,
                ((1, "Z"),): 2.0,
            }
        )
        bound = op.bind_parameters({t: 0.5})
        assert dict(bound)[((1, "Z"),)] == pytest.approx(2.0)


class TestQubitOperatorEquality:
    """Tests for equality and hashing."""

    def test_equal_same_terms(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 1.0})
        assert a == b

    def test_not_equal_different_terms(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "Z"),): 1.0})
        assert a != b

    def test_equal_different_order(self) -> None:
        a = QubitOperator({((0, "X"), (1, "Z")): 1.0})
        b = QubitOperator({((1, "Z"), (0, "X")): 1.0})
        assert a == b  # normalization sorts them

    def test_hash_consistency(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 1.0})
        assert hash(a) == hash(b)

    def test_equal_with_tolerance(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 1.0 + 1e-13})
        assert a == b

    def test_not_equal_different_coeff(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 2.0})
        assert a != b

    def test_equal_zero_operators(self) -> None:
        a = QubitOperator.zero()
        b = QubitOperator.zero()
        assert a == b

    def test_not_equal_to_non_operator(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        assert op != 42
        assert op != "hello"


class TestQubitOperatorRepresentation:
    """Tests for string representation."""

    def test_repr_zero(self) -> None:
        assert repr(QubitOperator.zero()) == "QubitOperator(0)"

    def test_repr_single_term(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        assert repr(op) in ("QubitOperator(1.0 X0)", "QubitOperator(X0)")

    def test_repr_negative_coeff(self) -> None:
        op = QubitOperator({((0, "X"),): -1.0})
        assert "QubitOperator(-1.0 X0)" in repr(op) or "QubitOperator(-X0)" in repr(op)

    def test_repr_identity(self) -> None:
        op = QubitOperator.identity(1.0)
        assert "I" in repr(op)

    def test_repr_multiple_terms(self) -> None:
        op = QubitOperator(
            {
                ((0, "X"),): 1.0,
                ((1, "Z"),): 2.0,
            }
        )
        r = repr(op)
        assert "X0" in r
        assert "Z1" in r

    def test_str(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        assert str(op) == repr(op)

    def test_latex_zero(self) -> None:
        op = QubitOperator.zero()
        assert op._repr_latex_() == "$0$"

    def test_latex_single_term(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0})
        latex = op._repr_latex_()
        assert "sigma" in latex or r"\s" in latex


class TestQubitOperatorComplex:
    """Tests with complex coefficients."""

    def test_complex_construction(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0 + 1j})
        assert isinstance(dict(op)[((0, "X"),)], complex)

    def test_complex_equality(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0 + 1j})
        b = QubitOperator({((0, "X"),): 1.0 + 1j})
        assert a == b

    def test_complex_hash(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0 + 1j})
        b = QubitOperator({((0, "X"),): 1.0 + 1j})
        assert hash(a) == hash(b)


class TestQubitOperatorEdgeCases:
    """Edge case tests."""

    def test_very_large_qubit_index(self) -> None:
        op = QubitOperator({((1000, "X"),): 1.0})
        assert op.qubit_count() == 1001

    def test_negative_qubit_index(self) -> None:
        # Negative indices are technically allowed but unusual
        op = QubitOperator({((-1, "X"),): 1.0})
        assert op.qubit_count() == 0  # max(-1) + 1 = 0

    def test_many_terms(self) -> None:
        terms = {((i, "X"),): float(i) for i in range(1, 101)}
        op = QubitOperator(terms)
        assert len(op) == 100

    def test_parameter_mul_chain(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"),): 1.0})
        result = 2.0 * (t * op)
        assert dict(result)[((0, "X"),)] == 2.0 * t


class TestQubitOperatorParameterExpressions:
    """Affine symbolic coefficients should behave like first-class values."""

    def test_scaled_parameter_tracks_and_binds(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        base = QubitOperator({((0, "X"),): 1.0})
        operator = (2 * theta - 0.5 * phi + 1) * base

        assert operator.parameters == {theta, phi}
        coefficient = operator.terms[((0, "X"),)]
        assert isinstance(coefficient, ParameterExpression)
        assert coefficient.terms == {theta: 2.0, phi: -0.5}

        bound = operator.bind_parameters({theta: 0.25, phi: 1.0})
        assert bound.parameters == set()
        assert bound.terms[((0, "X"),)] == pytest.approx(1.0)

    def test_affine_coefficients_merge_on_matching_term(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        term = ((0, "Z"),)
        left = theta * QubitOperator({term: 1.0})
        right = (2 * theta + phi + 0.25) * QubitOperator({term: 1.0})

        merged = left + right
        coefficient = merged.terms[term]
        assert isinstance(coefficient, ParameterExpression)
        assert coefficient == 3 * theta + phi + 0.25

    def test_symbolic_scalar_adds_identity_term(self) -> None:
        theta = Parameter("theta")
        operator = QubitOperator({((0, "X"),): 1.0}) + (2 * theta + 1)
        assert operator.parameters == {theta}
        assert operator.terms[()] == 2 * theta + 1

    def test_two_symbolic_coefficients_cannot_multiply(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        operator = theta * QubitOperator({((0, "X"),): 1.0})
        with pytest.raises(ValueError, match="not be affine"):
            _ = operator * phi

    def test_symbolic_coefficient_cannot_gain_imaginary_part(self) -> None:
        theta = Parameter("theta")
        operator = theta * QubitOperator({((0, "X"),): 1.0})
        with pytest.raises(ValueError, match="imaginary"):
            _ = operator * (1 + 1j)

    def test_constant_expression_scaling_and_division(self) -> None:
        two = ParameterExpression(constant=2)
        operator = QubitOperator({((0, "X"),): 4.0})
        assert (operator * two).terms[((0, "X"),)] == pytest.approx(8.0)
        assert (operator / two).terms[((0, "X"),)] == pytest.approx(2.0)

    def test_repr_latex_equality_and_hash_support_expression(self) -> None:
        theta = Parameter("theta")
        base = QubitOperator({((0, "X"),): 1.0})
        first = (2 * theta + 1) * base
        second = (2 * theta + 1) * base
        assert first == second
        assert hash(first) == hash(second)
        assert "2*theta + 1" in repr(first)
        assert "theta" in first._repr_latex_()


class TestQubitOperatorValidationRegressions:
    """Regression tests for Pauli-letter validation and repeated qubits."""

    @pytest.mark.parametrize(
        "term",
        [((0, "Q"),), ((0, "XX"),), (("0", "X"),), ((0,),)],
    )
    def test_invalid_terms_are_rejected(self, term: object) -> None:
        with pytest.raises(ValueError, match="Invalid Pauli term"):
            QubitOperator({term: 1.0})  # type: ignore[dict-item]

    def test_lowercase_and_identity_letters_are_accepted(self) -> None:
        op = QubitOperator({((0, "x"), (1, "i")): 1.0})
        assert op == QubitOperator({((0, "X"),): 1.0})

    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            (((0, "X"), (0, "Z")), {((0, "Y"),): -1j}),
            (((0, "Z"), (0, "X")), {((0, "Y"),): 1j}),
            (((0, "X"), (1, "Z"), (0, "Y")), {((0, "Z"), (1, "Z")): 1j}),
            (((2, "Y"), (2, "Y")), {(): 1.0}),
            (((0, "X"), (0, "Y"), (0, "Z")), {(): 1j}),
        ],
    )
    def test_repeated_qubits_are_reduced(
        self, term: tuple[tuple[int, str], ...], expected: dict[object, complex]
    ) -> None:
        assert QubitOperator({term: 1.0}) == QubitOperator(expected)  # type: ignore[arg-type]

    def test_repeated_qubit_with_real_phase_keeps_symbolic_coefficient(self) -> None:
        t = Parameter("t")
        op = QubitOperator({((0, "X"), (0, "X"), (1, "Z")): t})
        assert op == QubitOperator({((1, "Z"),): t})

    def test_latex_omits_unit_coefficients(self) -> None:
        op = QubitOperator({((0, "X"),): 1.0, ((1, "Z"),): -1.0})
        assert op._repr_latex_() == r"$\sigma_{0}^{X} - \sigma_{1}^{Z}$"
