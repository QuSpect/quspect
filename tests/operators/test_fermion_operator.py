"""Tests for FermionOperator."""

from __future__ import annotations

import random

import pytest

from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.operators.fermion_operator import FermionOperator


class TestFermionOperatorConstruction:
    """Tests for constructing FermionOperator instances."""

    def test_empty_construction(self) -> None:
        op = FermionOperator()
        assert op.is_zero
        assert len(op) == 0

    def test_single_term(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        assert len(op) == 1
        assert not op.is_zero

    def test_zero_coefficient_removed(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 0.0})
        assert op.is_zero

    def test_identity_term(self) -> None:
        op = FermionOperator({(): 1.0})
        assert op.is_identity

    def test_merges_duplicate_terms(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 1.0})
        b = FermionOperator({((0, 1), (0, 0)): 2.0})
        op = a + b
        assert len(op) == 1
        assert dict(op)[((0, 1), (0, 0))] == pytest.approx(3.0)

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError, match="Action must be"):
            FermionOperator({((0, 2),): 1.0})

    def test_invalid_term_type_raises(self) -> None:
        with pytest.raises(TypeError, match="tuple"):
            FermionOperator({"bad": 1.0})  # type: ignore[dict-item]

    def test_invalid_mode_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FermionOperator({((-1, 1),): 1.0})

    def test_non_tuple_element_raises(self) -> None:
        with pytest.raises(ValueError, match="mode, action"):
            FermionOperator({(0,): 1.0})  # type: ignore[dict-item]


class TestFermionOperatorConstructors:
    """Tests for classmethod constructors."""

    def test_zero(self) -> None:
        op = FermionOperator.zero()
        assert op.is_zero

    def test_identity(self) -> None:
        op = FermionOperator.identity()
        assert op.is_identity
        assert dict(op)[()] == pytest.approx(1.0)

    def test_identity_with_coeff(self) -> None:
        op = FermionOperator.identity(2.5)
        assert dict(op)[()] == pytest.approx(2.5)

    def test_number_operator(self) -> None:
        op = FermionOperator.number_operator(0)
        assert len(op) == 1
        assert dict(op)[((0, 1), (0, 0))] == pytest.approx(1.0)

    def test_number_operator_with_coeff(self) -> None:
        op = FermionOperator.number_operator(2, coefficient=0.5)
        assert dict(op)[((2, 1), (2, 0))] == pytest.approx(0.5)

    def test_number_operator_negative_mode(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FermionOperator.number_operator(-1)

    def test_hopping_term(self) -> None:
        op = FermionOperator.hopping_term(0, 1, coefficient=0.5)
        assert len(op) == 2
        assert dict(op)[((0, 1), (1, 0))] == pytest.approx(0.5)
        assert dict(op)[((1, 1), (0, 0))] == pytest.approx(0.5)

    def test_hopping_term_negative_mode(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FermionOperator.hopping_term(-1, 0)

    def test_coulomb_term(self) -> None:
        op = FermionOperator.coulomb_term(0, 1, 1, 0, coefficient=2.0)
        assert len(op) == 1
        assert dict(op)[((0, 1), (1, 1), (1, 0), (0, 0))] == pytest.approx(2.0)

    def test_coulomb_term_negative_mode(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FermionOperator.coulomb_term(-1, 0, 1, 0)


class TestFermionOperatorQuery:
    """Tests for query methods."""

    def test_is_zero(self) -> None:
        assert FermionOperator().is_zero

    def test_is_identity(self) -> None:
        assert FermionOperator({(): 1.0}).is_identity

    def test_mode_count(self) -> None:
        op = FermionOperator({((0, 1), (2, 0)): 1.0})
        assert op.mode_count() == 3

    def test_mode_count_empty(self) -> None:
        assert FermionOperator().mode_count() == 0

    def test_parameters(self) -> None:
        t = Parameter("t")
        op = FermionOperator({((0, 1), (0, 0)): t})
        assert op.parameters == {t}

    def test_iteration(self) -> None:
        op = FermionOperator(
            {
                ((0, 1), (0, 0)): 1.0,
                ((1, 1), (1, 0)): 2.0,
            }
        )
        items = list(op)
        assert len(items) == 2


class TestFermionOperatorArithmetic:
    """Tests for arithmetic operations."""

    def test_add_operators(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 1.0})
        b = FermionOperator({((1, 1), (1, 0)): 2.0})
        result = a + b
        assert len(result) == 2

    def test_add_merges_same_term(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 1.0})
        b = FermionOperator({((0, 1), (0, 0)): 2.0})
        result = a + b
        assert len(result) == 1
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(3.0)

    def test_sub_operators(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 3.0})
        b = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = a - b
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(2.0)

    def test_mul_scalar(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = op * 3.0
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(3.0)

    def test_rmul_scalar(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = 3.0 * op
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(3.0)

    def test_mul_parameter(self) -> None:
        t = Parameter("t")
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = t * op
        assert dict(result)[((0, 1), (0, 0))] == t

    def test_neg(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = -op
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(-1.0)

    def test_truediv(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 2.0})
        result = op / 2.0
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(1.0)

    def test_truediv_zero_raises(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        with pytest.raises(ZeroDivisionError):
            _ = op / 0

    def test_add_scalar(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = op + 2.0
        assert dict(result)[()] == pytest.approx(2.0)


class TestFermionOperatorSimplifyCompress:
    """Tests for simplify and compress."""

    def test_simplify(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        simplified = op.simplify()
        assert simplified == op

    def test_compress_removes_small(self) -> None:
        op = FermionOperator(
            {
                ((0, 1), (0, 0)): 1e-15,
                ((1, 1), (1, 0)): 1.0,
            }
        )
        result = op.compress()
        assert len(result) == 1

    def test_compress_keeps_parameters(self) -> None:
        t = Parameter("t")
        op = FermionOperator({((0, 1), (0, 0)): t})
        result = op.compress(tol=1.0)
        assert len(result) == 1


class TestFermionOperatorBindParameters:
    """Tests for parameter binding."""

    def test_bind_single_parameter(self) -> None:
        t = Parameter("t")
        op = FermionOperator({((0, 1), (0, 0)): t})
        bound = op.bind_parameters({t: 0.5})
        assert dict(bound)[((0, 1), (0, 0))] == pytest.approx(0.5)

    def test_bind_unbound_raises(self) -> None:
        t = Parameter("t")
        op = FermionOperator({((0, 1), (0, 0)): t})
        with pytest.raises(ValueError, match="not bound"):
            op.bind_parameters({})


class TestFermionOperatorEquality:
    """Tests for equality and hashing."""

    def test_equal_same_terms(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 1.0})
        b = FermionOperator({((0, 1), (0, 0)): 1.0})
        assert a == b

    def test_not_equal_different_terms(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 1.0})
        b = FermionOperator({((1, 1), (1, 0)): 1.0})
        assert a != b

    def test_hash_consistency(self) -> None:
        a = FermionOperator({((0, 1), (0, 0)): 1.0})
        b = FermionOperator({((0, 1), (0, 0)): 1.0})
        assert hash(a) == hash(b)

    def test_not_equal_to_non_operator(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        assert op != 42


class TestFermionOperatorNormalOrder:
    """Tests for normal ordering."""

    def test_identity_normal_order(self) -> None:
        op = FermionOperator.identity()
        result = op.normal_order()
        assert result == op

    def test_already_normal_ordered(self) -> None:
        # a†₀ a₀ is already normal ordered (creation before annihilation)
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        result = op.normal_order()
        assert result == op

    def test_simple_swap(self) -> None:
        # a₀ a†₁ (different modes) → -a†₁ a₀ + δ_{01}*I
        # Since 0 != 1, delta is 0
        op = FermionOperator({((0, 0), (1, 1)): 1.0})
        result = op.normal_order()
        # Should be: -a†₁ a₀
        assert len(result) == 1
        assert dict(result)[((1, 1), (0, 0))] == pytest.approx(-1.0)

    def test_same_mode_swap_with_delta(self) -> None:
        # a₀ a†₀ = I - a†₀ a₀
        op = FermionOperator({((0, 0), (0, 1)): 1.0})
        result = op.normal_order()
        # Should have two terms: I and -a†₀ a₀
        assert len(result) == 2
        assert dict(result)[()] == pytest.approx(1.0)  # Identity delta term
        assert dict(result)[((0, 1), (0, 0))] == pytest.approx(-1.0)

    def test_zero_normal_order(self) -> None:
        op = FermionOperator.zero()
        result = op.normal_order()
        assert result.is_zero

    def test_already_normal_ordered_multi_mode(self) -> None:
        # a†₀ a†₁ a₁ a₀ is already normal ordered
        op = FermionOperator({((0, 1), (1, 1), (1, 0), (0, 0)): 1.0})
        result = op.normal_order()
        assert result == op


class TestFermionOperatorParameterExpressions:
    """Affine symbolic coefficients remain usable through fermionic algebra."""

    def test_scaled_parameter_tracks_and_binds(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        operator = (2 * theta - phi + 1) * FermionOperator.number_operator(0)

        assert operator.parameters == {theta, phi}
        coefficient = operator.terms[((0, 1), (0, 0))]
        assert isinstance(coefficient, ParameterExpression)
        bound = operator.bind_parameters({theta: 0.5, phi: 0.25})
        assert bound.terms[((0, 1), (0, 0))] == pytest.approx(1.75)

    def test_matching_symbolic_terms_merge(self) -> None:
        theta = Parameter("theta")
        term = ((0, 1), (0, 0))
        merged = theta * FermionOperator({term: 1.0}) + (theta + 0.5) * FermionOperator(
            {term: 1.0}
        )
        assert merged.terms[term] == 2 * theta + 0.5

    def test_normal_order_preserves_expression(self) -> None:
        theta = Parameter("theta")
        operator = (2 * theta + 1) * FermionOperator({((0, 0), (1, 1)): 1.0})
        ordered = operator.normal_order()
        coefficient = ordered.terms[((1, 1), (0, 0))]
        assert coefficient == -2 * theta - 1
        assert ordered.parameters == {theta}

    def test_two_symbolic_coefficients_cannot_multiply(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        operator = theta * FermionOperator.number_operator(0)
        with pytest.raises(ValueError, match="not be affine"):
            _ = operator * phi

    def test_constant_expression_scaling_and_division(self) -> None:
        two = ParameterExpression(constant=2)
        operator = FermionOperator.number_operator(0, coefficient=4.0)
        term = ((0, 1), (0, 0))
        assert (operator * two).terms[term] == pytest.approx(8.0)
        assert (operator / two).terms[term] == pytest.approx(2.0)

    def test_repr_latex_equality_and_hash_support_expression(self) -> None:
        theta = Parameter("theta")
        first = (2 * theta + 1) * FermionOperator.number_operator(0)
        second = (2 * theta + 1) * FermionOperator.number_operator(0)
        assert first == second
        assert hash(first) == hash(second)
        assert "2*theta + 1" in repr(first)
        assert "theta" in first._repr_latex_()


class TestFermionOperatorRepresentation:
    """Tests for string representation."""

    def test_repr_zero(self) -> None:
        assert repr(FermionOperator.zero()) == "FermionOperator(0)"

    def test_repr_single_term(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        r = repr(op)
        assert "FermionOperator" in r
        assert "a†" in r or "a" in r

    def test_repr_identity(self) -> None:
        op = FermionOperator.identity()
        assert "I" in repr(op)

    def test_latex(self) -> None:
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        latex = op._repr_latex_()
        assert latex.startswith("$")
        assert latex.endswith("$")


class TestFermionOperatorRegressions:
    """Regression tests for hopping_term, normal_order and LaTeX output."""

    def test_hopping_term_same_mode_is_twice_number_operator(self) -> None:
        op = FermionOperator.hopping_term(1, 1, 0.5)
        assert op == FermionOperator.number_operator(1, 1.0)

    def test_hopping_term_same_mode_symbolic(self) -> None:
        t = Parameter("t")
        op = FermionOperator.hopping_term(0, 0, t)
        assert op.bind_parameters({t: 0.25}) == FermionOperator.number_operator(0, 0.5)

    def test_normal_order_delta_uses_sign_updated_coefficient(self) -> None:
        # a0 a†1 a†0 = -a†1 a0 a†0 = -a†1 + a†1 a†0 a0
        op = FermionOperator({((0, 0), (1, 1), (0, 1)): 1.0})
        expected = FermionOperator({((1, 1),): -1.0, ((1, 1), (0, 1), (0, 0)): 1.0})
        assert op.normal_order() == expected

    def test_normal_order_preserves_jordan_wigner_image(self) -> None:
        from quspect.operators.transforms import jordan_wigner

        rng = random.Random(7)
        for _ in range(30):
            length = rng.randint(1, 5)
            term = tuple((rng.randrange(3), rng.randrange(2)) for _ in range(length))
            op = FermionOperator({term: rng.uniform(-1.0, 1.0)})
            difference = jordan_wigner(op) - jordan_wigner(op.normal_order())
            assert difference.compress(tol=1e-12).is_zero, term

    def test_latex_omits_unit_coefficients(self) -> None:
        op = FermionOperator({((0, 1), (1, 0)): -1.0, (): 1.0})
        assert op._repr_latex_() == r"$\mathbb{I} - a^{\dagger}_{0}\, a_{1}$"
        assert FermionOperator({((0, 0),): 1})._repr_latex_() == r"$a_{0}$"
