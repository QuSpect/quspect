"""Comprehensive tests for the Parameter system.

Tests cover:
- Parameter creation and equality
- Hashing and set/dict usage
- Ordering / sorting
- Immutability
- ParamValue type union usage
- Edge cases: empty name, Unicode names, long names
"""

from __future__ import annotations

import pytest

from quspect.circuit.parameter import Parameter, ParameterExpression, ParamValue

# =============================================================================
# Parameter creation
# =============================================================================


class TestParameterCreation:
    """Parameter instances should be lightweight named placeholders."""

    def test_create_with_ascii_name(self) -> None:
        p = Parameter("theta")
        assert p.name == "theta"

    def test_create_with_unicode_name(self) -> None:
        p = Parameter("θ")
        assert p.name == "θ"

    def test_create_with_chinese_name(self) -> None:
        p = Parameter("参数")
        assert p.name == "参数"

    def test_create_with_empty_name(self) -> None:
        p = Parameter("")
        assert p.name == ""

    def test_create_with_long_name(self) -> None:
        long_name = "a" * 1000
        p = Parameter(long_name)
        assert p.name == long_name

    def test_create_with_numeric_like_name(self) -> None:
        p = Parameter("123")
        assert p.name == "123"

    def test_create_with_space_in_name(self) -> None:
        p = Parameter("my param")
        assert p.name == "my param"

    def test_create_with_special_characters(self) -> None:
        p = Parameter("θ_1")
        assert p.name == "θ_1"


# =============================================================================
# Equality and hashing
# =============================================================================


class TestParameterEquality:
    """Parameters with the same name must be equal."""

    def test_same_name_equal(self) -> None:
        assert Parameter("theta") == Parameter("theta")

    def test_different_name_not_equal(self) -> None:
        assert Parameter("theta") != Parameter("phi")

    def test_equality_is_case_sensitive(self) -> None:
        assert Parameter("Theta") != Parameter("theta")

    def test_equality_with_different_type(self) -> None:
        assert Parameter("theta") != "theta"
        assert Parameter("theta") != 42
        assert Parameter("theta") is not None


class TestParameterHashing:
    """Parameters must be hashable for use in dicts and sets."""

    def test_hash_consistent_with_equality(self) -> None:
        p1 = Parameter("theta")
        p2 = Parameter("theta")
        assert hash(p1) == hash(p2)

    def test_different_names_different_hash(self) -> None:
        # Not strictly required, but expected behavior
        p1 = Parameter("theta")
        p2 = Parameter("phi")
        assert hash(p1) != hash(p2)

    def test_parameter_in_set(self) -> None:
        s: set[Parameter] = {Parameter("a"), Parameter("b")}
        assert Parameter("a") in s
        assert Parameter("c") not in s

    def test_parameter_as_dict_key(self) -> None:
        d: dict[Parameter, float] = {
            Parameter("θ"): 0.5,
            Parameter("φ"): 1.2,
        }
        assert d[Parameter("θ")] == 0.5
        assert d[Parameter("φ")] == 1.2

    def test_large_set(self) -> None:
        """Many parameters should all be usable in a set."""
        params = {Parameter(f"p{i}") for i in range(1000)}
        assert len(params) == 1000
        # Adding duplicate should not change size
        params.add(Parameter("p500"))
        assert len(params) == 1000


# =============================================================================
# Ordering
# =============================================================================


class TestParameterOrdering:
    """Parameters should be orderable (by name) for sorting."""

    def test_sort_by_name(self) -> None:
        params = [Parameter("c"), Parameter("a"), Parameter("b")]
        sorted_params = sorted(params)
        assert [p.name for p in sorted_params] == ["a", "b", "c"]

    def test_less_than(self) -> None:
        assert Parameter("a") < Parameter("b")

    def test_greater_than(self) -> None:
        assert Parameter("b") > Parameter("a")

    def test_less_equal(self) -> None:
        assert Parameter("a") <= Parameter("a")
        assert Parameter("a") <= Parameter("b")

    def test_sort_stability_with_same_name(self) -> None:
        """All Parameter("x") instances are equal and sort together."""
        params = [Parameter("b"), Parameter("a"), Parameter("a")]
        sorted_params = sorted(params)
        assert sorted_params[0].name == "a"
        assert sorted_params[1].name == "a"
        assert sorted_params[2].name == "b"


# =============================================================================
# Immutability
# =============================================================================


class TestParameterImmutability:
    """Parameter is a frozen dataclass — modification should raise."""

    def test_cannot_set_name(self) -> None:
        p = Parameter("theta")
        with pytest.raises(Exception):  # noqa: B017
            p.name = "new_name"  # type: ignore[misc]


# =============================================================================
# Representation
# =============================================================================


class TestParameterRepr:
    """Parameter.__repr__ should be clear and eval-able."""

    def test_repr_includes_name(self) -> None:
        assert "theta" in repr(Parameter("theta"))

    def test_repr_format(self) -> None:
        assert repr(Parameter("θ")) == "Parameter('θ')"


# =============================================================================
# ParamValue type usage
# =============================================================================


class TestParamValue:
    """ParamValue = Union[float, Parameter] should accept both types."""

    def test_float_is_valid(self) -> None:
        v: ParamValue = 3.14
        assert isinstance(v, float)

    def test_parameter_is_valid(self) -> None:
        v: ParamValue = Parameter("theta")
        assert isinstance(v, Parameter)

    def test_int_is_valid_as_float(self) -> None:
        """int is a subtype of float for type checking purposes."""
        v: ParamValue = 42
        assert v == 42

    def test_function_accepts_paramvalue(self) -> None:
        """A function typed with ParamValue should accept both."""

        def set_param(value: ParamValue) -> str:
            if isinstance(value, Parameter):
                return f"param:{value.name}"
            return f"value:{value}"

        assert set_param(Parameter("θ")) == "param:θ"
        assert set_param(0.5) == "value:0.5"


# =============================================================================
# Edge cases
# =============================================================================


class TestParameterEdgeCases:
    """Unusual but valid usage patterns."""

    def test_parameter_with_unicode_and_numbers(self) -> None:
        p = Parameter("φ₂")
        assert p.name == "φ₂"

    def test_parameter_with_only_whitespace(self) -> None:
        p = Parameter("   ")
        assert p.name == "   "
        assert p == Parameter("   ")

    def test_thousands_of_parameters(self) -> None:
        """Creating many parameters should not cause issues."""
        params = [Parameter(f"p{i}") for i in range(10000)]
        assert len(params) == 10000
        # All should be unique
        assert len(set(params)) == 10000


# =============================================================================
# Affine expressions
# =============================================================================


class TestParameterExpression:
    """Parameter arithmetic should retain affine coefficients exactly."""

    def test_affine_arithmetic_and_evaluation(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        expr = 2 * theta - phi / 2 + 1.25

        assert isinstance(expr, ParameterExpression)
        assert expr.constant == 1.25
        assert expr.terms == {theta: 2.0, phi: -0.5}
        assert expr.parameters == frozenset({theta, phi})
        assert expr.evaluate({theta: 0.5, phi: 1.0}) == pytest.approx(1.75)

    def test_parameter_times_two_keeps_coefficient(self) -> None:
        theta = Parameter("theta")
        expr = 2 * theta
        assert isinstance(expr, ParameterExpression)
        assert expr.coefficient(theta) == 2.0

    def test_parameter_times_one_keeps_compatible_parameter_form(self) -> None:
        theta = Parameter("theta")
        assert 1 * theta is theta
        assert theta * 1 is theta

    def test_terms_are_combined_and_zero_terms_removed(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        expr = 2 * theta - theta + phi - phi
        assert expr.terms == {theta: 1.0}
        assert expr.parameters == frozenset({theta})

    def test_expression_equality_hash_and_canonical_order(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        lhs = theta + 2 * phi + 3
        rhs = ParameterExpression({phi: 2, theta: 1}, constant=3)
        assert lhs == rhs
        assert hash(lhs) == hash(rhs)
        assert {lhs, rhs} == {lhs}

    def test_expression_is_immutable(self) -> None:
        theta = Parameter("theta")
        expr = 2 * theta
        with pytest.raises(Exception):  # noqa: B017
            expr.constant = 4.0  # type: ignore[misc]
        with pytest.raises(TypeError):
            expr.terms[theta] = 3.0  # type: ignore[index]

    def test_missing_binding_raises(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        with pytest.raises(ValueError, match="phi"):
            (theta + phi).evaluate({theta: 1.0})

    def test_multiplication_of_symbolic_expressions_is_rejected(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        with pytest.raises(TypeError, match="non-constant"):
            _ = (theta + 1) * (phi - 1)

    def test_division_by_symbolic_expression_is_rejected(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        with pytest.raises(TypeError, match="non-constant"):
            _ = theta / (phi + 1)

    def test_constant_expression_may_scale_expression(self) -> None:
        theta = Parameter("theta")
        two = ParameterExpression(constant=2)
        assert two * theta == 2 * theta
        assert theta / two == 0.5 * theta

    def test_complex_coefficients_are_rejected(self) -> None:
        theta = Parameter("theta")
        with pytest.raises(TypeError, match="real"):
            _ = (1 + 2j) * theta

    def test_expression_is_part_of_param_value(self) -> None:
        value: ParamValue = 2 * Parameter("theta") + 1
        assert isinstance(value, ParameterExpression)

    def test_public_package_export(self) -> None:
        from quspect import ParameterExpression as PublicParameterExpression

        assert PublicParameterExpression is ParameterExpression
