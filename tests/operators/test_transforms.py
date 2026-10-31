"""Tests for fermion-to-qubit transforms."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.operators.fermion_operator import FermionOperator
from quspect.operators.qubit_operator import QubitOperator
from quspect.operators.transforms import (
    _build_bk_columns,
    _jw_ladder_operator,
    _operator_mul,
    bravyi_kitaev,
    jordan_wigner,
    reverse_jordan_wigner,
)


class TestJWTransformKnownResults:
    """Test Jordan-Wigner against analytically known results."""

    def test_number_operator_single_mode(self) -> None:
        """n_0 = a†₀ a₀ = (I - Z₀) / 2"""
        fop = FermionOperator.number_operator(0, coefficient=1.0)
        qop = jordan_wigner(fop)
        # Expected: 0.5 I - 0.5 Z₀
        assert len(qop) == 2
        assert dict(qop)[()] == pytest.approx(0.5)
        assert dict(qop)[((0, "Z"),)] == pytest.approx(-0.5)

    def test_number_operator_mode_1(self) -> None:
        """n_1 = a†₁ a₁ = (I - Z₁) / 2"""
        fop = FermionOperator.number_operator(1, coefficient=1.0)
        qop = jordan_wigner(fop)
        assert len(qop) == 2
        assert dict(qop)[()] == pytest.approx(0.5)
        assert dict(qop)[((1, "Z"),)] == pytest.approx(-0.5)

    def test_hopping_term(self) -> None:
        """a†₀ a₁ + a†₁ a₀ → (X₀ X₁ + Y₀ Y₁) / 2"""
        fop = FermionOperator.hopping_term(0, 1, coefficient=1.0)
        qop = jordan_wigner(fop)
        # Expected: 0.5 X₀ X₁ + 0.5 Y₀ Y₁
        assert len(qop) == 2
        terms = dict(qop)
        assert terms[((0, "X"), (1, "X"))] == pytest.approx(0.5)
        assert terms[((0, "Y"), (1, "Y"))] == pytest.approx(0.5)

    def test_identity_operator(self) -> None:
        """Identity should map to identity."""
        fop = FermionOperator.identity(coefficient=2.0)
        qop = jordan_wigner(fop)
        assert qop.is_identity
        assert dict(qop)[()] == pytest.approx(2.0)

    def test_zero_operator(self) -> None:
        """Zero should map to zero."""
        fop = FermionOperator.zero()
        qop = jordan_wigner(fop)
        assert qop.is_zero

    def test_density_density_interaction(self) -> None:
        """n₀ n₁ = (I - Z₀)(I - Z₁)/4 = (I - Z₀ - Z₁ + Z₀Z₁)/4"""
        # n₀ n₁ = a†₀ a₀ a†₁ a₁
        fop = FermionOperator(
            {
                ((0, 1), (0, 0), (1, 1), (1, 0)): 1.0,
            }
        )
        qop = jordan_wigner(fop)
        # Expected: 1/4 I - 1/4 Z₀ - 1/4 Z₁ + 1/4 Z₀ Z₁
        assert len(qop) == 4
        terms = dict(qop)
        assert terms[()] == pytest.approx(0.25)
        assert terms[((0, "Z"),)] == pytest.approx(-0.25)
        assert terms[((1, "Z"),)] == pytest.approx(-0.25)
        assert terms[((0, "Z"), (1, "Z"))] == pytest.approx(0.25)

    def test_chemical_potential_two_site(self) -> None:
        """-μ(n₀ + n₁) = -μ(I - (Z₀ + Z₁)/2)"""
        fop = FermionOperator.zero()
        fop -= FermionOperator.number_operator(0, coefficient=1.0)
        fop -= FermionOperator.number_operator(1, coefficient=1.0)
        qop = jordan_wigner(fop)
        # Expected: -1.0 I + 0.5 Z₀ + 0.5 Z₁
        terms = dict(qop)
        # n₀ + n₁ = I - 0.5*Z₀ - 0.5*Z₁, but we need to check...
        # n₀ = (I - Z₀)/2, n₁ = (I - Z₁)/2
        # n₀ + n₁ = I - (Z₀ + Z₁)/2
        # -(n₀ + n₁) = -I + (Z₀ + Z₁)/2
        assert terms[()] == pytest.approx(-1.0)
        assert terms[((0, "Z"),)] == pytest.approx(0.5)
        assert terms[((1, "Z"),)] == pytest.approx(0.5)


class TestJWTransformLargerSystems:
    """Test JW on systems with more modes."""

    def test_number_operator_mode_2(self) -> None:
        """n₂ = (I - Z₂)/2"""
        fop = FermionOperator.number_operator(2, coefficient=1.0)
        qop = jordan_wigner(fop)
        terms = dict(qop)
        assert terms[()] == pytest.approx(0.5)
        assert terms[((2, "Z"),)] == pytest.approx(-0.5)

    def test_hopping_0_2(self) -> None:
        """a†₀ a₂ + a†₂ a₀"""
        fop = FermionOperator.hopping_term(0, 2, coefficient=1.0)
        qop = jordan_wigner(fop)
        # Should have non-zero terms
        assert len(qop) >= 2
        # Each term should have a Z on qubit 1 (parity between 0 and 2)
        for term in dict(qop):
            # The parity Z₁ should be present
            assert (1, "Z") in term

    def test_coulomb_0_1_1_0(self) -> None:
        """a†₀ a†₁ a₁ a₀ with JW"""
        fop = FermionOperator.coulomb_term(0, 1, 1, 0, coefficient=1.0)
        qop = jordan_wigner(fop)
        # Should produce valid qubit operator (same as n₀ n₁)
        assert len(qop) > 0
        # n₀ n₁ = (I - Z₀)(I - Z₁)/4
        # n₀ n₁ = a†₀ a₀ a†₁ a₁  but our coulomb is a†₀ a†₁ a₁ a₀
        # These should be equivalent by fermion anti-commutation
        fop2 = FermionOperator(
            {
                ((0, 1), (0, 0), (1, 1), (1, 0)): 1.0,
            }
        )
        qop2 = jordan_wigner(fop2)
        # They should have the same number of terms
        assert len(qop) == len(qop2)


class TestOperatorMultiplication:
    """Tests for _operator_mul helper."""

    def test_mul_single_terms(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "X"),): 1.0})
        result = _operator_mul(a, b)
        # X₀ * X₀ = I
        assert result.is_identity
        assert dict(result)[()] == pytest.approx(1.0)

    def test_mul_different_qubits(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((1, "Z"),): 1.0})
        result = _operator_mul(a, b)
        # X₀ * Z₁ = X₀ Z₁
        assert len(result) == 1
        assert dict(result)[((0, "X"), (1, "Z"))] == pytest.approx(1.0)

    def test_mul_same_qubit_pauli(self) -> None:
        a = QubitOperator({((0, "X"),): 1.0})
        b = QubitOperator({((0, "Y"),): 1.0})
        result = _operator_mul(a, b)
        # X₀ * Y₀ = iZ₀
        assert len(result) == 1
        coeff = dict(result)[((0, "Z"),)]
        assert isinstance(coeff, complex)
        assert coeff == pytest.approx(1j)

    def test_mul_sums(self) -> None:
        # (X₀ + Y₀) * (X₁ + Y₁) = X₀X₁ + X₀Y₁ + Y₀X₁ + Y₀Y₁
        a = QubitOperator({((0, "X"),): 1.0, ((0, "Y"),): 1.0})
        b = QubitOperator({((1, "X"),): 1.0, ((1, "Y"),): 1.0})
        result = _operator_mul(a, b)
        assert len(result) == 4

    def test_mul_identity(self) -> None:
        a = QubitOperator({((0, "X"),): 2.0})
        i_op = QubitOperator.identity()
        result = _operator_mul(a, i_op)
        assert result == a


class TestJWLadderOperator:
    """Tests for _jw_ladder_operator helper."""

    def test_creation_mode_0(self) -> None:
        # a†₀ = (X₀ - iY₀)/2
        op = _jw_ladder_operator(0, 1, 1)
        assert len(op) == 2
        terms = dict(op)
        assert terms[((0, "X"),)] == pytest.approx(0.5)
        assert terms[((0, "Y"),)] == pytest.approx(-0.5j)

    def test_annihilation_mode_0(self) -> None:
        # a₀ = (X₀ + iY₀)/2
        op = _jw_ladder_operator(0, 0, 1)
        assert len(op) == 2
        terms = dict(op)
        assert terms[((0, "X"),)] == pytest.approx(0.5)
        assert terms[((0, "Y"),)] == pytest.approx(0.5j)

    def test_creation_mode_1(self) -> None:
        # a†₁ = (X₁ - iY₁)Z₀ / 2
        op = _jw_ladder_operator(1, 1, 2)
        assert len(op) == 2
        # Each term should include Z on qubit 0 (parity)
        for term in dict(op):
            assert (0, "Z") in term


class TestJWWithParameters:
    """Test JW transform with parameter interactions."""

    def test_parameter_coeff_raises(self) -> None:
        from quspect.circuit.parameter import Parameter

        t = Parameter("t")
        fop = t * FermionOperator.number_operator(0)
        with pytest.raises(ValueError, match="bind_parameters"):
            jordan_wigner(fop)

    def test_bound_then_transform(self) -> None:
        from quspect.circuit.parameter import Parameter

        t = Parameter("t")
        fop = t * FermionOperator.number_operator(0)
        bound = fop.bind_parameters({t: 1.0})
        qop = jordan_wigner(bound)
        assert dict(qop)[()] == pytest.approx(0.5)
        assert dict(qop)[((0, "Z"),)] == pytest.approx(-0.5)

    def test_affine_expression_requires_binding(self) -> None:
        from quspect.circuit.parameter import Parameter

        theta = Parameter("theta")
        operator = (2 * theta + 1) * FermionOperator.number_operator(0)
        with pytest.raises(ValueError, match="bind_parameters"):
            jordan_wigner(operator)
        with pytest.raises(ValueError, match="bind_parameters"):
            bravyi_kitaev(operator)

        bound = operator.bind_parameters({theta: 0.25})
        jw_result = jordan_wigner(bound)
        bk_result = bravyi_kitaev(bound)
        assert jw_result == bk_result
        assert jw_result.terms[()] == pytest.approx(0.75)
        assert jw_result.terms[((0, "Z"),)] == pytest.approx(-0.75)


class TestFutureTransforms:
    """Tests for placeholder transforms."""

    def test_bravyi_kitaev_single_site(self) -> None:
        fop = FermionOperator.number_operator(0)
        qop = bravyi_kitaev(fop)
        # n_0 = a†_0 a_0 → ½ I - ½ Z (same as JW for single site)
        assert dict(qop)[()] == pytest.approx(0.5)
        assert dict(qop)[((0, "Z"),)] == pytest.approx(-0.5)

    def test_reverse_jw_not_implemented(self) -> None:
        qop = QubitOperator({((0, "X"),): 1.0})
        with pytest.raises(NotImplementedError):
            reverse_jordan_wigner(qop)


# ---------------------------------------------------------------------------
# Bravyi-Kitaev (Seeley-Richard-Love) — numpy-only regression tests
# ---------------------------------------------------------------------------

_PAULI_MATS = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _dense(op: QubitOperator, n: int) -> np.ndarray:
    """Little-endian dense matrix (qubit 0 is the rightmost tensor factor)."""
    total = np.zeros((2**n, 2**n), dtype=complex)
    for term, coeff in op:
        letters = dict(term)
        mat = np.ones((1, 1), dtype=complex)
        for qubit in range(n - 1, -1, -1):
            factor = _PAULI_MATS.get(letters.get(qubit, ""), np.eye(2))
            mat = np.kron(mat, factor)
        total += complex(coeff) * mat  # type: ignore[arg-type]
    return total


def _ladder(mode: int, action: int) -> FermionOperator:
    return FermionOperator({((mode, action),): 1.0})


class TestBravyiKitaev:
    """Regression tests for the Seeley-Richard-Love BK transform."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_canonical_anticommutation(self, n: int) -> None:
        ops = [_dense(bravyi_kitaev(_ladder(j, 0), n_qubits=n), n) for j in range(n)]
        identity = np.eye(2**n)
        for i, a_i in enumerate(ops):
            for j, a_j in enumerate(ops):
                a_j_dag = a_j.conj().T
                np.testing.assert_allclose(
                    a_i @ a_j_dag + a_j_dag @ a_i, identity * (i == j), atol=1e-12
                )
                np.testing.assert_allclose(a_i @ a_j + a_j @ a_i, 0, atol=1e-12)

    def test_creation_is_adjoint_of_annihilation(self) -> None:
        for j in range(5):
            create = _dense(bravyi_kitaev(_ladder(j, 1), n_qubits=5), 5)
            annihilate = _dense(bravyi_kitaev(_ladder(j, 0), n_qubits=5), 5)
            np.testing.assert_allclose(create, annihilate.conj().T, atol=1e-14)

    def test_minimal_anticommutator_is_zero(self) -> None:
        op = FermionOperator({((0, 0), (2, 0)): 1.0, ((2, 0), (0, 0)): 1.0})
        assert bravyi_kitaev(op).is_zero

    def test_known_ladder_images_for_four_modes(self) -> None:
        # a_1 on 4 modes: U={3}, P={0}, F={0}, R={}.
        assert bravyi_kitaev(_ladder(1, 0), n_qubits=4) == QubitOperator(
            {((0, "Z"), (1, "X"), (3, "X")): 0.5, ((1, "Y"), (3, "X")): 0.5j}
        )
        # a_3 on 4 modes: U={}, P={1, 2}, F={1, 2}, R={}.
        assert bravyi_kitaev(_ladder(3, 0), n_qubits=4) == QubitOperator(
            {((1, "Z"), (2, "Z"), (3, "X")): 0.5, ((3, "Y"),): 0.5j}
        )

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_spectrum_matches_jordan_wigner(self, n: int) -> None:
        rng = np.random.default_rng(n)
        hamiltonian = FermionOperator.zero()
        for p in range(n):
            for q in range(p, n):
                value = float(rng.normal())
                if p == q:
                    hamiltonian += FermionOperator.number_operator(p, value)
                else:
                    hamiltonian += FermionOperator.hopping_term(p, q, value)
        hamiltonian += FermionOperator(
            {((0, 1), (n - 1, 1), (n - 1, 0), (0, 0)): float(rng.normal())}
        )
        jw = _dense(jordan_wigner(hamiltonian), n)
        bk = _dense(bravyi_kitaev(hamiltonian, n_qubits=n), n)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(jw), np.linalg.eigvalsh(bk), atol=1e-10
        )

    def test_n_qubits_aligns_encodings_for_addition(self) -> None:
        a0 = _ladder(0, 0)
        a2 = _ladder(2, 0)
        left = bravyi_kitaev(a0, n_qubits=3)
        right = bravyi_kitaev(a2, n_qubits=3)
        assert bravyi_kitaev(a0 + a2) == left + right
        # Default size follows the largest mode index.
        assert bravyi_kitaev(a0) == bravyi_kitaev(a0, n_qubits=1)
        assert bravyi_kitaev(a0) != left

    def test_n_qubits_validation(self) -> None:
        with pytest.raises(ValueError, match="n_qubits"):
            bravyi_kitaev(_ladder(3, 0), n_qubits=2)
        with pytest.raises(TypeError, match="n_qubits"):
            bravyi_kitaev(_ladder(0, 0), n_qubits=2.0)  # type: ignore[arg-type]
        assert bravyi_kitaev(FermionOperator.identity(2.0), n_qubits=0) == (
            QubitOperator.identity(2.0)
        )

    def test_column_sets_match_fenwick_tree(self) -> None:
        assert _build_bk_columns(4) == [{0, 1, 3}, {1, 3}, {2, 3}, {3}]
        assert _build_bk_columns(3) == [{0, 1}, {1}, {2}]
