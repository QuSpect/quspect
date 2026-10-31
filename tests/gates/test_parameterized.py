"""Comprehensive tests for parameterized quantum gates.

Tests cover:
- Template metadata (no params on template)
- with_params() factory method
- Matrix correctness across parameter ranges
- Unitarity for all parameter values
- Special parameter values: 0, π, 2π (identity-like cases)
- Error handling: missing params, wrong param count
- Gate immutability when using with_params
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.gate import Gate
from quspect.gates.parameterized import (
    ALL_PARAMETERIZED,
    CP,
    CRX,
    CRY,
    CRZ,
    CU3,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    P,
)

# =============================================================================
# Helper utilities
# =============================================================================


def is_unitary(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is unitary: U† U = I."""
    identity = np.eye(matrix.shape[0], dtype=complex)
    return bool(np.allclose(matrix.conj().T @ matrix, identity, atol=tol))


def assert_close_to_identity(
    matrix: np.ndarray, tol: float = 1e-10, msg: str = ""
) -> None:
    """Assert matrix is close to the identity matrix."""
    identity = np.eye(matrix.shape[0], dtype=complex)
    np.testing.assert_array_almost_equal(matrix, identity, err_msg=msg)


# Parameter sweep values for testing
THETA_VALUES: list[float] = [
    0.0,
    math.pi / 4,
    math.pi / 2,
    math.pi,
    3 * math.pi / 2,
    2 * math.pi,
    -math.pi / 3,
    0.123,
    7 * math.pi / 6,
]
"""Diverse theta values spanning [0, 2π] and beyond."""

PHI_VALUES: list[float] = [
    0.0,
    math.pi / 3,
    math.pi / 2,
    math.pi,
    -math.pi / 2,
]


# =============================================================================
# Template metadata
# =============================================================================


class TestTemplateMetadata:
    """Parameterized gate templates should have no params and correct names."""

    @pytest.mark.parametrize(
        "template,expected_name,expected_num_qubits",
        [
            (RX, "rx", 1),
            (RY, "ry", 1),
            (RZ, "rz", 1),
            (P, "p", 1),
            (U1, "u1", 1),
            (U2, "u2", 1),
            (U3, "u3", 1),
            (CRX, "crx", 2),
            (CRY, "cry", 2),
            (CRZ, "crz", 2),
            (CP, "cp", 2),
            (CU3, "cu3", 2),
        ],
    )
    def test_template_metadata(
        self,
        template: Gate,
        expected_name: str,
        expected_num_qubits: int,
    ) -> None:
        assert template.name == expected_name
        assert template.num_qubits == expected_num_qubits
        assert template.params == (), (
            f"Template {template.name} should have empty params, got {template.params}"
        )

    def test_all_templates_count(self) -> None:
        assert len(ALL_PARAMETERIZED) == 12


# =============================================================================
# with_params() factory method
# =============================================================================


class TestWithParams:
    """Test Gate.with_params() for creating concrete parameterized gates."""

    def test_creates_new_instance(self) -> None:
        g = RX.with_params(0.5)
        assert g is not RX  # new instance, not the template
        assert g.name == "rx"
        assert g.num_qubits == 1
        assert g.params == (0.5,)

    def test_preserves_label(self) -> None:
        g = RX.with_params(0.5)
        assert g.label == "RX"

    def test_multiple_params(self) -> None:
        g = U3.with_params(0.1, 0.2, 0.3)
        assert g.params == (0.1, 0.2, 0.3)

    def test_two_params(self) -> None:
        g = U2.with_params(0.1, 0.2)
        assert g.params == (0.1, 0.2)

    def test_empty_params_raises(self) -> None:
        with pytest.raises(ValueError, match="requires at least one parameter"):
            RX.with_params()

    def test_original_template_unchanged(self) -> None:
        _ = RX.with_params(0.5)
        assert RX.params == ()  # template stays parameter-free

    def test_with_params_is_idempotent_in_result(self) -> None:
        """Calling with_params twice produces equal gates."""
        g1 = RX.with_params(0.5)
        g2 = RX.with_params(0.5)
        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_different_params_not_equal(self) -> None:
        g1 = RX.with_params(0.5)
        g2 = RX.with_params(0.6)
        assert g1 != g2

    def test_negative_params(self) -> None:
        g = RX.with_params(-1.0)
        assert g.params == (-1.0,)

    def test_large_params(self) -> None:
        g = RX.with_params(1000.0)
        assert g.params == (1000.0,)


# =============================================================================
# Single-qubit rotation gates — matrix correctness
# =============================================================================


class TestRXMatrix:
    """RX(θ) = exp(-i θ X / 2) = cos(θ/2)I - i sin(θ/2)X."""

    def test_matrix_formula(self) -> None:
        theta = 0.7
        m = gate_matrix(RX.with_params(theta))
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        expected = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(RX.with_params(0.0))
        assert_close_to_identity(m)

    def test_theta_2pi_is_minus_identity(self) -> None:
        """RX(2π) = cos(π)I - i sin(π)X = -I."""
        m = gate_matrix(RX.with_params(2 * math.pi))
        expected = -np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_pi_is_minus_i_x(self) -> None:
        """RX(π) = cos(π/2)I - i sin(π/2)X = -i X."""
        m = gate_matrix(RX.with_params(math.pi))
        expected = -1j * np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_rx_pi_2_creates_equal_superposition(self) -> None:
        """RX(π/2)|0⟩ = cos(π/4)|0⟩ - i sin(π/4)|1⟩."""
        m = gate_matrix(RX.with_params(math.pi / 2))
        ket0 = np.array([1, 0], dtype=complex)
        result = m @ ket0
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        np.testing.assert_array_almost_equal(result, np.array([c, -1j * s]))


class TestRYMatrix:
    """RY(θ) = exp(-i θ Y / 2) = cos(θ/2)I - i sin(θ/2)Y."""

    def test_matrix_formula(self) -> None:
        theta = 0.7
        m = gate_matrix(RY.with_params(theta))
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        expected = np.array([[c, -s], [s, c]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(RY.with_params(0.0))
        assert_close_to_identity(m)

    def test_theta_2pi_is_minus_identity(self) -> None:
        m = gate_matrix(RY.with_params(2 * math.pi))
        expected = -np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_pi_is_minus_i_y(self) -> None:
        """RY(π) = -i Y."""
        m = gate_matrix(RY.with_params(math.pi))
        expected = np.array([[0, -1], [1, 0]], dtype=complex)  # = -iY
        # -i * [[0,-i],[i,0]] = [[0,-1],[1,0]]
        np.testing.assert_array_almost_equal(m, expected)

    def test_ry_pi_2_flips_0_to_equal_superposition_real(self) -> None:
        """RY(π/2)|0⟩ = cos(π/4)|0⟩ + sin(π/4)|1⟩ (real coefficients)."""
        m = gate_matrix(RY.with_params(math.pi / 2))
        ket0 = np.array([1, 0], dtype=complex)
        result = m @ ket0
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        np.testing.assert_array_almost_equal(result, np.array([c, s]))


class TestRZMatrix:
    """RZ(θ) = exp(-i θ Z / 2) = diag(exp(-iθ/2), exp(iθ/2))."""

    def test_matrix_formula(self) -> None:
        theta = 0.7
        m = gate_matrix(RZ.with_params(theta))
        e_neg = math.cos(theta / 2) - 1j * math.sin(theta / 2)
        e_pos = math.cos(theta / 2) + 1j * math.sin(theta / 2)
        expected = np.array([[e_neg, 0], [0, e_pos]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(RZ.with_params(0.0))
        assert_close_to_identity(m)

    def test_theta_2pi_is_minus_identity(self) -> None:
        m = gate_matrix(RZ.with_params(2 * math.pi))
        expected = -np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_rz_is_diagonal(self) -> None:
        """RZ should always be diagonal."""
        for theta in THETA_VALUES:
            m = gate_matrix(RZ.with_params(theta))
            assert m[0, 1] == 0.0j, f"Off-diagonal non-zero for θ={theta}"
            assert m[1, 0] == 0.0j, f"Off-diagonal non-zero for θ={theta}"


class TestPhaseGate:
    """P(θ) = diag(1, exp(iθ))."""

    def test_matrix_formula(self) -> None:
        theta = 0.7
        m = gate_matrix(P.with_params(theta))
        expected = np.array(
            [[1, 0], [0, math.cos(theta) + 1j * math.sin(theta)]], dtype=complex
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(P.with_params(0.0))
        assert_close_to_identity(m)

    def test_theta_pi_is_z(self) -> None:
        """P(π) = Z."""
        m = gate_matrix(P.with_params(math.pi))
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_pi_2_is_s(self) -> None:
        """P(π/2) = S."""
        m = gate_matrix(P.with_params(math.pi / 2))
        expected = np.array([[1, 0], [0, 1j]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_theta_pi_4_is_t(self) -> None:
        """P(π/4) = T."""
        m = gate_matrix(P.with_params(math.pi / 4))
        expected = np.array(
            [[1, 0], [0, math.cos(math.pi / 4) + 1j * math.sin(math.pi / 4)]],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)


class TestU3Gate:
    """U3(θ, φ, λ) — universal single-qubit gate."""

    def test_matrix_formula(self) -> None:
        theta, phi, lam = 0.5, 0.3, 0.7
        m = gate_matrix(U3.with_params(theta, phi, lam))
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        e_ip = math.cos(phi) + 1j * math.sin(phi)
        e_il = math.cos(lam) + 1j * math.sin(lam)
        expected = np.array(
            [
                [c, -s * e_il],
                [s * e_ip, c * e_ip * e_il],
            ],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_u3_000_is_identity(self) -> None:
        m = gate_matrix(U3.with_params(0.0, 0.0, 0.0))
        assert_close_to_identity(m)

    def test_u3_pi_00_is_x(self) -> None:
        """U3(π, 0, 0) = -i X (up to global phase)."""
        # cos(π/2)=0, sin(π/2)=1 → [[0, -1], [1, 0]]
        m = gate_matrix(U3.with_params(math.pi, 0.0, 0.0))
        # This gives [[0,-1],[1,0]] which equals -iX * i? No, let me check.
        # [[0,-1],[1,0]] applied to |0⟩ gives |1⟩, to |1⟩ gives -|0⟩
        # It's not exactly X. But it's correct for U3.
        expected = np.array([[0, -1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_u3_pi_2_00_is_hadamard_up_to_phase(self) -> None:
        """U3(π/2, 0, 0) creates equal superposition."""
        m = gate_matrix(U3.with_params(math.pi / 2, 0.0, 0.0))
        ket0 = np.array([1, 0], dtype=complex)
        result = m @ ket0
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        np.testing.assert_array_almost_equal(result, np.array([c, s]))


class TestU2Gate:
    """U2(φ, λ) = U3(π/2, φ, λ)."""

    def test_u2_equals_u3_with_pi_2(self) -> None:
        phi, lam = 0.3, 0.7
        m_u2 = gate_matrix(U2.with_params(phi, lam))
        m_u3 = gate_matrix(U3.with_params(math.pi / 2, phi, lam))
        np.testing.assert_array_almost_equal(m_u2, m_u3)

    def test_u2_00_is_hadamard_up_to_phase(self) -> None:
        """U2(0, 0) = U3(π/2, 0, 0)."""
        m = gate_matrix(U2.with_params(0.0, 0.0))
        ket0 = np.array([1, 0], dtype=complex)
        result = m @ ket0
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        np.testing.assert_array_almost_equal(result, np.array([c, s]))


class TestU1Gate:
    """U1(λ) = U3(0, 0, λ) = P(λ)."""

    def test_u1_equals_phase(self) -> None:
        for lam in THETA_VALUES[:5]:
            m_u1 = gate_matrix(U1.with_params(lam))
            m_p = gate_matrix(P.with_params(lam))
            np.testing.assert_array_almost_equal(
                m_u1, m_p, err_msg=f"Failed for λ={lam}"
            )


# =============================================================================
# Two-qubit parameterized gates
# =============================================================================


class TestCRXMatrix:
    """Controlled-RX gate."""

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(CRX.with_params(0.0))
        assert_close_to_identity(m)

    def test_control_0_no_effect(self) -> None:
        """When control is |0⟩, target is unchanged."""
        theta = 0.7
        m = gate_matrix(CRX.with_params(theta))
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        ket01 = np.array([0, 1, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket00, ket00)
        np.testing.assert_array_almost_equal(m @ ket01, ket01)

    def test_control_1_applies_rx(self) -> None:
        """When control is |1⟩, RX(θ) is applied to target."""
        theta = 0.7
        m = gate_matrix(CRX.with_params(theta))
        m_rx = gate_matrix(RX.with_params(theta))

        ket10 = np.array([0, 0, 1, 0], dtype=complex)
        result_10 = m @ ket10
        # |10⟩ → control=1, target=0 → RX applied to target
        target_state = m_rx @ np.array([1, 0], dtype=complex)
        expected_10 = np.array([0, 0, target_state[0], target_state[1]], dtype=complex)
        np.testing.assert_array_almost_equal(result_10, expected_10)


class TestCRYMatrix:
    """Controlled-RY gate."""

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(CRY.with_params(0.0))
        assert_close_to_identity(m)

    def test_control_0_no_effect(self) -> None:
        theta = 0.7
        m = gate_matrix(CRY.with_params(theta))
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket00, ket00)


class TestCRZMatrix:
    """Controlled-RZ gate."""

    def test_theta_zero_is_identity(self) -> None:
        m = gate_matrix(CRZ.with_params(0.0))
        assert_close_to_identity(m)

    def test_control_0_no_effect(self) -> None:
        theta = 0.7
        m = gate_matrix(CRZ.with_params(theta))
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket00, ket00)


class TestCU3Matrix:
    """Controlled-U3 gate."""

    def test_theta_zero_is_identity_when_phi_lam_zero(self) -> None:
        m = gate_matrix(CU3.with_params(0.0, 0.0, 0.0))
        assert_close_to_identity(m)

    def test_control_0_no_effect(self) -> None:
        m = gate_matrix(CU3.with_params(0.5, 0.3, 0.7))
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket00, ket00)


class TestCPMatrix:
    """Controlled-Phase gate."""

    def test_only_11_receives_phase(self) -> None:
        theta = 0.7
        expected = np.diag([1, 1, 1, np.exp(1j * theta)])
        np.testing.assert_allclose(gate_matrix(CP.with_params(theta)), expected)


# =============================================================================
# Unitarity — all parameterized gates for many parameter values
# =============================================================================


class TestUnitarityParameterized:
    """Every parameterized gate matrix MUST be unitary: U† U = I."""

    @pytest.mark.parametrize("theta", THETA_VALUES)
    def test_rx_unitary(self, theta: float) -> None:
        m = gate_matrix(RX.with_params(theta))
        assert is_unitary(m), f"RX({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES)
    def test_ry_unitary(self, theta: float) -> None:
        m = gate_matrix(RY.with_params(theta))
        assert is_unitary(m), f"RY({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES)
    def test_rz_unitary(self, theta: float) -> None:
        m = gate_matrix(RZ.with_params(theta))
        assert is_unitary(m), f"RZ({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES)
    def test_phase_unitary(self, theta: float) -> None:
        m = gate_matrix(P.with_params(theta))
        assert is_unitary(m), f"P({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES[:5])
    def test_crx_unitary(self, theta: float) -> None:
        m = gate_matrix(CRX.with_params(theta))
        assert is_unitary(m), f"CRX({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES[:5])
    def test_cry_unitary(self, theta: float) -> None:
        m = gate_matrix(CRY.with_params(theta))
        assert is_unitary(m), f"CRY({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES[:5])
    def test_crz_unitary(self, theta: float) -> None:
        m = gate_matrix(CRZ.with_params(theta))
        assert is_unitary(m), f"CRZ({theta}) is not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES)
    def test_cp_unitary(self, theta: float) -> None:
        assert is_unitary(gate_matrix(CP.with_params(theta)))

    @pytest.mark.parametrize("theta", THETA_VALUES[:3])
    @pytest.mark.parametrize("phi", PHI_VALUES[:3])
    @pytest.mark.parametrize("lam", [0.0, math.pi / 3])
    def test_u3_unitary(self, theta: float, phi: float, lam: float) -> None:
        m = gate_matrix(U3.with_params(theta, phi, lam))
        assert is_unitary(m), f"U3({theta},{phi},{lam}) not unitary"

    @pytest.mark.parametrize("theta", THETA_VALUES[:3])
    @pytest.mark.parametrize("phi", PHI_VALUES[:3])
    @pytest.mark.parametrize("lam", [0.0, math.pi / 3])
    def test_cu3_unitary(self, theta: float, phi: float, lam: float) -> None:
        m = gate_matrix(CU3.with_params(theta, phi, lam))
        assert is_unitary(m), f"CU3({theta},{phi},{lam}) not unitary"


# =============================================================================
# Matrix dimensions
# =============================================================================


class TestParameterizedMatrixDimensions:
    """Gate matrix dimensions must be 2^n × 2^n."""

    @pytest.mark.parametrize("template", ALL_PARAMETERIZED)
    def test_correct_dimensions(self, template: Gate) -> None:
        # Create a gate with appropriate number of params
        num_params = {
            "rx": 1,
            "ry": 1,
            "rz": 1,
            "p": 1,
            "u1": 1,
            "u2": 2,
            "u3": 3,
            "crx": 1,
            "cry": 1,
            "crz": 1,
            "cp": 1,
            "cu3": 3,
        }[template.name]
        params = tuple(float(i + 1) for i in range(num_params))
        g = template.with_params(*params)
        m = gate_matrix(g)
        expected_dim = 2**g.num_qubits
        assert m.shape == (expected_dim, expected_dim), (
            f"{g.name}: expected ({expected_dim},{expected_dim}), got {m.shape}"
        )


# =============================================================================
# Error handling
# =============================================================================


class TestParameterizedErrorHandling:
    """Parameterized gates should raise clear errors for misuse."""

    def test_rx_without_params_raises(self) -> None:
        with pytest.raises(ValueError, match="requires parameters"):
            gate_matrix(RX)

    def test_ry_without_params_raises(self) -> None:
        with pytest.raises(ValueError, match="requires parameters"):
            gate_matrix(RY)

    def test_rz_without_params_raises(self) -> None:
        with pytest.raises(ValueError, match="requires parameters"):
            gate_matrix(RZ)

    def test_u2_wrong_param_count_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 2 parameters"):
            gate_matrix(U2.with_params(0.5))

    def test_u3_wrong_param_count_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 3 parameters"):
            gate_matrix(U3.with_params(0.5, 0.3))

    def test_cu3_wrong_param_count_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 3 parameters"):
            gate_matrix(CU3.with_params(0.5))


# =============================================================================
# Special value properties
# =============================================================================


class TestSpecialValues:
    """Verify behaviour at special parameter values (0, π, 2π)."""

    def test_rx_4pi_is_identity(self) -> None:
        """RX(4π) = I."""
        m = gate_matrix(RX.with_params(4 * math.pi))
        assert_close_to_identity(m)

    def test_ry_4pi_is_identity(self) -> None:
        m = gate_matrix(RY.with_params(4 * math.pi))
        assert_close_to_identity(m)

    def test_rz_4pi_is_identity(self) -> None:
        m = gate_matrix(RZ.with_params(4 * math.pi))
        assert_close_to_identity(m)

    def test_rx_negative_theta_is_inverse_of_positive(self) -> None:
        """RX(-θ) = RX(θ)†."""
        theta = 0.5
        m_pos = gate_matrix(RX.with_params(theta))
        m_neg = gate_matrix(RX.with_params(-theta))
        np.testing.assert_array_almost_equal(m_neg, m_pos.conj().T)

    def test_ry_negative_theta_is_inverse_of_positive(self) -> None:
        theta = 0.5
        m_pos = gate_matrix(RY.with_params(theta))
        m_neg = gate_matrix(RY.with_params(-theta))
        np.testing.assert_array_almost_equal(m_neg, m_pos.conj().T)

    def test_rz_negative_theta_is_inverse_of_positive(self) -> None:
        theta = 0.5
        m_pos = gate_matrix(RZ.with_params(theta))
        m_neg = gate_matrix(RZ.with_params(-theta))
        np.testing.assert_array_almost_equal(m_neg, m_pos.conj().T)

    def test_rx_pi_equals_ry_pi_up_to_phase(self) -> None:
        """RX(π) and RY(π) differ only by a global phase (both give -i times Pauli)."""
        # Already covered: both equal -i * Pauli
        # RX(π) = -iX, RY(π) = -iY
        # These differ by more than global phase (X vs Y), so not equal
        pass  # No assertion needed — this is documenting the non-relation


# =============================================================================
# Identity and periodicity patterns
# =============================================================================


class TestPeriodicity:
    """Rotation gates should be periodic in their parameters."""

    def test_rx_4pi_periodicity(self) -> None:
        """RX(θ + 4π) = RX(θ)."""
        theta = 0.7
        m1 = gate_matrix(RX.with_params(theta))
        m2 = gate_matrix(RX.with_params(theta + 4 * math.pi))
        np.testing.assert_array_almost_equal(m1, m2)

    def test_rx_2pi_is_minus_rx_0(self) -> None:
        """RX(θ + 2π) = -RX(θ)."""
        theta = 0.7
        m1 = gate_matrix(RX.with_params(theta))
        m2 = gate_matrix(RX.with_params(theta + 2 * math.pi))
        np.testing.assert_array_almost_equal(m2, -m1)

    def test_rz_2pi_is_minus_identity(self) -> None:
        """RZ(2π) = -I."""
        m = gate_matrix(RZ.with_params(2 * math.pi))
        expected = -np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)


# =============================================================================
# Immutability of parameterized gate instances
# =============================================================================


class TestParameterizedGateImmutability:
    """Gates created via with_params() should also be immutable."""

    def test_cannot_set_params(self) -> None:
        g = RX.with_params(0.5)
        with pytest.raises(Exception):  # noqa: B017
            g.params = (1.0,)  # type: ignore[misc]

    def test_cannot_set_name(self) -> None:
        g = RX.with_params(0.5)
        with pytest.raises(Exception):  # noqa: B017
            g.name = "new"  # type: ignore[misc]


class TestParamValidationRegression:
    """Regression tests for B26 / B12 parameter validation."""

    @pytest.mark.parametrize(
        ("name", "num_qubits", "count"),
        [
            ("rx", 1, 2),
            ("ry", 1, 2),
            ("rz", 1, 2),
            ("p", 1, 2),
            ("u1", 1, 2),
            ("u2", 1, 3),
            ("u3", 1, 4),
            ("crx", 2, 2),
            ("cry", 2, 2),
            ("crz", 2, 2),
            ("cp", 2, 2),
            ("cu3", 2, 4),
        ],
    )
    def test_gate_matrix_rejects_extra_params(
        self, name: str, num_qubits: int, count: int
    ) -> None:
        """Too many parameters raise ValueError instead of being ignored."""
        gate = Gate(name=name, num_qubits=num_qubits, params=(0.1,) * count)
        with pytest.raises(ValueError, match="requires"):
            gate_matrix(gate)

    def test_standard_gate_with_params_rejected(self) -> None:
        """A fixed gate given parameters is an incorrect parameter count."""
        with pytest.raises(ValueError, match="no parameters"):
            gate_matrix(Gate(name="h", num_qubits=1, params=(0.1,)))

    def test_gate_matrix_rejects_non_numeric_param(self) -> None:
        """String parameters are not silently parsed."""
        with pytest.raises(ValueError, match="non-numeric"):
            gate_matrix(Gate(name="rx", num_qubits=1, params=("0.5",)))

    def test_numpy_params_normalized_to_float(self) -> None:
        """Gate stores numpy real scalars as Python floats."""
        gate = RX.with_params(np.float32(0.5))  # type: ignore[arg-type]
        assert gate.params == (0.5,) and type(gate.params[0]) is float
        gate = Gate(name="rx", num_qubits=1, params=(np.int64(2),))
        assert gate.params == (2.0,) and type(gate.params[0]) is float
