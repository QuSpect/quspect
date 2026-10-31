"""Comprehensive tests for standard (fixed-parameter) quantum gates.

Tests cover:
- Gate metadata (name, num_qubits, params, label)
- Unitary matrix correctness against known mathematical forms
- Matrix properties: unitarity, dimension
- Gate immutability and hashing
- gate_matrix() dispatch
- Edge cases and error handling
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.gate import Gate
from quspect.gates.standard import (
    ALL_STANDARD,
    CCX,
    CX,
    CZ,
    SDG,
    SWAP,
    TDG,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
)

# =============================================================================
# Helper utilities
# =============================================================================


def is_unitary(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is unitary: U† U = I."""
    identity = np.eye(matrix.shape[0], dtype=complex)
    return bool(np.allclose(matrix.conj().T @ matrix, identity, atol=tol))


def assert_gate_metadata(
    gate: Gate,
    expected_name: str,
    expected_num_qubits: int,
    expected_label: str,
) -> None:
    """Assert a gate's metadata matches expectations."""
    assert gate.name == expected_name, (
        f"Expected name '{expected_name}', got '{gate.name}'"
    )
    assert gate.num_qubits == expected_num_qubits, (
        f"Expected {expected_num_qubits} qubits, got {gate.num_qubits}"
    )
    assert gate.params == (), f"Expected empty params, got {gate.params}"
    assert gate.label == expected_label, (
        f"Expected label '{expected_label}', got '{gate.label}'"
    )


# =============================================================================
# Gate metadata tests — every standard gate
# =============================================================================


class TestStandardGateMetadata:
    """Verify metadata for every standard gate."""

    def test_identity_metadata(self) -> None:
        assert_gate_metadata(I, "id", 1, "I")

    def test_hadamard_metadata(self) -> None:
        assert_gate_metadata(H, "h", 1, "H")

    def test_pauli_x_metadata(self) -> None:
        assert_gate_metadata(X, "x", 1, "X")

    def test_pauli_y_metadata(self) -> None:
        assert_gate_metadata(Y, "y", 1, "Y")

    def test_pauli_z_metadata(self) -> None:
        assert_gate_metadata(Z, "z", 1, "Z")

    def test_s_metadata(self) -> None:
        assert_gate_metadata(S, "s", 1, "S")

    def test_sdg_metadata(self) -> None:
        assert_gate_metadata(SDG, "sdg", 1, "S†")

    def test_t_metadata(self) -> None:
        assert_gate_metadata(T, "t", 1, "T")

    def test_tdg_metadata(self) -> None:
        assert_gate_metadata(TDG, "tdg", 1, "T†")

    def test_cnot_metadata(self) -> None:
        assert_gate_metadata(CX, "cx", 2, "CX")

    def test_cz_metadata(self) -> None:
        assert_gate_metadata(CZ, "cz", 2, "CZ")

    def test_swap_metadata(self) -> None:
        assert_gate_metadata(SWAP, "swap", 2, "SWAP")

    def test_toffoli_metadata(self) -> None:
        assert_gate_metadata(CCX, "ccx", 3, "CCX")


class TestStandardGateUniqueness:
    """Verify each standard gate is a distinct instance with unique name."""

    def test_all_names_unique(self) -> None:
        names = [g.name for g in ALL_STANDARD]
        assert len(names) == len(set(names)), f"Duplicate gate names: {names}"

    def test_count_is_15(self) -> None:
        assert len(ALL_STANDARD) == 15


# =============================================================================
# Matrix correctness tests — single-qubit gates
# =============================================================================


class TestSingleQubitMatrices:
    """Verify the unitary matrix of every single-qubit standard gate."""

    def test_identity_matrix(self) -> None:
        m = gate_matrix(I)
        expected = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_pauli_x_matrix(self) -> None:
        m = gate_matrix(X)
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_pauli_x_flips_states(self) -> None:
        """X|0⟩ = |1⟩, X|1⟩ = |0⟩."""
        m = gate_matrix(X)
        ket0 = np.array([1, 0], dtype=complex)
        ket1 = np.array([0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket0, ket1)
        np.testing.assert_array_almost_equal(m @ ket1, ket0)

    def test_pauli_y_matrix(self) -> None:
        m = gate_matrix(Y)
        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_pauli_z_matrix(self) -> None:
        m = gate_matrix(Z)
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_pauli_z_flips_phase_of_one(self) -> None:
        """Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩."""
        m = gate_matrix(Z)
        ket0 = np.array([1, 0], dtype=complex)
        ket1 = np.array([0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket0, ket0)
        np.testing.assert_array_almost_equal(m @ ket1, -ket1)

    def test_hadamard_matrix(self) -> None:
        m = gate_matrix(H)
        inv_sqrt2 = 1.0 / math.sqrt(2)
        expected = np.array(
            [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=complex
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_hadamard_creates_superposition(self) -> None:
        """H|0⟩ = (|0⟩ + |1⟩)/√2."""
        m = gate_matrix(H)
        ket0 = np.array([1, 0], dtype=complex)
        result = m @ ket0
        inv_sqrt2 = 1.0 / math.sqrt(2)
        np.testing.assert_array_almost_equal(result, np.array([inv_sqrt2, inv_sqrt2]))

    def test_hadamard_is_self_inverse(self) -> None:
        """H H = I."""
        m = gate_matrix(H)
        np.testing.assert_array_almost_equal(m @ m, np.eye(2))

    def test_s_matrix(self) -> None:
        m = gate_matrix(S)
        expected = np.array([[1, 0], [0, 1j]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_s_times_sdg_is_identity(self) -> None:
        """S · S† = I."""
        m_s = gate_matrix(S)
        m_sdg = gate_matrix(SDG)
        np.testing.assert_array_almost_equal(m_s @ m_sdg, np.eye(2))

    def test_s_squared_is_z(self) -> None:
        """S² = Z."""
        m_s = gate_matrix(S)
        m_z = gate_matrix(Z)
        np.testing.assert_array_almost_equal(m_s @ m_s, m_z)

    def test_sdg_matrix(self) -> None:
        m = gate_matrix(SDG)
        expected = np.array([[1, 0], [0, -1j]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_t_matrix(self) -> None:
        m = gate_matrix(T)
        expected = np.array(
            [[1, 0], [0, math.cos(math.pi / 4) + 1j * math.sin(math.pi / 4)]],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_t_times_tdg_is_identity(self) -> None:
        """T · T† = I."""
        m_t = gate_matrix(T)
        m_tdg = gate_matrix(TDG)
        np.testing.assert_array_almost_equal(m_t @ m_tdg, np.eye(2))

    def test_t_squared_is_s(self) -> None:
        """T² = S."""
        m_t = gate_matrix(T)
        m_s = gate_matrix(S)
        np.testing.assert_array_almost_equal(m_t @ m_t, m_s)

    def test_tdg_matrix(self) -> None:
        m = gate_matrix(TDG)
        expected = np.array(
            [[1, 0], [0, math.cos(math.pi / 4) - 1j * math.sin(math.pi / 4)]],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)


# =============================================================================
# Matrix correctness tests — multi-qubit gates
# =============================================================================


class TestMultiQubitMatrices:
    """Verify the unitary matrix of every multi-qubit standard gate."""

    def test_cnot_matrix(self) -> None:
        m = gate_matrix(CX)
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_cnot_00_unchanged(self) -> None:
        """CNOT|00⟩ = |00⟩."""
        m = gate_matrix(CX)
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket00, ket00)

    def test_cnot_10_flips_to_11(self) -> None:
        """CNOT|10⟩ = |11⟩."""
        m = gate_matrix(CX)
        ket10 = np.array([0, 0, 1, 0], dtype=complex)
        ket11 = np.array([0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket10, ket11)

    def test_cnot_11_flips_to_10(self) -> None:
        """CNOT|11⟩ = |10⟩."""
        m = gate_matrix(CX)
        ket11 = np.array([0, 0, 0, 1], dtype=complex)
        ket10 = np.array([0, 0, 1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket11, ket10)

    def test_cnot_is_self_inverse(self) -> None:
        """CNOT · CNOT = I."""
        m = gate_matrix(CX)
        np.testing.assert_array_almost_equal(m @ m, np.eye(4))

    def test_cz_matrix(self) -> None:
        m = gate_matrix(CZ)
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_cz_is_symmetric(self) -> None:
        """CZ is symmetric between control and target."""
        m = gate_matrix(CZ)
        np.testing.assert_array_almost_equal(m, m.T)

    def test_cz_is_self_inverse(self) -> None:
        """CZ · CZ = I (diagonal with ±1)."""
        m = gate_matrix(CZ)
        np.testing.assert_array_almost_equal(m @ m, np.eye(4))

    def test_swap_matrix(self) -> None:
        m = gate_matrix(SWAP)
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )
        np.testing.assert_array_almost_equal(m, expected)

    def test_swap_exchanges_01_and_10(self) -> None:
        """SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩."""
        m = gate_matrix(SWAP)
        ket01 = np.array([0, 1, 0, 0], dtype=complex)
        ket10 = np.array([0, 0, 1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket01, ket10)
        np.testing.assert_array_almost_equal(m @ ket10, ket01)

    def test_swap_is_self_inverse(self) -> None:
        """SWAP · SWAP = I."""
        m = gate_matrix(SWAP)
        np.testing.assert_array_almost_equal(m @ m, np.eye(4))

    def test_swap_preserves_00_and_11(self) -> None:
        """SWAP|00⟩ = |00⟩, SWAP|11⟩ = |11⟩."""
        m = gate_matrix(SWAP)
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        ket11 = np.array([0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(m @ ket00, ket00)
        np.testing.assert_array_almost_equal(m @ ket11, ket11)

    def test_toffoli_matrix(self) -> None:
        m = gate_matrix(CCX)
        assert m.shape == (8, 8)

    def test_toffoli_acts_only_on_110_and_111(self) -> None:
        """Toffoli flips target when both controls are |1⟩."""
        m = gate_matrix(CCX)
        # |110⟩ (index 6) → |111⟩ (index 7)
        ket110 = np.zeros(8, dtype=complex)
        ket110[6] = 1
        ket111 = np.zeros(8, dtype=complex)
        ket111[7] = 1
        np.testing.assert_array_almost_equal(m @ ket110, ket111)
        np.testing.assert_array_almost_equal(m @ ket111, ket110)

    def test_toffoli_preserves_other_states(self) -> None:
        """States with at least one |0⟩ control are unchanged."""
        m = gate_matrix(CCX)
        for i in range(6):  # |000⟩ through |101⟩
            ket = np.zeros(8, dtype=complex)
            ket[i] = 1
            np.testing.assert_array_almost_equal(
                m @ ket, ket, err_msg=f"Failed for |{i:03b}⟩"
            )


# =============================================================================
# Matrix property tests — generic properties every gate must satisfy
# =============================================================================


class TestUnitarity:
    """Every gate matrix MUST be unitary: U† U = I."""

    @pytest.mark.parametrize("gate", ALL_STANDARD)
    def test_standard_gate_is_unitary(self, gate: Gate) -> None:
        m = gate_matrix(gate)
        assert is_unitary(m), f"Gate {gate.name} is not unitary:\n{m}"


class TestMatrixDimensions:
    """Gate matrix dimensions must be 2^n × 2^n where n = num_qubits."""

    @pytest.mark.parametrize("gate", ALL_STANDARD)
    def test_correct_dimensions(self, gate: Gate) -> None:
        m = gate_matrix(gate)
        expected_dim = 2**gate.num_qubits
        assert m.shape == (expected_dim, expected_dim), (
            f"Gate {gate.name}: expected shape "
            f"({expected_dim},{expected_dim}), got {m.shape}"
        )


class TestGateMatrixIsCopy:
    """gate_matrix() should return a copy, not a reference to internal data."""

    def test_modifying_returned_matrix_does_not_affect_internal(self) -> None:
        m1 = gate_matrix(H)
        m2 = gate_matrix(H)
        m1[0, 0] = 99.0
        np.testing.assert_array_almost_equal(m2, gate_matrix(H))


# =============================================================================
# Gate dataclass behaviour
# =============================================================================


class TestGateImmutability:
    """Gate is a frozen dataclass — attempting to modify it should raise."""

    def test_cannot_set_name(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            H.name = "new_name"  # type: ignore[misc]

    def test_cannot_set_num_qubits(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            H.num_qubits = 3  # type: ignore[misc]


class TestGateEquality:
    """Gates with identical fields should be equal (value semantics)."""

    def test_same_gate_equal(self) -> None:
        g1 = Gate(name="h", num_qubits=1)
        g2 = Gate(name="h", num_qubits=1)
        assert g1 == g2

    def test_different_name_not_equal(self) -> None:
        g1 = Gate(name="h", num_qubits=1)
        g2 = Gate(name="x", num_qubits=1)
        assert g1 != g2

    def test_different_num_qubits_not_equal(self) -> None:
        g1 = Gate(name="cx", num_qubits=2)
        g2 = Gate(name="cx", num_qubits=3)
        assert g1 != g2

    def test_singletons_are_equal(self) -> None:
        """The module-level singletons should equal newly created identical gates."""
        assert Gate(name="h", num_qubits=1) == H
        assert Gate(name="cx", num_qubits=2) == CX


class TestGateHashing:
    """Gates should be hashable (can be used in sets and as dict keys)."""

    def test_hash_consistent_with_equality(self) -> None:
        g1 = Gate(name="h", num_qubits=1)
        g2 = Gate(name="h", num_qubits=1)
        assert hash(g1) == hash(g2)

    def test_gate_in_set(self) -> None:
        gate_set: set[Gate] = {H, X, CX}
        assert Gate(name="h", num_qubits=1) in gate_set
        assert Gate(name="y", num_qubits=1) not in gate_set

    def test_gate_as_dict_key(self) -> None:
        d: dict[Gate, str] = {H: "Hadamard", CX: "CNOT"}
        assert d[H] == "Hadamard"
        assert d[Gate(name="h", num_qubits=1)] == "Hadamard"


class TestGateRepr:
    """Gate.__repr__ should be readable and include key fields."""

    def test_repr_includes_name(self) -> None:
        assert "h" in repr(H)

    def test_repr_includes_num_qubits(self) -> None:
        assert "1" in repr(H)

    def test_repr_includes_label(self) -> None:
        assert "H" in repr(H)


class TestGateConstruction:
    """Gate construction validation."""

    def test_num_qubits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            Gate(name="bad", num_qubits=0)

    def test_num_qubits_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            Gate(name="bad", num_qubits=-1)

    def test_default_params_empty(self) -> None:
        g = Gate(name="test", num_qubits=1)
        assert g.params == ()

    def test_default_label_empty(self) -> None:
        g = Gate(name="test", num_qubits=1)
        assert g.label == ""

    def test_custom_label(self) -> None:
        g = Gate(name="test", num_qubits=1, label="custom")
        assert g.label == "custom"


# =============================================================================
# gate_matrix error handling
# =============================================================================


class TestGateMatrixErrors:
    """gate_matrix() should raise appropriate errors for invalid input."""

    def test_unknown_gate_name_raises_keyerror(self) -> None:
        unknown = Gate(name="nonexistent", num_qubits=1)
        with pytest.raises(KeyError, match="nonexistent"):
            gate_matrix(unknown)


# =============================================================================
# Pauli algebra (relational tests between gates)
# =============================================================================


class TestPauliAlgebra:
    """Verify well-known Pauli matrix relationships."""

    def test_x_squared_is_i(self) -> None:
        """X² = I."""
        m = gate_matrix(X)
        np.testing.assert_array_almost_equal(m @ m, np.eye(2))

    def test_y_squared_is_i(self) -> None:
        """Y² = I."""
        m = gate_matrix(Y)
        np.testing.assert_array_almost_equal(m @ m, np.eye(2))

    def test_z_squared_is_i(self) -> None:
        """Z² = I."""
        m = gate_matrix(Z)
        np.testing.assert_array_almost_equal(m @ m, np.eye(2))

    def test_xyz_equals_ii(self) -> None:
        """X Y Z = i I."""
        m_x = gate_matrix(X)
        m_y = gate_matrix(Y)
        m_z = gate_matrix(Z)
        product = m_x @ m_y @ m_z
        expected = 1j * np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(product, expected)

    def test_h_times_x_times_h_equals_z(self) -> None:
        """H X H = Z."""
        m_h = gate_matrix(H)
        m_x = gate_matrix(X)
        m_z = gate_matrix(Z)
        np.testing.assert_array_almost_equal(m_h @ m_x @ m_h, m_z)

    def test_h_times_z_times_h_equals_x(self) -> None:
        """H Z H = X."""
        m_h = gate_matrix(H)
        m_x = gate_matrix(X)
        m_z = gate_matrix(Z)
        np.testing.assert_array_almost_equal(m_h @ m_z @ m_h, m_x)

    def test_anticommutation_xy(self) -> None:
        """X Y = -Y X."""
        m_x = gate_matrix(X)
        m_y = gate_matrix(Y)
        np.testing.assert_array_almost_equal(m_x @ m_y, -m_y @ m_x)

    def test_anticommutation_xz(self) -> None:
        """X Z = -Z X."""
        m_x = gate_matrix(X)
        m_z = gate_matrix(Z)
        np.testing.assert_array_almost_equal(m_x @ m_z, -m_z @ m_x)


# =============================================================================
# Specific quantum circuit patterns
# =============================================================================


class TestCircuitPatterns:
    """Verify small circuit patterns produce expected results."""

    def test_bell_state_cnot_h_on_00(self) -> None:
        """(CNOT)(H⊗I)|00⟩ = (|00⟩ + |11⟩)/√2."""
        m_h = gate_matrix(H)
        m_i = gate_matrix(I)
        m_cx = gate_matrix(CX)

        # H on qubit 0, I on qubit 1 → H ⊗ I
        h_i = np.kron(m_h, m_i)

        # Apply CNOT
        circuit = m_cx @ h_i

        # Apply to |00⟩
        ket00 = np.array([1, 0, 0, 0], dtype=complex)
        result = circuit @ ket00

        inv_sqrt2 = 1.0 / math.sqrt(2)
        expected = np.array([inv_sqrt2, 0, 0, inv_sqrt2], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected)

    def test_swap_from_three_cnots(self) -> None:
        """SWAP = CNOT₁₂ CNOT₂₁ CNOT₁₂."""
        m_cx = gate_matrix(CX)
        # CX₂₁ = (H⊗H) CX (H⊗H) — swapping control/target
        m_h2 = np.kron(gate_matrix(H), gate_matrix(H))
        cx_rev = m_h2 @ m_cx @ m_h2  # CX with control/target swapped
        constructed_swap = m_cx @ cx_rev @ m_cx
        m_swap = gate_matrix(SWAP)
        np.testing.assert_array_almost_equal(constructed_swap, m_swap)
