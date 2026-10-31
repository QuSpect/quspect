"""Regression tests for ``quspect.simulation.density_matrix.DensityMatrix``."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.simulation.density_matrix import DensityMatrix


def _product(*single: np.ndarray) -> DensityMatrix:
    """Product state; ``single[q]`` is the 2x2 state of qubit ``q``."""
    rho = np.array([[1.0]], dtype=complex)
    for s in single:
        rho = np.kron(s, rho)  # qubit q is bit q of the index (little-endian)
    return DensityMatrix.from_array(rho)


ZERO = np.diag([1.0, 0.0]).astype(complex)
ONE = np.diag([0.0, 1.0]).astype(complex)
PLUS = np.full((2, 2), 0.5, dtype=complex)
MIXED = np.eye(2, dtype=complex) / 2


class TestProbabilities:
    """``probabilities(qubits)`` keeps ``qubits`` with qubits[0] as the LSB."""

    def test_subset_keeps_measured_qubit(self) -> None:
        """|q2 q1 q0> = |001>: P(q0) = [0, 1]."""
        dm = _product(ONE, ZERO, ZERO)
        np.testing.assert_allclose(dm.probabilities((0,)), [0.0, 1.0])
        np.testing.assert_allclose(dm.probabilities((2,)), [1.0, 0.0])

    def test_order_follows_qubits(self) -> None:
        """Bit j of the index is the outcome of qubits[j]."""
        dm = _product(ONE, ZERO, ONE)  # q0=1, q1=0, q2=1
        # (1, 0): bit0 = q1 = 0, bit1 = q0 = 1 -> index 2
        np.testing.assert_allclose(dm.probabilities((1, 0)), [0, 0, 1, 0])
        # (2, 1, 0): q2=1 -> bit0, q1=0 -> bit1, q0=1 -> bit2 -> index 5
        expected = np.zeros(8)
        expected[5] = 1.0
        np.testing.assert_allclose(dm.probabilities((2, 1, 0)), expected)

    def test_marginal_of_mixed_qubit(self) -> None:
        """Marginal on a maximally mixed qubit is uniform."""
        dm = _product(ONE, MIXED)
        np.testing.assert_allclose(dm.probabilities((1,)), [0.5, 0.5])
        np.testing.assert_allclose(dm.probabilities((0, 1)), [0, 0.5, 0, 0.5])

    def test_rejects_duplicates_and_out_of_range(self) -> None:
        """Invalid qubit tuples raise ValueError."""
        dm = DensityMatrix(2)
        with pytest.raises(ValueError):
            dm.probabilities((0, 0))
        with pytest.raises(ValueError):
            dm.probabilities((2,))


def test_subsystem_purity_is_on_given_qubits() -> None:
    """q0 = |0>, q1 = I/2: purity on (0,) is 1 and on (1,) is 1/2."""
    dm = _product(ZERO, MIXED)
    assert dm.subsystem_purity((0,)) == pytest.approx(1.0)
    assert dm.subsystem_purity((1,)) == pytest.approx(0.5)
    assert dm.subsystem_purity((0, 1)) == pytest.approx(0.5)


class TestMeasure:
    """``measure`` measures exactly ``qubits`` and orders bits like ``qubits``."""

    def test_subset_collapses_correct_qubit(self) -> None:
        """|+>_q0 |1>_q1: measuring q1 always yields '1' and leaves q0 intact."""
        dm = _product(PLUS, ONE)
        rng = np.random.default_rng(1)
        for _ in range(20):
            bits, post = dm.measure((1,), rng)
            assert bits == "1"
            np.testing.assert_allclose(post._data, dm._data, atol=1e-12)

    def test_permuted_qubits(self) -> None:
        """q0 = 1, q1 = 0: measure((1, 0)) returns s = (q1, q0) = '01'."""
        dm = _product(ONE, ZERO)
        bits, post = dm.measure((1, 0), np.random.default_rng(0))
        assert bits == "01"
        np.testing.assert_allclose(post._data, dm._data, atol=1e-12)

    def test_partial_collapse_of_bell_state(self) -> None:
        """Measuring one qubit of a Bell state collapses both."""
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        dm = DensityMatrix.from_pure_state(bell)
        bits, post = dm.measure((0,), np.random.default_rng(3))
        idx = 0 if bits == "0" else 3
        expected = np.zeros((4, 4), dtype=complex)
        expected[idx, idx] = 1.0
        np.testing.assert_allclose(post._data, expected, atol=1e-12)
