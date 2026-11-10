"""Regression tests for ``quspect.simulation.density_matrix.DensityMatrix``."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from quspect import QuantumCircuit, Simulator
from quspect.simulation import DensityMatrixSimulator
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


class TestSmallEntries:
    """Gates and channels keep entries of any size.

    Every entry below ``sqrt(1e-15)`` (about 3.16e-8) used to be zeroed at each
    gate and channel, so small populations and trace leaked away.
    """

    def test_identity_gate_leaves_the_matrix_unchanged(self) -> None:
        rho = np.array([[1 - 1e-9, 2e-10 - 3e-10j], [2e-10 + 3e-10j, 1e-9]])
        dm = DensityMatrix.from_array(rho)
        dm.apply_gate(np.eye(2, dtype=complex), (0,))
        np.testing.assert_array_equal(dm._data, rho)

    def test_gates_keep_a_small_population(self) -> None:
        """ry(1e-4) puts 2.5e-9 on |1>; a later gate on another qubit keeps it."""
        circuit = QuantumCircuit().ry(1e-4, 0).x(1)
        diagonal = np.real(
            np.diag(DensityMatrixSimulator().density_matrix(circuit)._data)
        )
        small = np.sin(5e-5) ** 2
        np.testing.assert_allclose(diagonal, [0, 0, 1 - small, small], rtol=1e-12)

    def test_rotation_below_the_old_cut_is_applied(self) -> None:
        """rx(2e-9) has off-diagonal gate entries of 1e-9."""
        dm = DensityMatrix(1)
        half = 1e-9
        gate = np.array(
            [[np.cos(half), -1j * np.sin(half)], [-1j * np.sin(half), np.cos(half)]]
        )
        dm.apply_gate(gate, (0,))
        expected = gate @ np.diag([1.0, 0.0]) @ gate.conj().T
        np.testing.assert_allclose(dm._data, expected, rtol=1e-12, atol=0)
        assert abs(dm._data[0, 1]) == pytest.approx(1e-9)

    def test_long_small_angle_circuit_matches_the_statevector(self) -> None:
        """A noiseless Trotter circuit of 2000 small steps keeps trace 1."""
        num_qubits, steps, dt = 4, 2000, 1e-5
        circuit = QuantumCircuit()
        for _ in range(steps):
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1).rz(2 * dt, q + 1).cx(q, q + 1)
            for q in range(num_qubits):
                circuit.rx(2 * dt, q)
        rho = DensityMatrixSimulator().density_matrix(circuit)._data
        state = np.asarray(Simulator().statevector(circuit))
        assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(rho, np.outer(state, state.conj()), atol=1e-12)
        # P(q0 = 1) is about 4e-4: it lost 1 to 3 percent to the cut.
        odd = np.arange(1 << num_qubits) & 1 == 1
        assert np.real(np.diag(rho))[odd].sum() == pytest.approx(
            float((np.abs(state) ** 2)[odd].sum()), rel=1e-9
        )

    def test_channel_keeps_a_small_population(self) -> None:
        """Bit flip with probability p on diag(1 - e, e)."""
        p, e = 1e-10, 1e-12
        dm = DensityMatrix.from_array(np.diag([1 - e, e]).astype(complex))
        flip = np.array([[0, 1], [1, 0]], dtype=complex)
        dm.apply_channel([np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * flip], (0,))
        excited = (1 - p) * e + p * (1 - e)
        assert np.real(dm._data[1, 1]) == pytest.approx(excited, rel=1e-12)
        assert np.real(np.trace(dm._data)) == pytest.approx(1.0, abs=1e-15)

    def test_partial_trace_keeps_small_entries(self) -> None:
        rho = np.diag([1 - 3e-16, 1e-16, 1e-16, 1e-16]).astype(complex)
        reduced = DensityMatrix.from_array(rho).partial_trace((1,))
        np.testing.assert_allclose(
            np.diag(reduced._data), [rho[0, 0] + 1e-16, 2e-16], rtol=1e-12
        )


_SIZE_PROBE = """
from quspect import QuantumCircuit
from quspect.simulation import DensityMatrix, DensityMatrixSimulator

def outcome(action):
    try:
        action()
    except Exception as error:
        return type(error).__name__
    return "constructed"

for n in (-1, 26, 29, 30, 31, 32, 40, 63, 64, 65, 2**32, 2**64):
    print(n, outcome(lambda: DensityMatrix(n)))
circuit = QuantumCircuit().h(0).cx(0, 35).measure([0], [0])
simulator = DensityMatrixSimulator()
print("run", outcome(lambda: simulator.run(circuit, shots=1)))
print("density_matrix", outcome(lambda: simulator.density_matrix(circuit)))
"""


class TestSize:
    """Sizes that cannot exist raise instead of crashing the interpreter."""

    def test_impossible_sizes_raise(self) -> None:
        """32 to 63 qubits used to end the process with a segmentation fault.

        The probe runs in a child process so that a regression fails this
        test instead of killing the test run.
        """
        child = subprocess.run(
            [sys.executable, "-c", _SIZE_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert child.returncode == 0, child.stderr[-2000:]
        assert child.stdout.splitlines() == [
            "-1 ValueError",
            # 16 * 4**n bytes exceed every address space: the allocator refuses.
            "26 MemoryError",
            "29 MemoryError",
            # 4**n entries can no longer be counted in an allocation.
            "30 ValueError",
            "31 ValueError",
            "32 ValueError",
            "40 ValueError",
            "63 ValueError",
            "64 ValueError",
            "65 ValueError",
            "4294967296 ValueError",
            "18446744073709551616 ValueError",
            "run ValueError",
            "density_matrix ValueError",
        ]

    def test_messages_name_the_limit_and_the_size(self) -> None:
        with pytest.raises(ValueError, match=r"num_qubits must be <= 29 .* got 64"):
            DensityMatrix(64)
        with pytest.raises(MemoryError, match=r"29-qubit density matrix \(\d+ bytes\)"):
            DensityMatrix(29)
        with pytest.raises(TypeError):
            DensityMatrix(2.0)  # type: ignore[arg-type]

    def test_small_sizes_still_work(self) -> None:
        assert DensityMatrix(0).shape == (1, 1)
        assert DensityMatrix(np.int64(3)).shape == (8, 8)
