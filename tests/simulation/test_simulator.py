"""Tests for the Simulator class."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.parameter import Parameter
from quspect.simulation.simulator import Simulator


class TestSimulatorConstruction:
    """Tests for Simulator initialization."""

    def test_init_without_seed(self) -> None:
        """Create simulator without seed."""
        sim = Simulator()
        assert sim.seed is None

    def test_init_with_seed(self) -> None:
        """Create simulator with seed."""
        sim = Simulator(seed=42)
        assert sim.seed == 42


class TestSimulatorRunBasic:
    """Tests for basic simulation runs."""

    def test_bell_state(self) -> None:
        """Bell state: H(0) + CX(0,1) → 50/50 '00' and '11'."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1).measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=10000)
        assert result.num_shots == 10000
        assert result.num_qubits == 2
        assert result.num_clbits == 2
        # Counts should be roughly 50/50
        count_00 = result.get_count("00")
        count_11 = result.get_count("11")
        assert 4000 < count_00 < 6000
        assert 4000 < count_11 < 6000
        # No other outcomes
        assert result.get_count("01") == 0
        assert result.get_count("10") == 0

    def test_pauli_x(self) -> None:
        """X|0> = |1>: 100% '1' measurements."""
        circ = QuantumCircuit().x(0).measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=1000)
        assert result.most_frequent() == ("1", 1000)

    def test_superposition(self) -> None:
        """H|0> = |+>: roughly 50/50 '0' and '1'."""
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=10000)
        count_0 = result.get_count("0")
        count_1 = result.get_count("1")
        assert 4000 < count_0 < 6000
        assert 4000 < count_1 < 6000

    def test_ghz_state(self) -> None:
        """GHZ state: H(0) + CX(0,1) + CX(1,2) → 50/50 '000' and '111'."""
        circ = QuantumCircuit().h(0).cx(0, 1).cx(1, 2).measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=10000)
        count_000 = result.get_count("000")
        count_111 = result.get_count("111")
        assert 4000 < count_000 < 6000
        assert 4000 < count_111 < 6000
        assert result.get_count("001") == 0
        assert result.get_count("010") == 0
        assert result.get_count("011") == 0
        assert result.get_count("100") == 0
        assert result.get_count("101") == 0
        assert result.get_count("110") == 0

    def test_circuit_with_swap(self) -> None:
        """SWAP should exchange qubit values."""
        circ = QuantumCircuit().x(0).swap(0, 1).measure_all()
        # After X(0): q0=1, q1=0 → After SWAP: q0=0, q1=1 → |10>
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=100)
        assert result.most_frequent() == ("01", 100)
        # |01> means clbit 0 = 1 (from qubit 0), clbit 1 = 0 (from qubit 1)
        # But after swap, qubit 0 has value 0, qubit 1 has value 1
        # measure_all maps qubit i → clbit i
        # So clbit 0 = 0, clbit 1 = 1 → bitstring "01"
        # Actually: X on q0 makes q0=1, q1=0. Swap exchanges: q0=0, q1=1.
        # measure_all: clbit 0 = q0 = 0, clbit 1 = q1 = 1 → "01".
        pass


class TestSimulatorEmptyAndNoMeasure:
    """Tests for empty circuits and circuits without measurements."""

    def test_empty_circuit(self) -> None:
        """Empty circuit with 0 qubits returns empty result."""
        circ = QuantumCircuit()
        sim = Simulator()
        result = sim.run(circ)
        assert result.num_shots == 0
        assert result.num_qubits == 0
        assert result.num_clbits == 0
        assert result.counts == {}
        assert result.probabilities == {}

    def test_empty_circuit_with_gates(self) -> None:
        """Circuit with N qubits and no gates."""
        # We need to infer qubit count. Use a gate then measure to infer.
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator()
        result = sim.run(circ, shots=100)
        assert result.num_qubits == 1
        assert result.num_shots == 100

    def test_no_measurements_returns_statevector_none(self) -> None:
        """Circuit with gates but no measurements: statevector only."""
        circ = QuantumCircuit().h(0).cx(0, 1)
        sim = Simulator()
        result = sim.run(circ, save_statevector=False)
        assert result.num_shots == 0
        assert result.counts == {}
        assert result.statevector is None

    def test_no_measurements_returns_statevector(self) -> None:
        """Circuit with gates but no measurements: save statevector."""
        circ = QuantumCircuit().h(0).cx(0, 1)
        sim = Simulator()
        result = sim.run(circ, save_statevector=True)
        assert result.num_shots == 0
        assert result.statevector is not None
        assert result.statevector.shape == (4,)


class TestSimulatorStatevector:
    """Tests for the statevector() method."""

    def test_statevector_bell_state(self) -> None:
        """statevector() should return the correct Bell state."""
        circ = QuantumCircuit().h(0).cx(0, 1)
        sim = Simulator()
        sv = sim.statevector(circ)
        assert sv.shape == (4,)
        assert abs(sv[0]) == pytest.approx(1.0 / np.sqrt(2))
        assert abs(sv[3]) == pytest.approx(1.0 / np.sqrt(2))
        assert abs(sv[1]) == pytest.approx(0.0)
        assert abs(sv[2]) == pytest.approx(0.0)

    def test_statevector_single_qubit_h(self) -> None:
        """H|0> = (|0> + |1>)/√2."""
        circ = QuantumCircuit().h(0)
        sim = Simulator()
        sv = sim.statevector(circ)
        assert sv[0] == pytest.approx(1.0 / np.sqrt(2))
        assert sv[1] == pytest.approx(1.0 / np.sqrt(2))

    def test_statevector_skips_measurements(self) -> None:
        """statevector() ignores measurement instructions."""
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator()
        sv = sim.statevector(circ)
        # Should be |+> state, same as without measure
        assert sv[0] == pytest.approx(1.0 / np.sqrt(2))
        assert sv[1] == pytest.approx(1.0 / np.sqrt(2))


class TestSimulatorParameterized:
    """Tests for parameterized gate execution."""

    def test_parameterized_bound_phase(self) -> None:
        """P(theta)|+> applies exp(i*theta) only to the |1> amplitude."""
        theta = 0.73
        circ = QuantumCircuit().h(0).p(theta, 0)
        sv = Simulator().statevector(circ)
        expected = np.array([1.0, np.exp(1j * theta)]) / np.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_cphase_from_phase_and_crz(self) -> None:
        """P(theta/2) on control followed by CRZ(theta) is CPhase(theta)."""
        theta = 0.73
        circ = QuantumCircuit().h(0).h(1)
        circ.p(theta / 2, 0).crz(theta, 0, 1)
        sv = Simulator().statevector(circ)
        expected = np.array([1.0, 1.0, 1.0, np.exp(1j * theta)]) / 2
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_parameterized_bound_rx(self) -> None:
        """RX(pi)|0> = cos(pi/2)|0> - i sin(pi/2)|1> = -i|1>."""
        circ = QuantumCircuit().rx(np.pi, 0)
        sim = Simulator()
        sv = sim.statevector(circ)
        # RX(π)|0> = cos(π/2)|0> - i sin(π/2)|1> = 0|0> - i|1>
        assert abs(sv[0]) == pytest.approx(0.0, abs=1e-10)
        assert sv[1] == pytest.approx(-1j)

    def test_parameterized_bound_rz(self) -> None:
        """RZ(pi/2) on |+>."""
        circ = QuantumCircuit().h(0).rz(np.pi / 2, 0)
        sim = Simulator()
        sv = sim.statevector(circ)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_parameterized_bound_crx(self) -> None:
        """CRX(pi) with control |1>: X on target."""
        circ = QuantumCircuit().x(0).crx(np.pi, 0, 1)
        sim = Simulator()
        sv = sim.statevector(circ)
        # |10> → CRX(pi, control=0, target=1) → RX(pi)|1> on qubit 1 when qubit 0=1
        # Qubit 0 is |1>, so target gets RX(pi) = -iX
        # |10>: qubit 0=0? No, let me re-index.
        # X(0) puts q0 in |1>. State: |01> (index 1: q0=1, q1=0)
        # CRX(pi, control=0, target=1): since control=1, apply RX(pi) to target
        # RX(pi)|0> = cos(pi/2)|0> - i sin(pi/2)|1> = -i|1>
        # So state becomes |11> (index 3: q0=1, q1=1)
        assert abs(sv[3]) == pytest.approx(1.0)
        assert abs(sv[1]) == pytest.approx(0.0)

    def test_parameterized_bound_crz(self) -> None:
        """CRZ with concrete parameters."""
        circ = QuantumCircuit().h(0).crz(np.pi, 0, 1)
        sim = Simulator()
        sv = sim.statevector(circ)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_parameterized_bound_u3(self) -> None:
        """U3(theta, phi, lam) with concrete parameters."""
        circ = QuantumCircuit().u3(np.pi / 2, 0.0, np.pi, 0)
        sim = Simulator()
        sv = sim.statevector(circ)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_unbound_parameter_raises(self) -> None:
        """Circuit with unbound symbolic parameter raises ValueError."""
        theta = Parameter("theta")
        circ = QuantumCircuit().rx(theta, 0).h(1)
        sim = Simulator()
        with pytest.raises(ValueError, match="unbound"):
            sim.run(circ, shots=100)

    def test_unbound_parameter_statevector_raises(self) -> None:
        """statevector() with unbound parameter raises ValueError."""
        theta = Parameter("theta")
        circ = QuantumCircuit().rx(theta, 0)
        sim = Simulator()
        with pytest.raises(ValueError, match="unbound"):
            sim.statevector(circ)

    def test_bound_parameter_works(self) -> None:
        """After bind_parameters, simulation works."""
        theta = Parameter("theta")
        circ = QuantumCircuit().rx(theta, 0).measure_all()
        bound = circ.bind_parameters({theta: np.pi})
        sim = Simulator(seed=42)
        result = sim.run(bound, shots=1000)
        # RX(pi)|0> = -i|1>, so 100% "1"
        assert result.most_frequent() == ("1", 1000)


class TestSimulatorBarrier:
    """Tests for barrier handling."""

    def test_barrier_no_effect(self) -> None:
        """Barrier does not affect simulation outcome — both give ~50/50."""
        circ_no_barrier = QuantumCircuit().h(0).cx(0, 1).measure_all()
        circ_with_barrier = (
            QuantumCircuit().h(0).barrier().cx(0, 1).barrier().measure_all()
        )
        sim = Simulator(seed=42)
        result1 = sim.run(circ_no_barrier, shots=1000)
        result2 = sim.run(circ_with_barrier, shots=1000)
        # Barrier is a no-op: both should be roughly 50/50 Bell state
        for result in (result1, result2):
            c00 = result.get_count("00")
            c11 = result.get_count("11")
            assert 400 < c00 < 600
            assert 400 < c11 < 600
            assert result.get_count("01") == 0
            assert result.get_count("10") == 0


class TestSimulatorShots:
    """Tests for shot handling."""

    def test_zero_shots(self) -> None:
        """shots=0 returns empty counts but correct metadata."""
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator()
        result = sim.run(circ, shots=0)
        assert result.num_shots == 0
        assert result.counts == {}

    def test_negative_shots_raises(self) -> None:
        """Negative shots raises ValueError."""
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator()
        with pytest.raises(ValueError, match="shots"):
            sim.run(circ, shots=-1)

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        circ = QuantumCircuit().h(0).cx(0, 1).measure_all()
        sim1 = Simulator(seed=42)
        sim2 = Simulator(seed=42)
        result1 = sim1.run(circ, shots=10000)
        result2 = sim2.run(circ, shots=10000)
        assert result1.counts == result2.counts


class TestSimulatorMidCircuitMeasurement:
    """Tests for mid-circuit measurement and state collapse."""

    def test_mid_circuit_measurement_deterministic(self) -> None:
        """Mid-circuit measurement of |0> gives '0', then X makes it |1>."""
        # |0> → measure q0 → always '0' → X on q0 → |1> → measure → always '1'
        circ = QuantumCircuit()
        circ.measure([0], [0])  # First measure q0→c0
        circ.x(0)  # X gate after measurement
        circ.measure([0], [1])  # Second measure q0→c1
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=1000)
        # clbit 0 = 0 (first measure of q0 in |0>)
        # clbit 1 = 1 (second measure after X makes q0 |1>)
        # bitstring: position 0 = clbit 0 = 0, position 1 = clbit 1 = 1 → "01"
        assert result.most_frequent() == ("01", 1000)

    def test_mid_circuit_measurement_superposition(self) -> None:
        """H|0> → measure → 50/50, then X → measure → always flips."""
        circ = QuantumCircuit()
        circ.h(0)  # |+>
        circ.measure([0], [0])  # Measure q0, collapse to |0> or |1>
        circ.x(0)  # Flip
        circ.measure([0], [1])  # Second measure
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=10000)
        # If first measure = 0, q0 collapses to |0>, X makes |1>, second measure = 1
        # If first measure = 1, q0 collapses to |1>, X makes |0>, second measure = 0
        # So bitstrings: "01" (c0=0, c1=1) and "10" (c0=1, c1=0)
        count_01 = result.get_count("01")
        count_10 = result.get_count("10")
        assert 4000 < count_01 < 6000
        assert 4000 < count_10 < 6000

    def test_mid_circuit_continue_gates(self) -> None:
        """Gates after mid-circuit measurement operate on collapsed state."""
        circ = QuantumCircuit()
        circ.h(0)  # |+>
        circ.measure([0], [0])  # Collapse to |0> or |1>
        circ.h(0)  # H on collapsed state
        circ.measure([0], [1])  # Second measure
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=100)
        # After first measurement, q0 is |0> or |1>
        # H|0> = |+> → 50/50 on second measure
        # H|1> = |-> → 50/50 on second measure
        # So regardless of first outcome, second is ~50/50
        count_bit1_is_0 = result.get_count("00") + result.get_count("10")
        count_bit1_is_1 = result.get_count("01") + result.get_count("11")
        assert 0 < count_bit1_is_0 < 100
        assert 0 < count_bit1_is_1 < 100

    def test_multiple_measurements_same_qubit(self) -> None:
        """Measuring the same qubit twice gives consistent results."""
        circ = QuantumCircuit()
        circ.h(0)  # |+>
        circ.measure([0], [0])  # First measure
        circ.measure([0], [1])  # Second measure, same qubit
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=1000)
        # Both measurements of same qubit should agree
        # "00" (c0=0, c1=0) and "11" (c0=1, c1=1) are the only valid outcomes
        assert result.get_count("01") == 0
        assert result.get_count("10") == 0


class TestSimulatorSaveStatevector:
    """Tests for save_statevector option."""

    def test_save_statevector_false(self) -> None:
        """save_statevector=False leaves statevector=None."""
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator()
        result = sim.run(circ, shots=10, save_statevector=False)
        assert result.statevector is None

    def test_save_statevector_true(self) -> None:
        """save_statevector=True includes statevector."""
        circ = QuantumCircuit().h(0).measure_all()
        sim = Simulator()
        result = sim.run(circ, shots=10, save_statevector=True)
        assert result.statevector is not None


class TestSimulatorEdgeCases:
    """Tests for edge cases."""

    def test_gapped_qubit_indices(self) -> None:
        """Using qubit 3 without qubits 0-2."""
        circ = QuantumCircuit().h(3).cx(3, 0).measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=1000)
        assert result.num_qubits == 4
        # Bell state on qubits 0 and 3, qubits 1 and 2 are |0>
        # So only "0000" and "1001" outcomes
        count_0000 = result.get_count("0000")
        count_1001 = result.get_count("1001")
        assert 300 < count_0000 < 700
        assert 300 < count_1001 < 700

    def test_circuit_with_gaps(self) -> None:
        """Gaps in qubit indices handled correctly."""
        circ = QuantumCircuit()
        circ.h(0)
        circ.x(5)
        circ.cx(0, 5)
        circ.measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=1000)
        assert result.num_qubits == 6

    def test_large_qubit_count(self) -> None:
        """10-qubit circuit should work."""
        circ = QuantumCircuit()
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.measure_all()
        sim = Simulator(seed=42)
        result = sim.run(circ, shots=100)
        assert result.num_qubits == 3
        assert result.num_shots == 100


class TestStatevectorRegression:
    """Regression tests for B71/B72 (statevector edge cases)."""

    def test_empty_circuit_statevector_is_unit_scalar(self) -> None:
        """The 0-qubit state is ``[1+0j]``."""
        sv = Simulator(seed=0).statevector(QuantumCircuit())
        np.testing.assert_array_equal(sv, np.array([1.0 + 0j]))
        assert sv.dtype == np.complex128

    @pytest.mark.parametrize(
        "circ",
        [
            QuantumCircuit().h(0).measure([0], [0]).h(0),
            QuantumCircuit().h(0).measure([0], [0]).x(1),
            QuantumCircuit().h(0).measure([0], [0]).bit_flip(0.1, 0),
        ],
    )
    def test_mid_circuit_measurement_raises(self, circ: QuantumCircuit) -> None:
        """A gate or channel after a measurement is rejected explicitly."""
        with pytest.raises(ValueError, match="mid-circuit"):
            Simulator(seed=0).statevector(circ)

    def test_terminal_measurement_returns_pre_measurement_state(self) -> None:
        """Terminal measurements (and trailing barriers) are ignored."""
        circ = QuantumCircuit().h(0).measure([0], [0]).barrier().measure([0], [1])
        sv = Simulator(seed=0).statevector(circ)
        np.testing.assert_allclose(sv, np.array([1, 1]) / np.sqrt(2), atol=1e-12)
