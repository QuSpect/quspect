"""Regression tests for native C++ noisy-trajectory execution."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.noise.channel import KrausChannel
from quspect.simulation.monte_carlo_simulator import MonteCarloSimulator
from quspect.simulation.simulator import Simulator


def _append_surface_stabilizer_measurements(circuit: QuantumCircuit) -> None:
    """Append the four distance-three Z stabilizer measurements."""
    circuit.cx(0, 9).cx(1, 9).cx(3, 9).cx(4, 9)
    circuit.cx(1, 10).cx(2, 10).cx(4, 10).cx(5, 10)
    circuit.cx(3, 11).cx(4, 11).cx(6, 11).cx(7, 11)
    circuit.cx(4, 12).cx(5, 12).cx(7, 12).cx(8, 12)
    circuit.measure([9, 10, 11, 12], [0, 1, 2, 3])


class TestNativeNoiseRandomness:
    """Native trajectories use one advancing RNG stream per shot."""

    def test_consecutive_half_bit_flips_are_independent(self) -> None:
        circuit = QuantumCircuit(name="independent_noise")
        circuit.bit_flip(0.5, 0).bit_flip(0.5, 1)
        circuit.measure([0, 1], [0, 1])

        result = Simulator(seed=20260810).run(circuit, shots=4000)

        assert set(result.counts) == {"00", "01", "10", "11"}
        assert all(700 < result.get_count(outcome) < 1300 for outcome in result.counts)

    def test_noise_seed_is_reproducible(self) -> None:
        circuit = QuantumCircuit(name="reproducible_noise")
        circuit.h(0).bit_flip(0.37, 0).depolarize(0.41, 1)
        circuit.measure([0, 1], [0, 1])

        first = Simulator(seed=123456).run(circuit, shots=1000)
        second = Simulator(seed=123456).run(circuit, shots=1000)

        assert first.counts == second.counts


class TestNativeNoiseQubitOrder:
    """Two-qubit gates and generic Kraus maps respect logical qubit order."""

    @pytest.mark.parametrize(
        ("prepared_qubit", "expected"),
        [
            (0, "11"),
            (1, "01"),
        ],
    )
    def test_monte_carlo_cnot_direction(
        self, prepared_qubit: int, expected: str
    ) -> None:
        circuit = QuantumCircuit(name="mc_cnot_direction")
        circuit.x(prepared_qubit).bit_flip(0.0, 0).cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        result = MonteCarloSimulator(seed=7).run(circuit, shots=32)

        assert result.counts == {expected: 32}

    def test_asymmetric_two_qubit_custom_kraus_order(self) -> None:
        # Conventional big-endian |q0 q1> matrix: q0 controls q1.
        cnot = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        channel = KrausChannel(name="custom_cnot", num_qubits=2, kraus_ops=[cnot])
        circuit = QuantumCircuit(name="custom_kraus_order")
        circuit.x(0).append_channel(channel, (0, 1))
        circuit.measure([0, 1], [0, 1])

        result = Simulator(seed=11).run(circuit, shots=32)

        assert result.counts == {"11": 32}

    def test_amplitude_damping_uses_generic_kraus_path(self) -> None:
        circuit = QuantumCircuit(name="amplitude_damping")
        circuit.x(0).amplitude_damp(1.0, 0).measure([0], [0])

        result = Simulator(seed=17).run(circuit, shots=32)

        assert result.counts == {"0": 32}


class TestSurfaceSyndrome:
    """Noisy data qubits produce four correctly mapped syndrome bits."""

    @pytest.mark.parametrize(
        ("data_qubit", "expected_syndrome"),
        [
            (0, "1000"),
            (1, "1100"),
            (2, "0100"),
            (3, "1010"),
            (4, "1111"),
            (5, "0101"),
            (6, "0010"),
            (7, "0011"),
            (8, "0001"),
        ],
    )
    def test_single_data_bit_flip_syndrome(
        self, data_qubit: int, expected_syndrome: str
    ) -> None:
        circuit = QuantumCircuit(name=f"surface_x{data_qubit}")
        circuit.bit_flip(1.0, data_qubit)
        _append_surface_stabilizer_measurements(circuit)

        result = Simulator(seed=19).run(circuit, shots=8)

        assert result.num_clbits == 4
        assert result.counts == {expected_syndrome: 8}

    def test_depolarizing_two_percent_smoke(self) -> None:
        circuit = QuantumCircuit(name="d3_surface_noisy")
        for data_qubit in range(9):
            circuit.depolarize(0.02, data_qubit)
        _append_surface_stabilizer_measurements(circuit)

        result = Simulator(seed=20260810).run(circuit, shots=512)

        assert result.num_qubits == 13
        assert result.num_clbits == 4
        assert sum(result.counts.values()) == 512
        assert all(len(syndrome) == 4 for syndrome in result.counts)
        assert result.get_probability("0000") == pytest.approx(0.913522982824, abs=0.07)
        assert any(syndrome != "0000" for syndrome in result.counts)
