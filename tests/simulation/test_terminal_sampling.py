"""Terminal measurements are sampled from the final state without re-simulation."""

from __future__ import annotations

import time

import numpy as np

from quspect import QuantumCircuit
from quspect.simulation import Simulator


def test_terminal_sampling_time_does_not_grow_with_shots_times_dimension() -> None:
    """16 qubits, 100 000 shots: one cumulative distribution, then a search per shot.

    Rebuilding the 2^16 prefix sum for every shot took several seconds here;
    the bound leaves a wide margin for slow machines.
    """
    circuit = QuantumCircuit()
    for qubit in range(16):
        circuit.h(qubit)
    circuit.measure_all()
    simulator = Simulator(seed=3)

    start = time.perf_counter()
    result = simulator.run(circuit, shots=100_000)
    elapsed = time.perf_counter() - start

    assert sum(result.counts.values()) == 100_000
    assert all(len(key) == 16 for key in result.counts)
    assert elapsed < 2.0


def test_terminal_sampling_follows_the_state_distribution() -> None:
    """Counts of many shots agree with |amplitude|^2 (clbit 0 leftmost)."""
    circuit = QuantumCircuit().ry(0.9, 0).ry(1.7, 1).cx(0, 2).rx(0.5, 1)
    simulator = Simulator(seed=11)
    amplitudes = simulator.statevector(circuit)
    shots = 400_000
    counts = simulator.run(circuit.copy().measure_all(), shots=shots).counts

    assert sum(counts.values()) == shots
    for basis, amplitude in enumerate(amplitudes):
        key = "".join(str((basis >> qubit) & 1) for qubit in range(3))
        probability = float(abs(amplitude) ** 2)
        sigma = np.sqrt(shots * probability * (1 - probability))
        assert abs(counts.get(key, 0) - shots * probability) <= 5 * sigma + 1


def test_terminal_sampling_respects_the_clbit_mapping() -> None:
    circuit = QuantumCircuit().x(0).id(1).id(2).measure([0, 2], [3, 1])
    counts = Simulator(seed=1).run(circuit, shots=1000).counts
    assert counts == {"0001": 1000}
