"""Seeded simulators give a reproducible sequence of different runs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.backends import ExecutionRequest
from quspect.circuit.parameter import Parameter
from quspect.ir.qsal import dumps
from quspect.native import NativeExecutor
from quspect.operators import QubitOperator
from quspect.simulation import (
    DensityMatrixSimulator,
    MonteCarloSimulator,
    SimulationResult,
    Simulator,
)
from quspect.variational import ObservableObjective, SamplingEngine


def _uniform(num_qubits: int = 3) -> QuantumCircuit:
    circuit = QuantumCircuit()
    for qubit in range(num_qubits):
        circuit.h(qubit)
    return circuit.measure_all()


def _noisy() -> QuantumCircuit:
    circuit = QuantumCircuit().h(0).h(1).h(2)
    circuit.depolarize(0.2, 0)
    return circuit.measure_all()


def _mid_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit().h(0).h(1)
    circuit.measure([0, 1], [0, 1])
    return circuit.h(0).measure([0], [2])


_CIRCUITS = {"terminal": _uniform, "noisy": _noisy, "mid_circuit": _mid_circuit}


def _runs(
    simulator: Simulator | DensityMatrixSimulator, circuit: QuantumCircuit
) -> list[dict[str, int]]:
    return [simulator.run(circuit, shots=2000).counts for _ in range(4)]


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
@pytest.mark.parametrize("simulator_cls", [Simulator, MonteCarloSimulator])
def test_seeded_simulator_runs_differ(kind: str, simulator_cls: type) -> None:
    runs = _runs(simulator_cls(seed=7), _CIRCUITS[kind]())
    for index, counts in enumerate(runs):
        assert counts not in runs[:index]


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
def test_seeded_simulator_sequence_is_reproducible(kind: str) -> None:
    circuit = _CIRCUITS[kind]()
    assert _runs(Simulator(seed=7), circuit) == _runs(Simulator(seed=7), circuit)
    assert _runs(Simulator(seed=7), circuit) != _runs(Simulator(seed=8), circuit)


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
def test_first_run_uses_the_seed_itself(kind: str) -> None:
    """One-shot results of a seeded simulator are those of earlier releases."""
    circuit = _CIRCUITS[kind]()
    native = NativeExecutor(precision="double", seed=42).run(dumps(circuit), shots=500)
    expected = SimulationResult.from_native(
        native, circuit=circuit, shots=500, save_statevector=False
    )
    assert Simulator(seed=42).run(circuit, shots=500).counts == expected.counts


def test_unseeded_simulator_runs_differ() -> None:
    runs = _runs(Simulator(), _uniform())
    assert runs[0] != runs[1] or runs[1] != runs[2]


def test_statevector_does_not_advance_the_stream() -> None:
    circuit = _uniform()
    simulator = Simulator(seed=5)
    simulator.statevector(QuantumCircuit().h(0).cx(0, 1))
    first = simulator.run(circuit, shots=500).counts
    assert first == Simulator(seed=5).run(circuit, shots=500).counts


def test_rejected_run_does_not_advance_the_stream() -> None:
    circuit = _uniform()
    simulator = Simulator(seed=5)
    with pytest.raises(ValueError, match="shots"):
        simulator.run(circuit, shots=-1)
    first = simulator.run(circuit, shots=500).counts
    assert first == Simulator(seed=5).run(circuit, shots=500).counts


def test_assigning_seed_restarts_the_stream() -> None:
    circuit = _uniform()
    simulator = Simulator(seed=5)
    first = simulator.run(circuit, shots=500).counts
    simulator.seed = 5
    assert simulator.run(circuit, shots=500).counts == first
    density = DensityMatrixSimulator(seed=5)
    first = density.run(circuit, shots=500).counts
    density.seed = 5
    assert density.seed == 5
    assert density.run(circuit, shots=500).counts == first


def test_batch_experiments_use_successive_streams() -> None:
    circuit = _uniform()
    request = ExecutionRequest.batch(
        [circuit, circuit, circuit], labels=["a", "b", "c"], shots=2000
    )
    batch = Simulator(seed=7).execute(request)
    again = Simulator(seed=7).execute(request)
    assert batch.get_counts() == again.get_counts()
    assert batch.get_counts() == _runs(Simulator(seed=7), circuit)[:3]
    assert batch[0].counts != batch[1].counts
    assert batch[1].counts != batch[2].counts


def test_density_simulator_runs_differ_and_reproduce() -> None:
    circuit = _noisy()
    runs = _runs(DensityMatrixSimulator(seed=3), circuit)
    for index, counts in enumerate(runs):
        assert counts not in runs[:index]
    assert runs == _runs(DensityMatrixSimulator(seed=3), circuit)


def test_density_simulator_first_run_is_unchanged() -> None:
    """The first run still draws ``default_rng(seed).multinomial(shots, p)``."""
    circuit = QuantumCircuit().h(0).h(1).measure_all()
    expected = np.random.default_rng(11).multinomial(1000, [0.25] * 4)
    counts = DensityMatrixSimulator(seed=11).run(circuit, shots=1000).counts
    assert sorted(counts.values()) == sorted(expected.tolist())


def test_sampling_engine_estimates_are_not_repeated() -> None:
    theta = Parameter("theta")
    objective = ObservableObjective(
        QuantumCircuit().ry(theta, 0),
        {"x": QubitOperator.from_pauli_string("X")},
    )
    point = {theta: 0.4}

    def estimates() -> list[float]:
        prepared = SamplingEngine(Simulator(seed=7), shots=1024).compile(objective)
        return [prepared.evaluate(point).value() for _ in range(4)]

    values = estimates()
    assert len(set(values)) == len(values)
    assert values == estimates()
    assert values == pytest.approx([math.sin(0.4)] * 4, abs=0.15)


@pytest.mark.parametrize("simulator_cls", [Simulator, DensityMatrixSimulator])
def test_seed_accepts_integer_like_values(simulator_cls: type) -> None:
    simulator = simulator_cls(seed=np.int64(9))
    assert simulator.seed == 9
    assert type(simulator.seed) is int
    circuit = _uniform()
    plain = simulator_cls(seed=9).run(circuit, shots=200).counts
    assert simulator.run(circuit, shots=200).counts == plain


@pytest.mark.parametrize("simulator_cls", [Simulator, DensityMatrixSimulator])
@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_seed_of_wrong_type_is_refused(simulator_cls: type, seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be a non-negative integer or None"):
        simulator_cls(seed=seed)


@pytest.mark.parametrize("simulator_cls", [Simulator, DensityMatrixSimulator])
def test_negative_seed_is_refused(simulator_cls: type) -> None:
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        simulator_cls(seed=-1)


def test_simulator_seed_must_fit_64_bits() -> None:
    assert Simulator(seed=2**64 - 1).seed == 2**64 - 1
    with pytest.raises(ValueError, match="below 2\\*\\*64"):
        Simulator(seed=2**64)


def test_native_executor_run_accepts_a_per_call_seed() -> None:
    qsal = dumps(_uniform())
    executor = NativeExecutor(precision="double", seed=1)
    default = executor.run(qsal, shots=500).counts
    assert executor.run(qsal, shots=500).counts == default
    other = NativeExecutor(precision="double", seed=5).run(qsal, shots=500).counts
    assert executor.run(qsal, shots=500, seed=5).counts == other
    assert other != default
