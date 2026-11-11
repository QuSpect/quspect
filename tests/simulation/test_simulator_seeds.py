"""Seeds of the simulators: the default restart, per-run seeds and checks."""

from __future__ import annotations

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.backends import ExecutionRequest
from quspect.ir.qsal import dumps
from quspect.native import NativeExecutor
from quspect.simulation import (
    DensityMatrixSimulator,
    MonteCarloSimulator,
    SimulationResult,
    Simulator,
)

_SIMULATORS = [Simulator, MonteCarloSimulator, DensityMatrixSimulator]
_STATEVECTOR_SIMULATORS = [Simulator, MonteCarloSimulator]


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


def _counts(simulator, circuit: QuantumCircuit, **options) -> dict[str, int]:
    return simulator.run(circuit, shots=2000, **options).counts


# ---------------------------------------------------------------------------
# Default: a seeded simulator restarts from its seed on every run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_seeded_simulator_repeats_every_run(kind: str, simulator_cls: type) -> None:
    circuit = _CIRCUITS[kind]()
    simulator = simulator_cls(seed=7)
    first = _counts(simulator, circuit)
    assert [_counts(simulator, circuit) for _ in range(3)] == [first] * 3
    assert _counts(simulator_cls(seed=7), circuit) == first


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
def test_seeded_run_is_the_native_executor_run(kind: str) -> None:
    """The counts of a seeded run are those of earlier releases."""
    circuit = _CIRCUITS[kind]()
    native = NativeExecutor(precision="double", seed=42).run(dumps(circuit), shots=500)
    expected = SimulationResult.from_native(
        native, circuit=circuit, shots=500, save_statevector=False
    )
    simulator = Simulator(seed=42)
    assert simulator.run(circuit, shots=500).counts == expected.counts
    assert simulator.run(circuit, shots=500).counts == expected.counts


def test_execute_batch_keeps_the_simulator_seed() -> None:
    request = ExecutionRequest.batch([_uniform(), _uniform()], shots=2000)
    batch = Simulator(seed=7).execute(request)
    single = _counts(Simulator(seed=7), _uniform())
    assert [result.counts for result in batch.results] == [single, single]


# ---------------------------------------------------------------------------
# An unseeded simulator draws new entropy for every run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_unseeded_simulator_runs_differ(kind: str, simulator_cls: type) -> None:
    circuit = _CIRCUITS[kind]()
    simulator = simulator_cls()
    runs = [_counts(simulator, circuit) for _ in range(4)]
    # Eight outcomes at 2000 shots: two equal runs are a ~1e-7 coincidence.
    assert any(counts != runs[0] for counts in runs[1:])


def test_unseeded_simulator_reports_no_seed() -> None:
    simulator = Simulator()
    simulator.run(_uniform(), shots=10)
    assert simulator.seed is None


# ---------------------------------------------------------------------------
# run(..., seed=): the seed of one run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_CIRCUITS))
@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_run_seed_is_the_seed_of_that_run(kind: str, simulator_cls: type) -> None:
    circuit = _CIRCUITS[kind]()
    expected = _counts(simulator_cls(seed=11), circuit)
    assert _counts(simulator_cls(seed=7), circuit, seed=11) == expected
    assert _counts(simulator_cls(), circuit, seed=11) == expected
    assert expected != _counts(simulator_cls(seed=7), circuit)


@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_run_seed_leaves_the_simulator_seed_alone(simulator_cls: type) -> None:
    circuit = _uniform()
    simulator = simulator_cls(seed=7)
    default = _counts(simulator, circuit)
    assert _counts(simulator, circuit, seed=11) != default
    assert simulator.seed == 7
    assert _counts(simulator, circuit) == default


@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_run_seeds_give_different_samples(simulator_cls: type) -> None:
    simulator = simulator_cls(seed=7)
    runs = [_counts(simulator, _uniform(), seed=k) for k in range(4)]
    for index, counts in enumerate(runs):
        assert counts not in runs[:index]


def test_run_seed_reaches_a_saved_trajectory_state() -> None:
    circuit = _mid_circuit()
    simulator = Simulator(seed=7)
    states = [
        simulator.run(circuit, shots=1, save_statevector=True, seed=k).statevector
        for k in range(8)
    ]
    again = simulator.run(circuit, shots=1, save_statevector=True, seed=3).statevector
    np.testing.assert_array_equal(again, states[3])
    assert any(not np.array_equal(state, states[0]) for state in states[1:])


def test_run_seed_from_request_options() -> None:
    request = ExecutionRequest.single(_uniform(), shots=2000, options={"seed": 11})
    batch = Simulator(seed=7).execute(request)
    assert batch.results[0].counts == _counts(Simulator(seed=11), _uniform())


def test_native_executor_run_seed() -> None:
    qsal = dumps(_uniform())
    executor = NativeExecutor(precision="double", seed=7)
    default = executor.run(qsal, shots=500).counts
    assert executor.run(qsal, shots=500).counts == default
    other = executor.run(qsal, shots=500, seed=11).counts
    assert other == NativeExecutor(seed=11).run(qsal, shots=500).counts
    assert other != default
    assert executor.run(qsal, shots=500).counts == default


def test_statevector_needs_no_seed() -> None:
    circuit = QuantumCircuit().h(0).cx(0, 1)
    expected = Simulator(seed=7).statevector(circuit)
    np.testing.assert_array_equal(Simulator().statevector(circuit), expected)
    np.testing.assert_array_equal(Simulator(seed=8).statevector(circuit), expected)


# ---------------------------------------------------------------------------
# Assigning the seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_assigned_seed_is_used(simulator_cls: type) -> None:
    circuit = _uniform()
    simulator = simulator_cls(seed=7)
    simulator.seed = 11
    assert simulator.seed == 11
    assert _counts(simulator, circuit) == _counts(simulator_cls(seed=11), circuit)
    simulator.seed = None
    runs = [_counts(simulator, circuit) for _ in range(4)]
    assert any(counts != runs[0] for counts in runs[1:])


# ---------------------------------------------------------------------------
# Seed checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
@pytest.mark.parametrize("seed", [-1, -(2**70)])
def test_negative_seed_is_refused(simulator_cls: type, seed: int) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        simulator_cls(seed=seed)
    simulator = simulator_cls(seed=7)
    with pytest.raises(ValueError, match="non-negative integer"):
        simulator.run(_uniform(), shots=10, seed=seed)
    with pytest.raises(ValueError, match="non-negative integer"):
        simulator.seed = seed
    assert simulator.seed == 7


@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
@pytest.mark.parametrize("seed", [1.5, "7", True, [1, 2]])
def test_non_integer_seed_is_refused(simulator_cls: type, seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be a non-negative integer or None"):
        simulator_cls(seed=seed)
    with pytest.raises(TypeError, match="seed must be a non-negative integer or None"):
        simulator_cls().run(_uniform(), shots=10, seed=seed)


@pytest.mark.parametrize("simulator_cls", _STATEVECTOR_SIMULATORS)
def test_seed_must_fit_64_bits(simulator_cls: type) -> None:
    assert simulator_cls(seed=2**64 - 1).run(_uniform(), shots=10).num_shots == 10
    with pytest.raises(ValueError, match=r"below 2\*\*64"):
        simulator_cls(seed=2**64)
    with pytest.raises(ValueError, match=r"below 2\*\*64"):
        simulator_cls().run(_uniform(), shots=10, seed=2**64)


def test_density_simulator_takes_wide_seeds() -> None:
    wide = DensityMatrixSimulator(seed=2**70)
    assert _counts(wide, _uniform()) == _counts(wide, _uniform())


@pytest.mark.parametrize("simulator_cls", _SIMULATORS)
def test_numpy_integer_seeds_are_accepted(simulator_cls: type) -> None:
    circuit = _uniform()
    simulator = simulator_cls(seed=np.int64(7))
    assert simulator.seed == 7
    assert type(simulator.seed) is int
    expected = _counts(simulator_cls(seed=7), circuit)
    assert _counts(simulator, circuit) == expected
    assert _counts(simulator_cls(), circuit, seed=np.uint64(7)) == expected
