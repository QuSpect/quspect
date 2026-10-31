"""Cross-validate ``quspect.simulation.density_simulator.DensityMatrixSimulator``.

* ``density_matrix(circ)`` is compared element-wise with qiskit
  ``DensityMatrix`` evolution of ``to_qiskit(circ, noise="kraus")``.
* ``run(circ, shots)`` counts are compared with the exact qiskit distribution
  via :func:`qiskit_tests.helpers.assert_counts_match`.

``run`` semantics: keys follow the circuit's measure -> clbit map like
``Simulator.run`` (``num_clbits`` characters, clbit 0 leftmost) and mid-circuit
measurements are branched exactly.  Differences from ``Simulator.run`` that are
pinned by normal tests: all ``2^num_clbits`` keys are present (zero counts
included), and a circuit without any measurement is sampled as if every qubit
``q`` were measured into clbit ``q``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix

import quspect
from qiskit_tests._noise_helpers import channel_catalogue, nonzero
from qiskit_tests.helpers import (
    assert_counts_match,
    qiskit_probabilities,
    random_circuit_pair,
    to_qiskit,
)
from quspect.noise import depolarizing_model
from quspect.simulation import DensityMatrixSimulator, SimulationResult, Simulator

pytestmark = pytest.mark.qiskit_ref

SHOTS = 10000


def _with_all_channels(circ: quspect.QuantumCircuit, seed: int) -> None:
    """Append every catalogue channel at random (ordered) qubits of ``circ``."""
    rng = np.random.default_rng(seed)
    n = circ.num_qubits
    for ch in channel_catalogue().values():
        qubits = [int(q) for q in rng.permutation(n)[: ch.num_qubits]]
        circ.append_channel(ch, qubits)
        th = float(rng.uniform(-3, 3))
        circ.ry(th, int(rng.integers(n)))


def _rho(circ: quspect.QuantumCircuit) -> np.ndarray:
    """quspect density matrix as a numpy array."""
    return np.asarray(DensityMatrixSimulator(seed=0).density_matrix(circ)._data)


# =============================================================================
# density_matrix()
# =============================================================================


def test_name_and_seed() -> None:
    """Backend name and stored seed."""
    sim = DensityMatrixSimulator(seed=5)
    assert sim.name == "density_matrix_simulator"
    assert sim.seed == 5
    assert DensityMatrixSimulator().seed is None


@pytest.mark.parametrize(("n", "seed"), [(2, 0), (3, 1), (3, 2), (4, 3)])
def test_density_matrix_matches_qiskit(n: int, seed: int) -> None:
    """Random gates + every channel type on shuffled qubits: rho to 1e-10."""
    circ, _ = random_circuit_pair(n, 15, 900 + seed)
    _with_all_channels(circ, seed)
    circ.cx(0, n - 1).h(n - 1)
    ref = QiskitDensityMatrix(to_qiskit(circ, noise="kraus")).data
    np.testing.assert_allclose(_rho(circ), ref, atol=1e-10, rtol=0)


def test_density_matrix_noiseless_equals_statevector() -> None:
    """Without noise rho == |psi><psi| of Simulator.statevector."""
    circ, _ = random_circuit_pair(3, 30, 4)
    sv = Simulator(seed=0).statevector(circ)
    np.testing.assert_allclose(_rho(circ), np.outer(sv, sv.conj()), atol=1e-12)


def test_density_matrix_ignores_measurements() -> None:
    """density_matrix() skips measurements (documented: no collapse)."""
    circ = quspect.QuantumCircuit().h(0).measure([0], [0]).cx(0, 1)
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(_rho(circ), np.outer(bell, bell), atol=1e-12)


# =============================================================================
# run(): statistics against the exact distribution
# =============================================================================


def _bell() -> quspect.QuantumCircuit:
    return quspect.QuantumCircuit().h(0).cx(0, 1)


def _ghz() -> quspect.QuantumCircuit:
    return quspect.QuantumCircuit().h(0).cx(0, 1).cx(1, 2)


def _random3() -> quspect.QuantumCircuit:
    circ, _ = random_circuit_pair(3, 20, 31)
    return circ


@pytest.mark.statistical
@pytest.mark.parametrize("name", ["bell", "ghz", "random3"])
def test_run_counts_match_exact_distribution(name: str) -> None:
    """depolarizing_model + measure_all: counts ~ qiskit exact probabilities."""
    base = {"bell": _bell, "ghz": _ghz, "random3": _random3}[name]()
    noisy = depolarizing_model(0.04, 0.12).apply(base)
    noisy.amplitude_damp(0.2, 0).measure_all()
    result = DensityMatrixSimulator(seed=21).run(noisy, shots=SHOTS)
    assert result.num_shots == SHOTS
    assert_counts_match(nonzero(result.counts), qiskit_probabilities(noisy), SHOTS)


@pytest.mark.statistical
def test_run_and_simulator_agree_for_measure_all() -> None:
    """With measure_all both backends use the same keys and distribution."""
    noisy = depolarizing_model(0.05, 0.1).apply(_ghz()).measure_all()
    exact = qiskit_probabilities(noisy)
    dm = DensityMatrixSimulator(seed=3).run(noisy, shots=SHOTS)
    sv = Simulator(seed=3).run(noisy, shots=SHOTS)
    assert_counts_match(nonzero(dm.counts), exact, SHOTS)
    assert_counts_match(sv.counts, exact, SHOTS)


def test_run_result_fields_and_zero_padding() -> None:
    """All 2^n keys present (zero counts included), sorted; probabilities = f."""
    circ = quspect.QuantumCircuit().x(0).id(2).measure_all()
    result = DensityMatrixSimulator(seed=1).run(circ, shots=100)
    assert isinstance(result, SimulationResult)
    assert list(result.counts) == sorted(
        format(i, "03b") for i in range(8)
    )  # every key, sorted
    assert result.counts["100"] == 100
    assert sum(result.counts.values()) == 100
    assert result.probabilities["100"] == pytest.approx(1.0)
    assert result.probabilities["000"] == 0.0
    assert result.num_qubits == 3 and result.num_clbits == 3
    assert result.statevector is None


def test_run_zero_shots() -> None:
    """shots=0: no sampling, all counts zero."""
    result = DensityMatrixSimulator(seed=1).run(_bell(), shots=0)
    assert result.num_shots == 0
    assert set(result.counts.values()) == {0}
    assert len(result.counts) == 4


def test_run_seed_reproducible() -> None:
    """Same seed -> identical counts; unseeded run still samples correctly."""
    noisy = depolarizing_model(0.1, 0.1).apply(_ghz()).measure_all()
    a = DensityMatrixSimulator(seed=9).run(noisy, shots=500).counts
    b = DensityMatrixSimulator(seed=9).run(noisy, shots=500).counts
    c = DensityMatrixSimulator(seed=10).run(noisy, shots=500).counts
    assert a == b
    assert a != c
    assert sum(DensityMatrixSimulator().run(noisy, shots=50).counts.values()) == 50


def test_run_without_measurements_samples_all_qubits() -> None:
    """No measure: DM run still samples every qubit (Simulator returns {})."""
    circ = quspect.QuantumCircuit().x(1).id(2)
    dm = DensityMatrixSimulator(seed=1).run(circ, shots=10)
    assert nonzero(dm.counts) == {"010": 10}
    sv = Simulator(seed=1).run(circ, shots=10)
    assert sv.counts == {} and sv.num_shots == 0


# =============================================================================
# run(): key semantics vs Simulator.run (bugs)
# =============================================================================


def _permuted_clbits() -> quspect.QuantumCircuit:
    circ = quspect.QuantumCircuit().x(0).id(2)
    circ.measure([0, 2], [1, 0])  # c0 <- q2 = 0, c1 <- q0 = 1
    return circ


def _partial_measure() -> quspect.QuantumCircuit:
    return quspect.QuantumCircuit().x(1).h(0).measure([1], [0])  # c0 <- q1 = 1


def _swapped_all() -> quspect.QuantumCircuit:
    circ = quspect.QuantumCircuit().x(0).bit_flip(0.0, 1)
    circ.measure([0, 1], [1, 0])
    return circ


@pytest.mark.parametrize(
    "factory",
    [_permuted_clbits, _partial_measure, _swapped_all],
    ids=["permuted_clbits", "partial_measure", "swapped_all"],
)
def test_run_keys_follow_clbits_like_simulator(
    factory: Callable[[], quspect.QuantumCircuit],
) -> None:
    """Deterministic circuits: DM counts (nonzero) == Simulator counts."""
    circ = factory()
    expected = Simulator(seed=1).run(circ, shots=200).counts
    assert expected == {k: 200 for k, v in qiskit_probabilities(circ).items() if v}
    result = DensityMatrixSimulator(seed=1).run(circ, shots=200)
    assert nonzero(result.counts) == expected
    assert result.num_clbits == circ.num_clbits


@pytest.mark.statistical
def test_run_mid_circuit_measurement_distribution() -> None:
    """x, AD(0.3), measure q0, cx(0,1), measure q1: P(11)=0.7, P(00)=0.3."""
    circ = quspect.QuantumCircuit().x(0).amplitude_damp(0.3, 0)
    circ.measure([0], [0]).cx(0, 1).measure([1], [1])
    result = DensityMatrixSimulator(seed=4).run(circ, shots=SHOTS)
    assert_counts_match(nonzero(result.counts), {"00": 0.3, "11": 0.7}, SHOTS)


def test_run_empty_circuit() -> None:
    """A 0-qubit circuit yields the single empty bitstring for every shot."""
    result = DensityMatrixSimulator(seed=1).run(quspect.QuantumCircuit(), shots=5)
    assert result.counts == {"": 5}
    assert result.num_qubits == 0
