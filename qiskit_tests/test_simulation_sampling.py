"""Statistical cross-validation of ``Simulator.run`` counts against qiskit.

Reference distributions are exact: ``qiskit_probabilities`` (``Statevector``
/ ``DensityMatrix``) for terminal measurements, and the deferred-measurement
construction of :mod:`qiskit_tests._sim_helpers` (each mid-circuit
``measure(q -> c)`` becomes ``cx(q, ancilla)`` + terminal ancilla
measurement) for mid-circuit measurements.  Counts are judged with
``assert_counts_match`` (5-sigma per-outcome binomial tolerance + chi-square
p-value > 1e-6) at ``SHOTS = 4000``.
"""

from __future__ import annotations

import math

import pytest

from qiskit_tests._sim_helpers import deferred_probabilities
from qiskit_tests.helpers import (
    assert_counts_match,
    qiskit_probabilities,
    random_circuit_pair,
)
from quspect import QuantumCircuit
from quspect.simulation import MonteCarloSimulator, Simulator

SHOTS = 4000


def _check(circ: QuantumCircuit, expected: dict[str, float], seed: int) -> None:
    res = Simulator(seed=seed).run(circ, shots=SHOTS)
    assert res.num_shots == SHOTS
    assert sum(res.counts.values()) == SHOTS
    assert all(len(k) == circ.num_clbits for k in res.counts)
    for key, cnt in res.counts.items():
        assert res.probabilities[key] == pytest.approx(cnt / SHOTS)
    assert_counts_match(res.counts, expected, SHOTS)


# =============================================================================
# Terminal measurements
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize(("n", "seed"), [(2, 11), (3, 12), (4, 13), (5, 14)])
def test_terminal_measure_all_random_circuit(n: int, seed: int) -> None:
    """``measure_all`` counts follow the exact qiskit distribution."""
    circ, _ = random_circuit_pair(n, 6 * n, seed)
    circ.measure_all()
    _check(circ, qiskit_probabilities(circ), seed)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_terminal_partial_permuted_and_gapped_clbits() -> None:
    """Partial measurement into permuted / gapped clbits (unmeasured read 0)."""
    circ, _ = random_circuit_pair(4, 24, 21)
    circ.measure([3, 0, 2], [0, 4, 1])
    expected = qiskit_probabilities(circ)
    assert all(k[2] == "0" and k[3] == "0" for k in expected)
    _check(circ, expected, 21)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_terminal_biased_single_qubit() -> None:
    """A strongly biased outcome (p = sin^2(0.15)) is reproduced."""
    circ = QuantumCircuit().ry(0.3, 0).h(1).measure([0, 1], [0, 1])
    expected = qiskit_probabilities(circ)
    assert expected["10"] == pytest.approx(0.5 * math.sin(0.15) ** 2)
    _check(circ, expected, 22)


def test_deterministic_outcome_exact() -> None:
    """A basis-state circuit yields a single outcome with all shots."""
    circ = QuantumCircuit().x(0).x(2).measure_all()
    res = Simulator(seed=0).run(circ, shots=257)
    assert res.counts == {"101": 257}
    assert res.probabilities == {"101": 1.0}
    assert res.num_qubits == 3 and res.num_clbits == 3


def test_measurement_zero_shots() -> None:
    """``shots=0`` on a measured circuit returns no counts."""
    res = Simulator(seed=0).run(QuantumCircuit().h(0).measure_all(), shots=0)
    assert res.num_shots == 0
    assert res.counts == {}


# =============================================================================
# Mid-circuit measurements (deferred-measurement reference)
# =============================================================================


def _mid_circuit_cases() -> list[tuple[str, QuantumCircuit]]:
    cases: list[tuple[str, QuantumCircuit]] = []

    # Measured qubit never used again -> equivalent to a terminal measurement.
    c1 = QuantumCircuit().h(0).cx(0, 1).measure([0], [0]).ry(0.7, 1).h(2)
    c1.cx(1, 2).measure([1, 2], [1, 2])
    cases.append(("unused_after", c1))

    # Measured qubit is re-used: measurement destroys coherence.
    c2 = QuantumCircuit().h(0).measure([0], [0]).h(0).measure([0], [1])
    cases.append(("h_measure_h", c2))

    # Re-used as a control after measurement, plus a rotation.
    c3 = QuantumCircuit().ry(1.1, 0).measure([0], [0]).cx(0, 1).ry(0.4, 0)
    c3.measure([0, 1], [1, 2])
    cases.append(("control_after", c3))

    # Same clbit overwritten (last write wins).
    c4 = QuantumCircuit().h(0).measure([0], [0]).x(0).ry(0.5, 0).measure([0], [0])
    c4.h(1).measure([1], [1])
    cases.append(("overwrite_clbit", c4))

    # Random circuits with a measurement layer in the middle.
    for seed in (31, 32):
        first, _ = random_circuit_pair(3, 10, seed)
        second, _ = random_circuit_pair(3, 10, seed + 50)
        circ = first.copy()
        circ.measure([1], [3])
        circ.compose(second, [0, 1, 2], inplace=True)
        circ.measure([0, 1, 2], [0, 1, 2])
        cases.append((f"random_{seed}", circ))
    return cases


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize(
    "case", _mid_circuit_cases(), ids=[name for name, _ in _mid_circuit_cases()]
)
def test_mid_circuit_measurement_matches_deferred_reference(
    case: tuple[str, QuantumCircuit],
) -> None:
    """Mid-circuit counts match the exact deferred-measurement distribution."""
    _, circ = case
    _check(circ, deferred_probabilities(circ), 40)


@pytest.mark.qiskit_ref
def test_mid_circuit_unused_qubit_equals_terminal() -> None:
    """Deferred-measurement principle: unused-after == terminal measurement."""
    mid = QuantumCircuit().h(0).cx(0, 1).measure([0], [0]).ry(0.7, 1)
    mid.measure([1], [1])
    term = QuantumCircuit().h(0).cx(0, 1).ry(0.7, 1).measure([0, 1], [0, 1])
    ref_mid = deferred_probabilities(mid)
    ref_term = qiskit_probabilities(term)
    for key in set(ref_mid) | set(ref_term):
        assert ref_mid.get(key, 0.0) == pytest.approx(ref_term.get(key, 0.0))


@pytest.mark.qiskit_ref
def test_h_measure_h_reference_is_uniform() -> None:
    """Sanity check of the reference: H-M-H-M gives four equal outcomes."""
    circ = QuantumCircuit().h(0).measure([0], [0]).h(0).measure([0], [1])
    ref = deferred_probabilities(circ)
    assert ref == pytest.approx({"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25})


def test_mid_circuit_save_statevector_is_a_trajectory_state() -> None:
    """With mid-circuit measurement the saved state is a collapsed state."""
    circ = QuantumCircuit().h(0).measure([0], [0]).x(1).measure([1], [1])
    res = Simulator(seed=3).run(circ, shots=10, save_statevector=True)
    assert res.statevector is not None
    probs = abs(res.statevector) ** 2
    assert probs.max() == pytest.approx(1.0)


# =============================================================================
# Seeding
# =============================================================================


def _uniform5() -> QuantumCircuit:
    circ = QuantumCircuit()
    for q in range(5):
        circ.h(q)
    return circ.measure_all()


@pytest.mark.parametrize("mid", [False, True])
def test_same_seed_reproducible(mid: bool) -> None:
    """Same seed -> identical counts (terminal and mid-circuit paths)."""
    circ = _uniform5()
    if mid:
        circ.h(0).measure([0], [5])
    a = Simulator(seed=99).run(circ, shots=500).counts
    b = Simulator(seed=99).run(circ, shots=500).counts
    assert a == b


@pytest.mark.parametrize("mid", [False, True])
def test_different_seeds_differ(mid: bool) -> None:
    """Different seeds give different samples (32 outcomes, 500 shots)."""
    circ = _uniform5()
    if mid:
        circ.h(0).measure([0], [5])
    a = Simulator(seed=1).run(circ, shots=500).counts
    b = Simulator(seed=2).run(circ, shots=500).counts
    assert a != b


def test_unseeded_simulators_differ() -> None:
    """``seed=None`` draws system entropy, so two simulators differ."""
    circ = _uniform5()
    a = Simulator().run(circ, shots=500).counts
    b = Simulator().run(circ, shots=500).counts
    assert a != b


def test_repeated_runs_on_one_seeded_simulator_repeat() -> None:
    """Documented behaviour: each ``run`` restarts from the simulator seed.

    Repeated runs on one seeded instance therefore return identical counts
    (like qiskit-aer's ``seed_simulator``); averaging them adds no
    information.
    """
    sim = Simulator(seed=5)
    circ = _uniform5()
    assert sim.run(circ, shots=300).counts == sim.run(circ, shots=300).counts


# =============================================================================
# MonteCarloSimulator (noisy trajectories)
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_monte_carlo_simulator_bit_flip_matches_kraus_reference() -> None:
    """Trajectory sampling of a bit-flip circuit matches qiskit DensityMatrix."""
    circ = QuantumCircuit().ry(0.9, 0).cx(0, 1).bit_flip(0.2, 1).measure_all()
    expected = qiskit_probabilities(circ)
    res = MonteCarloSimulator(seed=8).run(circ, shots=SHOTS)
    assert_counts_match(res.counts, expected, SHOTS)
