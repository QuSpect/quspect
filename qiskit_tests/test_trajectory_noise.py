"""Statistical validation of noisy Monte Carlo trajectory simulation.

``Simulator.run`` (and its alias ``MonteCarloSimulator``) samples one
quantum trajectory per shot for circuits containing noise channels.  The
sampled counts are compared with the *exact* distribution from qiskit
``DensityMatrix`` evolution (``qiskit_probabilities``, which converts every
channel with ``to_qiskit(noise="kraus")``).

Tolerances (see :mod:`qiskit_tests.helpers`): per-outcome 5-sigma binomial
bounds plus a chi-square goodness-of-fit test (alpha = 1e-6) via
``assert_counts_match``; single-qubit ``<Z>`` estimates use
``assert_mean_close`` with the exact single-shot std ``sqrt(1 - <Z>^2)``.
``SHOTS = 20000`` gives a 5-sigma resolution of ~0.018 on p = 0.5 (~0.009
on <Z> near +-0.9), far below the O(0.1) shifts caused by a wrong channel
convention, while keeping every file well under the time budget.  All
simulators use fixed seeds.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy import stats

import quspect
from qiskit_tests._noise_helpers import (
    channel_catalogue,
    nonzero,
    z_expectation_from_counts,
    z_expectation_from_probs,
)
from qiskit_tests.helpers import (
    assert_counts_match,
    assert_mean_close,
    pm1_std,
    qiskit_probabilities,
    random_circuit_pair,
)
from quspect.noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    NoiseChannel,
    NoiseModel,
    NoiseRule,
    PhaseDampingChannel,
    depolarizing_model,
    thermal_relaxation_model,
)
from quspect.simulation import DensityMatrixSimulator, MonteCarloSimulator, Simulator

pytestmark = [pytest.mark.qiskit_ref, pytest.mark.statistical]

SHOTS = 20000
CATALOGUE = channel_catalogue()


def _check_counts_and_z(
    circ: quspect.QuantumCircuit, counts: dict[str, int], shots: int
) -> None:
    """assert_counts_match + per-clbit <Z> check against the exact reference."""
    exact = qiskit_probabilities(circ)
    assert_counts_match(counts, exact, shots)
    for c in range(len(next(iter(exact)))):
        mean, n = z_expectation_from_counts(counts, c)
        ez = z_expectation_from_probs(exact, c)
        assert_mean_close(mean, ez, pm1_std(ez), n)


def _single_channel_circuit(
    channel: NoiseChannel, qubits: tuple[int, ...]
) -> quspect.QuantumCircuit:
    """Entangled 2-qubit prep -> channel -> basis rotation -> measure_all.

    The post-rotations turn coherence (dephasing / Y / Z errors) into
    Z-basis population differences, so every channel is visible.
    """
    circ = quspect.QuantumCircuit()
    circ.ry(1.1, 0).cx(0, 1).rx(0.7, 1).u3(0.4, 0.3, -0.8, 0)
    circ.append_channel(channel, qubits)
    circ.h(0).ry(0.6, 1).rz(0.2, 1).rx(-0.5, 1)
    return circ.measure_all()


_SINGLE_CASES = [
    (key, qubits)
    for key, ch in CATALOGUE.items()
    for qubits in ([(0,), (1,)] if ch.num_qubits == 1 else [(0, 1), (1, 0)])
]


@pytest.mark.parametrize(("key", "qubits"), _SINGLE_CASES)
def test_each_channel_trajectory_matches_exact(
    key: str, qubits: tuple[int, ...]
) -> None:
    """Every channel type, on each qubit / ordering: counts and <Z> match."""
    circ = _single_channel_circuit(CATALOGUE[key], qubits)
    seed = 1000 + sum(ord(ch) for ch in key) + 10 * qubits[0]
    result = Simulator(seed=seed).run(circ, shots=SHOTS)
    assert result.num_shots == SHOTS
    _check_counts_and_z(circ, result.counts, SHOTS)


@pytest.mark.parametrize(
    "channel",
    [
        BitFlipChannel(name="bf", num_qubits=1, p=1.0),
        DepolarizingChannel(name="d", num_qubits=1, p=1.0),
        DepolarizingChannel(name="d2", num_qubits=2, p=1.0),
        AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=1.0),
        PhaseDampingChannel(name="pd", num_qubits=1, gamma=1.0),
        BitFlipChannel(name="bf0", num_qubits=1, p=0.0),
    ],
    ids=lambda c: c.name,
)
def test_boundary_strength_trajectories(channel: NoiseChannel) -> None:
    """p/gamma = 0 or 1 boundaries behave like the exact channel."""
    qubits = (0, 1) if channel.num_qubits == 2 else (1,)
    circ = _single_channel_circuit(channel, qubits)
    result = Simulator(seed=77).run(circ, shots=SHOTS)
    _check_counts_and_z(circ, result.counts, SHOTS)


# =============================================================================
# Composite noise models on Bell / GHZ / random circuits
# =============================================================================


def _base(name: str) -> quspect.QuantumCircuit:
    if name == "bell":
        return quspect.QuantumCircuit().h(0).cx(0, 1)
    if name == "ghz":
        return quspect.QuantumCircuit().h(0).cx(0, 1).cx(1, 2)
    circ, _ = random_circuit_pair(
        3, 18, 4242, gate_set=["h", "rx", "ry", "u3", "s", "t", "cx", "cz", "swap"]
    )
    return circ


def _per_qubit_thermal_model(t1: dict[int, float], t2: dict[int, float]) -> NoiseModel:
    """thermal_relaxation_model with each channel focused on one qubit.

    Uses the same per-qubit channels as ``thermal_relaxation_model`` but
    ``focus_on=(q,)`` so 2-qubit gates get one 1-qubit channel per qubit
    (works around the arity bug, see test_noise_model.py).
    """
    model = NoiseModel(name="thermal_fixed")
    for rule in thermal_relaxation_model(t1, t2):
        q = next(
            q for q in t1 if rule.qubit_filter is not None and rule.qubit_filter(q)
        )
        model.add_rule(
            NoiseRule(
                rule.channel,
                gate_filter=rule.gate_filter,
                qubit_filter=rule.qubit_filter,
                focus_on=(q,),
            )
        )
    return model


_T1 = {0: 900.0, 1: 1200.0, 2: 700.0}
_T2 = {0: 600.0, 1: 1500.0, 2: 500.0}


@pytest.mark.parametrize("name", ["bell", "ghz", "random3"])
def test_depolarizing_model_trajectories(name: str) -> None:
    """depolarizing_model(0.05, 0.15): counts and <Z> match qiskit."""
    circ = depolarizing_model(0.05, 0.15).apply(_base(name)).measure_all()
    result = Simulator(seed=zlib.crc32(name.encode()) % 1000).run(circ, shots=SHOTS)
    _check_counts_and_z(circ, result.counts, SHOTS)


@pytest.mark.parametrize("name", ["bell", "ghz", "random3"])
def test_thermal_model_trajectories(name: str) -> None:
    """Per-qubit T1/T2 relaxation after every gate: counts and <Z> match."""
    model = _per_qubit_thermal_model(_T1, _T2)
    circ = model.apply(_base(name))
    # Rotate so that dephasing is visible, then measure.
    circ.h(0).rx(0.8, 1).measure_all()
    result = Simulator(seed=31 + len(name)).run(circ, shots=SHOTS)
    _check_counts_and_z(circ, result.counts, SHOTS)


def test_thermal_relaxation_model_single_qubit_circuit() -> None:
    """thermal_relaxation_model as-is on a 1q-gate-only circuit."""
    circ, _ = random_circuit_pair(3, 15, 17, gate_set=["h", "rx", "u3", "t"])
    noisy = thermal_relaxation_model(_T1, _T2, gate_time_1q=120.0).apply(circ)
    noisy.measure_all()
    result = Simulator(seed=8).run(noisy, shots=SHOTS)
    _check_counts_and_z(noisy, result.counts, SHOTS)


def test_thermal_relaxation_model_bell_trajectories() -> None:
    """thermal_relaxation_model on a Bell circuit should simulate correctly."""
    circ = thermal_relaxation_model(_T1, _T2).apply(_base("bell")).measure_all()
    result = Simulator(seed=5).run(circ, shots=SHOTS)
    _check_counts_and_z(circ, result.counts, SHOTS)


def test_combined_models_and_explicit_channels() -> None:
    """Depolarizing model + thermal model + explicit channels together."""
    circ = depolarizing_model(0.03, 0.08).apply(_base("ghz"))
    circ = _per_qubit_thermal_model(_T1, _T2).apply(circ)
    circ.append_channel(CATALOGUE["kraus_2q"], (2, 0))
    circ.generalized_amplitude_damp(0.2, 0.3, 1).ry(0.4, 2).measure_all()
    result = Simulator(seed=99).run(circ, shots=SHOTS)
    _check_counts_and_z(circ, result.counts, SHOTS)


# =============================================================================
# Reproducibility
# =============================================================================


def test_seed_reproducibility() -> None:
    """Same seed -> identical counts; different seeds -> different counts."""
    circ = depolarizing_model(0.1, 0.2).apply(_base("ghz")).measure_all()
    a = Simulator(seed=123).run(circ, shots=2000).counts
    b = Simulator(seed=123).run(circ, shots=2000).counts
    c = Simulator(seed=124).run(circ, shots=2000).counts
    assert a == b
    assert a != c


def test_seed_reproducibility_with_mid_circuit_measurement() -> None:
    """Reproducibility also holds on the mid-circuit-measurement path."""
    circ = quspect.QuantumCircuit().h(0).depolarize(0.3, 0).measure([0], [0])
    circ.cx(0, 1).bit_flip(0.2, 1).measure([1], [1])
    a = Simulator(seed=5).run(circ, shots=3000).counts
    b = Simulator(seed=5).run(circ, shots=3000).counts
    assert a == b


# =============================================================================
# Channels followed by mid-circuit measurement
# =============================================================================


def test_channel_then_mid_measurement_deferred_equivalent() -> None:
    """measure q0 then use it only as a control == deferred measurement."""
    circ = quspect.QuantumCircuit().x(0).amplitude_damp(0.3, 0).measure([0], [0])
    circ.cx(0, 1).bit_flip(0.1, 1).measure([1], [1])
    deferred = quspect.QuantumCircuit().x(0).amplitude_damp(0.3, 0)
    deferred.cx(0, 1).bit_flip(0.1, 1).measure([0, 1], [0, 1])
    exact = qiskit_probabilities(deferred)
    assert exact == pytest.approx({"00": 0.27, "01": 0.03, "10": 0.07, "11": 0.63})
    result = Simulator(seed=17).run(circ, shots=SHOTS)
    assert_counts_match(result.counts, exact, SHOTS)


def test_channel_then_mid_measurement_collapse() -> None:
    """AD then measure q0 then H then measure again: outcomes independent."""
    circ = quspect.QuantumCircuit().x(0).amplitude_damp(0.3, 0)
    circ.measure([0], [0]).h(0).measure([0], [1])
    expected = {"00": 0.15, "01": 0.15, "10": 0.35, "11": 0.35}
    result = Simulator(seed=18).run(circ, shots=SHOTS)
    assert_counts_match(result.counts, expected, SHOTS)


def test_channel_after_mid_measurement() -> None:
    """Depolarizing(p) after a measurement flips the next outcome w.p. p/2."""
    p = 0.6
    circ = quspect.QuantumCircuit().h(0).measure([0], [0])
    circ.depolarize(p, 0).measure([0], [1])
    flip = p / 2
    expected = {
        "00": 0.5 * (1 - flip),
        "01": 0.5 * flip,
        "10": 0.5 * flip,
        "11": 0.5 * (1 - flip),
    }
    result = Simulator(seed=19).run(circ, shots=SHOTS)
    assert_counts_match(result.counts, expected, SHOTS)


# =============================================================================
# Trajectories vs DensityMatrixSimulator
# =============================================================================


def _all_channel_circuit() -> quspect.QuantumCircuit:
    """3-qubit random circuit interleaved with every catalogue channel."""
    rng = np.random.default_rng(2024)
    circ, _ = random_circuit_pair(3, 10, 2024)
    for ch in CATALOGUE.values():
        qubits = [int(q) for q in rng.permutation(3)[: ch.num_qubits]]
        circ.append_channel(ch, qubits)
        circ.ry(float(rng.uniform(-3, 3)), int(rng.integers(3)))
    return circ.measure_all()


def test_trajectory_matches_density_matrix_diagonal() -> None:
    """Trajectory counts ~ diagonal of DensityMatrixSimulator.density_matrix."""
    circ = _all_channel_circuit()
    rho = DensityMatrixSimulator().density_matrix(circ)
    diag = np.real(np.diag(np.asarray(rho._data)))
    probs = {
        "".join(str((i >> q) & 1) for q in range(3)): float(p)
        for i, p in enumerate(diag)
        if p > 1e-14
    }
    result = Simulator(seed=55).run(circ, shots=SHOTS)
    assert_counts_match(result.counts, probs, SHOTS)


def test_trajectory_vs_density_simulator_two_sample() -> None:
    """Two-sample chi-square: Simulator.run vs DensityMatrixSimulator.run."""
    circ = _all_channel_circuit()
    traj = Simulator(seed=56).run(circ, shots=SHOTS).counts
    dens = nonzero(DensityMatrixSimulator(seed=57).run(circ, shots=SHOTS).counts)
    keys = sorted(set(traj) | set(dens))
    table = np.array([[traj.get(k, 0) for k in keys], [dens.get(k, 0) for k in keys]])
    table = table[:, table.sum(axis=0) >= 10]
    pvalue = float(stats.chi2_contingency(table).pvalue)
    assert pvalue > 1e-6, f"two-sample chi-square p={pvalue:.3e}\n{table}"


# =============================================================================
# MonteCarloSimulator and other Simulator paths with noise
# =============================================================================


def test_monte_carlo_simulator_alias() -> None:
    """MonteCarloSimulator is a Simulator with its own name and same RNG path."""
    mc = MonteCarloSimulator(seed=7)
    assert isinstance(mc, Simulator)
    assert mc.name == "monte_carlo_simulator"
    assert Simulator(seed=7).name == "simulator"
    circ = depolarizing_model(0.05, 0.1).apply(_base("bell")).measure_all()
    assert (
        mc.run(circ, shots=3000).counts
        == Simulator(seed=7).run(circ, shots=3000).counts
    )


def test_monte_carlo_simulator_statistics() -> None:
    """MonteCarloSimulator counts match the exact distribution."""
    circ = _all_channel_circuit()
    result = MonteCarloSimulator(seed=58).run(circ, shots=SHOTS)
    _check_counts_and_z(circ, result.counts, SHOTS)


def test_noisy_run_without_measurement() -> None:
    """No measurements: no sampling; saved statevector is a normalised trajectory."""
    circ = quspect.QuantumCircuit().h(0).depolarize(0.5, 0).cx(0, 1)
    result = Simulator(seed=3).run(circ, shots=100, save_statevector=True)
    assert result.counts == {} and result.num_shots == 0
    assert result.statevector is not None
    assert np.linalg.norm(result.statevector) == pytest.approx(1.0)
