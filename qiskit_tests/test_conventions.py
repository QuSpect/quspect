"""Tests pinning the quspect <-> qiskit conventions and the shared helpers.

See the module docstring of ``qiskit_tests/conftest.py`` for the list of
conventions verified here.
"""

from __future__ import annotations

import itertools
import math
import zlib

import numpy as np
import pytest
import qiskit
import qiskit.circuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import Operator, Statevector

import quspect
from qiskit_tests.helpers import (
    GATE_SPECS,
    N_SIGMA,
    assert_counts_match,
    assert_mean_close,
    assert_probability_close,
    assert_states_equal,
    assert_unitary_equal,
    binomial_tolerance,
    circuit_unitary,
    global_phase,
    pm1_std,
    qiskit_circuit_unitary,
    qiskit_counts_to_quspect,
    qiskit_gate_matrix_be,
    qiskit_probabilities,
    qiskit_sample_counts,
    quspect_circuit_unitary,
    quspect_counts_to_qiskit,
    quspect_matrix,
    random_circuit_pair,
    to_qiskit,
)
from quspect.circuit.parameter import Parameter
from quspect.gates.parameterized import ALL_PARAMETERIZED
from quspect.gates.standard import ALL_STANDARD
from quspect.noise.channel import DepolarizingChannel, KrausChannel
from quspect.simulation import DensityMatrix, DensityMatrixSimulator, Simulator

pytestmark = pytest.mark.qiskit_ref

PARAM_SETS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (math.pi, math.pi, math.pi),
    (-math.pi, -math.pi / 2, math.pi / 2),
    (-0.7, -1.9, -2.6),
    (0.3, -1.2, 2.1),
    (2 * math.pi, -2 * math.pi, 3 * math.pi),
    (5.1, 0.4, -4.4),
]


def _random_input_layer(
    circ: quspect.QuantumCircuit,
    qc: qiskit.QuantumCircuit,
    n: int,
    rng: np.random.Generator,
) -> None:
    """Append the same entangled random state preparation to both circuits."""
    for q in range(n):
        th, ph, la = (float(v) for v in rng.uniform(-math.pi, math.pi, 3))
        circ.u3(th, ph, la, q)
        qc.u(th, ph, la, q)
    for q in range(n - 1):
        circ.cx(q, q + 1)
        qc.cx(q, q + 1)
    for q in range(n):
        th = float(rng.uniform(-math.pi, math.pi))
        circ.ry(th, q)
        qc.ry(th, q)


# =============================================================================
# 1. Statevector ordering
# =============================================================================


@pytest.mark.parametrize(("n", "q"), [(n, q) for n in range(1, 5) for q in range(n)])
def test_x_on_each_qubit_statevector_matches_qiskit(
    n: int, q: int, simulator: Simulator
) -> None:
    """X on qubit q sets amplitude index 2**q (little-endian), like qiskit."""
    circ = quspect.QuantumCircuit().id(n - 1).x(q)
    qc = qiskit.QuantumCircuit(n)
    qc.x(q)
    sv = simulator.statevector(circ)
    expected = np.zeros(2**n, dtype=complex)
    expected[1 << q] = 1.0
    assert_states_equal(sv, expected, up_to_phase=False)
    assert_states_equal(sv, Statevector(qc), up_to_phase=False)


# =============================================================================
# 2. Multi-qubit gate argument order
# =============================================================================

_FIXED_MULTI = [
    (name, n, qubits)
    for name in ("cx", "cz", "swap", "ccx")
    for n in (3, 4)
    for qubits in itertools.permutations(range(n), GATE_SPECS[name].num_qubits)
]


@pytest.mark.parametrize(("name", "n", "qubits"), _FIXED_MULTI)
def test_multiqubit_gate_on_basis_states(
    name: str, n: int, qubits: tuple[int, ...]
) -> None:
    """cx/cz/swap/ccx on every ordered qubit tuple: all basis inputs agree."""
    spec = GATE_SPECS[name]
    circ = quspect.QuantumCircuit().id(n - 1)
    spec.apply_quspect(circ, [], qubits)
    qc = qiskit.QuantumCircuit(n)
    spec.apply_qiskit(qc, [], qubits)
    getattr(qc, name)(*qubits)  # qiskit fluent API with the same order ...
    getattr(qc, name)(*qubits)  # ... applied twice = identity (all self-inverse)
    assert_unitary_equal(quspect_circuit_unitary(circ), Operator(qc).data)


@pytest.mark.parametrize(("name", "n", "qubits"), _FIXED_MULTI)
def test_multiqubit_gate_on_random_state(
    name: str, n: int, qubits: tuple[int, ...], simulator: Simulator
) -> None:
    """cx/cz/swap/ccx on every ordered qubit tuple: entangled random input."""
    rng = np.random.default_rng(zlib.crc32(repr((name, n, qubits)).encode()))
    circ = quspect.QuantumCircuit()
    qc = qiskit.QuantumCircuit(n)
    _random_input_layer(circ, qc, n, rng)
    getattr(circ, name)(*qubits)
    getattr(qc, name)(*qubits)
    assert_states_equal(simulator.statevector(circ), Statevector(qc), up_to_phase=False)


_PARAM_MULTI = [
    (name, qubits)
    for name in ("crx", "cry", "crz", "cp", "cu3")
    for qubits in itertools.permutations(range(3), 2)
]


@pytest.mark.parametrize(("name", "qubits"), _PARAM_MULTI)
def test_controlled_param_gate_argument_order(
    name: str, qubits: tuple[int, int], simulator: Simulator
) -> None:
    """crx/cry/crz/cp(theta, c, t) and cu3(t, p, l, c, t) match qiskit's order."""
    rng = np.random.default_rng(7)
    params = [float(v) for v in rng.uniform(-3, 3, GATE_SPECS[name].num_params)]
    circ = quspect.QuantumCircuit()
    qc = qiskit.QuantumCircuit(3)
    _random_input_layer(circ, qc, 3, rng)
    getattr(circ, name)(*params, *qubits)
    if name == "cu3":
        qc.cu(*params, 0.0, *qubits)
    else:
        getattr(qc, name)(*params, *qubits)
    assert_states_equal(simulator.statevector(circ), Statevector(qc), up_to_phase=False)


# =============================================================================
# 3. Gate matrices (big-endian) for every gate
# =============================================================================


def test_gate_specs_cover_all_quspect_gates() -> None:
    """GATE_SPECS lists exactly the gates exported by quspect.gates."""
    names = {g.name for g in ALL_STANDARD} | {g.name for g in ALL_PARAMETERIZED}
    assert names == set(GATE_SPECS)
    for g in [*ALL_STANDARD, *ALL_PARAMETERIZED]:
        assert GATE_SPECS[g.name].num_qubits == g.num_qubits


_MATRIX_CASES = [
    (name, params)
    for name, spec in GATE_SPECS.items()
    for params in (
        [()] if spec.num_params == 0 else [p[: spec.num_params] for p in PARAM_SETS]
    )
]


@pytest.mark.parametrize(("name", "params"), _MATRIX_CASES)
def test_gate_matrix_matches_qiskit_big_endian(
    name: str, params: tuple[float, ...]
) -> None:
    """gate_matrix(g) == Operator(qiskit_gate).reverse_qargs(), no phase."""
    spec = GATE_SPECS[name]
    mine = quspect_matrix(spec.quspect_gate(*params))
    ref = qiskit_gate_matrix_be(spec.qiskit_gate(*params))
    assert_unitary_equal(mine, ref)
    assert abs(global_phase(mine, ref) - 1.0) < 1e-12


@pytest.mark.parametrize(("name", "params"), _MATRIX_CASES)
def test_fluent_api_builds_expected_gate(name: str, params: tuple[float, ...]) -> None:
    """circ.<name>(*params, *qubits) stores the gate with params in order."""
    spec = GATE_SPECS[name]
    qubits = tuple(range(spec.num_qubits))[::-1]
    circ = quspect.QuantumCircuit()
    spec.apply_quspect(circ, params, qubits)
    (instr,) = circ.instructions
    assert instr.gate is not None
    assert instr.gate.name == name
    assert instr.qubits == qubits
    assert tuple(float(p) for p in instr.gate.params) == tuple(params)


@pytest.mark.parametrize("name", list(GATE_SPECS))
def test_gate_in_circuit_matches_qiskit_operator(name: str) -> None:
    """Every gate on reversed / non-adjacent qubits of a 3-qubit register."""
    spec = GATE_SPECS[name]
    rng = np.random.default_rng(99)
    params = [float(v) for v in rng.uniform(-4, 4, spec.num_params)]
    for qubits in itertools.permutations(range(3), spec.num_qubits):
        circ = quspect.QuantumCircuit().id(2)
        spec.apply_quspect(circ, params, qubits)
        qc = qiskit.QuantumCircuit(3)
        spec.apply_qiskit(qc, params, qubits)
        assert_unitary_equal(quspect_circuit_unitary(circ), Operator(qc).data)


# =============================================================================
# 4. Count bitstring convention
# =============================================================================


def test_counts_measure_all_bitstring_reversed(simulator: Simulator) -> None:
    """x(0) + measure_all -> quspect '1000', qiskit '0001'."""
    circ = quspect.QuantumCircuit().id(3).x(0).measure_all()
    result = simulator.run(circ, shots=50)
    assert result.counts == {"1000": 50}
    assert result.probabilities == {"1000": 1.0}
    qc = to_qiskit(circ)
    qcounts = BasicSimulator().run(qc, shots=50, seed_simulator=1).result().get_counts()
    assert qcounts == {"0001": 50}
    assert qiskit_counts_to_quspect(qcounts) == result.counts
    assert quspect_counts_to_qiskit(result.counts) == qcounts


def test_counts_with_permuted_clbits(simulator: Simulator) -> None:
    """measure into shuffled clbits: quspect s[c] is clbit c."""
    circ = quspect.QuantumCircuit().id(3).x(0).x(2)
    circ.measure([0, 1, 2, 3], [3, 0, 2, 1])
    result = simulator.run(circ, shots=20)
    # clbit0<-q1=0, clbit1<-q3=0, clbit2<-q2=1, clbit3<-q0=1
    assert result.counts == {"0011": 20}
    qcounts = BasicSimulator().run(to_qiskit(circ), shots=20).result().get_counts()
    assert qcounts == {"1100": 20}
    assert qiskit_probabilities(circ) == {"0011": 1.0}


def test_counts_partial_measurement_unmeasured_clbits() -> None:
    """Unmeasured clbits read 0 in the reference distribution."""
    circ = quspect.QuantumCircuit().x(0).x(1).measure([1], [2])
    assert qiskit_probabilities(circ) == {"001": 1.0}


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [3, 11])
def test_counts_random_circuit_statistics(seed: int) -> None:
    """Sampled counts from a random circuit with permuted clbits match qiskit."""
    circ, _ = random_circuit_pair(3, 25, seed)
    clbits = [int(c) for c in np.random.default_rng(seed).permutation(3)]
    circ.measure([0, 1, 2], clbits)
    shots = 8000
    result = Simulator(seed=seed).run(circ, shots=shots)
    expected = qiskit_probabilities(circ)
    assert_counts_match(result.counts, expected, shots)
    for key, prob in result.probabilities.items():
        assert prob == pytest.approx(result.counts[key] / shots)
    # Same support as qiskit's own sampler (keys reversed back to quspect).
    qcounts = BasicSimulator().run(to_qiskit(circ), shots=shots, seed_simulator=seed)
    ref = qiskit_counts_to_quspect(qcounts.result().get_counts())
    assert_counts_match(ref, expected, shots)


# =============================================================================
# Random circuit statevector equivalence
# =============================================================================


@pytest.mark.parametrize(
    ("n", "depth", "seed"),
    [(n, d, s) for n in range(2, 6) for d in (20, 30, 40) for s in range(3)],
)
def test_random_circuit_statevector(
    n: int, depth: int, seed: int, simulator: Simulator
) -> None:
    """Random circuits over all gates: quspect statevector == qiskit exactly."""
    circ, qc = random_circuit_pair(n, depth, 1000 * n + 10 * depth + seed)
    sv = simulator.statevector(circ)
    ref = Statevector(qc).data
    fidelity = abs(np.vdot(ref, sv)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)
    assert_states_equal(sv, ref, up_to_phase=False)


@pytest.mark.parametrize("seed", range(3))
def test_random_circuit_fixed_gates_only(seed: int, simulator: Simulator) -> None:
    """parametric=False restricts to fixed gates and still matches qiskit."""
    circ, qc = random_circuit_pair(4, 30, seed, parametric=False)
    assert all(not i.gate.params for i in circ.instructions if i.gate is not None)
    assert_states_equal(simulator.statevector(circ), Statevector(qc), up_to_phase=False)


# =============================================================================
# Helper self-tests: to_qiskit, unitaries
# =============================================================================


@pytest.mark.parametrize("seed", range(4))
def test_to_qiskit_matches_independent_construction(seed: int) -> None:
    """to_qiskit(circ) has the same operator as the independently built pair."""
    circ, qc = random_circuit_pair(3, 30, seed)
    assert_unitary_equal(Operator(to_qiskit(circ)).data, Operator(qc).data)
    assert_unitary_equal(qiskit_circuit_unitary(circ), Operator(qc).data)


@pytest.mark.parametrize("seed", range(4))
def test_quspect_circuit_unitary_matches_qiskit(seed: int) -> None:
    """quspect_circuit_unitary (quspect only) == qiskit Operator, no phase."""
    circ, qc = random_circuit_pair(3, 25, seed)
    u = quspect_circuit_unitary(circ)
    assert_unitary_equal(u @ u.conj().T, np.eye(8))
    assert_unitary_equal(u, Operator(qc).data)
    assert_unitary_equal(circuit_unitary(circ), u)


def test_circuit_unitary_padding_and_barrier() -> None:
    """num_qubits pads idle qubits; barriers are ignored."""
    circ = quspect.QuantumCircuit().h(0).barrier().cx(0, 1)
    u = quspect_circuit_unitary(circ, num_qubits=3)
    assert u.shape == (8, 8)
    assert_unitary_equal(u, qiskit_circuit_unitary(circ, num_qubits=3))


def test_unitary_helpers_reject_measurement() -> None:
    """Unitary helpers refuse circuits with measurements."""
    circ = quspect.QuantumCircuit().h(0).measure_all()
    with pytest.raises(ValueError):
        quspect_circuit_unitary(circ)
    with pytest.raises(ValueError):
        qiskit_circuit_unitary(circ)


def test_to_qiskit_measure_and_barrier() -> None:
    """Measurements keep (qubit, clbit) pairs; barriers are preserved."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 2).barrier(0, 2).barrier()
    circ.measure([2, 0], [0, 1])
    qc = to_qiskit(circ)
    assert qc.num_qubits == 3 and qc.num_clbits == 2
    names = [ci.operation.name for ci in qc.data]
    assert names == ["h", "cx", "barrier", "barrier", "measure", "measure"]
    # quspect's barrier() with no args covers the *active* qubits {0, 2}.
    assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [0, 2]
    meas = [
        (qc.find_bit(ci.qubits[0]).index, qc.find_bit(ci.clbits[0]).index)
        for ci in qc.data
        if ci.operation.name == "measure"
    ]
    assert meas == [(2, 0), (0, 1)]


def test_to_qiskit_parameters_and_expressions() -> None:
    """Parameter / affine expressions map to qiskit and bind identically."""
    a, b = Parameter("a"), Parameter("b")
    circ = quspect.QuantumCircuit()
    circ.rx(a, 0).ry(2 * a - 0.5, 1).cu3(b, a + b, 1.25, 0, 1).u2(-b, a / 2, 1)
    table: dict[str, qiskit.circuit.Parameter] = {}
    qc = to_qiskit(circ, parameter_table=table)
    assert set(table) == {"a", "b"}
    assert {p.name for p in qc.parameters} == {"a", "b"}
    values = {a: 0.37, b: -1.4}
    bound = circ.bind_parameters(values)
    qbound = qc.assign_parameters({table["a"]: 0.37, table["b"]: -1.4})
    assert_unitary_equal(quspect_circuit_unitary(bound), Operator(qbound).data)


def test_to_qiskit_rejects_noise_by_default() -> None:
    """Noise channels raise NotImplementedError unless noise='kraus'."""
    circ = quspect.QuantumCircuit().h(0).depolarize(0.1, 0)
    with pytest.raises(NotImplementedError):
        to_qiskit(circ)
    with pytest.raises(ValueError):
        to_qiskit(circ, noise="bogus")


def _noisy_circuit() -> quspect.QuantumCircuit:
    """A 3-qubit circuit exercising every channel, incl. asymmetric 2q Kraus."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q_mat, _ = np.linalg.qr(a)
    k0 = math.sqrt(0.6) * np.eye(4, dtype=complex)
    k1 = math.sqrt(0.4) * q_mat  # non-symmetric 2-qubit Kraus operator
    asym_cx = np.eye(4, dtype=complex)[[0, 1, 3, 2]]  # big-endian CX(first->second)
    circ = quspect.QuantumCircuit()
    circ.h(0).ry(0.7, 1).rx(-1.1, 2).cx(0, 2)
    circ.bit_flip(0.1, 0).phase_flip(0.2, 1).bit_phase_flip(0.15, 2)
    circ.depolarize(0.3, 1).pauli_error(0.05, 0.1, 0.15, 0)
    circ.amplitude_damp(0.25, 2).phase_damp(0.35, 0)
    circ.generalized_amplitude_damp(0.3, 0.2, 1)
    circ.append_channel(
        KrausChannel(name="k2", num_qubits=2, kraus_ops=[k0, k1]), (2, 0)
    )
    circ.append_channel(DepolarizingChannel(name="dep2", num_qubits=2, p=0.2), (1, 2))
    circ.append_channel(
        KrausChannel(name="cx", num_qubits=2, kraus_ops=[asym_cx]), (0, 1)
    )
    circ.cry(0.4, 2, 1)
    return circ


def test_to_qiskit_kraus_density_matrix_matches_quspect() -> None:
    """noise='kraus' evolves to the same rho as DensityMatrixSimulator."""
    circ = _noisy_circuit()
    rho_q = DensityMatrixSimulator(seed=0).density_matrix(circ)
    rho_ref = QiskitDensityMatrix(to_qiskit(circ, noise="kraus")).data
    assert_unitary_equal(np.asarray(rho_q._data), rho_ref, atol=1e-10)


def test_to_qiskit_kraus_endianness_basis_state() -> None:
    """A 2-qubit big-endian Kraus CX(q0 -> q1) on |q0=1> flips q1, like quspect."""
    cx_be = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
    circ = quspect.QuantumCircuit().x(0)
    circ.append_channel(
        KrausChannel(name="cx", num_qubits=2, kraus_ops=[cx_be]), (0, 1)
    )
    assert qiskit_probabilities(circ) == pytest.approx({"11": 1.0})
    rho = DensityMatrixSimulator(seed=0).density_matrix(circ)
    assert np.real(np.diag(np.asarray(rho._data)))[3] == pytest.approx(1.0)


def test_qiskit_probabilities_noisy_matches_density_diagonal() -> None:
    """qiskit_probabilities (DensityMatrix path) == quspect rho diagonal."""
    circ = _noisy_circuit()
    diag = np.real(
        np.diag(np.asarray(DensityMatrixSimulator().density_matrix(circ)._data))
    )
    probs = qiskit_probabilities(circ)
    for idx, p in enumerate(diag):
        key = "".join(str((idx >> q) & 1) for q in range(3))
        assert probs.get(key, 0.0) == pytest.approx(p, abs=1e-10)


def test_qiskit_probabilities_rejects_mid_circuit_measurement() -> None:
    """Mid-circuit measurement is not supported by the exact reference."""
    circ = quspect.QuantumCircuit().h(0).measure([0], [0]).x(0)
    with pytest.raises(ValueError):
        qiskit_probabilities(circ)


def test_count_conversion_handles_spaces_and_merging() -> None:
    """Spaces from multiple registers are dropped before reversing."""
    assert qiskit_counts_to_quspect({"01 10": 3, "0110": 2}) == {"0110": 5}
    assert qiskit_counts_to_quspect({"001": 4}) == {"100": 4}
    assert quspect_counts_to_qiskit({"100": 4, "110": 1}) == {"001": 4, "011": 1}


def test_qiskit_multi_register_counts_conversion() -> None:
    """Two classical registers: converted key has s[c] = global clbit c."""
    qr = qiskit.QuantumRegister(3)
    c0, c1 = qiskit.ClassicalRegister(1, "a"), qiskit.ClassicalRegister(2, "b")
    qc = qiskit.QuantumCircuit(qr, c0, c1)
    qc.x(0)
    qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    counts = BasicSimulator().run(qc, shots=10).result().get_counts()
    assert counts == {"10 1": 10}
    assert qiskit_counts_to_quspect(counts) == {"101": 10}


# =============================================================================
# Helper self-tests: statistics
# =============================================================================


def test_binomial_tolerance_values() -> None:
    """Tolerance follows n_sigma*sqrt(max(p(1-p),1/N)/N) with floor 1/N."""
    assert binomial_tolerance(0.5, 10000) == pytest.approx(N_SIGMA * 0.005)
    assert binomial_tolerance(0.0, 10000) == pytest.approx(N_SIGMA / 10000)
    assert binomial_tolerance(1.0, 100, n_sigma=0.1) == pytest.approx(1 / 100)
    with pytest.raises(ValueError):
        binomial_tolerance(0.5, 0)


@pytest.mark.statistical
def test_assert_counts_match_accepts_correct_samples(rng: np.random.Generator) -> None:
    """Multinomial samples from the true distribution pass; probabilities too."""
    probs = {"00": 0.5, "01": 0.3, "10": 0.199, "11": 0.001}
    shots = 10000
    for _ in range(20):
        draws = rng.multinomial(shots, list(probs.values()))
        counts = dict(zip(probs, (int(d) for d in draws), strict=True))
        assert_counts_match(counts, probs, shots)
        assert_probability_close(counts["00"] / shots, 0.5, shots)


def test_assert_counts_match_rejects_wrong_distribution() -> None:
    """Wrong distributions, zero-probability hits and bad totals all fail."""
    with pytest.raises(AssertionError, match="per-outcome"):
        assert_counts_match({"0": 5500, "1": 4500}, {"0": 0.5, "1": 0.5}, 10000)
    with pytest.raises(AssertionError, match="zero-probability"):
        assert_counts_match({"0": 9999, "1": 1}, {"0": 1.0}, 10000)
    assert_counts_match({"0": 9999, "1": 1}, {"0": 1.0}, 10000, allow_unexpected=True)
    with pytest.raises(AssertionError, match="shots"):
        assert_counts_match({"0": 10}, {"0": 1.0}, 11)
    # Many small coherent deviations: per-outcome OK, chi-square rejects.
    keys = [format(i, "05b") for i in range(32)]
    probs = dict.fromkeys(keys, 1 / 32)
    shots = 32000
    counts = {k: 1000 + (90 if i % 2 else -90) for i, k in enumerate(keys)}
    with pytest.raises(AssertionError, match="chi-square"):
        assert_counts_match(counts, probs, shots)


@pytest.mark.statistical
def test_qiskit_sample_counts_is_consistent() -> None:
    """The reference sampler reproduces the exact distribution."""
    circ, _ = random_circuit_pair(3, 20, 42)
    shots = 8000
    counts = qiskit_sample_counts(circ, shots, seed=1)
    assert sum(counts.values()) == shots
    assert_counts_match(counts, qiskit_probabilities(circ), shots)


@pytest.mark.statistical
def test_assert_mean_close(rng: np.random.Generator) -> None:
    """Sample means of +/-1 variables are judged with n_sigma*std/sqrt(n)."""
    e = 0.3
    samples = np.where(rng.random(20000) < (1 + e) / 2, 1.0, -1.0)
    assert_mean_close(samples, e)
    assert_mean_close(float(samples.mean()), e, pm1_std(e), samples.size)
    with pytest.raises(AssertionError):
        assert_mean_close(samples, e + 0.1, pm1_std(e))
    with pytest.raises(ValueError):
        assert_mean_close(0.3, 0.3)
    assert pm1_std(1.0) == 0.0


def test_assert_states_equal_phase_handling() -> None:
    """Global phase is tolerated only with up_to_phase=True."""
    a = np.array([1, 1j], dtype=complex) / math.sqrt(2)
    phase = assert_states_equal(np.exp(0.4j) * a, a)
    assert phase == pytest.approx(np.exp(0.4j))
    with pytest.raises(AssertionError):
        assert_states_equal(np.exp(0.4j) * a, a, up_to_phase=False)
    with pytest.raises(AssertionError):
        assert_unitary_equal(-np.eye(2), np.eye(2))
    assert_unitary_equal(-np.eye(2), np.eye(2), up_to_phase=True)


# =============================================================================
# 5. Known bug: DensityMatrix.probabilities on a subset of qubits
# =============================================================================


def test_density_matrix_probabilities_all_qubits_ok() -> None:
    """Measuring all qubits returns the little-endian diagonal (correct)."""
    sv = np.zeros(8, dtype=complex)
    sv[1] = 1.0  # |001>: qubit 0 = 1
    probs = DensityMatrix.from_pure_state(sv).probabilities((0, 1, 2))
    np.testing.assert_allclose(probs, np.eye(8)[1], atol=1e-12)


def test_density_matrix_probabilities_subset_bug() -> None:
    """probabilities((0,)) of |001> should be [0, 1]."""
    sv = np.zeros(8, dtype=complex)
    sv[1] = 1.0
    probs = DensityMatrix.from_pure_state(sv).probabilities((0,))
    np.testing.assert_allclose(probs, [0.0, 1.0], atol=1e-12)
