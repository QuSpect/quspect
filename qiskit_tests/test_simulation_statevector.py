"""Cross-validate ``Simulator.statevector`` / ``apply_gate`` against qiskit.

Conventions: quspect and qiskit statevectors are both little-endian, so they
are compared element-wise *without* a global-phase allowance.
``apply_gate(state, matrix, qubits)`` interprets ``matrix`` in the qiskit
``Operator`` convention (``qubits[0]`` is the least-significant bit of the
matrix index); ``gate_matrix`` is big-endian, so the pair
``apply_gate(state, gate_matrix(g), qubits[::-1])`` applies ``g`` to
``qubits``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import (
    Operator,
    Statevector,
    random_statevector,
    random_unitary,
)

import quspect.simulation.simulator as sim_mod
from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_states_equal,
    random_circuit_pair,
    to_qiskit,
)
from quspect import QuantumCircuit
from quspect.gates import gate_matrix
from quspect.simulation import Simulator, apply_gate, apply_gate_out_of_place


@pytest.mark.qiskit_ref
@pytest.mark.parametrize(
    ("n", "seed"), [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (3, 6), (4, 7)]
)
def test_random_circuit_statevector_matches_qiskit(
    n: int, seed: int, simulator: Simulator
) -> None:
    """Random circuits over all 25 gates: exact amplitude agreement."""
    circ, qc = random_circuit_pair(n, 12 * n, seed)
    sv = simulator.statevector(circ)
    assert sv.shape == (1 << n,)
    assert sv.dtype == np.complex128
    assert_states_equal(sv, Statevector(qc), up_to_phase=False, atol=1e-10)


@pytest.mark.qiskit_ref
def test_statevector_with_qubit_gaps(simulator: Simulator) -> None:
    """Untouched lower qubits are part of the state (width = max index + 1)."""
    circ = QuantumCircuit().h(4).cx(4, 2).ry(0.3, 2)
    sv = simulator.statevector(circ)
    assert sv.shape == (32,)
    assert_states_equal(sv, Statevector(to_qiskit(circ)), up_to_phase=False)


@pytest.mark.qiskit_ref
def test_run_save_statevector_matches_statevector(simulator: Simulator) -> None:
    """``run(save_statevector=True)`` without measurements = ``statevector``."""
    circ, qc = random_circuit_pair(3, 30, 42)
    res = simulator.run(circ, shots=100, save_statevector=True)
    assert res.num_shots == 0, "no measurement -> no sampling"
    assert res.counts == {} and res.probabilities == {}
    assert res.statevector is not None
    assert_states_equal(res.statevector, Statevector(qc), up_to_phase=False)
    res2 = simulator.run(circ, shots=100)
    assert res2.statevector is None


@pytest.mark.qiskit_ref
def test_terminal_measurement_statevector_is_pre_measurement(
    simulator: Simulator,
) -> None:
    """With terminal measurements, ``statevector`` returns the unmeasured state."""
    circ = QuantumCircuit().h(0).cx(0, 1).ry(0.4, 1)
    ref = Statevector(to_qiskit(circ))
    circ.measure_all()
    assert_states_equal(simulator.statevector(circ), ref, up_to_phase=False)


def test_run_empty_circuit(simulator: Simulator) -> None:
    """An empty circuit returns an empty 0-shot result."""
    res = simulator.run(QuantumCircuit(), shots=10, save_statevector=True)
    assert res.num_shots == 0
    assert res.counts == {}
    assert res.num_qubits == 0
    assert res.statevector is None


def test_statevector_empty_circuit_is_unit_scalar(simulator: Simulator) -> None:
    """The 0-qubit state is ``[1]``, as ``Statevector(QuantumCircuit(0))``."""
    ref = Statevector(qiskit.QuantumCircuit(0))
    np.testing.assert_allclose(simulator.statevector(QuantumCircuit()), ref.data)


def test_statevector_mid_circuit_measurement_is_valid_or_raises(
    simulator: Simulator,
) -> None:
    """Either raise a clear error or return a normalized post-measurement state."""
    circ = QuantumCircuit().h(0).measure([0], [0]).h(0)
    try:
        sv = simulator.statevector(circ)
    except (ValueError, RuntimeError):
        return
    assert sv.shape == (2,)
    assert np.linalg.norm(sv) == pytest.approx(1.0)


def test_negative_shots_rejected(simulator: Simulator) -> None:
    """``shots < 0`` raises ValueError."""
    with pytest.raises(ValueError, match="shots"):
        simulator.run(QuantumCircuit().h(0).measure_all(), shots=-1)


def test_simulator_names() -> None:
    """Backend names of the state-vector simulators."""
    from quspect.simulation import MonteCarloSimulator

    assert Simulator().name == "simulator"
    assert MonteCarloSimulator(seed=1).name == "monte_carlo_simulator"
    assert Simulator(seed=7).seed == 7
    assert Simulator().seed is None


def test_large_qubit_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ResourceWarning is emitted at the (patched) large-width threshold."""
    monkeypatch.setattr(sim_mod, "_LARGE_N_WARN", 3)
    circ = QuantumCircuit().h(2).measure_all()
    with pytest.warns(ResourceWarning, match=r"3 qubits.*2\^3 = 8"):
        Simulator(seed=0).run(circ, shots=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulator(seed=0).run(QuantumCircuit().h(1).measure_all(), shots=4)


def test_estimate_memory_values() -> None:
    """Memory estimate is 16 bytes per amplitude, in MiB."""
    assert sim_mod._estimate_memory(25) == pytest.approx(512.0)
    assert sim_mod._estimate_memory(20) == pytest.approx(16.0)


# =============================================================================
# apply_gate / apply_gate_out_of_place
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize(
    ("n", "qubits"),
    [(1, (0,)), (3, (2,)), (3, (0, 2)), (3, (2, 0)), (4, (3, 1, 0)), (4, (1, 3))],
)
def test_apply_gate_matches_qiskit_evolve(n: int, qubits: tuple[int, ...]) -> None:
    """``apply_gate`` with a qiskit Operator matrix == ``Statevector.evolve``."""
    seed = 100 * n + sum(qubits)
    psi = random_statevector(1 << n, seed=seed)
    op = random_unitary(1 << len(qubits), seed=seed + 1)
    state = np.array(psi.data, dtype=np.complex128)
    apply_gate(state, np.asarray(op.data), qubits)
    ref = psi.evolve(op, qargs=list(qubits))
    assert_states_equal(state, ref, up_to_phase=False, atol=1e-12)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(GATE_SPECS))
def test_apply_gate_with_gate_matrix_reversed_qubits(name: str) -> None:
    """``apply_gate(s, gate_matrix(g), qubits[::-1])`` applies g to qubits."""
    spec = GATE_SPECS[name]
    params = [0.37 * (i + 1) for i in range(spec.num_params)]
    qubits = (2, 0, 1)[: spec.num_qubits]
    psi = random_statevector(8, seed=len(name))
    state = np.array(psi.data, dtype=np.complex128)
    apply_gate(state, gate_matrix(spec.quspect_gate(*params)), qubits[::-1])
    qc = qiskit.QuantumCircuit(3)
    spec.apply_qiskit(qc, params, qubits)
    assert_states_equal(state, psi.evolve(Operator(qc)), up_to_phase=False)


def test_apply_gate_out_of_place_does_not_mutate() -> None:
    """The out-of-place variant returns a new array and leaves the input."""
    psi = np.array(random_statevector(4, seed=3).data, dtype=np.complex128)
    original = psi.copy()
    h = gate_matrix(GATE_SPECS["h"].quspect_gate())
    out = apply_gate_out_of_place(psi, h, (1,))
    np.testing.assert_array_equal(psi, original)
    ref = Statevector(original).evolve(Operator(h), qargs=[1])
    assert_states_equal(out, ref, up_to_phase=False)


@pytest.mark.parametrize(
    ("dim", "mat_dim", "qubits", "match"),
    [
        (4, 4, (0,), "inconsistent"),
        (4, 2, (2,), "out of range"),
        (4, 4, (1, 1), "Duplicate"),
        (2, 2, (), "inconsistent"),
    ],
)
def test_apply_gate_input_validation(
    dim: int, mat_dim: int, qubits: tuple[int, ...], match: str
) -> None:
    """Shape mismatch, out-of-range and duplicate qubits raise ValueError."""
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1
    with pytest.raises(ValueError, match=match):
        apply_gate(state, np.eye(mat_dim, dtype=np.complex128), qubits)


def test_apply_gate_rejects_non_power_of_two_state() -> None:
    """A length-3 state is not a valid qubit register."""
    state = np.array([1, 0, 0], dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    with pytest.raises(ValueError):
        apply_gate(state, x, (0,))


@pytest.mark.parametrize("kind", ["complex64", "float64", "strided"])
def test_apply_gate_non_complex128_state_updated_or_rejected(kind: str) -> None:
    """In-place application must either update the array or raise."""
    if kind == "strided":
        buf = np.zeros(4, dtype=np.complex128)
        state = buf[::2]
    else:
        state = np.zeros(2, dtype=kind)
    state[0] = 1
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    try:
        apply_gate(state, x, (0,))
    except (TypeError, ValueError):
        return
    np.testing.assert_allclose(state, [0, 1])
