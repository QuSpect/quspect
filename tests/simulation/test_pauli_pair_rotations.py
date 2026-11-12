"""Native execution of the two-qubit Pauli rotations ``rxx``/``ryy``/``rzz``.

The QSAL decoder hands them to the runtime as named gates (``rzz`` through
the diagonal kernel, ``rxx``/``ryy`` through the dense two-qubit kernel); the
density-matrix simulator applies their ``gate_matrix``.  Every path must give
``exp(-i θ/2 P⊗P)`` exactly as its exact CX decomposition does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from quspect.circuit.circuit import QuantumCircuit
from quspect.gates.gate import Gate
from quspect.ir.compilation import decompose_deep
from quspect.ir.qsal import dumps
from quspect.native import NativeExecutor
from quspect.noise.channel import DepolarizingChannel
from quspect.simulation.density_simulator import DensityMatrixSimulator
from quspect.simulation.simulator import Simulator

_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_PREP = [(0.3, 1.1), (1.7, -0.4), (2.2, 0.9)]


def _prepared() -> QuantumCircuit:
    """A generic product state on three qubits (no amplitude is special)."""
    circ = QuantumCircuit()
    for qubit, (ry, rz) in enumerate(_PREP):
        circ.ry(ry, qubit).rz(rz, qubit)
    return circ


def _reference(circ: QuantumCircuit) -> np.ndarray:
    """State vector of ``circ`` from its exact CX/RZ decomposition."""
    lowered = decompose_deep(
        circ, supported_gates={"h", "s", "sdg", "cx", "rx", "ry", "rz"}
    )
    assert {i.gate.name for i in lowered if i.gate} <= {
        "h",
        "s",
        "sdg",
        "cx",
        "rx",
        "ry",
        "rz",
    }
    return Simulator().statevector(lowered)


@pytest.mark.parametrize("name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize(("a", "b"), [(0, 2), (2, 1), (1, 0)])
@pytest.mark.parametrize("theta", [0.37, -1.21, math.pi, 5.0])
def test_statevector_matches_exact_decomposition(
    name: str, a: int, b: int, theta: float
) -> None:
    circ = getattr(_prepared(), name)(theta, a, b)
    np.testing.assert_allclose(
        Simulator().statevector(circ), _reference(circ), atol=1e-13
    )


@pytest.mark.parametrize("name", ["rxx", "ryy", "rzz"])
def test_statevector_matches_the_exponential(name: str) -> None:
    """Little-endian check against ``expm`` on qubits (0, 1) of |+>|0>."""
    theta = 0.83
    circ = QuantumCircuit().h(0).id(1)
    getattr(circ, name)(theta, 0, 1)
    pauli = _PAULI[name[1]]
    initial = np.kron(np.array([1, 0]), np.array([1, 1]) / math.sqrt(2))
    expected = expm(-0.5j * theta * np.kron(pauli, pauli)) @ initial
    np.testing.assert_allclose(Simulator().statevector(circ), expected, atol=1e-13)


def test_rzz_on_plus_states_rotates_x_expectation() -> None:
    """RZZ(θ)|++>: <X0> = cos θ (X0 anticommutes with Z0Z1)."""
    theta = 0.83
    sv = Simulator().statevector(QuantumCircuit().h(0).h(1).rzz(theta, 0, 1))
    x0 = np.kron(np.eye(2), _PAULI["x"])  # little-endian: qubit 0 is the last factor
    assert np.vdot(sv, x0 @ sv).real == pytest.approx(math.cos(theta), abs=1e-13)


def test_single_precision_runs_the_same_kernels() -> None:
    circ = _prepared().rxx(0.4, 0, 1).ryy(-0.9, 1, 2).rzz(1.3, 2, 0)
    result = NativeExecutor(precision="single").run(
        dumps(circ), return_statevector=True
    )
    np.testing.assert_allclose(result.statevector_f32, _reference(circ), atol=2e-6)


def test_sampling_matches_probabilities() -> None:
    circ = _prepared().rxx(0.4, 0, 1).ryy(-0.9, 1, 2).rzz(1.3, 2, 0)
    probabilities = np.abs(_reference(circ)) ** 2
    circ.measure([0, 1, 2], [0, 1, 2])
    shots = 20_000
    counts = Simulator(seed=7).run(circ, shots=shots).counts
    for index, p in enumerate(probabilities):
        # Bitstrings put classical bit 0 leftmost.
        key = "".join(str((index >> q) & 1) for q in range(3))
        assert counts.get(key, 0) / shots == pytest.approx(p, abs=0.015)


def test_density_matrix_simulator() -> None:
    circ = _prepared().rxx(0.4, 0, 1).ryy(-0.9, 1, 2).rzz(1.3, 2, 0)
    sv = _reference(circ)
    rho = np.asarray(DensityMatrixSimulator().density_matrix(circ)._data)
    np.testing.assert_allclose(rho, np.outer(sv, sv.conj()), atol=1e-13)


def test_noisy_trajectories_accept_pauli_pair_rotations() -> None:
    circ = _prepared().rzz(1.3, 2, 0)
    circ.append_channel(
        DepolarizingChannel(name="depolarizing", num_qubits=1, p=0.0), (0,)
    )
    circ.measure([0, 1, 2], [0, 1, 2])
    counts = Simulator(seed=3).run(circ, shots=200).counts
    assert sum(counts.values()) == 200


def test_wrong_qubit_count_is_a_validation_error() -> None:
    circ = QuantumCircuit()
    circ.append(Gate(name="rzz", num_qubits=1, params=(0.3,)), (0,))
    with pytest.raises(RuntimeError, match="rzz.*expected 2 qubits"):
        NativeExecutor().run(dumps(circ), return_statevector=True)


def test_unbound_angle_is_rejected() -> None:
    from quspect.circuit.parameter import Parameter

    circ = QuantumCircuit().rzz(Parameter("t"), 0, 1)
    with pytest.raises(ValueError, match="unbound"):
        Simulator().statevector(circ)
