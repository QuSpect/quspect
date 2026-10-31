"""Dense-unitary helpers for synthesis/consolidation tests (big-endian)."""

from __future__ import annotations

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix


def circuit_unitary(circuit: QuantumCircuit, num_qubits: int) -> np.ndarray:
    """Return the unitary of *circuit* with qubit 0 as the most significant bit."""
    dim = 2**num_qubits
    u = np.eye(dim, dtype=complex).reshape([2] * (2 * num_qubits))
    for ins in circuit.instructions:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
        axes_in = list(range(k, 2 * k))
        u = np.tensordot(g, u, axes=(axes_in, list(ins.qubits)))
        u = np.moveaxis(u, list(range(k)), list(ins.qubits))
    return u.reshape(dim, dim)


def phase_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Max-norm distance between *u* and *v* after optimal global phase."""
    overlap = np.vdot(v, u)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-300 else 1.0
    return float(np.max(np.abs(u - phase * v)))


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return np.asarray(q * (d / np.abs(d)))
