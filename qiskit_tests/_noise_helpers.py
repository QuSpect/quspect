"""Reference channels and utilities for the noise / density-matrix test files.

Reference channels are built from :mod:`qiskit.quantum_info` objects
(``Operator``, ``SuperOp``, ``Kraus``, ``Choi``) following the textbook
definitions (Nielsen & Chuang, "Quantum Computation and Quantum Information",
Sec. 8.3) and the definitions used by ``qiskit_aer.noise`` (which is not
installed here, so its formulas are reproduced by hand).  Channels are
compared through their **Choi matrices**, which are unique, so the
non-uniqueness of Kraus decompositions does not matter.

Parameter conventions (quspect parameter -> reference formula)
--------------------------------------------------------------
* ``BitFlipChannel(p)``      : ``(1-p) rho + p X rho X``  (N&C 8.3.3)
* ``PhaseFlipChannel(p)``    : ``(1-p) rho + p Z rho Z``  (N&C 8.3.3)
* ``BitPhaseFlipChannel(p)`` : ``(1-p) rho + p Y rho Y``  (N&C 8.3.3)
* ``PauliChannel(px,py,pz)`` : ``(1-px-py-pz) rho + px XrX + py YrY + pz ZrZ``
  (= aer ``pauli_error([("X",px),("Y",py),("Z",pz),("I",1-sum)])``)
* ``DepolarizingChannel(p, n)`` : ``(1-p) rho + p Tr(rho) I/2^n``.  This is
  N&C eq. (8.100) and **exactly** aer ``depolarizing_error(param=p, n)``.
  Each non-identity Pauli is applied with probability ``p/4^n`` (``p/4`` for
  one qubit, ``p/16`` for two).  It is *not* the "p/3 per Pauli" convention
  of N&C eq. (8.102): ``p_quspect = 4/3 * p_(p/3-convention)``.
* ``AmplitudeDampingChannel(gamma)`` : N&C (8.107)
  ``E0 = [[1,0],[0,sqrt(1-g)]], E1 = [[0,sqrt(g)],[0,0]]``
  (= aer ``amplitude_damping_error(gamma)``).
* ``PhaseDampingChannel(gamma)`` : N&C (8.127)/(8.128) with ``lambda = gamma``:
  off-diagonals scale by ``sqrt(1-gamma)``
  (= aer ``phase_damping_error(gamma)``; equivalently a phase flip with
  ``p = (1 - sqrt(1-gamma)) / 2``).
* ``GeneralizedAmplitudeDampingChannel(gamma, p_thermal)`` : N&C (8.108)-(8.111)
  with N&C's ``p = 1 - p_thermal``; ``p_thermal`` is the equilibrium |1>
  population (= aer ``amplitude_damping_error(gamma,
  excited_state_population=p_thermal)``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Choi, Kraus, Operator, SuperOp

from quspect.noise.channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    BitPhaseFlipChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    NoiseChannel,
    PauliChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
)

ComplexArray = NDArray[Any]

I2: ComplexArray = np.eye(2, dtype=complex)
X: ComplexArray = np.array([[0, 1], [1, 0]], dtype=complex)
Y: ComplexArray = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z: ComplexArray = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS_1Q: dict[str, ComplexArray] = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_string_matrix(label: str) -> ComplexArray:
    """Big-endian Kronecker product of a Pauli label (``"XZ"`` = X (x) Z)."""
    mat: ComplexArray = np.array([[1.0]], dtype=complex)
    for ch in label:
        mat = np.kron(mat, PAULIS_1Q[ch])
    return mat


# =============================================================================
# Choi matrices
# =============================================================================


def choi_of_kraus(ops: Sequence[Any]) -> ComplexArray:
    """Choi matrix (qiskit convention) of a list of Kraus operators."""
    return np.asarray(Choi(Kraus([np.asarray(k, dtype=complex) for k in ops])).data)


def choi_of_channel(channel: NoiseChannel) -> ComplexArray:
    """Choi matrix of a quspect channel's (big-endian) Kraus operators."""
    return choi_of_kraus(channel.kraus_operators())


def choi_of_superop(superop: SuperOp) -> ComplexArray:
    """Choi matrix of a qiskit ``SuperOp``."""
    return np.asarray(Choi(superop).data)


def choi_of_map(fn: Callable[[ComplexArray], ComplexArray], dim: int) -> ComplexArray:
    """Choi matrix of a linear map, ``sum_ij kron(|i><j|, E(|i><j|))``.

    The ordering matches qiskit's ``Choi`` convention (checked numerically
    against ``Choi(Kraus(...))``).
    """
    out = np.zeros((dim * dim, dim * dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            e = np.zeros((dim, dim), dtype=complex)
            e[i, j] = 1.0
            out += np.kron(e, fn(e))
    return out


def assert_choi_equal(
    actual: ComplexArray, expected: ComplexArray, atol: float = 1e-12
) -> None:
    """Assert two Choi matrices agree element-wise."""
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def kraus_completeness(ops: Sequence[Any]) -> ComplexArray:
    """Return ``sum_i K_i^dagger K_i``."""
    dim = np.asarray(ops[0]).shape[0]
    acc = np.zeros((dim, dim), dtype=complex)
    for k in ops:
        k_arr = np.asarray(k, dtype=complex)
        acc += k_arr.conj().T @ k_arr
    return acc


# =============================================================================
# Reference channels built from qiskit.quantum_info
# =============================================================================


def ref_pauli_superop(probs: Mapping[str, float]) -> SuperOp:
    """``sum_P probs[P] * P rho P`` as a qiskit ``SuperOp`` (labels big-endian).

    Equivalent to ``qiskit_aer.noise.pauli_error(list(probs.items()))``.
    """
    n = len(next(iter(probs)))
    total = SuperOp(np.zeros((4**n, 4**n), dtype=complex))
    for label, p in probs.items():
        total = total + p * SuperOp(Operator(pauli_string_matrix(label)))
    return total


def ref_depolarizing_superop(p: float, num_qubits: int) -> SuperOp:
    """``(1 - p) rho + p Tr(rho) I / 2^n`` (aer ``depolarizing_error(p, n)``).

    Built as ``(1 - p) * Id + p * D`` where ``D`` is the completely
    depolarizing channel ``(1/4^n) sum_P P rho P`` over all ``4^n`` Pauli
    strings.
    """
    dim = 2**num_qubits
    labels = ["".join(t) for t in itertools.product("IXYZ", repeat=num_qubits)]
    full = ref_pauli_superop(dict.fromkeys(labels, 1.0 / 4**num_qubits))
    ident = SuperOp(Operator(np.eye(dim, dtype=complex)))
    return (1.0 - p) * ident + p * full


def ref_depolarizing_map(p: float, dim: int) -> Callable[[ComplexArray], ComplexArray]:
    """Independent N&C (8.100) formula ``(1-p) rho + p Tr(rho) I/d``."""

    def fn(rho: ComplexArray) -> ComplexArray:
        return (1.0 - p) * rho + p * np.trace(rho) * np.eye(dim) / dim

    return fn


def ref_amplitude_damping_kraus(gamma: float) -> list[ComplexArray]:
    """N&C (8.107) amplitude damping Kraus operators."""
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def ref_amplitude_damping_map(
    gamma: float,
) -> Callable[[ComplexArray], ComplexArray]:
    """N&C (8.108): ``[[a, b], [c, d]] -> [[a + g d, sqrt(1-g) b], [., (1-g) d]]``."""
    s = math.sqrt(1.0 - gamma)

    def fn(rho: ComplexArray) -> ComplexArray:
        a, b, c, d = rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]
        return np.array([[a + gamma * d, s * b], [s * c, (1 - gamma) * d]])

    return fn


def ref_phase_damping_map(lam: float) -> Callable[[ComplexArray], ComplexArray]:
    """N&C (8.128): diagonal kept, off-diagonals scaled by ``sqrt(1-lambda)``."""
    s = math.sqrt(1.0 - lam)

    def fn(rho: ComplexArray) -> ComplexArray:
        return np.array([[rho[0, 0], s * rho[0, 1]], [s * rho[1, 0], rho[1, 1]]])

    return fn


def ref_generalized_amplitude_damping_kraus(
    gamma: float, p_thermal: float
) -> list[ComplexArray]:
    """N&C (8.108)-(8.111) with N&C ``p = 1 - p_thermal``."""
    p = 1.0 - p_thermal
    sg, s1g = math.sqrt(gamma), math.sqrt(1.0 - gamma)
    return [
        math.sqrt(p) * np.array([[1, 0], [0, s1g]], dtype=complex),
        math.sqrt(p) * np.array([[0, sg], [0, 0]], dtype=complex),
        math.sqrt(1 - p) * np.array([[s1g, 0], [0, 1]], dtype=complex),
        math.sqrt(1 - p) * np.array([[0, 0], [sg, 0]], dtype=complex),
    ]


def ref_thermal_relaxation_map(
    t1: float, t2: float, time: float, excited_population: float = 0.0
) -> Callable[[ComplexArray], ComplexArray]:
    """Analytic T1/T2 relaxation (aer ``thermal_relaxation_error``).

    Populations relax towards ``diag(1 - p1, p1)`` with rate ``1/T1``;
    coherences decay as ``exp(-t/T2)``.
    """
    e1 = math.exp(-time / t1)
    e2 = math.exp(-time / t2)
    p1 = excited_population

    def fn(rho: ComplexArray) -> ComplexArray:
        r11 = rho[1, 1] * e1 + np.trace(rho) * p1 * (1 - e1)
        r00 = np.trace(rho) - r11
        return np.array([[r00, e2 * rho[0, 1]], [e2 * rho[1, 0], r11]])

    return fn


# =============================================================================
# Channel catalogue
# =============================================================================


def random_cptp_kraus(num_qubits: int, num_ops: int, seed: int) -> list[ComplexArray]:
    """Random CPTP Kraus set built from a random isometry (QR decomposition)."""
    rng = np.random.default_rng(seed)
    dim = 2**num_qubits
    a = rng.normal(size=(dim * num_ops, dim)) + 1j * rng.normal(
        size=(dim * num_ops, dim)
    )
    q, _ = np.linalg.qr(a)
    return [np.ascontiguousarray(q[i * dim : (i + 1) * dim, :]) for i in range(num_ops)]


def channel_catalogue() -> dict[str, NoiseChannel]:
    """One representative instance of every channel type (non-trivial params)."""
    return {
        "pauli": PauliChannel(name="pauli", num_qubits=1, px=0.07, py=0.11, pz=0.19),
        "bit_flip": BitFlipChannel(name="bit_flip", num_qubits=1, p=0.23),
        "phase_flip": PhaseFlipChannel(name="phase_flip", num_qubits=1, p=0.31),
        "bit_phase_flip": BitPhaseFlipChannel(
            name="bit_phase_flip", num_qubits=1, p=0.17
        ),
        "depolarizing_1q": DepolarizingChannel(name="dep", num_qubits=1, p=0.37),
        "depolarizing_2q": DepolarizingChannel(name="dep2", num_qubits=2, p=0.29),
        "amplitude_damping": AmplitudeDampingChannel(
            name="amplitude_damping", num_qubits=1, gamma=0.33
        ),
        "phase_damping": PhaseDampingChannel(
            name="phase_damping", num_qubits=1, gamma=0.41
        ),
        "generalized_amplitude_damping": GeneralizedAmplitudeDampingChannel(
            name="gad", num_qubits=1, gamma=0.36, p_thermal=0.21
        ),
        "kraus_1q": KrausChannel(
            name="k1", num_qubits=1, kraus_ops=random_cptp_kraus(1, 3, seed=11)
        ),
        "kraus_2q": KrausChannel(
            name="k2", num_qubits=2, kraus_ops=random_cptp_kraus(2, 2, seed=12)
        ),
    }


# =============================================================================
# Shot statistics
# =============================================================================


def z_expectation_from_counts(
    counts: Mapping[str, int], clbit: int
) -> tuple[float, int]:
    """Estimate ``<Z>`` of a classical bit from quspect-keyed counts.

    Returns:
        ``(mean, shots)`` where ``mean = sum(count * (1 - 2 * bit)) / shots``.
    """
    shots = sum(counts.values())
    total = sum(c * (1 - 2 * int(k[clbit])) for k, c in counts.items())
    return total / shots, shots


def z_expectation_from_probs(probs: Mapping[str, float], clbit: int) -> float:
    """Exact ``<Z>`` of a classical bit from quspect-keyed probabilities."""
    return float(sum(p * (1 - 2 * int(k[clbit])) for k, p in probs.items()))


def nonzero(counts: Mapping[str, int]) -> dict[str, int]:
    """Drop zero-count entries (DensityMatrixSimulator pads all 2^n keys)."""
    return {k: int(v) for k, v in counts.items() if v}
