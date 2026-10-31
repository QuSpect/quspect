"""Cross-validate ``quspect.simulation.density_matrix.DensityMatrix`` with qiskit.

Every public method is compared with ``qiskit.quantum_info.DensityMatrix``
(``partial_trace``, ``state_fidelity``, ``evolve``...).

Conventions pinned here:

* The matrix index is little-endian (qubit 0 = least-significant bit), like
  qiskit -- ``rho._data`` equals ``qiskit.DensityMatrix(...).data``.
* ``from_labels`` reads labels like ``qiskit.DensityMatrix.from_label``:
  the **leftmost** character is the **highest** qubit.
* ``apply_gate(U, qubits)`` / ``apply_channel(kraus, qubits)`` interpret the
  matrices **little-endian over ``qubits``** (``qubits[0]`` is the LSB of the
  matrix index), i.e. exactly ``qiskit_dm.evolve(U, qargs=list(qubits))``.
  This is the *opposite* of quspect's big-endian ``gate_matrix`` /
  ``kraus_operators()``, which is why ``DensityMatrixSimulator`` passes
  ``instr.qubits[::-1]``.
* ``measure(qubits)`` returns a bitstring with ``s[j]`` = outcome of
  ``qubits[j]``; the probability vector index uses bit ``j`` for
  ``qubits[j]`` (qiskit ``probabilities(qargs)`` convention).
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest
import qiskit.circuit.library
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import (
    Kraus,
    Operator,
    Statevector,
    partial_trace,
    random_density_matrix,
    random_statevector,
    random_unitary,
    state_fidelity,
)

from qiskit_tests._noise_helpers import random_cptp_kraus
from qiskit_tests.helpers import assert_counts_match
from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.standard import CX
from quspect.simulation import DensityMatrix

pytestmark = pytest.mark.qiskit_ref

ATOL = 1e-12


def _q(dm: DensityMatrix) -> QiskitDensityMatrix:
    """quspect DensityMatrix -> qiskit DensityMatrix (same index order)."""
    return QiskitDensityMatrix(np.array(dm._data))


def _rand(n: int, seed: int, rank: int | None = None) -> QiskitDensityMatrix:
    """Seeded random qiskit density matrix on n qubits."""
    return random_density_matrix(2**n, rank=rank, seed=seed)


def _from_q(rho: QiskitDensityMatrix) -> DensityMatrix:
    """qiskit DensityMatrix -> quspect DensityMatrix."""
    return DensityMatrix.from_array(np.array(rho.data, dtype=complex))


# =============================================================================
# Construction
# =============================================================================


@pytest.mark.parametrize("n", [0, 1, 3])
def test_default_constructor_is_ground_state(n: int) -> None:
    """DensityMatrix(n) == |0...0><0...0|; num_qubits/dim/shape consistent."""
    dm = DensityMatrix(n)
    assert dm.num_qubits == n
    assert dm.dim == 2**n
    assert dm.shape == (2**n, 2**n)
    np.testing.assert_allclose(dm._data, QiskitDensityMatrix.from_int(0, 2**n).data)


def test_constructor_rejects_negative() -> None:
    """num_qubits < 0 raises ValueError."""
    with pytest.raises(ValueError):
        DensityMatrix(-1)


def test_from_array_copies_and_validates() -> None:
    """from_array copies the data; non-square / non-power-of-2 raise."""
    ref = _rand(2, seed=1)
    src = np.array(ref.data, dtype=complex)
    dm = DensityMatrix.from_array(src)
    assert dm.num_qubits == 2
    src[0, 0] = 99.0
    np.testing.assert_allclose(dm._data, ref.data, atol=ATOL)
    with pytest.raises(ValueError, match="square"):
        DensityMatrix.from_array(np.zeros((2, 4), dtype=complex))
    with pytest.raises(ValueError, match="square"):
        DensityMatrix.from_array(np.zeros(4, dtype=complex))
    with pytest.raises(ValueError, match="power of 2"):
        DensityMatrix.from_array(np.eye(3, dtype=complex))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_from_pure_state_matches_qiskit(n: int) -> None:
    """|psi><psi| equals qiskit DensityMatrix(Statevector) (little-endian)."""
    sv = random_statevector(2**n, seed=10 + n)
    dm = DensityMatrix.from_pure_state(np.asarray(sv.data))
    np.testing.assert_allclose(dm._data, QiskitDensityMatrix(sv).data, atol=ATOL)
    assert dm.purity() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        DensityMatrix.from_pure_state(np.ones(3, dtype=complex))


def test_from_pure_state_basis_index_is_little_endian() -> None:
    """Basis vector e_1 is |q0=1>, the same as qiskit Statevector.from_label('001')."""
    dm = DensityMatrix.from_pure_state(np.eye(8, dtype=complex)[1])
    ref = QiskitDensityMatrix(Statevector.from_label("001"))
    np.testing.assert_allclose(dm._data, ref.data)


@pytest.mark.parametrize("label", ["0", "1", "01", "10", "110", "0011"])
def test_from_labels_single_matches_qiskit_from_label(label: str) -> None:
    """from_labels([(label, 1)]) == qiskit from_label(label) (leftmost = MSB)."""
    dm = DensityMatrix.from_labels([(label, 1.0)])
    assert dm.num_qubits == len(label)
    np.testing.assert_allclose(dm._data, QiskitDensityMatrix.from_label(label).data)


def test_from_labels_mixture_matches_qiskit() -> None:
    """A classical mixture of labels is the weighted sum of qiskit projectors."""
    pairs = [("001", 0.5), ("110", 0.3), ("100", 0.2)]
    dm = DensityMatrix.from_labels(pairs)
    ref = sum(p * QiskitDensityMatrix.from_label(lab).data for lab, p in pairs)
    np.testing.assert_allclose(dm._data, ref)
    # qubit 2 (leftmost char) is 1 with probability 0.5
    np.testing.assert_allclose(dm.probabilities((0, 1, 2)), np.diag(ref).real)
    with pytest.raises(ValueError):
        DensityMatrix.from_labels([])


@pytest.mark.parametrize("n", [0, 1, 3])
def test_maximally_mixed(n: int) -> None:
    """I/2^n with purity 2^-n, like qiskit."""
    dm = DensityMatrix.maximally_mixed(n)
    ref = QiskitDensityMatrix(np.eye(2**n) / 2**n)
    np.testing.assert_allclose(dm._data, ref.data)
    assert dm.purity() == pytest.approx(ref.purity().real)


# =============================================================================
# Scalars: trace, normalize, purity, fidelity
# =============================================================================


def test_trace_and_normalize() -> None:
    """trace() of 3 rho is 3; normalize() rescales back to rho."""
    ref = _rand(2, seed=2)
    dm = DensityMatrix.from_array(3.0 * np.array(ref.data))
    assert complex(dm.trace()) == pytest.approx(3.0)
    dm.normalize()
    assert complex(dm.trace()) == pytest.approx(1.0)
    np.testing.assert_allclose(dm._data, ref.data, atol=ATOL)


@pytest.mark.parametrize(("n", "rank"), [(1, 1), (1, 2), (2, 3), (3, None)])
def test_purity_matches_qiskit(n: int, rank: int | None) -> None:
    """purity() == qiskit DensityMatrix.purity()."""
    ref = _rand(n, seed=30 + n, rank=rank)
    assert _from_q(ref).purity() == pytest.approx(ref.purity().real, abs=ATOL)


@pytest.mark.parametrize("seed", range(4))
def test_fidelity_matches_qiskit_state_fidelity(seed: int) -> None:
    """fidelity() == qiskit state_fidelity for mixed/mixed and pure/mixed."""
    a, b = _rand(2, seed=100 + seed), _rand(2, seed=200 + seed)
    assert _from_q(a).fidelity(_from_q(b)) == pytest.approx(
        state_fidelity(a, b), abs=1e-10
    )
    # Pure state: exact value <psi|sigma|psi>.  sqrt() of the ~1e-16 spurious
    # eigenvalues limits both quspect and qiskit to ~1e-8 here.
    psi = random_statevector(4, seed=300 + seed)
    exact = float(np.real(np.vdot(psi.data, b.data @ psi.data)))
    pure = DensityMatrix.from_pure_state(np.asarray(psi.data))
    assert pure.fidelity(_from_q(b)) == pytest.approx(exact, abs=1e-7)
    assert state_fidelity(psi, b) == pytest.approx(exact, abs=1e-12)
    assert _from_q(a).fidelity(_from_q(a)) == pytest.approx(1.0, abs=1e-10)


def test_fidelity_orthogonal_and_non_psd() -> None:
    """Orthogonal states have fidelity 0; a non-PSD matrix raises ValueError."""
    zero = DensityMatrix.from_labels([("0", 1.0)])
    one = DensityMatrix.from_labels([("1", 1.0)])
    assert zero.fidelity(one) == pytest.approx(0.0, abs=1e-12)
    bad = DensityMatrix.from_array(np.diag([1.5, -0.5]).astype(complex))
    with pytest.raises(ValueError, match="positive"):
        bad.fidelity(zero)


# =============================================================================
# Partial trace & subsystem purity
# =============================================================================

_SUBSETS_3 = [s for r in range(0, 4) for s in itertools.combinations(range(3), r)] + [
    (2, 0),
    (1, 0),
]


@pytest.mark.parametrize("traced", _SUBSETS_3)
def test_partial_trace_matches_qiskit(traced: tuple[int, ...]) -> None:
    """partial_trace(qubits) traces *out* qubits, like qiskit partial_trace."""
    ref = _rand(3, seed=7)
    out = _from_q(ref).partial_trace(traced)
    expected = partial_trace(ref, list(traced)).data
    assert out.num_qubits == 3 - len(set(traced))
    np.testing.assert_allclose(out._data, np.atleast_2d(expected), atol=ATOL)


def test_partial_trace_of_entangled_state() -> None:
    """Tracing one half of a Bell state gives I/2."""
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    red = DensityMatrix.from_pure_state(bell).partial_trace((1,))
    np.testing.assert_allclose(red._data, np.eye(2) / 2, atol=ATOL)


def test_subsystem_purity_pure_bipartite() -> None:
    """Pure 2-qubit state: both reduced purities agree with qiskit."""
    sv = random_statevector(4, seed=5)
    dm = DensityMatrix.from_pure_state(np.asarray(sv.data))
    expected = partial_trace(sv, [1]).purity().real
    assert dm.subsystem_purity((0,)) == pytest.approx(expected, abs=ATOL)
    assert dm.subsystem_purity((1,)) == pytest.approx(expected, abs=ATOL)


def test_subsystem_purity_is_on_given_qubits() -> None:
    """rho = |0><0|_q0 (x) I/2_q1: purity on (0,) is 1, on (1,) is 1/2."""
    rho = np.kron(np.eye(2) / 2, np.diag([1.0, 0.0])).astype(complex)
    dm = DensityMatrix.from_array(rho)
    ref = QiskitDensityMatrix(rho)
    assert dm.subsystem_purity((0,)) == pytest.approx(
        partial_trace(ref, [1]).purity().real
    )
    assert dm.subsystem_purity((1,)) == pytest.approx(0.5)


# =============================================================================
# Gate and channel application
# =============================================================================

_GATE_QUBITS = [(0,), (3,), (1, 3), (3, 1), (2, 0), (0, 3, 1), (2, 1, 3)]


@pytest.mark.parametrize("qubits", _GATE_QUBITS)
def test_apply_gate_matches_qiskit_evolve(qubits: tuple[int, ...]) -> None:
    """apply_gate(U, qubits) == qiskit evolve(U, qargs=qubits) (U little-endian)."""
    k = len(qubits)
    u = random_unitary(2**k, seed=50 + sum(q * 7**i for i, q in enumerate(qubits)))
    ref = _rand(4, seed=9)
    dm = _from_q(ref)
    dm.apply_gate(np.asarray(u.data), qubits)
    np.testing.assert_allclose(
        dm._data, ref.evolve(u, qargs=list(qubits)).data, atol=1e-12
    )


def test_apply_gate_with_big_endian_gate_matrix_needs_reversed_qubits() -> None:
    """gate_matrix (big-endian) needs qubits[::-1]: CX(2 -> 0) on |q2=1>."""
    dm = DensityMatrix.from_labels([("100", 1.0)])
    dm.apply_gate(gate_matrix(CX), (0, 2))  # reversed (control=2, target=0)
    np.testing.assert_allclose(
        dm._data, QiskitDensityMatrix.from_label("101").data, atol=ATOL
    )
    # Equivalently: little-endian qiskit CX matrix on (control, target).
    dm2 = DensityMatrix.from_labels([("100", 1.0)])
    dm2.apply_gate(np.asarray(Operator(qiskit.circuit.library.CXGate()).data), (2, 0))
    np.testing.assert_allclose(dm2._data, dm._data, atol=ATOL)


@pytest.mark.parametrize(
    ("qubits", "num_ops"), [((2,), 3), ((3, 1), 2), ((0, 2), 4), ((1, 3, 0), 2)]
)
def test_apply_channel_matches_qiskit_evolve(
    qubits: tuple[int, ...], num_ops: int
) -> None:
    """apply_channel(K, qubits) == qiskit evolve(Kraus(K), qargs=qubits)."""
    ops = random_cptp_kraus(len(qubits), num_ops, seed=60 + len(qubits) + num_ops)
    ref = _rand(4, seed=11)
    dm = _from_q(ref)
    dm.apply_channel(ops, qubits)
    expected = ref.evolve(Kraus(ops), qargs=list(qubits)).data
    np.testing.assert_allclose(dm._data, expected, atol=1e-12)
    assert complex(dm.trace()) == pytest.approx(1.0)


def test_apply_channel_rejects_wrong_shape() -> None:
    """Kraus operators of the wrong dimension raise ValueError."""
    dm = DensityMatrix(2)
    with pytest.raises(ValueError, match="shape"):
        dm.apply_channel([np.eye(2, dtype=complex)], (0, 1))


# =============================================================================
# probabilities / measure
# =============================================================================


def test_probabilities_all_qubits_matches_qiskit() -> None:
    """probabilities((0, ..., n-1)) == qiskit probabilities() (little-endian)."""
    ref = _rand(3, seed=13)
    np.testing.assert_allclose(
        _from_q(ref).probabilities((0, 1, 2)), ref.probabilities(), atol=ATOL
    )


_PROB_SUBSETS = [(0,), (1,), (2,), (0, 2), (2, 0), (1, 0), (2, 0, 1), (1, 2, 0)]


@pytest.mark.parametrize("qubits", _PROB_SUBSETS)
def test_probabilities_subset_and_order_matches_qiskit(
    qubits: tuple[int, ...],
) -> None:
    """probabilities(qubits)[i] has bit j of i = qubits[j] (qiskit qargs order)."""
    ref = _rand(3, seed=14)
    np.testing.assert_allclose(
        _from_q(ref).probabilities(qubits),
        ref.probabilities(list(qubits)),
        atol=ATOL,
    )


@pytest.mark.statistical
def test_measure_all_qubits_statistics_and_collapse() -> None:
    """measure((0,1)) samples qiskit's distribution; s[j] = qubit j."""
    ref = _rand(2, seed=15)
    dm = _from_q(ref)
    rng = np.random.default_rng(16)
    shots = 6000
    counts: Counter[str] = Counter()
    for _ in range(shots):
        bits, post = dm.measure((0, 1), rng)
        counts[bits] += 1
        idx = int(bits[0]) + 2 * int(bits[1])
        np.testing.assert_allclose(post._data, np.diag(np.eye(4)[idx]), atol=ATOL)
    probs = ref.probabilities()
    expected = {f"{i & 1}{i >> 1}": float(probs[i]) for i in range(4)}
    assert_counts_match(dict(counts), expected, shots)
    # the original state is untouched
    np.testing.assert_allclose(dm._data, ref.data, atol=ATOL)


def test_measure_default_rng_and_deterministic_outcome() -> None:
    """rng=None works; a basis state always yields its own label."""
    dm = DensityMatrix.from_labels([("10", 1.0)])  # q1 = 1, q0 = 0
    for _ in range(5):
        bits, post = dm.measure((0, 1))
        assert bits == "01"  # s[0] = q0, s[1] = q1
        np.testing.assert_allclose(post._data, dm._data)


def test_measure_subset_collapses_correct_qubit() -> None:
    """|+>_q0 (x) |1>_q1: measuring (1,) always gives '1', q0 stays |+>."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    sv = np.kron(np.array([0, 1], dtype=complex), plus)
    dm = DensityMatrix.from_pure_state(sv)
    rng = np.random.default_rng(1)
    for _ in range(20):
        bits, post = dm.measure((1,), rng)
        assert bits == "1"
        np.testing.assert_allclose(post._data, dm._data, atol=ATOL)


def test_measure_permuted_qubits() -> None:
    """|q0=1, q1=0>: measure((1, 0)) should return s = (q1, q0) = '01'."""
    dm = DensityMatrix.from_labels([("01", 1.0)])
    bits, _ = dm.measure((1, 0), np.random.default_rng(0))
    assert bits == "01"


# =============================================================================
# copy / repr
# =============================================================================


def test_copy_is_independent() -> None:
    """copy() is a deep copy: evolving the copy leaves the original intact."""
    ref = _rand(2, seed=17)
    dm = _from_q(ref)
    cp = dm.copy()
    np.testing.assert_allclose(cp._data, dm._data)
    cp.apply_gate(np.asarray(random_unitary(2, seed=1).data), (0,))
    np.testing.assert_allclose(dm._data, ref.data, atol=ATOL)
    assert not np.allclose(cp._data, dm._data)


def test_repr() -> None:
    """repr reports qubits, dimension, purity and trace."""
    assert repr(DensityMatrix.maximally_mixed(2)) == (
        "DensityMatrix(num_qubits=2, dim=4, purity=0.250000, trace=1.000000)"
    )
    assert repr(DensityMatrix(1)) == (
        "DensityMatrix(num_qubits=1, dim=2, purity=1.000000, trace=1.000000)"
    )
