"""Cross-validate ``quspect.dag.phase_polynomial`` with independent numpy math.

For a circuit built from CNOT / SWAP and diagonal gates (RZ, P, U1, T, Tdg, S,
Sdg, Z, CZ) the unitary acts as ``U |x> = exp(i phi(x)) |A x>`` where ``A`` is
a GF(2) matrix and ``phi(x) = sum_k angle_k * (c_k . x mod 2)`` up to a
global constant.  ``A`` and ``phi`` are read off the qiskit ``Operator`` of
the circuit (little-endian, bit ``j`` of the basis index = variable ``x_j``)
and compared with ``PhasePolynomial.parity_matrix`` / ``terms``.  Synthesised
circuits (``PhasePolynomial.to_circuit``) are compared with the original
unitary up to global phase.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import quspect
from qiskit_tests._dag_helpers import phase_function
from qiskit_tests.helpers import (
    assert_unitary_equal,
    qiskit_circuit_unitary,
    quspect_circuit_unitary,
)
from quspect.circuit.parameter import Parameter
from quspect.dag import (
    PhasePolynomial,
    PhasePolynomialCompiler,
    PhaseTerm,
)
from quspect.dag.phase_polynomial import ParityMatrix

pytestmark = pytest.mark.qiskit_ref

_DIAG_GATES = ("rz", "p", "u1", "t", "tdg", "s", "sdg", "z")


# =============================================================================
# Independent reference
# =============================================================================


def _affine_decomposition(
    u: NDArray[np.complex128], n: int
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Return ``(A, phi)`` with ``U|x> = e^{i phi(x)} |A x>``, ``phi(0) = 0``.

    Raises AssertionError if ``U`` is not a phased permutation of that form.
    """
    dim = 1 << n
    images = []
    phases = np.zeros(dim)
    for x in range(dim):
        col = u[:, x]
        y = int(np.argmax(np.abs(col)))
        assert abs(abs(col[y]) - 1.0) < 1e-9, "not a phased permutation"
        images.append(y)
        phases[x] = float(np.angle(col[y]))
    a = np.zeros((n, n), dtype=np.int8)
    for j in range(n):
        y = images[1 << j]
        for i in range(n):
            a[i, j] = (y >> i) & 1
    for x in range(dim):  # linearity over GF(2)
        bits = np.array([(x >> j) & 1 for j in range(n)])
        img = (a.astype(int) @ bits) % 2
        assert images[x] == sum(int(b) << i for i, b in enumerate(img))
    return a, phases - phases[0]


def _wrap(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map angles to (-pi, pi]."""
    return np.angle(np.exp(1j * angles))


def _assert_phases_match(
    pp: PhasePolynomial, circ: quspect.QuantumCircuit, n: int
) -> None:
    """Parity matrix and phase terms reproduce the circuit unitary."""
    a, phi = _affine_decomposition(qiskit_circuit_unitary(circ, num_qubits=n), n)
    np.testing.assert_array_equal(pp.parity_matrix.data.astype(np.int8), a)
    mine = phase_function(((t.angle, t.parity) for t in pp.terms), n)
    np.testing.assert_allclose(_wrap(mine - mine[0] - phi), 0.0, atol=1e-9)


def _random_cnot_diag_circuit(
    n: int, depth: int, seed: int, *, ordered_cx: bool
) -> quspect.QuantumCircuit:
    """Random {CX, SWAP, diagonal} circuit; ``ordered_cx`` forces control<target."""
    rng = np.random.default_rng(seed)
    circ = quspect.QuantumCircuit().id(n - 1)
    for _ in range(depth):
        r = rng.random()
        if r < 0.4 and n > 1:
            a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
            if ordered_cx:
                a, b = min(a, b), max(a, b)
            circ.cx(a, b)
        elif r < 0.45 and n > 1:
            a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
            circ.swap(a, b)
        else:
            name = _DIAG_GATES[int(rng.integers(len(_DIAG_GATES)))]
            q = int(rng.integers(n))
            if name in ("rz", "p", "u1"):
                getattr(circ, name)(float(rng.uniform(-math.pi, math.pi)), q)
            else:
                getattr(circ, name)(q)
    return circ


# =============================================================================
# Data structures
# =============================================================================


def test_phase_term_repr() -> None:
    """PhaseTerm repr lists the parity variables."""
    assert repr(PhaseTerm(0.5, np.array([1, 0, 1]))) == "PhaseTerm(0.5000 · [x0 ⊕ x2])"
    assert repr(PhaseTerm(0.25, np.array([0, 0]))) == "PhaseTerm(0.2500 · [0])"


def test_parity_matrix_row_operations() -> None:
    """apply_cnot adds the control row to the target row; apply_swap swaps rows."""
    pm = ParityMatrix(data=np.eye(3, dtype=np.int8))
    assert pm.num_qubits == 3
    pm.apply_cnot(0, 2)
    np.testing.assert_array_equal(pm.data, [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    cp = pm.copy()
    pm.apply_swap(1, 2)
    np.testing.assert_array_equal(pm.data, [[1, 0, 0], [1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(cp.data, [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    assert repr(cp).splitlines() == [
        "ParityMatrix:",
        "  [1 0 0]",
        "  [0 1 0]",
        "  [1 0 1]",
    ]


@pytest.mark.parametrize("seed", range(20))
def test_column_reduce_synthesises_the_parity_matrix(seed: int) -> None:
    """Reversed column_reduce CNOTs implement the matrix (checked with qiskit)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    while True:
        mat = rng.integers(0, 2, size=(n, n)).astype(np.int8)
        if round(abs(np.linalg.det(mat))) % 2 == 1:
            break
    pm = ParityMatrix(data=mat.copy())
    cnots = pm.column_reduce()
    np.testing.assert_array_equal(pm.data, np.eye(n, dtype=np.int8))
    circ = quspect.QuantumCircuit().id(n - 1)
    for c, t in reversed(cnots):
        circ.cx(c, t)
    a, phi = _affine_decomposition(qiskit_circuit_unitary(circ, num_qubits=n), n)
    np.testing.assert_array_equal(a, mat)
    np.testing.assert_allclose(phi, 0.0, atol=1e-12)


def test_column_reduce_singular_matrix_skips_missing_pivot() -> None:
    """A rank-deficient matrix leaves a zero column (no pivot found)."""
    pm = ParityMatrix(data=np.array([[1, 0], [1, 0]], dtype=np.int8))
    assert pm.column_reduce() == [(0, 1)]
    np.testing.assert_array_equal(pm.data, [[1, 0], [0, 0]])


def test_add_phase_merges_and_skips_zero() -> None:
    """Same-parity phases merge; |angle| < 1e-12 is dropped."""
    pp = PhasePolynomial(2, ParityMatrix(data=np.eye(2, dtype=np.int8)))
    pp.add_phase(0.3, 0)
    pp.add_phase(1e-14, 1)
    pp.add_phase(0.2, 0)
    assert len(pp.terms) == 1 and pp.terms[0].angle == pytest.approx(0.5)
    pp.apply_cnot(0, 1)
    pp.add_phase(0.7, 1)
    np.testing.assert_array_equal(pp.terms[1].parity, [1, 1])


# =============================================================================
# Extraction vs independent numpy reference
# =============================================================================


@pytest.mark.parametrize("name", _DIAG_GATES)
@pytest.mark.parametrize("q", [0, 2])
def test_single_diagonal_gate_term(name: str, q: int) -> None:
    """Every diagonal gate yields one term on x_q with the right angle."""
    circ = quspect.QuantumCircuit().id(2)
    if name in ("rz", "p", "u1"):
        getattr(circ, name)(0.731, q)
    else:
        getattr(circ, name)(q)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    assert len(pp.terms) == 1
    np.testing.assert_array_equal(pp.terms[0].parity, np.eye(3, dtype=int)[q])
    _assert_phases_match(pp, circ, 3)


@pytest.mark.parametrize(("n", "seed"), [(n, s) for n in (2, 3, 4) for s in range(8)])
def test_extract_matches_numpy_ordered_cnots(n: int, seed: int) -> None:
    """CX(control<target)+SWAP+diag: parity matrix and phase function exact."""
    circ = _random_cnot_diag_circuit(n, 25, 31 * n + seed, ordered_cx=True)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    _assert_phases_match(pp, circ, n)


def test_extract_ignores_non_diagonal_and_symbolic_gates() -> None:
    """H / RX and symbolic RZ do not contribute terms or parity changes."""
    theta = Parameter("theta")
    circ = quspect.QuantumCircuit().h(0).rx(0.3, 1).rz(theta, 0).t(1)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    assert len(pp.terms) == 1
    np.testing.assert_array_equal(pp.terms[0].parity, [0, 1])
    np.testing.assert_array_equal(pp.parity_matrix.data, np.eye(2))


def test_extract_reversed_cnot_bug() -> None:
    """cx(1, 0) must add row 1 into row 0: A = [[1, 1], [0, 1]]."""
    circ = quspect.QuantumCircuit().cx(1, 0)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    np.testing.assert_array_equal(pp.parity_matrix.data, [[1, 1], [0, 1]])


def test_extract_cz_phase_bug() -> None:
    """CZ phase function must be pi on |11> only."""
    circ = quspect.QuantumCircuit().cz(0, 1)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    _assert_phases_match(pp, circ, 2)


@pytest.mark.parametrize("seed", range(3))
def test_extract_matches_numpy_general_bug(seed: int) -> None:
    """Unrestricted CX direction + CZ: extraction should still be exact."""
    circ = _random_cnot_diag_circuit(3, 25, seed, ordered_cx=False)
    circ.cz(2, 0).cx(2, 1)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    _assert_phases_match(pp, circ, 3)


# =============================================================================
# Synthesis
# =============================================================================


@pytest.mark.parametrize("seed", range(6))
def test_to_circuit_diagonal_only(seed: int) -> None:
    """Diagonal-only circuits resynthesise to the same unitary (up to phase)."""
    rng = np.random.default_rng(seed)
    circ = quspect.QuantumCircuit().id(3)
    for _ in range(20):
        name = _DIAG_GATES[int(rng.integers(len(_DIAG_GATES)))]
        q = int(rng.integers(4))
        if name in ("rz", "p", "u1"):
            getattr(circ, name)(float(rng.uniform(-3, 3)), q)
        else:
            getattr(circ, name)(q)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    assert len(pp.terms) <= 4  # same-parity terms merged
    out = pp.to_circuit()
    assert out.name == "phase_poly_synth"
    assert {i.gate.name for i in out.instructions if i.gate} <= {"rz"}
    assert_unitary_equal(
        quspect_circuit_unitary(out, num_qubits=4),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


@pytest.mark.parametrize("seed", range(10))
def test_to_circuit_cnot_only(seed: int) -> None:
    """CX(control<target)-only circuits resynthesise exactly."""
    rng = np.random.default_rng(seed)
    n = 4
    circ = quspect.QuantumCircuit().id(n - 1)
    for _ in range(10):
        a, b = sorted(int(q) for q in rng.choice(n, size=2, replace=False))
        circ.cx(a, b)
    out = PhasePolynomialCompiler.extract(circ.to_dag()).to_circuit()
    assert_unitary_equal(
        quspect_circuit_unitary(out, num_qubits=n), qiskit_circuit_unitary(circ)
    )


def test_to_circuit_phase_on_unchanged_control() -> None:
    """Phase on a CNOT control (row unchanged by synthesis) is correct."""
    circ = quspect.QuantumCircuit().t(0).cx(0, 1).s(0)
    out = PhasePolynomialCompiler.extract(circ.to_dag()).to_circuit()
    assert_unitary_equal(
        quspect_circuit_unitary(out, num_qubits=2),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


_SYNTH_BUG_CASES = {
    "rz_cx_rz": quspect.QuantumCircuit().rz(0.5, 0).cx(0, 1).rz(0.3, 1),
    "cx_rz_cx": quspect.QuantumCircuit().cx(0, 1).rz(0.3, 1).cx(0, 1),
    "cx_cx_t": quspect.QuantumCircuit().cx(0, 1).cx(1, 2).t(2),
    "swap_t": quspect.QuantumCircuit().swap(0, 1).t(0),
}


@pytest.mark.parametrize("case", list(_SYNTH_BUG_CASES))
def test_to_circuit_general_bug(case: str) -> None:
    """Extracted polynomial (which is correct here) must resynthesise exactly."""
    circ = _SYNTH_BUG_CASES[case]
    n = circ.num_qubits
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    _assert_phases_match(pp, circ, n)  # extraction itself is right
    assert_unitary_equal(
        quspect_circuit_unitary(pp.to_circuit(), num_qubits=n),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


def test_optimize_keeps_parity_matrix_and_simplifies_terms() -> None:
    """optimize() keeps the linear part and merges / drops trivial terms."""
    circ = quspect.QuantumCircuit().cx(0, 1).t(1).cx(1, 2)
    pp = PhasePolynomialCompiler.extract(circ.to_dag())
    terms = [(t.angle, t.parity.copy()) for t in pp.terms]
    opt = PhasePolynomialCompiler.extract_and_optimize(circ.to_dag())
    np.testing.assert_array_equal(opt.parity_matrix.data, pp.parity_matrix.data)
    assert [(t.angle, list(t.parity)) for t in opt.terms] == [
        (a, list(p)) for a, p in terms
    ]
    raw = PhasePolynomial(
        2,
        ParityMatrix(data=np.eye(2, dtype=np.int8)),
        [
            PhaseTerm(0.3, np.array([1, 1])),
            PhaseTerm(math.pi, np.array([1, 0])),
            PhaseTerm(0.2, np.array([1, 1])),
            PhaseTerm(math.pi, np.array([1, 0])),
            PhaseTerm(0.7, np.array([0, 0])),
        ],
    )
    raw.optimize()
    assert [(round(t.angle, 12), list(t.parity)) for t in raw.terms] == [(0.5, [1, 1])]


def test_extract_and_optimize_preserves_semantics_bug() -> None:
    """extract_and_optimize(dag).to_circuit() must still implement the circuit."""
    circ = quspect.QuantumCircuit().cx(0, 1).cx(1, 2)
    opt = PhasePolynomialCompiler.extract_and_optimize(circ.to_dag())
    assert_unitary_equal(
        quspect_circuit_unitary(opt.to_circuit(), num_qubits=3),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )
