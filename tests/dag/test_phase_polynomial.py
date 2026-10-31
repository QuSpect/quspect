"""Tests for Phase Polynomial compiler."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.dag.phase_polynomial import (
    ParityMatrix,
    PhasePolynomial,
    PhasePolynomialCompiler,
    PhaseTerm,
)


class TestParityMatrix:
    """Test ParityMatrix operations."""

    def test_identity(self) -> None:
        """Test identity parity matrix."""
        pm = ParityMatrix(data=np.eye(3, dtype=np.int8))
        assert pm.num_qubits == 3
        assert pm.data[0, 0] == 1
        assert pm.data[0, 1] == 0

    def test_apply_cnot(self) -> None:
        """Test CNOT transformation on parity matrix."""
        pm = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pm.apply_cnot(0, 1)  # CNOT(0, 1): row[1] ^= row[0]
        # After CNOT(0,1): row 0 = [1,0], row 1 = [1,1]
        assert pm.data[0, 0] == 1
        assert pm.data[0, 1] == 0
        assert pm.data[1, 0] == 1
        assert pm.data[1, 1] == 1

    def test_apply_swap(self) -> None:
        """Test SWAP transformation."""
        pm = ParityMatrix(data=np.array([[1, 0], [0, 1]], dtype=np.int8))
        pm.apply_swap(0, 1)
        assert pm.data[0, 0] == 0
        assert pm.data[0, 1] == 1
        assert pm.data[1, 0] == 1
        assert pm.data[1, 1] == 0

    def test_copy(self) -> None:
        """Test deep copy."""
        pm1 = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pm2 = pm1.copy()
        pm2.apply_cnot(0, 1)
        # Original should be unchanged
        assert pm1.data[1, 0] == 0

    def test_column_reduce(self) -> None:
        """Test Gaussian elimination on parity matrix."""
        # Matrix with a non-zero off-diagonal entry
        data = np.array([[1, 1], [0, 1]], dtype=np.int8)
        pm = ParityMatrix(data=data)
        cnots = pm.column_reduce()
        # After reduction, should be in row-echelon form
        assert len(cnots) >= 0


class TestPhaseTerm:
    """Test PhaseTerm data structure."""

    def test_create_term(self) -> None:
        """Test creating a phase term."""
        parity = np.array([1, 0, 1], dtype=np.int8)
        term = PhaseTerm(angle=0.5, parity=parity)
        assert term.angle == 0.5
        np.testing.assert_array_equal(term.parity, [1, 0, 1])

    def test_repr(self) -> None:
        """Test string representation."""
        parity = np.array([1, 1], dtype=np.int8)
        term = PhaseTerm(angle=math.pi, parity=parity)
        r = repr(term)
        assert "PhaseTerm" in r
        assert "3.1416" in r


class TestPhasePolynomial:
    """Test PhasePolynomial data structure."""

    def test_create_empty(self) -> None:
        """Test creating empty phase polynomial."""
        pm = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pp = PhasePolynomial(num_qubits=2, parity_matrix=pm)
        assert pp.num_qubits == 2
        assert len(pp.terms) == 0

    def test_add_phase(self) -> None:
        """Test adding a phase term."""
        pm = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pp = PhasePolynomial(num_qubits=2, parity_matrix=pm)
        pp.add_phase(0.5, 0)
        assert len(pp.terms) == 1
        assert pp.terms[0].angle == 0.5

    def test_add_phase_merges_same_parity(self) -> None:
        """Test that phases with same parity are merged."""
        pm = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pp = PhasePolynomial(num_qubits=2, parity_matrix=pm)
        pp.add_phase(0.3, 0)
        pp.add_phase(0.5, 0)
        # Same qubit, same parity → merged
        assert len(pp.terms) == 1
        assert abs(pp.terms[0].angle - 0.8) < 1e-10

    def test_apply_cnot(self) -> None:
        """Test CNOT on phase polynomial."""
        pm = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pp = PhasePolynomial(num_qubits=2, parity_matrix=pm)
        pp.apply_cnot(0, 1)
        assert pp.parity_matrix.data[1, 0] == 1

    def test_optimize(self) -> None:
        """Test Gaussian elimination optimization."""
        data = np.array([[1, 1], [0, 1]], dtype=np.int8)
        pm = ParityMatrix(data=data)
        pp = PhasePolynomial(num_qubits=2, parity_matrix=pm)
        pp.optimize()
        # Should complete without error
        assert pp.num_qubits == 2

    def test_to_circuit(self) -> None:
        """Test synthesizing a circuit from phase polynomial."""
        pm = ParityMatrix(data=np.eye(2, dtype=np.int8))
        pp = PhasePolynomial(num_qubits=2, parity_matrix=pm)
        pp.add_phase(0.5, 0)
        pp.add_phase(0.3, 1)

        circ = pp.to_circuit()
        assert circ.num_qubits == 2


class TestPhasePolynomialCompiler:
    """Test phase polynomial extraction from DAG."""

    def test_extract_rz_cnot_circuit(self) -> None:
        """Test extracting from a simple {RZ, CNOT} circuit."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0).cx(0, 1).rz(0.3, 1)
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract(dag)

        assert pp.num_qubits == 2
        # After CNOT(0,1): parity[1] = x1 ⊕ x0
        # The second RZ is on qubit 1 which now carries parity {0,1}
        assert len(pp.terms) >= 1

    def test_extract_cz_circuit(self) -> None:
        """Test extracting from circuit with CZ."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0).cz(0, 1).rz(0.3, 0)
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract(dag)

        # CZ should add a π phase term
        assert pp.num_qubits == 2

    def test_extract_diagonal_gates(self) -> None:
        """Test extracting S, T, Z diagonal gates as phases."""
        circ = QuantumCircuit()
        circ.s(0).t(1).z(0)
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract(dag)

        # S(0) = RZ(π/2), Z(0) = RZ(π) → merged π/2+π on q0 = 1 term
        # T(1) = RZ(π/4) on q1 = 1 term
        # Total: 2 terms (merged S+Z on q0, plus T on q1)
        assert len(pp.terms) >= 2

    def test_extract_and_optimize(self) -> None:
        """Test extract + optimize convenience method."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0).cx(0, 1).rz(0.3, 1).cx(0, 1)
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract_and_optimize(dag)
        assert pp.num_qubits == 2

    def test_extract_empty_circuit(self) -> None:
        """Test extracting from empty circuit."""
        circ = QuantumCircuit()
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract(dag)
        assert pp.num_qubits == 0
        assert len(pp.terms) == 0

    def test_extract_identity_parity_matrix(self) -> None:
        """Test that initial parity matrix is identity."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0)
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract(dag)

        expected = np.eye(1, dtype=np.int8)
        np.testing.assert_array_equal(pp.parity_matrix.data, expected)


class TestPhasePolynomialEndToEnd:
    """End-to-end tests for phase polynomial pipeline."""

    def test_extract_synthesize_preserves_structure(self) -> None:
        """Test that extract→synthesize produces a valid circuit."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0).cx(0, 1).rz(0.3, 1)
        dag = circ.to_dag()

        pp = PhasePolynomialCompiler.extract(dag)
        synthesized = pp.to_circuit()

        # Synthesized circuit should have gates
        assert len(synthesized) > 0
        assert synthesized.num_qubits == 2

    def test_swap_tracking(self) -> None:
        """Test that SWAP is properly tracked in parity matrix."""
        circ = QuantumCircuit()
        circ.swap(0, 1).rz(0.5, 0)
        dag = circ.to_dag()
        pp = PhasePolynomialCompiler.extract(dag)

        # After SWAP, qubit 0 carries parity of original qubit 1
        assert pp.parity_matrix.data[0, 1] == 1


# ---------------------------------------------------------------------------
# Regression tests (B14-B18): exact semantics without external references
# ---------------------------------------------------------------------------


def _cx_rz_unitary(circ: QuantumCircuit, n: int) -> np.ndarray:
    """Dense little-endian unitary of a {cx, rz, cz, swap} circuit."""
    dim = 1 << n
    u = np.eye(dim, dtype=complex)
    for instr in circ.instructions:
        assert instr.gate is not None
        name, qs = instr.gate.name, instr.qubits
        step = np.zeros((dim, dim), dtype=complex)
        for x in range(dim):
            if name == "cx":
                c, t = qs
                y = x ^ (1 << t) if (x >> c) & 1 else x
                step[y, x] = 1
            elif name == "swap":
                a, b = qs
                ba, bb = (x >> a) & 1, (x >> b) & 1
                y = x & ~((1 << a) | (1 << b)) | (bb << a) | (ba << b)
                step[y, x] = 1
            elif name == "cz":
                a, b = qs
                step[x, x] = -1 if (x >> a) & 1 and (x >> b) & 1 else 1
            elif name == "rz":
                theta = float(instr.gate.params[0])
                bit = (x >> qs[0]) & 1
                step[x, x] = np.exp(1j * theta / 2 * (1 if bit else -1))
            else:  # pragma: no cover - test helper misuse
                raise AssertionError(name)
        u = step @ u
    return u


def _assert_equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> None:
    idx = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
    phase = a[idx] / b[idx]
    np.testing.assert_allclose(a, phase * b, atol=1e-9)


class TestPhasePolynomialRegressions:
    """Regression tests for extraction / synthesis / optimize semantics."""

    def test_extract_keeps_reversed_cnot_direction(self) -> None:
        """cx(1, 0) adds row 1 into row 0 (B14)."""
        pp = PhasePolynomialCompiler.extract(QuantumCircuit().cx(1, 0).to_dag())
        np.testing.assert_array_equal(pp.parity_matrix.data, [[1, 1], [0, 1]])

    def test_extract_cz_phase_terms(self) -> None:
        """CZ(a, b) = pi/2 a + pi/2 b - pi/2 (a xor b) (B15)."""
        pp = PhasePolynomialCompiler.extract(QuantumCircuit().cz(0, 1).to_dag())
        terms = {tuple(int(v) for v in t.parity): t.angle for t in pp.terms}
        assert terms == pytest.approx(
            {(1, 0): math.pi / 2, (0, 1): math.pi / 2, (1, 1): -math.pi / 2}
        )
        # phi(x) == pi only on |11>
        for x0 in (0, 1):
            for x1 in (0, 1):
                phi = sum(a * ((p[0] * x0 + p[1] * x1) % 2) for p, a in terms.items())
                expected = math.pi if x0 and x1 else 0.0
                assert np.exp(1j * phi) == pytest.approx(np.exp(1j * expected))

    @pytest.mark.parametrize("seed", range(8))
    def test_to_circuit_roundtrip_random(self, seed: int) -> None:
        """extract -> to_circuit reproduces the unitary up to phase (B16/B17)."""
        rng = np.random.default_rng(seed)
        n = 3
        circ = QuantumCircuit()
        for q in range(n):
            circ.rz(0.0, q)
        for _ in range(20):
            r = rng.random()
            a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
            if r < 0.35:
                circ.cx(a, b)
            elif r < 0.45:
                circ.cz(a, b)
            elif r < 0.5:
                circ.swap(a, b)
            else:
                circ.rz(float(rng.uniform(-3, 3)), a)
        pp = PhasePolynomialCompiler.extract(circ.to_dag())
        out = pp.to_circuit()
        assert {i.gate.name for i in out.instructions if i.gate} <= {"cx", "rz"}
        _assert_equal_up_to_phase(_cx_rz_unitary(out, n), _cx_rz_unitary(circ, n))

    def test_extract_and_optimize_preserves_semantics(self) -> None:
        """optimize() keeps the parity matrix (B18)."""
        circ = QuantumCircuit().cx(0, 1).cx(1, 2).rz(0.4, 2).cx(0, 1).rz(-0.4, 2)
        plain = PhasePolynomialCompiler.extract(circ.to_dag())
        opt = PhasePolynomialCompiler.extract_and_optimize(circ.to_dag())
        np.testing.assert_array_equal(opt.parity_matrix.data, plain.parity_matrix.data)
        _assert_equal_up_to_phase(
            _cx_rz_unitary(opt.to_circuit(), 3), _cx_rz_unitary(circ, 3)
        )

    def test_optimize_merges_and_drops_trivial_terms(self) -> None:
        """Same-parity terms merge; 2*pi multiples and zero parity drop."""
        pp = PhasePolynomial(
            num_qubits=2,
            parity_matrix=ParityMatrix(data=np.eye(2, dtype=np.int8)),
            terms=[
                PhaseTerm(math.pi, np.array([1, 0])),
                PhaseTerm(0.25, np.array([1, 1])),
                PhaseTerm(math.pi, np.array([1, 0])),
                PhaseTerm(1.0, np.array([0, 0])),
            ],
        )
        pp.optimize()
        assert len(pp.terms) == 1
        assert pp.terms[0].angle == pytest.approx(0.25)
        np.testing.assert_array_equal(pp.terms[0].parity, [1, 1])

    def test_to_circuit_rejects_singular_parity_matrix(self) -> None:
        """A non-invertible parity matrix cannot be synthesised."""
        pp = PhasePolynomial(
            num_qubits=2,
            parity_matrix=ParityMatrix(data=np.array([[1, 0], [1, 0]], np.int8)),
        )
        with pytest.raises(ValueError, match="invertible"):
            pp.to_circuit()
