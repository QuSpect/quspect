"""Tests for SingleQubitGateFusion pass."""

from __future__ import annotations

import numpy as np

from quspect import QuantumCircuit
from quspect.dag.passes.single_qubit_fusion import SingleQubitGateFusion, _decompose_u3


class TestDecomposeU3:
    """Tests for the ZYZ / U3 decomposition helper."""

    def test_decompose_identity(self) -> None:
        """Decompose identity matrix → U3(0, 0, 0)."""
        identity_matrix = np.eye(2, dtype=complex)
        theta, phi, lam = _decompose_u3(identity_matrix)
        # Identity: θ ≈ 0
        assert abs(theta) < 1e-10

    def test_decompose_hadamard(self) -> None:
        """Decompose Hadamard matrix."""
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        theta, phi, lam = _decompose_u3(h_mat)
        # H: θ = π/2, φ = 0 (or some valid value), λ ≈ π
        assert abs(theta - np.pi / 2) < 1e-9
        # Verify: the reconstructed matrix matches up to global phase
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3

        reconstructed = gate_matrix(U3.with_params(theta, phi, lam))
        # Check unitary equivalence: overlap should be 1
        overlap = abs(np.trace(h_mat @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9

    def test_decompose_pauli_x(self) -> None:
        """Decompose Pauli-X matrix."""
        x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        theta, phi, lam = _decompose_u3(x_mat)
        # X = U3(π, π/2, -π/2) up to phase
        assert abs(theta - np.pi) < 1e-9

    def test_decompose_roundtrip(self) -> None:
        """Reconstruction should match the original matrix up to global phase."""
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3

        theta = 1.3
        phi = 2.1
        lam = 0.7
        gate = U3.with_params(theta, phi, lam)
        original = gate_matrix(gate)

        theta2, phi2, lam2 = _decompose_u3(original)
        gate2 = U3.with_params(theta2, phi2, lam2)
        reconstructed = gate_matrix(gate2)

        # Check unitary equivalence up to global phase
        # The overlap |Tr(U V†)| / 2 should be 1 if U = e^{iα} V
        overlap = abs(np.trace(original @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9

    def test_decompose_removes_global_phase_before_extracting_angles(self) -> None:
        """A global phase must not change the reconstructed U3 operation."""
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3

        original = np.exp(0.73j) * gate_matrix(U3.with_params(1.2, -0.4, 0.9))
        theta, phi, lam = _decompose_u3(original)
        reconstructed = gate_matrix(U3.with_params(theta, phi, lam))

        overlap = abs(np.trace(original @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9

    def test_decompose_x_h_x_regression(self) -> None:
        """X-H-X must fuse to an equivalent U3 despite its π global phase."""
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3
        from quspect.gates.standard import H, X

        original = gate_matrix(X) @ gate_matrix(H) @ gate_matrix(X)
        theta, phi, lam = _decompose_u3(original)
        reconstructed = gate_matrix(U3.with_params(theta, phi, lam))

        overlap = abs(np.trace(original @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9


class TestSingleQubitGateFusionBasic:
    """Basic fusion tests."""

    def test_h_h_fuses(self) -> None:
        """Two Hadamard gates should fuse (H·H = I → U3(0, 0, 0))."""
        circ = QuantumCircuit()
        circ.h(0).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        # H·H = I, which is a single U3 gate (or identity is just removed)
        # 2 H gates become 1 fused gate
        assert optimized.num_nodes() == before - 1

    def test_x_h_x_fuses_to_equivalent_operation(self) -> None:
        """The fused X-H-X block must preserve its unitary up to global phase."""
        from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
        from quspect.gates._matrix_utils import gate_matrix

        circuit = QuantumCircuit().x(0).h(0).x(0)
        optimized = SingleQubitGateFusion().run(circuit.to_dag())
        reconstructed = DAGToCircuitCompiler.compile(optimized)

        original_matrix = (
            gate_matrix(circuit[2].gate)
            @ gate_matrix(circuit[1].gate)
            @ gate_matrix(circuit[0].gate)
        )
        fused_matrix = gate_matrix(reconstructed[0].gate)
        overlap = abs(np.trace(original_matrix @ fused_matrix.conj().T)) / 2

        assert len(reconstructed) == 1
        assert reconstructed[0].gate.name == "u3"
        assert abs(overlap - 1.0) < 1e-9

    def test_h_z_s_fuses(self) -> None:
        """H, Z, S on same qubit should fuse into single U3."""
        circ = QuantumCircuit()
        circ.h(0).z(0).s(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        # 3 single-qubit gates → 1 fused gate
        assert optimized.num_nodes() == before - 2

    def test_rx_rz_fuses(self) -> None:
        """RX(0.3) followed by RZ(0.5) should fuse."""
        circ = QuantumCircuit()
        circ.rx(0.3, 0).rz(0.5, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        assert optimized.num_nodes() == before - 1

    def test_no_fuse_different_qubits(self) -> None:
        """Single-qubit gates on different qubits should NOT fuse."""
        circ = QuantumCircuit()
        circ.h(0).x(1)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        assert optimized.num_nodes() == before


class TestSingleQubitGateFusionMultiQubit:
    """Fusion in multi-qubit context."""

    def test_fuse_with_cx_barrier(self) -> None:
        """CX gate should act as barrier for fusion."""
        circ = QuantumCircuit()
        circ.h(0).rz(0.3, 0).cx(0, 1).x(0)
        dag = circ.to_dag()
        optimized = SingleQubitGateFusion().run(dag)
        # H + RZ on q0 before CX → 1 fused gate + CX + X
        # 2 IN + 3 ops + 2 OUT = 7 (was 8 with H and RZ separate)
        assert optimized.num_nodes() == 7

    def test_independent_wires_fuse(self) -> None:
        """Fusion on different wires should work independently."""
        circ = QuantumCircuit()
        circ.h(0).x(0).h(1).z(1)
        dag = circ.to_dag()
        optimized = SingleQubitGateFusion().run(dag)
        # q0: H+X fused → 1, q1: H+Z fused → 1
        # 2 IN + 2 fused + 2 OUT = 6 (was 8 with 4 single-qubit gates)
        assert optimized.num_nodes() == 6

    def test_parameterized_not_fused(self) -> None:
        """Symbolic parameters should prevent fusion."""
        from quspect import Parameter

        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).rz(0.3, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        # RX(θ) has symbolic param → don't fuse
        assert optimized.num_nodes() == before


class TestSingleQubitGateFusionDAGReconstruct:
    """Test that the fused DAG can be reconstructed to a circuit."""

    def test_fuse_and_reconstruct(self) -> None:
        """Fuse gates and verify the circuit can be reconstructed."""
        from quspect.dag.dag_to_circuit import DAGToCircuitCompiler

        circ = QuantumCircuit()
        circ.h(0).s(0).t(0).z(0)
        dag = circ.to_dag()
        optimized = SingleQubitGateFusion().run(dag)
        reconstructed = DAGToCircuitCompiler.compile(optimized)
        # 4 single-qubit gates → 1 fused U3
        assert len(reconstructed) == 1
        assert reconstructed[0].gate.name == "u3"
