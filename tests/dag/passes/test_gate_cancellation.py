"""Tests for GateCancellation pass."""

from __future__ import annotations

from quspect import QuantumCircuit
from quspect.dag.passes.gate_cancellation import GateCancellation


class TestGateCancellationBasic:
    """Test basic cancellations."""

    def test_hh_cancels(self) -> None:
        """Test that two consecutive H gates on same qubit cancel."""
        circ = QuantumCircuit()
        circ.h(0).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        # Should remove 2 H gate nodes
        assert optimized.num_nodes() == before - 2

    def test_xx_cancels(self) -> None:
        """Test that two consecutive X gates cancel."""
        circ = QuantumCircuit()
        circ.x(0).x(0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2  # IN + OUT only

    def test_zz_cancels(self) -> None:
        """Test that two consecutive Z gates cancel."""
        circ = QuantumCircuit()
        circ.z(0).z(0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2

    def test_cxcx_cancels(self) -> None:
        """Test that two consecutive CNOT gates cancel."""
        circ = QuantumCircuit()
        circ.cx(0, 1).cx(0, 1)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        # Should remove 2 CNOT nodes
        assert optimized.num_nodes() == 4  # 2 inputs + 2 outputs

    def test_ssdg_cancels(self) -> None:
        """Test S followed by SDG cancels."""
        circ = QuantumCircuit()
        circ.s(0).sdg(0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2

    def test_ttdg_cancels(self) -> None:
        """Test T followed by TDG cancels."""
        circ = QuantumCircuit()
        circ.t(0).tdg(0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2


class TestGateCancellationNoCancel:
    """Test cases where cancellation should NOT happen."""

    def test_different_qubits_no_cancel(self) -> None:
        """Test H on different qubits doesn't cancel."""
        circ = QuantumCircuit()
        circ.h(0).h(1)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == before  # No change

    def test_gate_between_no_cancel(self) -> None:
        """Test H-X-H doesn't cancel the H gates."""
        circ = QuantumCircuit()
        circ.h(0).x(0).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        # X between H gates prevents cancellation
        assert optimized.num_nodes() == before

    def test_different_gates_no_cancel(self) -> None:
        """Test X-Z doesn't cancel."""
        circ = QuantumCircuit()
        circ.x(0).z(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == before


class TestParameterizedGateCancellation:
    """Test cancellation of parameterized gates (RX, RY, RZ, P, CRX, etc.)."""

    def test_rx_positive_negative_cancels(self) -> None:
        """RX(θ)·RX(-θ) should cancel."""
        circ = QuantumCircuit()
        circ.rx(0.5, 0).rx(-0.5, 0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2  # IN + OUT only

    def test_ry_positive_negative_cancels(self) -> None:
        """RY(θ)·RY(-θ) should cancel."""
        circ = QuantumCircuit()
        circ.ry(0.8, 0).ry(-0.8, 0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2

    def test_rz_sum_zero_cancels(self) -> None:
        """RZ(0.3) + RZ(-0.3) should cancel."""
        circ = QuantumCircuit()
        circ.rz(0.3, 0).rz(-0.3, 0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2

    def test_p_sum_zero_cancels(self) -> None:
        """Phase(π) + Phase(-π) should cancel."""
        import math

        circ = QuantumCircuit()
        circ.p(math.pi, 0).p(-math.pi, 0)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 2

    def test_rx_different_params_no_cancel(self) -> None:
        """RX(0.3)·RX(0.5) should NOT cancel (sum != 0)."""
        circ = QuantumCircuit()
        circ.rx(0.3, 0).rx(0.5, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == before

    def test_crx_cancels(self) -> None:
        """CRX(θ)·CRX(-θ) should cancel."""
        circ = QuantumCircuit()
        circ.crx(0.5, 0, 1).crx(-0.5, 0, 1)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 4  # 2 IN + 2 OUT

    def test_crz_cancels(self) -> None:
        """CRZ(θ)·CRZ(-θ) should cancel."""
        circ = QuantumCircuit()
        circ.crz(1.0, 0, 1).crz(-1.0, 0, 1)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 4

    def test_cry_cancels(self) -> None:
        """CRY(θ)·CRY(-θ) should cancel."""
        circ = QuantumCircuit()
        circ.cry(0.7, 0, 1).cry(-0.7, 0, 1)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        assert optimized.num_nodes() == 4

    def test_parameterized_not_cancelled_with_symbolic(self) -> None:
        """Symbolic parameters should NOT be cancelled."""
        from quspect import Parameter

        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).rx(0.5, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        # Symbolic param prevents cancellation
        assert optimized.num_nodes() == before


class TestGateCancellationMultiQubit:
    """Test cancellations in multi-qubit circuits."""

    def test_bell_with_redundant_h(self) -> None:
        """Test cancellations in a circuit with extra H."""
        circ = QuantumCircuit()
        circ.h(0).h(0).cx(0, 1)  # First two H cancel
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        # After H-H cancel: IN:q0, IN:q1, CX, OUT:q0, OUT:q1 = 5
        # (both H gates are removed, leaving only CX)
        assert optimized.num_nodes() == 5

    def test_multiple_cancellations(self) -> None:
        """Test multiple independent cancellations."""
        circ = QuantumCircuit()
        circ.h(0).h(0).x(1).x(1)
        dag = circ.to_dag()
        optimized = GateCancellation().run(dag)
        # All gates cancel: 2 inputs + 2 outputs = 4
        assert optimized.num_nodes() == 4

    def test_barrier_blocks_cancel(self) -> None:
        """Test that barrier prevents cancellation."""
        circ = QuantumCircuit()
        circ.h(0).barrier(0).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = GateCancellation().run(dag)
        # Barrier node between H gates prevents cancellation
        assert optimized.num_nodes() == before

    def test_cx_rz_target_cx_does_not_cancel(self) -> None:
        """The standard ZZ decomposition must retain both CNOT gates."""
        circ = QuantumCircuit()
        circ.cx(0, 1).rz(0.7, 1).cx(0, 1)
        dag = circ.to_dag()

        optimized = GateCancellation().run(dag)

        names = [
            node.instruction.gate.name
            for node in optimized.operation_nodes()
            if node.instruction is not None and node.instruction.gate is not None
        ]
        assert names == ["cx", "rz", "cx"]

    def test_reversed_cx_operands_do_not_cancel(self) -> None:
        """Control and target order is significant for CNOT cancellation."""
        circ = QuantumCircuit()
        circ.cx(0, 1).cx(1, 0)
        dag = circ.to_dag()

        optimized = GateCancellation().run(dag)

        assert len(optimized.operation_nodes()) == 2

    def test_multiqubit_cancel_reconnects_every_wire(self) -> None:
        """Removing a CNOT pair must leave valid chains on both qubits."""
        circ = QuantumCircuit()
        circ.h(0).x(1).cx(0, 1).cx(0, 1).z(0).y(1)

        optimized = GateCancellation().run(circ.to_dag())

        assert len(optimized.topological_order()) == optimized.num_nodes()
        assert [len(optimized.nodes_on_wire(f"q{q}")) for q in range(2)] == [4, 4]
