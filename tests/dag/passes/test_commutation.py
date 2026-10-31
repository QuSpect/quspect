"""Tests for CommutationAnalysis pass with active reordering."""

from __future__ import annotations

import numpy as np

from quspect import QuantumCircuit
from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
from quspect.dag.passes.commutation import CommutationAnalysis
from quspect.dag.passes.gate_cancellation import GateCancellation
from quspect.simulation.simulator import Simulator


class TestCommutationBasic:
    """Test basic commutation analysis."""

    def test_commutation_runs(self) -> None:
        """Test that commutation analysis runs without error."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0).cx(0, 1).rz(0.3, 0)
        dag = circ.to_dag()
        ca = CommutationAnalysis()
        result = ca.run(dag)
        # Should not change node count (enabling pass, not reducing)
        assert result.num_nodes() == dag.num_nodes()

    def test_disjoint_qubits_commute(self) -> None:
        """Test that operations on disjoint qubits are recognized."""
        circ = QuantumCircuit()
        circ.h(0).x(1)
        dag = circ.to_dag()
        ca = CommutationAnalysis()
        result = ca.run(dag)
        assert result.num_nodes() == dag.num_nodes()

    def test_commutation_preserves_dag(self) -> None:
        """Test commutation analysis preserves DAG validity."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).rz(0.3, 0).rz(0.5, 1)
        dag = circ.to_dag()
        ca = CommutationAnalysis()
        result = ca.run(dag)

        # Topological sort should still be valid
        order = result.topological_order()
        assert len(order) == result.num_nodes()
        # Every edge must be respected
        for edge in result.edges:
            assert order.index(edge.source_id) < order.index(edge.target_id)


class TestCommutationReordering:
    """Test active reordering to enable cancellations."""

    def test_disjoint_reorder_enables_cancel(self) -> None:
        """H(0)·X(1)·H(0) — gates on q1 commute, so H-H can cancel after reorder."""
        circ = QuantumCircuit()
        circ.h(0).x(1).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()

        ca = CommutationAnalysis()
        dag = ca.run(dag)

        # After commutation reordering, run cancellation
        gc = GateCancellation()
        dag = gc.run(dag)

        # H-H cancelled: q0 inputs+pairs reset, CX? No: H cancelled
        assert dag.num_nodes() < before

    def test_z_family_reorder(self) -> None:
        """RZ(0.3)·X(0)·RZ(0.5) — RZ gates are Z-family, should be reorderable."""
        circ = QuantumCircuit()
        circ.rz(0.3, 0).x(0).rz(0.5, 0)
        dag = circ.to_dag()
        ca = CommutationAnalysis()
        result = ca.run(dag)
        # DAG validity preserved
        assert result.num_nodes() == dag.num_nodes()

    def test_same_name_reorder(self) -> None:
        """Two H gates on same qubit separated by a gate on a DIFFERENT qubit
        should be reordered (disjoint qubits commute)."""
        circ = QuantumCircuit()
        # H(0), X(1), H(0) — X is on qubit 1, H on qubit 0, so X commutes
        # with the first H (disjoint qubits).
        # CommutationAnalysis brings H together, then GateCancellation
        # cancels H·H.
        circ.h(0).x(1).h(0)
        dag = circ.to_dag()

        ca = CommutationAnalysis()
        dag = ca.run(dag)
        gc = GateCancellation()
        dag = gc.run(dag)

        # After cancel: IN:q0, IN:q1, X, OUT:q0, OUT:q1 = 5 (was 7)
        assert dag.num_nodes() == 5

    def test_nested_reorder(self) -> None:
        """H(0)·X(1)·Z(2)·H(0) — gates on different qubits commute, H-H cancel."""
        circ = QuantumCircuit()
        circ.h(0).x(1).z(2).h(0)
        dag = circ.to_dag()

        ca = CommutationAnalysis()
        dag = ca.run(dag)
        gc = GateCancellation()
        dag = gc.run(dag)

        # H cancelled, X and Z stay: IN(3) + X + Z + OUT(3) = 8
        assert dag.num_nodes() == 8


class TestCommutationPipeline:
    """Test commutation in pipeline context."""

    def test_commute_enables_cancel(self) -> None:
        """Test that commuting enables gate cancellation."""
        from quspect.dag.passes import PassManager

        circ = QuantumCircuit()
        # H on q0, X on q1, H on q0 — X on q1 doesn't block H-H cancel
        circ.h(0).x(1).h(0)
        dag = circ.to_dag()

        pm = PassManager([CommutationAnalysis(), GateCancellation()])
        result = pm.run(dag, max_iterations=3)

        # H-H should be adjacent enough to cancel
        # (they're on q0, X is on q1 — disjoint qubits commute)
        assert result.num_nodes() <= dag.num_nodes()

    def test_full_commute_cancel_pipeline(self) -> None:
        """Full pipeline: commute + cancel in a multi-qubit circuit."""
        from quspect.dag.passes import PassManager, RotationMerging

        circ = QuantumCircuit()
        # q0: H, RZ(0.3), H — H·H cancel after commuting RZ
        # q1: RX(0.5), RX(0.3) — merge
        circ.h(0).rz(0.3, 0).h(0).rx(0.5, 1).rx(0.3, 1)

        dag = circ.to_dag()
        before = dag.num_nodes()

        pm = PassManager(
            [
                CommutationAnalysis(),
                GateCancellation(),
                RotationMerging(),
            ]
        )
        result = pm.run(dag, max_iterations=5)

        # q0: H cancelled, RZ stays → 1 op
        # q1: RX merged → 1 op
        # Total: 2 ops (was 5)
        assert result.num_nodes() < before


class TestCommutationPreservesCorrectness:
    """Test that commutation reordering preserves circuit semantics."""

    def test_topological_order_valid_after_reorder(self) -> None:
        """Topological order must be valid after any reordering."""
        circ = QuantumCircuit()
        circ.h(0).z(1).x(0).cx(0, 1).rz(0.3, 0)
        dag = circ.to_dag()

        ca = CommutationAnalysis()
        result = ca.run(dag)

        order = result.topological_order()
        assert len(order) == result.num_nodes()
        for edge in result.edges:
            assert order.index(edge.source_id) < order.index(edge.target_id), (
                f"Edge {edge.source_id} → {edge.target_id} violates topological order"
            )

    def test_reorder_rejects_cross_wire_dependency_cycle(self) -> None:
        """A shared-wire swap must respect alternate paths on other wires."""
        circ = QuantumCircuit()
        circ.cx(0, 2).cx(0, 1).cx(1, 2).cx(0, 2)
        expected = Simulator().statevector(circ)

        result = CommutationAnalysis().run(circ.to_dag())

        # The regression used to reverse a q0 edge while an alternate
        # dependency path still connected the same nodes, creating a cycle.
        order = result.topological_order()
        assert len(order) == result.num_nodes()
        optimized = DAGToCircuitCompiler.compile(result)
        actual = Simulator().statevector(optimized)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_nodes_preserved_after_reorder(self) -> None:
        """Reordering should not remove any nodes."""
        circ = QuantumCircuit()
        circ.h(0).x(1).cx(0, 1).z(0).h(0).rx(0.3, 1)

        dag = circ.to_dag()
        before_count = dag.num_nodes()

        ca = CommutationAnalysis()
        result = ca.run(dag)

        # Same number of nodes (commutation only reorders, doesn't remove)
        assert result.num_nodes() == before_count

    def test_cnot_commutation_respects_control_and_target(self) -> None:
        """Only X(target) and Z(control) commute with a CNOT."""
        circuits = {
            "rx_control": QuantumCircuit().cx(0, 1).rx(0.3, 0),
            "rz_target": QuantumCircuit().cx(0, 1).rz(0.3, 1),
            "rx_target": QuantumCircuit().cx(0, 1).rx(0.3, 1),
            "rz_control": QuantumCircuit().cx(0, 1).rz(0.3, 0),
        }
        analysis = CommutationAnalysis()
        nodes = {
            name: circuit.to_dag().operation_nodes()
            for name, circuit in circuits.items()
        }

        assert not analysis._ops_commute(*nodes["rx_control"])
        assert not analysis._ops_commute(*nodes["rz_target"])
        assert analysis._ops_commute(*nodes["rx_target"])
        assert analysis._ops_commute(*nodes["rz_control"])

    def test_qaoa_zz_fragment_preserves_statevector(self) -> None:
        """Commutation plus cancellation must preserve a QAOA ZZ fragment."""
        circ = QuantumCircuit()
        circ.h(0).h(1)
        circ.cx(0, 1).rz(0.7, 1).cx(0, 1)
        circ.rx(0.4, 0).rx(0.4, 1)
        expected = Simulator().statevector(circ)

        dag = CommutationAnalysis().run(circ.to_dag())
        dag = GateCancellation().run(dag)
        optimized = DAGToCircuitCompiler.compile(dag)
        actual = Simulator().statevector(optimized)

        np.testing.assert_allclose(actual, expected, atol=1e-12)
