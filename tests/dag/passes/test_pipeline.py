"""End-to-end pipeline integration tests for optimization passes.

These tests verify the complete optimization pipeline:
    Circuit → DAG → (optimize) → DAG → Circuit
"""

from __future__ import annotations

from quspect import Parameter, QuantumCircuit
from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
from quspect.dag.passes.base import PassManager
from quspect.dag.passes.commutation import CommutationAnalysis
from quspect.dag.passes.diagonal_before_measure import (
    EliminateDiagonalBeforeMeasurement,
)
from quspect.dag.passes.gate_cancellation import GateCancellation
from quspect.dag.passes.rotation_merging import RotationMerging
from quspect.dag.passes.single_qubit_fusion import SingleQubitGateFusion


def _build_default_pipeline() -> PassManager:
    """Build the recommended optimization pipeline."""
    return PassManager(
        [
            GateCancellation(),
            CommutationAnalysis(),
            RotationMerging(),
            SingleQubitGateFusion(),
            EliminateDiagonalBeforeMeasurement(),
        ]
    )


class TestPipelineEndToEnd:
    """Full pipeline: Circuit → DAG → optimize → DAG → Circuit."""

    def test_bell_state_no_change(self) -> None:
        """Optimized Bell state should be unchanged (no redundant gates)."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=3)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag, name="bell_opt")
        assert reconstructed.num_qubits == 2
        assert len(reconstructed) == 2

    def test_redundant_h_cancels(self) -> None:
        """Redundant H·H pair should cancel."""
        circ = QuantumCircuit()
        circ.h(0).h(0).cx(0, 1)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=3)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag)
        # H·H cancelled → only CX remains
        assert len(reconstructed) == 1
        assert reconstructed[0].gate.name == "cx"

    def test_rotation_merging_in_pipeline(self) -> None:
        """Rotation merging should work in pipeline context."""
        circ = QuantumCircuit()
        circ.rx(0.3, 0).rx(0.7, 0).cx(0, 1)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=3)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag)
        # RX(0.3)·RX(0.7) → RX(1.0), then fusion may → U3
        assert len(reconstructed) <= 2  # fused single-qubit + CX

    def test_full_cleanup(self) -> None:
        """Circuit with multiple optimization opportunities."""
        circ = QuantumCircuit()
        # H·H cancel
        circ.h(0).h(0)
        # RX merge
        circ.rx(0.3, 1).rx(0.7, 1)
        # Z before measurement
        circ.z(0).measure([0], [0])

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=5)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag)
        # H·H → eliminated, RX(0.3)+RX(0.7) → RX(1.0), Z before M → eliminated
        # Should have: RX(1.0) + M = 2 instructions
        assert len(reconstructed) == 2


class TestPipelineCommutation:
    """Pipeline tests involving commutation-enabled cancellations."""

    def test_commute_enables_cancel_in_pipeline(self) -> None:
        """Commutation should enable further cancellations in the pipeline."""
        circ = QuantumCircuit()
        # H(0), RZ(0.3)(0), H(0): RZ commutes with nothing special,
        # but H·H can cancel and RZ commutes past
        circ.h(0).rz(0.3, 0).h(0)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=5)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag)
        # H·H cancel, leaving RZ; or all fused to U3
        assert len(reconstructed) == 1


class TestPipelineDAGReconstruction:
    """Full roundtrip verification."""

    def test_roundtrip_preserves_gate_order(self) -> None:
        """Reconstruction should preserve topological gate order."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).z(1).cx(0, 1).h(0)

        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)

        assert len(reconstructed) == 5
        for i in range(5):
            assert reconstructed[i].gate.name == circ[i].gate.name

    def test_optimized_roundtrip(self) -> None:
        """Optimized circuit should reconstruct validly."""
        circ = QuantumCircuit()
        # H·H cancel
        circ.h(0).h(0)
        # RX merge
        circ.rx(1.0, 1).rx(2.0, 1)
        # CX
        circ.cx(0, 1)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=5)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag)
        # Should be a valid circuit
        assert reconstructed.num_qubits == 2
        assert len(reconstructed) >= 1  # at minimum CX

    def test_pipeline_with_parameterized(self) -> None:
        """Pipeline should preserve symbolic parameters."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).h(1).cx(0, 1).z(0)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        opt_dag = pm.run(dag, max_iterations=3)

        reconstructed = DAGToCircuitCompiler.compile(opt_dag)
        assert theta in reconstructed.parameters


class TestPipelineStatistics:
    """Statistics and summary tests."""

    def test_pipeline_produces_stats(self) -> None:
        """Pipeline should produce non-empty statistics."""
        circ = QuantumCircuit()
        circ.h(0).h(0).rx(0.3, 1).rx(0.7, 1)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        pm.run(dag, max_iterations=3)

        assert len(pm.stats_history) > 0
        assert pm.total_nodes_removed > 0
        assert pm.total_elapsed_ms > 0

    def test_pipeline_summary_readable(self) -> None:
        """Pipeline summary should be human-readable."""
        circ = QuantumCircuit()
        circ.h(0).h(0)

        dag = circ.to_dag()
        pm = _build_default_pipeline()
        pm.run(dag)

        summary = pm.summary()
        assert "==" in summary
        assert "PassManager" in summary
        assert "Total" in summary
