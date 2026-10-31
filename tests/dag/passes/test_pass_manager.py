"""Tests for PassManager with statistics tracking and convergence."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect.dag.passes.base import PassManager, PassStats


class TestPassStats:
    """Tests for PassStats dataclass."""

    def test_changed_detection(self) -> None:
        """Test that changed flag is computed correctly."""
        stats = PassStats(
            pass_name="Test", nodes_before=10, nodes_after=8, elapsed_ms=1.5
        )
        assert stats.changed is True
        assert stats.nodes_removed == 2
        assert stats.nodes_added == 0

    def test_no_change(self) -> None:
        """Test unchanged stats."""
        stats = PassStats(
            pass_name="Test", nodes_before=5, nodes_after=5, elapsed_ms=0.5
        )
        assert stats.changed is False
        assert stats.nodes_removed == 0

    def test_nodes_added(self) -> None:
        """Test nodes_added when count increases."""
        stats = PassStats(
            pass_name="Test", nodes_before=5, nodes_after=7, elapsed_ms=2.0
        )
        assert stats.changed is True
        assert stats.nodes_removed == 0
        assert stats.nodes_added == 2

    def test_reduction_pct(self) -> None:
        """Test reduction percentage calculation."""
        stats = PassStats(
            pass_name="Test", nodes_before=10, nodes_after=7, elapsed_ms=1.0
        )
        assert abs(stats.reduction_pct - 30.0) < 0.01

    def test_reduction_pct_zero(self) -> None:
        """Test reduction percentage when nodes_before is 0."""
        stats = PassStats(
            pass_name="Test", nodes_before=0, nodes_after=0, elapsed_ms=0.0
        )
        assert stats.reduction_pct == 0.0


class TestPassManagerBasic:
    """Basic PassManager functionality tests."""

    def test_empty_pass_list(self) -> None:
        """PassManager with no passes should return DAG unchanged."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = circ.to_dag()
        pm = PassManager([])
        result = pm.run(dag)
        assert result.num_nodes() == dag.num_nodes()
        assert len(pm.stats_history) == 0

    def test_single_pass(self) -> None:
        """Single pass should be tracked in stats."""
        from quspect.dag.passes import GateCancellation

        circ = QuantumCircuit()
        circ.h(0).h(0)  # Two H cancel
        dag = circ.to_dag()
        before = dag.num_nodes()  # Read BEFORE pass modifies the DAG
        pm = PassManager([GateCancellation()])
        result = pm.run(dag, max_iterations=2)

        assert result.num_nodes() < before
        assert len(pm.stats_history) >= 1
        assert pm.total_nodes_removed >= 2

    def test_convergence_detection(self) -> None:
        """PassManager should converge when no changes."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = circ.to_dag()
        pm = PassManager([])
        result = pm.run(dag, max_iterations=10)
        # Should converge on first iteration (no passes to run → no changes)
        assert result is not None

    def test_disabled_pass(self) -> None:
        """Disabled pass should be skipped."""
        from quspect.dag.passes import GateCancellation

        circ = QuantumCircuit()
        circ.h(0).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()

        gc = GateCancellation()
        gc.enabled = False
        pm = PassManager([gc])
        result = pm.run(dag, max_iterations=2)

        # Should skip and keep all nodes
        assert result.num_nodes() == before

    def test_summary_output(self) -> None:
        """Test that summary produces output."""
        from quspect.dag.passes import GateCancellation

        circ = QuantumCircuit()
        circ.h(0).h(0)
        dag = circ.to_dag()
        pm = PassManager([GateCancellation()])
        pm.run(dag)

        summary = pm.summary()
        assert "PassManager Optimization Summary" in summary
        assert "GateCancellation" in summary
        assert "Total" in summary

    def test_stats_history_accumulation(self) -> None:
        """Stats should accumulate across iterations."""
        from quspect.dag.passes import GateCancellation, RotationMerging

        circ = QuantumCircuit()
        circ.h(0).h(0).rx(0.3, 1).rx(0.7, 1)
        dag = circ.to_dag()
        pm = PassManager([GateCancellation(), RotationMerging()])
        pm.run(dag, max_iterations=3)

        # Each pass records stats per iteration
        assert len(pm.stats_history) > 0
        # Should see at least one cancellation
        assert pm.total_nodes_removed >= 3

    def test_total_elapsed_ms(self) -> None:
        """Test total elapsed time tracking."""
        from quspect.dag.passes import GateCancellation

        circ = QuantumCircuit()
        circ.h(0)
        dag = circ.to_dag()
        pm = PassManager([GateCancellation()])
        pm.run(dag)

        assert pm.total_elapsed_ms > 0


class TestPassManagerStructuralConvergence:
    """Convergence is structural, not node-count based (B69/B70/B77)."""

    def test_dag_signature_detects_reordering(self) -> None:
        """Same gates in a different wire order give different signatures."""
        from quspect.dag.passes.base import dag_signature

        a = QuantumCircuit().cx(0, 1).rz(0.3, 0).to_dag()
        b = QuantumCircuit().rz(0.3, 0).cx(0, 1).to_dag()
        c = QuantumCircuit().cx(0, 1).rz(0.3, 0).to_dag()
        assert dag_signature(a) != dag_signature(b)
        assert dag_signature(a) == dag_signature(c)

    def test_structure_changed_flag(self) -> None:
        """A reorder with equal node count counts as a change."""
        stats = PassStats(
            pass_name="T",
            nodes_before=5,
            nodes_after=5,
            elapsed_ms=0.0,
            structure_changed=True,
        )
        assert stats.changed is True

    def test_cancel_commute_recancel(self) -> None:
        """[cancel, commute] iterates until cx rz(control) cx -> rz."""
        from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
        from quspect.dag.passes import CommutationAnalysis, GateCancellation

        circ = QuantumCircuit().cx(0, 1).rz(0.3, 0).cx(0, 1)
        pm = PassManager([GateCancellation(), CommutationAnalysis()])
        out = DAGToCircuitCompiler.compile(pm.run(circ.to_dag()))
        assert [i.gate.name for i in out.instructions if i.gate] == ["rz"]

    def test_nonconvergence_raises(self) -> None:
        """A pass that always changes the DAG raises RuntimeError."""
        import pytest

        from quspect.dag.dag_circuit import DAGCircuit
        from quspect.dag.passes import BasePass

        class _Flip(BasePass):
            """Alternate between two equal-size circuits forever."""

            def __init__(self) -> None:
                super().__init__()
                self.toggle = False

            def run(self, dag: DAGCircuit) -> DAGCircuit:
                self.toggle = not self.toggle
                circ = QuantumCircuit()
                if self.toggle:
                    circ.x(0)
                else:
                    circ.h(0)
                return circ.to_dag()

        pm = PassManager([_Flip()])
        with pytest.raises(RuntimeError, match="did not converge"):
            pm.run(QuantumCircuit().h(0).to_dag(), max_iterations=3)
        assert len(pm.stats_history) == 3


class TestPassManagerSignatureSkip:
    """The structural signature is skipped only for passes that report edits."""

    def test_unreported_direct_mutation_is_detected(self) -> None:
        """A user pass that assigns ``node.instruction`` still counts as a change."""
        from quspect.circuit.instruction import CircuitInstruction, InstructionType
        from quspect.dag.passes.base import BasePass
        from quspect.gates.standard import X

        class FlipFirstH(BasePass):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def run(self, dag):  # type: ignore[no-untyped-def]
                self.calls += 1
                for node in dag.operation_nodes():
                    if node.instruction.gate.name == "h":
                        # Bypasses DAGCircuit methods: no modification_count bump.
                        node.instruction = CircuitInstruction(
                            InstructionType.GATE, gate=X, qubits=node.instruction.qubits
                        )
                        break
                return dag

        dag = QuantumCircuit().h(0).h(1).to_dag()
        flip = FlipFirstH()
        PassManager([flip]).run(dag, max_iterations=5)
        # Two changing iterations (one per h) and a final unchanged one.
        assert flip.calls == 3
        assert {n.instruction.gate.name for n in dag.operation_nodes()} == {"x"}

    def test_builtin_passes_report_modifications(self) -> None:
        from quspect.dag.passes import (
            CommutationAnalysis,
            EliminateDiagonalBeforeMeasurement,
            GateCancellation,
            RemoveIdentityGates,
            RotationMerging,
            SingleQubitGateFusion,
        )
        from quspect.dag.passes.base import BasePass

        assert BasePass.reports_modifications is False
        for cls in (
            CommutationAnalysis,
            EliminateDiagonalBeforeMeasurement,
            GateCancellation,
            RemoveIdentityGates,
            RotationMerging,
            SingleQubitGateFusion,
        ):
            assert cls.reports_modifications is True

    def test_rotation_merge_bumps_modification_count(self) -> None:
        from quspect.dag.passes import RotationMerging

        dag = QuantumCircuit().rz(0.1, 0).rz(0.2, 0).to_dag()
        before = dag.modification_count
        RotationMerging().run(dag)
        assert dag.modification_count > before
        (node,) = dag.operation_nodes()
        assert node.instruction.gate.params[0] == pytest.approx(0.3)


class TestNodesOnWireCache:
    def test_cached_result_is_a_copy_and_tracks_edits(self) -> None:
        dag = QuantumCircuit().h(0).x(0).to_dag()
        first = dag.nodes_on_wire("q0")
        first.append(-1)  # mutating the result must not leak into the cache
        assert dag.nodes_on_wire("q0") == first[:-1]
        nid = dag.operation_nodes()[0].node_id
        dag.remove_node_and_reconnect(nid)
        assert nid not in dag.nodes_on_wire("q0")
        assert len(dag.nodes_on_wire("q0")) == len(first) - 2
