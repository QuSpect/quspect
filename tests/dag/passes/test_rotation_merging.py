"""Tests for RotationMerging pass."""

from __future__ import annotations

import math

from quspect import QuantumCircuit
from quspect.dag.passes.rotation_merging import RotationMerging


class TestRotationMergingBasic:
    """Test basic rotation merging."""

    def test_rx_rx_merges(self) -> None:
        """Test RX(a) + RX(b) → RX(a+b)."""
        circ = QuantumCircuit()
        circ.rx(0.3, 0).rx(0.5, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = RotationMerging().run(dag)
        # One rotation removed
        assert optimized.num_nodes() == before - 1  # -1 op node

    def test_ry_ry_merges(self) -> None:
        """Test RY merging."""
        circ = QuantumCircuit()
        circ.ry(1.0, 0).ry(2.0, 0)
        dag = circ.to_dag()
        optimized = RotationMerging().run(dag)
        assert optimized.num_nodes() == 3  # IN + merged RY + OUT

    def test_rz_rz_merges(self) -> None:
        """Test RZ merging."""
        circ = QuantumCircuit()
        circ.rz(0.5, 0).rz(1.5, 0)
        dag = circ.to_dag()
        optimized = RotationMerging().run(dag)
        assert optimized.num_nodes() == 3

    def test_p_p_merges(self) -> None:
        """Test Phase gate merging."""
        circ = QuantumCircuit()
        circ.p(math.pi / 2, 0).p(math.pi / 4, 0)
        dag = circ.to_dag()
        optimized = RotationMerging().run(dag)
        assert optimized.num_nodes() == 3


class TestRotationMergingNoMerge:
    """Test cases where merging should NOT happen."""

    def test_different_axes_no_merge(self) -> None:
        """Test RX + RY doesn't merge."""
        circ = QuantumCircuit()
        circ.rx(0.5, 0).ry(0.5, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = RotationMerging().run(dag)
        assert optimized.num_nodes() == before

    def test_different_qubits_no_merge(self) -> None:
        """Test rotations on different qubits don't merge."""
        circ = QuantumCircuit()
        circ.rx(0.5, 0).rx(0.5, 1)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = RotationMerging().run(dag)
        assert optimized.num_nodes() == before

    def test_gate_between_no_merge(self) -> None:
        """Test gate between rotations prevents merging."""
        circ = QuantumCircuit()
        circ.rx(0.5, 0).x(0).rx(0.3, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = RotationMerging().run(dag)
        assert optimized.num_nodes() == before


class TestRotationMergingMultiQubit:
    """Test multi-qubit rotation merging."""

    def test_parallel_merges(self) -> None:
        """Test merges on different qubits independently."""
        circ = QuantumCircuit()
        circ.rx(0.3, 0).rx(0.5, 0).rz(0.2, 1).rz(0.8, 1)
        dag = circ.to_dag()
        optimized = RotationMerging().run(dag)
        # Two merges: 2 nodes removed
        assert optimized.num_nodes() == 6  # 2 IN + 2 merged + 2 OUT

    def test_merge_after_cx(self) -> None:
        """Test rotation merging works after a CNOT."""
        circ = QuantumCircuit()
        circ.cx(0, 1).rz(0.3, 0).rz(0.5, 0)
        dag = circ.to_dag()
        optimized = RotationMerging().run(dag)
        # CNOT stays, two RZ merge into one
        # 2 IN + CX + merged RZ + 2 OUT = 6
        assert optimized.num_nodes() == 6
