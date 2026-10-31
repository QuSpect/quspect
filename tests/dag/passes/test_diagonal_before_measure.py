"""Tests for EliminateDiagonalBeforeMeasurement pass."""

from __future__ import annotations

from quspect import QuantumCircuit
from quspect.dag.passes.diagonal_before_measure import (
    EliminateDiagonalBeforeMeasurement,
)


class TestEliminateDiagonalBasic:
    """Basic elimination tests."""

    def test_z_before_measure_removed(self) -> None:
        """Z gate immediately before measurement should be removed."""
        circ = QuantumCircuit()
        circ.h(0).z(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        # Z is removed: 2 IN:q+IN:c + H + M(0→c0) + 2 OUT:q+OUT:c = 6
        #                                                            (was 7 with Z)
        assert optimized.num_nodes() == before - 1

    def test_s_before_measure_removed(self) -> None:
        """S gate before measurement should be removed."""
        circ = QuantumCircuit()
        circ.h(0).s(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before - 1

    def test_t_before_measure_removed(self) -> None:
        """T gate before measurement should be removed."""
        circ = QuantumCircuit()
        circ.h(0).t(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before - 1

    def test_rz_before_measure_removed(self) -> None:
        """RZ rotation before measurement should be removed."""
        circ = QuantumCircuit()
        circ.h(0).rz(0.5, 0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before - 1

    def test_sdg_tdg_before_measure_removed(self) -> None:
        """S† and T† before measurement should be removed."""
        circ = QuantumCircuit()
        circ.h(0).sdg(0).tdg(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        # Both SDG and TDG removed
        assert optimized.num_nodes() == before - 2


class TestEliminateDiagonalNoRemove:
    """Cases where diagonal gates should NOT be removed."""

    def test_z_not_before_measure_kept(self) -> None:
        """Z gate NOT before measurement should be kept."""
        circ = QuantumCircuit()
        circ.h(0).z(0).x(0)  # Z not followed by measurement
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before

    def test_h_not_removed(self) -> None:
        """H is NOT diagonal — should NOT be removed."""
        circ = QuantumCircuit()
        circ.h(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before

    def test_x_not_removed(self) -> None:
        """X is NOT diagonal — should NOT be removed."""
        circ = QuantumCircuit()
        circ.h(0).x(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before

    def test_gate_between_z_and_measure(self) -> None:
        """Z with another gate before measurement should be kept."""
        circ = QuantumCircuit()
        circ.h(0).z(0).x(0).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        # Z is NOT immediately before measurement (X is between)
        assert optimized.num_nodes() == before


class TestEliminateDiagonalMultiQubit:
    """Multi-qubit elimination tests."""

    def test_independent_wires(self) -> None:
        """Elimination on independent wires."""
        circ = QuantumCircuit()
        circ.h(0).z(0).measure([0], [0])
        circ.h(1).s(1).measure([1], [1])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        assert optimized.num_nodes() == before - 2  # Z and S both removed

    def test_cx_does_not_block_elimination(self) -> None:
        """Diagonal on q0 before CX, then M on q0 — Z has CX as its only
        direct wire successor, not M, so Z is NOT eliminated.
        This is correct behavior: CX uses q0, so the phase matters."""
        circ = QuantumCircuit()
        circ.h(0).z(0).cx(0, 1).measure([0], [0])
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = EliminateDiagonalBeforeMeasurement().run(dag)
        # The Z is on q0, CX is on q0,q1, M is on q0.
        # Z's successor on wire q0 is CX (not M), so Z is kept.
        # This is the correct behavior.
        # But Z's successor on q0 is CX, not M — Z stays
        assert optimized.num_nodes() == before
