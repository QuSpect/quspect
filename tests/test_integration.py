"""End-to-end integration tests for the QuSpect workflow.

These tests verify the complete pipeline:
    Gate → Circuit → DAG Compilation
"""

from __future__ import annotations

import math

import pytest

from quspect import Parameter, QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.standard import (
    CX,
    H,
)


class TestEndToEndBellState:
    """Full Bell state workflow."""

    def test_bell_state_circuit(self) -> None:
        """Build and verify a Bell state circuit."""
        circ = QuantumCircuit(name="bell")
        circ.h(0)
        circ.cx(0, 1)
        circ.measure([0, 1], [0, 1])

        assert circ.num_qubits == 2
        assert circ.num_clbits == 2
        assert len(circ) == 3  # H, CX, MEASURE

        # Verify instruction types
        assert circ[0].is_gate
        assert circ[0].gate is H
        assert circ[1].is_gate
        assert circ[1].gate is CX
        assert circ[2].is_measure

    def test_bell_state_dag(self) -> None:
        """Compile Bell state to DAG and verify structure."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1)

        dag = circ.to_dag()
        assert dag.num_nodes() == 6  # 2 IN + 2 gates + 2 OUT
        assert dag.num_edges() == 5  # IN→H, H→CX(q0), IN→CX(q1), CX→OUT(q0), CX→OUT(q1)

        # Verify topological order
        topo = dag.topological_order()
        assert len(topo) == dag.num_nodes()

        # Verify layers
        layers = dag.layers()
        assert len(layers) >= 2  # at least input layer and gate layers


class TestEndToEndParameterizedCircuit:
    """Full parameterized circuit workflow."""

    def test_parameterized_workflow(self) -> None:
        """Build, bind, compile a parameterized circuit."""
        theta = Parameter("θ")
        phi = Parameter("φ")

        circ = QuantumCircuit(name="vqe")
        circ.rx(theta, 0)
        circ.ry(phi, 1)
        circ.cx(0, 1)
        circ.measure_all()

        # Before binding: check parameters
        assert circ.parameters == {theta, phi}
        assert circ.num_qubits == 2

        # Bind parameters
        bound = circ.bind_parameters({theta: 0.5, phi: 1.2})

        # After binding: no parameters left
        assert bound.parameters == set()

        # Should be able to compute matrix for the bound version
        # Verify via DAG compilation
        dag = bound.to_dag()
        assert dag.num_nodes() > 0

    def test_partial_binding_raises(self) -> None:
        """Test that incomplete binding raises ValueError."""
        theta = Parameter("θ")
        phi = Parameter("φ")

        circ = QuantumCircuit()
        circ.rx(theta, 0).ry(phi, 1)

        with pytest.raises(ValueError, match="not bound"):
            circ.bind_parameters({theta: 0.5})  # phi not bound


class TestEndToEndCircuitComposition:
    """Full circuit composition workflow."""

    def test_compose_and_dag(self) -> None:
        """Compose circuits and compile to DAG."""
        # Build a Bell pair sub-circuit
        sub = QuantumCircuit(name="bell_pair")
        sub.h(0).cx(0, 1)

        # Compose twice into a 4-qubit circuit
        main = QuantumCircuit(name="main")
        main = main.compose(sub, qubits=[0, 1])
        main = main.compose(sub, qubits=[2, 3])
        main.measure_all()

        assert main.num_qubits == 4
        assert len(main) == 5  # 2×2 gates + measure

        # Compile to DAG
        dag = main.to_dag()
        assert dag.num_nodes() > 0


class TestEndToEndSerialization:
    """Full serialization round-trip."""

    def test_json_roundtrip_bell(self) -> None:
        """Serialize and deserialize a Bell state circuit."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1).measure([0, 1], [0, 1])

        json_str = circ.to_json()
        restored = QuantumCircuit.from_json(json_str)

        assert restored.name == "bell"
        assert restored.num_qubits == 2
        assert restored.num_clbits == 2
        assert len(restored) == 3
        assert restored[0].gate is H
        assert restored[1].gate is CX

    def test_json_roundtrip_parameterized(self) -> None:
        """Serialize and deserialize a parameterized circuit."""
        theta = Parameter("θ")

        circ = QuantumCircuit(name="param")
        circ.rx(theta, 0).h(1).cx(0, 1)

        json_str = circ.to_json()
        restored = QuantumCircuit.from_json(json_str)

        assert restored.name == "param"
        assert restored.num_qubits == 2
        assert len(restored) == 3
        assert theta in restored.parameters

    def test_json_roundtrip_with_measurement(self) -> None:
        """Serialize and deserialize a circuit with measurements and barriers."""
        circ = QuantumCircuit(name="full")
        circ.h(0).x(1).barrier(0, 1).measure([0], [0])

        json_str = circ.to_json()
        restored = QuantumCircuit.from_json(json_str)

        assert len(restored) == 4
        assert restored[2].is_barrier
        assert restored[3].is_measure


class TestEndToEndDAGFeatures:
    """DAG topology and analysis end-to-end tests."""

    def test_dag_topological_order(self) -> None:
        """Verify topological ordering respects dependencies."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).z(0)

        dag = circ.to_dag()
        topo = dag.topological_order()

        # Every edge source must appear before its target in topo order
        pos = {node_id: i for i, node_id in enumerate(topo)}
        for edge in dag.edges:
            assert pos[edge.source_id] < pos[edge.target_id], (
                f"Edge {edge.source_id} → {edge.target_id} violates topological order"
            )

    def test_dag_qubit_dependencies(self) -> None:
        """Verify qubit dependency structure in DAG."""
        circ = QuantumCircuit()
        circ.h(0).x(1).cx(0, 1).z(0).y(1)

        dag = circ.to_dag()

        # Qubit 0 path: IN → H → CX → Z → OUT
        # Qubit 1 path: IN → X → CX → Y → OUT
        # CX is a 2-qubit gate — creates a synchronization point

        # Both qubit paths should exist
        assert dag.num_nodes() == 9  # 2 IN + 5 ops + 2 OUT


class TestEndToEndStandardGates:
    """End-to-end tests with all standard gates."""

    def test_all_13_standard_gates(self) -> None:
        """Build a circuit with all 13 standard gates."""
        circ = QuantumCircuit(name="all_gates")
        circ.id(0)
        circ.h(1)
        circ.x(2)
        circ.y(3)
        circ.z(4)
        circ.s(5)
        circ.sdg(6)
        circ.t(7)
        circ.tdg(8)
        circ.cx(0, 1)
        circ.cz(2, 3)
        circ.swap(4, 5)
        circ.ccx(0, 1, 2)

        assert circ.num_qubits == 9
        assert len(circ) == 13

        # Compile to DAG
        dag = circ.to_dag()
        assert dag.num_nodes() > 0

    def test_all_parameterized_gates(self) -> None:
        """Build a circuit with parameterized gates."""
        theta = Parameter("θ")

        circ = QuantumCircuit(name="all_param")
        circ.rx(0.5, 0)
        circ.ry(theta, 1)
        circ.rz(0.3, 2)
        circ.p(math.pi / 2, 3)
        circ.u3(0.5, 0.3, 0.1, 0)
        circ.crx(0.5, 0, 1)
        circ.cry(0.8, 2, 3)
        circ.crz(0.4, 4, 5)

        assert circ.num_qubits == 6
        assert theta in circ.parameters

        dag = circ.to_dag()
        assert dag.num_nodes() > 0


class TestEndToEndBarrier:
    """End-to-end barrier tests."""

    def test_barrier_specific_qubits(self) -> None:
        """Test barrier on specific qubits in full workflow."""
        circ = QuantumCircuit()
        circ.h(0).x(1).barrier(0, 1).z(0).y(1)

        dag = circ.to_dag()
        assert dag.num_nodes() == 9  # 2 IN + 5 ops + 2 OUT (barrier is an op)

    def test_barrier_all_qubits(self) -> None:
        """Test barrier on all active qubits."""
        circ = QuantumCircuit()
        circ.h(0).x(1).z(3).barrier()

        assert circ[-1].is_barrier
        assert circ[-1].qubits == (0, 1, 3)

        dag = circ.to_dag()
        assert dag.num_nodes() > 0


class TestEndToEndDynamicQubitCount:
    """End-to-end dynamic qubit count inference."""

    def test_gaps_in_qubit_indices(self) -> None:
        """Test that gaps in qubit indices work correctly."""
        circ = QuantumCircuit()
        circ.h(0)
        circ.x(5)  # gap: qubits 1-4 unused
        circ.cx(0, 5)

        assert circ.num_qubits == 6  # max index 5 + 1
        assert circ.active_qubits == {0, 5}

        dag = circ.to_dag()
        assert dag.num_qubits == 6

    def test_measure_all_dynamic(self) -> None:
        """Test measure_all with dynamic qubit count."""
        circ = QuantumCircuit()
        circ.h(0).x(3).z(7)
        circ.measure_all()

        assert circ[-1].is_measure
        assert circ[-1].qubits == (0, 3, 7)
        assert circ[-1].clbits == (0, 3, 7)


class TestEndToEndMatrixAccess:
    """End-to-end matrix access via gate_matrix."""

    def test_standard_gate_matrix_access(self) -> None:
        """Access matrix of standard gates via utility."""
        import numpy as np

        m = gate_matrix(H)
        assert m.shape == (2, 2)
        # H is unitary
        np.testing.assert_array_almost_equal(
            m @ m.conj().T, np.eye(2, dtype=complex), decimal=10
        )

    def test_parameterized_gate_matrix_access(self) -> None:
        """Access matrix of parameterized gates via utility."""
        import numpy as np

        from quspect.gates.parameterized import RX

        g = RX.with_params(0.5)
        m = gate_matrix(g)
        assert m.shape == (2, 2)
        # RX is unitary
        np.testing.assert_array_almost_equal(
            m @ m.conj().T, np.eye(2, dtype=complex), decimal=10
        )

    def test_symbolic_gate_matrix_raises(self) -> None:
        """Accessing matrix with symbolic params raises ValueError."""
        theta = Parameter("θ")
        from quspect.gates.gate import Gate

        g = Gate(name="rx", num_qubits=1, params=(theta,))
        with pytest.raises(ValueError, match="symbolic"):
            gate_matrix(g)


class TestEndToEndCopy:
    """End-to-end copy workflow."""

    def test_copy_independence(self) -> None:
        """Test that copy creates an independent circuit."""
        circ = QuantumCircuit(name="original")
        circ.h(0).cx(0, 1)

        copy = circ.copy(name="copy")
        assert copy.name == "copy"
        assert len(copy) == 2

        # Mutate original
        circ.z(2)

        # Copy should be unaffected
        assert len(copy) == 2
        assert copy.num_qubits == 2
        assert circ.num_qubits == 3
