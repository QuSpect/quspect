"""Tests for CircuitCompiler (QuantumCircuit → DAGCircuit)."""

from __future__ import annotations

from quspect import QuantumCircuit
from quspect.circuit.parameter import Parameter
from quspect.dag.compiler import CircuitCompiler


class TestCompilerEmptyCircuit:
    """Test compilation of empty circuits."""

    def test_empty_circuit(self) -> None:
        """Test compiling a completely empty circuit."""
        circ = QuantumCircuit()
        dag = CircuitCompiler.compile(circ)
        assert dag.num_qubits == 0
        assert dag.num_clbits == 0
        assert dag.num_nodes() == 0

    def test_empty_circuit_with_name(self) -> None:
        """Test compiling a named empty circuit."""
        circ = QuantumCircuit(name="empty")
        dag = CircuitCompiler.compile(circ)
        assert dag.num_nodes() == 0


class TestCompilerSingleQubit:
    """Test compilation of single-qubit circuits."""

    def test_single_h_gate(self) -> None:
        """Test compiling a circuit with one Hadamard gate."""
        circ = QuantumCircuit()
        circ.h(0)
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 1
        # Nodes: IN:q0, H, OUT:q0 = 3
        assert dag.num_nodes() == 3
        assert len(dag.input_nodes()) == 1
        assert len(dag.output_nodes()) == 1
        assert len(dag.operation_nodes()) == 1

        # Check topology
        assert dag.topological_order() == [0, 1, 2]

    def test_single_x_gate(self) -> None:
        """Test compiling a circuit with one X gate."""
        circ = QuantumCircuit()
        circ.x(0)
        dag = CircuitCompiler.compile(circ)
        assert len(dag.operation_nodes()) == 1

    def test_chain_of_gates(self) -> None:
        """Test compiling a chain of single-qubit gates."""
        circ = QuantumCircuit()
        circ.h(0).x(0).z(0)
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 1
        assert dag.num_nodes() == 5  # IN + 3 gates + OUT
        assert dag.num_edges() == 4  # IN→H, H→X, X→Z, Z→OUT
        assert dag.critical_path_length() == 3


class TestCompilerMultiQubit:
    """Test compilation of multi-qubit circuits."""

    def test_bell_state(self) -> None:
        """Test compiling a Bell state circuit."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 2
        # Nodes: IN:q0, IN:q1, H, CX, OUT:q0, OUT:q1 = 6
        assert dag.num_nodes() == 6
        assert dag.num_edges() == 5  # IN→H, H→CX(q0), IN→CX(q1), CX→OUT:q0, CX→OUT:q1

        # CX should have predecessors from both qubit lines
        layer = dag.topological_order()
        # Order must respect: IN→H→CX, IN→CX
        assert layer[0:2] == [0, 1]  # Both INPUT nodes first
        assert layer[2] == 2  # H gate

    def test_parallel_operations(self) -> None:
        """Test that independent operations on different qubits are parallel."""
        circ = QuantumCircuit()
        circ.h(0).x(1)
        dag = CircuitCompiler.compile(circ)

        layers = dag.layers()
        # IN:q0 and IN:q1 in layer 0
        # H and X should be in the same layer (parallel)
        assert len(layers[1]) == 2  # H and X in parallel

    def test_ghz_state(self) -> None:
        """Test compiling a 3-qubit GHZ state circuit."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).cx(0, 2)
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 3
        # IN×3 + H + CX×2 + OUT×3 = 9
        assert dag.num_nodes() == 9

        # Critical path: IN→H→CX(0,1)→CX(0,2)→OUT = 4 ops deep
        # But CXs use qubit 0, so they're sequential
        layers = dag.layers()
        op_depths = [
            layer
            for layer in layers
            if any(dag.get_node(nid).is_operation for nid in layer)
        ]
        assert len(op_depths) == 3  # H, first CX, second CX

    def test_gapped_qubits(self) -> None:
        """Test compiling circuit with gaps in qubit indices."""
        circ = QuantumCircuit()
        circ.h(0).x(5)
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 6  # qubits 0...5
        # 6 qubit inputs + 2 ops + 6 qubit outputs = 14
        assert dag.num_nodes() == 14


class TestCompilerMeasurement:
    """Test compilation with measurements."""

    def test_measurement(self) -> None:
        """Test compiling a circuit with measurement."""
        circ = QuantumCircuit()
        circ.h(0).measure([0], [0])
        dag = CircuitCompiler.compile(circ)

        assert dag.num_clbits == 1
        # Nodes: IN:q0, IN:c0, H, M, OUT:q0, OUT:c0
        assert dag.num_nodes() == 6

    def test_measure_all(self) -> None:
        """Test compiling measure_all."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).measure_all()
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 2
        assert dag.num_clbits == 2
        # IN:q0,IN:q1,IN:c0,IN:c1 + 3 ops + 4 OUTs = 11
        assert dag.num_nodes() == 11

    def test_measure_then_gate(self) -> None:
        """Test compiling measurement followed by gate on same qubit."""
        circ = QuantumCircuit()
        circ.h(0).measure([0], [0]).x(0)
        dag = CircuitCompiler.compile(circ)

        # After measurement, qubit 0 can still have a gate (no restriction in Phase 1)
        assert len(dag.operation_nodes()) == 3  # H, M, X


class TestCompilerBarrier:
    """Test compilation with barriers."""

    def test_barrier(self) -> None:
        """Test compiling a circuit with barrier."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).barrier(0, 1).h(0).h(1)
        dag = CircuitCompiler.compile(circ)

        # Barrier creates an OPERATION node
        barrier_nodes = [n for n in dag.nodes if n.is_barrier]
        assert len(barrier_nodes) == 1

        # Barrier node should involve both qubits
        barrier = barrier_nodes[0]
        assert barrier.qubits == {0, 1}

    def test_barrier_all(self) -> None:
        """Test barrier() with no args (applies to all active qubits)."""
        circ = QuantumCircuit()
        circ.h(0).x(1).barrier()
        dag = CircuitCompiler.compile(circ)

        barrier = [n for n in dag.nodes if n.is_barrier]
        assert len(barrier) == 1
        assert barrier[0].qubits == {0, 1}


class TestCompilerParameterized:
    """Test compilation of parameterized circuits."""

    def test_parameterized_gate(self) -> None:
        """Test compiling circuit with parameterized gate."""
        circ = QuantumCircuit()
        circ.rx(0.5, 0)
        dag = CircuitCompiler.compile(circ)
        assert len(dag.operation_nodes()) == 1

    def test_symbolic_parameter(self) -> None:
        """Test compiling circuit with symbolic parameters."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0)
        dag = CircuitCompiler.compile(circ)
        assert len(dag.operation_nodes()) == 1


class TestCompilerComplex:
    """Test compilation of complex circuit patterns."""

    def test_bell_with_measure(self) -> None:
        """End-to-end: Bell state with measurements."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1).measure([0, 1], [0, 1])
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 2
        assert dag.num_clbits == 2
        # 2 qubit IN + 2 clbit IN + 2 gates + 1 measure + 2 qubit OUT + 2 clbit OUT = 11
        assert dag.num_nodes() == 11

    def test_two_independent_blocks(self) -> None:
        """Test two completely independent blocks on different qubits."""
        circ = QuantumCircuit()
        circ.h(0).x(0).h(1).x(1)
        dag = CircuitCompiler.compile(circ)

        layers = dag.layers()
        # Both H gates should be in same layer, both X gates in same layer
        op_layer_0 = layers[1]
        op_layer_1 = layers[2]
        assert len(op_layer_0) == 2  # Two H gates
        assert len(op_layer_1) == 2  # Two X gates


class TestCompilerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_only_cx(self) -> None:
        """Test compiling circuit with only a CX gate."""
        circ = QuantumCircuit()
        circ.cx(0, 1)
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 2
        assert len(dag.operation_nodes()) == 1

    def test_only_measure(self) -> None:
        """Test compiling circuit with only a measurement."""
        circ = QuantumCircuit()
        circ.measure([0], [0])
        dag = CircuitCompiler.compile(circ)

        assert dag.num_qubits == 1
        assert dag.num_clbits == 1
        assert len(dag.operation_nodes()) == 1

    def test_large_qubit_index(self) -> None:
        """Test compilation with large qubit index."""
        circ = QuantumCircuit()
        circ.h(100)
        dag = CircuitCompiler.compile(circ)
        assert dag.num_qubits == 101

    def test_topological_order_valid(self) -> None:
        """Test that topological order respects all edges."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).cz(0, 1).h(0)
        dag = CircuitCompiler.compile(circ)

        order = dag.topological_order()
        # For every edge, source must appear before target
        for edge in dag.edges:
            assert order.index(edge.source_id) < order.index(edge.target_id), (
                f"Edge {edge.source_id}→{edge.target_id} violated in {order}"
            )

    def test_layers_no_duplicate_nodes(self) -> None:
        """Test that layers partition contains all nodes exactly once."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).x(2).measure_all()
        dag = CircuitCompiler.compile(circ)

        layers = dag.layers()
        all_node_ids = []
        for layer in layers:
            all_node_ids.extend(layer)

        assert sorted(all_node_ids) == sorted(range(dag.num_nodes()))
        assert len(all_node_ids) == dag.num_nodes()


class TestCompilerCircuitToDAG:
    """Test QuantumCircuit.to_dag() integration."""

    def test_to_dag_integration(self) -> None:
        """Test that QuantumCircuit.to_dag() uses the compiler."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = circ.to_dag()
        assert dag.num_qubits == 2
        assert dag.num_nodes() == 6  # 2 inputs + 2 ops + 2 outputs

    def test_to_dag_empty(self) -> None:
        """Test to_dag on empty circuit."""
        circ = QuantumCircuit()
        dag = circ.to_dag()
        assert dag.num_nodes() == 0

    def test_to_dag_summary(self) -> None:
        """Test that DAG summary works after compilation."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = circ.to_dag()
        s = dag.summary()
        assert "DAGCircuit" in s
        assert "qubits" in s.lower() or "Qubits" in s
