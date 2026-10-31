"""Tests for DAGCircuit data model."""

from __future__ import annotations

import pytest

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode, DAGNodeType


class TestDAGNode:
    """Test DAGNode creation and properties."""

    def test_input_node(self) -> None:
        """Test creating an INPUT node."""
        node = DAGNode(
            node_id=0, node_type=DAGNodeType.INPUT, qubits={0}, label="IN:q0"
        )
        assert node.node_id == 0
        assert node.node_type == DAGNodeType.INPUT
        assert node.is_input is True
        assert node.is_output is False
        assert node.is_operation is False
        assert node.qubits == {0}

    def test_output_node(self) -> None:
        """Test creating an OUTPUT node."""
        node = DAGNode(
            node_id=1, node_type=DAGNodeType.OUTPUT, qubits={0}, label="OUT:q0"
        )
        assert node.node_type == DAGNodeType.OUTPUT
        assert node.is_output is True

    def test_operation_node(self) -> None:
        """Test creating an OPERATION node."""
        from quspect.circuit.instruction import CircuitInstruction, InstructionType
        from quspect.gates.standard import H

        instr = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
        node = DAGNode(
            node_id=2,
            node_type=DAGNodeType.OPERATION,
            instruction=instr,
            qubits={0},
        )
        assert node.node_type == DAGNodeType.OPERATION
        assert node.is_operation is True
        assert node.is_gate is True

    def test_node_repr(self) -> None:
        """Test string representation."""
        node = DAGNode(
            node_id=0, node_type=DAGNodeType.INPUT, qubits={0}, label="IN:q0"
        )
        r = repr(node)
        assert "DAGNode" in r
        assert "INPUT" in r

    def test_defaults(self) -> None:
        """Test default values."""
        node = DAGNode(node_id=0, node_type=DAGNodeType.INPUT)
        assert node.instruction is None
        assert node.qubits == set()
        assert node.clbits == set()
        assert node.label == ""
        assert node.metadata == {}


class TestDAGEdge:
    """Test DAGEdge creation."""

    def test_create_edge(self) -> None:
        """Test creating a DAG edge."""
        edge = DAGEdge(source_id=0, target_id=1, wire="q0")
        assert edge.source_id == 0
        assert edge.target_id == 1
        assert edge.wire == "q0"

    def test_edge_default_wire(self) -> None:
        """Test edge with default wire."""
        edge = DAGEdge(source_id=0, target_id=1)
        assert edge.wire == ""


class TestDAGCircuitConstruction:
    """Test DAGCircuit node/edge management."""

    def test_empty_dag(self) -> None:
        """Test creating an empty DAG."""
        dag = DAGCircuit()
        assert dag.num_nodes() == 0
        assert dag.num_edges() == 0

    def test_add_node(self) -> None:
        """Test adding a node."""
        dag = DAGCircuit()
        node = DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0})
        dag.add_node(node)
        assert dag.num_nodes() == 1
        assert dag.get_node(0) is node

    def test_add_duplicate_node_raises(self) -> None:
        """Test adding duplicate node_id raises."""
        dag = DAGCircuit()
        n1 = DAGNode(node_id=0, node_type=DAGNodeType.INPUT)
        n2 = DAGNode(node_id=0, node_type=DAGNodeType.INPUT)
        dag.add_node(n1)
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node(n2)

    def test_add_edge(self) -> None:
        """Test adding an edge."""
        dag = DAGCircuit()
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_edge(DAGEdge(source_id=0, target_id=1, wire="q0"))
        assert dag.num_edges() == 1

    def test_set_qubit_count(self) -> None:
        """Test setting qubit count."""
        dag = DAGCircuit()
        dag.set_qubit_count(3)
        assert dag.num_qubits == 3

    def test_set_clbit_count(self) -> None:
        """Test setting clbit count."""
        dag = DAGCircuit()
        dag.set_clbit_count(4)
        assert dag.num_clbits == 4


class TestDAGCircuitTraversal:
    """Test DAG traversal operations."""

    def _build_linear_dag(self) -> DAGCircuit:
        """Build a simple linear 3-node DAG: IN:q0 → OP → OUT:q0."""
        dag = DAGCircuit()
        dag.set_qubit_count(1)

        n_in = DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0})
        n_op = DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0})
        n_out = DAGNode(node_id=2, node_type=DAGNodeType.OUTPUT, qubits={0})

        dag.add_node(n_in)
        dag.add_node(n_op)
        dag.add_node(n_out)

        dag.add_edge(DAGEdge(source_id=0, target_id=1, wire="q0"))
        dag.add_edge(DAGEdge(source_id=1, target_id=2, wire="q0"))

        return dag

    def test_predecessors(self) -> None:
        """Test predecessor query."""
        dag = self._build_linear_dag()
        assert dag.predecessors(1) == [0]
        assert dag.predecessors(2) == [1]
        assert dag.predecessors(0) == []

    def test_successors(self) -> None:
        """Test successor query."""
        dag = self._build_linear_dag()
        assert dag.successors(0) == [1]
        assert dag.successors(1) == [2]
        assert dag.successors(2) == []

    def test_input_nodes(self) -> None:
        """Test input nodes query."""
        dag = self._build_linear_dag()
        inputs = dag.input_nodes()
        assert len(inputs) == 1
        assert inputs[0].is_input

    def test_output_nodes(self) -> None:
        """Test output nodes query."""
        dag = self._build_linear_dag()
        outputs = dag.output_nodes()
        assert len(outputs) == 1
        assert outputs[0].is_output

    def test_operation_nodes(self) -> None:
        """Test operation nodes query."""
        dag = self._build_linear_dag()
        ops = dag.operation_nodes()
        assert len(ops) == 1
        assert ops[0].is_operation

    def test_nodes_property(self) -> None:
        """Test nodes property returns sorted list."""
        dag = self._build_linear_dag()
        nodes = dag.nodes
        assert len(nodes) == 3
        assert [n.node_id for n in nodes] == [0, 1, 2]


class TestDAGCircuitTopologicalOrder:
    """Test topological sorting."""

    def test_linear_chain(self) -> None:
        """Test topological order for linear chain."""
        dag = DAGCircuit()
        dag.set_qubit_count(1)
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=2, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_edge(DAGEdge(0, 1, "q0"))
        dag.add_edge(DAGEdge(1, 2, "q0"))

        order = dag.topological_order()
        assert order == [0, 1, 2]

    def test_diamond_shape(self) -> None:
        """Test topological order for diamond DAG."""
        dag = DAGCircuit()
        dag.set_qubit_count(1)
        # IN → A → C → OUT
        # IN → B → C → OUT
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=2, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=3, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=4, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_edge(DAGEdge(0, 1, "q0"))
        dag.add_edge(DAGEdge(0, 2, "q0"))
        dag.add_edge(DAGEdge(1, 3, "q0"))
        dag.add_edge(DAGEdge(2, 3, "q0"))
        dag.add_edge(DAGEdge(3, 4, "q0"))

        order = dag.topological_order()
        assert order[0] == 0  # INPUT first
        assert order[-1] == 4  # OUTPUT last
        assert {1, 2}.issubset(set(order[1:3]))  # A, B in middle
        assert order[3] == 3  # C after both

    def test_cycle_detected(self) -> None:
        """Test that cycles are detected."""
        dag = DAGCircuit()
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0}))
        # Create a cycle
        dag.add_edge(DAGEdge(0, 1, "q0"))
        dag.add_edge(DAGEdge(1, 0, "q0"))

        with pytest.raises(RuntimeError, match="Cycle"):
            dag.topological_order()


class TestDAGCircuitLayers:
    """Test layer (depth) computation."""

    def test_layers_linear(self) -> None:
        """Test layers for linear DAG."""
        dag = DAGCircuit()
        dag.set_qubit_count(1)
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=2, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_edge(DAGEdge(0, 1, "q0"))
        dag.add_edge(DAGEdge(1, 2, "q0"))

        layers = dag.layers()
        assert len(layers) == 3
        assert layers[0] == [0]  # INPUT
        assert layers[1] == [1]  # OP
        assert layers[2] == [2]  # OUTPUT

    def test_layers_parallel(self) -> None:
        """Test layers with parallel operations on different qubits."""
        dag = DAGCircuit()
        dag.set_qubit_count(2)
        # q0: IN → OP-A → OUT
        # q1: IN → OP-B → OUT
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.INPUT, qubits={1}))
        dag.add_node(DAGNode(node_id=2, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=3, node_type=DAGNodeType.OPERATION, qubits={1}))
        dag.add_node(DAGNode(node_id=4, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=5, node_type=DAGNodeType.OUTPUT, qubits={1}))
        dag.add_edge(DAGEdge(0, 2, "q0"))
        dag.add_edge(DAGEdge(1, 3, "q1"))
        dag.add_edge(DAGEdge(2, 4, "q0"))
        dag.add_edge(DAGEdge(3, 5, "q1"))

        layers = dag.layers()
        # Both OP-A and OP-B should be in the same layer (depth 1)
        assert len(layers) == 3
        assert len(layers[0]) == 2  # Two INPUT nodes
        assert len(layers[1]) == 2  # Two parallel OP nodes
        assert len(layers[2]) == 2  # Two OUTPUT nodes

    def test_critical_path_length(self) -> None:
        """Test critical path length."""
        dag = DAGCircuit()
        dag.set_qubit_count(1)
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=2, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=3, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_edge(DAGEdge(0, 1, "q0"))
        dag.add_edge(DAGEdge(1, 2, "q0"))
        dag.add_edge(DAGEdge(2, 3, "q0"))

        assert dag.critical_path_length() == 2  # Two OP nodes


class TestDAGCircuitWireQueries:
    """Test wire-related queries."""

    def test_nodes_on_wire(self) -> None:
        """Test nodes_on_wire query."""
        dag = DAGCircuit()
        dag.set_qubit_count(2)
        # q0: IN → OP1 → OP2 → OUT (node chain: 0 → 2 → 3 → 4)
        # q1: IN → OP2 → OUT      (node chain: 1 → 3 → 5)
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.INPUT, qubits={1}))
        dag.add_node(DAGNode(node_id=2, node_type=DAGNodeType.OPERATION, qubits={0}))
        dag.add_node(DAGNode(node_id=3, node_type=DAGNodeType.OPERATION, qubits={0, 1}))
        dag.add_node(DAGNode(node_id=4, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=5, node_type=DAGNodeType.OUTPUT, qubits={1}))
        # Edges with proper wire labels
        dag.add_edge(DAGEdge(0, 2, "q0"))  # IN:q0 → OP1
        dag.add_edge(DAGEdge(2, 3, "q0"))  # OP1 → OP2 (q0)
        dag.add_edge(DAGEdge(1, 3, "q1"))  # IN:q1 → OP2
        dag.add_edge(DAGEdge(3, 4, "q0"))  # OP2 → OUT:q0
        dag.add_edge(DAGEdge(3, 5, "q1"))  # OP2 → OUT:q1

        # nodes_on_wire looks at node qubits/clbits, not edge wires
        nodes_q0 = dag.nodes_on_wire("q0")
        assert len(nodes_q0) == 4  # IN:q0, OP1, OP2, OUT:q0

    def test_qubit_wires(self) -> None:
        """Test qubit_wires query."""
        dag = DAGCircuit()
        dag.set_qubit_count(3)
        assert dag.qubit_wires() == [0, 1, 2]

    def test_clbit_wires(self) -> None:
        """Test clbit_wires query."""
        dag = DAGCircuit()
        dag.set_clbit_count(2)
        assert dag.clbit_wires() == [0, 1]


class TestDAGCircuitSummary:
    """Test summary string."""

    def test_summary(self) -> None:
        """Test summary output."""
        dag = DAGCircuit()
        dag.set_qubit_count(1)
        dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
        dag.add_node(DAGNode(node_id=1, node_type=DAGNodeType.OUTPUT, qubits={0}))
        dag.add_edge(DAGEdge(0, 1, "q0"))

        s = dag.summary()
        assert "DAGCircuit" in s
        assert "1" in s  # num qubits
