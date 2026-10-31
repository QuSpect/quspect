"""Circuit compiler: QuantumCircuit → DAGCircuit.

This module implements the compilation algorithm that converts a sequential
:class:`QuantumCircuit` into a :class:`DAGCircuit` intermediate representation.

Algorithm (from architecture design doc, section 3.4.4):
    1. Create DAGCircuit instance
    2. Create INPUT nodes (one per qubit/clbit line)
    3. Create OUTPUT nodes (one per qubit/clbit line)
    4. For each instruction, create an OPERATION node and connect it
       to the preceding nodes on the same qubit/bit lines
    5. Connect the last node on each line to the OUTPUT node
    6. Return the DAGCircuit

Complexity: O(n + m) where n = num_qubits + num_clbits, m = num_instructions.
"""

from __future__ import annotations

from quspect.circuit.circuit import QuantumCircuit
from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode, DAGNodeType


class CircuitCompiler:
    """Compiler that converts a QuantumCircuit into a DAGCircuit.

    The compiler follows the algorithm described in the architecture
    design document (section 3.4.4). It handles all instruction types
    (GATE, MEASURE, BARRIER) and edge cases including empty circuits,
    sparse qubit indices, and multi-qubit operations.

    Example:
        >>> from quspect import QuantumCircuit
        >>> circ = QuantumCircuit()
        >>> circ.h(0).cx(0, 1).measure_all()
        >>> dag = CircuitCompiler.compile(circ)
        >>> dag.summary()
        DAGCircuit: 8 nodes, 7 edges
          Qubits: 2
          Clbits: 2
          ...
    """

    @staticmethod
    def compile(circuit: QuantumCircuit) -> DAGCircuit:
        """Compile a QuantumCircuit into a DAGCircuit.

        Args:
            circuit: The quantum circuit to compile.

        Returns:
            A DAGCircuit representing the dependency structure of the circuit.

        Raises:
            ValueError: If the circuit contains invalid instructions.
        """
        dag = DAGCircuit()
        n_qubits = circuit.num_qubits
        n_clbits = circuit.num_clbits

        dag.set_qubit_count(n_qubits)
        dag.set_clbit_count(n_clbits)

        # Edge case: empty circuit — only INPUT/OUTPUT nodes
        if len(circuit) == 0 and n_qubits == 0:
            return dag

        # Track the next available node_id
        next_id = 0

        # Step 2: Create INPUT nodes
        qubit_input_nodes: dict[int, int] = {}
        for q in range(n_qubits):
            node = DAGNode(
                node_id=next_id,
                node_type=DAGNodeType.INPUT,
                qubits={q},
                label=f"IN:q{q}",
            )
            dag.add_node(node)
            qubit_input_nodes[q] = next_id
            next_id += 1

        clbit_input_nodes: dict[int, int] = {}
        for c in range(n_clbits):
            node = DAGNode(
                node_id=next_id,
                node_type=DAGNodeType.INPUT,
                clbits={c},
                label=f"IN:c{c}",
            )
            dag.add_node(node)
            clbit_input_nodes[c] = next_id
            next_id += 1

        # Step 4: Maintain last_node per wire
        # qubit_last[q] = node_id of the last node touching qubit q
        qubit_last: dict[int, int] = dict(qubit_input_nodes)
        # clbit_last[c] = node_id of the last node touching clbit c
        clbit_last: dict[int, int] = dict(clbit_input_nodes)

        # Step 5: Process each instruction
        for instr in circuit:
            # Determine which qubits and clbits this instruction touches
            involved_qubits = set(instr.qubits)
            involved_clbits = set(instr.clbits)

            # For BARRIER, create a node that blocks all specified qubits
            # (No gate matrix, just a dependency node)

            # Create the OPERATION node
            op_node = DAGNode(
                node_id=next_id,
                node_type=DAGNodeType.OPERATION,
                instruction=instr,
                qubits=involved_qubits,
                clbits=involved_clbits,
            )
            dag.add_node(op_node)
            op_id = next_id
            next_id += 1

            # Add edges from the last node on each involved wire
            for q in involved_qubits:
                dag.add_edge(
                    DAGEdge(
                        source_id=qubit_last[q],
                        target_id=op_id,
                        wire=f"q{q}",
                    )
                )
                # Update last node for this qubit
                qubit_last[q] = op_id

            for c in involved_clbits:
                dag.add_edge(
                    DAGEdge(
                        source_id=clbit_last[c],
                        target_id=op_id,
                        wire=f"c{c}",
                    )
                )
                # Update last node for this clbit
                clbit_last[c] = op_id

        # Step 6: Create OUTPUT nodes and connect last nodes
        for q in range(n_qubits):
            out_node = DAGNode(
                node_id=next_id,
                node_type=DAGNodeType.OUTPUT,
                qubits={q},
                label=f"OUT:q{q}",
            )
            dag.add_node(out_node)
            out_id = next_id
            next_id += 1

            dag.add_edge(
                DAGEdge(
                    source_id=qubit_last[q],
                    target_id=out_id,
                    wire=f"q{q}",
                )
            )

        for c in range(n_clbits):
            out_node = DAGNode(
                node_id=next_id,
                node_type=DAGNodeType.OUTPUT,
                clbits={c},
                label=f"OUT:c{c}",
            )
            dag.add_node(out_node)
            out_id = next_id
            next_id += 1

            dag.add_edge(
                DAGEdge(
                    source_id=clbit_last[c],
                    target_id=out_id,
                    wire=f"c{c}",
                )
            )

        return dag
