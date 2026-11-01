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

from quspect.circuit._native import bit_counts
from quspect.circuit.circuit import QuantumCircuit
from quspect.dag.dag_circuit import DAGCircuit


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
        counts = None
        if type(circuit) is QuantumCircuit:
            # Both bit counts in one native pass over the instruction list
            # (None when the extension is missing or an instruction is not a
            # plain CircuitInstruction).
            counts = bit_counts(circuit._instructions)
        if counts is not None:
            n_qubits, n_clbits = counts
        else:
            n_qubits = circuit.num_qubits
            n_clbits = circuit.num_clbits

        dag.set_qubit_count(n_qubits)
        dag.set_clbit_count(n_clbits)

        # Edge case: empty circuit — only INPUT/OUTPUT nodes
        if len(circuit) == 0 and n_qubits == 0:
            return dag

        # Steps 2-6 (INPUT nodes, one OPERATION node per instruction wired to
        # the previous node on each of its lines, OUTPUT nodes) run in one
        # call into the DAG storage core.
        instructions = (
            circuit._instructions
            if type(circuit) is QuantumCircuit
            else circuit.instructions
        )
        dag._build_from_instructions(instructions, n_qubits, n_clbits)
        return dag
