"""DAG compiler: DAGCircuit → QuantumCircuit reconstruction.

This module converts an optimized :class:`DAGCircuit` back into a
:class:`QuantumCircuit`, enabling the full optimization pipeline:

    QuantumCircuit → DAGCircuit → (optimize) → DAGCircuit → QuantumCircuit
"""

from __future__ import annotations

import logging

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import InstructionType
from quspect.dag.dag_circuit import DAGCircuit

logger = logging.getLogger(__name__)


class DAGToCircuitCompiler:
    """Compiler that reconstructs a QuantumCircuit from a DAGCircuit.

    This is the reverse direction of :class:`CircuitCompiler`, enabling
    the DAG optimization pipeline to produce a usable circuit at the end.

    The reconstruction algorithm:
        1. Get all OPERATION nodes in topological order.
        2. Extract the instruction from each node.
        3. Add the instruction to a new QuantumCircuit. Gates,
           measurements, barriers and noise channels (``CHANNEL``
           instructions) are all preserved.

    .. note::

        INPUT and OUTPUT boundary nodes are ignored — they exist only
        to mark qubit/bit line boundaries in the DAG.

    Example:
        >>> circ = QuantumCircuit()
        >>> circ.h(0).cx(0, 1).measure_all()
        >>> dag = circ.to_dag()
        >>> # ... optimize dag ...
        >>> reconstructed = DAGToCircuitCompiler.compile(dag)
        >>> reconstructed.num_qubits
        2
    """

    @staticmethod
    def compile(dag: DAGCircuit, *, name: str = "") -> QuantumCircuit:
        """Reconstruct a QuantumCircuit from a DAGCircuit.

        Args:
            dag: The DAG to compile, typically after optimization passes.
            name: Optional name for the reconstructed circuit. If empty,
                defaults to ``"{original}_optimized"``.

        Returns:
            A new QuantumCircuit containing all operations from the DAG
            in topological order.

        Raises:
            ValueError: If a node has an invalid or missing instruction.
        """
        circuit_name = name if name else ""
        circuit = QuantumCircuit(name=circuit_name)

        # Collect OPERATION nodes in topological order
        op_nodes = dag.operation_nodes()
        logger.debug(
            "DAGToCircuit: reconstructing circuit from %d operation nodes",
            len(op_nodes),
        )

        for node in op_nodes:
            if node.instruction is None:
                logger.warning(
                    "DAGToCircuit: node %d has no instruction, skipping",
                    node.node_id,
                )
                continue

            instr = node.instruction

            if instr.type == InstructionType.GATE:
                if instr.gate is None:
                    logger.warning(
                        "DAGToCircuit: GATE node %d has no gate, skipping", node.node_id
                    )
                    continue
                circuit.append(instr.gate, instr.qubits, instr.clbits)

            elif instr.type == InstructionType.MEASURE:
                circuit.measure(instr.qubits, instr.clbits)

            elif instr.type == InstructionType.BARRIER:
                circuit.barrier(*instr.qubits)

            elif instr.type == InstructionType.CHANNEL:
                if instr.channel is None:
                    logger.warning(
                        "DAGToCircuit: CHANNEL node %d has no channel, skipping",
                        node.node_id,
                    )
                    continue
                circuit.append_channel(instr.channel, instr.qubits)

            else:
                logger.warning(
                    "DAGToCircuit: unknown instruction type %s on node %d, skipping",
                    instr.type,
                    node.node_id,
                )

        return circuit
