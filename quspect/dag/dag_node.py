"""DAG node and edge data models for quantum circuit intermediate representation.

The DAG (Directed Acyclic Graph) is the core intermediate representation (IR)
for quantum circuit compilation, optimization, and analysis. It explicitly
encodes qubit/bit dependencies between operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from quspect.circuit.instruction import CircuitInstruction


class DAGNodeType(Enum):
    """Type of a DAG node.

    Attributes:
        INPUT: Circuit input boundary (one per qubit/clbit line).
        OUTPUT: Circuit output boundary (one per qubit/clbit line).
        OPERATION: A gate, measurement, or barrier operation.
    """

    INPUT = auto()
    OUTPUT = auto()
    OPERATION = auto()


@dataclass
class DAGNode:
    """A node in the DAG representation of a quantum circuit.

    Each node represents either an input/output boundary point or an
    operation (gate, measurement, barrier).

    DAGNodes are mutable to allow construction by the compiler, but
    should be treated as immutable once added to a DAGCircuit.

    Attributes:
        node_id: Unique identifier within the DAG.
        node_type: The type of this node (INPUT, OUTPUT, or OPERATION).
        instruction: The circuit instruction, for OPERATION nodes.
        qubits: Set of qubit indices this node involves.
        clbits: Set of classical bit indices this node involves.
        label: Optional human-readable label.
        metadata: Optional arbitrary metadata attached to this node.
    """

    node_id: int
    node_type: DAGNodeType
    instruction: CircuitInstruction | None = None
    qubits: set[int] = field(default_factory=set)
    clbits: set[int] = field(default_factory=set)
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_input(self) -> bool:
        """Return True if this is an input boundary node."""
        return self.node_type == DAGNodeType.INPUT

    @property
    def is_output(self) -> bool:
        """Return True if this is an output boundary node."""
        return self.node_type == DAGNodeType.OUTPUT

    @property
    def is_operation(self) -> bool:
        """Return True if this is an operation node."""
        return self.node_type == DAGNodeType.OPERATION

    @property
    def is_gate(self) -> bool:
        """Return True if this is a gate operation node."""
        return (
            self.node_type == DAGNodeType.OPERATION
            and self.instruction is not None
            and self.instruction.is_gate
        )

    @property
    def is_measure(self) -> bool:
        """Return True if this is a measurement operation node."""
        return (
            self.node_type == DAGNodeType.OPERATION
            and self.instruction is not None
            and self.instruction.is_measure
        )

    @property
    def is_barrier(self) -> bool:
        """Return True if this is a barrier operation node."""
        return (
            self.node_type == DAGNodeType.OPERATION
            and self.instruction is not None
            and self.instruction.is_barrier
        )

    def __repr__(self) -> str:
        if self.is_input:
            qubits_str = ", ".join(f"q{q}" for q in sorted(self.qubits))
            return f"DAGNode(id={self.node_id}, INPUT, qubits=[{qubits_str}])"
        if self.is_output:
            qubits_str = ", ".join(f"q{q}" for q in sorted(self.qubits))
            return f"DAGNode(id={self.node_id}, OUTPUT, qubits=[{qubits_str}])"
        gate_name = (
            self.instruction.gate.name
            if self.instruction and self.instruction.gate
            else self.instruction.type.name
            if self.instruction
            else "?"
        )
        qubits_str = ", ".join(f"q{q}" for q in sorted(self.qubits))
        return f"DAGNode(id={self.node_id}, OP[{gate_name}], qubits=[{qubits_str}])"


@dataclass
class DAGEdge:
    """A directed edge in the circuit DAG.

    An edge represents a dependency between two operations on the same
    qubit or classical bit. The edge direction follows the execution
    order: from earlier to later operations.

    Attributes:
        source_id: The node_id of the source node.
        target_id: The node_id of the target node.
        wire: The qubit or classical bit index this edge carries,
            encoded as a string: ``"q0"``, ``"q1"``, ``"c0"``, etc.
    """

    source_id: int
    target_id: int
    wire: str = ""
