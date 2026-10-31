"""Rotation merging optimization pass.

This pass merges consecutive rotation gates around the same axis on the
same qubit into a single rotation. For example:

    RX(0.3) followed by RX(0.5) → RX(0.8)
    RZ(π/2) followed by RZ(π/4) → RZ(3π/4)

This is especially effective for QFT and QAOA circuits where rotation
angles accumulate along the same qubit lines.
"""

from __future__ import annotations

import logging

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode
from quspect.dag.passes.base import BasePass

logger = logging.getLogger(__name__)

#: Rotation gate names that belong to each axis family.
#: Gates in the same family on the same qubit can be merged.
ROTATION_FAMILIES: dict[str, str] = {
    "rx": "rx",  # → single merged RX
    "ry": "ry",  # → single merged RY
    "rz": "rz",  # → single merged RZ
    "p": "rz",  # Phase(θ) = RZ(θ) up to global phase
    "u1": "rz",  # U1 = Phase = RZ
}


class RotationMerging(BasePass):
    """Merge consecutive same-axis rotations on the same qubit.

    This pass scans each qubit wire for consecutive rotation gates
    belonging to the same rotation family and merges them into a
    single combined rotation.

    The algorithm:
    1. For each qubit wire, collect gate nodes
    2. Scan for consecutive pairs in the same rotation family
    3. When found, replace the first node with the merged rotation
       and remove the second node

    Note:
        Only merges rotations with **concrete float parameters**.
        Symbolic parameters (``Parameter`` instances) are left
        untouched to preserve variational circuit semantics.

    Example:
        >>> dag = circuit.to_dag()
        >>> merger = RotationMerging()
        >>> optimized = merger.run(dag)
        >>> # RX(0.3) RX(0.5) on qubit 0 → RX(0.8)
    """

    reports_modifications = True

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute rotation merging on the DAG.

        Args:
            dag: The input DAG.

        Returns:
            The optimized DAG.
        """
        changed = True
        while changed:
            changed = False
            for wire_idx in range(dag.num_qubits):
                if self._merge_on_wire(dag, f"q{wire_idx}"):
                    changed = True
        return dag

    def _merge_on_wire(self, dag: DAGCircuit, wire: str) -> bool:
        """Scan one wire and merge consecutive same-axis rotations.

        Args:
            dag: The DAG.
            wire: Wire identifier (e.g. ``"q0"``).

        Returns:
            True if at least one merge was performed.
        """
        op_nodes = dag.nodes_on_wire(wire)
        gate_nodes = [
            nid
            for nid in op_nodes
            if dag.get_node(nid).is_gate
            and dag.get_node(nid).instruction is not None
            and dag.get_node(nid).instruction.gate is not None  # type: ignore[union-attr]
        ]

        if len(gate_nodes) < 2:
            return False

        merged = False
        i = 0
        while i < len(gate_nodes) - 1:
            nid_a = gate_nodes[i]
            nid_b = gate_nodes[i + 1]

            node_a = dag.get_node(nid_a)
            node_b = dag.get_node(nid_b)

            # Both must be single-qubit gates on the same qubit
            if node_a.instruction is None or node_b.instruction is None:
                i += 1
                continue
            if node_a.instruction.gate is None or node_b.instruction.gate is None:
                i += 1
                continue

            name_a = node_a.instruction.gate.name
            name_b = node_b.instruction.gate.name

            # Check they belong to the same rotation family
            family_a = ROTATION_FAMILIES.get(name_a)
            family_b = ROTATION_FAMILIES.get(name_b)
            if family_a is None or family_b is None or family_a != family_b:
                i += 1
                continue

            # Both must have concrete float params
            angle_a = self._get_rotation_angle(node_a, name_a)
            angle_b = self._get_rotation_angle(node_b, name_b)
            if angle_a is None or angle_b is None:
                i += 1
                continue

            # Check they're consecutive
            if not self._are_consecutive(dag, nid_a, nid_b, wire):
                i += 1
                continue

            # Merge: remove node B, update node A with sum angle
            merged_angle = angle_a + angle_b
            self._update_rotation_node(dag, node_a, name_a, merged_angle)
            self._remove_node(dag, nid_b, wire)

            # nid_b (at position i + 1) is the only node that left the wire.
            del gate_nodes[i + 1]
            merged = True
            # Don't advance i — the updated node_a might now be
            # consecutive with the next node

        return merged

    def _get_rotation_angle(self, node: DAGNode, name: str) -> float | None:
        """Extract the rotation angle from a node's gate params.

        Args:
            node: The DAG node.
            name: Gate name.

        Returns:
            The rotation angle in radians, or None if parameters
            are not concrete floats.
        """
        if node.instruction is None or node.instruction.gate is None:
            return None

        gate = node.instruction.gate
        if gate.params:
            p = gate.params[0]
            if isinstance(p, (int, float)):
                return float(p)
            return None

        return None

    def _update_rotation_node(
        self, dag: DAGCircuit, node: DAGNode, gate_name: str, new_angle: float
    ) -> None:
        """Update a rotation node with a new merged angle.

        Args:
            dag: The DAG containing ``node`` (records the modification).
            node: The node to update.
            gate_name: Gate family name (rx, ry, rz, p).
            new_angle: The merged rotation angle.
        """
        from quspect.circuit.instruction import CircuitInstruction
        from quspect.gates.parameterized import RX, RY, RZ, P

        # Create new gate with merged angle
        if gate_name == "rx":
            new_gate = RX.with_params(float(new_angle))
        elif gate_name == "ry":
            new_gate = RY.with_params(float(new_angle))
        elif gate_name == "rz":
            new_gate = RZ.with_params(float(new_angle))
        else:  # p, u1
            new_gate = P.with_params(float(new_angle))

        # Update the node's instruction
        new_instr = CircuitInstruction(
            type=node.instruction.type,  # type: ignore[union-attr]
            gate=new_gate,
            qubits=node.instruction.qubits,  # type: ignore[union-attr]
            clbits=node.instruction.clbits,  # type: ignore[union-attr]
        )
        dag.substitute_node_instruction(node.node_id, new_instr)

    def _are_consecutive(
        self, dag: DAGCircuit, nid_a: int, nid_b: int, wire: str
    ) -> bool:
        """Check if two nodes are consecutive on a wire."""
        return dag.has_edge(nid_a, nid_b, wire)

    def _remove_node(self, dag: DAGCircuit, nid: int, wire: str) -> None:
        """Remove a node from the DAG and reconnect the wire.

        Args:
            dag: The DAG.
            nid: Node ID to remove.
            wire: Wire identifier.
        """
        # Find predecessor and successor on this wire
        pred_nid = dag.wire_predecessor(nid, wire)
        succ_nid = dag.wire_successor(nid, wire)

        # Remove the node
        dag.remove_node(nid)

        # Add reconnection edge
        if pred_nid is not None and succ_nid is not None:
            dag.add_edge(DAGEdge(source_id=pred_nid, target_id=succ_nid, wire=wire))
