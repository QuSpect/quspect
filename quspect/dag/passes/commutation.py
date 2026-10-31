"""Commutation analysis and reordering pass for DAG circuits.

This pass identifies pairs of adjacent operations that can be commuted
(reordered) without changing the circuit's overall unitary, then actively
reorders them to group cancellable gates together.

Key commutation rules:
    - Gates on disjoint qubits always commute
    - Z-rotation commutes with CNOT control
    - X-rotation commutes with CNOT target
    - CNOTs with same control commute with each other
    - CNOTs with same target commute with each other
    - RZ commutes with RZ (diagonal gates commute)
    - Same-name gates on same qubit → potential cancellation target
"""

from __future__ import annotations

import logging

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode
from quspect.dag.passes.base import BasePass

logger = logging.getLogger(__name__)

#: Gate names in the "Z-family" (diagonal in computational basis,
#: commute with each other and with CNOT controls).
Z_FAMILY: set[str] = {"z", "s", "sdg", "t", "tdg", "rz", "p", "cz"}

#: Gate names in the "X-family" (commute with CNOT targets).
X_FAMILY: set[str] = {"x", "rx"}

#: Gate names that commute when on same qubits (CNOT-like).
CNOT_LIKE: set[str] = {"cx", "cz"}


class CommutationAnalysis(BasePass):
    """Analyze and reorder operations to enable further cancellations.

    This pass performs commutation analysis on the DAGCircuit and
    actively reorders gates to bring cancellable pairs together.
    It does NOT remove any gates — that is left to subsequent
    :class:`GateCancellation` or :class:`RotationMerging` passes.

    The algorithm:
        1. Identify all pairs of adjacent operations that commute.
        2. On each qubit wire, scan for same-name gates separated
           by commuting gates.
        3. Swap commuting gates to bring identical gates adjacent
           to each other, creating cancellation opportunities.
        4. Apply the reordering directly on the DAG structure.

    .. note::

        This is an **enabling** pass — it doesn't reduce gate count
        on its own, but dramatically improves the effectiveness of
        the cancellation and merging passes.

    Example:
        >>> dag = circuit.to_dag()
        >>> ca = CommutationAnalysis()
        >>> dag = ca.run(dag)
        >>> # Now GateCancellation will find more pairs
    """

    # Maximum look-ahead distance when searching for cancellable pairs.
    MAX_LOOKAHEAD: int = 5

    reports_modifications = True

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute commutation analysis and reordering.

        Args:
            dag: The input DAG.

        Returns:
            The DAG with commutation-based reordering applied.
        """
        changed = True
        while changed:
            changed = False
            for wire_idx in range(dag.num_qubits):
                if self._reorder_on_wire(dag, f"q{wire_idx}"):
                    changed = True
        return dag

    # ------------------------------------------------------------------
    # Wire-level reordering
    # ------------------------------------------------------------------

    def _reorder_on_wire(self, dag: DAGCircuit, wire: str) -> bool:
        """Reorder gates on one wire to bring cancellable pairs together.

        For each gate node on the wire, look ahead for another gate
        with the same name on the same qubits. If the gates between
        them all commute with the first gate, swap them to adjacent
        positions.

        Args:
            dag: The DAG.
            wire: Wire identifier (e.g. ``"q0"``).

        Returns:
            True if any reordering was performed.
        """
        gate_nodes = self._get_gate_nodes(dag, wire)
        if len(gate_nodes) < 2:
            return False

        # 本函数只重排节点、不改指令：按节点缓存门名与操作数
        get_node = dag.get_node
        names: dict[int, str | None] = {}
        operands: dict[int, tuple[int, ...] | None] = {}
        for nid in gate_nodes:
            node = get_node(nid)
            names[nid] = self._gate_name(node)
            operands[nid] = (
                None if node.instruction is None else node.instruction.qubits
            )

        reordered = False
        i = 0
        while i < len(gate_nodes):
            nid_i = gate_nodes[i]
            name_i = names[nid_i]
            if name_i is None:
                i += 1
                continue
            qubits_i = operands[nid_i]

            # Look ahead for a same-name gate
            best_j = -1
            for j in range(i + 1, min(i + self.MAX_LOOKAHEAD + 1, len(gate_nodes))):
                nid_j = gate_nodes[j]
                if (
                    names[nid_j] == name_i
                    and qubits_i is not None
                    and operands[nid_j] == qubits_i
                    and self._can_skip_all(dag, wire, gate_nodes, i, j, get_node(nid_i))
                ):
                    best_j = j
                    break

            if best_j > i + 1:
                # Swap to bring node_j adjacent to node_i
                if self._swap_chain(dag, wire, gate_nodes, i, best_j):
                    reordered = True
                    # gate_nodes was updated in place by the swaps and
                    # matches the new wire order.
                    continue
                else:
                    i += 1
            else:
                i += 1

        return reordered

    def _get_gate_nodes(self, dag: DAGCircuit, wire: str) -> list[int]:
        """Get ordered list of gate node IDs on a wire."""
        op_nodes = dag.nodes_on_wire(wire)
        result: list[int] = []
        for nid in op_nodes:
            node = dag.get_node(nid)
            if (
                node.is_gate
                and node.instruction is not None
                and node.instruction.gate is not None
            ):
                result.append(nid)
        return result

    def _gate_name(self, node: DAGNode) -> str | None:
        """Extract gate name from a DAG node."""
        if node.instruction is None or node.instruction.gate is None:
            return None
        return node.instruction.gate.name

    def _same_qubits(self, a: DAGNode, b: DAGNode) -> bool:
        """Check if two nodes have the same ordered qubit operands."""
        if a.instruction is None or b.instruction is None:
            return False
        return a.instruction.qubits == b.instruction.qubits

    # ------------------------------------------------------------------
    # Commutation checks
    # ------------------------------------------------------------------

    def _ops_commute(self, a: DAGNode, b: DAGNode) -> bool:
        """Determine if two operations commute.

        Returns True if swapping their execution order does not change
        the overall unitary.

        Args:
            a: First operation node.
            b: Second operation node.

        Returns:
            True if the operations commute.
        """
        if a.instruction is None or b.instruction is None:
            return False
        if a.instruction.gate is None or b.instruction.gate is None:
            return False

        name_a = a.instruction.gate.name
        name_b = b.instruction.gate.name
        qubits_a = a.instruction.qubits
        qubits_b = b.instruction.qubits

        # 1. Disjoint qubits → always commute
        if not a.qubits & b.qubits:
            return True

        # 2. Both are diagonal (Z-family) → commute
        if name_a in Z_FAMILY and name_b in Z_FAMILY:
            return True

        # 3. Identical CNOT-like operations commute.  Operand tuples are
        # ordered: CX(control, target) is not CX(target, control).
        if name_a == name_b and name_a in CNOT_LIKE and qubits_a == qubits_b:
            return True

        # CNOTs also commute when they share a control or share a target.
        if (
            name_a == name_b == "cx"
            and len(qubits_a) == len(qubits_b) == 2
            and (qubits_a[0] == qubits_b[0] or qubits_a[1] == qubits_b[1])
        ):
            return True

        # 4. X-family on the target (operand 1) of CNOT → commute
        if (
            name_a == "cx"
            and name_b in X_FAMILY
            and len(qubits_a) == 2
            and qubits_b == (qubits_a[1],)
        ):
            return True
        if (
            name_b == "cx"
            and name_a in X_FAMILY
            and len(qubits_b) == 2
            and qubits_a == (qubits_b[1],)
        ):
            return True

        # 5. Z-family on the control (operand 0) of CNOT → commute
        if (
            name_a == "cx"
            and name_b in Z_FAMILY
            and len(qubits_a) == 2
            and qubits_b == (qubits_a[0],)
        ):
            return True
        return bool(
            name_b == "cx"
            and name_a in Z_FAMILY
            and len(qubits_b) == 2
            and qubits_a == (qubits_b[0],)
        )

    def _can_skip_all(
        self,
        dag: DAGCircuit,
        wire: str,
        gate_nodes: list[int],
        start: int,
        end: int,
        target_node: DAGNode,
    ) -> bool:
        """Check if ``target_node`` commutes with all intermediate gates.

        Args:
            dag: The DAG.
            wire: Wire identifier.
            gate_nodes: Ordered list of gate node IDs.
            start: Index of the target gate.
            end: Index of the same-name gate to swap toward.
            target_node: The target gate node.

        Returns:
            True if target commutes with all intermediate gates.
        """
        for k in range(start + 1, end):
            mid_node = dag.get_node(gate_nodes[k])
            if not self._ops_commute(target_node, mid_node):
                return False
        return True

    # ------------------------------------------------------------------
    # DAG reordering: swap a chain of commuting gates
    # ------------------------------------------------------------------

    def _swap_chain(
        self,
        dag: DAGCircuit,
        wire: str,
        gate_nodes: list[int],
        target_idx: int,
        source_idx: int,
    ) -> bool:
        """Bring the gate at ``source_idx`` adjacent to the gate at ``target_idx``
        by swapping it with each intermediate gate.

        This performs a sequence of adjacent swaps:
            [A, B, C, D] → swap D with C → [A, B, D, C]
                          → swap D with B → [A, D, B, C]
                          → D is now adjacent to A

        The whole chain is transactional: if one swap is blocked, the
        earlier swaps are rolled back through the DAG's undo journal,
        restoring the exact previous edge order.

        Args:
            dag: The DAG.
            wire: Wire identifier.
            gate_nodes: Ordered list of gate node IDs.
            target_idx: Index of target gate.
            source_idx: Index of source gate to move.

        Returns:
            True if the chain swap was successful.
        """
        nid_source = gate_nodes[source_idx]
        original_gate_nodes = list(gate_nodes)
        savepoint = dag.begin_transaction()
        try:
            # Bubble the source node backward through the chain
            for k in range(source_idx, target_idx + 1, -1):
                nid_left = gate_nodes[k - 1]
                nid_right = gate_nodes[k]
                if not self._swap_two_nodes(dag, wire, nid_left, nid_right):
                    # A later swap can be blocked by a dependency on another
                    # wire.  Roll back earlier swaps in this chain so a failed
                    # reorder cannot leave the DAG half-mutated or oscillating.
                    dag.rollback(savepoint)
                    gate_nodes[:] = original_gate_nodes
                    return False
                # Update gate_nodes in-place for the next iteration
                gate_nodes[k - 1], gate_nodes[k] = gate_nodes[k], gate_nodes[k - 1]
        finally:
            dag.end_transaction()

        logger.debug(
            "CommutationAnalysis: moved node %d adjacent to node %d on %s",
            nid_source,
            gate_nodes[target_idx],
            wire,
        )
        return True

    def _swap_two_nodes(
        self, dag: DAGCircuit, wire: str, nid_a: int, nid_b: int
    ) -> bool:
        """Swap two adjacent nodes on a wire in the DAG.

        Rewires the DAG so that B appears before A on the given wire.
        Also handles edges from other wires that connect to these nodes.

        Precondition: A comes before B on ``wire`` (edge A→B on wire exists).
                      A and B commute.

        Args:
            dag: The DAG to modify.
            wire: Wire identifier.
            nid_a: First node (currently before B).
            nid_b: Second node (currently after A).

        Returns:
            True on success.
        """
        node_a = dag.get_node(nid_a)
        node_b = dag.get_node(nid_b)
        shared_wires = [f"q{q}" for q in sorted(node_a.qubits & node_b.qubits)]
        shared_wires.extend(f"c{c}" for c in sorted(node_a.clbits & node_b.clbits))
        if wire not in shared_wires:
            return False

        # Multi-qubit operations must be adjacent on every shared wire.
        # Reversing just one shared edge would either corrupt ordering or
        # create a cycle with the untouched A→B edge.
        if not all(
            dag.has_edge(nid_a, nid_b, shared_wire) for shared_wire in shared_wires
        ):
            return False

        rewires: list[tuple[int, int, str]] = []
        for shared_wire in shared_wires:
            pred_a = dag.wire_predecessor(nid_a, shared_wire)
            succ_b = dag.wire_successor(nid_b, shared_wire)
            if pred_a is None or succ_b is None:
                return False
            rewires.append((pred_a, succ_b, shared_wire))

        dag.ensure_topological_positions()
        savepoint = dag.begin_transaction()
        try:
            # Remove the three wire edges around A and B on every shared wire.
            for pred_a, succ_b, shared_wire in rewires:
                dag.remove_edge(pred_a, nid_a, shared_wire)
                dag.remove_edge(nid_a, nid_b, shared_wire)
                dag.remove_edge(nid_b, succ_b, shared_wire)

            # Add new edges in swapped order on every shared wire.
            for pred_a, succ_b, shared_wire in rewires:
                for src, tgt in ((pred_a, nid_b), (nid_b, nid_a), (nid_a, succ_b)):
                    dag.add_edge(
                        DAGEdge(source_id=src, target_id=tgt, wire=shared_wire),
                        defer_order_check=True,
                    )

            # A and B can be adjacent on every shared wire while another wire
            # still contains an A -> ... -> B dependency path. Reversing the
            # direct edges would then create a cycle, so validate
            # transactionally (incremental check, local to A and B).
            if not dag.reverse_adjacent_edge_order(nid_a, nid_b):
                dag.rollback(savepoint)
                return False
        finally:
            dag.end_transaction()

        return True
