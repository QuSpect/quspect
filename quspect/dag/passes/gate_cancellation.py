"""Gate cancellation optimization pass.

This pass identifies adjacent inverse gate pairs on the same qubit wire(s)
and removes both gates. This is the highest-ROI optimization according to
2024 research — gate cancellation improves 68% of circuits and is the
foundation of the cancel-first pipeline.

Supported cancellations:
    - Single-qubit self-inverse: H², X², Y², Z²
    - Single-qubit inverse pairs: S·S†, T·T†
    - Single-qubit rotation: RX(θ)·RX(-θ), RY(θ)·RY(-θ), RZ(θ)·RZ(-θ), P(θ)·P(-θ)
    - Two-qubit: CNOT², CZ², SWAP²
    - Two-qubit rotation: CRX(θ)·CRX(-θ), CRY(θ)·CRY(-θ), CRZ(θ)·CRZ(-θ)
"""

from __future__ import annotations

import logging

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode
from quspect.dag.passes.base import BasePass

logger = logging.getLogger(__name__)

#: Gate names that are their own inverse (self-inverse).
SELF_INVERSE_GATES: set[str] = {"h", "x", "y", "z", "cx", "cz", "swap", "id"}

#: Gate name pairs that cancel each other (order-independent).
INVERSE_PAIRS: set[tuple[str, str]] = {
    ("s", "sdg"),
    ("sdg", "s"),
    ("t", "tdg"),
    ("tdg", "t"),
}

#: Rotation gate families — gates in the same family cancel if params sum to 0.
ROTATION_FAMILIES: set[str] = {"rx", "ry", "rz", "p", "crx", "cry", "crz", "cp"}

#: Tolerance for angle sum to be considered zero (in radians).
ANGLE_TOLERANCE: float = 1e-10


def _gates_cancel(
    name_a: str,
    name_b: str,
    params_a: tuple[float | int, ...] = (),
    params_b: tuple[float | int, ...] = (),
) -> bool:
    """Return True if two gates cancel each other.

    This handles three cases:
        1. Self-inverse gates: same name, appear twice → cancel.
        2. Explicit inverse pairs: S/SDG, T/TDG → cancel.
        3. Rotation gates: same name, params sum to zero → cancel.

    Args:
        name_a: Name of the first gate.
        name_b: Name of the second gate.
        params_a: Parameters of the first gate.
        params_b: Parameters of the second gate.

    Returns:
        True if applying both gates in sequence yields identity.
    """
    # Case 1: Self-inverse (same name, no params)
    if name_a == name_b:
        if name_a in SELF_INVERSE_GATES:
            return True
        # Case 3: Rotation gates with params that sum to zero
        if name_a in ROTATION_FAMILIES and params_a and params_b:
            # Check all params sum to zero
            if len(params_a) != len(params_b):
                return False
            if not all(isinstance(p, (int, float)) for p in params_a):
                return False
            if not all(isinstance(p, (int, float)) for p in params_b):
                return False
            return all(
                abs(float(params_a[i]) + float(params_b[i])) < ANGLE_TOLERANCE
                for i in range(len(params_a))
            )

    # Case 2: Explicit inverse pairs
    return (name_a, name_b) in INVERSE_PAIRS


def _get_gate_params(node: DAGNode) -> tuple[float | int, ...]:
    """Extract gate parameters from a DAG node.

    Args:
        node: A DAG operation node.

    Returns:
        Tuple of gate parameters, or empty tuple if none.
    """
    if node.instruction is None or node.instruction.gate is None:
        return ()
    return node.instruction.gate.params


class GateCancellation(BasePass):
    """Remove adjacent inverse gate pairs from a DAGCircuit.

    This pass scans each qubit wire for consecutive operations that
    are mutual inverses, removes both nodes, and reconnects the
    graph. It operates in-place on the DAG.

    The algorithm:
        1. For each qubit wire, walk the chain of nodes.
        2. When two consecutive gate nodes are found to cancel,
           remove both and reconnect the wire.
        3. Repeat until no more cancellations are found on any wire.

    .. note::

        Rotation gates with symbolic (``Parameter``) params are
        never cancelled — the cancellation is only applied when
        both gates have concrete numeric parameters that sum to
        (near) zero.

    Example:
        >>> dag = circuit.to_dag()
        >>> cancellor = GateCancellation()
        >>> optimized = cancellor.run(dag)
        >>> # Two consecutive H gates on same qubit → removed
    """

    reports_modifications = True

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute gate cancellation on the DAG.

        Args:
            dag: The input DAG to optimize.

        Returns:
            The optimized DAG (same object, modified in-place).
        """
        removed: set[int] = set()

        changed = True
        while changed:
            changed = False
            for wire_idx in range(dag.num_qubits):
                if self._cancel_on_wire(dag, f"q{wire_idx}", removed):
                    changed = True
        return dag

    def _cancel_on_wire(self, dag: DAGCircuit, wire: str, removed: set[int]) -> bool:
        """Scan one wire for cancellable gate pairs.

        Args:
            dag: The DAG to scan.
            wire: Wire identifier (e.g. ``"q0"``).
            removed: Set of already-removed node IDs.

        Returns:
            True if at least one cancellation was performed.
        """
        op_nodes = dag.nodes_on_wire(wire)
        gate_nodes = [
            nid for nid in op_nodes if nid not in removed and dag.get_node(nid).is_gate
        ]

        if len(gate_nodes) < 2:
            return False

        canceled = False
        i = 0
        while i < len(gate_nodes) - 1:
            nid_a = gate_nodes[i]
            nid_b = gate_nodes[i + 1]

            if nid_a in removed:
                i += 1
                continue
            if nid_b in removed:
                i += 2
                continue

            node_a = dag.get_node(nid_a)
            node_b = dag.get_node(nid_b)

            # The pair cancels iff the gates are inverse, act on the same
            # ordered operands (operand order matters for controlled gates,
            # so the instruction tuples are compared, not DAGNode.qubits)
            # and are consecutive on every wire they touch.  Checking only
            # the wire currently being scanned is not sufficient: in
            # CX(q0, q1) - RZ(q1) - CX(q0, q1), for example, the CX gates
            # are consecutive on q0 but not on q1.  The conditions are
            # side-effect free, so the cheap ones are tested first.
            instr_a = node_a.instruction
            instr_b = node_b.instruction
            if (
                instr_a is None
                or instr_b is None
                or instr_a.gate is None
                or instr_b.gate is None
                or instr_a.qubits != instr_b.qubits
            ):
                i += 1
                continue
            name_a = instr_a.gate.name
            name_b = instr_b.gate.name

            params_a = _get_gate_params(node_a)
            params_b = _get_gate_params(node_b)

            if _gates_cancel(
                name_a, name_b, params_a, params_b
            ) and self._are_consecutive_on_all_wires(dag, nid_a, nid_b):
                self._remove_node_pair(dag, nid_a, nid_b)
                removed.add(nid_a)
                removed.add(nid_b)
                # Only nid_a/nid_b (at positions i, i+1) left the wire.
                del gate_nodes[i : i + 2]
                canceled = True
                logger.debug(
                    "Cancelled %s·%s on %s (nodes %d, %d)",
                    name_a,
                    name_b,
                    wire,
                    nid_a,
                    nid_b,
                )
            else:
                i += 1

        return canceled

    def _are_consecutive(
        self, dag: DAGCircuit, nid_a: int, nid_b: int, wire: str
    ) -> bool:
        """Check if two nodes are consecutive on a given wire.

        Two nodes A and B are consecutive on wire W if there is an
        edge A→B with wire=W, and no other OPERATION node on wire W
        sits between them.

        Args:
            dag: The DAG.
            nid_a: First node ID.
            nid_b: Second node ID.
            wire: Wire identifier.

        Returns:
            True if A and B are consecutive on the wire.
        """
        return dag.has_edge(nid_a, nid_b, wire)

    def _are_consecutive_on_all_wires(
        self, dag: DAGCircuit, nid_a: int, nid_b: int
    ) -> bool:
        """Return whether two operations are consecutive on every operand wire."""
        node_a = dag.get_node(nid_a)
        node_b = dag.get_node(nid_b)
        if node_a.instruction is None or node_b.instruction is None:
            return False
        if node_a.instruction.qubits != node_b.instruction.qubits:
            return False
        if node_a.instruction.clbits != node_b.instruction.clbits:
            return False

        wires = [f"q{q}" for q in node_a.instruction.qubits]
        wires.extend(f"c{c}" for c in node_a.instruction.clbits)
        return bool(wires) and all(
            self._are_consecutive(dag, nid_a, nid_b, wire) for wire in wires
        )

    def _remove_node_pair(
        self,
        dag: DAGCircuit,
        nid_a: int,
        nid_b: int,
    ) -> None:
        """Remove two canceling nodes and reconnect every operand wire.

        Args:
            dag: The DAG to modify.
            nid_a: First node to remove.
            nid_b: Second node to remove.
        The caller guarantees that the nodes are consecutive on every wire.
        """
        node_a = dag.get_node(nid_a)
        if node_a.instruction is None:
            return

        wires = [f"q{q}" for q in node_a.instruction.qubits]
        wires.extend(f"c{c}" for c in node_a.instruction.clbits)
        reconnect: list[tuple[int, int, str]] = []
        for wire in wires:
            pred_a = dag.wire_predecessor(nid_a, wire)
            succ_b = dag.wire_successor(nid_b, wire)
            if pred_a is None or succ_b is None:
                raise RuntimeError(f"Wire {wire} of node {nid_a}/{nid_b} is broken.")
            reconnect.append((pred_a, succ_b, wire))

        # Remove both nodes
        dag.remove_node(nid_a)
        dag.remove_node(nid_b)

        # Restore each wire independently.
        for pred_a, succ_b, wire in reconnect:
            dag.add_edge(DAGEdge(source_id=pred_a, target_id=succ_b, wire=wire))
