"""Eliminate diagonal gates before Z-basis measurements.

This pass removes diagonal (Z-family) gates that appear immediately before
a Z-basis measurement on the same qubit wire. Since a Z-basis measurement
projects onto |0⟩/|1⟩ and diagonal gates only add phases, they do not
change the measurement outcome probabilities.

Formally, for any diagonal unitary D = diag(e^{iα}, e^{iβ}):

    P(|0⟩) = |⟨0| D |ψ⟩|² = |e^{iα} ⟨0|ψ⟩|² = |⟨0|ψ⟩|²
    P(|1⟩) = |⟨1| D |ψ⟩|² = |e^{iβ} ⟨1|ψ⟩|² = |⟨1|ψ⟩|²

So D can be removed without changing measurement statistics.

Eliminated gates:
    - Z, S, S†, T, T† (standard diagonal)
    - RZ(θ), Phase(θ), U1(λ) (parameterized diagonal)
"""

from __future__ import annotations

import logging

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge
from quspect.dag.passes.base import BasePass

logger = logging.getLogger(__name__)

#: Gate names of diagonal gates that can be eliminated before Z measurement.
DIAGONAL_ELIMINABLE: set[str] = {
    "id",  # identity — always removable
    "z",  # Pauli-Z
    "s",  # Phase gate π/2
    "sdg",  # S†
    "t",  # T gate π/4
    "tdg",  # T†
    "rz",  # RZ rotation
    "p",  # Phase gate
    "u1",  # U1 = Phase
}


class EliminateDiagonalBeforeMeasurement(BasePass):
    """Remove diagonal gates that appear immediately before Z-basis measurements.

    This pass scans each qubit wire and removes diagonal (Z-family) gates
    whose only successor on that wire is a MEASURE node. Since Z-basis
    measurements project onto the computational basis, diagonal phases
    do not affect the measurement outcome probabilities.

    The optimization is **sound** for Z-basis measurements. It is NOT
    applied before measurements in other bases (which would require
    basis-change tracking).

    Example:
        >>> circ = QuantumCircuit()
        >>> circ.h(0).z(0).measure([0], [0])
        >>> dag = circ.to_dag()
        >>> elim = EliminateDiagonalBeforeMeasurement()
        >>> optimized = elim.run(dag)
        >>> # Z gate before measurement is removed
    """

    reports_modifications = True

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute diagonal-before-measurement elimination.

        Args:
            dag: The input DAG.

        Returns:
            The optimized DAG (modified in-place).
        """
        changed = True
        while changed:
            changed = False
            for wire_idx in range(dag.num_qubits):
                if self._eliminate_on_wire(dag, f"q{wire_idx}"):
                    changed = True
        return dag

    def _eliminate_on_wire(self, dag: DAGCircuit, wire: str) -> bool:
        """Scan one wire and remove diagonal gates before measurements.

        Args:
            dag: The DAG.
            wire: Wire identifier (e.g. ``"q0"``).

        Returns:
            True if at least one gate was eliminated.
        """
        op_nodes = dag.nodes_on_wire(wire)

        # Build a map: node_id → list of successors on this wire
        succ_on_wire: dict[int, list[int]] = {}
        for nid in op_nodes:
            succ_on_wire[nid] = [
                tgt for tgt, w in dag.out_edges_with_wires(nid) if w == wire
            ]

        eliminated = False
        for nid in op_nodes:
            node = dag.get_node(nid)
            if not node.is_gate:
                continue
            if node.instruction is None or node.instruction.gate is None:
                continue

            gate_name = node.instruction.gate.name

            # Only eliminate diagonal gates
            if gate_name not in DIAGONAL_ELIMINABLE:
                continue

            # Must only have one successor on this wire that is a MEASURE
            successors = succ_on_wire.get(nid, [])
            if len(successors) != 1:
                continue

            succ_node = dag.get_node(successors[0])
            if not succ_node.is_measure:
                continue

            # Also check that the node has no other dependencies
            # (i.e., no two-qubit gate where this qubit is involved and
            # the result depends on the phase)
            all_succs = dag.successors(nid)
            for s in all_succs:
                sn = dag.get_node(s)
                if sn.is_gate and s != successors[0]:
                    # There is another gate depending on this diagonal —
                    # don't eliminate (the phase might matter for interference)
                    break
            else:
                # Safe to remove
                self._remove_node(dag, nid, wire)
                eliminated = True
                logger.debug(
                    "Eliminated diagonal gate '%s' (node %d) before measurement on %s",
                    gate_name,
                    nid,
                    wire,
                )
                # Update successors map
                del succ_on_wire[nid]

        return eliminated

    def _remove_node(self, dag: DAGCircuit, nid: int, wire: str) -> None:
        """Remove a node and reconnect the wire.

        Args:
            dag: The DAG.
            nid: Node ID to remove.
            wire: Wire identifier.
        """
        pred_nid = dag.wire_predecessor(nid, wire)
        succ_nid = dag.wire_successor(nid, wire)

        dag.remove_node(nid)

        if pred_nid is not None and succ_nid is not None:
            dag.add_edge(DAGEdge(source_id=pred_nid, target_id=succ_nid, wire=wire))
