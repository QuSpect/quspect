"""Removal of identity-equivalent single-qubit gates.

Rotation merging and fusion can leave gates such as ``rz(0)``,
``rz(2π)`` or ``u3(0, 0, 0)`` behind. They are the identity up to a
global phase and only add depth, so this pass deletes them and splices
their wires back together.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.passes.base import BasePass
from quspect.gates.gate import Gate

logger = logging.getLogger(__name__)

#: Single-parameter rotations that are the identity (up to phase) when
#: their angle is a multiple of 2π.
_ROTATIONS: frozenset[str] = frozenset({"rx", "ry", "rz", "p", "u1"})

#: Multi-parameter single-qubit gates checked through their matrix.
_MATRIX_CHECKED: frozenset[str] = frozenset({"u2", "u3"})

#: Angle tolerance in radians.
_ATOL: float = 1e-9


def is_identity_gate(gate: Gate) -> bool:
    """Return whether a single-qubit gate is the identity up to global phase.

    Recognises ``rx/ry/rz/p/u1`` with an angle ≡ 0 (mod 2π) and ``u2/u3``
    whose matrix is proportional to the identity. Explicit ``id`` gates
    are deliberately *not* reported: they are commonly used to pin the
    width of a circuit. Gates with symbolic parameters are never
    considered identities.

    Args:
        gate: The gate to test.

    Returns:
        True if the gate can be removed without changing the circuit
        (up to global phase).

    Example:
        >>> import math
        >>> from quspect.gates.parameterized import RZ
        >>> is_identity_gate(RZ.with_params(2 * math.pi))
        True
        >>> is_identity_gate(RZ.with_params(0.1))
        False
    """
    if gate.num_qubits != 1:
        return False
    if not gate.params or not all(isinstance(p, (int, float)) for p in gate.params):
        return False
    if gate.name in _ROTATIONS:
        t = math.fmod(float(gate.params[0]), 2.0 * math.pi)
        return abs(t) < _ATOL or abs(abs(t) - 2.0 * math.pi) < _ATOL
    if gate.name in _MATRIX_CHECKED:
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.synthesis import is_identity_up_to_phase

        return is_identity_up_to_phase(np.asarray(gate_matrix(gate)), _ATOL)
    return False


class RemoveIdentityGates(BasePass):
    """Delete single-qubit gates that are the identity up to global phase.

    Each removed gate's wire predecessor is reconnected to its wire
    successor, so the DAG remains a valid circuit DAG. Typical inputs are
    the ``rz(0)`` / ``rz(2π)`` leftovers of rotation merging and the
    ``u3(0, 0, 0)`` produced when an identity run is fused.

    Example:
        >>> from quspect import QuantumCircuit
        >>> dag = QuantumCircuit().rz(0.0, 0).h(0).to_dag()
        >>> dag = RemoveIdentityGates().run(dag)
        >>> [n.instruction.gate.name for n in dag.operation_nodes()]
        ['h']
    """

    reports_modifications = True

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute identity removal on the DAG.

        Args:
            dag: The input DAG.

        Returns:
            The optimized DAG (modified in place).
        """
        doomed: list[int] = []
        for wire_idx in range(dag.num_qubits):
            for nid in dag.nodes_on_wire(f"q{wire_idx}"):
                node = dag.get_node(nid)
                if (
                    node.is_gate
                    and node.instruction is not None
                    and node.instruction.gate is not None
                    and is_identity_gate(node.instruction.gate)
                ):
                    doomed.append(nid)
        for nid in doomed:
            if dag.has_node(nid):
                dag.remove_node_and_reconnect(nid)
                logger.debug("Removed identity gate (node %d)", nid)
        return dag
