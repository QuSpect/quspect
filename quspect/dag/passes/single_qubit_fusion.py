"""Single-qubit gate fusion optimization pass.

This pass fuses arbitrary sequences of consecutive single-qubit gates
on the same qubit into a single :class:`U3` gate using Euler decomposition.

Unlike :class:`RotationMerging`, which only merges same-axis rotations,
this pass handles arbitrary single-qubit gate combinations (H, X, Y, Z,
S, T, RX, RY, RZ, Phase, U1, U2, U3).

Algorithm:
    1. Scan each qubit wire for consecutive single-qubit gates.
    2. Compute the combined 2×2 unitary matrix of each fusible block.
    3. Decompose the combined matrix into a single U3(θ, φ, λ) gate.
    4. Replace the block with the fused gate.

Reference:
    Nielsen & Chuang, "Quantum Computation and Quantum Information",
    Section 4.2 — Single-qubit gates and ZYZ decomposition.
"""

from __future__ import annotations

import logging
from collections.abc import Collection

import numpy as np

from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge
from quspect.dag.passes.base import BasePass
from quspect.gates.gate import Gate

logger = logging.getLogger(__name__)

#: Gate names for single-qubit gates.
SINGLE_QUBIT_GATES: set[str] = {
    "id",
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u1",
    "u2",
    "u3",
}

#: Cost assigned to gates that cannot be decomposed into the basis.
_UNDECOMPOSABLE_COST: int = 10**6

#: Gate names that are diagonal and can be trivially merged.
DIAGONAL_GATES: set[str] = {"id", "z", "s", "sdg", "t", "tdg", "rz", "p", "u1"}


def _gate_to_matrix(gate: Gate) -> np.ndarray:
    """Compute the 2×2 unitary matrix for a single-qubit gate.

    Args:
        gate: A :class:`Gate` instance.

    Returns:
        A 2×2 complex numpy array.

    Raises:
        ValueError: If the gate has no matrix or symbolic parameters.
    """
    from quspect.gates._matrix_utils import gate_matrix

    # gate_matrix handles the central dispatch
    m = gate_matrix(gate)
    if m.shape != (2, 2):
        raise ValueError(
            f"Expected 2×2 matrix for single-qubit gate '{gate.name}', got {m.shape}"
        )
    return m


def _decompose_u3(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 2×2 unitary matrix into U3(θ, φ, λ) parameters.

    Uses the U3 convention from :func:`quspect.gates._matrix_utils._u3_matrix`:

        U3(θ, φ, λ) = [[cos(θ/2),               -sin(θ/2) · e^{iλ}  ],
                        [sin(θ/2) · e^{iφ},       cos(θ/2) · e^{i(φ+λ)}]]

    The matrix's global phase is removed using the phase of its ``u00``
    element whenever that element is non-zero. This is required because the
    QuSpect U3 convention fixes ``u00 = cos(θ/2)`` to be real and
    non-negative. For the ``θ ≈ π`` singularity, where ``u00 ≈ 0``, the
    global phase is absorbed into ``φ`` and ``λ`` instead.

    Args:
        matrix: A 2×2 unitary matrix.

    Returns:
        A tuple ``(theta, phi, lam)`` in radians.
    """
    u00 = matrix[0, 0]
    u01 = matrix[0, 1]
    u10 = matrix[1, 0]
    u11 = matrix[1, 1]

    # Extract θ from magnitudes: c = |u00|, s = |u10|
    c = float(abs(u00))
    s = float(abs(u10))

    # Clamp to [0, 1] to avoid floating-point issues
    c = min(c, 1.0)
    s = min(s, 1.0)

    theta = 2.0 * float(np.arctan2(s, c))
    eps = 1e-10

    # U3 fixes u00 to a real, non-negative value. Remove the input matrix's
    # global phase before extracting the relative phases from u01/u10/u11.
    # Without this step, for example, X-H-X is incorrectly mapped to
    # U3(π/2, 0, π) instead of the equivalent U3(π/2, π, 0).
    if abs(u00) >= eps:
        phase_correction = np.exp(-1j * np.angle(u00))
        u00 *= phase_correction
        u01 *= phase_correction
        u10 *= phase_correction
        u11 *= phase_correction

    if theta < eps:
        # θ ≈ 0: matrix is effectively diagonal
        # U ≈ [[1, 0], [0, e^{i(φ+λ)}]]
        phi = 0.0
        lam = float(np.angle(u11))
    elif abs(theta - np.pi) < eps:
        # θ ≈ π: matrix is off-diagonal
        # U ≈ [[0, -e^{iλ}], [e^{iφ}, 0]]
        phi = float(np.angle(u10))
        lam = float(np.angle(-u01))
    elif c >= s:
        # Mostly diagonal (θ ≤ π/2). arg(u10) and arg(-u01) carry an absolute
        # error of ~ulp/s, which for a small θ (e.g. 1e-8) is far above the
        # synthesis tolerance and does not cancel in φ + λ. Take φ + λ from
        # the well-conditioned u11 = c · e^{i(φ+λ)} and only φ − λ from the
        # off-diagonal entries (its error is scaled by s in the matrix).
        total = float(np.angle(u11))
        diff = float(np.angle(u10 * np.conj(-u01)))
        phi = 0.5 * (total + diff)
        lam = 0.5 * (total - diff)
        # Halving is ambiguous by π: pick the branch matching arg(u10).
        if np.real(np.exp(-1j * phi) * u10) < 0.0:
            phi += np.pi
            lam += np.pi
    else:
        # General case:
        # u10 = s · e^{iφ}  →  φ = arg(u10)
        # u01 = -s · e^{iλ}  →  λ = arg(-u01)
        phi = float(np.angle(u10))
        lam = float(np.angle(-u01))

    # Normalize to [0, 2π)
    phi = phi % (2.0 * np.pi)
    lam = lam % (2.0 * np.pi)

    return theta, phi, lam


class SingleQubitGateFusion(BasePass):
    """Fuse consecutive single-qubit gates into a single U3 gate.

    This pass is a generalization of :class:`RotationMerging`. While
    RotationMerging only merges same-axis rotations (e.g., RX + RX → RX),
    this pass handles **arbitrary** single-qubit gate sequences by:

    1. Computing the combined 2×2 unitary of each fusible block.
    2. Decomposing it into a single U3(θ, φ, λ) gate.

    Gates that are not single-qubit (e.g., CNOT, CZ, SWAP) act as
    "barriers" that terminate a fusible block.

    Note:
        Only gates with **concrete float parameters** are fused.
        Symbolic parameters (``Parameter`` instances) are left
        untouched to preserve variational circuit semantics.

    **Basis-aware mode.** When ``basis_gates`` is given, a run is no
    longer turned into ``u3`` unconditionally. Instead its unitary is
    resynthesized with :func:`quspect.gates.synthesis.synthesize_one_qubit`
    using only basis gates (e.g. ``rz·h·rz·h·rz`` for an ``{h, rz}``
    basis, with identity rotations dropped), and the run is replaced only
    if the replacement is strictly cheaper than the run. The cost of a
    run is its gate count, where a non-basis gate counts as the length of
    its decomposition into the basis. Runs equal to the identity (up to
    global phase) are removed. Single non-basis gates are resynthesized
    too when that is cheaper than decomposing them.

    Args:
        basis_gates: Optional target gate set. ``None`` (default) keeps
            the classic behaviour of fusing every run into one ``u3``.

    Example:
        >>> dag = circuit.to_dag()
        >>> fuser = SingleQubitGateFusion()
        >>> optimized = fuser.run(dag)
        >>> # H Z S X on qubit 0 → single U3 gate
        >>> fuser = SingleQubitGateFusion(basis_gates={"h", "rz", "cz"})
        >>> optimized = fuser.run(dag)
        >>> # Runs become at most rz·h·rz·h·rz, and only when shorter
    """

    reports_modifications = True

    def __init__(self, basis_gates: Collection[str] | None = None) -> None:
        super().__init__()
        self.basis_gates: frozenset[str] | None = (
            None if basis_gates is None else frozenset(basis_gates)
        )
        self._cost_cache: dict[tuple[str, tuple[float, ...]], int] = {}
        self._run_cache: dict[tuple[object, ...], list[Gate] | None] = {}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute single-qubit gate fusion on the DAG.

        Args:
            dag: The input DAG.

        Returns:
            The optimized DAG (modified in-place).
        """
        fuse_on_wire = (
            self._fuse_on_wire if self.basis_gates is None else self._resynth_on_wire
        )
        changed = True
        while changed:
            changed = False
            for wire_idx in range(dag.num_qubits):
                if fuse_on_wire(dag, f"q{wire_idx}"):
                    changed = True
        return dag

    # ------------------------------------------------------------------
    # Basis-aware resynthesis
    # ------------------------------------------------------------------

    def _gate_cost(self, gate: Gate) -> int:
        """Return the number of basis gates needed for ``gate``.

        Basis gates cost 1; other gates cost the length of their
        decomposition into the basis (or a large value if they cannot be
        decomposed).
        """
        basis = self.basis_gates
        assert basis is not None
        if gate.name in basis:
            return 1
        key = (gate.name, tuple(float(p) for p in gate.params))
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached
        from quspect.circuit.circuit import QuantumCircuit
        from quspect.ir.compilation import decompose_deep

        cost = _UNDECOMPOSABLE_COST
        try:
            circ = QuantumCircuit()
            circ.append(gate, (0,))
            out = decompose_deep(circ, supported_gates=set(basis))
            if all(
                i.gate is not None and i.gate.name in basis for i in out.instructions
            ):
                cost = len(out.instructions)
        except (ValueError, TypeError, RecursionError):
            cost = _UNDECOMPOSABLE_COST
        self._cost_cache[key] = cost
        return cost

    def _is_fusible_node(self, dag: DAGCircuit, nid: int) -> bool:
        """Return whether a node is a concrete single-qubit gate."""
        node = dag.get_node(nid)
        if node.instruction is None or node.instruction.gate is None:
            return False
        gate = node.instruction.gate
        return (
            gate.name in SINGLE_QUBIT_GATES
            and gate.num_qubits == 1
            and self._has_concrete_params(gate)
        )

    def _resynthesized_run(
        self, gates: list[Gate], basis: frozenset[str] | set[str]
    ) -> list[Gate] | None:
        """Shorter basis sequence for a run of single-qubit gates, or ``None``.

        ``None`` means the run is kept: a single basis gate, a run whose
        matrix is unavailable, or no strictly cheaper synthesis.
        """
        from quspect.gates.synthesis import synthesize_one_qubit

        cost = sum(self._gate_cost(g) for g in gates)
        if len(gates) == 1 and cost == 1:
            return None
        try:
            combined = np.eye(2, dtype=complex)
            for g in gates:
                combined = _gate_to_matrix(g) @ combined
        except (ValueError, TypeError):
            return None
        replacement = synthesize_one_qubit(combined, basis)
        if replacement is None or len(replacement) >= cost:
            return None
        return replacement

    def _resynth_on_wire(self, dag: DAGCircuit, wire: str) -> bool:
        """Resynthesize maximal single-qubit runs on one wire (basis mode).

        Args:
            dag: The DAG.
            wire: Wire identifier (e.g. ``"q0"``).

        Returns:
            True if at least one run was replaced.
        """
        from quspect.circuit.instruction import CircuitInstruction, InstructionType

        basis = self.basis_gates
        assert basis is not None
        nodes = dag.nodes_on_wire(wire)
        changed = False
        i = 0
        n = len(nodes)
        while i < n:
            if not self._is_fusible_node(dag, nodes[i]):
                i += 1
                continue
            j = i
            while (
                j + 1 < n
                and self._is_fusible_node(dag, nodes[j + 1])
                and dag.has_edge(nodes[j], nodes[j + 1], wire)
            ):
                j += 1
            block = nodes[i : j + 1]
            i = j + 1

            gates: list[Gate] = []
            for nid in block:
                instr = dag.get_node(nid).instruction
                assert instr is not None and instr.gate is not None
                gates.append(instr.gate)
            # The decision only depends on the gate sequence: memoize it (the
            # parameter reprs keep 0.0 and -0.0 apart, which Gate equality
            # does not).
            key = tuple((g, tuple(map(repr, g.params))) for g in gates)
            if key in self._run_cache:
                replacement = self._run_cache[key]
            else:
                replacement = self._resynthesized_run(gates, basis)
                self._run_cache[key] = replacement
            if replacement is None:
                continue

            # Reuse the first nodes of the run for the replacement gates
            # and splice out the remainder.
            for nid, new_gate in zip(block, replacement, strict=False):
                old_instr = dag.get_node(nid).instruction
                assert old_instr is not None
                dag.substitute_node_instruction(
                    nid,
                    CircuitInstruction(
                        type=InstructionType.GATE,
                        gate=new_gate,
                        qubits=old_instr.qubits,
                        clbits=old_instr.clbits,
                    ),
                )
            for nid in block[len(replacement) :]:
                dag.remove_node_and_reconnect(nid)
            changed = True
            logger.debug(
                "Resynthesized %d single-qubit gates on %s into %s",
                len(block),
                wire,
                [g.name for g in replacement],
            )
        return changed

    def _fuse_on_wire(self, dag: DAGCircuit, wire: str) -> bool:
        """Scan one wire and fuse consecutive single-qubit gate blocks.

        Args:
            dag: The DAG.
            wire: Wire identifier (e.g. ``"q0"``).

        Returns:
            True if at least one fusion was performed.
        """
        op_nodes = dag.nodes_on_wire(wire)
        gate_nodes: list[int] = []
        for nid in op_nodes:
            node = dag.get_node(nid)
            if (
                node.is_gate
                and node.instruction is not None
                and node.instruction.gate is not None
            ):
                gate_nodes.append(nid)

        if len(gate_nodes) < 2:
            return False

        fused = False
        i = 0
        while i < len(gate_nodes):
            # Try to build a fusible block starting at position i
            block_end = self._find_fusible_block(dag, gate_nodes, i, wire)
            if block_end is None or block_end <= i:
                i += 1
                continue

            # Extract the block
            block = gate_nodes[i : block_end + 1]
            if len(block) < 2:
                i += 1
                continue

            # Fuse the block
            if self._fuse_block(dag, block, wire):
                fused = True
                # Rebuild the gate_nodes list after the fusion
                op_nodes = dag.nodes_on_wire(wire)
                gate_nodes = []
                for nid in op_nodes:
                    node = dag.get_node(nid)
                    if (
                        node.is_gate
                        and node.instruction is not None
                        and node.instruction.gate is not None
                    ):
                        gate_nodes.append(nid)
                i = max(0, gate_nodes.index(block[0]) if block[0] in gate_nodes else 0)
            else:
                i += 1

        return fused

    def _find_fusible_block(
        self, dag: DAGCircuit, gate_nodes: list[int], start: int, wire: str
    ) -> int | None:
        """Find the end index of a fusible block of single-qubit gates.

        A fusible block is a consecutive sequence of single-qubit gates
        on the same wire, where no two-qubit or multi-qubit operation
        sits between them.

        Args:
            dag: The DAG.
            gate_nodes: Ordered list of gate node IDs on this wire.
            start: Starting index in ``gate_nodes``.
            wire: Wire identifier.

        Returns:
            The index of the last node in the fusible block, or None
            if no fusible block exists.
        """
        if start >= len(gate_nodes):
            return None

        first_nid = gate_nodes[start]
        first_node = dag.get_node(first_nid)
        if first_node.instruction is None or first_node.instruction.gate is None:
            return None

        first_gate = first_node.instruction.gate
        if first_gate.name not in SINGLE_QUBIT_GATES:
            return None

        # Check if the first gate has concrete params (not symbolic)
        if not self._has_concrete_params(first_gate):
            return None

        # Also check that it's truly a single-qubit gate (acts on 1 qubit)
        if first_gate.num_qubits != 1:
            return None

        end = start
        for j in range(start + 1, len(gate_nodes)):
            curr_nid = gate_nodes[j]
            curr_node = dag.get_node(curr_nid)
            if curr_node.instruction is None or curr_node.instruction.gate is None:
                break

            curr_gate = curr_node.instruction.gate

            # Check: must be single-qubit gate with concrete params
            if curr_gate.name not in SINGLE_QUBIT_GATES:
                break
            if curr_gate.num_qubits != 1:
                break
            if not self._has_concrete_params(curr_gate):
                break

            # Check: must be consecutive on this wire
            prev_nid = gate_nodes[j - 1]
            if not self._are_consecutive(dag, prev_nid, curr_nid, wire):
                break

            end = j

        return end if end > start else None

    def _has_concrete_params(self, gate: Gate) -> bool:
        """Check if all gate params are concrete floats/ints.

        Args:
            gate: A :class:`Gate` instance.

        Returns:
            True if all params are numeric.
        """
        if not gate.params:
            return True  # No params (standard gates) → fusible
        return all(isinstance(p, (int, float)) for p in gate.params)

    def _are_consecutive(
        self, dag: DAGCircuit, nid_a: int, nid_b: int, wire: str
    ) -> bool:
        """Check if two nodes are consecutive on a wire."""
        return dag.has_edge(nid_a, nid_b, wire)

    def _fuse_block(self, dag: DAGCircuit, block: list[int], wire: str) -> bool:
        """Fuse a block of single-qubit gates into a single U3 gate.

        Args:
            dag: The DAG to modify.
            block: Ordered list of node IDs in the fusible block.
            wire: Wire identifier.

        Returns:
            True on success.
        """
        if len(block) < 2:
            return False

        # Compute the combined matrix
        combined = np.eye(2, dtype=complex)
        for nid in block:
            node = dag.get_node(nid)
            if node.instruction is None or node.instruction.gate is None:
                return False
            try:
                m = _gate_to_matrix(node.instruction.gate)
                # Apply matrix from left: U_n · U_{n-1} · ... · U_1
                combined = m @ combined
            except (ValueError, TypeError):
                return False

        # Decompose into U3
        try:
            theta, phi, lam = _decompose_u3(combined)
        except ValueError:
            return False

        # Create the fused U3 gate
        from quspect.gates.parameterized import U3 as U3_TEMPLATE

        fused_gate = U3_TEMPLATE.with_params(theta, phi, lam)

        # Update the first node with the fused gate
        first_nid = block[0]
        first_node = dag.get_node(first_nid)
        if first_node.instruction is None:
            return False

        from quspect.circuit.instruction import CircuitInstruction, InstructionType

        dag.substitute_node_instruction(
            first_nid,
            CircuitInstruction(
                type=InstructionType.GATE,
                gate=fused_gate,
                qubits=first_node.instruction.qubits,
                clbits=first_node.instruction.clbits,
            ),
        )

        # Remove the remaining nodes in the block
        for nid in block[1:]:
            self._remove_node(dag, nid, wire)

        logger.debug(
            "Fused %d single-qubit gates on %s into U3(%.4f, %.4f, %.4f)",
            len(block),
            wire,
            theta,
            phi,
            lam,
        )
        return True

    def _remove_node(self, dag: DAGCircuit, nid: int, wire: str) -> None:
        """Remove a node from the DAG and reconnect the wire.

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
