"""Late resynthesis of SWAP-face two-qubit blocks.

Routing fuses SWAPs into the neighbouring two-qubit interactions; the
fused blocks sit on the SWAP face ``a = b = π/4`` of the Weyl chamber
(``SWAP·exp(-iθ/2·ZZ)`` up to single-qubit gates).  The generic 3-entangler
KAK template writes them with two general inner rotations, while
``SWAP·exp(-iθ/2·ZZ) = CX·CX'·Rz(θ)·CX`` needs a single inner ``rz``.

The pipeline keeps the generic template (its final CX cancellation
sometimes finds more pairs on it).  :func:`resynthesize_swap_face_blocks`
rewrites the SWAP-face blocks of the *final* circuit: every maximal block
of exactly three entanglers on one qubit pair (only single-qubit gates on
the pair in between) whose unitary is on the SWAP face is resynthesized
with :func:`~quspect.synthesis.two_qubit.synthesize_swap_face_block`.  The
entanglers stay three per block, on the same pairs; the outer single-qubit
layers are left for the single-qubit minimization to merge.  From level 2
on :func:`~quspect.transpiler.transpile` finishes three versions of the
selected routing — as is, rewritten here, and its post-routing pipeline
rerun under :func:`~quspect.synthesis.two_qubit.swap_face_templates` (if
that has no more two-qubit gates) — and keeps the best by (two-qubit
gates, depth, two-qubit depth, gates).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction

__all__ = ["resynthesize_swap_face_blocks"]

_I2 = np.eye(2, dtype=complex)


class _Block:
    __slots__ = ("indices", "num_entanglers", "pair", "pending")

    def __init__(self, pair: tuple[int, int], index: int) -> None:
        self.pair = pair
        self.indices = [index]  # instructions of the block (time order)
        self.num_entanglers = 1
        # single-qubit instructions after the last entangler (per wire)
        self.pending: dict[int, list[int]] = {pair[0]: [], pair[1]: []}


def _block_unitary(
    instrs: tuple[CircuitInstruction, ...], block: _Block
) -> np.ndarray | None:
    """Unitary of the block on ``block.pair`` (first qubit most significant)."""
    from quspect.gates._matrix_utils import gate_matrix

    q0, q1 = block.pair
    u = np.eye(4, dtype=complex)
    for idx in block.indices:
        ins = instrs[idx]
        assert ins.gate is not None
        try:
            m = np.asarray(gate_matrix(ins.gate), dtype=complex)
        except (KeyError, ValueError, TypeError):
            return None
        if len(ins.qubits) == 1:
            g = np.kron(m, _I2) if ins.qubits[0] == q0 else np.kron(_I2, m)
        elif ins.qubits == (q0, q1):
            g = m
        else:  # reversed operands: swap the tensor factors
            g = m.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
        u = g @ u
    return u


def resynthesize_swap_face_blocks(
    circuit: QuantumCircuit, native: frozenset[str]
) -> QuantumCircuit:
    """Rewrite the 3-entangler SWAP-face blocks with the SWAP-shaped template.

    Args:
        circuit: A circuit in the target's native gates.
        native: The target's native gate names (they choose the synthesis
            basis, see :meth:`~quspect.synthesis.two_qubit.TwoQubitBasis.from_gates`).

    Returns:
        An equivalent circuit (up to global phase) with the same entanglers
        on the same pairs (and in the same number), or ``circuit`` itself
        when no block was rewritten.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().cx(0, 1).rz(0.3, 1).cx(0, 1)
        >>> qc = qc.cx(0, 1).cx(1, 0).cx(0, 1)  # SWAP · ZZ(0.3)
        >>> native = frozenset({"cx", "h", "rz", "x"})
        >>> out = resynthesize_swap_face_blocks(qc, native)
        >>> sum(1 for i in out if i.gate.name == "cx")
        3
    """
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.synthesis.two_qubit import (
        TwoQubitBasis,
        ops_to_instructions,
        synthesize_swap_face_block,
    )

    basis = TwoQubitBasis.from_gates(native)
    if basis is None:
        return circuit
    instrs = circuit.instructions
    open_block: dict[int, _Block] = {}
    blocks: list[_Block] = []

    def close(q: int) -> None:
        b = open_block.pop(q, None)
        if b is None:
            return
        other = b.pair[1] if b.pair[0] == q else b.pair[0]
        open_block.pop(other, None)

    for idx, ins in enumerate(instrs):
        if not ins.is_gate or ins.gate is None:
            for q in ins.qubits:
                close(q)
            continue
        if len(ins.qubits) == 1:
            b = open_block.get(ins.qubits[0])
            if b is not None:
                b.pending[ins.qubits[0]].append(idx)
            continue
        if len(ins.qubits) != 2 or ins.gate.name != basis.entangler:
            for q in ins.qubits:
                close(q)
            continue
        a, c = ins.qubits
        b = open_block.get(a)
        if b is not None and b is open_block.get(c):
            for q in (a, c):
                b.indices += b.pending[q]
                b.pending[q] = []
            b.indices.append(idx)
            b.num_entanglers += 1
            continue
        close(a)
        close(c)
        nb = _Block((a, c), idx)
        open_block[a] = open_block[c] = nb
        blocks.append(nb)

    replace: dict[int, list[CircuitInstruction]] = {}
    drop: set[int] = set()
    for b in blocks:
        if b.num_entanglers != 3:
            continue
        u = _block_unitary(instrs, b)
        if u is None:
            continue
        ops = synthesize_swap_face_block(u, basis)
        if ops is None:
            continue
        new = ops_to_instructions(ops, b.pair)
        if any(i.gate is None or i.gate.name not in native for i in new):
            continue
        drop.update(b.indices)
        replace[b.indices[-1]] = new
    if not replace:
        return circuit
    out: list[CircuitInstruction] = []
    for idx, ins in enumerate(instrs):
        if idx in replace:
            out += replace[idx]
        elif idx not in drop:
            out.append(ins)
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)
