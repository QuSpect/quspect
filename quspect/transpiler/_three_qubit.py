"""Three-qubit block consolidation and resynthesis.

:class:`ThreeQubitBlockPass` collects maximal runs of gates confined to
three qubits, multiplies them into an ``8×8`` unitary and resynthesizes it
with a quantum Shannon decomposition (Shende–Bullock–Markov 2006,
quant-ph/0406176), keeping the replacement only when it needs fewer
two-qubit gates than the run (estimated after two-qubit block
resynthesis).

Two constructions are tried:

- **Controlled / multiplexed two-qubit unitary.**  If the block is
  block-diagonal with respect to one of its qubits ``c`` (``|0⟩⟨0| ⊗ A +
  |1⟩⟨1| ⊗ B``, e.g. a controlled-``U^(2^k)`` of phase estimation, whatever
  the number of repetitions), it is *demultiplexed*:
  ``A = V·D·W`` and ``B = V·D†·W`` with ``A B† = V D² V†`` (an eigen
  decomposition), so the block is ``W`` (two-qubit, ≤ 3 CZ), a
  multiplexed ``Rz`` on ``c`` selected by the other two qubits (4 CX, Gray
  code) and ``V`` (≤ 3 CZ): at most 10 two-qubit gates.
- **General block.**  A cosine–sine decomposition splits the block into two
  multiplexed two-qubit unitaries around a multiplexed ``Ry`` (4 CX); both
  multiplexors are demultiplexed as above, at most 24 two-qubit gates.

Every replacement is verified on the emitted gates: the product of the
emitted gate matrices must equal the block unitary up to global phase with
``max|U − e^{iφ}V| ≤ 1e-9`` (phase-aligned, element-wise), otherwise the
block is kept.  The emitted gates are ``cx``, ``ry``, ``rz`` and the
``{cz, h, rz, x}`` gates of the two-qubit synthesis; decompose afterwards.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.synthesis.two_qubit import synthesize_two_qubit_ops

if TYPE_CHECKING:
    from quspect.gates.gate import Gate

__all__ = ["ThreeQubitBlockPass", "ThreeQubitBlockStats", "synthesize_three_qubit"]

logger = logging.getLogger(__name__)

#: Verification budget of a replacement (phase-aligned, element-wise).
_VERIFY_ATOL = 1e-9
#: Off-diagonal norm below which a block counts as multiplexed.
_MUX_ATOL = 1e-10
#: Smallest estimated cost of a block worth trying (a demultiplexed block
#: needs up to 10 two-qubit gates).
_MIN_COST = 8

#: CZ cost of common gates; other two-qubit gates count 3, ``ccx`` 6.
_GATE_COST = {"cx": 1, "cz": 1, "cy": 1, "ch": 1, "swap": 3}
_TWO_CZ = frozenset({"cp", "cu1", "crz", "crx", "cry", "rzz", "rxx", "ryy", "rzx"})

# Gray-code sign matrix of a multiplexed rotation on the select qubits
# (s1, s2): rotation j acts with sign (-1)^(f_j · (s1, s2)), where f_j is
# the parity mask accumulated by the CX gates before it.
_GRAY = [(0, 0), (0, 1), (1, 1), (1, 0)]  # masks over (s1, s2)
_MUX_SIGNS = np.array(
    [
        [(-1) ** ((m1 & (k >> 1)) ^ (m2 & (k & 1))) for (m1, m2) in _GRAY]
        for k in range(4)
    ],
    dtype=float,
)
_MUX_INV = np.linalg.inv(_MUX_SIGNS)


@dataclass
class ThreeQubitBlockStats:
    """Counters of the last :meth:`ThreeQubitBlockPass.run`.

    Attributes:
        blocks_examined: Three-qubit blocks above the cost threshold.
        blocks_replaced: Blocks that were resynthesized.
        two_qubit_before: Estimated two-qubit cost of the replaced blocks.
        two_qubit_after: Two-qubit gates of their replacements.

    Example:
        >>> ThreeQubitBlockStats().blocks_replaced
        0
    """

    blocks_examined: int = 0
    blocks_replaced: int = 0
    two_qubit_before: int = 0
    two_qubit_after: int = 0


@dataclass
class _Block:
    qubits: list[int]
    members: list[int] = field(default_factory=list)


def _gate_cost(gate: Gate, num_qubits: int) -> int:
    if num_qubits == 3:
        return 6
    if num_qubits != 2:
        return 0
    if gate.name in _TWO_CZ:
        return 2
    return _GATE_COST.get(gate.name, 3)


def _estimate_cost(instrs: list[CircuitInstruction]) -> int:
    """Two-qubit cost of a gate run after two-qubit block resynthesis.

    Runs of two-qubit gates on one pair cost ``min(3, Σ cost)``; a
    three-qubit gate costs 6 and closes the runs on its qubits.
    """
    owner: dict[int, int] = {}
    runs: list[list[int]] = []  # [qa, qb, cost]
    for ins in instrs:
        qs = ins.qubits
        assert ins.gate is not None
        if len(qs) == 3:
            for q in qs:
                owner.pop(q, None)
            runs.append([-1, -1, 6])
        elif len(qs) == 2:
            a, b = qs
            r = owner.get(a)
            cost = _gate_cost(ins.gate, 2)
            if r is not None and owner.get(b) == r:
                runs[r][2] += cost
                continue
            for q in (a, b):
                old = owner.pop(q, None)
                if old is not None:
                    for p in runs[old][:2]:
                        if owner.get(p) == old:
                            del owner[p]
            owner[a] = owner[b] = len(runs)
            runs.append([a, b, cost])
    return sum(c if a < 0 else min(3, c) for a, _, c in runs)


def _apply(u: np.ndarray, gate_mat: np.ndarray, local: list[int]) -> np.ndarray:
    """Left-multiply the 8×8 ``u`` by a gate on local qubits (0 = high)."""
    k = len(local)
    psi = u.reshape([2, 2, 2, 8])
    g = gate_mat.reshape([2] * (2 * k))
    psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), local))
    psi = np.moveaxis(psi, list(range(k)), local)
    return psi.reshape(8, 8)


def _distance(u: np.ndarray, v: np.ndarray) -> float:
    overlap = complex(np.vdot(v, u))
    if abs(overlap) < 1e-300:
        return float("inf")
    phase = overlap / abs(overlap)
    return float(np.max(np.abs(u - phase * v)))


# (gate name, local qubits, params) — local qubits index the block order.
_Op = tuple[str, tuple[int, ...], tuple[float, ...]]


def _two_qubit_ops(u: np.ndarray, qa: int, qb: int) -> list[_Op]:
    ops = synthesize_two_qubit_ops(u)
    out: list[_Op] = []
    for name, local, params in ops:
        out.append((name, tuple((qa, qb)[i] for i in local), tuple(params)))
    return out


def _mux_ops(kind: str, angles: np.ndarray, c: int, s1: int, s2: int) -> list[_Op]:
    """Multiplexed ``kind`` (``"rz"``/``"ry"``) on ``c`` selected by (s1, s2).

    ``angles[2·s1 + s2]`` is the rotation angle applied for that selection.
    """
    alphas = _MUX_INV @ angles
    ops: list[_Op] = []
    for j, cx_ctrl in enumerate((s2, s1, s2, s1)):
        if abs(math.remainder(alphas[j], 4 * math.pi)) > 1e-15:
            ops.append((kind, (c,), (float(alphas[j]),)))
        ops.append(("cx", (cx_ctrl, c), ()))
    return ops


def _demultiplex(
    a: np.ndarray, b: np.ndarray, c: int, t1: int, t2: int
) -> list[_Op] | None:
    """Gates of ``|0⟩⟨0|_c ⊗ A + |1⟩⟨1|_c ⊗ B`` (A, B act on (t1, t2))."""
    import scipy.linalg  # type: ignore[import-untyped]

    m = a @ b.conj().T
    # m is unitary (normal): its complex Schur form is diagonal.
    t, v = scipy.linalg.schur(m, output="complex")
    if np.max(np.abs(t - np.diag(np.diag(t)))) > 1e-9:
        return None
    lam = np.diag(t)
    phases = np.angle(lam) / 2.0
    d = np.exp(1j * phases)
    w = np.diag(d) @ v.conj().T @ b
    # Rz(θ_k) = diag(e^{-iθ/2}, e^{iθ/2}) on c gives e^{±iφ_k} for θ_k = -2φ_k.
    ops = _two_qubit_ops(w, t1, t2)
    ops += _mux_ops("rz", -2.0 * phases, c, t1, t2)
    ops += _two_qubit_ops(v, t1, t2)
    return ops


def _block_unitary_in(order: list[int], instrs: list[CircuitInstruction]) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    pos = {q: i for i, q in enumerate(order)}
    u = np.eye(8, dtype=complex)
    for ins in instrs:
        assert ins.gate is not None
        mat = np.asarray(gate_matrix(ins.gate), dtype=complex)
        u = _apply(u, mat, [pos[q] for q in ins.qubits])
    return u


def _ops_unitary(ops: list[_Op]) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    u = np.eye(8, dtype=complex)
    for ins in _ops_to_instructions(ops, [0, 1, 2]):
        assert ins.gate is not None
        mat = np.asarray(gate_matrix(ins.gate), dtype=complex)
        u = _apply(u, mat, list(ins.qubits))
    return u


def _ops_to_instructions(ops: list[_Op], qubits: list[int]) -> list[CircuitInstruction]:
    from quspect.gates.parameterized import RY, RZ
    from quspect.gates.standard import CX, CZ, H, X

    fixed = {"cx": CX, "cz": CZ, "h": H, "x": X}
    out: list[CircuitInstruction] = []
    for name, local, params in ops:
        if name == "ry":
            gate = RY.with_params(*params)
        elif name == "rz":
            gate = RZ.with_params(*params)
        else:
            gate = fixed[name]
        out.append(
            CircuitInstruction(
                type=InstructionType.GATE,
                gate=gate,
                qubits=tuple(qubits[i] for i in local),
            )
        )
    return out


def synthesize_three_qubit(u: np.ndarray) -> list[_Op] | None:
    """Shannon-decompose an ``8×8`` unitary (local qubit 0 = high bit).

    Tries the demultiplexing of a block-diagonal unitary for each choice of
    the select qubit, then the general cosine–sine construction, and
    returns the cheapest verified gate list (``None`` if none verifies).

    Args:
        u: The unitary, local qubits ``(0, 1, 2)`` big-endian.

    Returns:
        ``(name, local qubits, params)`` tuples with names ``cx``, ``ry``,
        ``rz``, ``cz``, ``h``, ``x``, or ``None``.

    Example:
        >>> import numpy as np
        >>> ccz = np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex)
        >>> ops = synthesize_three_qubit(ccz)
        >>> sum(1 for name, q, _ in ops if len(q) == 2) <= 10
        True
    """
    import scipy.linalg

    best: list[_Op] | None = None
    best_cost = 1 << 30
    for c in range(3):
        t1, t2 = (q for q in range(3) if q != c)
        perm = _local_perm(c, t1, t2)
        up = perm @ u @ perm.T
        off = max(np.max(np.abs(up[:4, 4:])), np.max(np.abs(up[4:, :4])))
        if off > _MUX_ATOL:
            continue
        ops = _demultiplex(up[:4, :4], up[4:, 4:], c, t1, t2)
        if ops is None:
            continue
        cost = sum(1 for _, q, _ in ops if len(q) == 2)
        if cost < best_cost and _distance(_ops_unitary(ops), u) <= _VERIFY_ATOL:
            best, best_cost = ops, cost
    if best is not None:
        return best
    # General block: cosine–sine decomposition around qubit 0.
    (u1, u2), theta, (v1h, v2h) = scipy.linalg.cossin(u, p=4, q=4, separate=True)
    right = _demultiplex(v1h, v2h, 0, 1, 2)
    left = _demultiplex(u1, u2, 0, 1, 2)
    if right is None or left is None:
        return None
    ops = right + _mux_ops("ry", 2.0 * np.asarray(theta), 0, 1, 2) + left
    if _distance(_ops_unitary(ops), u) <= _VERIFY_ATOL:
        return ops
    return None


_Split = tuple[
    list[CircuitInstruction], list[CircuitInstruction], list[CircuitInstruction]
]


def _peelings(members: list[CircuitInstruction], order: list[int]) -> list[_Split]:
    """Ways to split a block into ``head + core + tail``.

    The whole block, and for each qubit ``c`` the split where ``head`` /
    ``tail`` are the single-qubit gates on ``c`` before its first / after
    its last multi-qubit gate (e.g. the Hadamard on a phase-estimation
    control): they commute with everything in between on the other qubits,
    so the block equals ``head · core · tail`` and ``core`` may be a
    multiplexor in ``c`` even though the block is not.
    """
    out: list[_Split] = [([], members, [])]
    for c in order:
        multi = [
            k
            for k, ins in enumerate(members)
            if len(ins.qubits) > 1 and c in ins.qubits
        ]
        if not multi:
            continue
        first, last = multi[0], multi[-1]
        on_c = [k for k, ins in enumerate(members) if ins.qubits == (c,)]
        head = [k for k in on_c if k < first]
        tail = [k for k in on_c if k > last]
        if not head and not tail:
            continue
        drop = set(head) | set(tail)
        out.append(
            (
                [members[k] for k in head],
                [ins for k, ins in enumerate(members) if k not in drop],
                [members[k] for k in tail],
            )
        )
    return out


def _local_perm(c: int, t1: int, t2: int) -> np.ndarray:
    """Permutation matrix ``P`` with ``P|x0 x1 x2⟩ = |x_c x_t1 x_t2⟩``."""
    p = np.zeros((8, 8))
    for idx in range(8):
        bits = [(idx >> (2 - i)) & 1 for i in range(3)]
        new = (bits[c] << 2) | (bits[t1] << 1) | bits[t2]
        p[new, idx] = 1.0
    return p


class ThreeQubitBlockPass:
    """Resynthesize three-qubit blocks with a quantum Shannon decomposition.

    The pass sweeps the instruction list once, growing blocks of gates that
    act within one set of at most three qubits (single-qubit gates on a free
    qubit open a one-qubit block; blocks on disjoint qubits are merged when
    a gate joins them and the union has at most three qubits).  A gate that
    does not fit closes the blocks it touches.  Measurements, barriers,
    noise channels, symbolic parameters and gates on more than three qubits
    are boundaries.

    Each closed three-qubit block whose estimated cost (two-qubit gates
    after two-qubit block resynthesis, ``ccx`` = 6) is at least 8 is
    resynthesized with :func:`synthesize_three_qubit`; the replacement is
    kept only if it has strictly fewer two-qubit gates.  Replaced blocks are
    emitted at the position of their last gate.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit()
        >>> for _ in range(4):
        ...     _ = qc.ccx(0, 1, 2).cx(0, 1).t(2)
        >>> out = ThreeQubitBlockPass().run(qc)
        >>> sum(1 for i in out if len(i.qubits) >= 2) < 4 * 7
        True
    """

    def __init__(self) -> None:
        self.stats = ThreeQubitBlockStats()

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of ``circuit`` with three-qubit blocks resynthesized.

        Args:
            circuit: Circuit to optimize (not modified).

        Returns:
            A new circuit, equal to ``circuit`` up to global phase.

        Example:
            >>> from quspect import QuantumCircuit
            >>> qc = QuantumCircuit().h(0).cx(0, 1)
            >>> len(ThreeQubitBlockPass().run(qc)) == 2
            True
        """
        self.stats = ThreeQubitBlockStats()
        instrs = list(circuit.instructions)
        slots: list[list[CircuitInstruction] | CircuitInstruction | None] = list(instrs)
        owner: dict[int, _Block] = {}

        def close(block: _Block) -> None:
            for q in block.qubits:
                if owner.get(q) is block:
                    del owner[q]
            if len(block.qubits) == 3:
                self._finalize(block, instrs, slots)

        for idx, ins in enumerate(instrs):
            gate = ins.gate
            qs = ins.qubits
            usable = (
                ins.is_gate
                and gate is not None
                and 1 <= len(qs) <= 3
                and all(isinstance(p, Real) for p in gate.params)
            )
            if not usable:
                targets = qs if qs else list(owner)
                for blk in {id(b): b for q in targets if (b := owner.get(q))}.values():
                    close(blk)
                continue
            touched = {id(b): b for q in qs if (b := owner.get(q)) is not None}
            union = set(qs)
            for b in touched.values():
                union.update(b.qubits)
            if len(union) <= 3:
                if len(touched) == 1:
                    blk = next(iter(touched.values()))
                    for q in qs:
                        if q not in blk.qubits:
                            blk.qubits.append(q)
                            owner[q] = blk
                else:
                    merged = _Block(qubits=[], members=[])
                    for b in touched.values():
                        merged.qubits.extend(b.qubits)
                        merged.members.extend(b.members)
                    merged.members.sort()
                    for q in qs:
                        if q not in merged.qubits:
                            merged.qubits.append(q)
                    blk = merged
                    for q in blk.qubits:
                        owner[q] = blk
                blk.members.append(idx)
                continue
            for b in touched.values():
                close(b)
            blk = _Block(qubits=list(qs), members=[idx])
            for q in qs:
                owner[q] = blk

        for blk in {id(b): b for b in owner.values()}.values():
            close(blk)

        flat: list[CircuitInstruction] = []
        for slot in slots:
            if slot is None:
                continue
            if isinstance(slot, list):
                flat.extend(slot)
            else:
                flat.append(slot)
        if self.stats.blocks_replaced:
            logger.debug(
                "ThreeQubitBlockPass: %d/%d blocks replaced, 2q %d -> %d",
                self.stats.blocks_replaced,
                self.stats.blocks_examined,
                self.stats.two_qubit_before,
                self.stats.two_qubit_after,
            )
        return QuantumCircuit._from_instruction_snapshot(flat, name=circuit.name)

    def _finalize(
        self,
        block: _Block,
        instrs: list[CircuitInstruction],
        slots: list[list[CircuitInstruction] | CircuitInstruction | None],
    ) -> None:
        members = [instrs[i] for i in block.members]
        if not any(len(i.qubits) >= 2 for i in members):
            return
        cost = _estimate_cost(members)
        if cost < _MIN_COST:
            return
        self.stats.blocks_examined += 1
        order = sorted(block.qubits)
        best: list[CircuitInstruction] | None = None
        best_cost = cost
        try:
            for head, core, tail in _peelings(members, order):
                ops = synthesize_three_qubit(_block_unitary_in(order, core))
                if ops is None:
                    continue
                new_cost = sum(1 for _, q, _ in ops if len(q) == 2)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = head + _ops_to_instructions(ops, order) + tail
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("Three-qubit resynthesis failed, keeping block: %s", exc)
            return
        if best is None:
            return
        new_cost = best_cost
        for i in block.members:
            slots[i] = None
        slots[block.members[-1]] = best
        self.stats.blocks_replaced += 1
        self.stats.two_qubit_before += cost
        self.stats.two_qubit_after += new_cost
