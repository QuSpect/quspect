"""Depth-oriented rescheduling of commuting gates (trace-monoid list scheduling).

Two gates *commute on a qubit* when both are rotations about the same Pauli
axis there — diagonal gates (``rz``, ``s``, ``t``, ``cz``, the control of a
``cx``…) about Z, ``x``/``sx``/``rx`` and the target of a ``cx`` about X.  An
operator that commutes with ``P`` on qubit ``q`` lies in the algebra spanned
by ``{I, P}`` on ``q`` (tensored with anything elsewhere), so two gates whose
axes agree on *every* qubit they share commute exactly.  Reordering a gate
sequence while keeping the relative order of every non-commuting pair gives
the same unitary: the circuit is a word in a *trace monoid* (Cartier–Foata,
Mazurkiewicz) and every linear extension of its dependence order is an
equivalent circuit.

On each qubit the gates therefore fall into *commutation groups* — maximal
runs of consecutive gates with the same axis (a gate with no axis is a group
of its own).  Members of a group may run in any order, each group starts
after the previous one on that qubit has finished, and two gates sharing a
qubit never run in the same layer.  Minimizing the depth over all equivalent
orders is a resource-constrained project scheduling problem (RCPSP) with
unit-time jobs and dedicated resources (the qubits); for one group of
mutually commuting two-qubit gates it reduces to edge-colouring their
interaction graph (Vizing: Δ or Δ + 1 layers).

:func:`schedule_for_depth` solves it heuristically with the classic RCPSP
machinery:

- a *serial schedule-generation scheme* with insertion: gates are taken by
  priority among those whose predecessors are scheduled, and each is put in
  the earliest layer at or after its predecessors where all its qubits are
  free (gaps left earlier can be filled later);
- critical-path priorities (Hu's bottom levels, extended by a per-qubit
  resource bound for the groups);
- *double justification* (Valls, Ballestín & Quintanilla 2005): the schedule
  is shifted right (scheduling the reversed circuit by latest finish
  time) and then left again (by earliest start time), which never lengthens
  it and usually closes gaps.

Total depth and two-qubit depth are different objectives (a long
single-qubit run on one qubit may be better placed where it delays a
two-qubit gate).  Three schedules are built — one for total depth, one that
keeps the per-qubit order of the two-qubit gates (so the two-qubit depth is
unchanged, since it only depends on that order) and one that first
minimizes the two-qubit depth (single-qubit gates taking no time) and then
the total depth with that two-qubit order fixed.  The result is the one with
the smallest depth among those whose two-qubit depth does not exceed the
input's.  Only the order of the instructions changes: the gates, their
count and the unitary are exactly those of the input.

Commutation is also used one level up: a maximal two-qubit block (the gates
on one pair with the single-qubit gates between them) gets, on each of its
qubits, the Pauli axis its 4×4 unitary commutes with, and the blocks are
scheduled like gates.  ``cx·rz·cx`` (a ZZ rotation) is diagonal, so a chain
of ZZ rotations — sequential gate by gate, because CX targets do not
commute with CX controls — becomes two layers of blocks (an edge-colouring
of the chain).
"""

from __future__ import annotations

import heapq
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate

__all__ = ["schedule_for_depth"]

#: Axis codes (``-1``: commutes with nothing that is not on disjoint qubits).
_NONE, _Z, _X, _Y = -1, 0, 1, 2
_AXES_1Q = {
    **dict.fromkeys(("rz", "z", "s", "sdg", "t", "tdg", "p", "u1", "id"), _Z),
    **dict.fromkeys(("x", "sx", "sxdg", "rx"), _X),
    **dict.fromkeys(("y", "ry"), _Y),
}
_AXES_MQ: dict[str, tuple[int, ...]] = {
    **dict.fromkeys(("cz", "cp", "crz", "rzz", "cu1"), (_Z, _Z)),
    "cx": (_Z, _X),
    "cy": (_Z, _Y),
    "rxx": (_X, _X),
    "ryy": (_Y, _Y),
    "ccx": (_Z, _Z, _X),
    "ccz": (_Z, _Z, _Z),
}
_PAULI = (
    np.array([[1, 0], [0, -1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
)
#: Double-justification rounds per schedule.
_ROUNDS = 2


def _matrix_axes(gate: Gate, num_qubits: int) -> tuple[int, ...]:
    """Commuting Pauli axis per qubit of a gate, from its matrix."""
    from quspect.gates._matrix_utils import gate_matrix

    try:
        m = np.asarray(gate_matrix(gate), dtype=complex)
    except (KeyError, ValueError, TypeError):
        return (_NONE,) * num_qubits
    dim = 2**num_qubits
    if m.shape != (dim, dim):
        return (_NONE,) * num_qubits
    out = []
    for slot in range(num_qubits):
        found = _NONE
        for code, p in enumerate(_PAULI):
            # qubit 0 of the gate is the most significant factor
            op = np.kron(
                np.kron(np.eye(2**slot), p), np.eye(2 ** (num_qubits - slot - 1))
            )
            if np.max(np.abs(op @ m - m @ op)) < 1e-12:
                found = code
                break
        out.append(found)
    return tuple(out)


def _axes(
    ins: CircuitInstruction, cache: dict[Gate, tuple[int, ...]]
) -> tuple[int, ...]:
    """Commuting axis per qubit of an instruction (``_NONE`` if not a gate)."""
    gate = ins.gate
    n = len(ins.qubits)
    if not ins.is_gate or gate is None:
        return (_NONE,) * n
    if n == 1 and gate.name in _AXES_1Q:
        return (_AXES_1Q[gate.name],)
    if n > 1 and gate.name in _AXES_MQ and len(_AXES_MQ[gate.name]) == n:
        return _AXES_MQ[gate.name]
    try:
        return cache[gate]
    except KeyError:
        pass
    except TypeError:  # unhashable gate
        return _matrix_axes(gate, n)
    ax = _matrix_axes(gate, n)
    cache[gate] = ax
    return ax


class _Problem:
    """Precedence structure of the commutation groups of a gate sequence.

    Built from a sequence ``seq`` of node ids; ``chain`` additionally keeps
    the order of the two-qubit nodes on every resource (within a group).
    Every valid reordering of ``seq`` has the same groups (and chains).
    """

    def __init__(
        self,
        seq: Sequence[int],
        res: Sequence[Sequence[int]],
        axis: Sequence[Sequence[int]],
        two: Sequence[bool],
        chain: bool,
    ) -> None:
        n = len(res)
        self.seq = list(seq)
        self.res = res
        self.group: list[list[int]] = [[0] * len(r) for r in res]
        self.members: list[list[int]] = []
        self.nxt: list[int] = []
        self.chain_pred: list[list[int]] = [[-1] * len(r) for r in res]
        self.chain_succ: list[list[int]] = [[] for _ in range(n)]
        last: dict[int, tuple[int, int]] = {}  # resource -> (group, axis)
        last_two: dict[int, int] = {}
        for g in self.seq:
            for k, q in enumerate(res[g]):
                a = axis[g][k]
                cur = last.get(q)
                if cur is not None and a != _NONE and cur[1] == a:
                    gid = cur[0]
                    self.members[gid].append(g)
                else:
                    gid = len(self.members)
                    self.members.append([g])
                    self.nxt.append(-1)
                    if cur is not None:
                        self.nxt[cur[0]] = gid
                    last[q] = (gid, a)
                self.group[g][k] = gid
                if chain and two[g]:
                    p = last_two.get(q, -1)
                    if p >= 0 and self.group[p][res[p].index(q)] == gid:
                        self.chain_pred[g][k] = p
                        self.chain_succ[p].append(g)
                    last_two[q] = g
        self.prev = [-1] * len(self.members)
        for gid, nx in enumerate(self.nxt):
            if nx >= 0:
                self.prev[nx] = gid

    def bottom_levels(self, dur: Sequence[int]) -> list[int]:
        """Critical-path priority of every node (longest remaining time).

        A group's tail is at least its members' bottom levels and at least
        the members' total duration plus the next group's tail (they share
        the resource).
        """
        res = self.res
        tail = [0] * len(self.members)
        bl = [0] * len(res)
        for g in reversed(self.seq):
            b = 0
            for k in range(len(res[g])):
                nx = self.nxt[self.group[g][k]]
                if nx >= 0 and tail[nx] > b:
                    b = tail[nx]
            for s in self.chain_succ[g]:
                if bl[s] > b:
                    b = bl[s]
            bl[g] = dur[g] + b
            for k in range(len(res[g])):
                gid = self.group[g][k]
                mem = self.members[gid]
                if mem[0] == g:  # the group's last node in reverse order
                    nx = self.nxt[gid]
                    t = sum(dur[h] for h in mem) + (tail[nx] if nx >= 0 else 0)
                    tail[gid] = max(t, max(bl[h] for h in mem))
        return bl

    def sgs(self, prio: Sequence[int], dur: Sequence[int]) -> list[int]:
        """Serial schedule generation with insertion; returns the node order.

        Among the nodes whose predecessors are scheduled, the one with the
        highest ``prio`` (lowest id on ties) is put at the earliest time at
        or after its predecessors' finish where all its resources are free
        (zero-duration nodes occupy nothing).  The order is by start time,
        zero-duration nodes first, then scheduling order.
        """
        res = self.res
        n = len(res)
        remaining = [len(m) for m in self.members]
        gfin = [0] * len(self.members)
        pending = [0] * n
        for g in self.seq:
            c = 0
            for k in range(len(res[g])):
                if self.prev[self.group[g][k]] >= 0:
                    c += 1
                if self.chain_pred[g][k] >= 0:
                    c += 1
            pending[g] = c
        heap = [(-prio[g], g) for g in self.seq if pending[g] == 0]
        heapq.heapify(heap)
        busy: dict[int, set[int]] = {}
        start = [0] * n
        fin = [0] * n
        pos = [0] * n
        count = 0
        while heap:
            _, g = heapq.heappop(heap)
            est = 0
            for k in range(len(res[g])):
                p = self.prev[self.group[g][k]]
                if p >= 0 and gfin[p] > est:
                    est = gfin[p]
                c = self.chain_pred[g][k]
                if c >= 0 and fin[c] > est:
                    est = fin[c]
            t = est
            if dur[g]:
                sets = [busy.setdefault(q, set()) for q in res[g]]
                while any(t in s for s in sets):
                    t += 1
                for s in sets:
                    s.add(t)
            start[g] = t
            fin[g] = t + dur[g]
            pos[g] = count
            count += 1
            for k in range(len(res[g])):
                gid = self.group[g][k]
                if fin[g] > gfin[gid]:
                    gfin[gid] = fin[g]
                remaining[gid] -= 1
                if remaining[gid] == 0:
                    nx = self.nxt[gid]
                    if nx >= 0:
                        for h in self.members[nx]:
                            pending[h] -= 1
                            if pending[h] == 0:
                                heapq.heappush(heap, (-prio[h], h))
            for succ in self.chain_succ[g]:
                pending[succ] -= 1
                if pending[succ] == 0:
                    heapq.heappush(heap, (-prio[succ], succ))
        if count != n:  # pragma: no cover - the structure is acyclic
            raise RuntimeError("schedule_for_depth: cyclic precedence")
        return sorted(self.seq, key=lambda g: (start[g], dur[g], pos[g]))


def _asap(
    order: Sequence[int], res: Sequence[Sequence[int]], dur: Sequence[int]
) -> tuple[list[int], list[int]]:
    """ASAP start / finish times of a node order (resources used in turn)."""
    avail: dict[int, int] = {}
    n = len(res)
    start = [0] * n
    fin = [0] * n
    for g in order:
        t = max((avail.get(q, 0) for q in res[g]), default=0)
        start[g] = t
        fin[g] = t + dur[g]
        for q in res[g]:
            avail[q] = fin[g]
    return start, fin


def _metric(
    order: Sequence[int], res: Sequence[Sequence[int]], weight: Sequence[int]
) -> int:
    """Depth of an order counting only nodes with ``weight`` 1 (others skipped)."""
    level: dict[int, int] = {}
    deepest = 0
    for g in order:
        if not weight[g]:
            continue
        lv = 1 + max((level.get(q, 0) for q in res[g]), default=0)
        for q in res[g]:
            level[q] = lv
        if lv > deepest:
            deepest = lv
    return deepest


def _justify(
    fwd: _Problem,
    bwd: _Problem,
    order: list[int],
    dur: Sequence[int],
    score: Callable[[list[int]], tuple[int, int]],
) -> list[int]:
    """Double justification rounds; keeps the best order by ``score``."""
    res = fwd.res
    best, best_score = order, score(order)
    cur = order
    for _ in range(_ROUNDS):
        _, fin = _asap(cur, res, dur)
        back = bwd.sgs(fin, dur)  # latest finish first, in reversed time
        cur = back[::-1]
        start, _ = _asap(cur, res, dur)
        cur = fwd.sgs([-s for s in start], dur)
        sc = score(cur)
        if sc < best_score:
            best, best_score = cur, sc
        else:
            break
    return best


def _best_order(
    res: Sequence[Sequence[int]],
    axis: Sequence[Sequence[int]],
    two: Sequence[bool],
    counted: Sequence[bool],
) -> list[int]:
    """Reference implementation of the scheduler (node ids in input order)."""
    n = len(res)
    seq = list(range(n))
    dur_all = [1 if c else 0 for c in counted]
    dur_two = [1 if t else 0 for t in two]

    def depth_score(order: list[int]) -> tuple[int, int]:
        return _metric(order, res, dur_all), _metric(order, res, dur_two)

    def two_score(order: list[int]) -> tuple[int, int]:
        d, d2 = depth_score(order)
        return d2, d

    def total(start_seq: list[int], chain: bool) -> list[int]:
        fwd = _Problem(start_seq, res, axis, two, chain)
        bwd = _Problem(start_seq[::-1], res, axis, two, chain)
        first = fwd.sgs(fwd.bottom_levels(dur_all), dur_all)
        return _justify(fwd, bwd, first, dur_all, depth_score)

    d0, d2q0 = depth_score(seq)
    best, best_score = seq, (d0, d2q0)
    cands = [total(seq, False), total(seq, True)]
    free = _Problem(seq, res, axis, two, False)
    free_r = _Problem(seq[::-1], res, axis, two, False)
    by_two = free.sgs(free.bottom_levels(dur_two), dur_two)
    by_two = _justify(free, free_r, by_two, dur_two, two_score)
    cands.append(total(by_two, True))
    for cand in cands:
        sc = depth_score(cand)
        if sc[1] <= d2q0 and sc < best_score:
            best, best_score = cand, sc
    return best


def schedule_for_depth(
    circuit: QuantumCircuit, *, blocks: bool = True
) -> QuantumCircuit:
    """Reorder commuting gates to minimize the depth of a circuit.

    Gates that act on disjoint qubits, or that are rotations about the same
    Pauli axis on every qubit they share (diagonal gates and CX controls
    about Z; ``x``, ``sx``, ``rx`` and CX targets about X), may be swapped;
    every other pair keeps its order (measurements, barriers and channels
    commute with nothing on their qubits and classical bits).  Among the
    equivalent orders a list scheduler with double justification (see the
    module docstring) picks one with the smallest depth whose two-qubit
    depth is not larger than the input's.

    With ``blocks`` the same scheduler also runs one level up, on the
    maximal two-qubit blocks (the gates on one pair with the single-qubit
    gates between them): a block whose unitary commutes with a Pauli on one
    of its qubits gets that axis there, so e.g. ``cx·rz·cx`` (a ZZ
    rotation) commutes with every diagonal gate or block on either qubit
    although its CX gates do not.  The block order is then refined on the
    gate level, and the better of the two results is kept.

    The gates themselves are unchanged, so the result is exactly equivalent
    and has the same gate counts; if no order is better, ``circuit`` is
    returned as is.

    Args:
        circuit: Any circuit.
        blocks: Also reorder whole two-qubit blocks (see above).

    Returns:
        A circuit with the same instructions in a possibly different order,
        with ``(depth, two-qubit depth)`` lexicographically no larger.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().cx(1, 2).cx(0, 1).rz(0.3, 0).cx(0, 3)
        >>> [(i.gate.name, i.qubits) for i in schedule_for_depth(qc)]
        [('cx', (1, 2)), ('cx', (0, 3)), ('cx', (0, 1)), ('rz', (0,))]

        ``cx(0, 3)`` commutes with ``cx(0, 1)`` and ``rz(0)`` (all diagonal
        on qubit 0) and now runs next to ``cx(1, 2)``: depth 4 → 3.  On the
        block level, two ZZ rotations sharing a qubit commute:

        >>> zz = QuantumCircuit().cx(0, 1).rz(0.3, 1).cx(0, 1)
        >>> zz.cx(1, 2).rz(0.2, 2).cx(1, 2).cx(3, 2).rz(0.1, 2).cx(3, 2)
        QuantumCircuit(, num_qubits=4, num_clbits=0, instructions=9)
        >>> [i.qubits for i in schedule_for_depth(zz) if len(i.qubits) == 2]
        [(0, 1), (0, 1), (3, 2), (3, 2), (1, 2), (1, 2)]

        ``ZZ(3, 2)`` now runs next to ``ZZ(0, 1)``: depth 9 → 6.
    """
    instrs = circuit.instructions
    if len(instrs) < 3:
        return circuit
    info = _Info(instrs)
    identity = list(range(len(instrs)))
    best, base, best_score = _order(instrs, info)
    if blocks:
        grouped = _block_order(instrs, info)
        if grouped is not None:
            cand, _, score = _order(instrs, info, grouped)
            if score[1] <= base[1] and score < best_score:
                best, best_score = cand, score
    if best == identity or not (best_score[1] <= base[1] and best_score < base):
        return circuit
    from quspect.circuit.circuit import QuantumCircuit

    return QuantumCircuit._from_instruction_snapshot(
        [instrs[g] for g in best], name=circuit.name
    )


def _instruction_resources(
    ins: CircuitInstruction, width: int, cache: dict[Gate, tuple[int, ...]]
) -> tuple[list[int], list[int]]:
    """Resources (qubits, then ``width + clbit``) and their axes."""
    if ins.is_barrier and not ins.qubits:
        qubits: tuple[int, ...] = tuple(range(width))
        ax: tuple[int, ...] = (_NONE,) * width
    else:
        qubits = ins.qubits
        ax = _axes(ins, cache)
    res: list[int] = []
    axes: list[int] = []
    for q, x in zip(qubits, ax, strict=True):
        if q not in res:
            res.append(q)
            axes.append(x)
    for c in ins.clbits:
        if width + c not in res:
            res.append(width + c)
            axes.append(_NONE)
    return res, axes


class _Info:
    """Scheduling nodes of a circuit's instructions (CSR, in circuit order)."""

    def __init__(self, instrs: Sequence[CircuitInstruction]) -> None:
        self.width = max((q for ins in instrs for q in ins.qubits), default=-1) + 1
        cache: dict[Gate, tuple[int, ...]] = {}
        self.offsets = [0]
        self.res: list[int] = []
        self.axes: list[int] = []
        for ins in instrs:
            res, axes = _instruction_resources(ins, self.width, cache)
            self.res += res
            self.axes += axes
            self.offsets.append(len(self.res))
        self.two = [ins.is_gate and len(ins.qubits) == 2 for ins in instrs]
        self.counted = [not ins.is_barrier and bool(ins.qubits) for ins in instrs]

    def node(self, i: int) -> tuple[list[int], list[int]]:
        """Resources and axes of instruction ``i``."""
        b, e = self.offsets[i], self.offsets[i + 1]
        return self.res[b:e], self.axes[b:e]


def _order(
    instrs: Sequence[CircuitInstruction],
    info: _Info,
    start: list[int] | None = None,
) -> tuple[list[int], tuple[int, int], tuple[int, int]]:
    """Best instruction order from ``start`` (default: circuit order).

    Returns the order (instruction indices) and the ``(depth, two-qubit
    depth)`` of ``start`` and of the result.  Uses the native core when
    available (same result).
    """
    core = _native_core()
    if core is not None:
        order, before, after = core._depth_schedule(
            info.offsets,
            info.res,
            info.axes,
            [int(t) for t in info.two],
            [int(c) for c in info.counted],
            info.width,
            start,
        )
        return list(order), (before[0], before[1]), (after[0], after[1])
    seq = list(range(len(instrs))) if start is None else start
    nodes = [info.node(i) for i in seq]
    local = _best_order(
        [r for r, _ in nodes],
        [a for _, a in nodes],
        [info.two[i] for i in seq],
        [info.counted[i] for i in seq],
    )
    result = [seq[g] for g in local]
    return (
        result,
        _depths([instrs[i] for i in seq]),
        _depths([instrs[i] for i in result]),
    )


def _depths(instrs: Sequence[CircuitInstruction]) -> tuple[int, int]:
    """``(depth, two-qubit depth)`` as reported by :func:`transpile`."""
    level: dict[int, int] = {}
    level2: dict[int, int] = {}
    deepest = deepest2 = 0
    for ins in instrs:
        qubits = ins.qubits
        if ins.is_barrier or not qubits:
            continue
        lv = 1 + max(level.get(q, 0) for q in qubits)
        for q in qubits:
            level[q] = lv
        if lv > deepest:
            deepest = lv
        if ins.is_gate and len(qubits) == 2:
            a, b = qubits
            lv2 = 1 + max(level2.get(a, 0), level2.get(b, 0))
            level2[a] = level2[b] = lv2
            if lv2 > deepest2:
                deepest2 = lv2
    return deepest, deepest2


def _blocks(instrs: Sequence[CircuitInstruction]) -> list[list[int]]:
    """Partition into maximal two-qubit blocks and single instructions.

    A block is a run of two-qubit gates on one pair together with the
    single-qubit gates on that pair between them.  Single-qubit gates not
    inside a block, and every other instruction, are nodes of their own.
    The nodes are listed in a valid order of the circuit and each keeps its
    instructions in circuit order.
    """
    nodes: list[list[int]] = []
    pair: list[tuple[int, int] | None] = []
    open_block: dict[int, int] = {}
    pending: dict[int, list[int]] = {}
    all_qubits: set[int] = set()

    def close(q: int) -> None:
        n = open_block.pop(q, None)
        if n is not None:
            p = pair[n]
            assert p is not None
            other = p[1] if p[0] == q else p[0]
            if open_block.get(other) == n:
                del open_block[other]

    def flush(q: int) -> None:
        for i in pending.pop(q, []):
            nodes.append([i])
            pair.append(None)

    for idx, ins in enumerate(instrs):
        qubits = ins.qubits
        all_qubits.update(qubits)
        if ins.is_gate and len(qubits) == 1 and not ins.clbits:
            pending.setdefault(qubits[0], []).append(idx)
            continue
        if ins.is_gate and len(qubits) == 2 and not ins.clbits:
            a, b = qubits
            n = open_block.get(a, -1)
            if n >= 0 and open_block.get(b, -1) == n:
                nodes[n] += sorted(pending.pop(a, []) + pending.pop(b, [])) + [idx]
                continue
            for q in qubits:
                close(q)
                flush(q)
            nodes.append([idx])
            pair.append((a, b))
            open_block[a] = open_block[b] = len(nodes) - 1
            continue
        touched = qubits if qubits or not ins.is_barrier else tuple(sorted(all_qubits))
        for q in touched:
            close(q)
            flush(q)
        nodes.append([idx])
        pair.append(None)
    for q in sorted(pending):
        flush(q)
    return nodes


#: Paulis on qubit 0 / qubit 1 of a 4×4 block, stacked (Z, X, Y each).
_PAULI_2Q = np.stack(
    [np.kron(p, np.eye(2)) for p in _PAULI] + [np.kron(np.eye(2), p) for p in _PAULI]
)
_SWAP_PERM = [0, 2, 1, 3]


def _embedded(
    gate: Gate, slot: int, cache: dict[tuple[Gate, int], np.ndarray | None]
) -> np.ndarray | None:
    """4×4 matrix of a block gate: 1q on qubit ``slot``, 2q (``slot`` 2 reversed)."""
    try:
        return cache[gate, slot]
    except KeyError:
        pass
    except TypeError:  # unhashable gate
        return None
    from quspect.gates._matrix_utils import gate_matrix

    try:
        m: np.ndarray | None = np.asarray(gate_matrix(gate), dtype=complex)
    except (KeyError, ValueError, TypeError):
        m = None
    if m is not None:
        if m.shape == (2, 2):
            m = np.kron(m, np.eye(2)) if slot == 0 else np.kron(np.eye(2), m)
        elif slot == 2:
            m = m[np.ix_(_SWAP_PERM, _SWAP_PERM)]
    cache[gate, slot] = m
    return m


def _block_axes(
    instrs: Sequence[CircuitInstruction],
    members: Sequence[int],
    cache: dict[tuple[Gate, int], np.ndarray | None],
) -> tuple[int, int] | None:
    """Pauli axis per qubit of a two-qubit block's unitary (``None``: unknown)."""
    a, b = next(instrs[i].qubits for i in members if len(instrs[i].qubits) == 2)
    u: np.ndarray | None = None
    for i in members:
        ins = instrs[i]
        assert ins.gate is not None
        if len(ins.qubits) == 1:
            slot = 0 if ins.qubits[0] == a else 1
        else:
            slot = 1 if ins.qubits == (a, b) else 2
        m = _embedded(ins.gate, slot, cache)
        if m is None:
            return None
        u = m if u is None else m @ u
    assert u is not None
    err = np.abs(_PAULI_2Q @ u - u @ _PAULI_2Q).max(axis=(1, 2)) < 1e-12
    out = []
    for slot in (0, 1):
        hits = np.flatnonzero(err[3 * slot : 3 * slot + 3])
        out.append(int(hits[0]) if len(hits) else _NONE)
    return out[0], out[1]


def _block_order(instrs: Sequence[CircuitInstruction], info: _Info) -> list[int] | None:
    """Instruction indices reordered by scheduling whole two-qubit blocks.

    ``None`` when no block (of more than one instruction) has a commuting
    axis, or when the block order does not change.
    """
    core = _native_core()
    if core is not None:
        return _block_order_native(core, instrs, info)
    nodes = _blocks(instrs)
    block_cache: dict[tuple[Gate, int], np.ndarray | None] = {}
    offsets = [0]
    flat_res: list[int] = []
    flat_axes: list[int] = []
    two: list[int] = []
    counted: list[int] = []
    useful = False
    for members in nodes:
        first = members[0]
        if len(members) == 1:
            res, axes = info.node(first)
            two.append(int(info.two[first]))
            counted.append(int(info.counted[first]))
        else:
            q2 = next(instrs[i].qubits for i in members if len(instrs[i].qubits) == 2)
            ax = _block_axes(instrs, members, block_cache)
            res = list(q2)
            axes = [_NONE, _NONE] if ax is None else list(ax)
            useful = useful or any(x != _NONE for x in axes)
            two.append(1)
            counted.append(1)
        flat_res += res
        flat_axes += axes
        offsets.append(len(flat_res))
    if not useful:
        return None
    spans = list(zip(offsets[:-1], offsets[1:], strict=True))
    order = _best_order(
        [flat_res[b:e] for b, e in spans],
        [flat_axes[b:e] for b, e in spans],
        [bool(t) for t in two],
        [bool(c) for c in counted],
    )
    if order == list(range(len(nodes))):
        return None
    return [i for k in order for i in nodes[k]]


def _block_order_native(
    core: Any, instrs: Sequence[CircuitInstruction], info: _Info
) -> list[int] | None:
    """:func:`_block_order` on the native core (same result)."""
    from quspect.gates._matrix_utils import gate_matrix

    qubit_offsets = [0]
    qubits: list[int] = []
    kind: list[int] = []
    matrix_offset: list[int] = []
    mats: list[np.ndarray] = []
    size = 0
    index: dict[Gate, int] = {}
    for ins in instrs:
        qs = ins.qubits
        qubits += qs
        qubit_offsets.append(len(qubits))
        gate = ins.gate
        k = len(qs) if ins.is_gate and len(qs) in (1, 2) and not ins.clbits else 0
        if k == 0 and ins.is_barrier and not qs:
            k = 3
        kind.append(k)
        off = -1
        if k in (1, 2) and gate is not None:
            try:
                off = index[gate]
            except KeyError:
                try:
                    m = np.asarray(gate_matrix(gate), dtype=complex)
                except (KeyError, ValueError, TypeError):
                    m = None
                if m is not None and m.shape == (2**k, 2**k):
                    off = size
                    mats.append(m.ravel())
                    size += m.size
                index[gate] = off
            except TypeError:  # unhashable gate
                off = -1
        matrix_offset.append(off)
    matrices = np.concatenate(mats) if mats else np.zeros(0, dtype=complex)
    order: list[int] = core._block_schedule(
        info.offsets,
        info.res,
        info.axes,
        [int(t) for t in info.two],
        [int(c) for c in info.counted],
        qubit_offsets,
        qubits,
        kind,
        matrix_offset,
        matrices,
    )
    return order or None


def _native_core() -> Any | None:
    """The loaded extension module if it provides the scheduler.

    The routing backend switch (``QUSPECT_ROUTING_BACKEND=python``) also
    selects the pure-Python scheduler; both give identical orders.
    """
    from quspect._native_loader import load_core
    from quspect.transpiler._routing import ROUTING_BACKEND_ENV

    if os.environ.get(ROUTING_BACKEND_ENV) == "python":
        return None
    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        return None
    if core is None or not hasattr(core, "_block_schedule"):
        return None
    return core
