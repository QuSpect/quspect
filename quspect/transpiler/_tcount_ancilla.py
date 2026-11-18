"""Hadamard gadgets: T-count optimization with ancillas (internal to the T-count pass).

Without ancillas, TODD only sees the T gates between two Hadamards as one
phase polynomial.  Heyfron & Campbell's Hadamard gadget removes the
Hadamards inside a Clifford+T region at the price of one clean ancilla, one
measurement and a classically controlled Clifford each, so that *all* T gates
of the region form a single phase polynomial:

1. The region is split into a Clifford head (no T in its causal past), a
   middle and a Clifford tail (no T in its causal future).  Every ``H`` left
   in the middle is *internal*.
2. An internal ``H`` on wire ``q`` becomes ``CX(q, a); H(a)`` with a fresh
   ancilla ``a`` in ``|0⟩``; the logical qubit continues on ``a``.  Measuring
   ``q`` in the X basis with outcome ``m`` leaves ``X^m H|ψ⟩`` on ``a``, so
   the gadget is completed by ``X^m`` on ``a``.
3. In the path sum of the gadgetized middle every wire holds a parity of the
   input variables and one variable per ancilla, all T gates are phases on
   parities and each gadget adds a quadratic (Clifford) term.  The T part is
   reduced by the pass's phase-polynomial search and accepted only after the
   exact Clifford-difference check, quadratic terms included.
4. The middle is synthesized again: ancillas in ``|+⟩``, one parity ladder
   with a ``T`` per column, the Clifford phases, and a CNOT network for the
   final wire labels.
5. The measurements are deferred to the end of the middle, in gadget order.
   The correction of gadget ``k`` is ``Q_k = U_{>k} X_{a_k} U_{>k}†`` for the
   rest ``U_{>k}`` of the sequential gadget circuit: a Pauli times a diagonal
   Clifford on parities, computed symbolically and applied under the
   measured bit (``x``, ``z``, ``s``, ``sdg`` as classically controlled gates
   around unconditional CNOT ladders).
6. Each logical qubit is moved back to its own wire (the wire was measured:
   a conditional ``x`` resets it, two ``cx`` move the state) and every other
   measured ancilla is reset by a conditional ``x``, so all ancillas end in
   ``|0⟩`` and can be reused.

:func:`min_h_emit` writes a rotation sequence with few internal Hadamards
(one frame change makes as many pending rotations diagonal as possible),
which is what decides the number of ancillas.

Everything here is integer bookkeeping; the pass's tests check the result
against an exact simulation of every measurement branch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from quspect.transpiler import _tcount_poly as poly
from quspect.transpiler import _tcount_rotations as rot
from quspect.transpiler._tcount_rotations import CX, SDG, TDG, GateT, H, Pauli, S, T

__all__: list[str] = []

#: Extra operation codes of a gadget program (besides the gate tuples).
MEASURE_X = "mx"  # ("mx", wire): measure the wire in the X basis
CONDITIONAL = "cc"  # ("cc", control wire, name, target wire): gate if the bit is 1
GATE = "g"  # ("g", gate tuple)

_PHASE = {S: 2, SDG: 6, T: 1, TDG: 7}

Reducer = Callable[[list[tuple[int, int]], list[int], int], list[int]]


@dataclass
class GadgetProgram:
    """A region rewritten with Hadamard gadgets.

    Attributes:
        num_wires: Logical wires ``0 … n−1`` followed by the ancilla wires.
        ops: ``("g", gate)``, ``("mx", wire)`` and ``("cc", control wire,
            gate name, target wire)`` in time order.  Every ancilla starts
            and ends in ``|0⟩`` and every logical qubit ends on its own wire.
        t_count: Number of ``T`` / ``T†`` gates.
        measured: The measured wires, in order.
    """

    num_wires: int
    ops: list[tuple[object, ...]] = field(default_factory=list)
    t_count: int = 0
    measured: list[int] = field(default_factory=list)

    @property
    def ancillas(self) -> int:
        return len(self.measured)


# ------------------------------------------------------------ few internal Hadamards
def _available(pending: Sequence[Pauli]) -> list[int]:
    return [
        i for i, p in enumerate(pending) if all(rot.commute(p, q) for q in pending[:i])
    ]


def _frame_change(p: Pauli, t: int) -> list[GateT]:
    x = p[0]
    gates: list[GateT] = [
        (CX, t, q) for q in range(x.bit_length()) if (x >> q) & 1 and q != t
    ]
    if (p[1] >> t) & 1:
        gates.append((SDG, t, -1))
    gates.append((H, t, -1))
    return gates


def _conjugate_all(pending: Sequence[Pauli], gates: Sequence[GateT]) -> list[Pauli]:
    out = []
    for x, z, s in pending:
        nx, nz, sg = rot.conj_circ((x, z, 1), gates)
        out.append((nx, nz, s * sg))
    return out


def min_h_emit(
    clifford: Sequence[GateT], rots: Sequence[Pauli], n: int, max_candidates: int = 64
) -> list[GateT]:
    """Gate list of ``F · R_m ⋯ R_1`` with few Hadamards between the T gates.

    The running Clifford frame is never undone.  A rotation that is diagonal
    in the current frame costs a CNOT ladder and one ``T``.  When no
    available rotation is diagonal, one frame change (CNOTs gathering the X
    part of a rotation on one qubit, an ``S†`` for a ``Y``, one ``H``) is
    chosen greedily to make as many pending rotations diagonal as possible.
    """
    pending = list(rots)
    out: list[GateT] = []
    frame: list[GateT] = []
    while pending:
        avail = _available(pending)
        diagonal = [i for i in avail if pending[i][0] == 0]
        if diagonal:
            p = pending.pop(diagonal[0])
            pre, tq, sign = rot.basis_change(p, n)
            out += pre
            out.append((T if p[2] * sign > 0 else TDG, tq, -1))
            frame += pre
            pending = _conjugate_all(pending, pre)
            continue
        best_key: tuple[int, int] | None = None
        best_gates: list[GateT] = []
        best_pending: list[Pauli] = []
        for i in avail[:max_candidates]:
            p = pending[i]
            for t in range(p[0].bit_length()):
                if not (p[0] >> t) & 1:
                    continue
                gates = _frame_change(p, t)
                new = _conjugate_all(pending, gates)
                key = (
                    -sum(1 for q in new if q[0] == 0),
                    sum(q[0].bit_count() for q in new),
                )
                if best_key is None or key < best_key:
                    best_key, best_gates, best_pending = key, gates, new
        out += best_gates
        frame += best_gates
        pending = best_pending
    final = rot.clifford_circuit(
        [rot.inv_gate(g) for g in reversed(frame)] + list(clifford), n
    )
    return out + final


# ---------------------------------------------------------------- head / middle / tail
def split_head_middle_tail(
    gates: Sequence[GateT], n: int
) -> tuple[list[GateT], list[GateT], list[GateT]]:
    """Clifford head (no T before), middle, Clifford tail (no T after)."""
    count = len(gates)
    past = [False] * count
    reach = [False] * n
    for i, g in enumerate(gates):
        wires = (g[1],) if g[2] < 0 else (g[1], g[2])
        seen = any(reach[w] for w in wires)
        past[i] = seen
        if g[0] == T or g[0] == TDG:
            seen = True
        for w in wires:
            reach[w] = seen
    future = [False] * count
    reach = [False] * n
    for i in range(count - 1, -1, -1):
        g = gates[i]
        wires = (g[1],) if g[2] < 0 else (g[1], g[2])
        seen = any(reach[w] for w in wires)
        future[i] = seen
        if g[0] == T or g[0] == TDG:
            seen = True
        for w in wires:
            reach[w] = seen
    head: list[GateT] = []
    mid: list[GateT] = []
    tail: list[GateT] = []
    for i, g in enumerate(gates):
        is_t = g[0] == T or g[0] == TDG
        if not is_t and not past[i]:
            head.append(g)
        elif not is_t and not future[i]:
            tail.append(g)
        else:
            mid.append(g)
    return head, mid, tail


def internal_hadamards(gates: Sequence[GateT], n: int) -> int:
    """Number of ancillas the gadget form of a gate list needs."""
    return sum(1 for g in split_head_middle_tail(gates, n)[1] if g[0] == H)


# ---------------------------------------------------------------- the gadgetized middle
@dataclass
class _PathSum:
    num_wires: int
    logical_wire: list[int]  # logical qubit -> wire holding it at the end
    label: list[int]  # wire -> parity mask over the variables
    phase: dict[int, int]  # parity mask -> coefficient mod 8
    quad: list[tuple[int, int]]  # (parity L, ancilla variable y): term 4·(L·v)·y
    seq: list[GateT]  # the sequential gadget circuit on the wires
    gadgets: list[tuple[int, int, int]]  # (frozen wire, ancilla, index in seq after it)


def _gadgetize(mid: Sequence[GateT], n: int) -> _PathSum:
    wire = list(range(n))
    label = [1 << i for i in range(n)]
    phase: dict[int, int] = {}
    quad: list[tuple[int, int]] = []
    seq: list[GateT] = []
    gadgets: list[tuple[int, int, int]] = []
    count = n
    for g in mid:
        op = g[0]
        if op == CX:
            c, t = wire[g[1]], wire[g[2]]
            label[t] ^= label[c]
            seq.append((CX, c, t))
        elif op == H:
            q = wire[g[1]]
            a = count
            count += 1
            label.append(1 << a)
            quad.append((label[q], a))
            seq += [(CX, q, a), (H, a, -1)]
            gadgets.append((q, a, len(seq)))
            wire[g[1]] = a
        else:
            w = wire[g[1]]
            phase[label[w]] = (phase.get(label[w], 0) + _PHASE[op]) % 8
            seq.append((op, w, -1))
    return _PathSum(count, wire, label, phase, quad, seq, gadgets)


def _clifford_remainder(
    ps: _PathSum, columns: Sequence[int]
) -> tuple[dict[int, int], list[tuple[int, int]]] | None:
    """``(S powers, CZ pairs)`` of ``phase − columns`` plus gadget terms, or ``None``.

    The exact acceptance test: the difference must be a Clifford phase
    polynomial (cubic coefficients 0 mod 8, quadratic 0 mod 4, linear 0 mod 2).
    """
    lin, quad, cub = poly.multilinear(
        list(ps.phase.items()) + [(p, -1) for p in columns]
    )
    for parity, y in ps.quad:
        for i in range(parity.bit_length()):
            if (parity >> i) & 1:
                key = (min(i, y), max(i, y))
                quad[key] = (quad.get(key, 0) + 4) % 8
    if (
        any(cub.values())
        or any(v % 4 for v in quad.values())
        or any(v % 2 for v in lin.values())
    ):
        return None
    return (
        {i: (v // 2) % 4 for i, v in lin.items() if v},
        sorted(pair for pair, v in quad.items() if v == 4),
    )


def _ladder(mask: int, op: int) -> list[GateT]:
    bits = [i for i in range(mask.bit_length()) if (mask >> i) & 1]
    target = bits[-1]
    ladder: list[GateT] = [(CX, b, target) for b in bits[:-1]]
    return ladder + [(op, target, -1)] + ladder[::-1]


def _synthesize_middle(
    ps: _PathSum,
    n: int,
    columns: Sequence[int],
    powers: dict[int, int],
    pairs: Sequence[tuple[int, int]],
) -> list[GateT]:
    out: list[GateT] = [(H, a, -1) for a in range(n, ps.num_wires)]  # ancillas in |+>
    for p in columns:
        out += _ladder(p, T)
    for i in sorted(powers):
        out += [(op, i, -1) for op in {1: (S,), 2: (S, S), 3: (SDG,)}[powers[i]]]
    for i, j in pairs:
        out += [(H, j, -1), (CX, i, j), (H, j, -1)]
    # CNOT network taking the identity labels to the final wire labels
    rows = list(ps.label)
    ops: list[tuple[int, int]] = []
    for col in range(ps.num_wires):
        pivot = next(
            (r for r in range(col, ps.num_wires) if (rows[r] >> col) & 1), None
        )
        if pivot is None:
            raise _GadgetError("final wire labels are not independent")
        if pivot != col:
            for c, t in ((pivot, col), (col, pivot), (pivot, col)):
                rows[t] ^= rows[c]
                ops.append((c, t))
        for r in range(ps.num_wires):
            if r != col and (rows[r] >> col) & 1:
                rows[r] ^= rows[col]
                ops.append((col, r))
    out += [(CX, c, t) for c, t in reversed(ops)]
    return out


class _GadgetError(Exception):
    """The gadget form of this gate list cannot be built (the caller skips it)."""


def _correction(
    seq: Sequence[GateT], start: int, ancilla: int
) -> tuple[int, int, dict[int, int]]:
    """``U X_a U†`` for ``U = seq[start:]``: ``(x mask, z mask, {parity: S power})``."""
    x, z = 1 << ancilla, 0
    diag: dict[int, int] = {}
    for op, p0, p1 in seq[start:]:
        if op == CX:
            if (x >> p0) & 1:
                x ^= 1 << p1
            if (z >> p1) & 1:
                z ^= 1 << p0
            if diag:
                moved: dict[int, int] = {}
                for parity, power in diag.items():
                    if (parity >> p1) & 1:
                        parity ^= 1 << p0
                    moved[parity] = (moved.get(parity, 0) + power) % 4
                diag = {parity: power for parity, power in moved.items() if power}
        elif op == H:
            if any((parity >> p0) & 1 for parity in diag):
                raise _GadgetError("a diagonal correction meets a Hadamard")
            xb, zb = (x >> p0) & 1, (z >> p0) & 1
            x = (x & ~(1 << p0)) | (zb << p0)
            z = (z & ~(1 << p0)) | (xb << p0)
        elif op in (S, SDG):
            if (x >> p0) & 1:
                z ^= 1 << p0
        elif (x >> p0) & 1:  # T or T†
            mask = 1 << p0
            diag[mask] = (diag.get(mask, 0) + (3 if op == T else 1)) % 4
            if not diag[mask]:
                del diag[mask]
    return x, z, diag


def build_gadget_program(
    gates: Sequence[GateT], n: int, reduce: Reducer
) -> GadgetProgram | None:
    """Rewrite a Clifford+T gate list with Hadamard gadgets.

    Args:
        gates: The region over ``{H, S, SDG, T, TDG, CX}`` on ``n`` wires.
        n: Number of logical wires.
        reduce: ``reduce(terms, columns, k)`` returning an equivalent, shorter
            column set of the phase polynomial with ``(column, +1)`` terms
            over ``k`` variables.

    Returns:
        The program, or ``None`` when the gadget form cannot be built for
        this gate list.
    """
    head, mid, tail = split_head_middle_tail(gates, n)
    ps = _gadgetize(mid, n)
    if ps.num_wires == n:
        return None  # no internal Hadamard: nothing an ancilla could buy
    odd = sorted(p for p, c in ps.phase.items() if c % 2)
    try:
        candidate = reduce([(p, 1) for p in odd], odd, ps.num_wires) if odd else []
        chosen = None
        for columns in (candidate, odd):
            remainder = _clifford_remainder(ps, columns)
            if remainder is not None:
                chosen = (list(columns), remainder)
                break
        if chosen is None:
            return None
        columns, (powers, pairs) = chosen
        ops: list[tuple[object, ...]] = [(GATE, g) for g in head]
        ops += [(GATE, g) for g in _synthesize_middle(ps, n, columns, powers, pairs)]
        measured: list[int] = []
        for frozen, ancilla, position in ps.gadgets:
            x, z, diag = _correction(ps.seq, position, ancilla)
            done = measured + [frozen]
            if any(((x | z) >> w) & 1 for w in done) or any(
                (parity >> w) & 1 for parity in diag for w in done
            ):
                raise _GadgetError("a correction acts on a measured wire")
            ops.append((MEASURE_X, frozen))
            for parity, power in sorted(diag.items()):
                bits = [i for i in range(parity.bit_length()) if (parity >> i) & 1]
                target = bits[-1]
                # an unconditional CNOT ladder around a conditional phase is the
                # conditional phase on the parity
                ladder: list[tuple[object, ...]] = [
                    (GATE, (CX, b, target)) for b in bits[:-1]
                ]
                name = {1: "s", 2: "z", 3: "sdg"}[power]
                ops += ladder + [(CONDITIONAL, frozen, name, target)] + ladder[::-1]
            for w in range(max(x, z).bit_length()):
                if (z >> w) & 1:
                    ops.append((CONDITIONAL, frozen, "z", w))
                if (x >> w) & 1:
                    ops.append((CONDITIONAL, frozen, "x", w))
            measured.append(frozen)
    except _GadgetError:
        return None
    # Every logical qubit back to its own wire, every ancilla back to |0>.
    holders = set(ps.logical_wire)
    for logical, wire in enumerate(ps.logical_wire):
        if wire == logical:
            continue
        ops.append((CONDITIONAL, logical, "x", logical))  # measured wire -> |0>
        ops.append((GATE, (CX, wire, logical)))
        ops.append(
            (GATE, (CX, logical, wire))
        )  # the state is on its wire, `wire` in |0>
    for wire in measured:
        if wire >= n and wire not in holders:
            ops.append((CONDITIONAL, wire, "x", wire))
    ops += [(GATE, g) for g in tail]
    return GadgetProgram(
        num_wires=ps.num_wires, ops=ops, t_count=len(columns), measured=measured
    )
