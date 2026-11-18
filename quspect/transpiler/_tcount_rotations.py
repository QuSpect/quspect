"""Pauli-rotation form of Clifford+T circuits (internal to the T-count pass).

A Clifford+T circuit equals ``F · R_m ⋯ R_1`` up to a global phase, where
``F`` is a Clifford and ``R_i = exp(∓iπ/8 · P_i)`` is a ``π/4`` rotation about
an n-qubit Pauli ``P_i`` (Gosset et al. 2014; Litinski 2019): every ``T`` on
qubit ``q`` becomes the rotation about ``C† Z_q C`` with ``C`` the Clifford
gates before it.  The T-count is the number of rotations.

This module holds the bookkeeping used by
:class:`~quspect.transpiler.TCountOptimizePass`:

- gates are tuples ``(op, a, b)`` over ``{H, S, SDG, T, TDG, CX}`` (``b = -1``
  for one-qubit gates, ``a`` the control of a ``CX``);
- a Pauli is ``(x, z, sign)`` with bit masks ``x``, ``z`` over the qubits,
  Hermitian convention ``Y = iXZ`` and ``sign = ±1``; for a rotation the sign
  is the exponent (``+1``: ``T``-like, ``-1``: ``T†``-like);
- conjugation of Paulis by Clifford gates, rotation merging, Clifford
  synthesis from a tableau and the emission of a rotation sequence as gates.

Everything is integer logic on Python ints, so it is exact and
platform-independent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__: list[str] = []

H, S, SDG, T, TDG, CX = 0, 1, 2, 3, 4, 5

GateT = tuple[int, int, int]
Pauli = tuple[int, int, int]

_INV = {H: H, S: SDG, SDG: S}


def g1(op: int, q: int) -> GateT:
    return (op, q, -1)


def cx(c: int, t: int) -> GateT:
    return (CX, c, t)


def inv_gate(g: GateT) -> GateT:
    return g if g[0] == CX else (_INV[g[0]], g[1], -1)


def conj(p: Pauli, g: GateT) -> Pauli:
    """``g P g†`` for a Clifford gate ``g``."""
    x, z, s = p
    op, a, b = g
    if op == CX:
        xa, za, xb, zb = (x >> a) & 1, (z >> a) & 1, (x >> b) & 1, (z >> b) & 1
        if xa & zb & (xb ^ za ^ 1):
            s = -s
        return x ^ (xa << b), z ^ (zb << a), s
    xa, za = (x >> a) & 1, (z >> a) & 1
    if op == H:
        if xa & za:
            s = -s
        if xa != za:
            x ^= 1 << a
            z ^= 1 << a
        return x, z, s
    if op == S:  # X -> Y, Y -> -X
        if xa & za:
            s = -s
    elif xa & (za ^ 1):  # S†: X -> -Y, Y -> X
        s = -s
    return x, z ^ (xa << a), s


def conj_circ(p: Pauli, circ: Iterable[GateT]) -> Pauli:
    """``C P C†`` with ``C`` given in time order."""
    for g in circ:
        p = conj(p, g)
    return p


def conj_circ_inv(p: Pauli, circ: Sequence[GateT]) -> Pauli:
    """``C† P C`` with ``C`` given in time order."""
    for g in reversed(circ):
        p = conj(p, inv_gate(g))
    return p


def commute(p: Pauli, q: Pauli) -> bool:
    return ((p[0] & q[1]).bit_count() + (p[1] & q[0]).bit_count()) % 2 == 0


def weight(p: Pauli) -> int:
    return (p[0] | p[1]).bit_count()


class _Frame:
    """The map ``P ↦ C† P C`` for a growing Clifford ``C`` (gates appended in order).

    Stored as the images of ``X_q`` and ``Z_q``; appending ``g`` replaces the
    map by ``P ↦ C† (g† P g) C``, which only changes the images of the
    generators on the qubits of ``g``.
    """

    def __init__(self, n: int) -> None:
        self.imx: list[Pauli] = [(1 << q, 0, 1) for q in range(n)]
        self.imz: list[Pauli] = [(0, 1 << q, 1) for q in range(n)]

    def append(self, g: GateT) -> None:
        op, a, b = g
        imx, imz = self.imx, self.imz
        if op == CX:  # g† X_a g = X_a X_b, g† Z_b g = Z_a Z_b
            imx[a] = _mul(imx[a], imx[b])
            imz[b] = _mul(imz[a], imz[b])
        elif op == H:
            imx[a], imz[a] = imz[a], imx[a]
        elif op == S:  # S† X S = -Y = -i X Z
            imx[a] = _mul_y(imx[a], imz[a], -1)
        else:  # S X S† = Y
            imx[a] = _mul_y(imx[a], imz[a], 1)


def _mul(p: Pauli, q: Pauli) -> Pauli:
    """Product of two commuting Paulis."""
    x, z = p[0] ^ q[0], p[1] ^ q[1]
    # P = i^{|x1 & z1|} X^{x1} Z^{z1}; moving Z^{z1} past X^{x2} gives (-1)^{|z1 & x2|}
    e = (p[0] & p[1]).bit_count() + (q[0] & q[1]).bit_count() - (x & z).bit_count()
    e += 2 * (p[1] & q[0]).bit_count()
    # commuting Hermitian Paulis multiply to a Hermitian Pauli: e is even
    return x, z, p[2] * q[2] * (1 if e % 4 == 0 else -1)


def _mul_y(px: Pauli, pz: Pauli, sign: int) -> Pauli:
    """``sign · i · A · B`` for anticommuting Hermitian Paulis (the image of ``±Y``)."""
    x, z = px[0] ^ pz[0], px[1] ^ pz[1]
    e = (px[0] & px[1]).bit_count() + (pz[0] & pz[1]).bit_count() - (x & z).bit_count()
    e += 2 * (px[1] & pz[0]).bit_count() + 1
    return x, z, sign * px[2] * pz[2] * (1 if e % 4 == 0 else -1)


def from_gates(gates: Iterable[GateT], n: int) -> tuple[list[GateT], list[Pauli]]:
    """Split a Clifford+T gate list into ``(F, rotations)``, rotations in time order."""
    frame = _Frame(n)
    cliff: list[GateT] = []
    rots: list[Pauli] = []
    for g in gates:
        if g[0] == T or g[0] == TDG:
            x, z, s = frame.imz[g[1]]
            rots.append((x, z, s if g[0] == T else -s))
        else:
            cliff.append(g)
            frame.append(g)
    return cliff, rots


def basis_change(
    p: Pauli, n: int, target: int | None = None
) -> tuple[list[GateT], int, int]:
    """``(V, t, sign)``: gates ``V`` and a qubit ``t`` with ``V P V† = sign · Z_t``."""
    x, z, _ = p
    pre: list[GateT] = []
    supp: list[int] = []
    m = x | z
    while m:
        low = m & -m
        q = low.bit_length() - 1
        m ^= low
        xa, za = (x >> q) & 1, (z >> q) & 1
        supp.append(q)
        if xa and za:
            pre += [g1(SDG, q), g1(H, q)]
        elif xa:
            pre.append(g1(H, q))
    tq = supp[-1] if target is None else target
    pre += [cx(q, tq) for q in supp if q != tq]
    return pre, tq, conj_circ((x, z, 1), pre)[2]


def rotation_square(p: Pauli, n: int) -> list[GateT]:
    """Gates of ``R(P)^{2s}``, the Clifford ``exp(∓iπ/4 · P)``."""
    pre, tq, sign = basis_change(p, n)
    mid = S if p[2] * sign > 0 else SDG
    return pre + [g1(mid, tq)] + [inv_gate(g) for g in reversed(pre)]


def push_clifford_left(
    clifford: list[GateT], rots: Sequence[Pauli], pos: int, c: Sequence[GateT]
) -> tuple[list[GateT], list[Pauli]]:
    """Insert the Clifford ``c`` before ``rots[pos:]`` and absorb it into ``F``."""
    out = list(rots[:pos])
    for x, z, s in rots[pos:]:
        nx, nz, sg = conj_circ_inv((x, z, 1), c)
        out.append((nx, nz, s * sg))
    return list(c) + clifford, out


def merge_rotations(
    clifford: list[GateT], rots: list[Pauli], n: int
) -> tuple[list[GateT], list[Pauli]]:
    """Merge rotations about the same Pauli that can be brought together.

    Two equal rotations make the Clifford ``exp(∓iπ/4·P)``, which is moved to
    ``F``; opposite rotations cancel.  A rotation only moves past rotations it
    commutes with.
    """
    i = 0
    while i < len(rots):
        p = rots[i]
        merged = False
        for j in range(i + 1, len(rots)):
            q = rots[j]
            if q[0] == p[0] and q[1] == p[1]:
                rest = rots[:i] + rots[i + 1 : j] + rots[j + 1 :]
                if q[2] == -p[2]:
                    rots = rest
                else:
                    clifford, rots = push_clifford_left(
                        clifford, rest, i, rotation_square(p, n)
                    )
                merged = True
                break
            if not commute(p, q):
                break
        if merged:
            # the removal can only create merges for rotations before position i that
            # were blocked by p; restarting from the front keeps the fixed point simple
            i = 0
        else:
            i += 1
    return clifford, rots


# ---------------------------------------------------------------- Clifford synthesis
def clifford_circuit(clifford: Sequence[GateT], n: int) -> list[GateT]:
    """Re-synthesize a Clifford gate list from its tableau (``O(n²)`` gates)."""
    imx = [conj_circ((1 << i, 0, 1), clifford) for i in range(n)]
    imz = [conj_circ((0, 1 << i, 1), clifford) for i in range(n)]
    return _clifford_from_images(imx, imz, n)


def _clifford_from_images(imx: list[Pauli], imz: list[Pauli], n: int) -> list[GateT]:
    """Gate list of the Clifford ``C`` given ``C X_i C†`` and ``C Z_i C†``."""
    applied: list[GateT] = []

    def do(g: GateT) -> None:
        applied.append(g)
        for i in range(n):
            imx[i] = conj(imx[i], g)
            imz[i] = conj(imz[i], g)

    def site(p: Pauli, q: int) -> int:
        return ((p[0] >> q) & 1) + 2 * ((p[1] >> q) & 1)

    for i in range(n):
        for q in range(i, n):
            t = site(imx[i], q)
            if t == 2:
                do(g1(H, q))
            elif t == 3:
                do(g1(S, q))
        xs = [q for q in range(i, n) if site(imx[i], q)]
        j = xs[0]
        for q in xs[1:]:
            do(cx(j, q))
        if j != i:
            do(cx(j, i))
            do(cx(i, j))
        for q in range(i + 1, n):
            t = site(imz[i], q)
            if t == 1:
                do(g1(H, q))
            elif t == 3:
                do(g1(S, q))
                do(g1(H, q))
        for q in range(i + 1, n):
            if site(imz[i], q):
                do(cx(q, i))
        if site(imz[i], i) == 3:
            for g in (g1(H, i), g1(S, i), g1(H, i)):
                do(g)
        if imx[i][2] < 0:
            do(g1(S, i))
            do(g1(S, i))
        if imz[i][2] < 0:
            for g in (g1(H, i), g1(S, i), g1(S, i), g1(H, i)):
                do(g)
    if any(imx[i] != (1 << i, 0, 1) or imz[i] != (0, 1 << i, 1) for i in range(n)):
        raise AssertionError(
            "Clifford synthesis failed"
        )  # pragma: no cover - internal invariant
    return [inv_gate(g) for g in reversed(applied)]


def emit(
    clifford: Sequence[GateT], rots: Sequence[Pauli], n: int, lookahead: int = 6
) -> list[GateT]:
    """Gate list of ``F · R_m ⋯ R_1`` that keeps the Clifford frame between rotations.

    ``R(P_2) R(P_1) = V† R(V P_2 V†) T V`` when ``V`` maps ``P_1`` to ``Z``, so
    the gates that bring a rotation onto one qubit are never undone: later
    rotations are expressed in the current frame and the accumulated frame is
    folded into the final Clifford.  Among the rotations that may come next
    (those commuting with every rotation still ahead of them) the one with the
    smallest support is emitted first, and the CX-ladder target is the qubit
    that leaves the next ``lookahead`` rotations with the smallest supports.
    """
    pending: list[Pauli] = list(rots)
    out: list[GateT] = []
    frame: list[GateT] = []
    while pending:
        best_w, best_i = -1, -1
        for i, p in enumerate(pending):
            if all(commute(p, q) for q in pending[:i]):
                w = weight(p)
                if best_i < 0 or w < best_w:
                    best_w, best_i = w, i
                    if w == 1:
                        break
        p = pending.pop(best_i)
        pre, tq, sign = basis_change(p, n)
        if lookahead and best_w > 1:
            look = pending[:lookahead]
            cost_best = -1
            support = p[0] | p[1]
            for cand in range(n):
                if not (support >> cand) & 1:
                    continue
                c_pre, c_tq, c_sign = basis_change(p, n, cand)
                c = sum(weight(conj_circ((x, z, 1), c_pre)) for x, z, _ in look)
                if cost_best < 0 or c < cost_best:
                    cost_best, pre, tq, sign = c, c_pre, c_tq, c_sign
        out += pre
        out.append(g1(T if p[2] * sign > 0 else TDG, tq))
        frame += pre
        moved = []
        for x, z, e in pending:
            nx, nz, sg = conj_circ((x, z, 1), pre)
            moved.append((nx, nz, e * sg))
        pending = moved
    final = clifford_circuit([inv_gate(g) for g in reversed(frame)] + list(clifford), n)
    return out + final


# -------------------------------------------------------- frames for commuting groups
def local_frame(
    paulis: Sequence[Pauli], n: int
) -> tuple[list[GateT], list[int]] | None:
    """One-qubit Clifford ``V`` making every Pauli Z-type (one Pauli type per qubit).

    Returns ``(V, used qubits)`` or ``None`` when some qubit carries two
    different non-identity Paulis across the group.
    """
    kind: dict[int, int] = {}
    for x, z, _ in paulis:
        m = x | z
        while m:
            low = m & -m
            q = low.bit_length() - 1
            m ^= low
            t = ((x >> q) & 1) + 2 * ((z >> q) & 1)
            if kind.setdefault(q, t) != t:
                return None
    gates: list[GateT] = []
    used = sorted(kind)
    for q in used:
        if kind[q] == 3:
            gates += [g1(SDG, q), g1(H, q)]
        elif kind[q] == 1:
            gates.append(g1(H, q))
    return gates, used


def compress_frame(paulis: Sequence[Pauli], n: int) -> tuple[list[GateT], list[int]]:
    """Clifford ``V`` and qubits ``used`` with every ``V P V†`` X-type on ``used``.

    Greedy symplectic reduction: each step moves one remaining Pauli onto a
    single free qubit and clears that qubit from the others.
    """
    work: list[Pauli] = [
        (x, z, 1) for x, z in dict.fromkeys((p[0], p[1]) for p in paulis)
    ]
    gates: list[GateT] = []
    free = list(range(n))
    used: list[int] = []

    def do(g: GateT) -> None:
        gates.append(g)
        for k in range(len(work)):
            work[k] = conj(work[k], g)

    while True:
        mask = 0
        for q in free:
            mask |= 1 << q
        cand = [k for k in range(len(work)) if (work[k][0] | work[k][1]) & mask]
        if not cand:
            break
        res = {k: (work[k][0] & mask, work[k][1] & mask) for k in cand}

        def has_partner(
            k: int, cand: list[int] = cand, res: dict[int, tuple[int, int]] = res
        ) -> bool:
            a = res[k]
            return any(
                ((a[0] & res[m][1]).bit_count() + (a[1] & res[m][0]).bit_count()) % 2
                for m in cand
                if m != k
            )

        # prefer a Pauli with an anticommuting partner on the free qubits
        k = next((k for k in cand if has_partner(k)), cand[0])
        for q in list(free):
            xa, za = (work[k][0] >> q) & 1, (work[k][1] >> q) & 1
            if za and not xa:
                do(g1(H, q))
            elif za and xa:
                do(g1(S, q))
        xs = [q for q in free if (work[k][0] >> q) & 1]
        i = xs[0]
        for q in xs[1:]:
            do(cx(i, q))
        partner = next(
            (m for m in range(len(work)) if m != k and (work[m][1] >> i) & 1), None
        )
        if partner is not None:
            for q in free:
                if q == i:
                    continue
                xa, za = (work[partner][0] >> q) & 1, (work[partner][1] >> q) & 1
                if xa and not za:
                    do(g1(H, q))
                elif xa and za:
                    do(g1(S, q))
                    do(g1(H, q))
            for q in free:
                if q != i and (work[partner][1] >> q) & 1:
                    do(cx(q, i))
        used.append(i)
        free.remove(i)
    return gates, used
