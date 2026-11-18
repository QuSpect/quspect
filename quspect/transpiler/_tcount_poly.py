"""T-count reduction of phase polynomials (internal to the T-count pass).

A set of mutually commuting ``π/4`` rotations is, in a suitable Clifford
frame, the diagonal unitary ``ω^{f(x)}`` (``ω = e^{iπ/4}``) of a *phase
polynomial* ``f(x) = Σ_j s_j · (c_j · x mod 2)  (mod 8)``: one parity ``c_j``
(a bit mask over ``k`` variables, a *column*) per rotation.  Two column sets
give the same unitary up to a diagonal Clifford iff the difference of their
polynomials has the form ``2·Σ a_i x_i + 4·Σ b_ij x_i x_j``; in multilinear
form: all cubic coefficients are ``0 mod 8``, all quadratic ones ``0 mod 4``
and all linear ones ``0 mod 2`` (Amy & Mosca 2019).  The T-count is the number
of columns, so the task is to find an equivalent set with fewer columns.

Three searches are implemented, each with a pure-Python reference and a native
twin (``cpp/src/transpiler/tcount.cpp``) that returns the same result:

- :func:`todd` — third-order duplicate-and-destroy (Heyfron & Campbell 2018):
  moves ``A → A + z yᵀ`` with ``y`` in the null space of the constraints that
  keep the cubic part, choosing for each step the move that removes the most
  columns over all ``z = c_a ⊕ c_b``, plus optional sideways moves.
- :func:`rm_search` — large-neighbourhood search with Reed–Muller moves: the
  indicators of 4-dimensional affine flats are the minimum-weight codewords of
  the punctured code ``RM(k−4, k)*`` whose cosets are the equivalence classes,
  so XOR-ing the column set with the points of a 4-flat is always allowed.
- :func:`tensor_starts` — when the cubic part is trilinear in a tripartition
  of the variables (``4·Σ T_ijk a_i b_j c_k``, e.g. a finite-field multiplier),
  every rank decomposition of ``T`` over GF(2) gives an equivalent column set
  (seven parities per rank-one term).  Low-rank decompositions are found by a
  flip-graph walk (Kauers & Moosbauer 2022) from the given terms or from a
  Karatsuba decomposition.

All randomness is drawn as integers (``randrange`` / ``getrandbits``) from a
seeded ``random.Random``, so results are deterministic and identical between
the Python and the native implementation.  Callers accept a result only after
:func:`clifford_difference` confirms the identity exactly.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from typing import Any

from quspect._native_loader import load_core

__all__: list[str] = []

Term = tuple[int, int, int]  # rank-one term u ⊗ v ⊗ w (masks over the three parts)

#: Columns must fit in 64 bits for the native implementation.
_NATIVE_MAX_VARS = 64
#: 4-flats sampled per step of the Reed–Muller search.
_FLAT_SAMPLES = 4000
#: Flip-graph walk: steps without progress before a "plus" move.
_PLUS_AFTER = 500


def _core_function(name: str, native: bool) -> Callable[..., Any] | None:
    if not native:
        return None
    try:
        core = load_core()
    except ImportError:
        return None
    return None if core is None else getattr(core, name, None)


# ---------------------------------------------------------------- exact identity check
def multilinear(
    terms: Iterable[tuple[int, int]],
) -> tuple[dict[int, int], dict[tuple[int, int], int], dict[tuple[int, int, int], int]]:
    """Multilinear form mod 8 of ``Σ c · (p·x mod 2)``: ``(linear, quadratic, cubic)``.

    Uses ``x_1 ⊕ ⋯ ⊕ x_w = Σ x_i − 2 Σ x_i x_j + 4 Σ x_i x_j x_l  (mod 8)``.
    """
    lin: dict[int, int] = {}
    quad: dict[tuple[int, int], int] = {}
    cub: dict[tuple[int, int, int], int] = {}
    for p, c in terms:
        c %= 8
        if not c:
            continue
        bits = [i for i in range(p.bit_length()) if (p >> i) & 1]
        for i in bits:
            lin[i] = (lin.get(i, 0) + c) % 8
        for pair in combinations(bits, 2):
            quad[pair] = (quad.get(pair, 0) - 2 * c) % 8
        if c % 2:
            for triple in combinations(bits, 3):
                cub[triple] = (cub.get(triple, 0) + 4) % 8
    return lin, quad, cub


def clifford_difference(
    old: Sequence[tuple[int, int]], new: Sequence[int]
) -> tuple[dict[int, int], list[tuple[int, int]]] | None:
    """Exact check that ``f_old − f_new`` is a Clifford phase polynomial.

    Args:
        old: ``(column, sign)`` terms of the current polynomial.
        new: Columns of the candidate (all with sign ``+1``).

    Returns:
        ``(a, b)`` with ``f_old − f_new = 2·Σ a_i x_i + 4·Σ_{(i,j)∈b} x_i x_j``
        (``a_i`` in ``Z_4``: the power of ``S`` on variable ``i``; ``b``: the
        ``CZ`` pairs), or ``None`` if the difference is not Clifford.
    """
    lin, quad, cub = multilinear([(c, s) for c, s in old] + [(c, -1) for c in new])
    if (
        any(v for v in cub.values())
        or any(v % 4 for v in quad.values())
        or any(v % 2 for v in lin.values())
    ):
        return None
    return (
        {i: (v // 2) % 4 for i, v in lin.items() if v},
        sorted(pair for pair, v in quad.items() if v == 4),
    )


# ---------------------------------------------------------------- TODD
def normalise(cols: Iterable[int]) -> list[int]:
    """Drop zero columns and cancel equal pairs (two equal parities make a Clifford)."""
    seen: dict[int, None] = {}
    for c in cols:
        if c:
            if c in seen:
                del seen[c]
            else:
                seen[c] = None
    return list(seen)


def _nullspace(rows: Iterable[int], m: int) -> list[int]:
    """Basis of ``{y ∈ GF(2)^m : r·y = 0 for every row r}`` (bit masks)."""
    pivots: dict[int, int] = {}
    for r in rows:
        while r:
            p = r.bit_length() - 1
            if p not in pivots:
                pivots[p] = r
                break
            r ^= pivots[p]
    for p in sorted(pivots):
        row = pivots[p]
        for q in pivots:
            if q != p and (pivots[q] >> p) & 1:
                pivots[q] ^= row
    basis = []
    for f in range(m):
        if f in pivots:
            continue
        v = 1 << f
        for p, r in pivots.items():
            if (r >> f) & 1:
                v |= 1 << p
        basis.append(v)
    return basis


def _constraints(cols: Sequence[int], k: int, z: int) -> list[int]:
    """Rows whose null space is ``{y : A + z yᵀ keeps the cubic part}``."""
    rows = [sum(((c >> a) & 1) << j for j, c in enumerate(cols)) for a in range(k)]
    cons = list(rows)
    zs = [i for i in range(k) if (z >> i) & 1]
    in_z = [(z >> i) & 1 for i in range(k)]
    # pairs and triples without a variable of z give zero rows
    for i, j in combinations(range(k), 2):
        if in_z[i] or in_z[j]:
            cons.append((rows[j] if in_z[i] else 0) ^ (rows[i] if in_z[j] else 0))
    if zs:
        for i, j, m in combinations(range(k), 3):
            if in_z[i] or in_z[j] or in_z[m]:
                cons.append(
                    ((rows[j] & rows[m]) if in_z[i] else 0)
                    ^ ((rows[i] & rows[m]) if in_z[j] else 0)
                    ^ ((rows[i] & rows[j]) if in_z[m] else 0)
                )
    return cons


def _move(cols: Sequence[int], z: int, y: int) -> list[int]:
    new = [c ^ z if (y >> j) & 1 else c for j, c in enumerate(cols)]
    if y.bit_count() & 1:
        new.append(z)
    return normalise(new)


def _todd_once(cols: Sequence[int], k: int) -> list[int]:
    """Greedy TODD: the best move over every distinct ``z = c_a ⊕ c_b`` is taken."""
    cols = list(cols)
    while True:
        m = len(cols)
        if m < 4:
            return cols
        best: list[int] | None = None
        seen_z: set[int] = set()
        for a in range(m):
            for b in range(a + 1, m):
                z = cols[a] ^ cols[b]
                if z in seen_z:
                    continue
                seen_z.add(z)
                basis = _nullspace(_constraints(cols, k, z), m)
                head = basis[:12]
                # two-vector combinations widen the choice of separated pairs
                cands = basis + [
                    u ^ v for i, u in enumerate(head) for v in head[i + 1 :]
                ]
                for y in cands:
                    new = _move(cols, z, y)
                    if len(new) < m and (best is None or len(new) < len(best)):
                        best = new
        if best is None:
            return cols
        cols = best


def _two_distinct(rng: random.Random, m: int) -> tuple[int, int]:
    a = rng.randrange(m)
    b = rng.randrange(m - 1)
    return a, b + 1 if b >= a else b


def _todd_python(cols: Sequence[int], k: int, plateau: int, seed: int) -> list[int]:
    best = _todd_once(normalise(cols), k)
    if plateau <= 0 or len(best) < 4:
        return best
    rng = random.Random(seed)
    cur = list(best)
    for _ in range(plateau):
        m = len(cur)
        if m < 4:
            break
        a, b = _two_distinct(rng, m)
        z = cur[a] ^ cur[b] if rng.randrange(10) < 7 else cur[a]
        basis = _nullspace(_constraints(cur, k, z), m)
        if not basis:
            continue
        y = 0
        for v in basis:
            if rng.getrandbits(1):
                y ^= v
        if not y:
            y = basis[rng.randrange(len(basis))]
        new = _move(cur, z, y)
        if not new or len(new) > m:
            continue
        cur = _todd_once(new, k)
        if len(cur) < len(best):
            best = list(cur)
        elif len(cur) > len(best) + 1:
            cur = list(best)
    return best


def todd_batch(
    problems: Sequence[tuple[Sequence[int], int, int, int]], *, native: bool = True
) -> list[list[int]]:
    """TODD on several ``(columns, k, plateau, seed)`` problems (native: concurrent)."""
    run = _core_function("_tcount_todd_batch", native)
    if run is not None and all(k <= _NATIVE_MAX_VARS for _, k, _, _ in problems):
        return [list(r) for r in run([(list(c), k, p, s) for c, k, p, s in problems])]
    return [_todd_python(c, k, p, s) for c, k, p, s in problems]


def todd(
    cols: Sequence[int], k: int, *, plateau: int = 0, seed: int = 0, native: bool = True
) -> list[int]:
    """Reduce the number of columns (bit masks over ``k`` variables) by TODD.

    Greedy TODD to a local minimum, then up to ``plateau`` sideways moves (a
    random cubic-part-preserving ``A + z yᵀ`` that does not add columns,
    followed by greedy TODD again); the smallest set seen is returned.
    """
    return todd_batch([(cols, k, plateau, seed)], native=native)[0]


# ------------------------------------------------------------ Reed–Muller flat search
def _flat(base: int, dirs: Sequence[int]) -> set[int]:
    pts = {base}
    for d in dirs:
        pts |= {p ^ d for p in pts}
    pts.discard(0)
    return pts


def _independent(dirs: Iterable[int]) -> bool:
    piv: dict[int, int] = {}
    for d in dirs:
        while d:
            h = d.bit_length() - 1
            if h not in piv:
                piv[h] = d
                break
            d ^= piv[h]
        if not d:
            return False
    return True


def _pick_flat(cs: Sequence[int], rng: random.Random) -> set[int] | None:
    """Points of the 4-flat through five random columns (``None`` if dependent)."""
    picked: list[int] = []
    while len(picked) < 5:
        i = rng.randrange(len(cs))
        if i not in picked:
            picked.append(i)
    base = cs[picked[0]]
    dirs = [cs[i] ^ base for i in picked[1:]]
    if not _independent(dirs):
        return None
    return _flat(base, dirs)


def _best_flat_move(cs: list[int], rng: random.Random) -> list[int] | None:
    if len(cs) < 5:
        return None
    sset = set(cs)
    best: set[int] | None = None
    gain_best = 0
    for _ in range(_FLAT_SAMPLES):
        pts = _pick_flat(cs, rng)
        if pts is None:
            continue
        inside = len(pts & sset)
        gain = 2 * inside - len(pts)
        if gain > gain_best:
            best, gain_best = pts, gain
    return None if best is None else sorted(sset ^ best)


def _rm_python(cols: Sequence[int], k: int, iters: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    cur = _todd_python(cols, k, 0, 0)
    best = list(cur)
    for _ in range(iters):
        nxt = _best_flat_move(cur, rng)
        if nxt is not None:
            cur = _todd_python(nxt, k, 0, 0)
        else:
            if len(cur) < 5:
                break
            pts = None
            for _try in range(50):
                pts = _pick_flat(cur, rng)
                if pts is not None:
                    break
            if pts is None:
                continue
            cand = _todd_python(sorted(set(cur) ^ pts), k, 0, 0)
            d = len(cand) - len(cur)
            # kick accepted with probability 2^-d (an exact Metropolis rule)
            if d <= 0 or (d <= 60 and rng.getrandbits(d) == 0):
                cur = cand
        if len(cur) < len(best):
            best = list(cur)
    return best


def rm_search(
    problems: Sequence[tuple[Sequence[int], int, int, int]], *, native: bool = True
) -> list[list[int]]:
    """Reed–Muller flat search on several ``(columns, k, iterations, seed)`` problems.

    Each run starts with TODD; every step applies the best improving 4-flat
    among sampled ones followed by TODD, or — at a local minimum — a random
    flat kick accepted with probability ``2^{-loss}``.  Native runs are
    concurrent.
    """
    run = _core_function("_tcount_rm_search", native)
    if run is not None and all(k <= _NATIVE_MAX_VARS for _, k, _, _ in problems):
        return [
            list(r)
            for r in run([(list(c), k, i, s) for c, k, i, s in problems], _FLAT_SAMPLES)
        ]
    return [_rm_python(c, k, i, s) for c, k, i, s in problems]


# ------------------------------------------------------------ trilinear decompositions
def _bits(x: int) -> list[int]:
    return [i for i in range(x.bit_length()) if (x >> i) & 1]


def tripartition(
    triples: Sequence[tuple[int, int, int]], budget: int = 200000
) -> list[list[int]] | None:
    """Three classes with one variable of each in every cubic monomial, or ``None``."""
    adj: dict[int, list[tuple[int, int, int]]] = {}
    for t in triples:
        for v in t:
            adj.setdefault(v, []).append(t)
    # order: every variable (but the first of a component) shares a monomial
    # with an earlier one
    order: list[int] = []
    seen: set[int] = set()
    for root in sorted(adj):
        if root in seen:
            continue
        seen.add(root)
        queue = [root]
        while queue:
            v = queue.pop(0)
            order.append(v)
            for t in adj[v]:
                for u in t:
                    if u not in seen:
                        seen.add(u)
                        queue.append(u)
    colour: dict[int, int] = {}
    steps = 0

    def ok(v: int) -> bool:
        for t in adj[v]:
            cs = [colour[x] for x in t if x in colour]
            if len(cs) != len(set(cs)):
                return False
        return True

    def dfs(i: int) -> bool:
        nonlocal steps
        if i == len(order):
            return True
        v = order[i]
        for c in range(3):
            steps += 1
            if steps > budget:
                return False
            colour[v] = c
            if ok(v) and dfs(i + 1):
                return True
            del colour[v]
        return False

    if not order or len(order) > 900 or not dfs(0):
        return None
    return [[v for v in sorted(colour) if colour[v] == c] for c in range(3)]


def local_tensor(
    triples: Iterable[tuple[int, int, int]], parts: Sequence[Sequence[int]]
) -> set[tuple[int, int, int]]:
    pos = [{v: i for i, v in enumerate(p)} for p in parts]
    out: set[tuple[int, int, int]] = set()
    for t in triples:
        a = next(x for x in t if x in pos[0])
        b = next(x for x in t if x in pos[1])
        c = next(x for x in t if x in pos[2])
        out ^= {(pos[0][a], pos[1][b], pos[2][c])}
    return out


def tensor_of(dec: Iterable[Term]) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for u, v, w in dec:
        for i in _bits(u):
            for j in _bits(v):
                for k in _bits(w):
                    out ^= {(i, j, k)}
    return out


def naive_decomposition(tensor: Iterable[tuple[int, int, int]]) -> list[Term]:
    return [(1 << i, 1 << j, 1 << k) for i, j, k in sorted(tensor)]


def _karatsuba_poly(m: int) -> list[Term]:
    """Rank decomposition of the product of two ``m``-term polynomials over GF(2)."""
    if m == 1:
        return [(1, 1, 1)]
    if m == 2:
        return [(1, 1, 0b011), (2, 2, 0b110), (3, 3, 0b010)]
    if m == 3:  # a_i b_i and (a_i + a_j)(b_i + b_j): rank 6
        return [
            (1, 1, 0b00111),
            (2, 2, 0b01110),
            (4, 4, 0b11100),
            (3, 3, 0b00010),
            (5, 5, 0b00100),
            (6, 6, 0b01000),
        ]
    h = (m + 1) // 2
    lo = m - h

    def fold(x: int) -> int:  # coefficient i of A0 + A1 is a_i + a_{i+h}
        out = 0
        for i in _bits(x):
            out |= 1 << i
            if i + h < m:
                out |= 1 << (i + h)
        return out

    dec = [(u, v, w ^ (w << h)) for u, v, w in _karatsuba_poly(h)]  # A0 B0
    dec += [
        (u << h, v << h, (w << 2 * h) ^ (w << h)) for u, v, w in _karatsuba_poly(lo)
    ]  # A1 B1
    dec += [
        (fold(u), fold(v), w << h) for u, v, w in _karatsuba_poly(h)
    ]  # (A0 + A1)(B0 + B1)
    return dec


def karatsuba_decomposition(
    tensor: set[tuple[int, int, int]], m: int
) -> list[Term] | None:
    """Karatsuba decomposition when ``T(e_i, e_j)`` depends only on ``i + j``."""
    for ra in (False, True):
        for rb in (False, True):
            col: dict[int, int] = {}
            consistent = True
            for i in range(m):
                for j in range(m):
                    ii = m - 1 - i if ra else i
                    jj = m - 1 - j if rb else j
                    w = 0
                    for k in range(m):
                        if (ii, jj, k) in tensor:
                            w |= 1 << k
                    if col.setdefault(i + j, w) != w:
                        consistent = False
            if not consistent:
                continue
            dec = []
            for u, v, w in _karatsuba_poly(m):
                z = 0
                for d in _bits(w):
                    z ^= col[d]
                uu = sum(1 << (m - 1 - i) for i in _bits(u)) if ra else u
                vv = sum(1 << (m - 1 - i) for i in _bits(v)) if rb else v
                dec.append((uu, vv, z))
            dec = _reduce(dec)
            if tensor_of(dec) == tensor:
                return dec
    return None


def _reduce(dec: list[Term]) -> list[Term]:
    """Drop terms with a zero factor; merge terms sharing two factors (fixed point)."""
    changed = True
    while changed:
        changed = False
        dec = [t for t in dec if t[0] and t[1] and t[2]]
        seen: dict[tuple[int, int, int], int] = {}
        for idx, t in enumerate(dec):
            for p in range(3):
                key = (p, t[(p + 1) % 3], t[(p + 2) % 3])
                j = seen.get(key)
                if j is not None:
                    s = list(dec[j])
                    s[p] ^= t[p]
                    dec[j] = (s[0], s[1], s[2])
                    dec.pop(idx)
                    changed = True
                    break
                seen[key] = idx
            if changed:
                break
    return dec


def _flip_walk_python(start: Sequence[Term], steps: int, seed: int) -> list[Term]:
    rng = random.Random(seed)
    dec = _reduce(list(start))
    best = list(dec)
    since = 0
    for _ in range(steps):
        p = rng.randrange(3)
        groups: dict[int, list[int]] = {}
        for i, t in enumerate(dec):
            groups.setdefault(t[p], []).append(i)
        cands = [g for g in groups.values() if len(g) >= 2]
        if not cands or since > _PLUS_AFTER:
            if len(dec) >= 2:
                i, j = _two_distinct(rng, len(dec))
                (a, b, c), (a2, b2, c2) = dec[i], dec[j]
                new = [(a ^ a2, b, c), (a2, b ^ b2, c), (a2, b2, c ^ c2)]
                dec = _reduce(
                    [t for idx, t in enumerate(dec) if idx != i and idx != j] + new
                )
            since = 0
            continue
        g = cands[rng.randrange(len(cands))]
        gi, gj = _two_distinct(rng, len(g))
        i, j = g[gi], g[gj]
        q, r = (p + 1) % 3, (p + 2) % 3
        if rng.getrandbits(1):
            q, r = r, q
        s, t2 = list(dec[i]), list(dec[j])
        # x⊗a_s⊗b_s + x⊗a_t⊗b_t = x⊗(a_s + a_t)⊗b_s + x⊗a_t⊗(b_t + b_s)
        s[q] ^= t2[q]
        t2[r] ^= s[r]
        dec[i], dec[j] = (s[0], s[1], s[2]), (t2[0], t2[1], t2[2])
        n0 = len(dec)
        dec = _reduce(dec)
        since = 0 if len(dec) < n0 else since + 1
        if len(dec) < len(best):
            best = list(dec)
    return best


def flip_walks(
    problems: Sequence[tuple[Sequence[Term], int, int]], *, native: bool = True
) -> list[list[Term]]:
    """Flip-graph walks ``(start, steps, seed)``; the lowest rank seen by each."""
    run = _core_function("_tcount_flip_walks", native)
    if run is not None and all(
        max(t) < (1 << 64) for start, _, _ in problems for t in start
    ):
        return [
            [(u, v, w) for u, v, w in r]
            for r in run([(list(d), s, e) for d, s, e in problems], _PLUS_AFTER)
        ]
    return [_flip_walk_python(d, s, e) for d, s, e in problems]


def parity_set(dec: Iterable[Term], parts: Sequence[Sequence[int]]) -> set[int]:
    """Columns of the CCZ-on-parities form of a decomposition; equal pairs cancel."""

    def glob(x: int, part: Sequence[int]) -> int:
        m = 0
        for i in _bits(x):
            m |= 1 << part[i]
        return m

    cols: set[int] = set()
    for u, v, w in dec:
        x, y, z = glob(u, parts[0]), glob(v, parts[1]), glob(w, parts[2])
        for p in (x, y, z, x ^ y, x ^ z, y ^ z, x ^ y ^ z):
            cols ^= {p}
    return cols


def with_residual(old: Sequence[tuple[int, int]], cols: set[int]) -> list[int] | None:
    """Add CS / T columns so that ``f_old − f_cols`` is Clifford, or ``None``."""
    lin, quad, cub = multilinear([(c, s) for c, s in old] + [(c, -1) for c in cols])
    if any(cub.values()):
        return None
    extra: set[int] = set()
    for (i, j), v in quad.items():
        if v % 4:  # 2 mod 4: a CS on (i, j) = parities i, j, i ⊕ j
            for p in (1 << i, 1 << j, (1 << i) | (1 << j)):
                extra ^= {p}
            lin[i] = lin.get(i, 0) + 2
            lin[j] = lin.get(j, 0) + 2
    for i, v in lin.items():
        if v % 2:
            extra ^= {1 << i}
    return sorted(cols ^ extra)


def tensor_starts(
    old: Sequence[tuple[int, int]],
    *,
    walks: int,
    steps: int,
    seed: int,
    native: bool = True,
) -> list[list[int]]:
    """Equivalent column sets from low-rank decompositions of a trilinear cubic part.

    Returns an empty list when the cubic part is not trilinear in a
    tripartition of the variables.  Every returned set has passed
    :func:`clifford_difference` against ``old``.
    """
    cub = multilinear(old)[2]
    triples = sorted(t for t, v in cub.items() if v)
    if not triples or walks <= 0:
        return []
    parts = tripartition(triples)
    if parts is None:
        return []
    tensor = local_tensor(triples, parts)
    starts = [naive_decomposition(tensor)]
    if len(parts[0]) == len(parts[1]) == len(parts[2]):
        kar = karatsuba_decomposition(tensor, len(parts[0]))
        if kar is not None:
            starts.append(kar)
    decs = flip_walks(
        [(starts[i % len(starts)], steps, seed * 1000003 + i) for i in range(walks)],
        native=native,
    )
    decs = [d for d in decs if tensor_of(d) == tensor]
    if not decs:
        return []
    rmin = min(len(d) for d in decs)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for d in decs:
        if len(d) > rmin + 1:
            continue
        cols = with_residual(old, parity_set(d, parts))
        if (
            cols is None
            or tuple(cols) in seen
            or clifford_difference(old, cols) is None
        ):
            continue
        seen.add(tuple(cols))
        out.append(cols)
    return out
