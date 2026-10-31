"""Low-dilation initial layouts: the interaction graph embedded in the device.

SABRE's layout search starts from random permutations and improves them by
forward/backward routing.  That works well for dense, time-varying
interaction patterns, but it is poor at *embedding* a large sparse
interaction graph — a CX chain (GHZ preparation), a nearest-neighbour
Trotter chain, an adder's ladder — into a sparse device when no exact
(SWAP-free) embedding exists: the refinement only moves qubits locally, so
a random start keeps its long-range defects (a 50-qubit chain on a
57-qubit heavy-hex lattice routes with ~35 SWAPs from the best of 26
SABRE starts, while one SWAP suffices).

This module computes starting layouts that minimize the **weighted
dilation** of the interaction graph ``H`` (edge weight ``w_ij`` = number
of routing nodes on the pair) in the coupling graph ``G``::

    cost(π) = Σ_{ij ∈ E(H)} w_ij · (d_G(π(i), π(j)) − 1)

— every unit of dilation of an edge is (at least) one SWAP that a static
layout forces on that pair.  Minimizing it is a Koopmans–Beckmann
quadratic assignment problem (flow ``W``, distance ``D − 1``) and contains
subgraph isomorphism / Hamiltonian path as the zero-cost case, so it is
NP-hard; two classical ideas give good solutions fast:

1. **Greedy growth with Warnsdorff's rule.**  Logical qubits are placed in
   *maximum-adjacency order* (the next qubit is the one most strongly
   connected to the placed ones, as in Prim's algorithm or a weighted
   Cuthill–McKee ordering), each on the free physical qubit with the
   smallest added dilation.  Ties go to the physical qubit with the
   *fewest* free neighbours — Warnsdorff's rule, the classic heuristic
   for Hamiltonian paths (knight's tours): it walks along the boundary of
   the free region instead of cutting it into pieces — unless that leaves
   fewer free neighbours than the qubit has unplaced partners (a capacity
   penalty that keeps room for the branches of trees and stars).  Several
   starts (peripheral / low-degree / central qubits of ``H`` and of ``G``,
   found by a double BFS sweep as in the Gibbs–Poole–Stockmeyer
   bandwidth heuristic) are tried.
2. **Simulated annealing on the QAP.**  The best greedy layouts are
   improved by 2-exchange moves (a logical qubit trades places with
   whatever occupies a physical qubit — another logical qubit or a free
   one), proposed mostly next to one of the qubit's partners, evaluated
   in ``O(degree)`` and accepted by the Metropolis rule under a linearly
   cooling temperature.

The layouts are *starts*: SABRE routes each one unrefined and refined,
and its usual selection (fewest SWAPs) decides, so a poor embedding (e.g.
for a dense, time-varying circuit) only costs its routing time.

Everything is integer arithmetic plus a SplitMix64 random source, and a
C++ implementation (``cpp/src/transpiler/dilation_layout.cpp``) gives
bit-identical results; :func:`dilation_layouts` uses it when the native
extension is built.
"""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Sequence
from typing import Any

_MASK64 = (1 << 64) - 1
#: Greedy starts per side (logical × physical); the best ``_SA_STARTS``
#: distinct greedy layouts are annealed.
_SA_STARTS = 3
#: Annealing moves per interacting logical qubit (and per greedy layout).
_SA_MOVES_PER_QUBIT = 400
#: Probability (in 1/1024) that a move targets a neighbour of a partner.
_LOCAL_MOVE = 820


class _SplitMix64:
    """SplitMix64 (Steele, Lea & Flood 2014); mirrored in C++."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Integer in ``[0, n)`` (plain modulo; the bias is irrelevant here)."""
        return self.next() % n

    def unit(self) -> float:
        """Float in ``[0, 1)`` with 53 random bits."""
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)


def _bfs_far(adj: Sequence[Sequence[int]], start: int, allowed: Sequence[bool]) -> int:
    """Last vertex reached by a BFS from ``start`` (smallest index on ties)."""
    seen = {start}
    queue = deque([start])
    last = start
    while queue:
        u = queue.popleft()
        last = u
        for v in adj[u]:
            if allowed[v] and v not in seen:
                seen.add(v)
                queue.append(v)
    return last


def _pseudo_peripheral(adj: Sequence[Sequence[int]], start: int) -> int:
    """Endpoint of a double BFS sweep (Gibbs–Poole–Stockmeyer style)."""
    allowed = [True] * len(adj)
    return _bfs_far(adj, _bfs_far(adj, start, allowed), allowed)


def _eccentric_centre(dist: Sequence[Sequence[int]], nodes: Sequence[int]) -> int:
    """Vertex of smallest eccentricity (smallest index on ties)."""
    return min(nodes, key=lambda p: (max(dist[p][q] for q in nodes), p))


def _starts(
    nbr: list[list[tuple[int, int]]],
    adj: Sequence[Sequence[int]],
    dist: Sequence[Sequence[int]],
    active: list[int],
) -> tuple[list[int], list[int]]:
    """Logical and physical start vertices of the greedy growth."""
    wdeg = {q: sum(w for _, w in nbr[q]) for q in active}
    log_adj = [[o for o, _ in nbr[q]] for q in range(len(nbr))]
    logical: list[int] = []
    for q in (
        _pseudo_peripheral(log_adj, active[0]),
        min(active, key=lambda q: (wdeg[q], q)),
        max(active, key=lambda q: (wdeg[q], -q)),
    ):
        if q not in logical:
            logical.append(q)
    n_phys = len(adj)
    nodes = list(range(n_phys))
    physical: list[int] = []
    for p in (
        _pseudo_peripheral(adj, 0),
        min(nodes, key=lambda p: (len(adj[p]), p)),
        _eccentric_centre(dist, nodes),
        max(nodes, key=lambda p: (len(adj[p]), -p)),
    ):
        if p not in physical:
            physical.append(p)
    return logical, physical


def _greedy(
    num_logical: int,
    nbr: list[list[tuple[int, int]]],
    adj: Sequence[Sequence[int]],
    dist: Sequence[Sequence[int]],
    active: list[int],
    start_logical: int,
    start_physical: int,
) -> list[int]:
    """Maximum-adjacency growth with Warnsdorff tie-breaking (see module doc)."""
    n_phys = len(adj)
    pos = [-1] * num_logical
    free = [True] * n_phys
    free_deg = [len(adj[p]) for p in range(n_phys)]
    open_deg = [len(nbr[q]) for q in range(num_logical)]  # unplaced partners
    conn = [0] * num_logical
    wdeg = [sum(w for _, w in nbr[q]) for q in range(num_logical)]

    def place(q: int, p: int) -> None:
        pos[q] = p
        free[p] = False
        for r in adj[p]:
            free_deg[r] -= 1
        for o, w in nbr[q]:
            open_deg[o] -= 1
            conn[o] += w

    place(start_logical, start_physical)
    remaining = [q for q in active if q != start_logical]
    while remaining:
        best_i = 0
        for i in range(1, len(remaining)):
            q, b = remaining[i], remaining[best_i]
            if (conn[q], wdeg[q], -q) > (conn[b], wdeg[b], -b):
                best_i = i
        q = remaining.pop(best_i)
        best_key: tuple[int, int, int] | None = None
        best_p = -1
        if conn[q] == 0:
            # 新的相互作用连通分量：放在空闲邻居最多的空闲比特上
            for p in range(n_phys):
                if free[p]:
                    key = (-free_deg[p], 0, p)
                    if best_key is None or key < best_key:
                        best_key, best_p = key, p
        else:
            placed = [(pos[o], w) for o, w in nbr[q] if pos[o] >= 0]
            need = open_deg[q]
            for p in range(n_phys):
                if not free[p]:
                    continue
                row = dist[p]
                cost = 0
                for po, w in placed:
                    cost += w * (row[po] - 1)
                if best_key is not None and cost > best_key[0]:
                    continue
                short = need - free_deg[p]
                key = (cost + (short if short > 0 else 0), free_deg[p], p)
                if best_key is None or key < best_key:
                    best_key, best_p = key, p
        place(q, best_p)
    return pos


def _anneal(
    num_logical: int,
    nbr: list[list[tuple[int, int]]],
    adj: Sequence[Sequence[int]],
    dist: Sequence[Sequence[int]],
    active: list[int],
    pos: list[int],
    rng: _SplitMix64,
    moves: int,
    t0: float,
) -> tuple[int, list[int]]:
    """Simulated annealing with 2-exchange moves; returns the best layout."""
    n_phys = len(adj)
    occ = [-1] * n_phys
    cur_pos = list(pos)
    for q in active:
        occ[cur_pos[q]] = q

    def part(t: int, p: int, excl: int) -> int:
        if t < 0:
            return 0
        row = dist[p]
        s = 0
        for o, w in nbr[t]:
            if o != excl:
                s += w * (row[cur_pos[o]] - 1)
        return s

    cur = _cost_active(nbr, dist, cur_pos, active)
    best, best_pos = cur, list(cur_pos)
    n_active = len(active)
    for k in range(moves):
        t = active[rng.below(n_active)]
        p = cur_pos[t]
        edges = nbr[t]
        if rng.below(1024) < _LOCAL_MOVE:
            o = edges[rng.below(len(edges))][0]
            around = adj[cur_pos[o]]
            q = around[rng.below(len(around))]
        else:
            q = rng.below(n_phys)
        if q == p:
            continue
        u = occ[q]
        delta = part(t, q, u) + part(u, p, t) - part(t, p, u) - part(u, q, t)
        if delta > 0:
            temp = t0 * float(moves - k) / float(moves)
            if not rng.unit() < math.exp(-float(delta) / temp):
                continue
        cur_pos[t] = q
        occ[q] = t
        occ[p] = u
        if u >= 0:
            cur_pos[u] = p
        cur += delta
        if cur < best:
            best, best_pos = cur, list(cur_pos)
    return best, best_pos


def _cost_active(
    nbr: list[list[tuple[int, int]]],
    dist: Sequence[Sequence[int]],
    pos: list[int],
    active: list[int],
) -> int:
    total = 0
    for a in active:
        for b, w in nbr[a]:
            if a < b:
                total += w * (dist[pos[a]][pos[b]] - 1)
    return total


def dilation_layouts_python(
    num_logical: int,
    edges: Sequence[tuple[int, int, int]],
    adjacency: Sequence[Sequence[int]],
    distance: Sequence[Sequence[int]],
    seed: int,
    count: int,
) -> list[tuple[int, list[int]]]:
    """Pure-Python reference implementation of :func:`dilation_layouts`."""
    nbr: list[list[tuple[int, int]]] = [[] for _ in range(num_logical)]
    for a, b, w in edges:
        nbr[a].append((b, w))
        nbr[b].append((a, w))
    for row in nbr:
        row.sort()
    active = [q for q in range(num_logical) if nbr[q]]
    if not active:
        return []
    logical, physical = _starts(nbr, adjacency, distance, active)
    greedy: list[tuple[int, list[int]]] = []
    for sl in logical:
        for sp in physical:
            pos = _greedy(num_logical, nbr, adjacency, distance, active, sl, sp)
            greedy.append((_cost_active(nbr, distance, pos, active), pos))
    greedy = _distinct_best(greedy, _SA_STARTS)
    total_w = sum(w for _, _, w in edges)
    t0 = max(1.0, float(total_w) / float(len(edges)))
    moves = _SA_MOVES_PER_QUBIT * len(active)
    rng = _SplitMix64(seed)
    results = list(greedy)
    for _, pos in greedy:
        results.append(
            _anneal(num_logical, nbr, adjacency, distance, active, pos, rng, moves, t0)
        )
    return _distinct_best(results, count)


def _distinct_best(
    items: list[tuple[int, list[int]]], count: int
) -> list[tuple[int, list[int]]]:
    """The ``count`` cheapest distinct layouts (stable on ties)."""
    out: list[tuple[int, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for cost, pos in sorted(items, key=lambda item: item[0]):
        key = tuple(pos)
        if key in seen:
            continue
        seen.add(key)
        out.append((cost, pos))
        if len(out) == count:
            break
    return out


def dilation_layouts(
    num_logical: int,
    edges: Sequence[tuple[int, int, int]],
    adjacency: Sequence[Sequence[int]],
    distance: Sequence[Sequence[int]],
    *,
    seed: int,
    count: int = 1,
) -> list[tuple[int, list[int]]]:
    """Layouts of small weighted dilation of an interaction graph.

    Args:
        num_logical: Number of logical qubits.
        edges: Interaction graph as ``(a, b, weight)`` with ``a < b``,
            distinct pairs and positive integer weights.
        adjacency: Coupling graph adjacency lists (sorted), connected.
        distance: All-pairs shortest-path distances of the coupling graph.
        seed: Seed of the annealing random source.
        count: Number of distinct layouts to return (best first).

    Returns:
        ``(cost, layout)`` pairs, best first: ``layout[q]`` is the physical
        qubit of interacting logical qubit ``q`` and ``-1`` for logical
        qubits without interactions; ``cost`` is the weighted dilation
        ``Σ w · (d − 1)``.  Empty without edges.

    Example:
        >>> adj = [[1], [0, 2], [1, 3], [2]]  # a 4-qubit line
        >>> dist = [[abs(i - j) for j in range(4)] for i in range(4)]
        >>> chain = [(0, 1, 1), (1, 2, 1)]
        >>> cost, layout = dilation_layouts(3, chain, adj, dist, seed=0)[0]
        >>> cost, layout in ([0, 1, 2], [1, 2, 3], [3, 2, 1], [2, 1, 0])
        (0, True)
    """
    core = _native_core()
    if core is not None:
        result: list[tuple[int, list[int]]] = core._dilation_layouts(
            num_logical,
            [tuple(e) for e in edges],
            [list(a) for a in adjacency],
            [list(r) for r in distance],
            seed & _MASK64,
            count,
        )
        return result
    return dilation_layouts_python(num_logical, edges, adjacency, distance, seed, count)


def _native_core() -> Any | None:
    """The loaded extension module if it provides the dilation layout."""
    from quspect._native_loader import load_core
    from quspect.transpiler._routing import ROUTING_BACKEND_ENV

    if os.environ.get(ROUTING_BACKEND_ENV) == "python":
        return None

    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        return None
    if core is None or not hasattr(core, "_dilation_layouts"):
        return None
    return core
