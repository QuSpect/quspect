"""Exact qubit layout by VF2++-style subgraph monomorphism search.

A circuit can be executed without any routing SWAP exactly when its
*interaction graph* (logical qubits, with an edge for every pair that shares
a two-qubit gate) has a **monomorphism** into the device coupling graph: an
injective map ``logical -> physical`` that sends every interaction edge to a
coupling edge.  Extra coupling edges between mapped qubits are allowed, so
this is *not* an induced-subgraph isomorphism test.

The search follows the ideas of

    Cordella, Foggia, Sansone and Vento. "A (sub)graph isomorphism algorithm
    for matching large graphs." IEEE TPAMI 26(10), 2004 (VF2).

    Jüttner and Madarasi. "VF2++ — An improved subgraph isomorphism
    algorithm." Discrete Applied Mathematics 242, 2018.

* **Fast rejection**: too many nodes or edges, a degree sequence that the
  coupling graph cannot dominate, or an odd cycle (non-bipartite interaction
  graph) when the coupling graph is bipartite (line, even ring, grid and
  heavy-hex all are).
* **Matching order** (VF2++): per connected component, breadth-first from
  the node of maximum degree; inside a BFS level the node with the most
  already-ordered neighbours (then the highest degree) goes first.
* **Candidates**: a node whose earlier-ordered neighbours are already mapped
  can only go to a free physical neighbour of the image of one of them that
  is adjacent to the images of all of them, has enough degree and enough
  free neighbours for the node's still-unmatched neighbours (look-ahead).
  Candidates are tried in Warnsdorff order (fewest free neighbours first),
  which finds long chain embeddings in grids and heavy-hex lattices quickly.
* **Budget**: the number of explored states (candidate assignments) is
  bounded, so hard negative instances return "not found" instead of
  hanging.  An optional wall-clock limit is available as a second guard;
  only the state budget keeps the result machine independent.

The search is pure Python and deterministic.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_VF2_MAX_STATES = 100_000
"""Default state budget of :func:`find_subgraph_monomorphism`."""

_TIME_CHECK_INTERVAL = 256


def find_subgraph_monomorphism(
    num_pattern: int,
    pattern_edges: Iterable[tuple[int, int]],
    target_adjacency: Sequence[Sequence[int]],
    *,
    max_states: int = DEFAULT_VF2_MAX_STATES,
    time_limit: float | None = None,
) -> list[int] | None:
    """Find an injective edge-preserving map of a pattern graph into a target.

    Every pattern node that has at least one edge is mapped to a distinct
    target node such that every pattern edge ``(u, v)`` becomes a target
    edge ``(m[u], m[v])``.  Pattern nodes without edges are left unmapped
    (``-1``) so the caller can place them freely.

    Args:
        num_pattern: Number of pattern nodes (``0 .. num_pattern - 1``).
        pattern_edges: Undirected pattern edges; duplicates and both
            orientations are allowed, self loops are rejected.
        target_adjacency: Adjacency lists of the undirected target graph.
        max_states: Maximum number of explored candidate assignments
            (at least 1).  The search gives up (returns ``None``) once the
            budget is spent.
        time_limit: Optional wall-clock limit in seconds.  ``None`` (the
            default) keeps the result fully deterministic.

    Returns:
        ``mapping`` with ``mapping[u]`` the target node of pattern node
        ``u`` (``-1`` for edgeless pattern nodes), or ``None`` if no
        monomorphism exists or none was found within the budget.

    Raises:
        ValueError: If an edge endpoint is out of range, an edge is a self
            loop, or ``max_states`` is smaller than 1.

    Example:
        >>> line = [[1], [0, 2], [1, 3], [2]]
        >>> find_subgraph_monomorphism(3, [(0, 1), (1, 2)], line)
        [0, 1, 2]
        >>> find_subgraph_monomorphism(3, [(0, 1), (1, 2), (0, 2)], line) is None
        True
    """
    if max_states < 1:
        raise ValueError("max_states must be at least 1.")
    padj: list[set[int]] = [set() for _ in range(num_pattern)]
    for u, v in pattern_edges:
        if not (0 <= u < num_pattern and 0 <= v < num_pattern):
            raise ValueError(f"Pattern edge ({u}, {v}) is out of range.")
        if u == v:
            raise ValueError(f"Pattern edge ({u}, {v}) is a self loop.")
        padj[u].add(v)
        padj[v].add(u)

    num_target = len(target_adjacency)
    tadj: list[set[int]] = [set(neighbors) for neighbors in target_adjacency]
    tlist: list[list[int]] = [sorted(neighbors) for neighbors in tadj]
    tdeg = [len(neighbors) for neighbors in tadj]
    pdeg = [len(neighbors) for neighbors in padj]

    active = [u for u in range(num_pattern) if pdeg[u] > 0]
    mapping = [-1] * num_pattern
    if not active:
        return mapping
    if _quick_reject(active, padj, pdeg, tadj, tdeg, num_target):
        return None

    order = _matching_order(active, padj, pdeg)
    position = {u: i for i, u in enumerate(order)}
    back: list[list[int]] = []
    for u in order:
        earlier = [v for v in padj[u] if position[v] < position[u]]
        earlier.sort(key=lambda v: position[v])
        back.append(earlier)
    roots_by_degree = sorted(range(num_target), key=lambda p: (tdeg[p], p))

    used = [False] * num_target
    free_deg = list(tdeg)  # number of unused neighbours of each target node

    def candidates(level: int) -> list[int]:
        u = order[level]
        need = pdeg[u]
        earlier = back[level]
        unmatched = need - len(earlier)
        if not earlier:
            return [
                p
                for p in roots_by_degree
                if not used[p] and tdeg[p] >= need and free_deg[p] >= unmatched
            ]
        images = [mapping[v] for v in earlier]
        anchor = min(images, key=lambda p: tdeg[p])
        others = [p for p in images if p != anchor]
        result = [
            p
            for p in tlist[anchor]
            if not used[p]
            and tdeg[p] >= need
            and free_deg[p] >= unmatched
            and all(q in tadj[p] for q in others)
        ]
        result.sort(key=lambda p: (free_deg[p], p))
        return result

    def assign(u: int, p: int) -> None:
        mapping[u] = p
        used[p] = True
        for z in tlist[p]:
            free_deg[z] -= 1

    def unassign(u: int) -> None:
        p = mapping[u]
        mapping[u] = -1
        used[p] = False
        for z in tlist[p]:
            free_deg[z] += 1

    deadline = None if time_limit is None else time.perf_counter() + time_limit
    depth = len(order)
    stack: list[list[int]] = [candidates(0)]
    cursor = [0]
    states = 0
    level = 0
    while True:
        u = order[level]
        if mapping[u] != -1:
            unassign(u)
        cands = stack[level]
        if cursor[level] >= len(cands):
            stack.pop()
            cursor.pop()
            level -= 1
            if level < 0:
                logger.debug(
                    "VF2: no monomorphism (search exhausted, %d states)", states
                )
                return None
            continue
        p = cands[cursor[level]]
        cursor[level] += 1
        states += 1
        if states > max_states:
            logger.debug("VF2: state budget %d exhausted", max_states)
            return None
        if (
            deadline is not None
            and states % _TIME_CHECK_INTERVAL == 0
            and time.perf_counter() > deadline
        ):
            logger.debug("VF2: time limit %.3fs exceeded", time_limit)
            return None
        assign(u, p)
        level += 1
        if level == depth:
            logger.debug("VF2: monomorphism found after %d states", states)
            return mapping
        stack.append(candidates(level))
        cursor.append(0)


def _quick_reject(
    active: list[int],
    padj: list[set[int]],
    pdeg: list[int],
    tadj: list[set[int]],
    tdeg: list[int],
    num_target: int,
) -> bool:
    """Cheap necessary conditions; ``True`` means "certainly no embedding"."""
    if len(active) > num_target:
        return True
    if sum(pdeg) > sum(tdeg):
        return True
    pattern_degrees = sorted((pdeg[u] for u in active), reverse=True)
    target_degrees = sorted(tdeg, reverse=True)
    if any(d > t for d, t in zip(pattern_degrees, target_degrees, strict=False)):
        return True
    return _is_bipartite(range(num_target), tadj) and not _is_bipartite(active, padj)


def _is_bipartite(nodes: Iterable[int], adjacency: Sequence[set[int]]) -> bool:
    """2-colourability of the subgraph spanned by ``nodes`` (all components)."""
    colour: dict[int, int] = {}
    for start in nodes:
        if start in colour:
            continue
        colour[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in colour:
                    colour[neighbor] = 1 - colour[node]
                    queue.append(neighbor)
                elif colour[neighbor] == colour[node]:
                    return False
    return True


def _matching_order(
    active: list[int], padj: list[set[int]], pdeg: list[int]
) -> list[int]:
    """VF2++ node order: BFS levels, most constrained node first per level."""
    order: list[int] = []
    conn = dict.fromkeys(active, 0)  # neighbours already placed in ``order``
    remaining = set(active)
    while remaining:
        root = max(remaining, key=lambda u: (pdeg[u], -u))
        level = [root]
        seen = {root}
        while level:
            pending = list(level)
            start = len(order)
            while pending:
                best = max(pending, key=lambda u: (conn[u], pdeg[u], -u))
                pending.remove(best)
                order.append(best)
                for v in padj[best]:
                    conn[v] += 1
            next_level: list[int] = []
            for u in order[start:]:
                for v in sorted(padj[u]):
                    if v not in seen:
                        seen.add(v)
                        next_level.append(v)
            level = next_level
        remaining -= seen
    return order
