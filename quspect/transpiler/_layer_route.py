"""Layered routing of SU(4)-block circuits on a path (quantum-volume style).

When every two-qubit block of a circuit is a generic SU(4) (its mirror cost
``cz(U · SWAP) - cz(U)`` is 0), a SWAP right before or after a block on the
same pair is free: block and SWAP resynthesize into one two-qubit unitary.
On a path (line coupling graph) the routing problem then has an exact
combinatorial form.  Split the blocks into dependency layers (each a
matching of the qubits).  If all blocks of layer ``l`` execute while their
qubits sit next to each other, the arrangement ``σ_l`` is determined by the
*order of the blocks* only (the order inside a block is free), and the
fewest paid SWAPs between consecutive layers is their Kendall-τ distance
(inversion count, the bubble-sort distance)::

    d(σ_{l-1}, σ_l) = #{(u, v) in different blocks of both σ_{l-1} and σ_l,
                        ordered differently}.

Exhaustive search on small instances shows that this *aligned* schedule
loses nothing against arbitrary asynchronous routings, so routing becomes
the optimization of a chain of ordered partitions,
``min Σ_l d(σ_{l-1}, σ_l)`` — a Kemeny-median-like problem.  With both
neighbours fixed, re-ordering one layer is a linear ordering problem (LOP)
with weights ``w(A, B)`` = inversions against both neighbours if block
``A`` precedes ``B``; it is solved by insertion local search inside block
coordinate descent over the chain, followed by iterated local search
(random kicks of a few consecutive layers, local descent, keep if better).
The SWAPs between layers come from odd-even transposition sort, which swaps
only inverted neighbours (so the count is exactly the inversion number) in
at most ``n`` rounds; the free flips inside the previous layer's blocks are
emitted right after those blocks, where block consolidation absorbs them.

On QV-100 on a 100-qubit line this needs about 30 % fewer CZ than SABRE.

On a *ring* the same chain model holds with the cyclic analogue of the
Kendall-τ distance: lift the ring to its universal cover (the integer line,
qubit ``t`` has a copy at ``x_t + k·n`` for every ``k``) and give every qubit
a lifted displacement ``δ_t ≡ y_t - x_t (mod n)`` with ``Σ δ_t = 0`` (a SWAP
moves one qubit by +1 and one by -1).  For given lifts the fewest adjacent
transpositions is the number of crossings of lifted copies (Jerrum 1985)::

    C(δ) = Σ_{t<u} |⌊(x_t + δ_t - x_u - δ_u) / n⌋ - ⌊(x_t - x_u) / n⌋|,

realized exactly by a periodic odd-even transposition sort.  With the
orientations inside the blocks fixed, the optimal lifts follow a threshold
rule (the ``K = Σ d_t / n`` qubits of largest forward displacement ``d_t``
go backwards); the orientations and each layer's cyclic block order and
rotation are chosen by local search (see :func:`layer_route_events`).

On every other coupling graph (grids, heavy-hex, ...) no such exact chain
model is known: the aligned schedule turns each transition into a token
swapping problem, and on QV-100 / grid 10x10 even good token-swapping
heuristics need about 1.7 times the half total displacement D/2 of good
placement chains, more SWAPs than SABRE.  Instead
:func:`su4_route_events` routes asynchronously with a *truncated rollout*
of SABRE's greedy policy (approximate dynamic programming): every
candidate SWAP is scored by its SABRE cost plus the best SABRE cost one
more SWAP ahead (ties go to the SWAP whose qubits are free earliest, which
keeps the routing shallow), and the free SWAP of every executed block is
used when it brings the qubits closer to their next partners.  On QV-100 / grid 10x10
this needs about 6 % fewer paid SWAPs than SABRE.

The routings are offered as extra candidates (:func:`layer_route_plan`);
they are only built when the circuit has no three-qubit nodes and every
two-qubit node is a generic SU(4) block.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from quspect.transpiler._mapping import estimate_two_qubit_cost
from quspect.transpiler._route_lns import node_cost_model
from quspect.transpiler._routing import _native_core

if TYPE_CHECKING:
    from quspect.transpiler._mapping import SABRE, _CandidatePlan, _Prepared

#: Iterated-local-search kicks per layer (capped by ``MAX_PERTURBATIONS``).
PERTURBATIONS_PER_LAYER = 1000
#: Upper bound on the kicks of one routing.
MAX_PERTURBATIONS = 20_000
#: Iterated-local-search kicks per layer of the ring chain (each costs more).
RING_PERTURBATIONS_PER_LAYER = 25
#: Upper bound on the kicks of one ring routing.
RING_MAX_PERTURBATIONS = 500
#: Largest ring routed by the cyclic chain model (its local search is cubic).
RING_MAX_QUBITS = 24
#: Widths of the rollout's second-level expansion (1 means plain greedy).
ROLLOUT_WIDTHS = (16, 1)
#: Random restarts of the rollout routing per width, ``ROLLOUT_WORK // blocks``
#: clamped to ``[1, ROLLOUT_MAX_SEEDS]``.
ROLLOUT_WORK = 20_000
ROLLOUT_MAX_SEEDS = 8
#: Forward/backward layout iterations of each rollout routing.
ROLLOUT_LAYOUT_ITERS = 2
#: Restarts are ranked by ``paid SWAPs + ROLLOUT_DEPTH_WEIGHT * depth`` (depth
#: proxy: blocks and paid SWAPs one layer each).
ROLLOUT_DEPTH_WEIGHT = 0.5
#: Largest circuit (blocks) routed by the rollout router.
ROLLOUT_MAX_BLOCKS = 20_000


def path_order(adjacency: list[set[int]]) -> list[int] | None:
    """The vertices of a path graph in path order, or ``None``.

    Args:
        adjacency: Neighbour sets of an undirected graph.

    Returns:
        ``[v_0, …, v_{n-1}]`` with ``v_i`` adjacent to ``v_{i+1}`` if the
        graph is a single path (one vertex counts), else ``None``.

    Example:
        >>> path_order([{1}, {0, 2}, {1}])
        [0, 1, 2]
        >>> path_order([{1, 2}, {0, 2}, {0, 1}]) is None
        True
    """
    n = len(adjacency)
    if n == 0:
        return None
    if n == 1:
        return [0]
    if any(len(a) > 2 for a in adjacency):
        return None
    ends = [v for v in range(n) if len(adjacency[v]) == 1]
    if len(ends) != 2:
        return None
    order = [ends[0]]
    prev = -1
    while len(order) < n:
        cur = order[-1]
        nxt = [v for v in adjacency[cur] if v != prev]
        if not nxt:
            return None
        prev = cur
        order.append(nxt[0])
    return order if len(set(order)) == n else None


def cycle_order(adjacency: list[set[int]]) -> list[int] | None:
    """The vertices of a cycle graph in cyclic order, or ``None``.

    Args:
        adjacency: Neighbour sets of an undirected graph.

    Returns:
        ``[v_0, …, v_{n-1}]`` with ``v_i`` adjacent to ``v_{i+1 mod n}`` if
        the graph is a single cycle of at least 3 vertices, else ``None``.

    Example:
        >>> cycle_order([{1, 3}, {0, 2}, {1, 3}, {2, 0}])
        [0, 1, 2, 3]
        >>> cycle_order([{1}, {0, 2}, {1}]) is None
        True
    """
    n = len(adjacency)
    if n < 3 or any(len(a) != 2 for a in adjacency):
        return None
    order = [0]
    prev = -1
    while len(order) < n:
        cur = order[-1]
        nxt = [v for v in sorted(adjacency[cur]) if v != prev]
        if not nxt or nxt[0] == 0:
            return None
        prev = cur
        order.append(nxt[0])
    return order if 0 in adjacency[order[-1]] and len(set(order)) == n else None


def block_layers(prep: _Prepared) -> list[list[int]] | None:
    """Dependency layers of the two-qubit nodes if layered routing applies.

    Args:
        prep: Prepared routing input (block-node DAG with mirror costs).

    Returns:
        ``layers[l]`` = the two-qubit DAG nodes of layer ``l`` (longest
        path from the inputs), or ``None`` if the DAG has three-qubit nodes,
        no two-qubit node, or a two-qubit node that is not a generic SU(4)
        block (mirror cost other than 0).

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(2, [(0, 1)])
        >>> router = SABRE(target, block_nodes=True, mirror_gates=True)
        >>> block_layers(router._prepare(QuantumCircuit().cx(0, 1), False)) is None
        True
    """
    dag = prep.dag
    if dag.has_three_qubit or any(q >= 0 for q in dag.nq3):
        return None
    mirror, _ = node_cost_model(prep)
    num = len(dag.nodes)
    depth = [0] * num
    layers: list[list[int]] = []
    for n in range(num):
        d = max((depth[p] for p in dag.pred[n]), default=0)
        if dag.nq1[n] >= 0:
            if mirror[n] != 0:
                return None
            while len(layers) <= d:
                layers.append([])
            layers[d].append(n)
            d += 1
        depth[n] = d
    return layers or None


def _topological(dag: Any) -> bool:
    """Whether every predecessor index is smaller than its node's."""
    return all(p < n for n in range(len(dag.nodes)) for p in dag.pred[n])


def _interleave(dag: Any, steps: list[tuple[int, list[int]]]) -> list[int] | None:
    """Routing events from block executions and SWAPs, measurements drained.

    ``steps`` is a list of ``(node, swaps)``: execute two-qubit node ``node``
    (``-1``: none), then the encoded SWAP events ``swaps``.  Non-two-qubit
    nodes (measurements) run as soon as their predecessors have.  Returns
    ``None`` if some node is never executed.
    """
    num = len(dag.nodes)
    done = [False] * num
    others = [v for v in range(num) if dag.nq1[v] < 0]
    events: list[int] = []

    def drain_others() -> None:
        for v in others:
            if not done[v] and all(done[p] for p in dag.pred[v]):
                done[v] = True
                events.append(v)

    for node, swaps in steps:
        if node >= 0:
            drain_others()
            done[node] = True
            events.append(node)
        events.extend(swaps)
    drain_others()
    return events if all(done) else None


def layer_route_events(
    router: SABRE, prep: _Prepared, *, seed: int = 1, perturbations: int | None = None
) -> tuple[list[int], list[int], int] | None:
    """Layered routing of a prepared circuit on a path or a ring (module docs).

    Args:
        router: Router of the target (its coupling graph must be a path or
            a cycle of at most ``RING_MAX_QUBITS`` vertices).
        prep: Prepared routing input.
        seed: Seed of the iterated local search.
        perturbations: Kicks of the iterated local search (default
            ``PERTURBATIONS_PER_LAYER`` per layer, at most
            ``MAX_PERTURBATIONS``; on rings ``RING_PERTURBATIONS_PER_LAYER``
            and ``RING_MAX_PERTURBATIONS``).

    Returns:
        ``(layout, events, paid_swaps)`` — full start permutation
        ``layout[token] = physical``, routing events, and the number of
        SWAPs not absorbed by a block — or ``None`` if layered routing
        does not apply or the native core is missing.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> router = SABRE(target, block_nodes=True, mirror_gates=True)
        >>> prep = router._prepare(QuantumCircuit().cx(0, 2), False)
        >>> layer_route_events(router, prep) is None  # a lone CX is no SU(4)
        True
    """
    core = _native_core()
    if core is None or not hasattr(core, "_route_layers_on_path"):
        return None
    ring = False
    sites = path_order(router._adj_sets)
    if sites is None:
        sites = cycle_order(router._adj_sets)
        if (
            sites is None
            or len(sites) > RING_MAX_QUBITS
            or not hasattr(core, "_route_layers_on_ring")
        ):
            return None
        ring = True
    dag = prep.dag
    if not _topological(dag):
        return None
    layers = block_layers(prep)
    if layers is None:
        return None
    n = len(sites)
    pairs = [[(dag.nq1[v], dag.nq2[v]) for v in layer] for layer in layers]
    if ring:
        if perturbations is None:
            perturbations = min(
                RING_MAX_PERTURBATIONS, RING_PERTURBATIONS_PER_LAYER * len(layers)
            )
        solve = core._route_layers_on_ring
    else:
        if perturbations is None:
            perturbations = min(
                MAX_PERTURBATIONS, PERTURBATIONS_PER_LAYER * len(layers)
            )
        solve = core._route_layers_on_path
    cost, initial, swaps, _, _ = solve(
        n, pairs, 64, perturbations, 6, seed & 0xFFFFFFFFFFFFFFFF
    )
    layout = [0] * n
    for pos, token in enumerate(initial):
        layout[token] = sites[pos]
    steps: list[tuple[int, list[int]]] = []
    for li, layer in enumerate(layers):
        steps.extend((v, []) for v in layer)
        if li < len(swaps):
            codes = []
            for i in swaps[li]:
                x, y = sites[i], sites[(i + 1) % n]
                if x > y:
                    x, y = y, x
                codes.append(-1 - (x * n + y))
            steps.append((-1, codes))
    events = _interleave(dag, steps)
    if events is None:
        return None
    return layout, events, int(cost)


def su4_route_events(
    router: SABRE,
    prep: _Prepared,
    layout: list[int],
    *,
    seed: int = 1,
    width: int = 16,
    layout_iters: int | None = None,
) -> tuple[list[int], list[int], int] | None:
    """Truncated-rollout routing of an SU(4)-block circuit (see module docs).

    Applies to any connected coupling graph.  The routing starts from
    ``layout`` and from ``layout_iters`` forward/backward refinements of it
    and keeps the start with the fewest paid SWAPs.

    Args:
        router: Router of the target coupling graph.
        prep: Prepared routing input.
        layout: Full start permutation ``layout[token] = physical``.
        seed: Seed of the tie-breaking.
        width: Candidates expanded one SWAP further (``1``: plain greedy).
        layout_iters: Forward/backward layout iterations (default
            ``ROLLOUT_LAYOUT_ITERS``).

    Returns:
        ``(layout, events, paid_swaps)`` as :func:`layer_route_events`, or
        ``None`` if the circuit is not an SU(4)-block circuit (or too
        large) or the native core is missing.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> router = SABRE(target, block_nodes=True, mirror_gates=True)
        >>> prep = router._prepare(QuantumCircuit().cx(0, 2), False)
        >>> su4_route_events(router, prep, [0, 1, 2]) is None
        True
    """
    out = _su4_route(router, prep, layout, seed, width, layout_iters)
    return None if out is None else out[:3]


def _su4_route(
    router: SABRE,
    prep: _Prepared,
    layout: list[int],
    seed: int,
    width: int,
    layout_iters: int | None,
) -> tuple[list[int], list[int], int, int] | None:
    """:func:`su4_route_events` plus the routing's depth proxy."""
    core = _native_core()
    if core is None or not hasattr(core, "_route_su4_blocks"):
        return None
    dag = prep.dag
    if not _topological(dag) or block_layers(prep) is None:
        return None
    nodes = [v for v in range(len(dag.nodes)) if dag.nq1[v] >= 0]
    if len(nodes) > ROLLOUT_MAX_BLOCKS:
        return None
    n = router._num_physical
    if len(layout) != n:
        return None
    if layout_iters is None:
        layout_iters = ROLLOUT_LAYOUT_ITERS
    adjacency = [sorted(a) for a in router._adj_sets]
    blocks = [(dag.nq1[v], dag.nq2[v]) for v in nodes]
    paid, _, start, codes, depth = core._route_su4_blocks(
        adjacency, blocks, list(layout), width, 0.5, 0.5, layout_iters, seed
    )
    steps: list[tuple[int, list[int]]] = []
    for code in codes:
        if code >= 0:
            steps.append((nodes[code], []))
        elif steps:
            steps[-1][1].append(int(code))
        else:
            steps.append((-1, [int(code)]))
    events = _interleave(dag, steps)
    if events is None:
        return None
    return list(start), events, int(paid), int(depth)


def _plan_from(
    router: SABRE,
    base: _CandidatePlan,
    out: tuple[list[int], list[int], int],
    label: str,
) -> _CandidatePlan:
    """A candidate plan from a routing ``(layout, events, paid)`` of ``base``."""
    assert base.prep is not None
    layout, events, _ = out
    gates, initial, final = router._finish(base.prep, events, layout)
    return replace(
        base,
        gates=gates,
        initial_mapping=initial,
        final_mapping=final,
        num_swaps=sum(1 for e in events if e < 0),
        estimated_two_qubit_gates=estimate_two_qubit_cost(gates, router._num_physical),
        label=f"{label}/{base.label}",
        is_default=False,
        embedded=False,
        route_layout=layout,
        route_events=events,
    )


def layer_route_plan(
    router: SABRE, base: _CandidatePlan, *, seed: int = 1
) -> _CandidatePlan | None:
    """Best layered / rollout routing of ``base``'s circuit as a candidate.

    On a path or a ring the layered chain routing (:func:`layer_route_events`)
    is built; on every topology except a path the truncated-rollout routing
    (:func:`su4_route_events`, widths ``ROLLOUT_WIDTHS``, a few seeds
    depending on the circuit size) starts from ``base``'s layout.  The plan
    with the fewest estimated two-qubit gates is returned.

    Args:
        router: Router of the target coupling graph.
        base: Candidate whose preparation (block-node DAG) is routed.
        seed: Seed of the local searches.

    Returns:
        A plan labelled ``layers/<base label>`` or ``rollout/<base label>``
        (never the default), or ``None`` if no such routing applies.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> plan = SABRE(target)._solve_plan(QuantumCircuit().cx(0, 2), seed=1)
        >>> layer_route_plan(SABRE(target), plan) is None  # a lone CZ is no SU(4)
        True
    """
    prep = base.prep
    if prep is None:
        return None
    plans: list[_CandidatePlan] = []
    out = layer_route_events(router, prep, seed=seed)
    if out is not None:
        plans.append(_plan_from(router, base, out, "layers"))
    if path_order(router._adj_sets) is None and base.route_layout:
        num_blocks = sum(1 for q in prep.dag.nq1 if q >= 0)
        seeds = max(1, min(ROLLOUT_MAX_SEEDS, ROLLOUT_WORK // max(1, num_blocks)))
        best: tuple[list[int], list[int], int] | None = None
        best_key = (0.0, 0)
        for width in ROLLOUT_WIDTHS:
            for k in range(seeds):
                rolled = _su4_route(
                    router, prep, base.route_layout, seed + 7919 * k, width, None
                )
                if rolled is None:
                    break
                key = (rolled[2] + ROLLOUT_DEPTH_WEIGHT * rolled[3], rolled[2])
                if best is None or key < best_key:
                    best, best_key = rolled[:3], key
        if best is not None:
            # 所有块都是一般 SU(4)：估计的双比特门数随付费 SWAP 单调，只完成最好的一个
            plans.append(_plan_from(router, base, best, "rollout"))
    if not plans:
        return None
    return min(plans, key=lambda p: (p.estimated_two_qubit_gates, p.num_swaps))
