"""Exact window re-routing and large-neighbourhood search over SABRE routings.

A routing is a sequence of *events*: executions of routing-DAG nodes and
physical SWAPs.  Its SWAP cost is measured in CZ gates with the block-merge
model of the mirror-gate router: a SWAP costs 3, unless the operation right
before or after it on both of its physical qubits is the same two-qubit
block on that pair, in which case block resynthesis merges them and the SWAP
costs only the block's *mirror cost* ``cz(U · SWAP) - cz(U)`` (0 for a
generic SU(4) block, 1 for a lone CZ).  The cheaper side counts.

Exact search (A*)
-----------------

:func:`exact_route` finds a routing of a whole DAG from a fixed initial
layout with the minimum cost (or minimum SWAP count with
``merge_costs=False``).  The search state is the permutation of the logical
qubits, the set of executed nodes and the open merge opportunities; nodes
are executed as soon as they are executable.  The lower bound is admissible
and built from *movement*: a SWAP moves two logical qubits by one step each.

* Meeting and exit: the qubits ``a, b`` of a pending node must meet on some
  coupling edge ``(m, m')``, and a qubit that is used after the window must
  then reach its exit position ``e``, so together they move at least
  ``min over edges of D(a, m) + D(m, e_a) + D(b, m') + D(m', e_b)``.
  Requirements of qubit-disjoint nodes add up (chosen greedily, largest
  first), as do the exit distances of the remaining constrained qubits.
* Hub chain: the qubit with the most pending nodes meets its partners in a
  fixed order.  A dynamic program over its position,
  ``f_i(x) = min_u [f_{i-1}(u) + D(u, x)] + min_{y ∈ N(x)} [D(p_b, y) +
  D(y, e_b)]`` (the first term is a min-plus distance transform, i.e. a
  multi-source BFS), bounds the movement of the hub and its distinct
  partners together.
* Cost: half of a SWAP's cost is charged to each qubit it moves.  A qubit
  can make a cheap (merged) step at most once per node it is part of, at
  that node's mirror cost; every other step costs ``3 / 2``.  SWAPs needed
  ``>= ceil(movement / 2)`` also bound the cost with the cheapest merge
  opportunities.

The expensive parts (meeting/exit and hub chain) are evaluated lazily, when
a state is popped (Lazy A*).  Commuting SWAPs on disjoint edges are only
tried in one order (partial-order reduction).  For the pure SWAP count the
search is exact; with merge costs the greedy execution of executable nodes
can, in rare cases, miss a merge.

On small circuits (at most ten qubits) where A* proves optimality from the
level-3 initial layout, the SWAP count of the level-3 routing is typically
15–30 % above the minimum, but its merge-aware cost is within a few CZ of
the optimum: the merges, not the SWAP count, are what matter.

Large-neighbourhood search
--------------------------

:func:`refine_routing` takes a SABRE routing and repeatedly cuts out a
*window* — ``k`` consecutive SWAPs together with the nodes executed between
them (from the previous SWAP to the next one).  The layout entering the
window is fixed, and so is the exit position of every logical qubit that is
still used after the window; qubits without later nodes (idle and finished
qubits, workspace ancillas) may end anywhere, since their placement is
absorbed into the final mapping.  The window is re-routed by the same A*
with the original window cost as upper bound (branch and bound: the search
only explores routings that are strictly cheaper) and a budget of A*
expansions, including the merges with the operations just before and after
the window.  Improvements are spliced in immediately; the window sizes are
swept left to right, twice.  The result executes the
same nodes, so it is exactly equivalent; its cost never increases.

Depth as secondary objective
----------------------------

Every physical qubit carries the time (in CZ layers) at which it becomes
free: a SWAP adds its cost, a node its weight (:func:`node_weights`), and a
node merged into the SWAP before it only the part the merge adds.  The rest
of the routing is fixed around a window, so the depth proxy of the whole
routing is exactly ``max_q (exit time[q] + tail[q])``, where ``tail[q]`` is
the longest path from the first operation after the window on ``q``.  The
window A* then orders by ``(cost, depth)`` lexicographically: the cost bound
is unchanged (the primary objective keeps its optimality) and the depth part
only uses the current times, which never decrease (admissible).  With
``depth_ties`` the cost sweeps take the shallowest of the cheapest window
routings; ``depth_passes`` adds sweeps that also replace a window by an
equally cheap, shallower routing (:func:`deepen_plan`).

The native core (``cpp/src/transpiler/route_lns.cpp``) is required; without
it the functions return ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from quspect.transpiler._mapping import (
    _GATE_CZ_COST,
    _collect_pair_blocks,
    _mirror_cost_arrays,
    estimate_two_qubit_cost,
)
from quspect.transpiler._routing import _native_core, native_handles

if TYPE_CHECKING:
    from quspect.transpiler._mapping import (
        SABRE,
        _CandidatePlan,
        _Prepared,
    )

#: Window sizes (SWAPs per window) swept by :func:`refine_routing`.
DEFAULT_WINDOW_SWAPS = (1, 2, 3, 4)
#: A* expansions allowed per window.
DEFAULT_WINDOW_EXPANSIONS = 4000
#: Windows with more nodes than this are skipped.
DEFAULT_MAX_WINDOW_NODES = 64


@dataclass
class ExactRouting:
    """Result of :func:`exact_route`.

    Attributes:
        found: A routing was found within the budget.
        optimal: The routing is proven optimal (A* finished).
        cost: Its SWAP cost (CZ units; ``3 * num_swaps`` without merges).
        num_swaps: Its number of SWAPs.
        lower_bound: Proven lower bound on the optimal cost (``-1``
            if unknown).
        expansions: A* states expanded.
        events: The routing events (node index ``>= 0`` or encoded SWAP
            ``-1 - (x * n + y)``).
    """

    found: bool
    optimal: bool
    cost: int
    num_swaps: int
    lower_bound: int
    expansions: int
    events: list[int]


@dataclass
class RefinedRouting:
    """Result of :func:`refine_routing`.

    Attributes:
        events: Refined routing events (same nodes, legal, not costlier).
        old_cost: Cost of the input routing.
        new_cost: Cost of the refined routing.
        old_swaps: SWAPs of the input routing.
        new_swaps: SWAPs of the refined routing.
        windows: Windows searched.
        improved: Windows replaced by a cheaper (or, in the depth sweeps,
            an equally cheap but shallower) routing.
        expansions: A* states expanded in total.
        old_depth: Depth proxy (two-qubit layers) of the input routing
            (``0`` without ``depth_ties``).
        new_depth: Depth proxy of the refined routing.
    """

    events: list[int]
    old_cost: int
    new_cost: int
    old_swaps: int
    new_swaps: int
    windows: int
    improved: int
    expansions: int
    old_depth: int = 0
    new_depth: int = 0


def lns_available() -> bool:
    """Whether the native core of the exact search is available.

    Returns:
        ``True`` if :func:`exact_route` and :func:`refine_routing` can run.

    Example:
        >>> isinstance(lns_available(), bool)
        True
    """
    core = _native_core()
    return core is not None and hasattr(core, "_refine_routing")


def node_cost_model(prep: _Prepared) -> tuple[list[int], list[int]]:
    """Per-node merge costs and measured qubits of a prepared routing input.

    The mirror cost of a node is taken from the DAG's mirror annotation when
    present, else computed the same way (cached on ``prep``).  On a DAG
    whose nodes are single gates of a longer pair block, the first / last
    node of the block carries the block's cost; inner nodes get ``-1``.

    Args:
        prep: Prepared routing input (gates, DAG, source circuit).

    Returns:
        ``(mirror_cost, measure_token)`` lists, one entry per DAG node.
    """
    if prep.lns_model is not None:
        return prep.lns_model
    dag = prep.dag
    if dag.mirror_fwd:
        fwd, bwd = dag.mirror_fwd, dag.mirror_bwd
    else:
        try:
            fwd, bwd = _mirror_cost_arrays(dag, prep.gates, prep.circuit)
        except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
            fwd = [
                _fallback_mirror_cost(g.name) if dag.nq1[n] >= 0 else -1
                for n, g in enumerate(dag.nodes)
            ]
            bwd = fwd
    mirror = [f if f >= 0 else b for f, b in zip(fwd, bwd, strict=True)]
    measure = [g.q1 if dag.nq1[n] < 0 else -1 for n, g in enumerate(dag.nodes)]
    prep.lns_model = (mirror, measure)
    return prep.lns_model


#: Two-qubit layers of a routed three-qubit (``ccx``) node in the depth proxy.
_THREE_QUBIT_WEIGHT = 6


def node_weights(prep: _Prepared) -> list[int]:
    """Per-node two-qubit layer counts of the LNS depth proxy.

    A node's weight is the number of CZ layers it contributes to the
    two-qubit depth: its CZ count, where every maximal pair block is capped
    at 3 (it is resynthesized into at most three CZ), so later gates of a
    long block weigh 0.  A routed ``ccx`` weighs ``6``; measurements ``0``.
    Cached on ``prep``.

    Args:
        prep: Prepared routing input (gates, DAG).

    Returns:
        One weight per DAG node.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(2, [(0, 1)])
        >>> router = SABRE(target)
        >>> qc = QuantumCircuit().cx(0, 1).h(0).cp(0.3, 0, 1)
        >>> node_weights(router._prepare(qc, False))
        [1, 2]
    """
    if prep.lns_weight is not None:
        return prep.lns_weight
    dag = prep.dag
    node_of = {id(g): n for n, g in enumerate(dag.nodes)}
    for n, members in enumerate(dag.extra):
        for _, g in members:
            node_of[id(g)] = n
    weight = [_THREE_QUBIT_WEIGHT if q3 >= 0 else 0 for q3 in dag.nq3]
    blocks, _ = _collect_pair_blocks(prep.gates)
    for block in blocks:
        left = 3
        for i in block:
            g = prep.gates[i]
            node = node_of.get(id(g))
            if not g.is_two_qubit or node is None:
                continue
            c = min(left, 3 if g.name == "swap" else _GATE_CZ_COST.get(g.name, 1))
            weight[node] += c
            left -= c
    prep.lns_weight = weight
    return weight


def _fallback_mirror_cost(name: str) -> int:
    """Mirror cost of a lone gate by the estimate rules (``SWAP · CZ`` = 2 CZ)."""
    return max(0, 3 - _GATE_CZ_COST.get(name, 1)) if name else -1


def routing_cost(
    router: SABRE,
    prep: _Prepared,
    layout: list[int],
    events: list[int],
    *,
    merge_costs: bool = True,
) -> int | None:
    """SWAP cost (CZ units) of a routing under the block-merge model.

    Args:
        router: Router of the coupling graph the routing is for.
        prep: Prepared routing input the events refer to.
        layout: Full start permutation ``layout[token] = physical``.
        events: Routing events.
        merge_costs: Use the merge model (else every SWAP costs 3).

    Returns:
        The cost, or ``None`` without the native core.

    Raises:
        ValueError: If the events are not a legal routing.
    """
    core = _native_core()
    if core is None or not hasattr(core, "_route_swap_cost"):
        return None
    graph, ndag = native_handles(core, router, prep.dag)
    mirror, measure = node_cost_model(prep)
    return int(
        core._route_swap_cost(graph, ndag, mirror, measure, layout, events, merge_costs)
    )


def exact_route(
    router: SABRE,
    prep: _Prepared,
    layout: list[int],
    *,
    merge_costs: bool = True,
    max_expansions: int = 200_000,
    upper_bound: int = 0,
) -> ExactRouting | None:
    """Optimal routing of a whole prepared circuit from a fixed layout (A*).

    Args:
        router: Router of the target coupling graph.
        prep: Prepared routing input (see ``SABRE._prepare``).
        layout: Full start permutation ``layout[token] = physical``.
        merge_costs: Minimize the block-merge cost (else the SWAP count).
        max_expansions: A* state budget; when it runs out the best routing
            found so far (if any) is returned with ``optimal=False``.
        upper_bound: Only look for routings strictly cheaper than this
            (``0``: no bound).

    Returns:
        The search result, or ``None`` without the native core.

    Raises:
        ValueError: If ``layout`` is not a permutation of the physical
            qubits.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> router = SABRE(target)
        >>> prep = router._prepare(QuantumCircuit().cx(0, 2), False)
        >>> res = exact_route(router, prep, [0, 1, 2], merge_costs=False)
        >>> res is None or (res.optimal and res.num_swaps == 1)
        True
    """
    core = _native_core()
    if core is None or not hasattr(core, "_exact_route"):
        return None
    graph, ndag = native_handles(core, router, prep.dag)
    mirror, measure = node_cost_model(prep)
    found, optimal, cost, swaps, expansions, lower, events = core._exact_route(
        graph,
        ndag,
        mirror,
        measure,
        list(layout),
        max_expansions,
        merge_costs,
        upper_bound,
    )
    return ExactRouting(
        found=found,
        optimal=optimal,
        cost=cost,
        num_swaps=swaps,
        lower_bound=lower,
        expansions=expansions,
        events=list(events),
    )


def refine_routing(
    router: SABRE,
    prep: _Prepared,
    layout: list[int],
    events: list[int],
    *,
    window_swaps: tuple[int, ...] = DEFAULT_WINDOW_SWAPS,
    passes: int = 2,
    window_expansions: int = DEFAULT_WINDOW_EXPANSIONS,
    total_expansions: int = 4_000_000,
    max_window_nodes: int = DEFAULT_MAX_WINDOW_NODES,
    merge_costs: bool = True,
    weight: float = 1.0,
    stop_at_first: bool = False,
    depth_ties: bool = False,
    depth_passes: int = 0,
    depth_expansions: int = 0,
) -> RefinedRouting | None:
    """Large-neighbourhood search: exact re-routing of SWAP windows.

    See the module documentation.  The refined events execute the same
    nodes from the same layout, so the routed circuit stays exactly
    equivalent; its cost never exceeds the input's.

    Args:
        router: Router of the target coupling graph.
        prep: Prepared routing input the events refer to.
        layout: Full start permutation ``layout[token] = physical``.
        events: A legal routing of ``prep.dag`` from ``layout``.
        window_swaps: Window sizes (SWAPs per window) to sweep.
        passes: Sweeps over all window sizes (stops early without gain).
        window_expansions: A* state budget per window.
        total_expansions: A* state budget of the whole search.
        max_window_nodes: Skip windows with more DAG nodes.
        merge_costs: Use the block-merge cost model.
        weight: Search order ``g + weight * h`` of the window A* (``> 1``:
            weighted A*, pruning stays admissible).
        stop_at_first: Stop a window at its first improvement.
        depth_ties: Use the depth proxy as secondary objective: among the
            cheapest window routings take the shallowest (see the module
            documentation).
        depth_passes: Further sweeps after the cost sweeps that also
            replace a window by an equally cheap, shallower routing
            (requires ``depth_ties``).
        depth_expansions: A* state budget of those sweeps.

    Returns:
        The refined routing, or ``None`` without the native core.

    Raises:
        ValueError: If ``events`` is not a legal routing.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> router = SABRE(target)
        >>> prep = router._prepare(QuantumCircuit().cx(0, 2).cx(0, 2), False)
        >>> swap01 = -1 - (0 * 3 + 1)
        >>> events = [swap01, 0, swap01, swap01, 1]  # a wasteful round trip
        >>> res = refine_routing(router, prep, [0, 1, 2], events)
        >>> res is None or res.new_cost <= res.old_cost
        True
    """
    core = _native_core()
    if core is None or not hasattr(core, "_refine_routing"):
        return None
    graph, ndag = native_handles(core, router, prep.dag)
    mirror, measure = node_cost_model(prep)
    out: tuple[Any, ...] = core._refine_routing(
        graph,
        ndag,
        mirror,
        measure,
        list(layout),
        list(events),
        list(window_swaps),
        passes,
        window_expansions,
        total_expansions,
        max_window_nodes,
        merge_costs,
        weight,
        stop_at_first,
        node_weights(prep) if depth_ties or depth_passes > 0 else [],
        depth_ties,
        depth_passes,
        depth_expansions,
    )
    new_events, old_cost, new_cost, old_swaps, new_swaps, windows, improved, exp = out[
        :8
    ]
    return RefinedRouting(
        events=list(new_events),
        old_cost=old_cost,
        new_cost=new_cost,
        old_swaps=old_swaps,
        new_swaps=new_swaps,
        windows=windows,
        improved=improved,
        expansions=exp,
        old_depth=out[8],
        new_depth=out[9],
    )


def refine_plan(
    router: SABRE,
    plan: _CandidatePlan,
    *,
    prefix: str = "lns",
    **options: Any,
) -> _CandidatePlan | None:
    """LNS-refined copy of a routing candidate, or ``None`` if not cheaper.

    Args:
        router: Router of the target coupling graph (any configuration).
        plan: Candidate from ``SABRE._candidate_plans`` (carries its
            routing).
        prefix: Label prefix of the refined plan.
        **options: Keyword arguments of :func:`refine_routing`.

    Returns:
        A new plan (label ``<prefix>/<label>``, never the default) whose routing
        is strictly cheaper under the merge model, or equally cheap with a
        smaller depth proxy (``depth_ties``), else ``None``.
    """
    prep = plan.prep
    if prep is None or not plan.route_events or plan.num_swaps == 0:
        return None
    res = refine_routing(router, prep, plan.route_layout, plan.route_events, **options)
    if res is None or res.improved == 0:
        return None
    cheaper = res.new_cost < res.old_cost
    if not cheaper and not (
        res.new_cost == res.old_cost and res.new_depth < res.old_depth
    ):
        return None
    gates, initial, final = router._finish(prep, res.events, plan.route_layout)
    return replace(
        plan,
        gates=gates,
        initial_mapping=initial,
        final_mapping=final,
        num_swaps=res.new_swaps,
        estimated_two_qubit_gates=estimate_two_qubit_cost(gates, router._num_physical),
        label=f"{prefix}/{plan.label}",
        is_default=False,
        route_events=res.events,
    )


def deepen_plan(
    router: SABRE,
    plan: _CandidatePlan,
    *,
    window_swaps: tuple[int, ...] = DEFAULT_WINDOW_SWAPS,
    window_expansions: int = DEFAULT_WINDOW_EXPANSIONS,
    expansions: int = 1_000_000,
    passes: int = 1,
) -> _CandidatePlan | None:
    """Equally cheap, shallower copy of a routing candidate (depth sweeps only).

    Runs only the depth sweeps of :func:`refine_routing`: a window is replaced
    by a routing that is better in ``(merge cost, depth proxy)``
    lexicographically — usually equally cheap with a shorter two-qubit
    critical path through the window (the rest of the routing fixed).  The
    cost under the merge model never increases.

    Args:
        router: Router of the target coupling graph.
        plan: Candidate carrying its routing (possibly an LNS refinement).
        window_swaps: Window sizes to sweep.
        window_expansions: A* state budget per window.
        expansions: A* state budget of all depth sweeps.
        passes: Depth sweeps over all window sizes.

    Returns:
        A new plan labelled ``lnsd/<label>`` that is cheaper, or equally
        cheap with a smaller depth proxy, else ``None``.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> router = SABRE(target)
        >>> plan = router._solve_plan(QuantumCircuit().cx(0, 2), seed=1)
        >>> deep = deepen_plan(router, plan)
        >>> deep is None or deep.label.startswith("lnsd/")
        True
    """
    return refine_plan(
        router,
        plan,
        prefix="lnsd",
        window_swaps=window_swaps,
        passes=0,
        window_expansions=window_expansions,
        total_expansions=0,
        depth_passes=passes,
        depth_expansions=expansions,
    )
