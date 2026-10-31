"""Unified SABRE routing core (two- and three-qubit nodes, any routing DAG).

One routing pass of :class:`~quspect.transpiler.SABRE` — front layer,
bounded extended set, decay, optional SWAP-fusion bonus, seeded random
tie-breaking and the release valve — is implemented once here, in
:func:`route_dag`, for every kind of routing DAG the mapper builds:

* plain dependency DAGs of two-qubit gates and measurements;
* DAGs with three-qubit ``ccx`` nodes (routed as one unit, see
  :mod:`quspect.transpiler._toffoli`);
* commutation-aware DAGs, where several front-layer nodes may share a
  qubit.

Extension points
----------------

The loop is written around a few small, separately documented pieces so
that new node kinds or scoring terms do not need another copy of it:

* **Node cost**: the distance term of a node.  A two-qubit node costs
  ``D[pa][pb]``; a three-qubit node costs the size of the smallest tree
  spanning its three physical qubits minus one (:func:`cost3`), so a node
  is executable iff its cost is ``1``.  Two-qubit nodes take an inlined
  fast path; a node with ``dag.nq3[n] >= 0`` goes through the generic
  three-qubit path (exact cost change per candidate SWAP).
* **Front / lookahead bookkeeping**: per logical token, the front-layer
  two-qubit nodes (``front2``) and three-qubit nodes (``front3``) on it,
  and the other endpoints of its front-layer (``fpart``) and extended-set
  (``partners``) two-qubit nodes.  Lists rather than single slots, so a
  commutation-aware DAG may put several front nodes on one qubit.
* **Candidate scoring**: candidate SWAPs come from a per-qubit table of
  coupling edges (:func:`incident_edges`), merged in front-layer order.
  For every token with two-qubit partners, a lazily built *delta table*
  holds, per neighbouring physical qubit ``z``, the change of its front and
  extended-set distance sums if it moved to ``z``; a table is dropped as
  soon as the token, one of its partners, or its partner lists change, so
  most candidates are scored with two table lookups.
* **Score modifiers**: the raw lookahead score is multiplied by the decay
  factor of the SWAP and, with a SWAP-fusion bonus, by ``1 - bonus`` when
  the SWAP's pair just executed the same two-qubit node.
* **Mirror gates** (``router.mirror_gates`` and a DAG annotated by
  ``_annotate_mirror_costs``): after a round of executions, every executed
  node that ends its pair block (and is still the last node on both its
  qubits) may be followed by a SWAP on the same pair for its *mirror cost*
  ``cz(U · SWAP) - cz(U)`` (0 for a generic SU(4) block).  The SWAP is
  inserted when exchanging the two logical qubits lowers the front +
  extended-set score (no decay) by more than ``mirror_threshold +
  mirror_cz_penalty * cost``.  Search SWAPs on such a pair are counted at
  the mirror cost, and the SWAP-fusion discount becomes ``bonus * (3 -
  cost) / 3``.  ``_RouteResult.swap_cost`` sums the CZ cost of all SWAPs
  (``3 * num_swaps`` without mirror gates).
* **Release valve**: force-routes the cheapest front-layer node along
  shortest paths (both ends of a two-qubit node walk towards each other;
  the two outer qubits of a three-qubit node walk to the middle one).

All arithmetic on distances is integer, so the order in which the per-node
contributions are summed never changes a score; the floating-point score
expression and the candidate order are fixed, and ties are broken with the
caller's random source.  The routing is therefore a deterministic function
of ``(layout, dag, direction, rng state)``.

Native core
-----------

The same loop is implemented in C++ (``cpp/src/transpiler/sabre_routing.cpp``,
exposed by the ``quspect._core_*`` extension module).  :func:`route_dag`
uses it automatically when the extension is built and falls back to the
pure-Python reference implementation otherwise.  The native core is
bit-identical to the Python one: it walks the front layer and candidate
SWAPs in the same order, keeps the floating-point score expression (no FMA
contraction), and reproduces CPython's Mersenne Twister and
``Random.choice`` exactly — the caller's :class:`random.Random` state is
handed to C++ and the advanced state written back, so later draws are
unchanged.  Set the environment variable ``QUSPECT_ROUTING_BACKEND`` to
``python`` to force the reference implementation, or to ``native`` to
require the native core (``auto``, the default, picks native when
available).
"""

# 记号沿用 SABRE 论文：D（距离矩阵）、G（邻接表）、W（扩展集权重）。
# ruff: noqa: N803, N806

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quspect.transpiler._mapping import SABRE, _RoutingDag

_TIE_EPS = 1e-10

ROUTING_BACKEND_ENV = "QUSPECT_ROUTING_BACKEND"
"""Name of the environment variable that selects the routing core.

Values: ``auto`` (default; native when available), ``native`` or ``python``.
It is read on every :func:`route_dag` call.
"""

_BACKENDS = ("auto", "native", "python")

_DISCONNECTED = (
    "No routing SWAP is available for the current mapping. "
    "The selected physical-qubit topology may be disconnected."
)


@dataclass
class _RouteResult:
    """Outcome of one routing pass."""

    num_swaps: int
    final_pi: list[int]
    events: list[int]  # node index (>= 0) or encoded swap (< 0)
    depth: int = 0
    # SWAP 的估计 CZ 代价：普通 SWAP 计 3，紧跟在块末节点之后（镜像门）计该块的
    # 镜像代价；未启用镜像门时恒为 3 * num_swaps（未给出时按此补全）。
    swap_cost: int = -1
    num_mirrors: int = 0

    def __post_init__(self) -> None:
        if self.swap_cost < 0:
            self.swap_cost = 3 * self.num_swaps


# =============================================================================
# Node cost model
# =============================================================================


def cost3(D: list[list[int]], pa: int, pb: int, pc: int) -> int:
    """Routing cost of a three-qubit node on physical qubits ``pa, pb, pc``.

    The smallest tree spanning three vertices of a graph whose shortest
    paths meet is ``min`` over the middle vertex of the two distances to the
    others; ``1`` means the three qubits are connected (a path or a
    triangle), like the distance of an executable two-qubit gate.

    Args:
        D: All-pairs distance matrix.
        pa: Physical qubit of the first control.
        pb: Physical qubit of the second control.
        pc: Physical qubit of the target.

    Returns:
        ``min(D[a][b] + D[a][c], D[a][b] + D[b][c], D[a][c] + D[b][c]) - 1``.

    Example:
        >>> line = [[abs(i - j) for j in range(4)] for i in range(4)]
        >>> cost3(line, 0, 2, 1), cost3(line, 0, 3, 1)
        (1, 2)
    """
    dab, dac, dbc = D[pa][pb], D[pa][pc], D[pb][pc]
    return min(dab + dac, dab + dbc, dac + dbc) - 1


def _node_cost(
    D: list[list[int]],
    pi: list[int],
    nq1: list[int],
    nq2: list[int],
    nq3: list[int],
    n: int,
) -> int:
    """Distance term of routing node ``n`` under mapping ``pi`` (``1`` = executable)."""
    c = nq3[n]
    if c >= 0:
        return cost3(D, pi[nq1[n]], pi[nq2[n]], pi[c])
    return D[pi[nq1[n]]][pi[nq2[n]]]


def incident_edges(G: list[list[int]]) -> list[dict[int, tuple[int, int]]]:
    """Candidate-SWAP table: per physical qubit, its coupling edges.

    ``out[p]`` maps ``code = x * n + y`` to ``(x, y)`` (``x < y``) for every
    neighbour of ``p``, in neighbour order; :func:`route_dag` builds its
    ordered candidate set by merging these dicts.

    Args:
        G: Adjacency lists of the coupling graph.

    Returns:
        One ``{code: (x, y)}`` dict per physical qubit.

    Example:
        >>> incident_edges([[1], [0, 2], [1]])[1]
        {1: (0, 1), 5: (1, 2)}
    """
    n = len(G)
    return [
        {(p * n + z if p < z else z * n + p): (p, z) if p < z else (z, p) for z in G[p]}
        for p in range(n)
    ]


def _tokens(nq1: list[int], nq2: list[int], nq3: list[int], n: int) -> tuple[int, ...]:
    c = nq3[n]
    return (nq1[n], nq2[n]) if c < 0 else (nq1[n], nq2[n], c)


# =============================================================================
# Routing pass
# =============================================================================


def route_dag(
    router: SABRE,
    layout: list[int],
    dag: _RoutingDag,
    *,
    forward: bool,
    rng: random.Random,
    record: bool = False,
) -> _RouteResult:
    """Run one SABRE routing pass over ``dag``.

    Every SWAP candidate touches a physical qubit of a front-layer node and
    is scored by::

        H = max(decay[x], decay[y]) * fusion(x, y)
            * (sum_F cost + W * sum_E cost / |E|)

    where ``cost`` is the node cost (see the module documentation) and
    ``fusion`` is ``1 - bonus`` for a SWAP on a pair whose last operation
    was the same two-qubit node (only with ``router.swap_fusion_bonus > 0``
    and a DAG without three-qubit nodes), else ``1``.

    Args:
        router: Configured router: distance matrix ``_D``, adjacency
            ``_adj_sets``, candidate table ``_edges_at`` (see
            :func:`incident_edges`), ``lookahead_weight``,
            ``extended_set_size``, ``decay_delta``,
            ``decay_reset_interval``, ``release_valve_threshold`` and
            ``swap_fusion_bonus``.
        layout: Full permutation ``layout[token] = physical`` (logical
            qubits first, then ancillas).
        dag: Routing DAG (``nq1[n] < 0``: measurement; ``nq3[n] >= 0``:
            three-qubit node).
        forward: Traverse the DAG forwards (``True``) or reversed.
        rng: Random source for tie-breaking.
        record: Record the execution/SWAP event sequence and the
            two-qubit depth.

    Returns:
        Routing statistics, the final full permutation and (if
        ``record``) the events.

    Raises:
        ValueError: If no routing SWAP is available (disconnected topology)
            or the routing backend variable holds an unknown value.
        RuntimeError: If the native core is required but not available.

    Example:
        >>> import random
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> from quspect.transpiler._mapping import _build_routing_dag, _extract_gates
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> gates, n = _extract_gates(QuantumCircuit().cx(0, 2))
        >>> dag = _build_routing_dag(gates, n)
        >>> res = route_dag(
        ...     SABRE(target), [0, 1, 2], dag, forward=True, rng=random.Random(0)
        ... )
        >>> res.num_swaps
        1
    """
    backend = os.environ.get(ROUTING_BACKEND_ENV, "auto") or "auto"
    if backend not in _BACKENDS:
        raise ValueError(
            f"{ROUTING_BACKEND_ENV} must be one of {', '.join(_BACKENDS)}; "
            f"got {backend!r}"
        )
    if backend != "python":
        core = _native_core()
        if (
            core is not None
            and type(rng) is random.Random
            and len(layout) == router._num_physical
        ):
            return _route_dag_native(
                core, router, layout, dag, forward=forward, rng=rng, record=record
            )
        if backend == "native":
            raise RuntimeError(
                "The native routing core is not available (build the C++ "
                "extension, see quspect._native_loader.BUILD_HINT)."
            )
    return _route_dag_python(
        router, layout, dag, forward=forward, rng=rng, record=record
    )


def routing_backend() -> str:
    """Name of the routing core :func:`route_dag` currently uses.

    Returns:
        ``"native"`` when the C++ core is available and not disabled via
        :data:`ROUTING_BACKEND_ENV`, else ``"python"``.

    Example:
        >>> routing_backend() in ("native", "python")
        True
    """
    if os.environ.get(ROUTING_BACKEND_ENV) == "python":
        return "python"
    return "native" if _native_core() is not None else "python"


def _native_core() -> Any | None:
    """The loaded extension module if it provides the routing core."""
    from quspect._native_loader import load_core

    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        return None
    if core is None or not hasattr(core, "_route_dag"):
        return None
    return core


def _route_dag_native(
    core: Any,
    router: SABRE,
    layout: list[int],
    dag: _RoutingDag,
    *,
    forward: bool,
    rng: random.Random,
    record: bool,
) -> _RouteResult:
    """:func:`route_dag` on the C++ core (bit-identical to the Python one).

    The coupling graph and the DAG are converted once and cached on the
    router / DAG.  The Mersenne Twister state of ``rng`` is passed in and,
    if the core drew any random numbers, the advanced state is written
    back.
    """
    graph, native_dag = native_handles(core, router, dag)
    version, internal, gauss_next = rng.getstate()
    (
        num_swaps,
        final_pi,
        events,
        depth,
        swap_cost,
        num_mirrors,
        state,
    ) = core._route_dag(
        graph,
        native_dag,
        layout,
        forward,
        record,
        router.lookahead_weight,
        min(router.extended_set_size, 2**31 - 1),
        router.decay_delta,
        min(router.decay_reset_interval, 2**31 - 1),
        min(router.release_valve_threshold or 0, 2**31 - 1),
        router.swap_fusion_bonus,
        router.mirror_gates,
        router.mirror_threshold,
        router.mirror_cz_penalty,
        internal,
    )
    if state is not None:
        rng.setstate((version, state, gauss_next))
    return _RouteResult(
        num_swaps=num_swaps,
        final_pi=final_pi,
        events=events,
        depth=depth,
        swap_cost=swap_cost,
        num_mirrors=num_mirrors,
    )


def native_handles(core: Any, router: SABRE, dag: _RoutingDag) -> tuple[Any, Any]:
    """Native copies ``(graph, dag)`` of the coupling graph and the routing DAG.

    Both are built once and cached on the router / DAG (keyed by the
    extension module, so a reloaded module rebuilds them).

    Args:
        core: The loaded native extension module.
        router: Router whose ``_G`` / ``_D`` describe the coupling graph.
        dag: Routing DAG (not modified after the first native use).

    Returns:
        The native ``_RoutingGraph`` and ``_RoutingDag`` objects.
    """
    graph = getattr(router, "_native_graph", None)
    if graph is None or graph[0] is not core:
        graph = (core, core._RoutingGraph(router._G, router._D))
        router._native_graph = graph
    native_dag = dag.native
    if native_dag is None or native_dag[0] is not core:
        native_dag = (
            core,
            core._RoutingDag(
                dag.nq1,
                dag.nq2,
                dag.nq3,
                dag.succ,
                dag.pred,
                dag.has_three_qubit,
                dag.mirror_fwd,
                dag.mirror_bwd,
            ),
        )
        dag.native = native_dag
    return graph[1], native_dag[1]


def _route_dag_python(
    router: SABRE,
    layout: list[int],
    dag: _RoutingDag,
    *,
    forward: bool,
    rng: random.Random,
    record: bool = False,
) -> _RouteResult:
    """Pure-Python reference implementation of :func:`route_dag`."""
    n_phys = router._num_physical
    D = router._D
    adj = router._adj_sets
    nq1, nq2, nq3 = dag.nq1, dag.nq2, dag.nq3
    has3 = dag.has_three_qubit
    succ = dag.succ if forward else dag.pred
    indeg = [len(p) for p in (dag.pred if forward else dag.succ)]
    W = router.lookahead_weight
    ext_size = router.extended_set_size
    delta = router.decay_delta
    reset_interval = router.decay_reset_interval
    valve = router.release_valve_threshold or 10 * max(n_phys, 1)

    pi = list(layout)
    rpi = [0] * n_phys
    for token, physical in enumerate(pi):
        rpi[physical] = token

    front: dict[int, None] = {}
    # 每个逻辑比特上的前沿节点：两比特节点与三比特节点分开记录（对易感知 DAG
    # 中一个比特上可以有多个前沿两比特节点）。
    front2: list[list[int]] = [[] for _ in range(n_phys)]
    front3: list[list[int]] = [[] for _ in range(n_phys)]
    events: list[int] = []
    num_swaps = 0
    level = [0] * n_phys if record else []
    # 融合感知：每个物理比特上最后一个两比特操作（节点编号；SWAP 用负的唯一编号）。
    # 与第 7 轮一致，含三比特节点的 DAG 不使用融合奖励。
    fusion = 1.0 - router.swap_fusion_bonus
    use_fusion = fusion != 1.0 and not has3
    # 镜像门（MIRAGE）：mcost[n] >= 0 表示 n 是其比特对块在本方向上的末节点，紧跟一个
    # 同对 SWAP 时块 U 变成 U·SWAP，多出 mcost[n] 个 CZ（通用 SU(4) 块为 0）。
    mirror_on = router.mirror_gates and bool(dag.mirror_fwd)
    mcost = (dag.mirror_fwd if forward else dag.mirror_bwd) if mirror_on else []
    track_last = use_fusion or mirror_on
    bonus = router.swap_fusion_bonus
    mfusion = [1.0 - bonus * (3 - d) / 3 for d in range(4)]
    mlimit = [
        router.mirror_threshold + d * router.mirror_cz_penalty + _TIE_EPS
        for d in range(4)
    ]
    last_op = [-1] * n_phys
    op_counter = -2
    # 本轮 drain 中执行、可能镜像的块末节点；每个逻辑比特上最后执行的节点
    mcands: list[int] = []
    last_exec = [-1] * n_phys
    swap_cost = 0
    num_mirrors = 0

    def drain(stack: list[int]) -> None:
        nonlocal op_counter
        while stack:
            n = stack.pop()
            a = nq1[n]
            if a >= 0:
                c = nq3[n]
                if c < 0:
                    b = nq2[n]
                    pa, pb = pi[a], pi[b]
                    if pb not in adj[pa]:
                        front[n] = None
                        front2[a].append(n)
                        front2[b].append(n)
                        continue
                    if record:
                        lv = max(level[pa], level[pb]) + 1
                        level[pa] = level[pb] = lv
                    if track_last:
                        last_op[pa] = last_op[pb] = n
                    if mirror_on:
                        last_exec[a] = last_exec[b] = n
                        if mcost[n] >= 0:
                            mcands.append(n)
                else:
                    b = nq2[n]
                    pa, pb, pc = pi[a], pi[b], pi[c]
                    if cost3(D, pa, pb, pc) != 1:
                        front[n] = None
                        front3[a].append(n)
                        front3[b].append(n)
                        front3[c].append(n)
                        continue
                    if record:
                        lv = max(level[pa], level[pb], level[pc]) + 1
                        level[pa] = level[pb] = level[pc] = lv
                    if mirror_on:
                        op_counter -= 1
                        last_op[pa] = last_op[pb] = last_op[pc] = op_counter
                        last_exec[a] = last_exec[b] = last_exec[c] = n
            elif mirror_on:
                # 测量是调度栅栏：之前的块都不再与之后的 SWAP 相邻
                mcands.clear()
                for p in range(n_phys):
                    last_op[p] = -1
            if record:
                events.append(n)
            for s in succ[n]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)

    def apply_swap(x: int, y: int, ready: list[int]) -> int:
        """Apply SWAP ``(x, y)``; returns its CZ cost estimate."""
        nonlocal op_counter
        cost = 3
        if mirror_on:
            op = last_op[x]
            if op >= 0 and op == last_op[y] and mcost[op] >= 0:
                cost = mcost[op]
        tx, ty = rpi[x], rpi[y]
        pi[tx], pi[ty] = y, x
        rpi[x], rpi[y] = ty, tx
        # tx、ty 移动了：它们自己以及以它们为伙伴的比特的增量表失效
        tables[tx] = tables[ty] = None
        for u in fpart[tx]:
            tables[u] = None
        for u in partners[tx]:
            tables[u] = None
        for u in fpart[ty]:
            tables[u] = None
        for u in partners[ty]:
            tables[u] = None
        if track_last:
            op_counter -= 1
            last_op[x] = last_op[y] = op_counter
        for token in (tx, ty):
            for g in front2[token]:
                if g not in ready and pi[nq2[g]] in adj[pi[nq1[g]]]:
                    ready.append(g)
            if has3:
                for g in front3[token]:
                    if g not in ready and _node_cost(D, pi, nq1, nq2, nq3, g) == 1:
                        ready.append(g)
        return cost

    def retire(ready: list[int]) -> None:
        """Remove executable nodes from the front layer and drain them."""
        for g in ready:
            del front[g]
            c = nq3[g]
            if c < 0:
                front2[nq1[g]].remove(g)
                front2[nq2[g]].remove(g)
            else:
                for token in (nq1[g], nq2[g], c):
                    front3[token].remove(g)
        drain(ready)

    edges_at = router._edges_at
    # 每个逻辑比特：前沿两比特节点的另一端 fpart[t]（partners[t] 对扩展集同理），
    # 以及代价增量表 tables[t][z] = (前沿增量, 扩展集增量)：t 移到相邻物理比特
    # z 时，它的两比特节点距离之和的变化（不计与 z 上比特之间的节点）。表只取决于
    # t 与其伙伴的位置以及伙伴列表，其中任何一项变化时置 None（惰性重算）。
    fpart: list[list[int]] = [[] for _ in range(n_phys)]
    front_tokens: list[int] = []
    tables: list[dict[int, tuple[int, int]] | None] = [None] * n_phys
    has_part = [False] * n_phys

    def delta_table(t: int) -> dict[int, tuple[int, int]]:
        p = pi[t]
        dp = D[p]
        fps = [pi[o] for o in fpart[t]]
        eps = [pi[o] for o in partners[t]]
        table: dict[int, tuple[int, int]] = {}
        for z in router._G[p]:
            dz = D[z]
            df = 0
            for po in fps:
                if po != z:
                    df += dz[po] - dp[po]
            de = 0
            for po in eps:
                if po != z:
                    de += dz[po] - dp[po]
            table[z] = (df, de)
        tables[t] = table
        return table

    drain([n for n in range(len(indeg)) if indeg[n] == 0])

    decay = [1.0] * n_phys
    partners: list[list[int]] = [[] for _ in range(n_phys)]
    # 扩展集中的三比特节点按精确代价 cost3 计入（多种近似中实测最好）。
    ext3_of: list[list[int]] = [[] for _ in range(n_phys)]
    touched: list[int] = []
    ext: list[int] = []
    ext2: list[int] = []
    ext3: list[int] = []
    ext_dirty = True

    while front:
        if ext_dirty:
            old_partners = {token: partners[token] for token in touched}
            for token in touched:
                partners[token] = []
                ext3_of[token] = []
            ext = _extended_set(front, succ, indeg, nq1, ext_size)
            touched = []
            ext2 = []
            ext3 = []
            for g in ext:
                a, b = nq1[g], nq2[g]
                c = nq3[g]
                if c >= 0:
                    ext3.append(g)
                    for token in (a, b, c):
                        ext3_of[token].append(g)
                        touched.append(token)
                    continue
                ext2.append(g)
                partners[a].append(b)
                partners[b].append(a)
                touched.append(a)
                touched.append(b)
            ext_dirty = False
            # 前沿只在外层迭代之间变化：重建每个比特的前沿伙伴表
            old_fpart = {t: fpart[t] for t in front_tokens}
            for t in front_tokens:
                fpart[t] = []
            front_tokens = []
            for g in front:
                if nq3[g] < 0:
                    a, b = nq1[g], nq2[g]
                    fpart[a].append(b)
                    fpart[b].append(a)
                    front_tokens.append(a)
                    front_tokens.append(b)
            # 伙伴列表变化的比特：增量表失效
            for t, lst in old_partners.items():
                if partners[t] != lst:
                    tables[t] = None
            for t, lst in old_fpart.items():
                if fpart[t] != lst:
                    tables[t] = None
            for t in touched:
                if t not in old_partners:
                    tables[t] = None
                has_part[t] = True
            for t in front_tokens:
                if t not in old_fpart:
                    tables[t] = None
                has_part[t] = True
            for t in old_partners:
                has_part[t] = bool(fpart[t] or partners[t])
            for t in old_fpart:
                has_part[t] = bool(fpart[t] or partners[t])

        ready: list[int] = []
        if mcands:
            # 镜像门：刚执行的块末节点 n（其比特上之后没有再执行别的节点）后紧跟
            # 同对 SWAP 的代价只有 mcost[n]；若交换 n 的两个逻辑比特让前沿 + 扩展集
            # 的启发式代价下降超过 mlimit[mcost[n]]，就执行这个"免费" SWAP。
            e_scale = W / len(ext) if ext else 0.0
            for n in mcands:
                a, b = nq1[n], nq2[n]
                if last_exec[a] != n or last_exec[b] != n:
                    continue
                pa, pb = pi[a], pi[b]
                dpa, dpb = D[pa], D[pb]
                df = 0
                for o in fpart[a]:
                    if o != b:
                        po = pi[o]
                        df += dpb[po] - dpa[po]
                for o in fpart[b]:
                    if o != a:
                        po = pi[o]
                        df += dpa[po] - dpb[po]
                de = 0
                for o in partners[a]:
                    if o != b:
                        po = pi[o]
                        de += dpb[po] - dpa[po]
                for o in partners[b]:
                    if o != a:
                        po = pi[o]
                        de += dpa[po] - dpb[po]
                if has3:
                    if front3[a] or front3[b]:
                        df += _delta3(D, pi, dag, front3[a], front3[b], a, b, pa, pb)
                    if ext3_of[a] or ext3_of[b]:
                        de += _delta3(D, pi, dag, ext3_of[a], ext3_of[b], a, b, pa, pb)
                d = mcost[n]
                if -(df + de * e_scale) > mlimit[d]:
                    x, y = (pa, pb) if pa < pb else (pb, pa)
                    apply_swap(x, y, ready)
                    swap_cost += d
                    num_mirrors += 1
                    num_swaps += 1
                    if record:
                        events.append(-1 - (x * n_phys + y))
                    last_exec[a] = last_exec[b] = -1
            mcands.clear()
            if ready:
                retire(ready)
                decay = [1.0] * n_phys
                ext_dirty = True
                continue

        pending: list[tuple[int, int]] = []
        pending_cost = 0
        steps = 0
        while not ready:
            if len(pending) >= valve:
                for x, y in reversed(pending):
                    apply_swap(x, y, ready)
                ready.clear()
                pending = _release_valve(router, front, pi, dag)
                pending_cost = 0
                for x, y in pending:
                    pending_cost += apply_swap(x, y, ready)
                break

            f_sum = 0
            front3_nodes = False
            cands: dict[int, tuple[int, int]] = {}
            for g in front:
                c = nq3[g]
                pa, pb = pi[nq1[g]], pi[nq2[g]]
                if c < 0:
                    f_sum += D[pa][pb]
                    cands.update(edges_at[pa])
                    cands.update(edges_at[pb])
                else:
                    pc = pi[c]
                    f_sum += cost3(D, pa, pb, pc)
                    cands.update(edges_at[pa])
                    cands.update(edges_at[pb])
                    cands.update(edges_at[pc])
                    front3_nodes = True
            if not cands:
                raise ValueError(_DISCONNECTED)
            e_sum = 0
            for g in ext2:
                e_sum += D[pi[nq1[g]]][pi[nq2[g]]]
            for g in ext3:
                e_sum += cost3(D, pi[nq1[g]], pi[nq2[g]], pi[nq3[g]])
            e_scale = W / len(ext) if ext else 0.0

            best_score = float("inf")
            best: list[tuple[int, int]] = []
            for xy in cands.values():
                x, y = xy
                tx, ty = rpi[x], rpi[y]
                if has_part[tx]:
                    df, de = (tables[tx] or delta_table(tx))[y]
                    if has_part[ty]:
                        df2, de2 = (tables[ty] or delta_table(ty))[x]
                        df += df2
                        de += de2
                elif has_part[ty]:
                    df, de = (tables[ty] or delta_table(ty))[x]
                else:
                    df = de = 0
                if front3_nodes and (front3[tx] or front3[ty]):
                    df += _delta3(D, pi, dag, front3[tx], front3[ty], tx, ty, x, y)
                if ext3 and (ext3_of[tx] or ext3_of[ty]):
                    de += _delta3(D, pi, dag, ext3_of[tx], ext3_of[ty], tx, ty, x, y)
                score = f_sum + df + (e_sum + de) * e_scale
                dec = decay[x] if decay[x] > decay[y] else decay[y]
                score *= dec
                op = last_op[x]
                if use_fusion and op >= 0 and op == last_op[y]:
                    if not mcost:
                        score *= fusion
                    elif mcost[op] >= 0:
                        # 镜像感知：按该块的精确镜像代价折扣（0 个额外 CZ = 全额）
                        score *= mfusion[mcost[op]]
                if score < best_score - _TIE_EPS:
                    best_score = score
                    best = [xy]
                elif score <= best_score + _TIE_EPS:
                    best.append(xy)

            x, y = best[0] if len(best) == 1 else rng.choice(best)
            pending_cost += apply_swap(x, y, ready)
            pending.append((x, y))
            steps += 1
            if steps % reset_interval == 0:
                decay = [1.0] * n_phys
            else:
                decay[x] += delta
                decay[y] += delta

        num_swaps += len(pending)
        swap_cost += pending_cost
        if record:
            for x, y in pending:
                events.append(-1 - (x * n_phys + y))
                lv = max(level[x], level[y]) + 1
                level[x] = level[y] = lv
        retire(ready)
        decay = [1.0] * n_phys
        ext_dirty = True

    return _RouteResult(
        num_swaps=num_swaps,
        final_pi=pi,
        events=events,
        depth=max(level) if level else 0,
        swap_cost=swap_cost,
        num_mirrors=num_mirrors,
    )


def _delta3(
    D: list[list[int]],
    pi: list[int],
    dag: _RoutingDag,
    nodes_x: list[int],
    nodes_y: list[int],
    tx: int,
    ty: int,
    x: int,
    y: int,
) -> int:
    """Cost change of the three-qubit nodes on ``tx``/``ty`` if ``x, y`` swap.

    Each node is counted once even when it touches both tokens; after the
    SWAP token ``tx`` sits on ``y`` and ``ty`` on ``x``.
    """
    nq1, nq2, nq3 = dag.nq1, dag.nq2, dag.nq3
    total = 0
    for g in dict.fromkeys(nodes_x + nodes_y):
        ps = [
            y if tok == tx else x if tok == ty else pi[tok]
            for tok in (nq1[g], nq2[g], nq3[g])
        ]
        total += cost3(D, ps[0], ps[1], ps[2]) - cost3(
            D, pi[nq1[g]], pi[nq2[g]], pi[nq3[g]]
        )
    return total


def _extended_set(
    front: dict[int, None],
    succ: list[list[int]],
    indeg: list[int],
    nq1: list[int],
    limit: int,
) -> list[int]:
    """Breadth-first lookahead set of at most ``limit`` routing nodes.

    A successor joins only once all of its predecessors are in the front
    layer or already visited (temporary in-degree bookkeeping, restored
    before returning).  Measurements are walked through but not included.

    Args:
        front: Front-layer nodes (insertion ordered).
        succ: Successor lists in the routing direction.
        indeg: Remaining in-degrees (restored on return).
        nq1: First logical qubit per node (``< 0`` for measurements).
        limit: Maximum size of the set.

    Returns:
        The extended-set nodes in breadth-first order.
    """
    result: list[int] = []
    if limit == 0:
        return result
    visit = list(front)
    decremented: list[int] = []
    found = 0
    # 遍历时向 visit 追加是合法的：for 循环会继续访问新追加的节点
    for v in visit:
        for s in succ[v]:
            d = indeg[s] - 1
            indeg[s] = d
            decremented.append(s)
            if d == 0:
                if nq1[s] >= 0:
                    result.append(s)
                    found += 1
                    if found >= limit:
                        break
                visit.append(s)
        if found >= limit:
            break
    for s in decremented:
        indeg[s] += 1
    return result


# =============================================================================
# Release valve
# =============================================================================


def _release_valve(
    router: SABRE,
    front: dict[int, None],
    pi: list[int],
    dag: _RoutingDag,
) -> list[tuple[int, int]]:
    """SWAPs that make the cheapest front-layer node executable.

    For a two-qubit node both endpoints walk towards each other along a
    shortest path.  For a three-qubit node the qubit ``m`` minimising the
    distance sum stays put; the other two (nearer first) walk towards it
    along shortest paths until the three qubits are connected.

    Args:
        router: Router providing ``_D`` and ``_G``.
        front: Front-layer nodes (the first cheapest one is routed).
        pi: Current mapping ``pi[token] = physical``.
        dag: Routing DAG.

    Returns:
        SWAPs ``(x, y)`` with ``x < y``, in application order.

    Raises:
        ValueError: If the chosen node's qubits are disconnected.
    """
    D = router._D
    G = router._G
    nq1, nq2, nq3 = dag.nq1, dag.nq2, dag.nq3
    target = min(front, key=lambda g: _node_cost(D, pi, nq1, nq2, nq3, g))
    if nq3[target] < 0:
        pa, pb = pi[nq1[target]], pi[nq2[target]]
        path = _shortest_path(G, pa, pb)
        if path is None:
            raise ValueError(_DISCONNECTED)
        k = len(path) - 1
        left = k // 2
        right = (k - 1) - left
        swaps = [(path[i], path[i + 1]) for i in range(left)]
        swaps += [(path[k - i], path[k - i - 1]) for i in range(right)]
        return [(x, y) if x < y else (y, x) for x, y in swaps]

    toks = _tokens(nq1, nq2, nq3, target)
    pos = {tok: pi[tok] for tok in toks}
    middle = min(toks, key=lambda m: sum(D[pos[m]][pos[o]] for o in toks if o != m))
    ends = sorted(
        (o for o in toks if o != middle), key=lambda o: D[pos[o]][pos[middle]]
    )
    occupant = {p: tok for tok, p in enumerate(pi)}

    def connected() -> bool:
        a, b, c = (pos[tok] for tok in toks)
        return cost3(D, a, b, c) == 1

    out: list[tuple[int, int]] = []
    for end in ends:
        while not connected() and D[pos[end]][pos[middle]] > 1:
            path = _shortest_path(G, pos[end], pos[middle])
            if path is None:
                raise ValueError(_DISCONNECTED)
            x, y = path[0], path[1]
            tx, ty = occupant[x], occupant.get(y, -1)
            occupant[x], occupant[y] = ty, tx
            pos[end] = y
            if ty in pos:
                pos[ty] = x
            out.append((x, y) if x < y else (y, x))
    return out


def _shortest_path(G: list[list[int]], source: int, dest: int) -> list[int] | None:
    """Breadth-first shortest physical path ``[source, ..., dest]`` or ``None``.

    Args:
        G: Adjacency lists of the coupling graph (neighbour order decides
            between equally short paths).
        source: Start qubit.
        dest: End qubit.

    Returns:
        The path including both ends, or ``None`` if ``dest`` is unreachable.
    """
    parent = {source: source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == dest:
            break
        for neighbor in G[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if dest not in parent:
        return None
    path = [dest]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path
