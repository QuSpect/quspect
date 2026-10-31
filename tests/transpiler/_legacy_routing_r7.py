"""Frozen copy of the round-7 routing code (``evolve/07-phase``, ``af6228e``).

Reference implementation for the differential tests in
``test_routing_differential.py``: the unified routing core in
:mod:`quspect.transpiler._routing` must reproduce these loops bit for bit
(same SWAPs, same events, same final permutation, same depth).

Copied verbatim from ``quspect/transpiler/_mapping.py``
(``_build_routing_dag``, ``SABRE._route``, ``SABRE._extended_set``,
``SABRE._release_valve``, ``SABRE._shortest_path``) and
``quspect/transpiler/_toffoli.py`` (``route_with_toffoli``,
``_release_valve``); methods became functions taking the router as
``self``/``sabre``.  Do not edit, except to follow renames of the shared
data classes.
"""
# ruff: noqa: N806, C901, PLR0912, PLR0915

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

from quspect.transpiler._mapping import (
    SABRE,
    _MappingGate,
    _RouteResult,
    _RoutingDag,
)

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.gates.gate import Gate

_TIE_EPS = 1e-10


def legacy_build_routing_dag(
    gates: list[_MappingGate],
    num_logical: int,
    circuit: QuantumCircuit | None = None,
    *,
    commutation: bool = False,
) -> _RoutingDag:
    """Compress a gate list into the two-qubit/measurement routing DAG.

    A measurement is a scheduling fence: it depends on every earlier node
    and every later node depends on it, so multi-qubit final measurements
    are never interleaved with routing SWAPs on other qubits.

    With ``commutation`` (needs ``circuit`` to look up the gates), nodes
    that commute are not ordered: on each qubit, consecutive multi-qubit
    gates that act diagonally in the same basis there — ``Z`` (``cz``,
    ``cp``, ``crz``, ``rzz``, a ``cx``/``ccx`` control) or ``X`` (a
    ``cx``/``ccx`` target) — form a *commuting group*, and the single-qubit
    gates between them may belong to it if they are diagonal in the same
    basis (``rz``, ``t``… or ``x``, ``rx``…).  A group member only depends
    on the nodes before the group, a node after the group on all its
    members.  Single-qubit gates stay attached (as prefix) to the next node
    on their qubit: inside a group they commute with every member; gates
    that open a group (e.g. an ``h`` before a run of ``cp``) are emitted
    with its first node, which the other members then depend on.  Two
    gates diagonal in the same basis on every shared qubit commute, so any
    execution order of the DAG gives the same unitary.

    Args:
        gates: Logical gate list from :func:`_extract_gates`.
        num_logical: Number of logical qubits.
        circuit: Source circuit of ``gates`` (required for ``commutation``).
        commutation: Build the commutation-aware DAG.

    Returns:
        The compressed routing DAG.
    """
    from quspect.transpiler._highlevel import basis_on

    dag = _RoutingDag()
    pending: list[list[_MappingGate]] = [[] for _ in range(num_logical)]
    # 每个比特：当前对易组的成员、组的基（None 表示无法加入）、组之前的节点，
    # 以及待挂的单比特门是否都与组同基。
    members: list[list[int]] = [[] for _ in range(num_logical)]
    group_basis: list[str | None] = [None] * num_logical
    before: list[list[int]] = [[] for _ in range(num_logical)]
    pending_ok: list[bool] = [True] * num_logical

    def gate_of(g: _MappingGate) -> Gate | None:
        if g.gate is not None:
            return g.gate
        if circuit is None or g.index < 0:
            return None
        return circuit.instructions[g.index].gate

    for g in gates:
        if g.name == "measure":
            preds = {p for q in range(num_logical) for p in members[q]}
            pre = sorted(
                (h for q in range(num_logical) for h in pending[q]),
                key=lambda h: h.index,
            )
            for q in range(num_logical):
                pending[q] = []
            qs: tuple[int, ...] = ()
            q1 = q2 = q3 = -1
        else:
            if g.q3 >= 0:
                qs = (g.q1, g.q2, g.q3)
                dag.has_three_qubit = True
            elif g.is_two_qubit:
                qs = (g.q1, g.q2)
            else:
                if commutation and pending_ok[g.q1]:
                    gate = gate_of(g)
                    basis = None if gate is None else basis_on(gate, (g.q1,), g.q1)
                    if basis is None or basis != group_basis[g.q1]:
                        pending_ok[g.q1] = False
                pending[g.q1].append(g)
                continue
            q1, q2 = qs[0], qs[1]
            q3 = qs[2] if len(qs) == 3 else -1
            pre = sorted((h for q in qs for h in pending[q]), key=lambda h: h.index)
            preds = set()
            gate = gate_of(g) if commutation else None
            had_prefix = [bool(pending[q]) for q in qs]
            for q in qs:
                basis = None if gate is None else basis_on(gate, qs, q)
                if basis is not None and basis == group_basis[q] and pending_ok[q]:
                    preds.update(before[q])
                else:
                    preds.update(members[q])
                pending[q] = []

        idx = len(dag.nodes)
        dag.nodes.append(g)
        dag.nq1.append(q1)
        dag.nq2.append(q2)
        dag.nq3.append(q3)
        dag.prefix.append(pre)
        dag.succ.append([])
        dag.pred.append(sorted(preds))
        for p in dag.pred[idx]:
            dag.succ[p].append(idx)
        if q1 == -1:
            for q in range(num_logical):
                before[q] = members[q]
                members[q] = [idx]
                group_basis[q] = None
                pending_ok[q] = True
        else:
            for q, prefixed in zip(qs, had_prefix, strict=True):
                basis = None if gate is None else basis_on(gate, qs, q)
                if basis is not None and basis == group_basis[q] and pending_ok[q]:
                    members[q].append(idx)
                else:
                    # 新组的首节点带着组前的单比特门（如 h）时，这些门必须先于
                    # 组内所有成员执行，因此后续成员都依赖首节点。
                    before[q] = [idx] if prefixed else members[q]
                    members[q] = [idx]
                    group_basis[q] = basis
                pending_ok[q] = True

    dag.trailing = sorted(
        (h for q in range(num_logical) for h in pending[q]), key=lambda h: h.index
    )
    return dag


def legacy_route(
    self: SABRE,
    layout: list[int],
    dag: _RoutingDag,
    *,
    forward: bool,
    rng: random.Random,
    record: bool = False,
) -> _RouteResult:
    """Run one SABRE routing pass.

    Args:
        layout: Full permutation ``layout[token] = physical`` (logical
            qubits first, then ancillas).
        dag: Routing DAG.
        forward: Traverse the DAG forwards (``True``) or reversed.
        rng: Random source for tie-breaking.
        record: Record the execution/SWAP event sequence.

    Returns:
        Routing statistics, the final full permutation and (if
        ``record``) the events.
    """
    if dag.has_three_qubit:
        return legacy_route_with_toffoli(
            self, layout, dag, forward=forward, rng=rng, record=record
        )
    n_phys = self._num_physical
    D = self._D
    G = self._G
    adj = self._adj_sets
    nq1 = dag.nq1
    nq2 = dag.nq2
    succ = dag.succ if forward else dag.pred
    indeg = [len(p) for p in (dag.pred if forward else dag.succ)]
    W = self.lookahead_weight
    ext_size = self.extended_set_size
    delta = self.decay_delta
    reset_interval = self.decay_reset_interval
    valve = self.release_valve_threshold or 10 * max(n_phys, 1)

    pi = list(layout)
    rpi = [0] * n_phys
    for token, physical in enumerate(pi):
        rpi[physical] = token

    front: dict[int, None] = {}
    # 对易感知 DAG 中一个比特上可以同时有多个前沿门
    front_of: list[list[int]] = [[] for _ in range(n_phys)]
    events: list[int] = []
    num_swaps = 0
    level = [0] * n_phys if record else []
    # 融合感知：每个物理比特上最后一个两比特操作（节点编号；SWAP 用负的唯一编号）
    fusion = 1.0 - self.swap_fusion_bonus
    use_fusion = fusion != 1.0
    last_op = [-1] * n_phys
    op_counter = [-2]

    def drain(stack: list[int]) -> None:
        while stack:
            n = stack.pop()
            a = nq1[n]
            if a >= 0:
                b = nq2[n]
                pa, pb = pi[a], pi[b]
                if pb not in adj[pa]:
                    front[n] = None
                    front_of[a].append(n)
                    front_of[b].append(n)
                    continue
                if record:
                    lv = max(level[pa], level[pb]) + 1
                    level[pa] = level[pb] = lv
                if use_fusion:
                    last_op[pa] = last_op[pb] = n
            if record:
                events.append(n)
            for s in succ[n]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)

    def apply_swap(x: int, y: int, ready: list[int]) -> None:
        tx, ty = rpi[x], rpi[y]
        pi[tx], pi[ty] = y, x
        rpi[x], rpi[y] = ty, tx
        if use_fusion:
            op_counter[0] -= 1
            last_op[x] = last_op[y] = op_counter[0]
        for token in (tx, ty):
            for g in front_of[token]:
                if g not in ready and pi[nq2[g]] in adj[pi[nq1[g]]]:
                    ready.append(g)

    drain([n for n in range(len(indeg)) if indeg[n] == 0])

    decay = [1.0] * n_phys
    partners: list[list[int]] = [[] for _ in range(n_phys)]
    touched: list[int] = []
    ext: list[int] = []
    ext_dirty = True

    while front:
        if ext_dirty:
            for token in touched:
                partners[token] = []
            ext = _extended_set(front, succ, indeg, nq1, ext_size)
            touched = []
            for g in ext:
                a, b = nq1[g], nq2[g]
                partners[a].append(b)
                partners[b].append(a)
                touched.append(a)
                touched.append(b)
            ext_dirty = False

        pending: list[tuple[int, int]] = []
        ready: list[int] = []
        steps = 0
        while not ready:
            if len(pending) >= valve:
                for x, y in reversed(pending):
                    apply_swap(x, y, ready)
                ready.clear()
                pending = _release_valve_2q(self, front, pi, nq1, nq2)
                for x, y in pending:
                    apply_swap(x, y, ready)
                break

            f_sum = 0
            cands: dict[int, None] = {}
            for g in front:
                pa, pb = pi[nq1[g]], pi[nq2[g]]
                f_sum += D[pa][pb]
                for p in (pa, pb):
                    for z in G[p]:
                        cands[p * n_phys + z if p < z else z * n_phys + p] = None
            if not cands:
                raise ValueError(
                    "No routing SWAP is available for the current mapping. "
                    "The selected physical-qubit topology may be disconnected."
                )
            e_sum = 0
            for g in ext:
                e_sum += D[pi[nq1[g]]][pi[nq2[g]]]
            e_scale = W / len(ext) if ext else 0.0

            best_score = float("inf")
            best: list[int] = []
            for code in cands:
                x, y = divmod(code, n_phys)
                tx, ty = rpi[x], rpi[y]
                dx, dy = D[x], D[y]
                df = 0
                for g in front_of[tx]:
                    other = nq2[g] if nq1[g] == tx else nq1[g]
                    if other != ty:
                        po = pi[other]
                        df += dy[po] - dx[po]
                for g in front_of[ty]:
                    other = nq2[g] if nq1[g] == ty else nq1[g]
                    if other != tx:
                        po = pi[other]
                        df += dx[po] - dy[po]
                de = 0
                for other in partners[tx]:
                    if other != ty:
                        po = pi[other]
                        de += dy[po] - dx[po]
                for other in partners[ty]:
                    if other != tx:
                        po = pi[other]
                        de += dx[po] - dy[po]
                score = f_sum + df + (e_sum + de) * e_scale
                dec = decay[x] if decay[x] > decay[y] else decay[y]
                score *= dec
                if use_fusion and last_op[x] >= 0 and last_op[x] == last_op[y]:
                    score *= fusion
                if score < best_score - _TIE_EPS:
                    best_score = score
                    best = [code]
                elif score <= best_score + _TIE_EPS:
                    best.append(code)

            code = best[0] if len(best) == 1 else rng.choice(best)
            x, y = divmod(code, n_phys)
            apply_swap(x, y, ready)
            pending.append((x, y))
            steps += 1
            if steps % reset_interval == 0:
                decay = [1.0] * n_phys
            else:
                decay[x] += delta
                decay[y] += delta

        num_swaps += len(pending)
        for x, y in pending:
            if record:
                events.append(-1 - (x * n_phys + y))
                lv = max(level[x], level[y]) + 1
                level[x] = level[y] = lv
        for g in ready:
            del front[g]
            front_of[nq1[g]].remove(g)
            front_of[nq2[g]].remove(g)
        drain(ready)
        decay = [1.0] * n_phys
        ext_dirty = True

    return _RouteResult(
        num_swaps=num_swaps,
        final_pi=pi,
        events=events,
        depth=max(level) if level else 0,
    )


def _extended_set(
    front: dict[int, None],
    succ: list[list[int]],
    indeg: list[int],
    nq1: list[int],
    limit: int,
) -> list[int]:
    """Breadth-first lookahead set of at most ``limit`` two-qubit nodes.

    A successor joins only once all of its predecessors are in the
    front layer or already visited (temporary in-degree bookkeeping,
    restored before returning).
    """
    result: list[int] = []
    if limit == 0:
        return result
    visit = list(front)
    decremented: list[int] = []
    i = 0
    while i < len(visit) and len(result) < limit:
        for s in succ[visit[i]]:
            indeg[s] -= 1
            decremented.append(s)
            if indeg[s] == 0:
                if nq1[s] >= 0:
                    result.append(s)
                    if len(result) >= limit:
                        break
                visit.append(s)
        i += 1
    for s in decremented:
        indeg[s] += 1
    return result


def _release_valve_2q(
    self: SABRE,
    front: dict[int, None],
    pi: list[int],
    nq1: list[int],
    nq2: list[int],
) -> list[tuple[int, int]]:
    """SWAPs that bring the closest front-layer gate together.

    Both endpoints walk towards each other along a shortest path.

    Raises:
        ValueError: If the closest gate's qubits are disconnected.
    """
    D = self._D
    target = min(front, key=lambda g: D[pi[nq1[g]]][pi[nq2[g]]])
    pa, pb = pi[nq1[target]], pi[nq2[target]]
    path = _shortest_path(self, pa, pb)
    if path is None:
        raise ValueError(
            "No routing SWAP is available for the current mapping. "
            "The selected physical-qubit topology may be disconnected."
        )
    k = len(path) - 1
    left = k // 2
    right = (k - 1) - left
    swaps = [(path[i], path[i + 1]) for i in range(left)]
    swaps += [(path[k - i], path[k - i - 1]) for i in range(right)]
    return [(x, y) if x < y else (y, x) for x, y in swaps]


def _shortest_path(self: SABRE, source: int, dest: int) -> list[int] | None:
    """Shortest physical path ``[source, ..., dest]`` or ``None``."""
    parent = {source: source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == dest:
            break
        for neighbor in self._G[node]:
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


def legacy_route_with_toffoli(
    sabre: SABRE,
    layout: list[int],
    dag: _RoutingDag,
    *,
    forward: bool,
    rng: random.Random,
    record: bool = False,
) -> _RouteResult:
    """One SABRE routing pass over a DAG with three-qubit ``ccx`` nodes.

    Same algorithm and parameters as :meth:`SABRE._route` (front layer,
    bounded extended set, decay, release valve); a three-qubit node counts
    ``min_m (D[m][a] + D[m][c]) - 1`` in the front term and the sum of its
    three pairwise distances in the extended-set term.

    Args:
        sabre: The configured router (distance matrix, parameters).
        layout: Full permutation ``layout[token] = physical``.
        dag: Routing DAG (``dag.nq3[n] >= 0`` marks a three-qubit node).
        forward: Traverse the DAG forwards (``True``) or reversed.
        rng: Random source for tie-breaking.
        record: Record the execution/SWAP event sequence.

    Returns:
        Routing statistics, the final permutation and (if ``record``) the
        events, as :class:`~quspect.transpiler._mapping._RouteResult`.

    Raises:
        ValueError: If no routing SWAP is available (disconnected topology).
    """

    n_phys = sabre._num_physical
    D = sabre._D  # noqa: N806 - SABRE paper notation
    G = sabre._G  # noqa: N806
    nq1, nq2, nq3 = dag.nq1, dag.nq2, dag.nq3
    succ = dag.succ if forward else dag.pred
    indeg = [len(p) for p in (dag.pred if forward else dag.succ)]
    W = sabre.lookahead_weight  # noqa: N806
    ext_size = sabre.extended_set_size
    delta = sabre.decay_delta
    reset_interval = sabre.decay_reset_interval
    valve = sabre.release_valve_threshold or 10 * max(n_phys, 1)

    pi = list(layout)
    rpi = [0] * n_phys
    for token, physical in enumerate(pi):
        rpi[physical] = token

    def cost3(pa: int, pb: int, pc: int) -> int:
        dab, dac, dbc = D[pa][pb], D[pa][pc], D[pb][pc]
        return min(dab + dac, dab + dbc, dac + dbc) - 1

    def cost(g: int) -> int:
        c = nq3[g]
        if c >= 0:
            return cost3(pi[nq1[g]], pi[nq2[g]], pi[c])
        return D[pi[nq1[g]]][pi[nq2[g]]]

    front: dict[int, None] = {}
    front_of = [-1] * n_phys
    events: list[int] = []
    num_swaps = 0
    level = [0] * n_phys if record else []

    def tokens(g: int) -> tuple[int, ...]:
        c = nq3[g]
        return (nq1[g], nq2[g]) if c < 0 else (nq1[g], nq2[g], c)

    def drain(stack: list[int]) -> None:
        while stack:
            n = stack.pop()
            if nq1[n] >= 0:
                if cost(n) != 1:
                    front[n] = None
                    for token in tokens(n):
                        front_of[token] = n
                    continue
                if record:
                    ps = [pi[token] for token in tokens(n)]
                    lv = max(level[p] for p in ps) + 1
                    for p in ps:
                        level[p] = lv
            if record:
                events.append(n)
            for s in succ[n]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)

    def apply_swap(x: int, y: int, ready: list[int]) -> None:
        tx, ty = rpi[x], rpi[y]
        pi[tx], pi[ty] = y, x
        rpi[x], rpi[y] = ty, tx
        for token in (tx, ty):
            g = front_of[token]
            if g >= 0 and g not in ready and cost(g) == 1:
                ready.append(g)

    drain([n for n in range(len(indeg)) if indeg[n] == 0])

    decay = [1.0] * n_phys
    partners: list[list[int]] = [[] for _ in range(n_phys)]
    touched: list[int] = []
    ext: list[int] = []
    # 扩展集中的三比特节点按精确代价 cost3 计入（多种近似中实测最好）。
    ext3_of: list[list[int]] = [[] for _ in range(n_phys)]
    ext3: list[int] = []
    ext_dirty = True

    while front:
        if ext_dirty:
            for token in touched:
                partners[token] = []
                ext3_of[token] = []
            ext = _extended_set(front, succ, indeg, nq1, ext_size)
            touched = []
            ext3 = []
            for g in ext:
                if nq3[g] >= 0:
                    ext3.append(g)
                    for token in tokens(g):
                        ext3_of[token].append(g)
                        touched.append(token)
                    continue
                a, b = nq1[g], nq2[g]
                partners[a].append(b)
                partners[b].append(a)
                touched.append(a)
                touched.append(b)
            ext_dirty = False

        pending: list[tuple[int, int]] = []
        ready: list[int] = []
        steps = 0
        while not ready:
            if len(pending) >= valve:
                for x, y in reversed(pending):
                    apply_swap(x, y, ready)
                ready.clear()
                pending = _release_valve_3q(sabre, front, pi, nq1, nq2, nq3)
                for x, y in pending:
                    apply_swap(x, y, ready)
                break

            f_sum = 0
            cands: dict[int, None] = {}
            for g in front:
                f_sum += cost(g)
                for token in tokens(g):
                    p = pi[token]
                    for z in G[p]:
                        cands[p * n_phys + z if p < z else z * n_phys + p] = None
            if not cands:
                raise ValueError(
                    "No routing SWAP is available for the current mapping. "
                    "The selected physical-qubit topology may be disconnected."
                )
            e_sum = 0
            for g in ext:
                e_sum += cost(g)
            e_scale = W / len(ext) if ext else 0.0

            best_score = float("inf")
            best: list[int] = []
            for code in cands:
                x, y = divmod(code, n_phys)
                tx, ty = rpi[x], rpi[y]
                dx, dy = D[x], D[y]
                df = 0
                gx, gy = front_of[tx], front_of[ty]
                for g in (gx, gy) if gx != gy else (gx,):
                    if g < 0:
                        continue
                    if nq3[g] >= 0:
                        # 交换后的物理位置：tx → y，ty → x。
                        ps = [
                            y if tok == tx else x if tok == ty else pi[tok]
                            for tok in tokens(g)
                        ]
                        df += cost3(ps[0], ps[1], ps[2]) - cost(g)
                    elif g == gx:
                        other = nq2[g] if nq1[g] == tx else nq1[g]
                        if other != ty:
                            po = pi[other]
                            df += dy[po] - dx[po]
                    else:
                        other = nq2[g] if nq1[g] == ty else nq1[g]
                        if other != tx:
                            po = pi[other]
                            df += dx[po] - dy[po]
                de = 0
                for other in partners[tx]:
                    if other != ty:
                        po = pi[other]
                        de += dy[po] - dx[po]
                for other in partners[ty]:
                    if other != tx:
                        po = pi[other]
                        de += dx[po] - dy[po]
                if ext3:
                    seen3 = set(ext3_of[tx]) | set(ext3_of[ty])
                    for g in seen3:
                        ps = [
                            y if tok == tx else x if tok == ty else pi[tok]
                            for tok in tokens(g)
                        ]
                        de += cost3(ps[0], ps[1], ps[2]) - cost(g)
                score = f_sum + df + (e_sum + de) * e_scale
                dec = decay[x] if decay[x] > decay[y] else decay[y]
                score *= dec
                if score < best_score - _TIE_EPS:
                    best_score = score
                    best = [code]
                elif score <= best_score + _TIE_EPS:
                    best.append(code)

            code = best[0] if len(best) == 1 else rng.choice(best)
            x, y = divmod(code, n_phys)
            apply_swap(x, y, ready)
            pending.append((x, y))
            steps += 1
            if steps % reset_interval == 0:
                decay = [1.0] * n_phys
            else:
                decay[x] += delta
                decay[y] += delta

        num_swaps += len(pending)
        for x, y in pending:
            if record:
                events.append(-1 - (x * n_phys + y))
                lv = max(level[x], level[y]) + 1
                level[x] = level[y] = lv
        for g in ready:
            del front[g]
            for token in tokens(g):
                front_of[token] = -1
        drain(ready)
        decay = [1.0] * n_phys
        ext_dirty = True

    return _RouteResult(
        num_swaps=num_swaps,
        final_pi=pi,
        events=events,
        depth=max(level) if level else 0,
    )


def _release_valve_3q(
    sabre: SABRE,
    front: dict[int, None],
    pi: list[int],
    nq1: list[int],
    nq2: list[int],
    nq3: list[int],
) -> list[tuple[int, int]]:
    """SWAPs that make the cheapest front-layer node executable.

    A two-qubit node is handled like :meth:`SABRE._release_valve`.  For a
    three-qubit node the qubit ``m`` minimising the distance sum stays put;
    the other two walk towards it along shortest paths until the three
    qubits are connected.

    Raises:
        ValueError: If the chosen node's qubits are disconnected.
    """
    D = sabre._D  # noqa: N806

    def cost(g: int) -> int:
        a, b, c = pi[nq1[g]], pi[nq2[g]], nq3[g]
        if c < 0:
            return D[a][b]
        pc = pi[c]
        dab, dac, dbc = D[a][b], D[a][pc], D[b][pc]
        return min(dab + dac, dab + dbc, dac + dbc) - 1

    target = min(front, key=cost)
    if nq3[target] < 0:
        return _release_valve_2q(sabre, {target: None}, pi, nq1, nq2)

    toks = (nq1[target], nq2[target], nq3[target])
    pos = {tok: pi[tok] for tok in toks}
    middle = min(toks, key=lambda m: sum(D[pos[m]][pos[o]] for o in toks if o != m))
    ends = sorted(
        (o for o in toks if o != middle), key=lambda o: D[pos[o]][pos[middle]]
    )
    occupant = {p: tok for tok, p in enumerate(pi)}

    def connected() -> bool:
        a, b, c = (pos[tok] for tok in toks)
        dab, dac, dbc = D[a][b], D[a][c], D[b][c]
        return min(dab + dac, dab + dbc, dac + dbc) == 2

    swaps: list[tuple[int, int]] = []
    for end in ends:
        while not connected() and D[pos[end]][pos[middle]] > 1:
            path = _shortest_path(sabre, pos[end], pos[middle])
            if path is None:
                raise ValueError(
                    "No routing SWAP is available for the current mapping. "
                    "The selected physical-qubit topology may be disconnected."
                )
            x, y = path[0], path[1]
            tx, ty = occupant[x], occupant.get(y, -1)
            occupant[x], occupant[y] = ty, tx
            pos[end] = y
            if ty in pos:
                pos[ty] = x
            swaps.append((x, y) if x < y else (y, x))
    return swaps
