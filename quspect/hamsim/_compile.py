"""``compile_evolution``: ``exp(-iHt)`` to a circuit for a compilation target."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from quspect.circuit.circuit import QuantumCircuit
from quspect.hamsim._formulas import (
    STATE_ERROR_MAX_QUBITS,
    TERM_ORDERS,
    Rotation,
    commutator_prefactor,
    order_terms,
    product_formula,
    state_error,
)
from quspect.hamsim._network import (
    connected_active_set,
    ladder_circuit,
    long_path,
    merge_commuting,
    synthesize_network,
)
from quspect.hamsim._terms import PauliTerm, pauli_terms
from quspect.transpiler._target import CompilationTarget
from quspect.transpiler._transpile import TranspileResult, transpile

DEFAULT_NATIVE_GATES = frozenset(
    {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}
)

METHODS = ("network", "ladder")

#: Settings ``(window, front, next_weight, decay, block_mode, block_bias)``
#: tried by the network candidate (the cheapest result is kept; see
#: :func:`~quspect.hamsim.synthesize_network`).  The greedy synthesis is
#: sensitive to its look-ahead (the best single setting needs 4–12 % more
#: CX than the best of a 72–108 setting grid); the first twelve were picked
#: greedily from such grids (all-to-all and device graphs, within about 1 %
#: of the whole grid), the last four complete the earlier default settings.
#: The block settings (``block_mode=1``) also consider emitting two-qubit
#: blocks on coupling edges in place or with a mirror SWAP.  Runs are
#: pruned by branch and bound, so the portfolio costs far less than its
#: size suggests.
NETWORK_CONFIGS: tuple[tuple[int, int, int, int, int, int], ...] = (
    (32, 256, 1024, 850, 0, 1024),
    (64, 64, 512, 1000, 1, 1024),
    (64, 16, 1024, 1000, 0, 1024),
    (32, 64, 1024, 850, 1, 1024),
    (16, 64, 1024, 700, 0, 1024),
    (64, 16, 512, 1000, 0, 1024),
    (16, 16, 512, 1000, 0, 1024),
    (16, 64, 512, 850, 1, 1024),
    (16, 256, 256, 700, 0, 1024),
    (64, 64, 1024, 1000, 1, 1024),
    (16, 64, 512, 850, 0, 1024),
    (16, 16, 1024, 850, 1, 1024),
    (32, 16, 512, 900, 0, 1024),
    (16, 16, 1024, 1000, 0, 1024),
    (32, 64, 512, 800, 0, 1024),
    (16, 64, 512, 800, 0, 1024),
)

#: Further settings of the default portfolio, tried after
#: :data:`NETWORK_CONFIGS` within :data:`PORTFOLIO_BUDGET`: longer windows
#: (up to 128) and the full grid of fronts, weights, decays and block modes.
#: Small formulas (e.g. single-step UCCSD ansätze) gain 2–7 % from them.
EXTENDED_CONFIGS: tuple[tuple[int, int, int, int, int, int], ...] = tuple(
    cfg
    for cfg in (
        (w, f, nw, d, bm, 1024)
        for w in (128, 64, 32, 16, 8)
        for nw in (256, 512, 1024, 2048)
        for d in (1000, 850, 700)
        for f in (64, 16, 256)
        for bm in (1, 0)
    )
    if cfg not in NETWORK_CONFIGS
)

#: Work budget of the default portfolio, in textbook CX × runs: the
#: first ``budget // (textbook CX × layouts × graphs)`` settings of
#: :data:`EXTENDED_CONFIGS` are added to :data:`NETWORK_CONFIGS`.
PORTFOLIO_BUDGET = 2_000_000

#: Network runs within ``FINALIST_SLACK / 1024`` of the fewest CX are
#: finalists; the :data:`FINALISTS` cheapest are transpiled and the fewest
#: two-qubit gates after transpilation wins (its savings differ by run).
FINALIST_SLACK = 31
FINALISTS = 4

#: The network candidate is abandoned once its CX count exceeds this
#: multiple of the ladder candidate's two-qubit count (it cannot win).
NETWORK_ABORT_RATIO = 1.6

#: Coupling graphs the network candidate is synthesized on: ``"device"``
#: (the coupling graph itself) and ``"path"`` (a long simple path of it,
#: see :func:`~quspect.hamsim.long_path`).  The greedy synthesis is not
#: monotone in the edge set: on a denser graph the frame moves Paulis along
#: many routes and the images of later rotations spread, so a sparser
#: spanning subgraph can need far fewer CX (e.g. a 32-qubit Jordan–Wigner
#: Hubbard chain: 4 629 CX on a serpentine path of a 5×7 grid against
#: 11 040 on the whole grid for four Trotter steps).  Every CX of the
#: subgraph is a coupling edge, so its output is valid on the device.
NETWORK_GRAPHS = ("device", "path")

#: With ``term_order="auto"`` the later term orders' network runs are
#: abandoned above this multiple of the best two-qubit count so far.
SELECT_ABORT_RATIO = 1.4

#: Formulas of more than four steps whose textbook CX count exceeds this
#: are costed by :func:`select_term_order` from two and four steps
#: (steady-state slope) instead of being compiled in full.
SELECT_FULL_CX = 60_000

#: Formulas with more textbook CX than this compile the network first.
LARGE_FORMULA_CX = 50_000

#: ... and skip the ladder when the network needs fewer than
#: ``1 / LADDER_SKIP_RATIO`` of the textbook CX.
LADDER_SKIP_RATIO = 4


@dataclass
class EvolutionResult:
    """Output of :func:`compile_evolution`.

    Attributes:
        circuit: Compiled circuit on the target's physical qubits; logical
            qubit ``q`` starts on ``initial_layout[q]`` and ends on
            ``final_layout[q]`` (other physical qubits start in ``|0⟩`` and
            are returned to it).
        initial_layout: ``initial_layout[logical] = physical`` at the start.
        final_layout: ``final_layout[logical] = physical`` at the end.
        order: Product-formula order (1, 2 or 4).
        steps: Number of Trotter steps.
        term_order: Term order of the formula (``"input"`` / ``"support"``;
            the selected one for ``"auto"``).
        rotations: The product formula as a rotation sequence (time order,
            before the exact reordering of commuting rotations); its
            unitary equals the circuit's up to layout and global phase.
        method: Selected synthesis (``"network"`` or ``"ladder"``).
        two_qubit_gates: Two-qubit gates of ``circuit``.
        two_qubit_depth: Two-qubit depth of ``circuit``.
        error_estimate: Estimated product-formula error (``None`` when
            ``steps`` was given without ``eps``).
        error_metric: How ``error_estimate`` was obtained (``"state"``:
            simulated state error; ``"bound"``: commutator bound).
        candidates: ``{label: two-qubit gates}`` of every compiled
            candidate.
        transpile_result: The :class:`TranspileResult` of the selected
            candidate.
        network_layout: Initial layout of the network candidate when it
            was selected (pass it as ``layouts=[...]`` to reproduce it).
        network_config: Its synthesis setting (for ``network_configs``).
        network_graph: The coupling graph it was synthesized on (``"device"``
            or ``"path"``, for ``network_graphs``).
        term_order_table: With ``term_order="auto"``: ``{term order:
            (steps, error estimate, estimated two-qubit gates)}``.
    """

    circuit: QuantumCircuit
    initial_layout: list[int]
    final_layout: list[int]
    order: int
    steps: int
    term_order: str
    rotations: list[Rotation]
    method: str
    two_qubit_gates: int
    two_qubit_depth: int
    error_estimate: float | None = None
    error_metric: str | None = None
    candidates: dict[str, int] = field(default_factory=dict)
    transpile_result: TranspileResult | None = None
    network_layout: list[int] | None = None
    network_config: tuple[int, int, int, int, int, int] | None = None
    network_graph: str | None = None
    term_order_table: dict[str, tuple[int, float, int]] | None = None


def _full_target(n: int) -> CompilationTarget:
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return CompilationTarget.from_coupling_map(
        max(n, 1), edges, native_gates=set(DEFAULT_NATIVE_GATES)
    )


def _edges(target: CompilationTarget) -> list[tuple[int, int]]:
    return sorted({(min(a, b), max(a, b)) for a, b in target.coupling_graph if a != b})


def _is_full(target: CompilationTarget, n: int) -> bool:
    edges = set(_edges(target))
    return all((a, b) in edges for a in range(n) for b in range(a + 1, n))


def _min_steps(
    err: Callable[[int], float], eps: float, r_max: int
) -> tuple[int, float]:
    """Smallest ``r`` with ``err(r) ≤ eps`` (doubling, then bisection)."""
    probes: dict[int, float] = {}

    def e(r: int) -> float:
        if r not in probes:
            probes[r] = err(r)
        return probes[r]

    r, lo = 1, 0
    while e(r) > eps:
        if r >= r_max:
            raise ValueError(f"no step count up to {r_max} reaches eps = {eps}")
        lo, r = r, min(2 * r, r_max)
    hi = r
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if e(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi, e(hi)


def _layout_candidates(
    n: int,
    terms: Sequence[PauliTerm],
    rotations: Sequence[Rotation],
    target: CompilationTarget,
    seed: int,
    trials: int,
) -> list[list[int]]:
    """Initial layouts to try: low-dilation embeddings of the interaction graph
    (consecutive support qubits of each term) and SABRE's layout of the ladder
    circuit of one step."""
    from quspect.transpiler._embedding import dilation_layouts

    edges = _edges(target)
    big = target.num_qubits
    adj: list[list[int]] = [[] for _ in range(big)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for row in adj:
        row.sort()
    weights: dict[tuple[int, int], int] = {}
    for _, qs, _ in terms:
        s = sorted(qs)
        for a, b in zip(s, s[1:], strict=False):
            weights[(a, b)] = weights.get((a, b), 0) + 1
    out: list[list[int]] = []
    if weights:
        from collections import deque

        dist: list[list[int]] = []
        for src in range(big):
            d = [big + 1] * big
            d[src] = 0
            dq = deque([src])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if d[v] > d[u] + 1:
                        d[v] = d[u] + 1
                        dq.append(v)
            dist.append(d)
        found = dilation_layouts(
            n,
            [(a, b, w) for (a, b), w in sorted(weights.items())],
            adj,
            dist,
            seed=seed,
            count=trials,
        )
        for _, lay in found:
            out.append(_complete_layout(list(lay), big, adj))
    ladder = ladder_circuit(product_formula(terms, 1.0, 1, 1), n)
    _pad(ladder, range(n))
    res = transpile(ladder, target, optimization_level=1, seed=seed)
    out.append(list(res.initial_layout)[:n])
    uniq: list[list[int]] = []
    for lay in out:
        if lay not in uniq:
            uniq.append(lay)
    return uniq


def _complete_layout(lay: list[int], big: int, adj: list[list[int]]) -> list[int]:
    """Put unplaced (``-1``) logical qubits on free physical qubits near placed ones."""
    used = {p for p in lay if p >= 0}
    for q, p in enumerate(lay):
        if p >= 0:
            continue
        pick = -1
        for u in sorted(used):
            for v in adj[u]:
                if v not in used:
                    pick = v
                    break
            if pick >= 0:
                break
        if pick < 0:
            pick = min(v for v in range(big) if v not in used)
        lay[q] = pick
        used.add(pick)
    return lay


def _pad(qc: QuantumCircuit, qubits: Sequence[int] | range) -> None:
    """Make every qubit in ``qubits`` part of the circuit (identity gates)."""
    for q in qubits:
        qc.id(q)


def _norm_config(cfg: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    """``(window, front, next_weight, decay[, block_mode, block_bias])``."""
    c = tuple(int(v) for v in cfg)
    if len(c) == 4:
        c = c + (0, 1024)
    if len(c) != 6:
        raise ValueError("a network config has 4 or 6 integers")
    return c


def _resolve_metric(error_metric: str, n: int) -> str:
    metric = error_metric
    if metric == "auto":
        metric = "state" if n <= STATE_ERROR_MAX_QUBITS else "bound"
    if metric not in ("state", "bound"):
        raise ValueError("error_metric must be 'auto', 'state' or 'bound'")
    return metric


def _steps_for_eps(
    n: int,
    terms: Sequence[PauliTerm],
    time: float,
    order: int,
    eps: float,
    metric: str,
    max_steps: int,
    reference: Any = None,
) -> tuple[int, float]:
    """Fewest steps whose estimated error is at most ``eps``."""
    if metric == "state":
        from quspect.hamsim._formulas import _exact_evolve, _test_states

        ref = (
            _exact_evolve(n, terms, time, _test_states(n, 2, 20261102))
            if reference is None
            else reference
        )

        def err(r: int) -> float:
            return state_error(
                n, terms, time, product_formula(terms, time, r, order), reference=ref
            )

    else:
        if order == 4:
            raise ValueError("error_metric='bound' supports order 1 and 2")
        # the bound is C / r**order: one evaluation of C
        pref = commutator_prefactor(n, terms, time, order)

        def err(r: int) -> float:
            return pref / float(r**order)

    return _min_steps(err, eps, max_steps)


@dataclass
class _Candidate:
    key: tuple[int, int]
    method: str
    circuit: QuantumCircuit
    init: list[int]
    fin: list[int]
    res: TranspileResult
    layout: list[int] | None = None
    config: tuple[int, int, int, int, int, int] | None = None
    graph: str | None = None


def _naive_cx(rotations: Sequence[Rotation]) -> int:
    return sum(2 * (len(q) - 1) for _, q, _ in rotations)


def _network_graphs(
    target: CompilationTarget, n: int, names: Sequence[str]
) -> list[tuple[str, list[tuple[int, int]], CompilationTarget]]:
    """``(name, edges, target of those edges)`` for each requested graph.

    ``"path"`` is skipped on all-to-all connectivity, when the coupling
    graph is itself a path, or when the longest path found has fewer than
    ``n`` qubits.
    """
    edges = _edges(target)
    out: list[tuple[str, list[tuple[int, int]], CompilationTarget]] = []
    for name in names:
        if name == "device":
            out.append(("device", edges, target))
        elif name == "path":
            if _is_full(target, n):
                continue
            path = long_path(target.num_qubits, edges)
            if len(path) < n:
                continue
            pe = sorted(
                {(min(a, b), max(a, b)) for a, b in zip(path, path[1:], strict=False)}
            )
            if len(pe) == len(edges):
                continue
            sub = CompilationTarget.from_coupling_map(
                target.num_qubits, pe, native_gates=set(DEFAULT_NATIVE_GATES)
            )
            out.append(("path", pe, sub))
        else:
            raise ValueError(
                f"unknown network graph {name!r}; expected one of {NETWORK_GRAPHS}"
            )
    if not out:  # e.g. only "path" on all-to-all connectivity
        out.append(("device", edges, target))
    return out


def _compile_rotations(
    n: int,
    terms: Sequence[PauliTerm],
    merged: Sequence[Rotation],
    target: CompilationTarget,
    *,
    methods: Sequence[str],
    optimization_level: int,
    ladder_level: int | None,
    seed: int,
    layout_trials: int,
    network_configs: Sequence[Sequence[int]] | None,
    layouts: Sequence[Sequence[int]] | None,
    finalize_permutation: bool,
    network_graphs: Sequence[str] = NETWORK_GRAPHS,
    bound: int = 0,
    portfolio_budget: int = PORTFOLIO_BUDGET,
) -> list[_Candidate]:
    """Compile a (merged) rotation sequence with every requested method.

    Normally the ladder is compiled first and bounds the network runs
    (:data:`NETWORK_ABORT_RATIO`).  For formulas above
    :data:`LARGE_FORMULA_CX` textbook CX the network goes first (bounded by
    ``NETWORK_ABORT_RATIO`` times the textbook count) and the expensive
    ladder transpilation is skipped when the network needs fewer than
    ``1 / LADDER_SKIP_RATIO`` of the textbook CX.  ``bound`` (``> 0``)
    additionally abandons network runs above that CX count.
    """
    out: list[_Candidate] = []
    naive = _naive_cx(merged)
    want_ladder = "ladder" in methods or not merged
    want_network = "network" in methods and bool(merged)
    large = naive > LARGE_FORMULA_CX

    def ladder() -> _Candidate:
        qc = ladder_circuit(merged, n)
        _pad(qc, range(n))
        res = transpile(
            qc, target, optimization_level=ladder_level or optimization_level, seed=seed
        )
        key = (res.stats.two_qubit_gates, res.stats.two_qubit_depth)
        return _Candidate(
            key,
            "ladder",
            res.circuit,
            list(res.initial_layout)[:n],
            list(res.final_layout)[:n],
            res,
        )

    if want_ladder and not (large and want_network):
        out.append(ladder())
    if want_network:
        # without a ladder candidate the network is the only result: no abort bound
        limit = 0
        if out:
            limit = int(NETWORK_ABORT_RATIO * out[0].key[0]) + 1
        elif want_ladder:
            limit = int(NETWORK_ABORT_RATIO * naive) + 1
        if bound > 0:
            limit = bound if limit == 0 else min(limit, bound)
        net = _network_candidate(
            n,
            terms,
            merged,
            target,
            optimization_level=optimization_level,
            seed=seed,
            layout_trials=layout_trials,
            network_configs=network_configs,
            layouts=layouts,
            finalize_permutation=finalize_permutation,
            limit=limit,
            network_graphs=network_graphs,
            portfolio_budget=portfolio_budget,
        )
        if net is not None:
            out.append(net)
    if want_ladder and large and want_network:
        net_cx = next((c.key[0] for c in out if c.method == "network"), None)
        if net_cx is None or net_cx * LADDER_SKIP_RATIO >= naive or not merged:
            out.append(ladder())
    return out


def _network_candidate(
    n: int,
    terms: Sequence[PauliTerm],
    merged: Sequence[Rotation],
    target: CompilationTarget,
    *,
    optimization_level: int,
    seed: int,
    layout_trials: int,
    network_configs: Sequence[Sequence[int]] | None,
    layouts: Sequence[Sequence[int]] | None,
    finalize_permutation: bool,
    limit: int,
    network_graphs: Sequence[str] = NETWORK_GRAPHS,
    portfolio_budget: int = PORTFOLIO_BUDGET,
) -> _Candidate | None:
    plan: list[tuple[str, list[tuple[int, int]], list[list[int]]]] = []
    for gname, gedges, gtarget in _network_graphs(target, n, network_graphs):
        if layouts is not None:
            lays = [list(lay) for lay in layouts]
        elif _is_full(target, n):
            lays = [list(range(n))]
        elif gname == "path":
            # layouts on the path as a line of its own (the qubits off the path are
            # not connected to it), mapped back; logical qubits in index order along
            # the path first (Jordan–Wigner chains)
            path = long_path(target.num_qubits, _edges(target))
            line = CompilationTarget.from_coupling_map(
                len(path),
                [(k, k + 1) for k in range(len(path) - 1)],
                native_gates=set(DEFAULT_NATIVE_GATES),
            )
            along = path[:n]
            lays = [along]
            for lay in _layout_candidates(n, terms, merged, line, seed, layout_trials):
                mapped = [path[k] for k in lay]
                if mapped not in lays:
                    lays.append(mapped)
        else:
            lays = _layout_candidates(n, terms, merged, gtarget, seed, layout_trials)
        plan.append((gname, gedges, lays))
    if network_configs is None:  # default portfolio within the work budget
        runs = max(1, _naive_cx(merged) * sum(len(lays) for _, _, lays in plan))
        extra = min(len(EXTENDED_CONFIGS), max(0, portfolio_budget) // runs)
        network_configs = NETWORK_CONFIGS + EXTENDED_CONFIGS[:extra]
    configs = [_norm_config(c) for c in network_configs]
    if not configs:
        raise ValueError("network_configs is empty")
    # branch and bound with a small slack: runs above (1 + slack) times the best CX
    # count so far cannot be among the finalists; the cheapest few finalists are
    # all transpiled (the transpiler's savings differ between them) and the
    # fewest two-qubit gates after transpilation wins
    done: list[tuple[int, int, Any, list[int], tuple[int, ...], str]] = []
    best_cx = 0
    runs = 0  # run counter: stable order of equal CX counts
    seen: set[tuple[Any, ...]] = set()
    for gname, gedges, lays in plan:
        for lay in lays:
            used = sorted({lay[q] for _, qs, _ in merged for q in qs})
            active = connected_active_set(target.num_qubits, gedges, used)
            for cfg in configs:
                window, front, next_weight, decay, block_mode, block_bias = cfg
                cap = limit
                if best_cx:
                    slack = best_cx * FINALIST_SLACK // 1024
                    cap = best_cx + slack if cap == 0 else min(cap, best_cx + slack)
                net = synthesize_network(
                    merged,
                    target.num_qubits,
                    gedges,
                    lay,
                    active=active,
                    window=window,
                    front=front,
                    next_weight=next_weight,
                    decay=decay,
                    block_mode=block_mode,
                    block_bias=block_bias,
                    finalize_permutation=finalize_permutation,
                    cx_limit=cap,
                )
                if net.aborted:
                    continue
                # settings that only differ in an unused look-ahead give identical
                # runs: keep one of each (by CX count, blocks and rotation order)
                sig = (
                    net.num_cx,
                    net.blocks,
                    net.swap_blocks,
                    tuple(net.order),
                    tuple(lay),
                    gname,
                )
                if sig in seen:
                    continue
                seen.add(sig)
                runs += 1
                done.append((net.num_cx, runs, net, lay, cfg, gname))
                if not best_cx or net.num_cx < best_cx:
                    best_cx = net.num_cx
                    # keep only the runs that can still be finalists (memory)
                    keep = best_cx + best_cx * FINALIST_SLACK // 1024
                    done = sorted(d for d in done if d[0] <= keep)[:FINALISTS]
                elif len(done) > 4 * FINALISTS:
                    done = sorted(done)[:FINALISTS]
    if not done:
        return None
    cut = best_cx + best_cx * FINALIST_SLACK // 1024
    finalists = sorted(d for d in done if d[0] <= cut)[:FINALISTS]
    out: _Candidate | None = None
    for _, _, f_net, f_lay, f_cfg, f_graph in finalists:
        cand = _transpile_network(
            n, f_net, f_lay, f_cfg, f_graph, target, optimization_level, seed
        )
        if out is None or cand.key < out.key:
            out = cand
    return out


def _transpile_network(
    n: int,
    net: Any,
    lay: list[int],
    cfg: tuple[int, ...],
    gname: str,
    target: CompilationTarget,
    optimization_level: int,
    seed: int,
) -> _Candidate:
    qc = net.circuit
    _pad(qc, sorted(set(lay)))
    res = transpile(
        qc,
        target,
        optimization_level=optimization_level,
        seed=seed,
        initial_layout=list(range(qc.num_qubits)),
    )
    perm = net.final_permutation
    init = [res.initial_layout[lay[q]] for q in range(n)]
    fin = [res.final_layout[perm[lay[q]]] for q in range(n)]
    key = (res.stats.two_qubit_gates, res.stats.two_qubit_depth)
    config = cast(tuple[int, int, int, int, int, int], tuple(cfg))
    return _Candidate(key, "network", res.circuit, init, fin, res, lay, config, gname)


def _formula_cost(
    n: int,
    terms: Sequence[PauliTerm],
    time: float,
    steps: int,
    order: int,
    seed: int,
    target: CompilationTarget | None = None,
    cap: int = 20000,
) -> int:
    """Estimated two-qubit count of the ``steps``-step formula on ``target``
    (default all-to-all; network with two settings and the ladder at level
    1; the count of large formulas is extrapolated from two and four steps)."""
    if target is None:
        target = _full_target(n)

    def count(k: int, dt: float) -> int:
        merged = merge_commuting(product_formula(terms, dt, k, order), n)
        cands = _compile_rotations(
            n,
            terms,
            merged,
            target,
            methods=METHODS,
            optimization_level=1,
            ladder_level=1,
            seed=seed,
            layout_trials=1,
            network_configs=NETWORK_CONFIGS[:2],
            layouts=None,
            finalize_permutation=True,
        )
        return min(c.key[0] for c in cands)

    stages = {1: 1, 2: 2, 4: 10}[order]
    per_step = stages * sum(2 * (len(q) - 1) for _, q, _ in terms)
    if steps <= 4 or per_step * steps <= cap:
        return count(steps, time)
    c2 = count(2, 2 * time / steps)
    c4 = count(4, 4 * time / steps)
    return _extrapolate(c2, c4, steps)


def _extrapolate(c2: int, c4: int, steps: int) -> int:
    """``c2 + (steps − 2)(c4 − c2)/2`` rounded to the nearest integer (half up)."""
    num = 2 * c2 + (steps - 2) * (c4 - c2)
    return (2 * num + 2) // 4


def _textbook_cx(terms: Sequence[PauliTerm], steps: int, order: int) -> int:
    """Textbook CX count of the ``steps``-step formula (adjacent merges included).

    Merges happen inside a step and at the identical step boundaries only, so
    the count is affine in ``steps``.
    """
    c1 = _naive_cx(product_formula(terms, 1.0, 1, order))
    if steps == 1:
        return c1
    c2 = _naive_cx(product_formula(terms, 2.0, 2, order))
    return c1 + (steps - 1) * (c2 - c1)


def _result(
    n: int,
    term_order: str,
    order: int,
    steps: int,
    rotations: list[Rotation],
    cands: list[_Candidate],
) -> EvolutionResult:
    best = min(cands, key=lambda c: c.key)
    return EvolutionResult(
        circuit=best.circuit,
        initial_layout=best.init,
        final_layout=best.fin,
        order=order,
        steps=steps,
        term_order=term_order,
        rotations=rotations,
        method=best.method,
        two_qubit_gates=best.key[0],
        two_qubit_depth=best.key[1],
        candidates={c.method: c.key[0] for c in cands},
        transpile_result=best.res,
        network_layout=best.layout,
        network_config=best.config,
        network_graph=best.graph,
    )


def _compile_formula(
    n: int,
    terms: Sequence[PauliTerm],
    time: float,
    steps: int,
    order: int,
    target: CompilationTarget,
    term_order: str,
    options: dict[str, Any],
    bound: int = 0,
) -> EvolutionResult | None:
    """Compile the ``steps``-step formula of ``terms`` (already in term order)."""
    rotations = product_formula(terms, time, steps, order)
    merged = merge_commuting(rotations, n)
    cands = _compile_rotations(n, terms, merged, target, bound=bound, **options)
    if not cands:
        return None
    return _result(n, term_order, order, steps, rotations, cands)


def _fixed_options(options: dict[str, Any], res: EvolutionResult) -> dict[str, Any]:
    """``options`` restricted to the candidate selected in ``res``."""
    out = dict(options)
    out["methods"] = (res.method,)
    if res.method == "network":
        out["layouts"] = [res.network_layout]
        out["network_configs"] = [res.network_config]
        out["network_graphs"] = [res.network_graph]
    return out


@dataclass
class TermOrderChoice:
    """Output of :func:`select_term_order`.

    Attributes:
        term_order: The selected term order.
        steps: Its step count for ``eps``.
        error_estimate: Its estimated error.
        error_metric: ``"state"`` or ``"bound"``.
        table: ``{term order: (steps, error estimate, two-qubit gates)}``
            for every candidate (compiled or estimated, see ``cost``).
        cost: ``"compile"`` or ``"estimate"``.
        results: With ``cost="compile"``: ``{term order: compiled
            results}`` — one :class:`EvolutionResult` of the whole formula,
            or the two- and four-step results ``(two, four)`` of an
            extrapolated one (missing when every run was abandoned by the
            bound).
    """

    term_order: str
    steps: int
    error_estimate: float
    error_metric: str
    table: dict[str, tuple[int, float, int]]
    cost: str = "compile"
    results: dict[str, Any] = field(default_factory=dict, repr=False)


def select_term_order(
    hamiltonian: Any,
    time: float,
    eps: float,
    order: int = 2,
    *,
    candidates: Sequence[str] = TERM_ORDERS,
    error_metric: str = "auto",
    num_qubits: int | None = None,
    seed: int = 0,
    max_steps: int = 4096,
    target: CompilationTarget | None = None,
    cost: str = "compile",
    compile_options: dict[str, Any] | None = None,
) -> TermOrderChoice:
    """Choose the term order of a product formula by its own error estimate and cost.

    Reordering non-commuting terms changes the Trotter error, so a term
    order that makes each step cheaper can need more steps.  For every
    candidate order this finds the fewest steps meeting ``eps`` by this
    module's error estimate (``error_metric``: simulated state error on its
    own random states, or the commutator bound) and the two-qubit count of
    that formula on ``target`` (default all-to-all connectivity); the
    cheapest order wins (ties: the earlier candidate).  The best order can
    depend on the target: a support-grouped order that saves gates on
    all-to-all connectivity may need far more routing on a sparse device.

    ``cost="compile"`` (default) compiles every candidate formula exactly as
    :func:`compile_evolution` would (``compile_options``: its keyword
    arguments ``methods``, ``optimization_level``, ``ladder_level``,
    ``layout_trials``, ``network_configs``, ``network_graphs``,
    ``finalize_permutation``), by branch and bound: once an order has been
    compiled, the network runs of the later ones are abandoned above
    :data:`SELECT_ABORT_RATIO` times its count.  Formulas of more than four
    steps with more than :data:`SELECT_FULL_CX` textbook CX are costed from
    their two- and four-step circuits, ``c2 + (r − 2)(c4 − c2)/2`` (the
    steady-state slope; the two-step circuit reuses the four-step choice of
    candidate).  ``cost="estimate"``: a cheap estimate instead (network
    with two settings and the ladder at level 1).

    Args:
        hamiltonian: Pauli sum (see :func:`quspect.hamsim.pauli_terms`).
        time: Evolution time.
        eps: Error target.
        order: Product-formula order (1, 2 or 4).
        candidates: Term orders to compare (see
            :func:`quspect.hamsim.order_terms`).
        error_metric: ``"auto"``, ``"state"`` or ``"bound"``.
        num_qubits: Number of qubits (default from the Hamiltonian).
        seed: Seed of the compilation.
        max_steps: Largest step count tried.
        target: Compilation target (default all-to-all connectivity on the
            Hamiltonian's qubits).
        cost: ``"compile"`` or ``"estimate"``.
        compile_options: Options of the compilation (``cost="compile"``).

    Returns:
        The selected order, the comparison table and (``cost="compile"``)
        the compiled results.
    """
    n, terms = pauli_terms(hamiltonian, num_qubits)
    if not candidates:
        raise ValueError("candidates is empty")
    if cost not in ("compile", "estimate"):
        raise ValueError("cost must be 'compile' or 'estimate'")
    if target is None:
        target = _full_target(n)
    opts = _compile_defaults(seed)
    unknown = set(compile_options or {}) - set(opts)
    if unknown:
        raise ValueError(f"unknown compile options {sorted(unknown)}")
    opts.update(compile_options or {})
    metric = _resolve_metric(error_metric, n)
    reference = None
    if metric == "state":
        from quspect.hamsim._formulas import _exact_evolve, _test_states

        # the exact evolution does not depend on the term order
        reference = _exact_evolve(n, terms, time, _test_states(n, 2, 20261102))
    table: dict[str, tuple[int, float, int]] = {}
    results: dict[str, Any] = {}
    best: str | None = None
    best_four = 0  # four-step count of the best extrapolated order
    for name in candidates:
        ordered = order_terms(terms, name)
        try:
            r, e = _steps_for_eps(
                n, ordered, time, order, eps, metric, max_steps, reference
            )
        except ValueError:  # no step count up to max_steps meets eps
            continue
        if cost == "estimate":
            c = _formula_cost(n, ordered, time, r, order, seed, target)
        else:
            bound = 0
            if best is not None:
                bound = int(SELECT_ABORT_RATIO * table[best][2]) + 1
            if r <= 4 or _textbook_cx(ordered, r, order) <= SELECT_FULL_CX:
                res = _compile_formula(
                    n, ordered, time, r, order, target, name, opts, bound
                )
                if res is None:  # every run abandoned: cannot win
                    continue
                c = res.two_qubit_gates
                results[name] = res
            else:
                b4 = int(SELECT_ABORT_RATIO * best_four) + 1 if best_four else 0
                four = _compile_formula(
                    n, ordered, 4 * time / r, 4, order, target, name, opts, b4
                )
                if four is None:
                    continue
                two = _compile_formula(
                    n,
                    ordered,
                    2 * time / r,
                    2,
                    order,
                    target,
                    name,
                    _fixed_options(opts, four),
                )
                if two is None:
                    continue
                c = _extrapolate(two.two_qubit_gates, four.two_qubit_gates, r)
                results[name] = (two, four)
                if best is None or c < table[best][2]:
                    best_four = four.two_qubit_gates
        table[name] = (r, e, c)
        if best is None or c < table[best][2]:
            best = name
    if best is None:
        raise ValueError(f"no term order reaches eps = {eps} within {max_steps} steps")
    r, e, _ = table[best]
    return TermOrderChoice(best, r, e, metric, table, cost, results)


def _compile_defaults(seed: int) -> dict[str, Any]:
    return {
        "methods": METHODS,
        "optimization_level": 2,
        "ladder_level": None,
        "seed": seed,
        "layout_trials": 2,
        "network_configs": None,
        "layouts": None,
        "finalize_permutation": True,
        "network_graphs": NETWORK_GRAPHS,
    }


def compile_evolution(
    hamiltonian: Any,
    time: float,
    target: CompilationTarget | None = None,
    *,
    steps: int | None = None,
    order: int = 2,
    eps: float | None = None,
    term_order: str = "input",
    error_metric: str = "auto",
    methods: Sequence[str] = METHODS,
    optimization_level: int = 2,
    ladder_level: int | None = None,
    seed: int = 0,
    num_qubits: int | None = None,
    layout_trials: int = 2,
    network_configs: Sequence[Sequence[int]] | None = None,
    layouts: Sequence[Sequence[int]] | None = None,
    finalize_permutation: bool = True,
    max_steps: int = 4096,
    network_graphs: Sequence[str] = NETWORK_GRAPHS,
    term_order_cost: str = "compile",
) -> EvolutionResult:
    """Compile ``exp(-i H t)`` with a product formula for a target.

    The product formula (``order`` ∈ {1, 2, 4}, ``steps`` steps,
    ``term_order``) is fixed first; its rotations are then compiled
    exactly — every candidate implements the same unitary, so the choice
    between them is by gate count alone:

    - ``"network"``: Clifford-frame Pauli-network synthesis directly on the
      coupling graph (see :func:`quspect.hamsim.synthesize_network`):
      rotations about the same Pauli separated only by commuting rotations
      are merged first, consecutive rotations share Clifford prefixes,
      Paulis are moved on the device through the frame instead of by SWAP
      routing, two-qubit blocks on coupling edges may be emitted in place
      (optionally with a mirror SWAP absorbed into the frame), and the final
      frame is synthesized on the same graph — with
      ``finalize_permutation`` only up to a qubit permutation, which the
      output layout absorbs.  The synthesis runs on each graph of
      ``network_graphs`` (the coupling graph and a long path of it, see
      :data:`NETWORK_GRAPHS`); the initial layout is the best (fewest CX) of
      ``layout_trials`` low-dilation embeddings of the interaction graph
      and SABRE's layout of the ladder circuit on that graph (on the path
      also the logical qubits in index order along it; or the given
      ``layouts``), each with every setting in ``network_configs``; runs are abandoned
      as soon as they exceed the best CX count so far or
      :data:`NETWORK_ABORT_RATIO` times the ladder candidate.  The result
      is then optimized by :func:`~quspect.transpiler.transpile` at
      ``optimization_level`` with the layout fixed (no routing is needed).
    - ``"ladder"``: the textbook circuit (basis change, CX ladder, ``rz``
      per rotation) through :func:`~quspect.transpiler.transpile` at
      ``ladder_level``.

    The candidate with the fewest two-qubit gates (then the smallest
    two-qubit depth) is returned.

    With ``eps`` and no ``steps``, the smallest ``steps`` whose estimated
    error is at most ``eps`` is used.  ``error_metric``: ``"state"`` (the
    worst simulated state error over random states and ``|0101…⟩``, up to
    16 qubits), ``"bound"`` (commutator bound, orders 1 and 2) or
    ``"auto"`` (``"state"`` when possible).  ``term_order="auto"`` (needs
    ``eps``) picks the term order by :func:`select_term_order` on
    ``target``: with ``term_order_cost="compile"`` (default) every
    candidate order is compiled with these options (branch and bound) and
    the cheapest compiled circuit is returned directly when ``steps`` is
    not given; ``"estimate"`` uses a cheap cost estimate.

    Args:
        hamiltonian: Pauli sum (see :func:`quspect.hamsim.pauli_terms`).
        time: Evolution time ``t``.
        target: Compilation target; default: all-to-all connectivity on
            the Hamiltonian's qubits with a CZ gate set.
        steps: Number of Trotter steps.
        order: Product-formula order (1, 2 or 4).
        eps: Error target, used when ``steps`` is not given.
        term_order: ``"input"``, ``"support"`` (see
            :func:`quspect.hamsim.order_terms`) or ``"auto"``.
        error_metric: ``"auto"``, ``"state"`` or ``"bound"``.
        methods: Candidates to compile, a subset of ``("network", "ladder")``.
        optimization_level: :func:`transpile` level for the network
            candidate.
        ladder_level: :func:`transpile` level for the ladder candidate
            (default ``optimization_level``).
        seed: Seed for layouts and routing.
        num_qubits: Number of logical qubits (default from the Hamiltonian).
        layout_trials: Low-dilation layouts tried for the network candidate
            on a non-complete coupling graph.
        network_configs: Settings ``(window, front, next_weight, decay[,
            block_mode, block_bias])`` of
            :func:`~quspect.hamsim.synthesize_network` tried for every
            layout (the fewest CX wins); default (``None``):
            :data:`NETWORK_CONFIGS` plus as many of :data:`EXTENDED_CONFIGS`
            as :data:`PORTFOLIO_BUDGET` allows.
        layouts: Initial layouts (``layout[logical] = physical``) for the
            network candidate instead of the automatic ones.
        finalize_permutation: Synthesize the network's final frame only up
            to a qubit permutation (absorbed into ``final_layout``).
        max_steps: Largest step count tried with ``eps``.
        network_graphs: Graphs the network candidate is synthesized on
            (subset of :data:`NETWORK_GRAPHS`).
        term_order_cost: ``"compile"`` or ``"estimate"`` (``term_order="auto"``).

    Returns:
        The compiled circuit and its product formula.

    Raises:
        ValueError: Bad arguments, or a target with fewer qubits than the
            Hamiltonian.

    Example:
        >>> from quspect.hamsim import compile_evolution
        >>> h = [("ZZ", (0, 1), 1.0), ("X", (0,), 0.5), ("X", (1,), 0.5)]
        >>> res = compile_evolution(h, 0.5, steps=2, order=2)
        >>> res.steps, res.order, res.method in ("network", "ladder")
        (2, 2, True)
    """
    n, terms = pauli_terms(hamiltonian, num_qubits)
    if order not in (1, 2, 4):
        raise ValueError("order must be 1, 2 or 4")
    unknown = set(methods) - set(METHODS)
    if unknown or not methods:
        raise ValueError(f"methods must be a non-empty subset of {METHODS}")
    if target is None:
        target = _full_target(n)
    if target.num_qubits < n:
        raise ValueError(f"target has {target.num_qubits} qubits, the Hamiltonian {n}")

    options: dict[str, Any] = {
        "methods": tuple(methods),
        "optimization_level": optimization_level,
        "ladder_level": ladder_level,
        "seed": seed,
        "layout_trials": layout_trials,
        "network_configs": network_configs,
        "layouts": layouts,
        "finalize_permutation": finalize_permutation,
        "network_graphs": tuple(network_graphs),
    }
    err_est: float | None = None
    metric: str | None = None
    choice: TermOrderChoice | None = None
    if term_order == "auto":
        if eps is None:
            raise ValueError("term_order='auto' needs eps")
        choice = select_term_order(
            terms,
            time,
            eps,
            order,
            error_metric=error_metric,
            num_qubits=n,
            seed=seed,
            max_steps=max_steps,
            target=target,
            cost=term_order_cost,
            compile_options=options,
        )
        term_order = choice.term_order
        if steps is None:
            steps, err_est, metric = (
                choice.steps,
                choice.error_estimate,
                choice.error_metric,
            )
            done = choice.results.get(term_order)
            if isinstance(done, EvolutionResult):  # compiled during the selection
                done.error_estimate, done.error_metric = err_est, metric
                done.term_order_table = choice.table
                return done
    terms = order_terms(terms, term_order)

    if steps is None:
        if eps is None:
            raise ValueError("give steps or eps")
        metric = _resolve_metric(error_metric, n)
        steps, err_est = _steps_for_eps(n, terms, time, order, eps, metric, max_steps)
    elif steps < 1:
        raise ValueError("steps must be >= 1")

    res = _compile_formula(n, terms, time, steps, order, target, term_order, options)
    if res is None:
        raise RuntimeError("no candidate compiled (network aborted without a ladder)")
    res.error_estimate, res.error_metric = err_est, metric
    res.term_order_table = None if choice is None else choice.table
    return res
