"""``compile_evolution``: ``exp(-iHt)`` to a circuit for a compilation target."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
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
    cycle_of_length,
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

#: Further settings of the default portfolio: longer windows (up to 128)
#: and the full grid of fronts, weights, decays and block modes.  Small
#: formulas (e.g. single-step UCCSD ansätze) gain 2–7 % from them.  They
#: are tried on the best graph and layout of :data:`NETWORK_CONFIGS` only
#: (screened on a prefix of the formula, see :data:`WORK_BUDGET`).
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

#: Work budget of the default portfolio for one formula.  A network run's work is its
#: scoring work (:attr:`~quspect.hamsim.NetworkCircuit.work`, the inner iterations of
#: the support cost estimates) plus ``(rotations processed + CX emitted) × (window +
#: 16)``, about a nanosecond per unit on one core; it is measured after the run from
#: deterministic counts, so every decision below — and the result — is the same on every
#: machine.  The portfolio runs in two stages.  Stage 1, within half of the budget:
#: :data:`NETWORK_CONFIGS` on every graph and layout (setting-major priority), all in
#: full when the work predicted from the first run fits, otherwise screened: each run on
#: the first ``1 / divisor`` of the rotations (``divisor`` from :data:`SCREEN_DIVISOR`
#: up to 16 times it, the smallest that fits; as many runs as fit) and the
#: :data:`SCREEN_KEEP` cheapest distinct prefixes in full.  Stage 2, within the rest:
#: the settings of :data:`EXTENDED_CONFIGS` on the :data:`STAGE2_GROUPS` best graphs and
#: layouts of stage 1 only, screened the same way.  In every case measured the best
#: stage-1 layout was the best layout overall and a quarter of the formula kept the best
#: runs among the survivors, so this keeps the quality of trying every setting on every
#: layout at a fraction of its work, and bounds the work of very large formulas (e.g.
#: 100-qubit lattices on a line) that a fixed portfolio cannot.  Small formulas (see
#: :data:`EXHAUSTIVE_WORK`) still try everything.  Every run also stops (and is
#: abandoned) once the scoring work reaches the end of its stage (the ``work_limit`` of
#: :func:`~quspect.hamsim.synthesize_network`, a deterministic count): the first run at
#: half of the budget — when it does not fit there, a formula of several steps is
#: compiled as repeated one-step blocks (see :data:`BLOCK_RATIO`) and otherwise the
#: ladder is kept — so no single run can exceed the budget, however much the work of
#: a run grows once its frame images widen (a prefix of half the formula can need a
#: thousandth of the work of the whole).
WORK_BUDGET = 60_000_000_000

#: With ``block_steps=None`` (automatic), a formula of several steps whose
#: single full network run is predicted (from runs on its first
#: ``1 / PROBE_DIVISOR``, … ``1 / 2``) to need more than
#: ``WORK_BUDGET / BLOCK_RATIO`` is compiled as a repeated one-step block
#: instead (see
#: :func:`compile_evolution`): its compile time no longer grows with the
#: number of steps.
BLOCK_RATIO = 4

#: See :data:`BLOCK_RATIO`.
PROBE_DIVISOR = 16

#: Formulas whose whole default portfolio (every setting on every graph and
#: layout, predicted from the first run) needs at most this much work run
#: it all in full (see :data:`WORK_BUDGET`).
EXHAUSTIVE_WORK = 1_000_000_000

#: Stage 2 of the default portfolio runs on at least this many best graphs
#: and layouts of stage 1 (see :data:`WORK_BUDGET` and
#: :data:`STAGE2_FULL_WORK`).
STAGE2_GROUPS = 2

#: Stage 2 is screened on prefixes when its runs do not fit in the rest of
#: the budget (``False``: in full while it fits, as stage 1).
STAGE2_SCREEN = True

#: Stage 2 runs in full (not screened) on as many best groups as fit in
#: this much work (and in the rest of the budget), when its runs on the
#: :data:`STAGE2_GROUPS` best groups fit: cheap formulas try every setting
#: on more layouts, and are not screened.
STAGE2_FULL_WORK = 20_000_000_000

#: Screening runs take the first ``1 / SCREEN_DIVISOR`` of the rotations.
SCREEN_DIVISOR = 4

#: Distinct screening prefixes compiled in full.
SCREEN_KEEP = 16

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
#: ``"cycle"``: a cycle of exactly ``n`` qubits (see
#: :func:`~quspect.hamsim.cycle_of_length`), used when the interaction
#: graph wraps around (a term couples logical qubits ``0`` and ``n − 1``,
#: e.g. periodic chains): the logical qubits in index order along it have
#: every chain bond on a coupling edge.
NETWORK_GRAPHS = ("device", "path", "cycle")

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

#: Ladders of more textbook CX than this are transpiled at level 1 only:
#: routing hundreds of thousands of CX at level 3 takes minutes (four steps
#: of a 20-qubit molecule on heavy-hex, 400 000 CX: 190 s at level 3, 48 s
#: at level 1).  Such ladders are far behind the network (molecules: by a
#: factor of 1.5–4), so they remain only a fallback and a bound.
HUGE_LADDER_CX = 350_000

#: Repeated blocks whose copies have more CX than this are transpiled once
#: and the transpiled block is repeated (see :func:`_repeat_block`); smaller
#: ones are transpiled as a whole (the optimizations across the copies save
#: a few gates).
REPEAT_TRANSPILE_CX = 100_000

#: ... and only the cheapest ``REPEAT_FINALISTS`` finalists are transpiled.
REPEAT_FINALISTS = 2


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
        block_steps: Steps of the repeated block when the circuit is copies
            of one compiled block (``None`` otherwise; see ``block_steps`` of
            :func:`compile_evolution`).  ``transpile_result`` is then that
            of the copies, or of one block when the transpiled block is
            repeated (see :data:`REPEAT_TRANSPILE_CX`).
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
    block_steps: int | None = None


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


def _wraps(terms: Sequence[PauliTerm], n: int) -> bool:
    """Some term couples logical qubits ``0`` and ``n − 1`` and nothing between."""
    if n < 4:
        return False
    return any(sorted(qs) == [0, n - 1] for _, qs, _ in terms)


def _network_graphs(
    target: CompilationTarget,
    n: int,
    names: Sequence[str],
    terms: Sequence[PauliTerm] = (),
) -> list[tuple[str, list[tuple[int, int]], CompilationTarget, list[int]]]:
    """``(name, edges, target of those edges, qubits along it)`` for each graph.

    ``"path"`` is skipped on all-to-all connectivity, when the coupling
    graph is itself a path, or when the longest path found has fewer than
    ``n`` qubits; ``"cycle"`` unless ``terms`` wrap around (see
    :func:`_wraps`) and a cycle of exactly ``n`` qubits that is not the
    whole coupling graph is found.  The qubit list is empty for
    ``"device"``.
    """
    edges = _edges(target)
    out: list[tuple[str, list[tuple[int, int]], CompilationTarget, list[int]]] = []
    for name in names:
        if name == "device":
            out.append(("device", edges, target, []))
            continue
        if name not in ("path", "cycle"):
            raise ValueError(
                f"unknown network graph {name!r}; expected one of {NETWORK_GRAPHS}"
            )
        if _is_full(target, n):
            continue
        if name == "path":
            walk = long_path(target.num_qubits, edges)
            if len(walk) < n:
                continue
            pairs = list(zip(walk, walk[1:], strict=False))
        else:
            if not _wraps(terms, n):
                continue
            walk = cycle_of_length(target.num_qubits, edges, n)
            if not walk:
                continue
            pairs = list(zip(walk, walk[1:] + walk[:1], strict=False))
        pe = sorted({(min(a, b), max(a, b)) for a, b in pairs})
        if len(pe) == len(edges):
            continue
        sub = CompilationTarget.from_coupling_map(
            target.num_qubits, pe, native_gates=set(DEFAULT_NATIVE_GATES)
        )
        out.append((name, pe, sub, walk))
    if not out:  # e.g. only "path" on all-to-all connectivity
        out.append(("device", edges, target, []))
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
    work_budget: int | None = None,
    probe: bool = False,
    repeat: int = 1,
) -> list[_Candidate]:
    """Compile a (merged) rotation sequence with every requested method.

    Normally the ladder is compiled first and bounds the network runs
    (:data:`NETWORK_ABORT_RATIO`).  For formulas above
    :data:`LARGE_FORMULA_CX` textbook CX the network goes first (bounded by
    ``NETWORK_ABORT_RATIO`` times the textbook count) and the expensive
    ladder transpilation is skipped when the network needs fewer than
    ``1 / LADDER_SKIP_RATIO`` of the textbook CX.  ``bound`` (``> 0``)
    additionally abandons network runs above that CX count.  ``probe``:
    raise :class:`_TooBigError` (before any expensive work) when one full
    network run is predicted to exceed ``work_budget / BLOCK_RATIO``.
    """
    if work_budget is None:
        work_budget = WORK_BUDGET
    out: list[_Candidate] = []
    naive = _naive_cx(merged)
    want_ladder = "ladder" in methods or not merged
    want_network = "network" in methods and bool(merged)
    large = naive > LARGE_FORMULA_CX

    def ladder() -> _Candidate:
        qc = ladder_circuit(merged, n)
        _pad(qc, range(n))
        level = ladder_level or optimization_level
        if naive > HUGE_LADDER_CX:
            level = min(level, 1)
        res = transpile(qc, target, optimization_level=level, seed=seed)
        key = (res.stats.two_qubit_gates, res.stats.two_qubit_depth)
        return _Candidate(
            key,
            "ladder",
            res.circuit,
            list(res.initial_layout)[:n],
            list(res.final_layout)[:n],
            res,
        )

    if probe and want_network:
        _probe(n, terms, merged, target, layouts, network_graphs, work_budget)
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
        try:
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
                work_budget=work_budget,
                repeat=repeat,
                too_big=probe,
            )
        except _TooBigError as exc:
            exc.ladder = out  # the ladder already compiled (reused by the blocks)
            raise
        if net is not None:
            out.append(net)
    if want_ladder and large and want_network:
        net_cx = next((c.key[0] for c in out if c.method == "network"), None)
        if net_cx is None or net_cx * LADDER_SKIP_RATIO >= naive or not merged:
            out.append(ladder())
    return out


class _TooBigError(Exception):
    """A full network run of the formula would exceed its share of the budget.

    ``ladder``: the ladder candidates compiled before (if any).
    """

    def __init__(self) -> None:
        super().__init__()
        self.ladder: list[_Candidate] = []


#: A completed network run: ``(CX, run number, network, layout, setting, graph
#: name, group index)``.
_Run = tuple[int, int, Any, list[int], tuple[int, ...], str, int]


class _Portfolio:
    """Network runs of one formula within a work budget: branch and bound over
    full runs, screening on prefixes, and the finalists (see
    :data:`WORK_BUDGET`).

    The work of a run is measured after it, from deterministic counts only
    (the native core's scoring work plus rotations processed and CX emitted
    times the window), so the decisions — and the result — are the same on
    every machine.
    """

    def __init__(
        self,
        merged: Sequence[Rotation],
        num_physical: int,
        groups: list[tuple[str, list[tuple[int, int]], list[int], list[int]]],
        finalize_permutation: bool,
        limit: int,
        budget: int,
    ) -> None:
        self.merged = merged
        self.num_physical = num_physical
        self.groups = groups  # (graph name, edges, layout, active qubits)
        self.finalize_permutation = finalize_permutation
        self.limit = limit
        self.budget = budget
        self.spent = 0
        self.rate = 0  # work of a full run per (window + 16), from the first one
        self.done: list[_Run] = []
        self.best_cx = 0
        self.best_group = -1
        # without a completed run: the group whose abandoned run got furthest
        self.reach = (0, -1)
        self.runs = 0  # run counter: stable order of equal CX counts
        self.seen: set[tuple[Any, ...]] = set()
        self.group_best: dict[int, int] = {}  # fewest CX of each group's runs
        # work at which the current stage ends: no run starts at or after it and
        # every run stops there (``work_limit`` of the native core); 0: no limit
        self.end = 0
        self.exceeded = False  # the last run stopped at the work limit

    def _synth(self, gi: int, cfg: tuple[int, ...], length: int, cap: int) -> Any:
        _, gedges, lay, active = self.groups[gi]
        window, front, next_weight, decay, block_mode, block_bias = cfg
        full = length >= len(self.merged)
        work_limit = max(1, self.end - self.spent) if self.end else 0
        net = synthesize_network(
            self.merged if full else self.merged[:length],
            self.num_physical,
            gedges,
            lay,
            active=active,
            window=window,
            front=front,
            next_weight=next_weight,
            decay=decay,
            block_mode=block_mode,
            block_bias=block_bias,
            finalize=full,
            finalize_permutation=self.finalize_permutation and full,
            cx_limit=cap,
            work_limit=work_limit,
        )
        self.exceeded = net.work_exceeded
        work = net.work + (len(net.order) + net.num_cx) * (window + 16)
        self.spent += work
        if full and not net.aborted and not self.rate:
            self.rate = max(1, work // (window + 16))
        return net

    def estimate(self, cfg: tuple[int, ...]) -> int:
        """Predicted work of a full run with setting ``cfg``."""
        return self.rate * (cfg[0] + 16)

    def full(self, gi: int, cfg: tuple[int, ...]) -> None:
        """One full run, abandoned above the branch-and-bound cap."""
        cap = self.limit
        if self.best_cx:
            slack = self.best_cx * FINALIST_SLACK // 1024
            cap = self.best_cx + slack if cap == 0 else min(cap, self.best_cx + slack)
        net = self._synth(gi, cfg, len(self.merged), cap)
        if net.aborted:
            if len(net.order) > self.reach[0]:
                self.reach = (len(net.order), gi)
            return
        gname, _, lay, _ = self.groups[gi]
        # settings that only differ in an unused look-ahead give identical
        # runs: keep one of each (by CX count, blocks and rotation order)
        sig = (net.num_cx, net.blocks, net.swap_blocks, tuple(net.order), gi)
        if sig in self.seen:
            return
        self.seen.add(sig)
        self.runs += 1
        if net.num_cx < self.group_best.get(gi, net.num_cx + 1):
            self.group_best[gi] = net.num_cx
        self.done.append((net.num_cx, self.runs, net, lay, cfg, gname, gi))
        if not self.best_cx or net.num_cx < self.best_cx:
            self.best_cx = net.num_cx
            self.best_group = gi
            # keep only the runs that can still be finalists (memory)
            keep = self.best_cx + self.best_cx * FINALIST_SLACK // 1024
            self.done = sorted(d for d in self.done if d[0] <= keep)[:FINALISTS]
        elif len(self.done) > 4 * FINALISTS:
            self.done = sorted(self.done)[:FINALISTS]

    def stage(
        self, pool: Sequence[tuple[int, tuple[int, ...]]], end: int, screen: bool
    ) -> None:
        """Run ``pool`` (in priority order) until the work reaches ``end``.

        In full when the predicted work fits (and not ``screen``), else every
        run that fits on the first ``1 / divisor`` of the rotations
        (``divisor`` from :data:`SCREEN_DIVISOR` up to 16 times it, the
        smallest that fits; at least one per graph and layout) and the
        cheapest distinct prefixes in full (:data:`SCREEN_KEEP`, halved
        while their predicted work exceeds three quarters of the rest of the
        budget).  No run starts once the work has reached ``end``.
        """
        pool = list(pool)
        if not pool:
            return
        self.end = end
        if not self.rate:  # the first run of the formula measures the work rate
            self.full(*pool[0])
            pool = pool[1:]
        need = [self.estimate(cfg) for _, cfg in pool]
        if not screen and self.spent + sum(need) <= end or len(pool) <= SCREEN_KEEP:
            for item in pool:
                if self.spent >= end:
                    break
                self.full(*item)
            return
        # survivors: up to SCREEN_KEEP, as many as leave a quarter of the rest
        # of the budget to the prefixes
        avg = max(1, sum(need) // len(need))
        room = end - self.spent
        survivors = SCREEN_KEEP
        while survivors > 2 and survivors * avg > room - room // 4:
            survivors //= 2
        keep = survivors * avg
        divisor = SCREEN_DIVISOR
        while (
            self.spent + sum(need) // divisor + keep > end
            and divisor < 16 * SCREEN_DIVISOR
        ):
            divisor *= 2
        length = max(1, len(self.merged) // divisor)
        stop = end - keep
        ranked: list[tuple[int, int, int, tuple[int, ...]]] = []
        sigs: set[tuple[Any, ...]] = set()
        groups = {gi for gi, _ in pool}
        for k, (gi, cfg) in enumerate(pool):
            # every graph and layout of the pool gets at least one prefix
            if ranked and self.spent >= stop and k >= len(groups):
                break
            net = self._synth(gi, cfg, length, self.limit)
            if net.aborted:  # a prefix above the bound: the full run is too
                continue
            sig = (net.num_cx, net.blocks, net.swap_blocks, tuple(net.order), gi)
            if sig in sigs:
                continue
            sigs.add(sig)
            ranked.append((net.num_cx, k, gi, cfg))
        for _, _, gi, cfg in sorted(ranked)[:survivors]:
            if self.spent >= end:
                break
            self.full(gi, cfg)

    def top_groups(self, k: int) -> list[int]:
        """The ``k`` groups with the cheapest completed runs (else the group of
        the furthest abandoned run)."""
        if not self.group_best:
            return [self.reach[1]] if self.reach[1] >= 0 else []
        return sorted(self.group_best, key=lambda g: (self.group_best[g], g))[:k]

    def finalists(self) -> list[_Run]:
        if not self.done:
            return []
        cut = self.best_cx + self.best_cx * FINALIST_SLACK // 1024
        return sorted(d for d in self.done if d[0] <= cut)[:FINALISTS]


def _probe(
    n: int,
    terms: Sequence[PauliTerm],
    merged: Sequence[Rotation],
    target: CompilationTarget,
    layouts: Sequence[Sequence[int]] | None,
    network_graphs: Sequence[str],
    work_budget: int,
) -> None:
    """Raise :class:`_TooBigError` when a full network run of ``merged`` is
    predicted to exceed ``work_budget / BLOCK_RATIO``.

    The work of a run grows faster than its length (the frame images of
    later rotations widen), so the first :data:`NETWORK_CONFIGS` setting
    runs on the first ``1 / 16``, ``1 / 8``, ``1 / 4`` and ``1 / 2`` of the
    rotations (first graph and layout) and the work of the whole run is
    extrapolated from the last prefix with the growth per doubling seen so
    far (at least 2); the probe stops as soon as the prediction is above
    the threshold, or far below it.  Formulas of fewer than ``4 ·
    PROBE_DIVISOR`` rotations are never too big.
    """
    if len(merged) < 4 * PROBE_DIVISOR:
        return
    gname, gedges, gtarget, walk = _network_graphs(target, n, network_graphs, terms)[0]
    if layouts is not None:
        lay = list(layouts[0])
    elif _is_full(target, n):
        lay = list(range(n))
    elif walk:
        lay = walk[:n]
    else:
        lay = _layout_candidates(n, terms, merged, gtarget, 0, 1)[0]
    used = sorted({lay[q] for _, qs, _ in merged for q in qs})
    active = connected_active_set(target.num_qubits, gedges, used)
    group = [(gname, gedges, lay, active)]
    cfg = _norm_config(NETWORK_CONFIGS[0])
    limit = work_budget // BLOCK_RATIO
    prev = 0
    div = PROBE_DIVISOR
    while div >= 2:
        pf = _Portfolio(merged, target.num_qubits, group, True, 0, 0)
        pf._synth(0, cfg, len(merged) // div, 0)
        work = max(1, pf.spent)
        num, den = (work, prev) if prev and work > 2 * prev else (2, 1)
        pred = work
        d = div
        while d > 1:  # one growth factor per doubling up to the whole run
            pred = pred * num // den
            d //= 2
        if pred > limit:
            raise _TooBigError
        if pred * 16 <= limit:
            return
        prev = work
        div //= 2


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
    work_budget: int = 0,
    repeat: int = 1,
    too_big: bool = False,
) -> _Candidate | None:
    """The network candidate (see :data:`WORK_BUDGET`); ``None`` when every run
    was abandoned.  ``too_big``: raise :class:`_TooBigError` when the first
    full run alone exceeds half of ``work_budget`` (stopped there)."""
    groups: list[tuple[str, list[tuple[int, int]], list[int], list[int]]] = []
    for gname, gedges, gtarget, walk in _network_graphs(
        target, n, network_graphs, terms
    ):
        if layouts is not None:
            lays = [list(lay) for lay in layouts]
        elif _is_full(target, n):
            lays = [list(range(n))]
        elif gname in ("path", "cycle"):
            # layouts on the path (cycle) as a line (ring) of its own (the qubits
            # off it are not connected to it), mapped back; logical qubits in
            # index order along it first (Jordan–Wigner and periodic chains)
            m = len(walk)
            ring = [(k, k + 1) for k in range(m - 1)]
            if gname == "cycle":
                ring.append((0, m - 1))
            sub = CompilationTarget.from_coupling_map(
                m, ring, native_gates=set(DEFAULT_NATIVE_GATES)
            )
            lays = [walk[:n]]
            for lay in _layout_candidates(n, terms, merged, sub, seed, layout_trials):
                mapped = [walk[k] for k in lay]
                if mapped not in lays:
                    lays.append(mapped)
        else:
            lays = _layout_candidates(n, terms, merged, gtarget, seed, layout_trials)
        for lay in lays:
            used = sorted({lay[q] for _, qs, _ in merged for q in qs})
            active = connected_active_set(target.num_qubits, gedges, used)
            groups.append((gname, gedges, lay, active))
    explicit = network_configs is not None
    configs = [
        _norm_config(c)
        for c in (NETWORK_CONFIGS if network_configs is None else network_configs)
    ]
    if not configs:
        raise ValueError("network_configs is empty")
    pf = _Portfolio(
        merged, target.num_qubits, groups, finalize_permutation, limit, work_budget
    )
    if explicit:
        # given settings: each on every graph and layout in full, by branch and
        # bound with a small slack (runs above (1 + slack) times the best CX
        # count so far cannot be finalists)
        for gi in range(len(groups)):
            for cfg in configs:
                pf.full(gi, cfg)
    else:
        # stage 1: the base settings on every graph and layout (setting-major
        # priority) within half of the budget; stage 2: the extended settings
        # on the graph and layout of the best run, screened, within the rest
        pool = [(gi, cfg) for cfg in configs for gi in range(len(groups))]
        # the first run measures the work rate; it stops at half of the budget
        pf.end = work_budget // 2
        pf.full(*pool[0])
        if pf.exceeded:
            if too_big:  # a single run does not fit: compile the formula as blocks
                raise _TooBigError
            if limit > 0:  # a ladder (or the bound) competes: no network result
                return None
            pf.end = 0  # the network is the only candidate: one run in full
            pf.full(*pool[0])
        extended = [_norm_config(c) for c in EXTENDED_CONFIGS]
        whole = sum(pf.estimate(c) for c in configs + extended) * len(groups)
        if pf.rate and whole <= EXHAUSTIVE_WORK:
            # small formulas: every setting on every graph and layout in full
            for gi in range(len(groups)):
                for cfg in configs + extended:
                    if (gi, cfg) != pool[0]:
                        pf.full(gi, cfg)
            top = []
            screen = False
        else:
            pf.stage(pool[1:], work_budget // 2, False)
            # stage 2 in full on as many best groups as fit in the rest of the
            # budget, up to STAGE2_FULL_WORK; otherwise screened on the
            # STAGE2_GROUPS best groups
            per = sum(pf.estimate(c) for c in extended)
            room = min(work_budget - pf.spent, STAGE2_FULL_WORK)
            k = 0 if per <= 0 or room <= 0 else min(len(groups), room // per)
            top = pf.top_groups(max(k, STAGE2_GROUPS))
            screen = STAGE2_SCREEN and len(top) * per > room
        if top:
            pf.stage(
                [(gi, c) for c in extended for gi in top],
                work_budget,
                screen,
            )
    finalists = pf.finalists()
    if finalists and repeat * finalists[0][0] > REPEAT_TRANSPILE_CX:
        finalists = finalists[:REPEAT_FINALISTS]  # large repeated blocks
    # the cheapest few finalists are all transpiled (the transpiler's savings
    # differ between them) and the fewest two-qubit gates after transpilation wins
    out: _Candidate | None = None
    for _, _, f_net, f_lay, f_cfg, f_graph, _ in finalists:
        cand = None
        if repeat > 1 and repeat * f_net.num_cx > REPEAT_TRANSPILE_CX:
            # copies of a block with an exact final frame
            cand = _repeat_block(
                n,
                f_net,
                f_lay,
                f_cfg,
                f_graph,
                target,
                optimization_level,
                seed,
                repeat,
            )
        if cand is None:
            if repeat > 1:
                qc = QuantumCircuit()
                wires = list(range(f_net.circuit.num_qubits))
                for _ in range(repeat):
                    qc.compose(f_net.circuit, wires, inplace=True)
                f_net = replace(f_net, circuit=qc, num_cx=repeat * f_net.num_cx)
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
    elide: bool = True,
) -> _Candidate:
    qc = net.circuit
    _pad(qc, sorted(set(lay)))
    res = transpile(
        qc,
        target,
        optimization_level=optimization_level,
        seed=seed,
        initial_layout=list(range(qc.num_qubits)),
        elide_permutations=elide,
    )
    perm = net.final_permutation
    init = [res.initial_layout[lay[q]] for q in range(n)]
    fin = [res.final_layout[perm[lay[q]]] for q in range(n)]
    key = (res.stats.two_qubit_gates, res.stats.two_qubit_depth)
    config = cast(tuple[int, int, int, int, int, int], tuple(cfg))
    return _Candidate(key, "network", res.circuit, init, fin, res, lay, config, gname)


def _repeat_block(
    n: int,
    net: Any,
    lay: list[int],
    cfg: tuple[int, ...],
    gname: str,
    target: CompilationTarget,
    optimization_level: int,
    seed: int,
    repeat: int,
) -> _Candidate | None:
    """``repeat`` copies of the transpiled block (``None`` when the transpiled
    block does not start and end on the same layout).

    The block is transpiled once and its output repeated: the transpile time
    no longer grows with the number of copies (only the optimizations across
    the block boundaries are given up).  When the transpiler turned SWAPs
    into an output permutation, the block is transpiled again without that
    (the copies must start and end on the same layout).
    """
    from quspect.transpiler._transpile import _score

    def same(c: _Candidate) -> bool:
        return c.init == c.fin and list(c.res.initial_layout) == list(
            c.res.final_layout
        )

    one = _transpile_network(n, net, lay, cfg, gname, target, optimization_level, seed)
    if not same(one):
        one = _transpile_network(
            n, net, lay, cfg, gname, target, optimization_level, seed, elide=False
        )
        if not same(one):
            return None
    qc = QuantumCircuit()
    wires = list(range(one.circuit.num_qubits))
    for _ in range(repeat):
        qc.compose(one.circuit, wires, inplace=True)
    two_q, depth, _ = _score(qc)
    return replace(one, key=(two_q, depth), circuit=qc)


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
    """Compile the ``steps``-step formula of ``terms`` (already in term order).

    ``options["block_steps"]``: ``None`` (automatic, see :data:`BLOCK_RATIO`),
    ``0`` (never) or the steps of a repeated block.
    """
    opts = dict(options)
    block = opts.pop("block_steps", None)
    rotations = product_formula(terms, time, steps, order)
    ladder: list[_Candidate] | None = None
    network = "network" in opts["methods"]
    if block is None:
        block = 0
        if steps >= 2 and network:
            merged = merge_commuting(rotations, n)
            try:
                cands = _compile_rotations(
                    n, terms, merged, target, bound=bound, probe=True, **opts
                )
            except _TooBigError as exc:
                block = 1
                ladder = exc.ladder
            else:
                if not cands:
                    return None
                return _result(n, term_order, order, steps, rotations, cands)
    if block and network:
        if block > steps:
            raise ValueError("block_steps must not exceed steps")
        return _compile_blocks(
            n, terms, time, steps, order, target, term_order, opts, bound, block, ladder
        )
    merged = merge_commuting(rotations, n)
    cands = _compile_rotations(n, terms, merged, target, bound=bound, **opts)
    if not cands:
        return None
    return _result(n, term_order, order, steps, rotations, cands)


def _compile_blocks(
    n: int,
    terms: Sequence[PauliTerm],
    time: float,
    steps: int,
    order: int,
    target: CompilationTarget,
    term_order: str,
    options: dict[str, Any],
    bound: int,
    block: int,
    ladder: list[_Candidate] | None = None,
) -> EvolutionResult | None:
    """The ``steps``-step formula as ``steps / block`` copies of one
    ``block``-step network circuit.

    The steps of a product formula are identical, so the formula is the
    block formula repeated (the merges of rotations across block boundaries
    are given up).  The block is synthesized by the network with its final
    frame synthesized exactly (no output permutation), so it starts and ends
    on the same layout; the portfolio runs on the block alone and the copies
    of the finalists are transpiled together (large ones: the transpiled block
    is repeated, see :data:`REPEAT_TRANSPILE_CX`).  The ladder of the whole
    formula (when requested) is compiled first (or taken from ``ladder``) and
    competes; it also bounds the block runs (:data:`NETWORK_ABORT_RATIO`).
    """
    if steps % block:
        raise ValueError(f"steps ({steps}) must be a multiple of block_steps ({block})")
    reps = steps // block
    rotations = product_formula(terms, time, steps, order)
    cands: list[_Candidate] = []
    if "ladder" in options["methods"]:
        # the ladder of the whole formula competes and bounds the block runs
        if ladder:
            cands += ladder
        else:
            lad = dict(options)
            lad["methods"] = ("ladder",)
            whole = merge_commuting(rotations, n)
            cands += _compile_rotations(n, terms, whole, target, **lad)
        cap = int(NETWORK_ABORT_RATIO * cands[0].key[0]) + 1
        bound = cap if bound <= 0 else min(bound, cap)
    sub = product_formula(terms, time * block / steps, block, order)
    merged = merge_commuting(sub, n)
    opts = dict(options)
    opts["methods"] = ("network",)
    opts["finalize_permutation"] = False
    each = 0 if bound <= 0 else bound // reps + 1
    cands += _compile_rotations(
        n, terms, merged, target, bound=each, repeat=reps, **opts
    )
    if not cands:
        return None
    res = _result(n, term_order, order, steps, rotations, cands)
    if res.method == "network":
        res.block_steps = block
    return res


def _fixed_options(options: dict[str, Any], res: EvolutionResult) -> dict[str, Any]:
    """``options`` restricted to the candidate selected in ``res``."""
    out = dict(options)
    out["methods"] = (res.method,)
    out["block_steps"] = res.block_steps or 0
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
                if four is None or (b4 and four.two_qubit_gates > b4):
                    continue  # above the bound: cannot win
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
        "block_steps": None,
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
    block_steps: int | None = None,
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
            :data:`NETWORK_CONFIGS` on every graph and layout, then
            :data:`EXTENDED_CONFIGS` on the best ones, within
            :data:`WORK_BUDGET` (screened on prefixes when needed).
        layouts: Initial layouts (``layout[logical] = physical``) for the
            network candidate instead of the automatic ones.
        finalize_permutation: Synthesize the network's final frame only up
            to a qubit permutation (absorbed into ``final_layout``).
        max_steps: Largest step count tried with ``eps``.
        network_graphs: Graphs the network candidate is synthesized on
            (subset of :data:`NETWORK_GRAPHS`).
        term_order_cost: ``"compile"`` or ``"estimate"`` (``term_order="auto"``).
        block_steps: Compile the formula as copies of one compiled
            ``block_steps``-step circuit (``steps`` must be a multiple; the
            block's final frame is synthesized exactly so that copies
            compose; network only).  ``0``: never.  ``None`` (default):
            a one-step block when a full network run of the whole formula
            is predicted to exceed ``WORK_BUDGET / BLOCK_RATIO`` (see
            :data:`BLOCK_RATIO`) — very large formulas then compile in a
            time independent of ``steps``.

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
        "block_steps": block_steps,
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
    if block_steps is not None and (block_steps < 0 or block_steps > steps):
        raise ValueError("block_steps must be None or in [0, steps]")

    res = _compile_formula(n, terms, time, steps, order, target, term_order, options)
    if res is None:
        raise RuntimeError("no candidate compiled (network aborted without a ladder)")
    res.error_estimate, res.error_metric = err_est, metric
    res.term_order_table = None if choice is None else choice.table
    return res
