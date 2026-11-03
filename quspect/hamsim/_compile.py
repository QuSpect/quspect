"""``compile_evolution``: ``exp(-iHt)`` to a circuit for a compilation target."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

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
#: CX than the best of a 72–108 setting grid); this portfolio was picked
#: greedily from such grids (all-to-all and device graphs) and comes within
#: about 1 % of the whole grid.  The block settings (``block_mode=1``) also
#: consider emitting two-qubit blocks on coupling edges in place or with a
#: mirror SWAP.  Runs are pruned by branch and bound, so the portfolio costs
#: far less than its size suggests.
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
)

#: The network candidate is abandoned once its CX count exceeds this
#: multiple of the ladder candidate's two-qubit count (it cannot win).
NETWORK_ABORT_RATIO = 1.6

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


def _naive_cx(rotations: Sequence[Rotation]) -> int:
    return sum(2 * (len(q) - 1) for _, q, _ in rotations)


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
    network_configs: Sequence[Sequence[int]],
    layouts: Sequence[Sequence[int]] | None,
    finalize_permutation: bool,
) -> list[_Candidate]:
    """Compile a (merged) rotation sequence with every requested method.

    Normally the ladder is compiled first and bounds the network runs
    (:data:`NETWORK_ABORT_RATIO`).  For formulas above
    :data:`LARGE_FORMULA_CX` textbook CX the network goes first (bounded by
    ``NETWORK_ABORT_RATIO`` times the textbook count) and the expensive
    ladder transpilation is skipped when the network needs fewer than
    ``1 / LADDER_SKIP_RATIO`` of the textbook CX.
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
        limit = int(NETWORK_ABORT_RATIO * (out[0].key[0] if out else naive)) + 1
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
    network_configs: Sequence[Sequence[int]],
    layouts: Sequence[Sequence[int]] | None,
    finalize_permutation: bool,
    limit: int,
) -> _Candidate | None:
    edges = _edges(target)
    if layouts is not None:
        lays = [list(lay) for lay in layouts]
    elif _is_full(target, n):
        lays = [list(range(n))]
    else:
        lays = _layout_candidates(n, terms, merged, target, seed, layout_trials)
    configs = [_norm_config(c) for c in network_configs]
    if not configs:
        raise ValueError("network_configs is empty")
    best = None
    for lay in lays:
        used = sorted({lay[q] for _, qs, _ in merged for q in qs})
        active = connected_active_set(target.num_qubits, edges, used)
        for cfg in configs:
            window, front, next_weight, decay, block_mode, block_bias = cfg
            # branch and bound: a run above the best CX count so far cannot win
            cap = limit if best is None else min(limit, best[0].num_cx)
            net = synthesize_network(
                merged,
                target.num_qubits,
                edges,
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
            if best is None or net.num_cx < best[0].num_cx:
                best = (net, lay, cfg)
    if best is None:
        return None
    net, lay, cfg = best
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
    return _Candidate(key, "network", res.circuit, init, fin, res, lay, cfg)


def _formula_cost(
    n: int,
    terms: Sequence[PauliTerm],
    time: float,
    steps: int,
    order: int,
    seed: int,
    cap: int = 20000,
) -> int:
    """Estimated two-qubit count of the ``steps``-step formula on all-to-all
    connectivity (network with two settings and the ladder at level 1; the
    count of large formulas is extrapolated from one and two steps)."""
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
    if steps <= 2 or per_step * steps <= cap:
        return count(steps, time)
    c1 = count(1, time / steps)
    c2 = count(2, 2 * time / steps)
    return c1 + (steps - 1) * max(0, c2 - c1)


@dataclass
class TermOrderChoice:
    """Output of :func:`select_term_order`.

    Attributes:
        term_order: The selected term order.
        steps: Its step count for ``eps``.
        error_estimate: Its estimated error.
        error_metric: ``"state"`` or ``"bound"``.
        table: ``{term order: (steps, error estimate, estimated two-qubit
            gates)}`` for every candidate.
    """

    term_order: str
    steps: int
    error_estimate: float
    error_metric: str
    table: dict[str, tuple[int, float, int]]


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
) -> TermOrderChoice:
    """Choose the term order of a product formula by its own error estimate and cost.

    Reordering non-commuting terms changes the Trotter error, so a term
    order that makes each step cheaper can need more steps.  For every
    candidate order this finds the fewest steps meeting ``eps`` by this
    module's error estimate (``error_metric``: simulated state error on its
    own random states, or the commutator bound) and estimates the
    two-qubit count of that formula on all-to-all connectivity (Pauli
    network and ladder, see :func:`compile_evolution`); the cheapest order
    wins (ties: the earlier candidate).

    Args:
        hamiltonian: Pauli sum (see :func:`quspect.hamsim.pauli_terms`).
        time: Evolution time.
        eps: Error target.
        order: Product-formula order (1, 2 or 4).
        candidates: Term orders to compare (see
            :func:`quspect.hamsim.order_terms`).
        error_metric: ``"auto"``, ``"state"`` or ``"bound"``.
        num_qubits: Number of qubits (default from the Hamiltonian).
        seed: Seed of the cost estimate.
        max_steps: Largest step count tried.

    Returns:
        The selected order and the comparison table.
    """
    n, terms = pauli_terms(hamiltonian, num_qubits)
    if not candidates:
        raise ValueError("candidates is empty")
    metric = _resolve_metric(error_metric, n)
    reference = None
    if metric == "state":
        from quspect.hamsim._formulas import _exact_evolve, _test_states

        # the exact evolution does not depend on the term order
        reference = _exact_evolve(n, terms, time, _test_states(n, 2, 20261102))
    table: dict[str, tuple[int, float, int]] = {}
    best: str | None = None
    for name in candidates:
        ordered = order_terms(terms, name)
        r, e = _steps_for_eps(
            n, ordered, time, order, eps, metric, max_steps, reference
        )
        cost = _formula_cost(n, ordered, time, r, order, seed)
        table[name] = (r, e, cost)
        if best is None or cost < table[best][2]:
            best = name
    assert best is not None
    r, e, _ = table[best]
    return TermOrderChoice(best, r, e, metric, table)


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
    network_configs: Sequence[Sequence[int]] = NETWORK_CONFIGS,
    layouts: Sequence[Sequence[int]] | None = None,
    finalize_permutation: bool = True,
    max_steps: int = 4096,
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
      output layout absorbs.  The initial layout is the best (fewest CX) of
      ``layout_trials`` low-dilation embeddings of the interaction graph
      and SABRE's layout of the ladder circuit (or the given ``layouts``),
      each with every setting in ``network_configs``; runs are abandoned
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
    ``eps``) picks the term order by :func:`select_term_order`.

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
            layout (the fewest CX wins); default :data:`NETWORK_CONFIGS`.
        layouts: Initial layouts (``layout[logical] = physical``) for the
            network candidate instead of the automatic ones.
        finalize_permutation: Synthesize the network's final frame only up
            to a qubit permutation (absorbed into ``final_layout``).
        max_steps: Largest step count tried with ``eps``.

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
        )
        term_order = choice.term_order
        if steps is None:
            steps, err_est, metric = (
                choice.steps,
                choice.error_estimate,
                choice.error_metric,
            )
    terms = order_terms(terms, term_order)

    if steps is None:
        if eps is None:
            raise ValueError("give steps or eps")
        metric = _resolve_metric(error_metric, n)
        steps, err_est = _steps_for_eps(n, terms, time, order, eps, metric, max_steps)
    elif steps < 1:
        raise ValueError("steps must be >= 1")

    rotations = product_formula(terms, time, steps, order)
    merged = merge_commuting(rotations, n)
    cands = _compile_rotations(
        n,
        terms,
        merged,
        target,
        methods=methods,
        optimization_level=optimization_level,
        ladder_level=ladder_level,
        seed=seed,
        layout_trials=layout_trials,
        network_configs=network_configs,
        layouts=layouts,
        finalize_permutation=finalize_permutation,
    )
    if not cands:
        raise RuntimeError("no candidate compiled (network aborted without a ladder)")
    scores = {c.method: c.key[0] for c in cands}
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
        error_estimate=err_est,
        error_metric=metric,
        candidates=scores,
        transpile_result=best.res,
        network_layout=best.layout,
        network_config=best.config,
        term_order_table=None if choice is None else choice.table,
    )
