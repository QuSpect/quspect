"""``compile_evolution``: ``exp(-iHt)`` to a circuit for a compilation target."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from quspect.circuit.circuit import QuantumCircuit
from quspect.hamsim._formulas import (
    STATE_ERROR_MAX_QUBITS,
    Rotation,
    commutator_bound,
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

#: Gate-score settings ``(window, front, next_weight, decay)`` tried by the
#: network candidate (the cheapest result is kept).  The greedy synthesis is
#: sensitive to its look-ahead; a few settings that differ in horizon and
#: decay cover each other's bad cases (about 15 % fewer CX than the best
#: single setting on the development set).
NETWORK_CONFIGS: tuple[tuple[int, int, int, int], ...] = (
    (16, 16, 512, 1000),
    (32, 16, 512, 900),
    (16, 16, 1024, 1000),
    (32, 64, 512, 800),
    (16, 64, 512, 800),
)


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
        term_order: Term order of the formula (``"input"`` / ``"support"``).
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
    network_configs: Sequence[tuple[int, int, int, int]] = NETWORK_CONFIGS,
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
      routing, and the final frame is synthesized on the same graph.  The
      initial layout is the best (fewest CX) of ``layout_trials``
      low-dilation embeddings of the interaction graph and SABRE's layout
      of the ladder circuit.  The result is then optimized by
      :func:`~quspect.transpiler.transpile` at ``optimization_level`` with
      the layout fixed (no routing is needed).
    - ``"ladder"``: the textbook circuit (basis change, CX ladder, ``rz``
      per rotation) through :func:`~quspect.transpiler.transpile` at
      ``ladder_level``.

    The candidate with the fewest two-qubit gates (then the smallest
    two-qubit depth) is returned.

    With ``eps`` and no ``steps``, the smallest ``steps`` whose estimated
    error is at most ``eps`` is used.  ``error_metric``: ``"state"`` (the
    worst simulated state error over random states and ``|0101…⟩``, up to
    16 qubits), ``"bound"`` (commutator bound, orders 1 and 2) or
    ``"auto"`` (``"state"`` when possible).

    Args:
        hamiltonian: Pauli sum (see :func:`quspect.hamsim.pauli_terms`).
        time: Evolution time ``t``.
        target: Compilation target; default: all-to-all connectivity on
            the Hamiltonian's qubits with a CZ gate set.
        steps: Number of Trotter steps.
        order: Product-formula order (1, 2 or 4).
        eps: Error target, used when ``steps`` is not given.
        term_order: ``"input"`` or ``"support"`` (see
            :func:`quspect.hamsim.order_terms`).
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
        network_configs: ``(window, front, next_weight, decay)`` settings
            of :func:`~quspect.hamsim.synthesize_network` tried for every
            layout (the fewest CX wins); default :data:`NETWORK_CONFIGS`.
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
    terms = order_terms(terms, term_order)

    err_est: float | None = None
    metric: str | None = None
    if steps is None:
        if eps is None:
            raise ValueError("give steps or eps")
        metric = error_metric
        if metric == "auto":
            metric = "state" if n <= STATE_ERROR_MAX_QUBITS else "bound"
        if metric == "state":
            from quspect.hamsim._formulas import _exact_evolve, _test_states

            ref = _exact_evolve(n, terms, time, _test_states(n, 2, 20261102))

            def err(r: int) -> float:
                return state_error(
                    n,
                    terms,
                    time,
                    product_formula(terms, time, r, order),
                    reference=ref,
                )

        elif metric == "bound":
            if order == 4:
                raise ValueError("error_metric='bound' supports order 1 and 2")

            def err(r: int) -> float:
                return commutator_bound(n, terms, time, r, order)

        else:
            raise ValueError("error_metric must be 'auto', 'state' or 'bound'")
        steps, err_est = _min_steps(err, eps, max_steps)
    elif steps < 1:
        raise ValueError("steps must be >= 1")

    rotations = product_formula(terms, time, steps, order)
    merged = merge_commuting(rotations, n)
    results: list[
        tuple[
            tuple[int, int], str, QuantumCircuit, list[int], list[int], TranspileResult
        ]
    ] = []
    scores: dict[str, int] = {}

    if "network" in methods and merged:
        edges = _edges(target)
        if _is_full(target, n):
            layouts = [list(range(n))]
        else:
            layouts = _layout_candidates(n, terms, merged, target, seed, layout_trials)
        if not network_configs:
            raise ValueError("network_configs is empty")
        best = None
        for lay in layouts:
            used = sorted({lay[q] for _, qs, _ in merged for q in qs})
            active = connected_active_set(target.num_qubits, edges, used)
            for window, front, next_weight, decay in network_configs:
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
                )
                if best is None or net.num_cx < best[0].num_cx:
                    best = (net, lay)
        assert best is not None
        net, lay = best
        qc = net.circuit
        _pad(qc, sorted(set(lay)))
        res = transpile(
            qc,
            target,
            optimization_level=optimization_level,
            seed=seed,
            initial_layout=list(range(qc.num_qubits)),
        )
        init = [res.initial_layout[lay[q]] for q in range(n)]
        fin = [res.final_layout[lay[q]] for q in range(n)]
        key = (res.stats.two_qubit_gates, res.stats.two_qubit_depth)
        results.append((key, "network", res.circuit, init, fin, res))
        scores["network"] = key[0]

    if "ladder" in methods or not merged:
        qc = ladder_circuit(merged, n)
        _pad(qc, range(n))
        res = transpile(
            qc, target, optimization_level=ladder_level or optimization_level, seed=seed
        )
        key = (res.stats.two_qubit_gates, res.stats.two_qubit_depth)
        results.append(
            (
                key,
                "ladder",
                res.circuit,
                list(res.initial_layout)[:n],
                list(res.final_layout)[:n],
                res,
            )
        )
        scores["ladder"] = key[0]

    results.sort(key=lambda item: item[0])
    key, method, circuit, init, fin, res = results[0]
    return EvolutionResult(
        circuit=circuit,
        initial_layout=init,
        final_layout=fin,
        order=order,
        steps=steps,
        term_order=term_order,
        rotations=rotations,
        method=method,
        two_qubit_gates=key[0],
        two_qubit_depth=key[1],
        error_estimate=err_est,
        error_metric=metric,
        candidates=scores,
        transpile_result=res,
    )
