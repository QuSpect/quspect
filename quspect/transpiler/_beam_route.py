"""Global beam-search SWAP routing as an extra level-3 candidate.

SABRE (and its LNS refinement) follows one greedy routing per layout; the
beam router (``cpp/src/transpiler/beam_routing.cpp``, see
``beam_routing.hpp``) keeps the ``BEAM_WIDTH`` best partial routings at every
step instead.  All of them advance by exactly one SWAP per step and then
execute every block that became adjacent; children are ranked by front-layer
distance, a decayed per-qubit lookahead, a bonus per executed block and a
SABRE-like decay penalty for reusing the same physical qubits without
progress, and states already reached earlier (same layout, same progress)
are dropped.  The layout is refined by forward/backward passes of the beam
router itself.

It optimizes the plain SWAP count, not the merge-aware CZ estimate, so its
routing only becomes a candidate: it is scored like every other one
(``SABRE._finish_scored``) and fully evaluated by level 3, which keeps the
best result.  A routing that needs more than ``BEAM_MAX_SWAP_RATIO`` times
the fewest SABRE SWAPs is abandoned, and the beam narrows (or drops the
layout iterations, or is skipped) to stay within a deterministic work
budget.

Measured results (quspect level 3, CZ basis, beam candidate on vs off,
median of seeds per case): the geometric-mean two-qubit count drops by
0.47 % on Standard-42 (210 cases), 0.79 % on the synthetic suite (196) and
0.42 % on Large-20 (80), with up to 13 % on single circuits and no case
worse; compile time grows by 20 %, 69 % and 19 %.  Routing alone (SWAPs
inserted, no gate merging), on the 72 instances of SimpleTES (arXiv
2604.19341; 24 SABRE circuits on IBM Q20, Google Willow and IBM Heron) it
inserts 18.4 % fewer SWAPs than SimpleTES's evolved router and 30 % fewer
than LightSABRE with 20 layout x 20 swap trials, and 18.5 % / 19.3 % fewer
on 117 held-out instances.

Beam search for qubit routing has been studied before (Qiu, Zhu and Wei,
Entropy 27:232, 2025; Nishio et al., ACM JETC 16:3, 2020; Qiskit's
``LookaheadSwap``).  This implementation keeps one global beam to the end of
the circuit with a compact state (per-qubit chain pointers plus layout),
never revisits a state, has a release valve, and starts from the best
distinct layouts of the SABRE candidates.

The routing walks the blocks in program order on every qubit, so it applies
to DAGs without three-qubit nodes whose measurements only precede other
measurements; measurement nodes are drained as soon as their predecessors
ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quspect.transpiler._layer_route import _interleave, _plan_from
from quspect.transpiler._routing import _native_core

if TYPE_CHECKING:
    from quspect.transpiler._mapping import SABRE, _CandidatePlan, _RoutingDag

#: Partial routings kept per step.
BEAM_WIDTH = 128
#: Narrowest beam used when the work budget shrinks it.
BEAM_MIN_WIDTH = 16
#: Lookahead blocks per qubit, their decay and weight, the bonus per executed
#: block and the decay penalty per repeated use of a physical qubit since the
#: last execution (tuned on the SimpleTES suite, the same for all devices).
BEAM_EXT_DEPTH = 8
BEAM_GAMMA = 0.7
BEAM_EXT_WEIGHT = 0.5
BEAM_EXEC_BONUS = 3.0
BEAM_DECAY = 1.0
#: Forward/backward layout refinement passes per start (dropped to 0 when the
#: budget does not allow the narrowest beam otherwise).
BEAM_LAYOUT_ITERS = 2
#: Distinct start layouts (best-ranked SABRE candidates first), routed
#: concurrently.
BEAM_STARTS = 4
#: A beam routing is abandoned once it needs more than this multiple of the
#: fewest SWAPs among the SABRE candidates (it could not win anyway; this
#: bounds the time on circuits where the beam search does badly).
BEAM_MAX_SWAP_RATIO = 1.5
#: Deterministic work budget per start, in elementary steps of the native
#: router (about 1.2 ns each on an Apple M3, so 6e9 is about 7 s): routing
#: passes x SWAP steps (at most the cap above) x beam width x candidate SWAPs
#: per state x cost of one state (see :func:`beam_config`).
BEAM_WORK = 6_000_000_000


def beam_applicable(dag: _RoutingDag) -> bool:
    """Whether the beam router can route ``dag`` (see the module docs)."""
    if dag.has_three_qubit:
        return False
    for v, q in enumerate(dag.nq1):
        if q < 0 and any(dag.nq1[s] >= 0 for s in dag.succ[v]):
            return False
    return any(q >= 0 for q in dag.nq1)


def beam_config(
    fewest_swaps: int,
    num_physical: int,
    active_qubits: int,
    num_edges: int,
    work: int = BEAM_WORK,
) -> tuple[int, int] | None:
    """``(width, layout_iters)`` within ``work``, or ``None`` if too costly.

    One routing pass of width ``w`` takes at most ``steps`` beam steps
    (``BEAM_MAX_SWAP_RATIO * fewest_swaps``), each expanding ``w`` states into
    at most ``cands`` children (coupling edges at the front qubits, bounded
    by ``min(num_edges, 2 * num_edges / num_physical * active_qubits)``), and
    a child costs a state copy plus the lookahead (``num_physical +
    active_qubits * (BEAM_EXT_DEPTH + 1)``).  The widest beam up to
    ``BEAM_WIDTH`` that fits is used with ``BEAM_LAYOUT_ITERS`` layout
    iterations (``2 * iters + 1`` passes), else with a single pass; below
    ``BEAM_MIN_WIDTH`` the beam router is skipped.

    Example:
        >>> beam_config(100, 20, 16, 30)
        (128, 2)
        >>> beam_config(10**7, 100, 100, 180) is None
        True
    """
    steps = max(1, int(BEAM_MAX_SWAP_RATIO * fewest_swaps))
    mean_degree = 2 * num_edges / max(1, num_physical)
    cands = max(1, min(num_edges, round(mean_degree * active_qubits)))
    per_child = num_physical + active_qubits * (BEAM_EXT_DEPTH + 1)
    per_width = steps * cands * per_child
    for iters in (BEAM_LAYOUT_ITERS, 0):
        width = min(BEAM_WIDTH, work // max(1, (2 * iters + 1) * per_width))
        if width >= BEAM_MIN_WIDTH:
            return width, iters
    return None


def beam_route_plan(
    router: SABRE,
    plans: list[_CandidatePlan],
    *,
    seed: int = 1,
    work_scale: float = 1.0,
) -> _CandidatePlan | None:
    """Beam-search routing of the default candidate's circuit, as a candidate.

    Args:
        router: Router of the target coupling graph (scores the routing).
        plans: SABRE candidates, best ranked first; the default one supplies
            the routing DAG, and the first ``BEAM_STARTS`` distinct layouts
            of the candidates sharing its preparation are the starts.
        seed: Base seed of the tie-breaking noise (start ``i`` uses
            ``seed + 7919 * i``).
        work_scale: Multiplier of the work budget ``BEAM_WORK`` (level 3
            passes ``work_limit`` relative to its default).

    Returns:
        A plan labelled ``beam/<default label>`` (never the default), or
        ``None`` if the beam router does not apply, the native core is
        missing, the circuit exceeds the work budget or every start needs
        more than ``BEAM_MAX_SWAP_RATIO`` times the fewest SABRE SWAPs.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import SABRE, CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> sabre = SABRE(target)
        >>> plans = sabre._candidate_plans(
        ...     QuantumCircuit().cx(0, 2).cx(0, 1).cx(1, 2), seed=1, exact_layout=False
        ... )
        >>> plan = beam_route_plan(sabre, plans)
        >>> plan is None or plan.label.startswith("beam/")
        True
    """
    base = next((p for p in plans if p.is_default and p.prep is not None), None)
    if base is None or not base.route_layout:
        return None
    prep = base.prep
    assert prep is not None
    dag = prep.dag
    n = router._num_physical
    if not beam_applicable(dag) or len(base.route_layout) != n:
        return None
    core: Any = _native_core()
    if core is None or not hasattr(core, "_route_beam_many"):
        return None
    nodes = [v for v in range(len(dag.nodes)) if dag.nq1[v] >= 0]
    fewest = min(p.num_swaps for p in plans if p.prep is prep)
    active = len({q for v in nodes for q in (dag.nq1[v], dag.nq2[v])})
    num_edges = sum(len(a) for a in router._adj_sets) // 2
    config = beam_config(fewest, n, active, num_edges, int(BEAM_WORK * work_scale))
    if config is None:
        return None
    width, iters = config
    starts: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for p in plans:
        if p.prep is not prep or len(p.route_layout) != n:
            continue
        key = tuple(p.route_layout)
        if key in seen:
            continue
        seen.add(key)
        starts.append(list(p.route_layout))
        if len(starts) >= BEAM_STARTS:
            break
    max_swaps = max(1, int(BEAM_MAX_SWAP_RATIO * fewest))
    adjacency = [sorted(a) for a in router._adj_sets]
    blocks = [(dag.nq1[v], dag.nq2[v]) for v in nodes]
    outs = core._route_beam_many(
        adjacency,
        blocks,
        starts,
        [seed + 7919 * i for i in range(len(starts))],
        width,
        BEAM_EXT_DEPTH,
        BEAM_GAMMA,
        BEAM_EXT_WEIGHT,
        BEAM_EXEC_BONUS,
        iters,
        max_swaps,
        BEAM_DECAY,
    )
    outs = [out for out in outs if int(out[0]) >= 0]
    if not outs:
        return None
    # 最少 SWAP 的起点（并列取先出现的）
    swaps, layout, codes = min(outs, key=lambda out: int(out[0]))
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
    return _plan_from(router, base, (list(layout), events, int(swaps)), "beam")
