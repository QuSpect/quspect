"""SABRE qubit mapping and routing (LightSABRE-style "SABRE v2").

Implements SWAP-based bidirectional heuristic search from:

    Li, Gushu, Yufei Ding, and Yuan Xie.
    "Tackling the qubit mapping problem for NISQ-era quantum devices."
    ASPLOS 2019.

with the refinements of:

    Zou, Henry, Matthew Treinish, Kevin Hartman, Alexander Ivrii, and
    Jake Lishman. "LightSABRE: A Lightweight and Enhanced SABRE Algorithm."
    arXiv:2409.08368 (2024).

Algorithm overview:

* The circuit is compressed into a dependency DAG whose nodes are the
  two-qubit gates and measurements (measurements act as scheduling fences).
  Single-qubit gates are attached to the next node on their qubit and are
  emitted just before it, so they never influence routing decisions.
* Routing keeps a front layer ``F`` of blocked two-qubit gates and a
  bounded extended set ``E`` (at most ``extended_set_size`` gates, found by
  a breadth-first walk of the DAG successors of ``F``).  Candidate SWAPs
  touch a physical qubit of a front-layer gate and are scored by::

      H = max(decay[a], decay[b]) * (sum_F D + W * sum_E D / |E|)

  The front term is not divided by ``|F|`` (as in Qiskit's SABRE; ``|F|``
  is the same for every candidate, so this only re-weights it against the
  extended-set term, which measured slightly better).
  Only the change caused by the SWAP on gates touching the two swapped
  qubits is evaluated, so scoring a candidate costs ``O(degree)`` rather
  than ``O(|F| + |E|)``.  Ties are broken uniformly at random (seeded).
* Decay follows the reference implementation: each chosen SWAP adds
  ``decay_delta`` to its two physical qubits, every ``decay_reset_interval``
  SWAPs all decay values reset to 1, and they also reset whenever a gate
  becomes executable.
* Release valve: if ``release_valve_threshold`` SWAPs were inserted without
  any gate becoming executable, those SWAPs are undone and the front gate
  with the smallest distance is routed along a shortest path, which
  guarantees progress (no livelock).
* Layout search: ``layout_trials`` random initial layouts plus the trivial
  layout (and, with ``dilation_layouts``, low-dilation embeddings of the
  interaction graph, see :mod:`quspect.transpiler._embedding`) are each
  refined by ``iter_num`` forward/backward passes; the trivial and the
  low-dilation layouts are also tried unrefined.  The candidate whose final
  forward routing needs the fewest SWAPs (then the smallest two-qubit
  depth, then the lowest trial index) is kept.

Variable naming follows the original paper: F (front layer), E (extended
set), D (distance matrix), H (heuristic function), pi (mapping).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from quspect.synthesis.two_qubit import SynthOp, ops_to_instructions
from quspect.transpiler._embedding import dilation_layouts
from quspect.transpiler._routing import _RouteResult, incident_edges, route_dag
from quspect.transpiler._target import CompilationTarget
from quspect.transpiler._toffoli import ROUTABLE_THREE_QUBIT_GATES, expand_toffolis
from quspect.transpiler._vf2 import DEFAULT_VF2_MAX_STATES, find_subgraph_monomorphism

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.gates.gate import Gate

logger = logging.getLogger(__name__)

_INF = 10**9
_TIE_EPS = 1e-10

DEFAULT_LAYOUT_TRIALS = 6
"""Default number of random layout trials used by :meth:`SABRE.solve`."""

# 候选打分（回放、Toffoli 综合、代价估计）计入 work_limit 的工作量：每个输入门与
# SWAP 计 1，每个整体路由的 Toffoli 另计这么多（实测约 150 µs / 个 vs 约 1 µs / 门）。
_SCORE_WORK_PER_CCX = 150
# 低伸缩布局起点（见 _embedding）的路由种子下标，与 layout_trials 无关
_DILATION_SEED_INDEX = 1 << 20


# =============================================================================
# Simplified gate representation for mapping
# =============================================================================


@dataclass
class _MappingGate:
    """Lightweight gate representation used internally by the mapping pass.

    Only distinguishes single-qubit gates from two-qubit gates, since
    single-qubit gates are always executable regardless of mapping.  A
    ``ccx`` routed as a three-qubit unit has ``is_two_qubit=False`` and
    ``q3 >= 0`` (controls ``q1``, ``q2``, target ``q3``).
    """

    is_two_qubit: bool
    q1: int  # first logical qubit
    q2: int  # second logical qubit (equals q1 for single-qubit gates)
    index: int  # original position in the instruction list
    name: str = ""  # for debugging
    clbit: int | None = None  # original classical target for measurements
    gate: Gate | None = None  # synthesized gate (overrides the original one)
    q3: int = -1  # third logical qubit (target) of a routed ``ccx``, else -1


@dataclass
class _RoutingDag:
    """Compressed dependency DAG used by the router.

    Nodes are two-qubit gates and measurements.  ``nq1[n]``/``nq2[n]`` are
    the logical qubits of a two-qubit node, or ``-1`` for a measurement
    (which is always executable); ``nq3[n]`` is the target of a three-qubit
    ``ccx`` node and ``-1`` otherwise.  ``prefix[n]`` holds the single-qubit
    gates that must be emitted right before node ``n``; ``trailing`` holds
    the single-qubit gates after the last node on their qubit.
    """

    nodes: list[_MappingGate] = field(default_factory=list)
    nq1: list[int] = field(default_factory=list)
    nq2: list[int] = field(default_factory=list)
    nq3: list[int] = field(default_factory=list)
    has_three_qubit: bool = False
    succ: list[list[int]] = field(default_factory=list)
    pred: list[list[int]] = field(default_factory=list)
    prefix: list[list[_MappingGate]] = field(default_factory=list)
    trailing: list[_MappingGate] = field(default_factory=list)
    # 块节点（merge_pairs）：节点 n 之后紧接着在同一比特对上执行的两比特门，每个带着
    # 它自己前面的单比特门 (prefix, gate)。
    extra: list[list[tuple[list[_MappingGate], _MappingGate]]] = field(
        default_factory=list
    )
    # 镜像门代价（见 _annotate_mirror_costs）：mirror_fwd[n] >= 0 表示 n 是其比特对块的
    # 最后一个两比特节点（正向路由），mirror_bwd[n] 同理对应第一个（反向路由）；值为
    # cz(U·SWAP) - cz(U)。为空表示未标注（不做镜像）。
    mirror_fwd: list[int] = field(default_factory=list)
    mirror_bwd: list[int] = field(default_factory=list)
    # 原生路由核心的 DAG 副本 (core 模块, 对象)，首次原生路由时创建；
    # DAG 建好后不再修改。
    native: Any = field(default=None, repr=False, compare=False)


@dataclass
class _Prepared:
    """Routing input shared by every layout/routing trial of one circuit."""

    circuit: QuantumCircuit
    gates: list[_MappingGate]
    num_logical: int
    sigma: list[int]
    dag: _RoutingDag
    groups: list[list[int]]
    elide: bool
    #: per-node merge costs and measured qubits (see ``_route_lns``), cached
    lns_model: tuple[list[int], list[int]] | None = field(default=None, repr=False)
    #: per-node two-qubit layer weights of the LNS depth proxy, cached
    lns_weight: list[int] | None = field(default=None, repr=False)


@dataclass
class RoutingCandidate:
    """One routed version of a circuit produced by :meth:`SABRE.route_candidates`.

    The circuit is in *wire space*, exactly like the output of
    :meth:`SABRE.solve`: wire ``w`` sits on physical qubit
    ``initial_mapping[w]`` for the whole circuit and SWAPs move state
    between wires.

    Attributes:
        circuit: Routed circuit (wire space, SWAPs not yet decomposed).
        initial_mapping: ``initial_mapping[wire] = physical``; logical
            wires first, then workspace wires.
        final_mapping: ``final_mapping[wire] = physical`` where the state
            of the wire ends (includes elided permutations).
        num_swaps: Routing SWAPs inserted by the router (before trailing
            SWAP elision).
        depth: Two-qubit depth of the routed circuit (SWAP = 1 layer).
        estimated_two_qubit_gates: Cheap estimate of the CZ count after
            the routed circuit is decomposed and its two-qubit blocks are
            resynthesized (see :func:`estimate_two_qubit_cost`).
        label: Human-readable origin, e.g. ``"exact"`` or ``"trial3/swap1"``.
        is_default: True for the candidate :meth:`SABRE.solve` returns
            with the same arguments (fewest SWAPs, then smallest depth,
            then lowest trial index; see ``reference_trials``).

    Example:
        >>> from quspect import QuantumCircuit
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> qc = QuantumCircuit().cx(0, 1).cx(1, 2).cx(0, 2)
        >>> cands = SABRE(target).route_candidates(qc, seed=1, layout_trials=2)
        >>> any(c.is_default for c in cands)
        True
    """

    circuit: QuantumCircuit
    initial_mapping: list[int]
    final_mapping: list[int]
    num_swaps: int
    depth: int
    estimated_two_qubit_gates: int
    label: str
    is_default: bool = False


@dataclass
class _CandidatePlan:
    """A :class:`RoutingCandidate` whose circuit is built on demand."""

    source: QuantumCircuit
    num_logical: int
    gates: list[_MappingGate]
    initial_mapping: list[int]
    final_mapping: list[int]
    num_swaps: int
    depth: int
    estimated_two_qubit_gates: int
    label: str
    is_default: bool = False
    #: routed from a low-dilation layout start (see ``dilation_layouts``)
    embedded: bool = False
    #: the routing itself: shared preparation, full start permutation and
    #: event sequence (input of the large-neighbourhood search, see
    #: :mod:`quspect.transpiler._route_lns`)
    prep: _Prepared | None = field(default=None, repr=False, compare=False)
    route_layout: list[int] = field(default_factory=list, repr=False, compare=False)
    route_events: list[int] = field(default_factory=list, repr=False, compare=False)

    def build_circuit(self) -> QuantumCircuit:
        """The routed wire-space circuit (built anew on every call)."""
        return _build_output_circuit(
            self.source, self.gates, self.initial_mapping, self.num_logical
        )

    def materialize(self) -> RoutingCandidate:
        """The equivalent :class:`RoutingCandidate`."""
        return RoutingCandidate(
            circuit=self.build_circuit(),
            initial_mapping=self.initial_mapping,
            final_mapping=self.final_mapping,
            num_swaps=self.num_swaps,
            depth=self.depth,
            estimated_two_qubit_gates=self.estimated_two_qubit_gates,
            label=self.label,
            is_default=self.is_default,
        )


# =============================================================================
# Circuit DAG builder
# =============================================================================


def _build_routing_dag(
    gates: list[_MappingGate],
    num_logical: int,
    circuit: QuantumCircuit | None = None,
    *,
    commutation: bool = False,
    merge_pairs: bool = False,
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

    With ``merge_pairs`` (ignored with ``commutation``), a two-qubit gate
    whose qubits' last node is the same two-qubit node on the same pair is
    merged into that node (listed in ``extra``), so every node is a whole
    two-qubit block — as when routing consolidated ``unitary`` blocks — and
    the extended set looks that many blocks ahead.

    Args:
        gates: Logical gate list from :func:`_extract_gates`.
        num_logical: Number of logical qubits.
        circuit: Source circuit of ``gates`` (required for ``commutation``).
        commutation: Build the commutation-aware DAG.
        merge_pairs: Merge runs of two-qubit gates on one pair into one node.

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
    merge = merge_pairs and not commutation

    def gate_of(g: _MappingGate) -> Gate | None:
        if g.gate is not None:
            return g.gate
        if circuit is None or g.index < 0:
            return None
        return circuit[g.index].gate

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
            if merge and len(qs) == 2:
                last = members[q1]
                if (
                    len(last) == 1
                    and members[q2] == last
                    and dag.nq3[last[0]] < 0
                    and {dag.nq1[last[0]], dag.nq2[last[0]]} == {q1, q2}
                ):
                    dag.extra[last[0]].append((pre, g))
                    pending[q1] = []
                    pending[q2] = []
                    continue
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
        dag.extra.append([])
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


# =============================================================================
# All-pairs shortest paths
# =============================================================================


def _floyd_warshall(adjacency: list[list[int]], num_qubits: int) -> list[list[int]]:
    """Compute all-pairs shortest path distances on the coupling graph.

    Kept for backward compatibility; :class:`SABRE` uses the equivalent and
    faster :func:`_bfs_distances`.

    Args:
        adjacency: Adjacency list of the physical coupling graph.
        num_qubits: Number of physical qubits.

    Returns:
        ``dist[i][j]`` = shortest path length (number of edges) between
        physical qubits ``i`` and ``j``; ``10**9`` if disconnected.
    """
    dist: list[list[int]] = [
        [0 if i == j else _INF for j in range(num_qubits)] for i in range(num_qubits)
    ]

    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            dist[i][j] = 1
            dist[j][i] = 1

    for k in range(num_qubits):
        dk = dist[k]
        for i in range(num_qubits):
            dik = dist[i][k]
            if dik == _INF:
                continue
            di = dist[i]
            for j in range(num_qubits):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd

    return dist


def _bfs_distances(adjacency: list[list[int]], num_qubits: int) -> list[list[int]]:
    """All-pairs shortest path lengths via one BFS per source qubit.

    Args:
        adjacency: Adjacency list of the physical coupling graph.
        num_qubits: Number of physical qubits.

    Returns:
        Distance matrix with ``10**9`` for disconnected pairs.
    """
    dist: list[list[int]] = []
    for source in range(num_qubits):
        row = [_INF] * num_qubits
        row[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            nd = row[node] + 1
            for neighbor in adjacency[node]:
                if row[neighbor] == _INF:
                    row[neighbor] = nd
                    queue.append(neighbor)
        dist.append(row)
    return dist


# =============================================================================
# SABRE algorithm
# =============================================================================


class SABRE:
    """SABRE qubit mapping and routing pass (LightSABRE-style).

    Given a logical circuit and a physical coupling graph, SABRE produces:

    1. An **initial mapping** ``pi0`` from logical to physical qubits.
    2. A **routed circuit** with SWAP gates inserted so that every
       two-qubit gate acts on adjacent physical qubits.

    The initial mapping is chosen among several layout trials, each refined
    by alternating forward and backward routing passes.  Routing uses the
    lookahead heuristic with a bounded extended set, reference decay
    semantics, seeded random tie-breaking and a release valve that rules
    out livelock.  See the module docstring for details.

    Args:
        target: Compilation target describing the coupling graph.
        lookahead_weight: Weight ``W`` of the extended-set term.
        extended_set_size: Maximum number of gates in the extended set.
        decay_delta: Decay increment applied to the two physical qubits
            of each chosen SWAP.
        decay_reset_interval: Reset all decay values to 1 after this many
            SWAPs without progress.
        release_valve_threshold: Number of SWAPs without any gate becoming
            executable after which the release valve force-routes the
            closest front-layer gate.  ``None`` means ``10 * num_qubits``
            of the target.
        permutation_synthesis: With ``elide_permutations``, also replace
            every input block ``U`` on a qubit pair for which ``SWAP · U``
            needs fewer CZ gates (e.g. a DCNOT-class block: two CZ → one)
            by ``SWAP · U`` and absorb the exchange into the qubit
            labelling.  Skipped unless the target's native gates include
            ``{cz, h, rz, x}``.
        commutation_aware: Route on a commutation-aware DAG: multi-qubit
            gates that commute (diagonal in the same basis on every shared
            qubit, e.g. ``cp``/``rzz``/``cz`` gates, or ``cx`` gates sharing
            a control or a target) may be executed in any order, so the
            router can pick whichever is closest.  Most useful when the
            input keeps such gates undecomposed (``cx``, ``cp``, ``rzz``).
            Ignored for circuits with routed Toffolis.
        swap_fusion_bonus: Relative score discount (``0 ≤ bonus < 1``) for a
            SWAP on a pair whose last operation on both qubits is the same
            two-qubit gate: block resynthesis merges the two (e.g. ``CP`` +
            ``SWAP`` → 3 CZ instead of 5), so such SWAPs are nearly free.
            ``0`` (the default) is plain SABRE.  Applies to circuits
            without routed Toffolis.  With ``mirror_gates`` the discount
            is scaled by the exact saving of the merged block,
            ``bonus * (3 - extra) / 3`` (``extra`` = mirror cost below).
        mirror_gates: Mirror-gate-aware routing (MIRAGE, McKinney et al.
            2023): a SWAP right after the last two-qubit gate of a block
            ``U`` on the same pair is merged by block resynthesis into
            ``U · SWAP``, which for a generic SU(4) block needs no more CZ
            than ``U`` itself.  Every block gets its exact *mirror cost*
            ``cz(U · SWAP) - cz(U)`` from its Weyl coordinates; when the
            router executes the last gate of a block it also executes its
            mirror if exchanging the two logical qubits lowers the
            lookahead score (front layer + extended set, no decay) by more
            than ``mirror_threshold + mirror_cz_penalty * cost``.  Routing
            SWAPs chosen on such a pair are counted at the mirror cost.
            Ignored on commutation-aware DAGs.
        mirror_threshold: Score gain a free (cost 0) mirror must exceed.
        mirror_cz_penalty: Score gain required per extra CZ of a mirror.
        block_nodes: Route on whole two-qubit blocks: consecutive two-qubit
            gates on one qubit pair form a single routing node (as Qiskit
            routes consolidated ``unitary`` blocks), so the extended set
            looks ahead ``extended_set_size`` blocks instead of gates.
            Ignored on commutation-aware DAGs.
        dilation_layouts: Number of low-dilation layouts added to the
            layout search (each routed unrefined and refined, right after
            the trivial layout): embeddings of the interaction graph that
            minimize ``Σ w · (d − 1)`` over its pairs (``w`` routing nodes
            on the pair, ``d`` their coupling distance), found by greedy
            growth with Warnsdorff's rule and simulated annealing (see
            :func:`quspect.transpiler._embedding.dilation_layouts`).  They
            fix SABRE's weak spot of large sparse interaction graphs (CX
            chains, nearest-neighbour Trotter chains, adders) that have no
            exact embedding.  Their routing seeds do not shift the other
            trials.  ``0`` (the default) disables them; skipped on
            disconnected coupling graphs.

    Raises:
        ValueError: If a numeric parameter is out of range.

    Example:
        >>> from quspect import QuantumCircuit
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> circuit = QuantumCircuit().cx(0, 2)
        >>> routed, pi0, pi_final = SABRE(target).solve(circuit, seed=7)
        >>> sum(1 for i in routed if i.is_gate and i.gate.name == "swap")
        0
    """

    def __init__(
        self,
        target: CompilationTarget,
        *,
        lookahead_weight: float = 0.5,
        extended_set_size: int = 20,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        release_valve_threshold: int | None = None,
        permutation_synthesis: bool = False,
        commutation_aware: bool = False,
        swap_fusion_bonus: float = 0.0,
        mirror_gates: bool = False,
        mirror_threshold: float = 0.0,
        mirror_cz_penalty: float = 0.5,
        block_nodes: bool = False,
        dilation_layouts: int = 0,
    ) -> None:
        if dilation_layouts < 0:
            raise ValueError("dilation_layouts must be non-negative.")
        if extended_set_size < 0:
            raise ValueError("extended_set_size must be non-negative.")
        if decay_reset_interval < 1:
            raise ValueError("decay_reset_interval must be at least 1.")
        if release_valve_threshold is not None and release_valve_threshold < 1:
            raise ValueError("release_valve_threshold must be at least 1.")
        if not 0.0 <= swap_fusion_bonus < 1.0:
            raise ValueError("swap_fusion_bonus must be in [0, 1).")
        if mirror_threshold < 0.0 or mirror_cz_penalty < 0.0:
            raise ValueError("mirror_threshold and mirror_cz_penalty must be >= 0.")

        self._num_physical = target.num_qubits
        self._G = target.adjacency
        # 置换消去后残余单比特门的综合门集：目标原生门，未声明时用 u3。
        self._residual_basis = frozenset(target.native_gates) or frozenset({"u3"})
        self._adj_sets = [set(neighbors) for neighbors in self._G]
        self._D = _bfs_distances(self._G, self._num_physical)
        self._edges_at = incident_edges(self._G)
        # 原生路由核心的耦合图副本 (core 模块, 对象)，首次原生路由时创建。
        self._native_graph: Any = None
        self._components = _connected_components(self._G, self._num_physical)
        self._component_of = [0] * self._num_physical
        for idx, component in enumerate(self._components):
            for physical in component:
                self._component_of[physical] = idx

        self.lookahead_weight = lookahead_weight
        self.extended_set_size = extended_set_size
        self.decay_delta = decay_delta
        self.decay_reset_interval = decay_reset_interval
        self.release_valve_threshold = release_valve_threshold
        self.permutation_synthesis = permutation_synthesis
        self.commutation_aware = commutation_aware
        self.swap_fusion_bonus = swap_fusion_bonus
        self.mirror_gates = mirror_gates
        self.mirror_threshold = mirror_threshold
        self.mirror_cz_penalty = mirror_cz_penalty
        self.block_nodes = block_nodes
        self.dilation_layouts = dilation_layouts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        circuit: QuantumCircuit,
        *,
        iter_num: int = 3,
        seed: int | None = None,
        initial_mapping: Sequence[int] | None = None,
        layout_trials: int = DEFAULT_LAYOUT_TRIALS,
        exact_layout: bool = True,
        vf2_max_states: int = DEFAULT_VF2_MAX_STATES,
        vf2_time_limit: float | None = None,
        elide_permutations: bool = True,
    ) -> tuple[QuantumCircuit, list[int], list[int]]:
        """Run SABRE layout and routing on a logical circuit.

        Without ``initial_mapping``, an exact layout is tried first (when
        ``exact_layout``): if the trivial layout or a VF2++ subgraph
        monomorphism of the interaction graph into the coupling graph
        (:func:`~quspect.transpiler._vf2.find_subgraph_monomorphism`)
        places every interacting pair on a coupling edge, the circuit is
        routed from it without any SWAP and returned.  Otherwise the
        trivial layout (unrefined and refined) and ``layout_trials`` seeded
        random layouts are each refined by ``iter_num`` forward/backward
        passes and routed; the candidate with the fewest SWAPs is returned
        (ties: smaller two-qubit depth, then lower trial index).

        With ``elide_permutations`` (the default, like Qiskit's
        ``ElidePermutations``), SWAPs are absorbed into the returned
        ``final_mapping`` instead of being executed:

        - every two-qubit block of the input equal to ``SWAP · (A ⊗ B)``
          (an explicit ``swap`` gate or a decomposed one, e.g. three CZs
          with Hadamards) is removed before layout and routing; ``A`` and
          ``B`` are re-synthesized in the target's native gates (``u3``
          without native gates) and all later gates, including
          measurements, act on the exchanged qubits;
        - routing SWAPs followed by no gate other than measurements on
          their qubits are dropped and those measurements redirected.

        The returned mappings stay exact: the output circuit applied with
        wire ``w`` on physical ``initial_mapping[w]`` equals the input
        circuit followed by moving logical ``q`` to physical
        ``final_mapping[q]`` (up to global phase).

        The returned circuit is in *wire space*: wire ``w`` is pinned to
        physical qubit ``initial_mapping[w]`` for the whole circuit and the
        emitted SWAPs move state between wires.  ``initial_mapping`` may be
        longer than the circuit's qubit count when routing passes through
        otherwise unused physical qubits (workspace wires).

        Args:
            circuit: The logical quantum circuit to map.  Gates must act on
                at most two qubits, except ``ccx``: a Toffoli is routed as a
                three-qubit unit (executable once its qubits are connected
                on the coupling graph) and then synthesized for the
                realised placement — six CX on a coupling triangle, eight
                nearest-neighbour CX on a path (see
                :mod:`quspect.transpiler._toffoli`).
            iter_num: Number of forward+backward refinement iterations per
                layout trial (default 3, as in the original paper).
            seed: Optional random seed; the same seed yields an identical
                result.  With ``initial_mapping`` and no seed, routing is
                deterministic (seed 0 is used for tie-breaking).
            initial_mapping: Optional fixed starting mapping where
                ``initial_mapping[logical] = physical``.  When supplied,
                no layout search is performed and the circuit is routed
                from this assignment.
            layout_trials: Number of random layout trials (at least 1).
                Ignored when ``initial_mapping`` is given.
            exact_layout: Try a SWAP-free exact layout (trivial, then
                VF2++) before the SABRE layout trials.  Ignored when
                ``initial_mapping`` is given.
            vf2_max_states: State budget of the VF2++ search (at least 1);
                the search gives up deterministically when it is spent.
            vf2_time_limit: Optional wall-clock limit in seconds for the
                VF2++ search.  ``None`` keeps results machine independent.
            elide_permutations: Absorb SWAP-equivalent input blocks and
                trailing routing SWAPs into ``final_mapping`` (see above).

        Returns:
            A tuple ``(physical_circuit, initial_mapping, final_mapping)``:

            - ``physical_circuit``: routed circuit with SWAPs inserted
              (in wire space, see above).
            - ``initial_mapping``: ``pi0[wire] = physical``.
            - ``final_mapping``: ``pi_final[wire] = physical`` where the
              state of ``wire`` ends; for a logical wire ``q`` this is
              where logical qubit ``q`` of the *input* circuit ends,
              including elided permutations.

        Raises:
            ValueError: If the circuit needs more qubits than the device
                has, if a gate other than ``ccx`` acts on more than two
                qubits, if
                ``initial_mapping`` or ``layout_trials`` is invalid, or if
                interacting logical qubits cannot be placed in a single
                connected component of the coupling graph (SWAPs never
                move a qubit between components, so such circuits are
                unroutable), or if ``vf2_max_states`` is smaller than 1.
            TypeError: If ``initial_mapping`` is not a sequence of ints or
                ``layout_trials`` is not an int.

        Example:
            >>> from quspect import QuantumCircuit
            >>> target = CompilationTarget.from_coupling_map(
            ...     4, [(0, 1), (1, 2), (2, 3)]
            ... )
            >>> ghz = QuantumCircuit().h(0).cx(0, 1).cx(1, 2).cx(2, 3)
            >>> routed, pi0, _ = SABRE(target).solve(ghz, seed=1, layout_trials=4)
            >>> sum(1 for i in routed if i.is_gate and i.gate.name == "swap")
            0
            >>> star = QuantumCircuit().cx(0, 1).cx(0, 2)  # needs the middle qubit
            >>> _, pi0, _ = SABRE(target).solve(star, seed=1)
            >>> pi0[0] in (1, 2)
            True
            >>> rev = QuantumCircuit().h(0).swap(0, 1)  # elided into the layout
            >>> routed, pi0, pi_final = SABRE(target).solve(rev, seed=1)
            >>> [i.gate.name for i in routed], pi_final[:2] == [pi0[1], pi0[0]]
            (['h'], True)
        """
        plan = self._solve_plan(
            circuit,
            iter_num=iter_num,
            seed=seed,
            initial_mapping=initial_mapping,
            layout_trials=layout_trials,
            exact_layout=exact_layout,
            vf2_max_states=vf2_max_states,
            vf2_time_limit=vf2_time_limit,
            elide_permutations=elide_permutations,
        )
        return plan.build_circuit(), plan.initial_mapping, plan.final_mapping

    def _solve_plan(
        self,
        circuit: QuantumCircuit,
        *,
        iter_num: int = 3,
        seed: int | None = None,
        initial_mapping: Sequence[int] | None = None,
        layout_trials: int = DEFAULT_LAYOUT_TRIALS,
        exact_layout: bool = True,
        vf2_max_states: int = DEFAULT_VF2_MAX_STATES,
        vf2_time_limit: float | None = None,
        elide_permutations: bool = True,
    ) -> _CandidatePlan:
        """:meth:`solve` as a candidate plan that keeps its routing (same result)."""
        if not isinstance(layout_trials, int) or isinstance(layout_trials, bool):
            raise TypeError("layout_trials must be an integer.")
        if layout_trials < 1:
            raise ValueError("layout_trials must be at least 1.")
        if vf2_max_states < 1:
            raise ValueError("vf2_max_states must be at least 1.")

        prep = self._prepare(circuit, elide_permutations)
        dag = prep.dag
        num_logical = prep.num_logical
        master_rng = random.Random(seed)

        if initial_mapping is not None:
            pi = self._validate_initial_mapping(initial_mapping, num_logical)
            self._check_mapping_connectivity(pi, prep.groups)
            layout = self._complete_layout(pi)
            # Routing from a fixed layout stays deterministic without a seed.
            route_rng = random.Random(0 if seed is None else seed)
            best = self._route(
                layout,
                dag,
                forward=True,
                rng=random.Random(route_rng.getrandbits(64)),
                record=True,
            )
        else:
            exact = (
                self._exact_layout(
                    prep.gates, dag, num_logical, vf2_max_states, vf2_time_limit
                )
                if exact_layout
                else None
            )
            if exact is not None:
                layout, best = exact
            else:
                layout, best = self._search_layout(
                    dag,
                    prep.groups,
                    num_logical,
                    iter_num,
                    layout_trials,
                    master_rng,
                )

        physical_gates, initial, final = self._finish(prep, best.events, layout)
        return _CandidatePlan(
            source=circuit,
            num_logical=num_logical,
            gates=physical_gates,
            initial_mapping=initial,
            final_mapping=final,
            num_swaps=best.num_swaps,
            depth=best.depth,
            estimated_two_qubit_gates=estimate_two_qubit_cost(
                physical_gates, self._num_physical
            ),
            label="default",
            is_default=True,
            prep=prep,
            route_layout=layout,
            route_events=best.events,
        )

    def route_candidates(
        self,
        circuit: QuantumCircuit,
        *,
        iter_num: int = 3,
        seed: int | None = None,
        initial_mapping: Sequence[int] | None = None,
        layout_trials: int = DEFAULT_LAYOUT_TRIALS,
        swap_trials: int = 1,
        exact_layout: bool = True,
        vf2_max_states: int = DEFAULT_VF2_MAX_STATES,
        vf2_time_limit: float | None = None,
        elide_permutations: bool = True,
        swap_layouts: int | None = None,
        reference_trials: int | None = None,
        top_k: int | None = None,
        work_limit: int | None = None,
        time_limit: float | None = None,
        swap_keep: int | None = None,
    ) -> list[RoutingCandidate]:
        """Route a circuit several ways and return the ranked candidates.

        Runs the same layout search as :meth:`solve` (same seeds, same
        refinement), but instead of keeping only the trial with the fewest
        SWAPs it keeps every routed trial.  With ``swap_trials > 1`` the
        refined layouts (all, or the ``swap_layouts`` best ranked) are then
        re-routed ``swap_trials - 1`` more times with different
        tie-breaking seeds (Qiskit's ``swap_trials``).  Duplicate routings
        are dropped.

        Candidates are ranked by ``(estimated_two_qubit_gates, num_swaps,
        depth, generation order)``: the estimate accounts for SWAPs that
        merge with a neighbouring gate on the same pair (``SWAP · CZ``
        needs 2 CZ instead of 4), which a raw SWAP count ignores.  The
        candidate :meth:`solve` would return (with ``reference_trials``
        layout trials) is flagged ``is_default`` and is always included,
        even when it falls outside ``top_k``.  When an exact (SWAP-free)
        layout exists, it is the only candidate.

        Args:
            circuit: Logical circuit (gates on at most two qubits, plus
                ``ccx`` routed as a three-qubit unit as in :meth:`solve`;
                every candidate has its Toffolis synthesized for their
                placement before it is scored).
            iter_num: Forward+backward refinement iterations per trial.
            seed: Random seed; the same seed yields identical candidates
                (unless ``time_limit`` cuts the search short).
            initial_mapping: Optional fixed ``initial_mapping[logical] =
                physical``.  No layout search is done; the circuit is
                routed from it ``swap_trials`` times (the first routing
                equals :meth:`solve` with the same arguments).
            layout_trials: Number of random layout trials (at least 1).
            swap_trials: Routing passes per refined layout (at least 1);
                the first one is the routing :meth:`solve` uses.
            exact_layout: Try a SWAP-free exact layout first.
            vf2_max_states: State budget of the VF2++ search.
            vf2_time_limit: Optional wall-clock limit of the VF2++ search.
            elide_permutations: Absorb permutations into the final mapping
                (see :meth:`solve`).
            swap_layouts: Re-route only this many refined layouts (the
                best ranked after their first routing) ``swap_trials - 1``
                more times.  ``None`` re-routes all of them.
            reference_trials: The default candidate (``is_default``) is
                the result of :meth:`solve` with ``layout_trials =
                reference_trials`` (default: ``layout_trials``).  Because
                trial seeds are drawn in order, those trials are a prefix
                of this search; they are always run, whatever the budget.
            top_k: Return at most this many best-ranked candidates (plus
                the default one).  ``None`` returns all of them.
            work_limit: Optional deterministic budget on the total routing
                work: DAG nodes plus inserted SWAPs summed over all routing
                passes (about 1 µs each with the native routing core,
                10–20 µs in CPython), plus, for every scored candidate,
                its input gates and SWAPs and ``150`` per routed Toffoli
                (replay, Toffoli synthesis and cost estimate).  Three
                quarters of it may go to the layout trials, the rest to the
                extra routings; once spent, no further trial starts (at
                least one trial is always routed).
            time_limit: Optional wall-clock budget in seconds for the
                trials; once spent, no further trial is started (at least
                one is always routed).  Results then depend on machine
                speed, so leave it ``None`` for reproducibility.
            swap_keep: Of the ``swap_trials - 1`` extra routings of each
                layout, keep (replay and score) only the ``swap_keep`` with
                the fewest SWAPs (then the smallest depth, then the
                earliest), as Qiskit selects among its swap trials.
                Routing is far cheaper than replaying and scoring a
                candidate, so many swap trials stay affordable.  ``None``
                keeps every routing.

        Returns:
            Candidates sorted best first.

        Raises:
            ValueError: For the same reasons as :meth:`solve`, or if
                ``swap_trials``, ``swap_layouts``, ``top_k``,
                ``work_limit`` or ``swap_keep`` is smaller than 1, or
                ``reference_trials`` is outside ``[1, layout_trials]``.
            TypeError: If ``layout_trials`` is not an int.

        Example:
            >>> from quspect import QuantumCircuit
            >>> target = CompilationTarget.from_coupling_map(
            ...     4, [(0, 1), (1, 2), (2, 3)]
            ... )
            >>> qc = QuantumCircuit().cx(0, 3).cx(1, 2).cx(0, 2).cx(1, 3)
            >>> cands = SABRE(target).route_candidates(
            ...     qc, seed=3, layout_trials=3, swap_trials=2
            ... )
            >>> best = cands[0]
            >>> best.estimated_two_qubit_gates <= cands[-1].estimated_two_qubit_gates
            True
        """
        plans = self._candidate_plans(
            circuit,
            iter_num=iter_num,
            seed=seed,
            initial_mapping=initial_mapping,
            layout_trials=layout_trials,
            swap_trials=swap_trials,
            exact_layout=exact_layout,
            vf2_max_states=vf2_max_states,
            vf2_time_limit=vf2_time_limit,
            elide_permutations=elide_permutations,
            swap_layouts=swap_layouts,
            reference_trials=reference_trials,
            top_k=top_k,
            work_limit=work_limit,
            time_limit=time_limit,
            swap_keep=swap_keep,
        )
        return [plan.materialize() for plan in plans]

    def _candidate_plans(
        self,
        circuit: QuantumCircuit,
        *,
        iter_num: int = 3,
        seed: int | None = None,
        initial_mapping: Sequence[int] | None = None,
        layout_trials: int = DEFAULT_LAYOUT_TRIALS,
        swap_trials: int = 1,
        exact_layout: bool = True,
        vf2_max_states: int = DEFAULT_VF2_MAX_STATES,
        vf2_time_limit: float | None = None,
        elide_permutations: bool = True,
        swap_layouts: int | None = None,
        reference_trials: int | None = None,
        top_k: int | None = None,
        work_limit: int | None = None,
        time_limit: float | None = None,
        swap_keep: int | None = None,
    ) -> list[_CandidatePlan]:
        """:meth:`route_candidates` without building the output circuits.

        Returns the same candidates, in the same order, as plans whose
        circuit is only built on :meth:`_CandidatePlan.materialize` /
        :meth:`_CandidatePlan.build_circuit` (level 3 fully evaluates only a
        few of them).
        """
        import time

        if not isinstance(layout_trials, int) or isinstance(layout_trials, bool):
            raise TypeError("layout_trials must be an integer.")
        if layout_trials < 1:
            raise ValueError("layout_trials must be at least 1.")
        if swap_trials < 1:
            raise ValueError("swap_trials must be at least 1.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if reference_trials is None:
            reference_trials = layout_trials
        if not 1 <= reference_trials <= layout_trials:
            raise ValueError("reference_trials must be in [1, layout_trials].")
        if swap_layouts is not None and swap_layouts < 1:
            raise ValueError("swap_layouts must be at least 1.")
        if work_limit is not None and work_limit < 1:
            raise ValueError("work_limit must be at least 1.")
        if swap_keep is not None and swap_keep < 1:
            raise ValueError("swap_keep must be at least 1.")
        if vf2_max_states < 1:
            raise ValueError("vf2_max_states must be at least 1.")

        prep = self._prepare(circuit, elide_permutations)
        dag = prep.dag
        n_phys = self._num_physical

        if exact_layout and initial_mapping is None:
            exact = self._exact_layout(
                prep.gates, dag, prep.num_logical, vf2_max_states, vf2_time_limit
            )
            if exact is not None:
                layout, result = exact
                gates, initial, final = self._finish(prep, result.events, layout)
                return [
                    _CandidatePlan(
                        source=circuit,
                        num_logical=prep.num_logical,
                        gates=gates,
                        initial_mapping=initial,
                        final_mapping=final,
                        num_swaps=0,
                        depth=result.depth,
                        estimated_two_qubit_gates=estimate_two_qubit_cost(
                            gates, n_phys
                        ),
                        label="exact",
                        is_default=True,
                        prep=prep,
                        route_layout=layout,
                        route_events=result.events,
                    )
                ]

        deadline = None if time_limit is None else time.perf_counter() + time_limit
        if initial_mapping is not None:
            pi = self._validate_initial_mapping(initial_mapping, prep.num_logical)
            self._check_mapping_connectivity(pi, prep.groups)
            starts = [(self._complete_layout(pi), False, 0)]
            # Same seed derivation as solve() with a fixed layout.
            route_rng = random.Random(0 if seed is None else seed)
            base_seeds = [route_rng.getrandbits(64)]
        else:
            starts, trial_seeds = self._layout_starts(
                prep.groups,
                prep.num_logical,
                layout_trials,
                random.Random(seed),
                dag,
            )
            # The first starts replay solve(layout_trials=reference_trials),
            # whose routing seeds cycle over its own (shorter) seed list.
            n_ref = len(starts) - (layout_trials - reference_trials)
            base_seeds = [
                trial_seeds[
                    index % (reference_trials if t < n_ref else len(trial_seeds))
                ]
                ^ index
                for t, (_, _, index) in enumerate(starts)
            ]
        # found[i] = (rank key, label, routing); finished[i] = replayed output
        found: list[tuple[tuple[int, int, int, int], str, _RouteResult]] = []
        finished: list[tuple[list[_MappingGate], list[int], list[int]]] = []
        embedded: list[bool] = []
        layouts: list[list[int]] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        default_key: tuple[int, int, int] | None = None
        default_pos = -1

        work = 0
        nodes = len(dag.nodes)
        score_work = len(prep.gates) + _SCORE_WORK_PER_CCX * sum(
            1 for g in prep.gates if g.q3 >= 0
        )

        def add(result: _RouteResult, layout: list[int], trial: int, s_idx: int) -> int:
            nonlocal work
            sig = (tuple(layout), tuple(result.events))
            if sig in seen:
                return -1
            seen.add(sig)
            label = f"trial{trial}/swap{s_idx}"
            embedded.append(starts[trial][2] >= _DILATION_SEED_INDEX)
            # 回放 + 估计（含 Toffoli 按落点综合）比一次原生路由贵得多，也计入预算
            work += score_work + result.num_swaps
            gates, initial, final = self._finish(prep, result.events, layout)
            est = estimate_two_qubit_cost(gates, n_phys)
            pos = len(found)
            found.append(((est, result.swap_cost, result.depth, pos), label, result))
            finished.append((gates, initial, final))
            layouts.append(layout)
            return pos

        phase1_limit = None if work_limit is None else work_limit * 3 // 4

        def out_of_budget(limit: int | None) -> bool:
            if limit is not None and work >= limit:
                return True
            return deadline is not None and time.perf_counter() > deadline

        def route(
            layout: list[int], fwd: bool, rng: random.Random, record: bool = True
        ) -> _RouteResult:
            nonlocal work
            res = self._route(layout, dag, forward=fwd, rng=rng, record=record)
            work += nodes + res.num_swaps
            return res

        # Phase 1: every layout trial, refined and routed once (as solve()).
        # The first n_reference starts are exactly those of
        # solve(layout_trials=reference_trials); they ignore the budget.
        n_reference = len(starts) - (layout_trials - reference_trials)
        if initial_mapping is not None:
            n_reference = 1
        routed_layouts: list[tuple[int, int, list[int], int]] = []
        for trial, (layout, refine, _) in enumerate(starts):
            if trial >= n_reference and out_of_budget(phase1_limit):
                logger.debug("SABRE candidates: budget spent after %d trials", trial)
                break
            base_seed = base_seeds[trial]
            rng = random.Random(base_seed)
            if refine:
                for _ in range(iter_num):
                    forward = route(layout, True, rng, record=False)
                    layout = route(forward.final_pi, False, rng, record=False).final_pi
            result = route(layout, True, rng)
            pos = add(result, layout, trial, 0)
            key = (result.swap_cost, result.depth, trial)
            if (
                trial < n_reference
                and pos >= 0
                and (default_key is None or key < default_key)
            ):
                default_key, default_pos = key, pos
            if result.num_swaps == 0:
                routed_layouts = []
                break
            if pos >= 0:
                routed_layouts.append((found[pos][0][0], trial, layout, base_seed))

        # Phase 2: re-route the most promising layouts with other seeds.
        routed_layouts.sort(key=lambda item: (item[0], item[1]))
        if swap_layouts is not None:
            routed_layouts = routed_layouts[:swap_layouts]
        # With swap_keep, each layout keeps its swap_keep best extra routings
        # (fewest SWAPs, then depth, then trial order) and only those are
        # replayed and scored, in trial order, after all routings ran.
        kept: list[list[tuple[tuple[int, int, int], _RouteResult]]] = [
            [] for _ in routed_layouts
        ]
        for s_idx in range(1, swap_trials):
            for j, (_, trial, layout, base_seed) in enumerate(routed_layouts):
                if out_of_budget(work_limit):
                    break
                swap_rng = random.Random(base_seed * 1_000_003 + s_idx)
                result = route(layout, True, swap_rng)
                if swap_keep is None:
                    add(result, layout, trial, s_idx)
                    continue
                best = kept[j]
                best.append(((result.swap_cost, result.depth, s_idx), result))
                best.sort(key=lambda item: item[0])
                del best[swap_keep:]
        for j, (_, trial, layout, _) in enumerate(routed_layouts):
            for (_, _, s_idx), result in sorted(kept[j], key=lambda item: item[0][2]):
                add(result, layout, trial, s_idx)

        order = sorted(range(len(found)), key=lambda i: found[i][0])
        chosen = order if top_k is None else order[:top_k]
        if default_pos >= 0 and default_pos not in chosen:
            chosen = [*chosen, default_pos]
        out: list[_CandidatePlan] = []
        for i in chosen:
            (est, _, _, _), label, result = found[i]
            gates, initial, final = finished[i]
            out.append(
                _CandidatePlan(
                    source=circuit,
                    num_logical=prep.num_logical,
                    gates=gates,
                    initial_mapping=initial,
                    final_mapping=final,
                    num_swaps=result.num_swaps,
                    depth=result.depth,
                    estimated_two_qubit_gates=est,
                    label=label,
                    is_default=i == default_pos,
                    embedded=embedded[i],
                    prep=prep,
                    route_layout=layouts[i],
                    route_events=result.events,
                )
            )
        logger.debug(
            "SABRE candidates: %d routed, best estimate %d CZ (default %d)",
            len(found),
            out[0].estimated_two_qubit_gates,
            found[default_pos][0][0] if default_pos >= 0 else -1,
        )
        return out

    # ------------------------------------------------------------------
    # Internal: shared preparation and output
    # ------------------------------------------------------------------

    def _prepare(self, circuit: QuantumCircuit, elide: bool) -> _Prepared:
        """Extract gates, elide input permutations and build the routing DAG."""
        gates, num_logical = _extract_gates(circuit)
        if num_logical > self._num_physical:
            raise ValueError(
                f"Circuit uses {num_logical} logical qubits but the target "
                f"device has only {self._num_physical} physical qubits."
            )
        sigma = list(range(num_logical))
        if elide:
            gates, sigma = _elide_swap_blocks(
                gates,
                circuit,
                num_logical,
                self._residual_basis,
                resynthesize=self.permutation_synthesis,
            )
        # Toffoli 路由器假定每个比特至多一个前沿门：含 ccx 时不启用对易
        commutation = self.commutation_aware and not any(g.q3 >= 0 for g in gates)
        dag = _build_routing_dag(
            gates,
            num_logical,
            circuit,
            commutation=commutation,
            merge_pairs=self.block_nodes,
        )
        if self.mirror_gates and not commutation:
            _annotate_mirror_costs(dag, gates, circuit)
        return _Prepared(
            circuit=circuit,
            gates=gates,
            num_logical=num_logical,
            sigma=sigma,
            dag=dag,
            groups=_interaction_groups(gates, num_logical),
            elide=elide,
        )

    def _finish(
        self, prep: _Prepared, events: list[int], layout: list[int]
    ) -> tuple[list[_MappingGate], list[int], list[int]]:
        """Replay routing events (Toffolis synthesized) into gates and wire mappings."""
        num_logical = prep.num_logical
        physical_gates, initial, final = _replay(
            events, layout, prep.dag, num_logical, self._num_physical
        )
        physical_gates = expand_toffolis(physical_gates, self._adj_sets, prep.circuit)
        if prep.elide:
            physical_gates, initial, final = _elide_trailing_swaps(
                physical_gates, initial, num_logical
            )
            sigma = prep.sigma
            final = [
                final[sigma[w]] if w < num_logical else final[w]
                for w in range(len(final))
            ]
        return physical_gates, initial, final

    # ------------------------------------------------------------------
    # Internal: layout search
    # ------------------------------------------------------------------

    def _search_layout(
        self,
        dag: _RoutingDag,
        logical_groups: list[list[int]],
        num_logical: int,
        iter_num: int,
        layout_trials: int,
        master_rng: random.Random,
    ) -> tuple[list[int], _RouteResult]:
        """Run all layout trials and return the best ``(layout, routing)``."""
        starts, trial_seeds = self._layout_starts(
            logical_groups, num_logical, layout_trials, master_rng, dag
        )

        best_key: tuple[int, int, int] | None = None
        best_layout: list[int] = []
        best_result: _RouteResult | None = None
        for trial, (layout, refine, index) in enumerate(starts):
            rng = random.Random(trial_seeds[index % len(trial_seeds)] ^ index)
            if refine:
                for _ in range(iter_num):
                    forward = self._route(layout, dag, forward=True, rng=rng)
                    backward = self._route(
                        forward.final_pi, dag, forward=False, rng=rng
                    )
                    layout = backward.final_pi
            result = self._route(layout, dag, forward=True, rng=rng, record=True)
            key = (result.swap_cost, result.depth, trial)
            logger.debug(
                "SABRE layout trial %d: %d swaps, 2q depth %d",
                trial,
                result.num_swaps,
                result.depth,
            )
            if best_key is None or key < best_key:
                best_key, best_layout, best_result = key, layout, result
            if result.num_swaps == 0:
                break
        assert best_result is not None
        return best_layout, best_result

    def _layout_starts(
        self,
        logical_groups: list[list[int]],
        num_logical: int,
        layout_trials: int,
        master_rng: random.Random,
        dag: _RoutingDag | None = None,
    ) -> tuple[list[tuple[list[int], bool, int]], list[int]]:
        """Starting layouts ``(layout, refine, seed index)`` and the trial seeds.

        The routing seed of a start is derived from ``trial_seeds[index %
        len(trial_seeds)] ^ index``.  The trivial layout (unrefined and
        refined) and the random trials have the indices of their position
        without dilation layouts; the dilation layouts (see
        :mod:`quspect.transpiler._embedding`), each tried unrefined and
        refined, come right after the trivial layout with indices beyond
        the others, so adding them leaves every other trial unchanged.
        """
        starts: list[tuple[list[int], bool, int]] = []
        trivial = list(range(num_logical))
        if self._mapping_is_connected(trivial, logical_groups):
            full = self._complete_layout(trivial)
            starts.append((full, False, len(starts)))
            starts.append((list(full), True, len(starts)))
        trial_seeds = [master_rng.getrandbits(64) for _ in range(layout_trials)]
        n_trivial = len(starts)
        index = _DILATION_SEED_INDEX
        for embedded in self._dilation_starts(dag, num_logical, trial_seeds[0]):
            starts.append((embedded, False, index))
            starts.append((list(embedded), True, index + 1))
            index += 2
        for j, trial_seed in enumerate(trial_seeds):
            rng = random.Random(trial_seed)
            if len(self._components) <= 1:
                perm = list(range(self._num_physical))
                rng.shuffle(perm)
            else:
                perm = self._complete_layout(
                    self._component_respecting_layout(logical_groups, num_logical, rng)
                )
            starts.append((perm, True, n_trivial + j))
        return starts, trial_seeds

    def _dilation_starts(
        self, dag: _RoutingDag | None, num_logical: int, seed: int
    ) -> list[list[int]]:
        """Full layouts of small weighted dilation (``dilation_layouts`` many).

        The interaction graph has one unit of weight per routing node on a
        pair (a ``ccx`` node counts on its two control–target pairs).
        Skipped on disconnected coupling graphs.
        """
        if dag is None or self.dilation_layouts == 0 or len(self._components) > 1:
            return []
        weight: dict[tuple[int, int], int] = {}
        for a, b, c in zip(dag.nq1, dag.nq2, dag.nq3, strict=True):
            if a < 0:
                continue
            pairs = ((a, c), (b, c)) if c >= 0 else ((a, b),)
            for x, y in pairs:
                key = (x, y) if x < y else (y, x)
                weight[key] = weight.get(key, 0) + 1
        if not weight:
            return []
        edges = [(a, b, w) for (a, b), w in sorted(weight.items())]
        found = dilation_layouts(
            num_logical,
            edges,
            self._G,
            self._D,
            seed=seed,
            count=self.dilation_layouts,
        )
        out: list[list[int]] = []
        for _, pos in found:
            used = {p for p in pos if p >= 0}
            free = iter(p for p in range(self._num_physical) if p not in used)
            out.append(
                self._complete_layout([p if p >= 0 else next(free) for p in pos])
            )
        return out

    def _exact_layout(
        self,
        gates: list[_MappingGate],
        dag: _RoutingDag,
        num_logical: int,
        max_states: int,
        time_limit: float | None,
    ) -> tuple[list[int], _RouteResult] | None:
        """SWAP-free layout (trivial or VF2++) and its routing, if any.

        Returns ``None`` when the circuit has no two-qubit gate (the
        regular search handles it with the trivial layout) or when no
        embedding was found within the budget.
        """
        edges = {(min(a, b), max(a, b)) for a, b in _interaction_pairs(gates)}
        if not edges:
            return None
        adj = self._adj_sets
        if all(b in adj[a] for a, b in edges):
            pi = list(range(num_logical))
        else:
            found = find_subgraph_monomorphism(
                num_logical,
                sorted(edges),
                self._G,
                max_states=max_states,
                time_limit=time_limit,
            )
            if found is None:
                return None
            used = {p for p in found if p >= 0}
            free = iter(p for p in range(self._num_physical) if p not in used)
            pi = [p if p >= 0 else next(free) for p in found]
        layout = self._complete_layout(pi)
        result = self._route(
            layout, dag, forward=True, rng=random.Random(0), record=True
        )
        if result.num_swaps:  # pragma: no cover - an embedding needs no SWAP
            logger.warning(
                "Exact layout unexpectedly needed %d SWAPs", result.num_swaps
            )
            return None
        logger.debug("SABRE: exact layout found, routing needs no SWAP")
        return layout, result

    def _complete_layout(self, pi: list[int]) -> list[int]:
        """Extend a logical layout to a full permutation of physical qubits.

        Free physical qubits are assigned (in increasing order) to ancilla
        indices ``num_logical, num_logical + 1, ...``.
        """
        used = set(pi)
        return list(pi) + [p for p in range(self._num_physical) if p not in used]

    def _mapping_is_connected(
        self, pi: list[int], logical_groups: list[list[int]]
    ) -> bool:
        return all(
            len({self._component_of[pi[q]] for q in group}) <= 1
            for group in logical_groups
        )

    # ------------------------------------------------------------------
    # Internal: heuristic routing pass
    # ------------------------------------------------------------------

    def _route(
        self,
        layout: list[int],
        dag: _RoutingDag,
        *,
        forward: bool,
        rng: random.Random,
        record: bool = False,
    ) -> _RouteResult:
        """Run one SABRE routing pass (see :func:`~._routing.route_dag`).

        Args:
            layout: Full permutation ``layout[token] = physical`` (logical
                qubits first, then ancillas).
            dag: Routing DAG (two- and three-qubit nodes, plain or
                commutation-aware).
            forward: Traverse the DAG forwards (``True``) or reversed.
            rng: Random source for tie-breaking.
            record: Record the execution/SWAP event sequence.

        Returns:
            Routing statistics, the final full permutation and (if
            ``record``) the events.
        """
        return route_dag(self, layout, dag, forward=forward, rng=rng, record=record)

    # ------------------------------------------------------------------
    # Internal: validation helpers
    # ------------------------------------------------------------------

    def _validate_initial_mapping(
        self,
        initial_mapping: Sequence[int],
        num_logical: int,
    ) -> list[int]:
        """Validate and copy a logical-to-physical starting mapping."""
        if not isinstance(initial_mapping, (list, tuple)):
            raise TypeError("initial_mapping must be a list or tuple of integers.")
        mapping = list(initial_mapping)
        if len(mapping) != num_logical:
            raise ValueError(
                f"initial_mapping length ({len(mapping)}) must equal the circuit "
                f"qubit count ({num_logical})."
            )
        if not all(
            isinstance(physical, int) and not isinstance(physical, bool)
            for physical in mapping
        ):
            raise TypeError("initial_mapping must contain only integer qubits.")
        if len(set(mapping)) != len(mapping):
            raise ValueError("initial_mapping must not contain duplicates.")
        invalid = [
            physical
            for physical in mapping
            if physical < 0 or physical >= self._num_physical
        ]
        if invalid:
            raise ValueError(
                "initial_mapping contains indices outside the target range "
                f"[0, {self._num_physical - 1}]: {invalid}."
            )
        return mapping

    def _check_mapping_connectivity(
        self, pi: list[int], logical_groups: list[list[int]]
    ) -> None:
        """Raise ValueError if an interaction group spans several components.

        Logical qubits that interact (directly or transitively through
        two-qubit gates) must all live in one connected component of the
        coupling graph: SWAPs only move qubits along coupling edges.
        """
        for group in logical_groups:
            components = {self._component_of[pi[q]] for q in group}
            if len(components) > 1:
                placement = {q: pi[q] for q in group}
                raise ValueError(
                    "Unroutable mapping: interacting logical qubits "
                    f"{group} are mapped to physical qubits {placement} in "
                    "different connected components of the (disconnected) "
                    "coupling graph."
                )

    def _component_respecting_layout(
        self,
        logical_groups: list[list[int]],
        num_logical: int,
        rng: random.Random,
    ) -> list[int]:
        """Build a random initial layout on a disconnected coupling graph.

        Every interaction group is placed entirely inside one connected
        component (exact backtracking over the component capacities);
        idle logical qubits fill the remaining physical qubits.

        Raises:
            ValueError: If the interaction groups cannot be packed into the
                connected components.
        """
        groups = sorted(
            (g for g in logical_groups if len(g) > 1), key=len, reverse=True
        )
        capacity = [len(c) for c in self._components]
        assignment: list[int] = [-1] * len(groups)

        def place(i: int) -> bool:
            if i == len(groups):
                return True
            tried: set[int] = set()
            for comp in range(len(capacity)):
                if capacity[comp] < len(groups[i]) or capacity[comp] in tried:
                    continue
                tried.add(capacity[comp])
                capacity[comp] -= len(groups[i])
                assignment[i] = comp
                if place(i + 1):
                    return True
                capacity[comp] += len(groups[i])
            return False

        if not place(0):
            raise ValueError(
                "Unroutable circuit: the interacting logical qubit groups "
                f"{groups} cannot each be placed inside a single connected "
                "component of the disconnected coupling graph (component sizes "
                f"{[len(c) for c in self._components]})."
            )

        free = [list(c) for c in self._components]
        for component in free:
            rng.shuffle(component)
        pi = [-1] * num_logical
        for group, comp in zip(groups, assignment, strict=True):
            for q in group:
                pi[q] = free[comp].pop()
        leftovers = [p for component in free for p in component]
        rng.shuffle(leftovers)
        for q in range(num_logical):
            if pi[q] == -1:
                pi[q] = leftovers.pop()
        return pi


# =============================================================================
# Post-routing cost estimate
# =============================================================================


#: CZ cost of two-qubit gates that need two CZ (others count 1, SWAP 3).
_GATE_CZ_COST = dict.fromkeys(
    ("cp", "cu1", "crz", "crx", "cry", "rzz", "rxx", "ryy", "rzx"), 2
)


def _block_cost(total: int, swaps: int) -> int:
    """CZ count of a resynthesized two-qubit block (upper-bound estimate)."""
    if swaps == 1 and total == 4:
        return 2  # SWAP · CZ is locally equivalent to iSWAP
    return min(3, total)


def estimate_two_qubit_cost(gates: Sequence[_MappingGate], num_physical: int) -> int:
    """Estimate the CZ count of routed gates after block resynthesis.

    Gates are grouped into maximal runs on one physical pair (like
    :class:`~quspect.transpiler.ConsolidateBlocksPass`); a measurement or a
    two-qubit gate on another pair closes the run.  A SWAP costs 3, a
    controlled rotation or ``rzz``-type gate 2, any other two-qubit gate 1.
    A run costs ``min(3, total)``, except a SWAP with a single other gate,
    which costs 2 (``SWAP · CZ`` ~ iSWAP).  This ignores cancellations
    found by later optimization, so it is a ranking heuristic, not an exact
    count.

    :meth:`SABRE.route_candidates` scores the gates *after* routed Toffolis
    were synthesized for their placement, so their CX gates are counted like
    any other two-qubit gate.  A still unexpanded three-qubit ``ccx`` closes
    the runs on its qubits and costs 6 (its cheapest synthesis).

    Args:
        gates: Routed gates on physical qubits.
        num_physical: Number of physical qubits.

    Returns:
        The estimated number of two-qubit gates.
    """
    owner = [-1] * num_physical
    # block: [qubit a, qubit b, total cost, swap count]
    blocks: list[list[int]] = []
    open_blocks: dict[int, None] = {}
    cost = 0

    def close(b: int) -> None:
        nonlocal cost
        blk = blocks[b]
        cost += _block_cost(blk[2], blk[3])
        owner[blk[0]] = owner[blk[1]] = -1
        del open_blocks[b]

    for g in gates:
        if g.q3 >= 0:
            for q in (g.q1, g.q2, g.q3):
                if owner[q] >= 0:
                    close(owner[q])
            cost += 6
        elif g.is_two_qubit:
            a, b = g.q1, g.q2
            is_swap = g.name == "swap"
            weight = 3 if is_swap else _GATE_CZ_COST.get(g.name, 1)
            oa = owner[a]
            if oa >= 0 and oa == owner[b]:
                blk = blocks[oa]
                blk[2] += weight
                blk[3] += is_swap
                continue
            if oa >= 0:
                close(oa)
            if owner[b] >= 0:
                close(owner[b])
            idx = len(blocks)
            blocks.append([a, b, weight, int(is_swap)])
            owner[a] = owner[b] = idx
            open_blocks[idx] = None
        elif g.name == "measure" and owner[g.q1] >= 0:
            close(owner[g.q1])
    for b in list(open_blocks):
        close(b)
    return cost


# =============================================================================
# Event replay
# =============================================================================


def _replay(
    events: list[int],
    layout: list[int],
    dag: _RoutingDag,
    num_logical: int,
    num_physical: int,
) -> tuple[list[_MappingGate], list[int], list[int]]:
    """Turn routing events into physical gates and wire-space mappings.

    Args:
        events: Recorded events (node index or encoded SWAP).
        layout: Full initial permutation ``layout[token] = physical``.
        dag: Routing DAG.
        num_logical: Number of logical qubits.
        num_physical: Number of physical qubits.

    Returns:
        ``(physical_gates, initial_mapping, final_mapping)``.  Physical
        qubits first touched by a SWAP while hosting no logical qubit are
        appended to ``initial_mapping`` as workspace wires (lower physical
        index of the SWAP first).
    """
    pi = list(layout)
    rpi = [0] * num_physical
    for token, physical in enumerate(pi):
        rpi[physical] = token
    initial = list(layout[:num_logical])
    has_wire = set(initial)
    out: list[_MappingGate] = []

    def emit_1q(g: _MappingGate) -> None:
        p = pi[g.q1]
        out.append(_MappingGate(False, p, p, g.index, g.name, g.clbit, g.gate))

    for event in events:
        if event >= 0:
            for g in dag.prefix[event]:
                emit_1q(g)
            node = dag.nodes[event]
            if node.is_two_qubit:
                out.append(
                    _MappingGate(
                        True,
                        pi[node.q1],
                        pi[node.q2],
                        node.index,
                        node.name,
                        gate=node.gate,
                    )
                )
                for pre, g2 in dag.extra[event] if dag.extra else ():
                    for g in pre:
                        emit_1q(g)
                    out.append(
                        _MappingGate(
                            True, pi[g2.q1], pi[g2.q2], g2.index, g2.name, gate=g2.gate
                        )
                    )
            elif node.q3 >= 0:
                out.append(
                    _MappingGate(
                        False,
                        pi[node.q1],
                        pi[node.q2],
                        node.index,
                        node.name,
                        q3=pi[node.q3],
                    )
                )
            else:
                emit_1q(node)
        else:
            x, y = divmod(-1 - event, num_physical)
            for p in (x, y):
                if p not in has_wire:
                    has_wire.add(p)
                    initial.append(p)
            tx, ty = rpi[x], rpi[y]
            pi[tx], pi[ty] = y, x
            rpi[x], rpi[y] = ty, tx
            out.append(_MappingGate(True, x, y, -1, "swap"))
    for g in dag.trailing:
        emit_1q(g)

    initial_rpi = [0] * num_physical
    for token, physical in enumerate(layout):
        initial_rpi[physical] = token
    final = [pi[initial_rpi[p]] for p in initial]
    return out, initial, final


# =============================================================================
# Connectivity helpers
# =============================================================================


def _connected_components(
    adjacency: list[list[int]], num_qubits: int
) -> list[list[int]]:
    """Return the connected components of the coupling graph (sorted lists)."""
    seen = [False] * num_qubits
    components: list[list[int]] = []
    for start in range(num_qubits):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _interaction_pairs(gates: list[_MappingGate]) -> list[tuple[int, int]]:
    """Logical qubit pairs that must interact.

    A two-qubit gate contributes its pair; a three-qubit ``ccx`` contributes
    ``(c1, t)`` and ``(c2, t)``: a placement coupling both pairs makes it
    executable (target in the middle of a path).
    """
    pairs: list[tuple[int, int]] = []
    for g in gates:
        if g.is_two_qubit:
            pairs.append((g.q1, g.q2))
        elif g.q3 >= 0:
            pairs.append((g.q1, g.q3))
            pairs.append((g.q2, g.q3))
    return pairs


def _interaction_groups(gates: list[_MappingGate], num_logical: int) -> list[list[int]]:
    """Group logical qubits connected (transitively) by two-qubit gates.

    Returns:
        Sorted lists of logical qubits, one per connected component of the
        interaction graph (singletons included).
    """
    parent = list(range(num_logical))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in _interaction_pairs(gates):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for q in range(num_logical):
        groups.setdefault(find(q), []).append(q)
    return sorted(groups.values())


# =============================================================================
# Permutation elision
# =============================================================================

_SWAP_4 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
_EYE_2 = np.eye(2, dtype=complex)
_PRODUCT_TOL = 1e-8
#: Gates needed by permutation-aware block resynthesis.
_RESYNTH_BASIS = frozenset({"cz", "h", "rz", "x"})

# 两比特门的算符 Schmidt 秩 ≤ 2（局部等价于受控门）：少于 3 个这样的门
# 无法组成 SWAP（Weyl 坐标 c3 恒为 0），可以跳过矩阵计算。
# 与 CZ 局部等价的两比特门（单门块的镜像代价为 1：CZ·SWAP ~ iSWAP 需 2 CZ）
_CZ_CLASS_GATES = frozenset({"cx", "cy", "cz", "ch", "ecr"})

_RANK2_GATES = frozenset(
    {
        "cx",
        "cy",
        "cz",
        "ch",
        "cp",
        "cu1",
        "cu3",
        "cu",
        "crx",
        "cry",
        "crz",
        "cs",
        "csdg",
        "csx",
        "rxx",
        "ryy",
        "rzz",
        "rzx",
        "ecr",
    }
)


def _collect_pair_blocks(
    gates: list[_MappingGate],
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Find maximal gate runs confined to one logical qubit pair.

    A block starts and ends with a two-qubit gate on the pair ``(a, b)``
    and contains every single-qubit gate on ``a``/``b`` in between; no
    other gate touches ``a`` or ``b`` inside it.  Measurements close the
    block on their qubit.

    Returns:
        ``(blocks, pairs)``: positions into ``gates`` (increasing) and the
        ``(a, b)`` pair of each block, ``a`` being the first operand of the
        block's first gate.
    """
    blocks: list[list[int]] = []
    pairs: list[tuple[int, int]] = []
    open_block: dict[int, int] = {}
    tail: dict[int, list[int]] = {}

    def close(q: int) -> None:
        bid = open_block.get(q)
        if bid is None:
            return
        for r in pairs[bid]:
            del open_block[r]
            tail.pop(r, None)

    for pos, g in enumerate(gates):
        if g.q3 >= 0:
            for q in (g.q1, g.q2, g.q3):
                close(q)
        elif g.is_two_qubit:
            a, b = g.q1, g.q2
            bid = open_block.get(a)
            if bid is not None and open_block.get(b) == bid:
                blocks[bid].extend(sorted(tail.pop(a, []) + tail.pop(b, [])))
                blocks[bid].append(pos)
            else:
                close(a)
                close(b)
                open_block[a] = open_block[b] = len(blocks)
                blocks.append([pos])
                pairs.append((a, b))
        elif g.name == "measure":
            close(g.q1)
        elif g.q1 in open_block:
            tail.setdefault(g.q1, []).append(pos)
    return blocks, pairs


def _block_unitary(
    block: list[int],
    a: int,
    gates: list[_MappingGate],
    circuit: QuantumCircuit,
) -> np.ndarray | None:
    """``4×4`` unitary of a pair block (qubit ``a`` is the high qubit).

    Returns ``None`` if a gate of the block has no numeric matrix.
    """
    from quspect.gates._matrix_utils import gate_matrix

    u = np.eye(4, dtype=complex)
    for i in block:
        g = gates[i]
        gate = g.gate if g.gate is not None else circuit[g.index].gate
        if gate is None:
            return None
        try:
            m = np.asarray(gate_matrix(gate), dtype=complex)
        except (KeyError, TypeError, ValueError):
            return None
        if g.is_two_qubit:
            if g.q1 != a:
                m = _SWAP_4 @ m @ _SWAP_4
        elif g.q1 == a:
            m = np.kron(m, _EYE_2)
        else:
            m = np.kron(_EYE_2, m)
        u = m @ u
    return u


def _annotate_mirror_costs(
    dag: _RoutingDag, gates: list[_MappingGate], circuit: QuantumCircuit
) -> None:
    """Fill ``dag.mirror_fwd`` / ``dag.mirror_bwd`` with block mirror costs.

    Every maximal pair block ``U`` (see :func:`_collect_pair_blocks`) gets the
    number of extra CZ its mirror ``U · SWAP`` needs after resynthesis,
    ``max(0, cz(U · SWAP) - min(n2q, cz(U)))`` from the Weyl coordinates
    (``SWAP · U`` and ``U · SWAP`` are locally equivalent, so the same cost
    holds for a SWAP before the block).  The cost is stored on the block's
    last two-qubit node (forward routing) and its first one (backward
    routing); all other nodes get ``-1``.  A lone CZ-class gate costs 1
    (``CZ · SWAP`` ~ iSWAP needs 2 CZ), a generic SU(4) block 0.
    """
    dag.mirror_fwd, dag.mirror_bwd = _mirror_cost_arrays(dag, gates, circuit)
    dag.native = None


def _mirror_cost_arrays(
    dag: _RoutingDag, gates: list[_MappingGate], circuit: QuantumCircuit
) -> tuple[list[int], list[int]]:
    """The ``(mirror_fwd, mirror_bwd)`` arrays of :func:`_annotate_mirror_costs`."""
    from quspect.synthesis.two_qubit import num_cz_required

    num_nodes = len(dag.nodes)
    fwd = [-1] * num_nodes
    bwd = [-1] * num_nodes
    node_of = {id(g): n for n, g in enumerate(dag.nodes)}
    for n, members in enumerate(dag.extra):
        for _, g in members:
            node_of[id(g)] = n
    blocks, pairs = _collect_pair_blocks(gates)
    for block, (a, _) in zip(blocks, pairs, strict=True):
        two = [i for i in block if gates[i].is_two_qubit]
        first, last = node_of.get(id(gates[two[0]])), node_of.get(id(gates[two[-1]]))
        if first is None or last is None:  # pragma: no cover - defensive
            continue
        if len(two) == 1 and gates[two[0]].name in _CZ_CLASS_GATES:
            cost = 1
        else:
            u = _block_unitary(block, a, gates, circuit)
            if u is None:
                continue
            before = min(len(two), num_cz_required(u))
            cost = max(0, num_cz_required(u @ _SWAP_4) - before)
        fwd[last] = cost
        bwd[first] = cost
    return fwd, bwd


def _swap_residual(
    block: list[int],
    a: int,
    gates: list[_MappingGate],
    circuit: QuantumCircuit,
    u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(A, B)`` if the block equals ``SWAP · (A ⊗ B)``, else ``None``.

    ``A`` acts on qubit ``a`` (the block's first operand), ``B`` on the
    other qubit; both are applied *before* the exchange.  ``u`` is the
    block unitary if already known.
    """
    if u is None:
        two = [gates[i] for i in block if gates[i].is_two_qubit]
        if len(two) < 3 and all(g.name in _RANK2_GATES for g in two):
            return None
        u = _block_unitary(block, a, gates, circuit)
        if u is None:
            return None
    w = _SWAP_4 @ u
    # 重排 (a_out b_out, a_in b_in) → (a_out a_in, b_out b_in)：秩 1 ⇔ 张量积。
    r = w.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    left, sv, right = np.linalg.svd(r)
    if sv[1] > _PRODUCT_TOL * sv[0]:
        return None
    scale = np.sqrt(sv[0])
    mat_a = left[:, 0].reshape(2, 2) * scale
    mat_b = right[0, :].reshape(2, 2) * scale
    if not np.allclose(np.kron(mat_a, mat_b), w, atol=_PRODUCT_TOL):
        return None
    mat_a = mat_a / np.sqrt(np.linalg.det(mat_a))
    mat_b = mat_b / np.sqrt(np.linalg.det(mat_b))
    return mat_a, mat_b


def _cheaper_with_swap(u: np.ndarray, num_2q: int) -> list[SynthOp] | None:
    """Synthesis of ``V = SWAP · u`` if ``V`` needs fewer CZ than ``u``.

    The block ``u`` (with ``num_2q`` two-qubit gates) equals ``SWAP · V``;
    when the exchange is absorbed into a qubit relabelling only ``V`` has to
    be implemented (permutation-aware synthesis).  For instance a
    DCNOT-class block (two CZ) becomes one CZ, and a SWAP-class block none.

    Returns:
        The verified ``{cz, h, rz, x}`` gate tuples of ``V`` on local qubits
        ``0`` (the block's high qubit) and ``1``, or ``None`` when ``V`` is
        not cheaper than what block resynthesis gives for ``u``.
    """
    from quspect.synthesis.two_qubit import num_cz_required, synthesize_two_qubit_ops

    if num_2q < 2:
        return None
    before = min(num_2q, num_cz_required(u))
    v = _SWAP_4 @ u
    if num_cz_required(v) >= before:
        return None
    try:
        ops = synthesize_two_qubit_ops(v)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    if sum(1 for name, _, _ in ops if name == "cz") >= before:
        return None
    return ops


def _is_plain_swap(circuit: QuantumCircuit, index: int) -> bool:
    """Whether instruction ``index`` is an unparameterized ``swap`` gate."""
    gate = circuit[index].gate
    return gate is not None and gate.name == "swap" and not gate.params


def _elide_swap_blocks(
    gates: list[_MappingGate],
    circuit: QuantumCircuit,
    num_logical: int,
    basis: Collection[str],
    *,
    resynthesize: bool = False,
) -> tuple[list[_MappingGate], list[int]]:
    """Absorb SWAP-equivalent two-qubit blocks into a qubit relabelling.

    Each block that implements ``SWAP · (A ⊗ B)`` (an explicit ``swap``
    gate, or e.g. its three-CZ decomposition) is removed: ``A`` and ``B``
    are re-synthesized into ``basis`` and every later gate has the two
    logical labels exchanged.  With ``resynthesize`` (requires ``{cz, h,
    rz, x}`` in ``basis``), any block ``U`` with at least two two-qubit gates
    for which ``SWAP · U`` needs fewer CZ than ``U`` is replaced the same way
    by the synthesis of ``SWAP · U`` (see :func:`_cheaper_with_swap`).
    Explicit ``swap`` gates inside other blocks are removed individually the
    same way.  After the rewrite, the state of original logical qubit ``q``
    lives on token ``sigma[q]``.

    Returns:
        ``(gates, sigma)``; blocks whose residual cannot be expressed in
        ``basis`` are kept.
    """
    from quspect.gates.synthesis import synthesize_one_qubit

    sigma = list(range(num_logical))
    blocks, pairs = _collect_pair_blocks(gates)
    dropped: set[int] = set()
    residual_at: dict[int, tuple[int, int, list[Gate], list[Gate]]] = {}
    resynth_at: dict[int, tuple[int, int, list[SynthOp]]] = {}
    resynthesize = resynthesize and frozenset(basis) >= _RESYNTH_BASIS
    for block, (a, b) in zip(blocks, pairs, strict=True):
        u = None
        num_2q = 0
        if resynthesize:
            num_2q = sum(1 for i in block if gates[i].is_two_qubit)
            if num_2q >= 2:
                u = _block_unitary(block, a, gates, circuit)
        res = _swap_residual(block, a, gates, circuit, u)
        if res is None:
            ops = None if u is None else _cheaper_with_swap(u, num_2q)
            if ops is not None:
                dropped.update(block)
                resynth_at[block[-1]] = (a, b, ops)
            continue
        seq_a = synthesize_one_qubit(res[0], basis)
        seq_b = synthesize_one_qubit(res[1], basis)
        if seq_a is None or seq_b is None:
            continue
        dropped.update(block)
        residual_at[block[-1]] = (a, b, seq_a, seq_b)
    # 其余块中的显式 SWAP 门本身就是精确置换，单独消去。
    for pos, g in enumerate(gates):
        if (
            pos not in dropped
            and g.is_two_qubit
            and g.name == "swap"
            and g.gate is None
            and _is_plain_swap(circuit, g.index)
        ):
            dropped.add(pos)
            residual_at[pos] = (g.q1, g.q2, [], [])
    if not dropped:
        return gates, sigma

    out: list[_MappingGate] = []
    for pos, g in enumerate(gates):
        if pos in dropped:
            synth = resynth_at.get(pos)
            if synth is not None:
                a, b, ops = synth
                ta, tb = sigma[a], sigma[b]
                for ins in ops_to_instructions(ops, (ta, tb)):
                    assert ins.gate is not None
                    two = len(ins.qubits) == 2
                    out.append(
                        _MappingGate(
                            two,
                            ins.qubits[0],
                            ins.qubits[-1],
                            g.index,
                            ins.gate.name,
                            gate=ins.gate,
                        )
                    )
                sigma[a], sigma[b] = tb, ta
                continue
            info = residual_at.get(pos)
            if info is not None:
                a, b, seq_a, seq_b = info
                ta, tb = sigma[a], sigma[b]
                for token, seq in ((ta, seq_a), (tb, seq_b)):
                    for gate in seq:
                        out.append(
                            _MappingGate(
                                False, token, token, g.index, gate.name, gate=gate
                            )
                        )
                sigma[a], sigma[b] = tb, ta
            continue
        out.append(
            replace(
                g,
                q1=sigma[g.q1],
                q2=sigma[g.q2],
                q3=sigma[g.q3] if g.q3 >= 0 else -1,
            )
        )
    logger.debug(
        "Elided %d SWAP-equivalent block(s) into the output permutation "
        "(%d resynthesized as SWAP · V)",
        len(residual_at) + len(resynth_at),
        len(resynth_at),
    )
    return out, sigma


def _elide_trailing_swaps(
    physical_gates: list[_MappingGate],
    initial: list[int],
    num_logical: int,
) -> tuple[list[_MappingGate], list[int], list[int]]:
    """Drop routing SWAPs followed by no gate on either qubit.

    A routing SWAP whose two physical qubits are afterwards only measured
    (or swapped again by other dropped SWAPs) is removed; the later
    measurements are redirected to where the state now stays.  Workspace
    wires and the final mapping are recomputed from the kept SWAPs.

    Args:
        physical_gates: Routed gates on physical qubits (``_replay``).
        initial: Wire → physical mapping including workspace wires.
        num_logical: Number of logical wires (prefix of ``initial``).

    Returns:
        ``(physical_gates, initial_mapping, final_mapping)``.
    """
    touched: set[int] = set()
    measures_at: dict[int, list[int]] = {}
    dropped: set[int] = set()
    for k in range(len(physical_gates) - 1, -1, -1):
        g = physical_gates[k]
        if g.name == "swap" and g.index < 0:
            x, y = g.q1, g.q2
            if x not in touched and y not in touched:
                dropped.add(k)
                mx = measures_at.pop(x, None)
                my = measures_at.pop(y, None)
                if my is not None:
                    measures_at[x] = my
                if mx is not None:
                    measures_at[y] = mx
                continue
            touched.update((x, y))
        elif g.name == "measure":
            measures_at.setdefault(g.q1, []).append(k)
        else:
            touched.update((g.q1, g.q2))

    if dropped:
        target_of = {k: p for p, ks in measures_at.items() for k in ks}
        kept: list[_MappingGate] = []
        for k, g in enumerate(physical_gates):
            if k in dropped:
                continue
            p = target_of.get(k)
            if p is not None and p != g.q1:
                g = replace(g, q1=p, q2=p)
            kept.append(g)
        physical_gates = kept
        logger.debug("Elided %d trailing routing SWAP(s)", len(dropped))

    wires = list(initial[:num_logical])
    occupant: dict[int, int] = {p: w for w, p in enumerate(wires)}
    position = list(wires)
    for g in physical_gates:
        if g.name != "swap" or g.index >= 0:
            continue
        for p in (g.q1, g.q2):
            if p not in occupant:
                occupant[p] = len(wires)
                wires.append(p)
                position.append(p)
        wx, wy = occupant[g.q1], occupant[g.q2]
        occupant[g.q1], occupant[g.q2] = wy, wx
        position[wx], position[wy] = g.q2, g.q1
    return physical_gates, wires, position


# =============================================================================
# Circuit extraction and reconstruction
# =============================================================================


def _extract_gates(
    circuit: QuantumCircuit,
) -> tuple[list[_MappingGate], int]:
    """Extract a simplified gate list from a :class:`QuantumCircuit`.

    Returns:
        ``(gates, num_logical_qubits)``.
    """
    gates: list[_MappingGate] = []
    for idx, instr in enumerate(circuit):
        if instr.is_gate and instr.gate is not None:
            qs = instr.qubits
            if len(qs) == 3 and instr.gate.name in ROUTABLE_THREE_QUBIT_GATES:
                gates.append(
                    _MappingGate(
                        is_two_qubit=False,
                        q1=qs[0],
                        q2=qs[1],
                        index=idx,
                        name=instr.gate.name,
                        q3=qs[2],
                    )
                )
                continue
            if len(qs) > 2:
                raise ValueError(
                    f"SABRE cannot route {len(qs)}-qubit gate "
                    f"{instr.gate.name!r} at instruction {idx}; decompose it "
                    "into one- and two-qubit gates before routing (only 'ccx' "
                    "is routed as a three-qubit unit)."
                )
            is_two = len(qs) == 2
            q1 = qs[0]
            q2 = qs[1] if is_two else q1
            gates.append(
                _MappingGate(
                    is_two_qubit=is_two,
                    q1=q1,
                    q2=q2,
                    index=idx,
                    name=instr.gate.name,
                )
            )
        elif instr.is_measure:
            for q, c in zip(instr.qubits, instr.clbits, strict=True):
                gates.append(
                    _MappingGate(
                        is_two_qubit=False,
                        q1=q,
                        q2=q,
                        index=idx,
                        name="measure",
                        clbit=c,
                    )
                )

    return gates, circuit.num_qubits


def _build_output_circuit(
    original: QuantumCircuit,
    physical_gates: list[_MappingGate],
    initial_mapping: list[int],
    num_logical: int,
) -> QuantumCircuit:
    """Build the routed :class:`QuantumCircuit` from physical gate list.

    The heuristic search produces gates with **physical** qubit indices.
    Convert those physical operands to stable circuit-wire indices using
    the inverse of ``initial_mapping``.  The mapping must stay fixed here:
    emitted SWAP gates move quantum state between those wires, so using an
    inverse mapping that evolves with the SWAPs would apply later gates to
    the wrong physical qubits when ``initial_mapping`` is forwarded to a
    provider as a static ``initial_layout``.
    """
    from quspect.circuit.circuit import QuantumCircuit

    # Fixed physical → circuit-wire mapping.  ``initial_mapping`` may include
    # extra workspace wires added by SABRE when it routes through an otherwise
    # unused physical qubit.
    p2l: dict[int, int] = {}
    for logical, physical in enumerate(initial_mapping):
        if physical >= 0:
            p2l[physical] = logical

    out = QuantumCircuit(name=original.name)

    for pg in physical_gates:
        if pg.name == "swap":
            out.swap(p2l[pg.q1], p2l[pg.q2])
        elif pg.name == "measure":
            if pg.clbit is None:
                raise ValueError("Routed measurement is missing its classical bit.")
            out.measure([p2l[pg.q1]], [pg.clbit])
        elif pg.gate is not None:
            operands = (pg.q1, pg.q2) if pg.is_two_qubit else (pg.q1,)
            out.append(pg.gate, tuple(p2l[p] for p in operands))
        else:
            if pg.index >= 0:
                orig_instr = original[pg.index]
                if orig_instr.is_gate and orig_instr.gate is not None:
                    physical_operands = (pg.q1, pg.q2)
                    routed_operands = tuple(
                        p2l[physical]
                        for physical in physical_operands[: len(orig_instr.qubits)]
                    )
                    out.append(orig_instr.gate, routed_operands)

    return out
