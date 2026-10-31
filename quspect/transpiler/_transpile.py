"""One-call, quality-first ``transpile()`` built from the transpiler passes.

The pipeline for ``optimization_level=2`` (the default) is::

    HighLevelOptimizePass → ThreeQubitBlockPass
    → decompose to CX level → PhaseFoldingPass → HighLevelOptimizePass
    → decompose (keeping ccx) → consolidate
    → SABRE (permutation-aware layout + Toffoli-aware routing)
    → decompose → consolidate → OptimizePass → decompose → consolidate
    → decompose → single-qubit minimization → depth scheduling

Before routing, three-qubit blocks are resynthesized when a Shannon
decomposition is cheaper (e.g. the repeated controlled-``U`` of phase
estimation collapse to at most 10 two-qubit gates each), diagonal
rotations on the same parity are merged on the CX level (phase folding,
which also unlocks CX cancellations), and SABRE absorbs the SWAP part of
two-qubit blocks ``U`` with a cheaper ``SWAP · U`` into the output
permutation.  All rewrites are exact up to global phase.

From level 2 on SABRE routes on whole two-qubit blocks (consecutive gates
on one pair form a single routing node, so the lookahead counts blocks, as
when routing consolidated ``unitary`` blocks) and is mirror-gate aware
(MIRAGE): after the last gate of a block ``U`` it may insert a SWAP on the
same pair, which the post-routing consolidation merges into ``U · SWAP`` —
free for a generic SU(4) block.

Toffolis are kept through the pre-routing decomposition: SABRE routes each
``ccx`` as a three-qubit unit and synthesizes it for the coupling it lands
on (six CX on a triangle, eight nearest-neighbour CX on a path), which is
much cheaper than routing its fifteen-gate textbook decomposition.

Level 3 replaces the single SABRE result by a candidate search: SABRE
routes the circuit many ways (layout trials × routing seeds, see
:meth:`SABRE.route_candidates`), the candidates are ranked by a cheap
estimate of their two-qubit count after block resynthesis, the best few
(plus the one level 2 would use) go through the full post-routing
pipeline, and the winner is chosen by ``(two-qubit gates, two-qubit depth,
total gates)``.  A second candidate family routes the CX-level circuit
(``cx``, ``cp``, ``crz``, ``rzz`` kept whole) on a commutation-aware DAG
with a bonus for SWAPs that merge with the gate just executed on their
pair; this is what turns QAOA-like and CX-ladder circuits into
swap-network-like routings.  A third family routes on block nodes without
mirror gates (mirror decisions are greedy and occasionally hurt, e.g. 2-D
Trotter circuits on a line); all families are merged by estimated
two-qubit count.  Because the level-2 routing is always among
the fully evaluated candidates, level 3 never produces more two-qubit
gates than level 2 for the same seed.

From level 2 on the selected routings are refined by a large-neighbourhood
search: windows of consecutive SWAPs are re-routed exactly (A* with an
admissible bound, see :mod:`quspect.transpiler._route_lns`) and replaced
when the SWAP cost, counting SWAPs merged into neighbouring blocks at
their mirror cost, drops; equally cheap but shallower windows are offered
as additional candidates.  Circuits made of generic SU(4) blocks are also
routed layer by layer on paths and rings, and by a truncated rollout of
SABRE's greedy policy on other coupling graphs
(:mod:`quspect.transpiler._layer_route`).
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from quspect.synthesis.two_qubit import swap_face_templates
from quspect.transpiler._basis import plan_basis, translate_to_target
from quspect.transpiler._consolidate import SYNTHESIS_BASIS, ConsolidateBlocksPass
from quspect.transpiler._gauge import minimize_single_qubit_gates, orient_cx_gates
from quspect.transpiler._highlevel import HighLevelOptimizePass
from quspect.transpiler._layer_route import block_layers, layer_route_plan
from quspect.transpiler._mapping import (
    DEFAULT_LAYOUT_TRIALS,
    SABRE,
    _CandidatePlan,
)
from quspect.transpiler._passes import DecomposePass, OptimizePass
from quspect.transpiler._phase_fold import PhaseFoldingPass
from quspect.transpiler._route_lns import deepen_plan, refine_plan
from quspect.transpiler._schedule import schedule_for_depth
from quspect.transpiler._swap_face import resynthesize_swap_face_blocks
from quspect.transpiler._three_qubit import ThreeQubitBlockPass

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.transpiler._target import CompilationTarget

logger = logging.getLogger(__name__)

# 第 3 级候选搜索的默认参数。路由工作量以"DAG 节点 + 插入的 SWAP"计（候选打分另计，
# 见 SABRE.route_candidates），原生路由核心下每单位约 1 µs（纯 Python 约 10–20 µs）；
# 完整后优化每个估计双比特门约 0.5 ms。
_LEVEL3_LAYOUT_TRIALS = 20
_LEVEL3_SWAP_TRIALS = 12
_LEVEL3_WORK_LIMIT = 3_000_000
_LEVEL3_MAX_EVALUATIONS = 2
# 每个布局的额外路由（swap trial）只保留 SWAP 最少的这么多个做回放和估计
_LEVEL3_SWAP_KEEP: int | None = 2
# 额外完整评估（第一个之外）的估计双比特门总量上限
_LEVEL3_EVAL_BUDGET_2Q = 2_500
# 相位折叠（路由前）从这一级开始启用
_PHASE_FOLD_LEVEL = 2
# 对易感知候选：完整评估的个数、SWAP 融合奖励、占 work_limit 的比例，
# 以及路由前保留不分解的门（对易规则需要看到完整的 cx / cp / crz / rzz；
# ccx 分解掉后其 CX 也能参与对易，实测比整体路由 Toffoli 更好）
_COMM_EVALUATIONS = 1
_COMM_FUSION = 0.3
_COMM_WORK = 1.0
_COMMUTING_KEEP = frozenset({"cx", "cp", "crz", "rzz"})
# 大电路（路由 DAG 节点数超过 _COMM_SMALL）只有当对易 DAG 的深度缩短到
# _COMM_MIN_GAIN 以下时才做对易感知搜索：随机电路、Heisenberg 链几乎没有
# 可利用的对易性，跳过可省近一半路由时间；小电路上这一族候选仍常常更好
# （SWAP 融合奖励、ccx 分解后的路由），开销可忽略，总是搜索。
_COMM_MIN_GAIN = 0.95
_COMM_SMALL = 500
# 镜像门感知路由（MIRAGE）：从这一级开始启用；接受阈值与每个额外 CZ 的分数要求，
# 以及镜像感知的 SWAP 融合奖励（按块的精确镜像代价缩放）
_MIRROR_LEVEL = 2
_MIRROR_THRESHOLD = 0.25
_MIRROR_CZ_PENALTY = 0.5
_MIRROR_FUSION = 0.2
_BLOCK_NODES = True
# 第 3 级：不做镜像的块节点候选族占 work_limit 的比例（0 关闭）
_MIRROR_ALT_WORK = 1.0
_MIRROR_GATES = True
# 近似（approximation_degree）：路由前只做"整个相互作用消失"的近似（每块保留 ≤ 0 个
# 纠缠门，即弱的 cp / crz / 合并块变成单比特门，路由少了一条边）；把 SU(4) 块从 3 个
# CZ 降到 2 个这类近似放到路由后，路由前的线路与精确流水线一致（否则只改了几个块的
# 代价估计，SABRE 的结果会随机地变好或变坏，如 QV-100 网格 +0.5%）。
_PRE_ROUTING_MAX_ENTANGLERS = 0
# 第 2 级：路由前近似改变了线路时，精确线路也路由一次，按"路由开销"代理（路由后估计
# 双比特门数 − 路由前估计）选一个；两者相差不超过这个比例时都做完整后处理再选
_ROUTING_PROXY_MARGIN = 0.1
# 低伸缩初始布局（交互图在耦合图中的加权伸缩最小化，见 _embedding）：从这一级开始
# 作为 SABRE 的额外布局起点（不精炼 + 精炼各一次），个数
_DILATION_LEVEL = 1
_DILATION_LAYOUTS = 1
# CX 朝向的全局选择（对偶分解，见 _gauge.orient_cx_gates）与 SWAP 面块的 SWAP 形
# 模板（选中路由的路由后流程再走一遍 / 最终线路改写）从这一级开始
_ORIENT_LEVEL = 2
_SWAP_POST = True
_SWAP_LATE = True
# 对易感知的深度重排（迹幺半群上的列表调度，见 _schedule）从这一级开始
_SCHEDULE_LEVEL = 1
# 超过这么多条指令时，最终单比特最小化只在先重排的顺序上做一次（见 finish）
_SCHEDULE_BOTH_LIMIT = 20_000
# 路由的大邻域搜索（窗口精确重路由，见 _route_lns）从这一级开始：所有完整评估的候选
# 先做一轮小窗口（1–4 个 SWAP）扫描，第 3 级估计最好的一个再做一轮大窗口
# （6/8/12 个 SWAP）扫描；
# 每个窗口的 A* 展开上限与每轮的总展开预算（约 8 µs / 次）
_LNS_LEVEL = 2
# 一般 SU(4) 块线路的分层路由（路径 / 环）与截断 rollout 路由（见 _layer_route）
# 从这一级开始
_LAYER_ROUTE_LEVEL = 2
# 同一类线路（所有两比特块都是一般 SU(4)）的 SABRE 前瞻：一层就有很多个块，默认的
# 20 个块看不到下一层。扩展集取 _SU4_EXT_LAYERS 层的块数（不少于默认值，
# 不超过 _SU4_EXT_MAX）
_SU4_EXT_LAYERS = 2
_SU4_EXT_MAX = 100
# 同一类线路的前瞻权重（默认 0.5）：前瞻覆盖两层时，下一层与当前层同等重要
_SU4_LOOKAHEAD_WEIGHT = 1.0
_LNS_LIGHT_WINDOWS = (1, 2, 3, 4)
_LNS_HEAVY_WINDOWS = (6, 8, 12)
_LNS_HEAVY_TOP = 1
_LNS_WINDOW_EXPANSIONS = 1000
_LNS_MIN_WINDOW_EXPANSIONS = 32
_LNS_LIGHT_BUDGET = 1_000_000
_LNS_HEAVY_BUDGET = 2_000_000
# 总预算随 SWAP 数增长（每个 SWAP 这么多次展开，不少于 *_MIN）；超过这么多物理比特的
# 设备上按比例缩小（每次展开更贵）
_LNS_LIGHT_PER_SWAP = 300
_LNS_LIGHT_MIN = 20_000
_LNS_HEAVY_PER_SWAP = 1000
_LNS_HEAVY_MIN = 50_000
_LNS_DEVICE_QUBITS = 27
# 深度精炼（见 _route_lns.deepen_plan）：在（精炼后的）路由上做"同代价、深度代理
# 更小就替换"的窗口扫描，作为额外候选与原路由一起完整评估（双比特门数因此不会变多）。
# 第 3 级只对估计双比特门数最低的 _LNS_DEPTH_TOP 个候选做；预算占代价扫描预算的比例
_LNS_DEPTH = True
_LNS_DEPTH_TOP = 2
_LNS_DEPTH_SHARE = 0.5
# 估计双比特门数不超过它的候选：原路由与精炼后的路由都做完整评估；
# 更大的（非默认）候选只评估精炼后的
_LNS_KEEP_BOTH_2Q = 1500
# 深度精炼（lnsd/）胜出时，估计双比特门数不超过它的线路也完成最好的非深度精炼候选再取优
# （finish 阶段不单调：更浅的路由偶尔在最后多出几个双比特门）
_LNS_DEPTH_FINISH_2Q = 20_000
# 第 2 级只在估计双比特门数不超过它时做 LNS（两条路由都完整评估）
_LNS_LEVEL2_MAX_2Q = 5000


@dataclass
class TranspileStats:
    """Statistics of one :func:`transpile` run.

    Attributes:
        optimization_level: The level that was run.
        two_qubit_gates: Two-qubit gates in the output.
        two_qubit_depth: Two-qubit depth of the output.
        total_gates: Gates (any arity, measurements excluded) in the output.
        depth: Depth of the output (measurements included).
        num_swaps: Routing SWAPs inserted by SABRE for the selected
            candidate (before they were decomposed and merged).
        num_candidates: Routed candidates considered (1 below level 3).
        num_evaluated: Candidates run through the full post-routing
            pipeline.
        selected: Label of the selected candidate (``"exact"``,
            ``"trial3/swap1"``, ...).
        candidate_scores: ``(label, two_qubit_gates, two_qubit_depth,
            total_gates)`` of every fully evaluated candidate.
        layout_trials: Layout trials used by SABRE.
        swap_trials: Routing passes per refined layout.
        seconds: Total wall-clock time.
        stage_seconds: Wall-clock time per stage (``"pre"``, ``"route"``,
            ``"post"``).
        approximation_tolerance: Per-operation approximation budget used
            (average gate infidelity, see ``approximation_degree`` of
            :func:`transpile`; ``0.0`` = no deliberate approximation).
        approximated_operations: Gates and two-qubit blocks replaced by an
            approximation with fewer two-qubit gates.
        approximation_error: Upper bound on the operator-norm distance, up
            to global phase, between the output and the input caused by
            these approximations (the sum of their individual distances).
            The output's process infidelity with the input is at most
            ``approximation_error²`` plus the numerical error of the exact
            rewrites (each verified to ``1e-9`` element-wise).
    """

    optimization_level: int
    two_qubit_gates: int = 0
    two_qubit_depth: int = 0
    total_gates: int = 0
    depth: int = 0
    num_swaps: int = 0
    num_candidates: int = 1
    num_evaluated: int = 1
    selected: str = ""
    candidate_scores: list[tuple[str, int, int, int]] = field(default_factory=list)
    layout_trials: int = 0
    swap_trials: int = 1
    seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    approximation_tolerance: float = 0.0
    approximated_operations: int = 0
    approximation_error: float = 0.0


@dataclass
class TranspileResult:
    """Output of :func:`transpile`.

    Two equivalent views of the compiled circuit are provided:

    - ``circuit`` is in **physical-qubit space**: qubit index = physical
      qubit of the target, and every two-qubit gate acts on a coupling
      edge.  Logical qubit ``q`` of the input starts on physical
      ``initial_layout[q]`` and ends on physical ``final_layout[q]``.
    - ``wire_circuit`` is in SABRE's **wire space**: wire ``w`` is pinned
      to physical ``wire_layout[w]`` (logical wires first, then workspace
      wires borrowed by routing), as returned by :meth:`SABRE.solve`.
      ``circuit`` is ``wire_circuit`` with wire ``w`` renamed to
      ``wire_layout[w]``.

    Semantics: applying ``circuit`` to a state where logical qubit ``q``
    sits on physical ``initial_layout[q]`` (other physical qubits in
    ``|0⟩``) equals applying the input circuit and then moving logical
    ``q`` to physical ``final_layout[q]``, up to global phase.

    Attributes:
        circuit: Compiled circuit indexed by physical qubits.
        initial_layout: ``initial_layout[logical] = physical`` at the start.
        final_layout: ``final_layout[logical] = physical`` at the end.
        wire_circuit: Compiled circuit in wire space.
        wire_layout: ``wire_layout[wire] = physical`` (length ≥ number of
            logical qubits).
        stats: Run statistics.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import CompilationTarget, transpile
        >>> target = CompilationTarget.from_coupling_map(
        ...     3, [(0, 1), (1, 2)], native_gates={"cz", "h", "rz", "x"}
        ... )
        >>> res = transpile(QuantumCircuit().h(0).cx(0, 2), target, seed=1)
        >>> res.stats.two_qubit_gates
        1
        >>> sorted(res.circuit.active_qubits) == sorted(
        ...     res.wire_layout[w] for w in res.wire_circuit.active_qubits
        ... )
        True
    """

    circuit: QuantumCircuit
    initial_layout: list[int]
    final_layout: list[int]
    wire_circuit: QuantumCircuit
    wire_layout: list[int]
    stats: TranspileStats


def transpile(
    circuit: QuantumCircuit,
    target: CompilationTarget,
    *,
    optimization_level: int = 2,
    seed: int | None = None,
    initial_layout: Sequence[int] | None = None,
    iter_num: int = 3,
    layout_trials: int | None = None,
    swap_trials: int | None = None,
    max_evaluations: int | None = None,
    work_limit: int | None = None,
    time_limit: float | None = None,
    exact_layout: bool = True,
    elide_permutations: bool = True,
    approximation_degree: float | None = 1.0,
    depth_weight: float = 0.0,
) -> TranspileResult:
    """Compile a circuit to a target's gate set and coupling graph.

    Optimization levels:

    - ``0``: decompose → SABRE → decompose.  No optimization.
    - ``1``: :class:`HighLevelOptimizePass` (cancel self-inverse
      ``ccx``/``cx``/``cz``/``swap`` pairs) → decompose → consolidate
      two-qubit blocks → SABRE → decompose → consolidate → decompose (no
      DAG optimization; fast).  From this level on SABRE's layout search
      also starts from a low-dilation embedding of the interaction graph
      (see ``dilation_layouts`` of :class:`~quspect.transpiler.SABRE`),
      which routes e.g. a 50-qubit CX chain on a 57-qubit heavy-hex
      lattice with one SWAP.
    - ``2`` (default): level 1 plus, before routing,
      :class:`ThreeQubitBlockPass` (Shannon resynthesis of three-qubit
      blocks), :class:`PhaseFoldingPass` on the CX level followed by
      another :class:`HighLevelOptimizePass`, and permutation-aware block
      synthesis and mirror-gate-aware block routing in SABRE; after
      routing :class:`OptimizePass` and another
      consolidate/decompose sweep.  When the pre-routing approximation
      (see ``approximation_degree``) changed the circuit, the exact
      circuit is routed as well and the routing with the smaller
      estimated routing overhead is kept (both are fully optimized and
      compared when the estimates are within 10 %); ``stats.selected``
      is then ``"exact-routing"``.  The three-qubit and folding passes need
      declared native gates.
    - ``3``: level 2 with a candidate search over many SABRE routings
      (layout trials × routing seeds, plus commutation-aware routings of
      the CX-level circuit with a SWAP-fusion bonus, and block routings
      without mirror gates) and selection by the
      final ``(two-qubit gates, two-qubit depth, total gates)``.  Never
      worse than level 2 in two-qubit count for the same ``seed``.

    From level 2 on the routing is refined by a large-neighbourhood search
    (:mod:`quspect.transpiler._route_lns`): windows of 1–4 consecutive
    SWAPs are re-routed exactly by A* with their entry layout and the exit
    positions of the qubits used later fixed, and replaced when the merged
    SWAP cost drops.  Level 2 keeps the better of the original and the
    refined routing after the post-routing pipeline (very large circuits
    are not refined at level 2); level 3 refines every fully evaluated
    candidate this way, the best one also with windows of 6, 8 and 12
    SWAPs, and evaluates the refinements alongside the originals (large
    non-default candidates are replaced by their refinement).

    Depth is the secondary objective of the search: equal-cost depth sweeps
    (``lnsd/`` candidates) replace windows by routings with a shorter
    two-qubit critical path and are evaluated next to the routings they
    refine, so they never add two-qubit gates.

    For circuits whose two-qubit blocks are all generic SU(4) (quantum
    volume), level 2 and 3 widen SABRE's lookahead to two layers of blocks
    and also route by the methods of :mod:`quspect.transpiler._layer_route`:
    on a line by the layered ordered-partition method (about 30 % fewer CZ
    on QV-100), on a ring by its cyclic analogue, and on every other graph
    by a truncated rollout of SABRE's greedy policy (5-6 % fewer CZ on
    QV-100 on a 10x10 grid and on heavy-hex).

    At every level the pre-routing decomposition keeps ``ccx``: SABRE
    routes each Toffoli as a three-qubit unit and synthesizes it for its
    final placement (see :meth:`SABRE.solve`); the post-routing
    decomposition lowers everything to the native gates.

    Target basis: any native gate set with a CZ-class entangler (``cz``,
    ``cx`` or one registered with
    :func:`~quspect.synthesis.two_qubit.register_entangler`) and a universal
    single-qubit set (e.g. ``{rz, h}``, ``{rz, sx}`` as on IBM devices
    ``{cx, rz, sx, x}``, ``{rz, rx}``, ``{u3}``).  The whole pipeline runs in
    a CZ-centred *working basis* (the target's single-qubit gates plus
    ``{cz, h, rz, x}``) and the result is translated at the end: each CZ
    becomes the target entangler (for CX a Hadamard pair on the target qubit,
    oriented per gate so that the Hadamards merge into the neighbouring
    single-qubit runs) and the touched single-qubit runs are resynthesized in
    the target's gates; from level 2 on, :class:`OptimizePass` then runs
    once more in the target's gates.  From level 1 on, the selected result
    finally gets a single-qubit minimization (any basis, CZ included):
    rotations are moved through the two-qubit gates they commute with
    (``rz`` through a CX control or either CZ qubit, ``rx`` through a CX
    target) and CX gates are reversed when the Hadamard pairs this costs
    are absorbed, chosen per wire (beam search) so that the runs need the
    fewest target gates; the two-qubit gates are unchanged.  From level 2
    on it also starts from CX orientations chosen for the whole circuit at
    once (:func:`~quspect.transpiler._gauge.orient_cx_gates`, a dual
    decomposition over the wires) and, after the depth reordering below,
    keeps the shallower result; blocks on the SWAP face of the Weyl chamber
    (routing SWAPs fused with an interaction) also get the SWAP-shaped
    template with a single inner ``rz`` (see
    :mod:`quspect.transpiler._swap_face`).  Its gates are
    then reordered for depth by
    :func:`~quspect.transpiler._schedule.schedule_for_depth` (commuting
    gates and two-qubit blocks are list-scheduled; the two-qubit depth never
    grows and the gates stay the same; when
    reordering the two-qubit blocks first lowers the two-qubit depth, the
    single-qubit minimization is also run on that order and the shallower
    result is kept; above 20 000 instructions only that order is used).
    CX and CZ
    differ only by single-qubit
    gates, so layout and routing are identical for a CZ and a CX target and
    the two-qubit count is the same or (rarely, after the final
    optimization) lower (see :mod:`quspect.transpiler._basis`).  With no
    declared native gates, decomposition is skipped and single-qubit runs
    fuse into ``u3``.

    Level 3 re-routes every refined layout ``swap_trials - 1`` more times
    with other tie-breaking seeds and scores only the two routings with the
    fewest SWAPs per layout (routing is cheap with the native core, scoring
    a candidate is not).  Its work is bounded deterministically: routing
    stops once ``work_limit`` units (DAG nodes + SWAPs per routing pass,
    plus the scoring work of every candidate) are spent, and
    extra full evaluations stop once their estimated two-qubit count
    exceeds a fixed budget, so large circuits get fewer trials and a fixed
    ``seed`` still gives a reproducible result.  ``time_limit`` adds a
    wall-clock cap on the routing trials, at the cost of reproducibility.

    Args:
        circuit: Logical circuit to compile.  Qubit ``q`` is logical ``q``.
        target: Compilation target (coupling graph and native gates).
        optimization_level: 0, 1, 2 or 3 (see above).
        seed: Random seed for SABRE; the same seed gives the same output.
        initial_layout: Optional fixed ``initial_layout[logical] =
            physical``; disables the layout search (level 3 still tries
            several routing seeds from it).
        iter_num: SABRE forward/backward refinement iterations per trial.
        layout_trials: Random layout trials.  Default: the SABRE default
            (6) below level 3, 20 at level 3 (subject to ``work_limit``).
        swap_trials: Level 3 only: routing passes per refined layout
            (default 12; the two with the fewest SWAPs per layout are
            scored).
        max_evaluations: Level 3 only: number of best-ranked candidates
            fully optimized (the level-2 candidate is evaluated in
            addition, and so are the LNS refinements of the evaluated
            candidates).  Default: 1, or 2 for small circuits.
        work_limit: Level 3 only: routing work budget passed to
            :meth:`SABRE.route_candidates` (default 3 000 000, about
            1–3 s with the native routing core, 10–20 times more in pure
            Python); the commutation-aware candidate search gets a
            budget of the same size.
        time_limit: Level 3 only: optional wall-clock budget in seconds
            for the routing trials.
        depth_weight: Level 3 only: trade two-qubit gates for two-qubit
            depth in the candidate selection, which then minimizes
            ``two-qubit gates + depth_weight · two-qubit depth`` (ties by
            the default key).  ``0`` (default) selects by the fewest
            two-qubit gates; a positive weight may select a candidate with
            more two-qubit gates than level 2 (e.g. ``0.5`` prefers a
            routing with 7 % more gates and a third of the two-qubit depth).
            Must be non-negative.
        exact_layout: Try a SWAP-free VF2++ layout first.
        elide_permutations: Absorb SWAP-equivalent blocks and trailing
            routing SWAPs into ``final_layout`` (see :meth:`SABRE.solve`).
        approximation_degree: How much the output may deviate from the
            input, with Qiskit's convention (``1.0`` = no approximation
            beyond a numerical floor, lower = more).  From level 1 on,
            two-qubit gates and consolidated two-qubit blocks are replaced by
            circuits with fewer two-qubit gates when each replacement's
            average gate infidelity ``1 - F̄(U, V)`` is at most
            ``max(1 - approximation_degree, 1e-12)`` (see
            :func:`approximation_tolerance`): before routing only when the
            whole interaction can be dropped (the gate or block becomes
            single-qubit gates, so routing no longer needs to connect the
            pair), after routing any reduction (e.g. 3 → 2 CZ).  The weak
            part of the interaction is dropped and the local part kept, e.g.
            a ``CP(θ)`` with ``θ²/20 ≤ 1e-12`` (``|θ| ≤ 4.47e-6``) becomes
            two ``rz(θ/2)``.  The default ``1.0`` is at least as strict as
            Qiskit's default (whose ``RemoveIdentityEquivalent`` drops whole
            gates up to the same ``1e-12`` and whose
            ``CommutativeCancellation`` drops merged rotations below
            ``1e-5`` rad).  ``None`` disables deliberate approximation.
            Guarantee: the output differs from the input, up to global phase,
            by at most ``stats.approximation_error`` in operator norm (the
            sum of the replacements' distances), so its process infidelity
            is at most ``stats.approximation_error²`` plus the numerical
            error of the exact rewrites.

    Returns:
        A :class:`TranspileResult`.

    Raises:
        ValueError: If ``optimization_level`` is not 0–3, if
            ``approximation_degree`` is not ``None`` or in ``[0, 1]``, if
            ``depth_weight`` is negative, if the circuit does not fit the
            target, for invalid SABRE arguments, or if the target's native
            gates cannot express the circuit (single-qubit set not
            universal, or no supported entangler for a circuit with
            multi-qubit gates).

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import CompilationTarget, transpile
        >>> target = CompilationTarget.from_coupling_map(
        ...     4,
        ...     [(0, 1), (1, 2), (2, 3)],
        ...     native_gates={"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"},
        ... )
        >>> qc = QuantumCircuit().h(0).cx(0, 3).cx(1, 2).cx(0, 2)
        >>> res = transpile(qc, target, optimization_level=3, seed=0)
        >>> {i.gate.name for i in res.circuit if i.is_gate} <= target.native_gates
        True
        >>> all(
        ...     tuple(sorted(i.qubits)) in target.coupling_graph
        ...     for i in res.circuit
        ...     if i.is_gate and len(i.qubits) == 2
        ... )
        True
        >>> len(res.initial_layout) == len(res.final_layout) == qc.num_qubits
        True
        >>> ibm = CompilationTarget.from_coupling_map(
        ...     4, [(0, 1), (1, 2), (2, 3)], native_gates={"cx", "rz", "sx", "x"}
        ... )
        >>> res = transpile(qc.cz(1, 3), ibm, seed=0)
        >>> {i.gate.name for i in res.circuit if i.is_gate} <= ibm.native_gates
        True
    """
    tolerance = approximation_tolerance(approximation_degree)
    with _relaxed_gc():
        return _transpile(
            circuit,
            target,
            optimization_level=optimization_level,
            seed=seed,
            initial_layout=initial_layout,
            iter_num=iter_num,
            layout_trials=layout_trials,
            swap_trials=swap_trials,
            max_evaluations=max_evaluations,
            work_limit=work_limit,
            time_limit=time_limit,
            exact_layout=exact_layout,
            elide_permutations=elide_permutations,
            tolerance=tolerance,
            depth_weight=depth_weight,
        )


#: Per-operation approximation floor (average gate infidelity) at
#: ``approximation_degree=1.0``: the cutoff of Qiskit's
#: ``RemoveIdentityEquivalent`` at its default ``approximation_degree=1.0``.
APPROXIMATION_FLOOR = 1e-12


def approximation_tolerance(approximation_degree: float | None) -> float:
    """Per-operation approximation budget of :func:`transpile`.

    ``None`` disables deliberate approximation (``0.0``); otherwise the
    budget is ``max(1 - approximation_degree, APPROXIMATION_FLOOR)``, an
    average gate infidelity ``1 - F̄`` with
    ``F̄ = (1 + d·|tr(U^† V)|²/d²) / (1 + d)``.

    Args:
        approximation_degree: ``None`` or a number in ``[0, 1]``.

    Returns:
        The largest accepted average gate infidelity of one approximated
        gate or two-qubit block.

    Raises:
        ValueError: If ``approximation_degree`` is not ``None`` and not in
            ``[0, 1]``.

    Example:
        >>> approximation_tolerance(1.0), approximation_tolerance(None)
        (1e-12, 0.0)
        >>> approximation_tolerance(0.999)  # doctest: +ELLIPSIS
        0.00099999...
    """
    if approximation_degree is None:
        return 0.0
    if isinstance(approximation_degree, bool) or not (
        isinstance(approximation_degree, (int, float))
        and 0.0 <= approximation_degree <= 1.0
    ):
        raise ValueError(
            "approximation_degree must be None or a number in [0, 1], "
            f"got {approximation_degree!r}."
        )
    return max(1.0 - float(approximation_degree), APPROXIMATION_FLOOR)


#: Generation-0 threshold of the cyclic garbage collector while
#: :func:`transpile` runs (CPython's default is 2000).  Compilation allocates
#: millions of short-lived, acyclic objects (instructions, gates, small
#: arrays); with the default threshold a large circuit spends ~10 % of its
#: time in full collections that free nothing.  Only the collection
#: frequency changes, never a result.
_GC_THRESHOLD0 = 50_000


@contextmanager
def _relaxed_gc() -> Iterator[None]:
    """Raise the generation-0 GC threshold for the duration of a transpile."""
    old = gc.get_threshold()
    if 0 < old[0] < _GC_THRESHOLD0:
        gc.set_threshold(_GC_THRESHOLD0, *old[1:])
    try:
        yield
    finally:
        gc.set_threshold(*old)


def _transpile(
    circuit: QuantumCircuit,
    target: CompilationTarget,
    *,
    optimization_level: int = 2,
    seed: int | None = None,
    initial_layout: Sequence[int] | None = None,
    iter_num: int = 3,
    layout_trials: int | None = None,
    swap_trials: int | None = None,
    max_evaluations: int | None = None,
    work_limit: int | None = None,
    time_limit: float | None = None,
    exact_layout: bool = True,
    elide_permutations: bool = True,
    tolerance: float = 0.0,
    depth_weight: float = 0.0,
) -> TranspileResult:
    """Body of :func:`transpile` (same arguments; ``tolerance`` from
    :func:`approximation_tolerance`)."""
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError(
            f"optimization_level must be 0, 1, 2 or 3, got {optimization_level!r}."
        )
    if not depth_weight >= 0.0:
        raise ValueError(f"depth_weight must be non-negative, got {depth_weight!r}.")
    t_start = time.perf_counter()
    stats = TranspileStats(optimization_level=optimization_level)
    # 内部流水线在以 CZ 为中心的工作门集上进行（目标单比特门 + {cz, h, rz, x}），
    # 最后一步再翻译为目标的纠缠门 / 单比特门：CZ 与 CX 等只差单比特门，双比特门数不变。
    plan = plan_basis(target.native_gates, circuit)
    target = plan.working_target(target)
    native = frozenset(target.native_gates)
    decompose = DecomposePass(target)
    # 路由前保留 ccx：SABRE 把 Toffoli 当作三比特单元路由并按落点综合
    pre_decompose = DecomposePass(target, keep={"ccx"})
    consolidate = (
        ConsolidateBlocksPass(target)
        if optimization_level >= 1 and (not native or native >= SYNTHESIS_BASIS)
        else None
    )
    optimize = OptimizePass(basis_gates=native or None)
    final_optimize = OptimizePass(basis_gates=plan.native or None)
    sabre = SABRE(
        target,
        permutation_synthesis=optimization_level >= _PHASE_FOLD_LEVEL,
        dilation_layouts=_dilation_layouts(optimization_level),
        **_mirror_options(optimization_level),
    )

    approximate = consolidate is not None and tolerance > 0
    post_consolidate = (
        ConsolidateBlocksPass(target, max_infidelity=tolerance)
        if approximate
        else consolidate
    )
    pre_approx = (
        ConsolidateBlocksPass(
            target,
            max_infidelity=tolerance,
            max_entanglers=_PRE_ROUTING_MAX_ENTANGLERS,
        )
        if approximate
        else None
    )

    def post(c: QuantumCircuit) -> tuple[QuantumCircuit, int, float]:
        """Post-routing pipeline; also returns its approximations (count, error)."""
        c = decompose.run(c)
        if optimization_level == 0:
            return translate_to_target(c, plan), 0, 0.0
        n_approx, err = 0, 0.0
        if post_consolidate is not None:
            c = post_consolidate.run(c)
            n_approx += post_consolidate.stats.blocks_approximated
            err += post_consolidate.stats.approximation_error
        if optimization_level >= 2:
            c = decompose.run(optimize.run(c))
            if post_consolidate is not None:
                c = post_consolidate.run(c)
                n_approx += post_consolidate.stats.blocks_approximated
                err += post_consolidate.stats.approximation_error
        out = translate_to_target(decompose.run(c), plan)
        if optimization_level >= 2 and plan.needs_translation:
            # 翻译后的单比特门再在目标门集里优化一遍（对易 / 合并 / 重综合）：
            # CX 门集约 −1%，IBM {cx, rz, sx, x} 约 −11% 单比特门
            tuned = final_optimize.run(out)
            if _count_two_qubit(tuned) <= _count_two_qubit(out) and all(
                i.gate.name in plan.native for i in tuned if i.is_gate and i.gate
            ):
                out = tuned
        return out, n_approx, err

    def finish(c: QuantumCircuit, routed: QuantumCircuit | None) -> QuantumCircuit:
        # 只对选中的结果做一次：单比特旋转穿过与之对易的双比特门推移 + CX 反向
        # （H⊗H 吸收），逐线束搜索后在目标门集里重综合（双比特门不变）；
        # 然后按对易关系重排以减小深度（门不变）。
        out = finish_one(c)
        if optimization_level < _ORIENT_LEVEL or not plan.native:
            return out
        # SWAP 面上的三纠缠门块（路由把 SWAP 并进相邻的相互作用）用 SWAP 形模板综合
        # （内层只剩一个 rz）：选中的路由用它再走一遍路由后流程；另外把最终线路里的
        # 这种块直接改写（纠缠门不变）。都走完下面的流程，按 (双比特门, 深度, …) 取优
        # （最终线路里没有这种块时两条都跳过）
        swapped = resynthesize_swap_face_blocks(c, plan.native)
        if swapped is c:
            return out
        alts: list[QuantumCircuit] = []
        if routed is not None and _SWAP_POST:
            with swap_face_templates():
                alt, _, _ = post(routed)
            if _count_two_qubit(alt) <= _count_two_qubit(c):
                alts.append(alt)
        if _SWAP_LATE:
            alts.append(swapped)
        for alt in alts:
            done = finish_one(alt)
            if _finish_key(done) < _finish_key(out):
                out = done
        return out

    def finish_one(c: QuantumCircuit) -> QuantumCircuit:
        def gauge(x: QuantumCircuit) -> QuantumCircuit:
            if optimization_level >= 1 and plan.one_qubit:
                return minimize_single_qubit_gates(x, plan.one_qubit)
            return x

        def variants(x: QuantumCircuit, *, blocks: bool = True) -> list[QuantumCircuit]:
            # 单比特最小化后按深度重排。第 2 级起再从全局选定的 CX 朝向（对偶分解）出发
            # 最小化一遍（朝向改变对易关系，由调用方按重排后的深度取优）
            out = [schedule_for_depth(gauge(x), blocks=blocks)]
            if optimization_level >= _ORIENT_LEVEL and plan.one_qubit:
                oriented = orient_cx_gates(x, plan.one_qubit)
                if oriented is not x:
                    out.append(schedule_for_depth(gauge(oriented), blocks=blocks))
            return out

        def shallowest(cands: list[QuantumCircuit]) -> QuantumCircuit:
            best = cands[0]
            for alt in cands[1:]:
                if _depth_key(alt) < _depth_key(best):
                    best = alt
            return best

        if optimization_level < _SCHEDULE_LEVEL:
            return gauge(c)
        # 单比特最小化会把旋转并进两比特块、破坏块的对易性（如 ZZ 链），而它不改
        # 双比特门的顺序：先按块重排再最小化，双比特门深度保持重排后的值。小线路两条
        # 路都算，取 (深度, 双比特门深度, 总门数) 更好的一个（只在提前重排压低了
        # 双比特门深度时多算一遍，按只做局部最小化的结果判断）；大线路只走先重排的一条
        # 路，省一次最小化。
        early = schedule_for_depth(c)
        if early is c:
            return shallowest(variants(c))
        if len(c.instructions) > _SCHEDULE_BOTH_LIMIT:
            return shallowest(variants(early, blocks=False))
        cands = variants(c)
        if _depth(early, two_qubit_only=True) < _depth(cands[0], two_qubit_only=True):
            cands += variants(early)
        return shallowest(cands)

    # --- pre-routing -----------------------------------------------------
    t0 = time.perf_counter()
    logical = circuit
    if approximate:
        stats.approximation_tolerance = tolerance
    # 路由前的近似 (个数, 误差)：逐门的（对易感知候选的线路也含有）与逐块的（只在
    # 块路由候选的线路里）分开记，最后按选中的候选累加
    gate_approx: tuple[int, float] = (0, 0.0)
    block_approx: tuple[int, float] = (0, 0.0)
    if optimization_level >= 1:
        logical = HighLevelOptimizePass().run(logical)
    if optimization_level >= _PHASE_FOLD_LEVEL and native:
        if consolidate is not None:
            logical = ThreeQubitBlockPass().run(logical)
        logical = _fold_phases(logical, native)
    exact_high = logical
    if pre_approx is not None:
        # 在三比特块重综合与相位折叠之后：它们的输入与精确流水线相同
        logical = pre_approx.approximate_gates(logical)
        gate_approx = _approximations(pre_approx)
    high_level = logical
    logical = pre_decompose.run(logical)
    if native and "id" not in native:
        logical = _drop_identity_gates(logical)
    if consolidate is not None:
        pre_consolidate = pre_approx if pre_approx is not None else consolidate
        logical = pre_decompose.run(pre_consolidate.run(logical))
        if pre_approx is not None:
            block_approx = _approximations(pre_approx)
    exact_logical = None
    if pre_approx is not None and (gate_approx[0] > 0 or block_approx[0] > 0):
        assert consolidate is not None
        exact_logical = pre_decompose.run(exact_high)
        if native and "id" not in native:
            exact_logical = _drop_identity_gates(exact_logical)
        exact_logical = pre_decompose.run(consolidate.run(exact_logical))
    num_logical = max(circuit.num_qubits, logical.num_qubits)
    stats.stage_seconds["pre"] = time.perf_counter() - t0

    # --- routing ---------------------------------------------------------
    t0 = time.perf_counter()
    ext_size = (
        _su4_extended_set(sabre, logical, elide_permutations)
        if optimization_level >= _LAYER_ROUTE_LEVEL
        else None
    )
    ext_kw: dict[str, Any] = (
        {}
        if ext_size is None
        else {
            "extended_set_size": ext_size,
            "lookahead_weight": _SU4_LOOKAHEAD_WEIGHT,
        }
    )
    if optimization_level < 3:
        trials = DEFAULT_LAYOUT_TRIALS if layout_trials is None else layout_trials
        solve_args: dict[str, Any] = {
            "iter_num": iter_num,
            "seed": seed,
            "initial_mapping": initial_layout,
            "layout_trials": trials,
            "exact_layout": exact_layout,
            "elide_permutations": elide_permutations,
        }
        route_plan = sabre._solve_plan(logical, **solve_args)
        if ext_size is not None:
            # 宽前瞻的路由也做一次，按估计（QV 类线路上与完整评估一致）取优
            wide = SABRE(
                target,
                permutation_synthesis=optimization_level >= _PHASE_FOLD_LEVEL,
                dilation_layouts=_dilation_layouts(optimization_level),
                **_mirror_options(optimization_level),
                **ext_kw,
            )._solve_plan(logical, **solve_args)
            if wide.estimated_two_qubit_gates < route_plan.estimated_two_qubit_gates:
                route_plan = replace(wide, label="wide")
        if optimization_level >= _LAYER_ROUTE_LEVEL and initial_layout is None:
            # 分层路由的估计更省就替换（QV 类线路上估计与完整评估一致）
            layered = layer_route_plan(sabre, route_plan, seed=_layer_seed(seed))
            if (
                layered is not None
                and layered.estimated_two_qubit_gates
                < route_plan.estimated_two_qubit_gates
            ):
                route_plan = replace(layered, label="layers")
        routed, wires, final = (
            route_plan.build_circuit(),
            route_plan.initial_mapping,
            route_plan.final_mapping,
        )
        stats.layout_trials = 0 if initial_layout is not None else trials
        posted: tuple[QuantumCircuit, int, float] | None = None
        use_exact = False
        if exact_logical is not None:
            routed_x, wires_x, final_x = sabre.solve(
                exact_logical,
                iter_num=iter_num,
                seed=seed,
                initial_mapping=initial_layout,
                layout_trials=trials,
                exact_layout=exact_layout,
                elide_permutations=elide_permutations,
            )
            choice = _pick_routing(routed, logical, routed_x, exact_logical, post)
            use_exact = choice.use_exact
            if choice.posted is not None:
                posted = choice.posted
        if use_exact:
            routed, wires, final = routed_x, wires_x, final_x
            block_approx = (0, 0.0)
            gate_approx = (0, 0.0)
        stats.num_swaps = _count_swaps(routed)
        stats.selected = "exact-routing" if use_exact else "default"
        stats.stage_seconds["route"] = time.perf_counter() - t0
        extra_plans: list[_CandidatePlan] = []
        fallbacks: list[_Choice] = []
        if (
            optimization_level >= _LNS_LEVEL
            and not use_exact
            and route_plan.estimated_two_qubit_gates <= _LNS_LEVEL2_MAX_2Q
        ):
            # 原路由与精炼后的路由都做完整评估（更大的线路上两次评估太贵，不做 LNS）
            t0 = time.perf_counter()
            refined = refine_plan(
                sabre, route_plan, **_lns_options(route_plan, _LNS_LIGHT_WINDOWS)
            )
            if refined is not None:
                extra_plans.append(refined)
            if _LNS_DEPTH:
                deep = _deepen(sabre, refined if refined is not None else route_plan)
                if deep is not None:
                    extra_plans.append(deep)
            stats.stage_seconds["lns"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        best, n_approx, err = post(routed) if posted is None else posted
        if extra_plans:
            stats.candidate_scores.append(("default", *_score(best)))
            current = _Choice(
                best,
                routed,
                wires,
                final,
                stats.selected,
                stats.num_swaps,
                [gate_approx, block_approx, (n_approx, err)],
            )
            evaluated = [current]
            for extra in extra_plans:
                plan_routed = extra.build_circuit()
                plan_post, plan_n, plan_err = post(plan_routed)
                stats.candidate_scores.append((extra.label, *_score(plan_post)))
                extra_choice = _Choice(
                    plan_post,
                    plan_routed,
                    extra.initial_mapping,
                    extra.final_mapping,
                    extra.label,
                    extra.num_swaps,
                    [gate_approx, block_approx, (plan_n, plan_err)],
                )
                evaluated.append(extra_choice)
                if _score(plan_post) < _score(current.post):
                    current = extra_choice
            if route_plan.estimated_two_qubit_gates <= _LNS_KEEP_BOTH_2Q:
                fallbacks = [c for c in evaluated if c is not current]
            elif (
                current.label.startswith("lnsd/")
                and route_plan.estimated_two_qubit_gates <= _LNS_DEPTH_FINISH_2Q
            ):
                rest = [c for c in evaluated if not c.label.startswith("lnsd/")]
                fallbacks = [min(rest, key=lambda c: _score(c.post))]
            best = current.post
            if current.routed is not None:
                routed = current.routed
            wires, final = current.wires, current.final
            stats.num_swaps, stats.selected = current.num_swaps, current.label
            n_approx, err = current.approx[2]
        best_routed: QuantumCircuit | None = routed
        approx_parts = [gate_approx, block_approx, (n_approx, err)]
        stats.stage_seconds["post"] = time.perf_counter() - t0
        if not stats.candidate_scores:
            stats.candidate_scores = [("default", *_score(best))]
    else:
        lt = _LEVEL3_LAYOUT_TRIALS if layout_trials is None else layout_trials
        st = _LEVEL3_SWAP_TRIALS if swap_trials is None else swap_trials
        if work_limit is None:
            work_limit = _LEVEL3_WORK_LIMIT
        stats.layout_trials, stats.swap_trials = lt, st
        cands = _route_candidates(
            target,
            logical,
            iter_num=iter_num,
            seed=seed,
            initial_layout=initial_layout,
            layout_trials=lt,
            swap_trials=st,
            exact_layout=exact_layout,
            elide_permutations=elide_permutations,
            work_limit=work_limit,
            time_limit=time_limit,
            permutation_synthesis=optimization_level >= _PHASE_FOLD_LEVEL,
            swap_keep=_LEVEL3_SWAP_KEEP,
            mirror={**_mirror_options(optimization_level), **ext_kw},
        )
        if _MIRROR_ALT_WORK > 0 and cands[0].num_swaps > 0 and _MIRROR_LEVEL <= 3:
            # 镜像门是贪心决策，在少数结构化电路（链上的 Trotter）上反而更差：再用不做
            # 镜像的块节点路由一族候选，与主族按估计的双比特门数合并排序。
            alt = _route_candidates(
                target,
                logical,
                iter_num=iter_num,
                seed=seed,
                initial_layout=initial_layout,
                layout_trials=lt,
                swap_trials=st,
                exact_layout=False,
                elide_permutations=elide_permutations,
                work_limit=max(1, int(_MIRROR_ALT_WORK * work_limit)),
                time_limit=time_limit,
                permutation_synthesis=optimization_level >= _PHASE_FOLD_LEVEL,
                swap_keep=_LEVEL3_SWAP_KEEP,
                mirror={
                    # 不做镜像的这一族保留默认的扩展集：两种前瞻都进入候选
                    **_mirror_options(optimization_level),
                    "mirror_gates": False,
                    "swap_fusion_bonus": 0.0,
                },
                reference_trials=1,
            )
            alt = [replace(c, is_default=False, label=f"block/{c.label}") for c in alt]
            cands = sorted(
                cands + alt,
                key=lambda c: (c.estimated_two_qubit_gates, c.num_swaps, c.depth),
            )
        to_eval = _pick_evaluations(cands, max_evaluations)
        if optimization_level >= _LAYER_ROUTE_LEVEL and initial_layout is None:
            base = next((c for c in cands if c.is_default and c.prep is not None), None)
            layered = (
                layer_route_plan(sabre, base, seed=_layer_seed(seed))
                if base is not None
                else None
            )
            if layered is not None:
                cands = sorted(
                    [*cands, layered],
                    key=lambda c: (c.estimated_two_qubit_gates, c.num_swaps, c.depth),
                )
                to_eval.append(layered)
        if exact_logical is not None:
            # 第 2 级可能选用精确线路的路由：把它（第 2 级的那一个）也完整评估，
            # 保证第 3 级不比第 2 级差
            exact_cands = _route_candidates(
                target,
                exact_logical,
                iter_num=iter_num,
                seed=seed,
                initial_layout=initial_layout,
                layout_trials=min(DEFAULT_LAYOUT_TRIALS, lt),
                swap_trials=1,
                exact_layout=exact_layout,
                elide_permutations=elide_permutations,
                work_limit=None,
                time_limit=None,
                permutation_synthesis=optimization_level >= _PHASE_FOLD_LEVEL,
                mirror={**_mirror_options(optimization_level), **ext_kw},
            )
            to_eval += [
                replace(c, is_default=False, label=f"exact/{c.label}")
                for c in exact_cands
                if c.is_default
            ]
        comm_logical = (
            DecomposePass(target, keep=_COMMUTING_KEEP).run(high_level)
            if consolidate is not None and cands[0].num_swaps > 0
            else None
        )
        if comm_logical is not None and _commutation_worthwhile(comm_logical):
            comm = _commutation_candidates(
                target,
                comm_logical,
                iter_num=iter_num,
                seed=seed,
                initial_layout=initial_layout,
                layout_trials=lt,
                swap_trials=st,
                elide_permutations=elide_permutations,
                work_limit=work_limit,
                time_limit=time_limit,
            )
            cands = cands + comm
            to_eval += comm[:_COMM_EVALUATIONS]
        stats.num_candidates = len(cands)
        stats.stage_seconds["route"] = time.perf_counter() - t0
        if optimization_level >= _LNS_LEVEL:
            t0 = time.perf_counter()
            to_eval = _with_refinements(to_eval, _lns_refinements(sabre, to_eval))
            if _LNS_DEPTH:
                to_eval += _depth_refinements(sabre, to_eval)
            stats.stage_seconds["lns"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        best_key: tuple[float, int, int, int, int] | None = None
        chosen: _CandidatePlan | None = None
        fallbacks = []
        best = logical
        best_routed = None
        best_approx: tuple[int, float] = (0, 0.0)
        # 估计最好的未经 LNS 精炼的候选（见下面的 fallback）
        orig_best: tuple[Any, ...] | None = None
        # 最好的非深度精炼（lnsd/）候选：深度精炼胜出时也完成它
        base_best: tuple[Any, ...] | None = None
        for order, cand in enumerate(to_eval):
            built = cand.build_circuit()
            out, n_approx, err = post(built)
            score = _score(out)
            stats.candidate_scores.append((cand.label, *score))
            key = (score[0] + depth_weight * score[1], *score, order)
            if best_key is None or key < best_key:
                best_key, chosen, best = key, cand, out
                best_approx = (n_approx, err)
                best_routed = built
            if not cand.label.startswith("lns") and (
                orig_best is None or key < orig_best[0]
            ):
                orig_best = (key, cand, out, built, (n_approx, err))
            if not cand.label.startswith("lnsd/") and (
                base_best is None or key < base_best[0]
            ):
                base_best = (key, cand, out, built, (n_approx, err))
        assert chosen is not None

        def parts(cand: _CandidatePlan, post_approx: tuple[int, float]) -> list[Any]:
            # LNS 精炼（lns/、lns2/、lnsd/ 前缀）不改变候选路由的是哪条线路
            family = cand.label
            while family.startswith("lns"):
                family = family.split("/", 1)[1]
            return [
                (0, 0.0) if family.startswith("exact/") else gate_approx,
                # 对易感知候选路由的是逐门近似后的 CX 级线路，不含逐块近似；
                # exact/ 候选路由的是未近似的线路
                (0, 0.0) if family.startswith(("comm/", "exact/")) else block_approx,
                post_approx,
            ]

        approx_parts = parts(chosen, best_approx)
        alt_choices: list[tuple[Any, ...] | None] = []
        if (
            chosen.label.startswith("lns")
            and chosen.estimated_two_qubit_gates <= _LNS_KEEP_BOTH_2Q
        ):
            alt_choices = [orig_best]
            if chosen.label.startswith("lnsd/") and base_best is not orig_best:
                alt_choices.append(base_best)
        elif (
            chosen.label.startswith("lnsd/")
            and chosen.estimated_two_qubit_gates <= _LNS_DEPTH_FINISH_2Q
        ):
            alt_choices = [base_best]
        for alt_choice in alt_choices:
            if alt_choice is None:
                continue
            _, ocand, oout, obuilt, oapprox = alt_choice
            fallbacks.append(
                _Choice(
                    oout,
                    obuilt,
                    ocand.initial_mapping,
                    ocand.final_mapping,
                    ocand.label,
                    ocand.num_swaps,
                    parts(ocand, oapprox),
                )
            )
        stats.num_evaluated = len(to_eval)
        stats.selected = chosen.label
        stats.num_swaps = chosen.num_swaps
        wires, final = chosen.initial_mapping, chosen.final_mapping
        stats.stage_seconds["post"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    best = finish(best, best_routed)
    for fallback in fallbacks:
        # 最后的单比特 / SWAP 面阶段还能去掉双比特门，与路由后流程的打分不完全单调：
        # LNS 精炼胜出时（小线路）也完成最好的原候选，按最终结果取优
        alt_final = finish(fallback.post, fallback.routed)
        if _finish_key(alt_final) < _finish_key(best):
            best = alt_final
            wires, final = fallback.wires, fallback.final
            stats.selected, stats.num_swaps = fallback.label, fallback.num_swaps
            approx_parts = fallback.approx
    _record_approximations(stats, *approx_parts)
    stats.stage_seconds["finish"] = time.perf_counter() - t0
    two_q, two_q_depth, total = _score(best)
    stats.two_qubit_gates, stats.two_qubit_depth, stats.total_gates = (
        two_q,
        two_q_depth,
        total,
    )
    stats.depth = _depth(best, two_qubit_only=False)

    initial_layout_out, final_layout_out = _logical_layouts(
        wires, final, logical.num_qubits, num_logical, target.num_qubits
    )
    physical = _to_physical(best, wires)
    result = TranspileResult(
        circuit=physical,
        initial_layout=initial_layout_out,
        final_layout=final_layout_out,
        wire_circuit=best,
        wire_layout=list(wires),
        stats=stats,
    )
    stats.seconds = time.perf_counter() - t_start
    logger.debug(
        "transpile O%d: %d 2q gates (2q depth %d), candidate %s of %d, %.2f s",
        optimization_level,
        two_q,
        two_q_depth,
        stats.selected,
        stats.num_candidates,
        stats.seconds,
    )
    return result


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class _Choice:
    """A fully post-processed routing kept for the final comparison."""

    post: QuantumCircuit
    routed: QuantumCircuit | None
    wires: list[int]
    final: list[int]
    label: str
    num_swaps: int
    approx: list[tuple[int, float]]


def _approximations(approx: ConsolidateBlocksPass) -> tuple[int, float]:
    """``(count, summed distance)`` of the last run of ``approx``."""
    return approx.stats.blocks_approximated, approx.stats.approximation_error


def _record_approximations(stats: TranspileStats, *parts: tuple[int, float]) -> None:
    """Add the approximations that the selected output went through."""
    for count, err in parts:
        stats.approximated_operations += count
        stats.approximation_error += err


def _route_candidates(
    target: CompilationTarget,
    logical: QuantumCircuit,
    *,
    iter_num: int,
    seed: int | None,
    initial_layout: Sequence[int] | None,
    layout_trials: int,
    swap_trials: int,
    exact_layout: bool,
    elide_permutations: bool,
    work_limit: int | None,
    time_limit: float | None,
    permutation_synthesis: bool,
    swap_keep: int | None = None,
    mirror: dict[str, Any] | None = None,
    reference_trials: int | None = None,
) -> list[_CandidatePlan]:
    """Level-3 candidates; the ``is_default`` one is the level-2 routing.

    ``reference_trials`` (default: the level-2 trial count) layout trials are
    exempt from the budget.
    """
    sabre = SABRE(
        target,
        permutation_synthesis=permutation_synthesis,
        dilation_layouts=_DILATION_LAYOUTS,
        **(mirror or {}),
    )
    return sabre._candidate_plans(
        logical,
        iter_num=iter_num,
        seed=seed,
        initial_mapping=initial_layout,
        layout_trials=layout_trials,
        swap_trials=swap_trials,
        reference_trials=(
            min(DEFAULT_LAYOUT_TRIALS, layout_trials)
            if reference_trials is None
            else reference_trials
        ),
        exact_layout=exact_layout,
        elide_permutations=elide_permutations,
        work_limit=work_limit,
        time_limit=time_limit,
        swap_keep=swap_keep,
    )


def _lns_options(
    plan: _CandidatePlan, windows: tuple[int, ...], heavy: bool = False
) -> dict[str, Any]:
    """Keyword arguments of ``refine_plan`` for one LNS sweep of ``plan``.

    The total A* budget grows with the SWAP count (``_LNS_*_PER_SWAP``
    expansions per SWAP, between ``_LNS_*_MIN`` and ``_LNS_*_BUDGET``) and
    shrinks on devices with more than ``_LNS_DEVICE_QUBITS`` qubits, where an
    expansion costs more; the per-window budget shrinks so that the total
    covers the whole routing.
    """
    per_swap, low, high = (
        (_LNS_HEAVY_PER_SWAP, _LNS_HEAVY_MIN, _LNS_HEAVY_BUDGET)
        if heavy
        else (_LNS_LIGHT_PER_SWAP, _LNS_LIGHT_MIN, _LNS_LIGHT_BUDGET)
    )
    budget = min(high, max(low, per_swap * plan.num_swaps))
    n_phys = max(1, len(plan.route_layout))
    if n_phys > _LNS_DEVICE_QUBITS:
        budget = budget * _LNS_DEVICE_QUBITS // n_phys
    per_window = budget // max(1, plan.num_swaps * len(windows))
    return {
        "window_swaps": windows,
        "window_expansions": max(
            _LNS_MIN_WINDOW_EXPANSIONS, min(_LNS_WINDOW_EXPANSIONS, per_window)
        ),
        "total_expansions": budget,
    }


def _su4_extended_set(sabre: SABRE, logical: QuantumCircuit, elide: bool) -> int | None:
    """SABRE extended-set size for a layered SU(4)-block circuit, else ``None``.

    Applies when every two-qubit block of ``logical`` is a generic SU(4)
    (see :func:`quspect.transpiler._layer_route.block_layers`) and its
    widest layer has more than half the default extended set.
    """
    if not sabre.block_nodes or not sabre.mirror_gates:
        return None
    try:
        prep = sabre._prepare(logical, elide)
    except ValueError:
        return None
    layers = block_layers(prep)
    if layers is None:
        return None
    size = _SU4_EXT_LAYERS * max(len(layer) for layer in layers)
    if size <= sabre.extended_set_size:
        return None
    return min(_SU4_EXT_MAX, size)


def _layer_seed(seed: int | None) -> int:
    """Seed of the layered routing's local search (deterministic default)."""
    return 1 if seed is None else (seed * 0x9E3779B1 + 1) & 0xFFFFFFFF


def _deepen(sabre: SABRE, plan: _CandidatePlan) -> _CandidatePlan | None:
    """Depth-refined copy of ``plan`` (``lnsd/...``), ``None`` if no gain."""
    opts = _lns_options(plan, _LNS_LIGHT_WINDOWS)
    return deepen_plan(
        sabre,
        plan,
        window_swaps=_LNS_LIGHT_WINDOWS,
        window_expansions=opts["window_expansions"],
        expansions=int(_LNS_DEPTH_SHARE * opts["total_expansions"]),
    )


def _lns_refinements(
    sabre: SABRE, plans: list[_CandidatePlan]
) -> list[_CandidatePlan | None]:
    """LNS-refined copies (``lns/...``) of the candidates, ``None`` where no gain.

    Every routing first gets a cheap sweep of exact window re-routing
    (windows of ``_LNS_LIGHT_WINDOWS`` SWAPs, see
    :mod:`quspect.transpiler._route_lns`); the ``_LNS_HEAVY_TOP`` routings
    with the lowest estimate after it get a second sweep with larger windows.
    """
    out = [
        refine_plan(sabre, plan, **_lns_options(plan, _LNS_LIGHT_WINDOWS))
        for plan in plans
    ]
    cur = [r if r is not None else p for r, p in zip(out, plans, strict=True)]
    order = sorted(
        range(len(plans)), key=lambda i: (cur[i].estimated_two_qubit_gates, i)
    )
    extra: list[_CandidatePlan] = []
    for i in _top_with_layered(order, plans, _LNS_HEAVY_TOP):
        heavy = _lns_options(cur[i], _LNS_HEAVY_WINDOWS, heavy=True)
        again = refine_plan(sabre, cur[i], **heavy)
        if again is not None:
            light = out[i]
            out[i] = replace(again, label=f"lns2/{plans[i].label}")
            if light is not None and plans[i].is_default:
                # 第 2 级选的正是默认候选的小窗口精炼：保留它，第 3 级才不比第 2 级差
                extra.append(light)
    return out + extra


def _depth_refinements(
    sabre: SABRE, plans: list[_CandidatePlan]
) -> list[_CandidatePlan]:
    """Depth-refined copies (``lnsd/...``) of the most promising candidates.

    The ``_LNS_DEPTH_TOP`` candidates with the lowest estimated two-qubit
    count (LNS refinements included) get the equal-cost depth sweeps of
    :func:`quspect.transpiler._route_lns.deepen_plan`; the copies are
    evaluated in addition to their originals.
    """
    order = sorted(
        range(len(plans)), key=lambda i: (plans[i].estimated_two_qubit_gates, i)
    )
    out = []
    for i in _top_with_layered(order, plans, _LNS_DEPTH_TOP):
        deep = _deepen(sabre, plans[i])
        if deep is not None:
            out.append(deep)
    return out


def _top_with_layered(
    order: list[int], plans: list[_CandidatePlan], k: int
) -> list[int]:
    """The ``k`` best of ``order`` among the SABRE candidates, plus better layered ones.

    Layered / rollout routings (:mod:`quspect.transpiler._layer_route`) do not
    take the place of a SABRE candidate, so the SABRE candidates refined are
    the same as without them (adding a candidate never costs gates).
    """
    picked: list[int] = []
    sabre = 0
    for i in order:
        if sabre >= k:
            break
        picked.append(i)
        if not _is_layered(plans[i]):
            sabre += 1
    return picked


def _is_layered(plan: _CandidatePlan) -> bool:
    """Whether ``plan`` is a layered / rollout routing (``layer_route_plan``)."""
    return plan.label.split("/", 1)[0] in ("layers", "rollout")


def _with_refinements(
    plans: list[_CandidatePlan], refined: list[_CandidatePlan | None]
) -> list[_CandidatePlan]:
    """Evaluation list with the LNS refinements.

    ``refined[i]`` (for ``i < len(plans)``) refines ``plans[i]``; further
    entries are evaluated as they are.  A refined routing is evaluated in
    addition to its original when the original is the level-2 (default)
    candidate, a commutation-aware one (its nodes are single gates that the
    post-routing consolidation regroups, so the merge model is less
    faithful) or small (estimated two-qubit count at most
    ``_LNS_KEEP_BOTH_2Q``; the refinement is better after the full
    post-routing pipeline in about nine of ten cases, but not always); for
    any other one it replaces the original, saving a full evaluation.
    Originals come first, so ties keep the original.
    """
    kept = []
    for i, p in enumerate(plans):
        r = refined[i] if i < len(refined) else None
        if (
            r is None
            or p.is_default
            or p.label.startswith("comm/")
            or p.estimated_two_qubit_gates <= _LNS_KEEP_BOTH_2Q
        ):
            kept.append(p)
    return kept + [r for r in refined if r is not None]


def _dilation_layouts(optimization_level: int) -> int:
    """Low-dilation layout starts of SABRE at this level (see ``_embedding``)."""
    return _DILATION_LAYOUTS if optimization_level >= _DILATION_LEVEL else 0


def _mirror_options(optimization_level: int) -> dict[str, Any]:
    """SABRE keyword arguments of mirror-gate-aware routing at this level."""
    if optimization_level < _MIRROR_LEVEL:
        return {}
    return {
        "block_nodes": _BLOCK_NODES,
        "mirror_gates": _MIRROR_GATES,
        "mirror_threshold": _MIRROR_THRESHOLD,
        "mirror_cz_penalty": _MIRROR_CZ_PENALTY,
        "swap_fusion_bonus": _MIRROR_FUSION,
    }


def _commutation_worthwhile(circuit: QuantumCircuit) -> bool:
    """Whether the level-3 commutation-aware candidate search is run.

    Always for small circuits (at most ``_COMM_SMALL`` routing nodes); for
    larger ones only if commutation shortens the routing DAG noticeably
    (:func:`_commutation_gain` below ``_COMM_MIN_GAIN``).
    """
    from quspect.transpiler._mapping import _extract_gates

    gates, _ = _extract_gates(circuit)
    nodes = sum(1 for g in gates if g.is_two_qubit or g.q3 >= 0)
    return nodes <= _COMM_SMALL or _commutation_gain(circuit) < _COMM_MIN_GAIN


def _commutation_gain(circuit: QuantumCircuit) -> float:
    """Routing-DAG depth with commutation divided by the depth without.

    A cheap, linear-time test of whether a commutation-aware routing can
    reorder anything useful (``1.0``: nothing commutes).
    """
    from quspect.transpiler._mapping import _build_routing_dag, _extract_gates

    gates, n = _extract_gates(circuit)

    def depth(commutation: bool) -> int:
        dag = _build_routing_dag(gates, n, circuit, commutation=commutation)
        level = [0] * len(dag.nodes)
        for i, preds in enumerate(dag.pred):
            level[i] = 1 + max((level[p] for p in preds), default=0)
        return max(level, default=0)

    plain = depth(False)
    return depth(True) / plain if plain else 1.0


def _commutation_candidates(
    target: CompilationTarget,
    logical: QuantumCircuit,
    *,
    iter_num: int,
    seed: int | None,
    initial_layout: Sequence[int] | None,
    layout_trials: int,
    swap_trials: int,
    elide_permutations: bool,
    work_limit: int,
    time_limit: float | None,
) -> list[_CandidatePlan]:
    """Level-3 candidates routed on a commutation-aware DAG.

    ``logical`` is the CX-level circuit (``cx``/``cp``/``crz``/``rzz`` kept
    whole).  The search gets ``_COMM_WORK`` of the routing budget; the
    returned candidates are relabelled ``comm/...`` and never flagged as
    the level-2 default.
    """
    sabre = SABRE(
        target,
        permutation_synthesis=True,
        commutation_aware=True,
        swap_fusion_bonus=_COMM_FUSION,
        dilation_layouts=_DILATION_LAYOUTS,
    )
    cands = sabre._candidate_plans(
        logical,
        iter_num=iter_num,
        seed=seed,
        initial_mapping=initial_layout,
        layout_trials=layout_trials,
        swap_trials=swap_trials,
        reference_trials=1,
        exact_layout=False,
        elide_permutations=elide_permutations,
        work_limit=max(1, int(_COMM_WORK * work_limit)),
        time_limit=time_limit,
        swap_keep=_LEVEL3_SWAP_KEEP,
    )
    return [replace(c, is_default=False, label=f"comm/{c.label}") for c in cands]


def _pick_evaluations(
    cands: list[_CandidatePlan], max_evaluations: int | None
) -> list[_CandidatePlan]:
    """Candidates to run through the full post-routing pipeline.

    The best-ranked candidate and the level-2 (default) candidate are always
    evaluated.  Further candidates follow in rank order up to
    ``max_evaluations`` (default 2), and by default only while their summed
    estimated two-qubit count stays within a fixed budget, so large circuits
    get fewer full evaluations.  When all of them were routed from
    low-dilation layout starts, the best-ranked candidates from other starts
    are evaluated as well (as many, within the same budget).
    """
    limit = _LEVEL3_MAX_EVALUATIONS if max_evaluations is None else max_evaluations
    budget = _LEVEL3_EVAL_BUDGET_2Q if max_evaluations is None else None
    chosen = cands[:1]
    spent = 0
    for cand in cands[1:]:
        if len(chosen) >= limit:
            break
        spent += cand.estimated_two_qubit_gates
        if budget is not None and spent > budget:
            break
        chosen.append(cand)
    if not any(c.is_default for c in chosen):
        chosen += [c for c in cands if c.is_default]
    if chosen and all(c.embedded for c in chosen):
        # 候选多样性：低伸缩嵌入起点的 SWAP 往往最少，估计排序会把它的几个路由排在
        # 最前面，挤掉更好的随机布局路由（估计与最终门数并不单调）；此时再完整评估
        # 估计最好的非嵌入候选（同样受估计门数预算约束）
        others = [c for c in cands if not c.embedded][:limit]
        for other in others:
            spent += other.estimated_two_qubit_gates
            if budget is not None and spent > budget:
                break
            chosen.append(other)
    return chosen


def _fold_phases(circuit: QuantumCircuit, native: frozenset[str]) -> QuantumCircuit:
    """Phase folding on the CX level, then commutation-aware CX cancellation.

    The circuit is lowered to the native gates plus ``cx`` (and ``ccx``, kept
    for routing), so that CX networks are visible to
    :class:`PhaseFoldingPass`; the merged circuit then goes through
    :class:`HighLevelOptimizePass` again, which cancels the CX pairs that the
    removed rotations used to separate.
    """
    from quspect.ir.compilation import decompose_deep

    lowered = decompose_deep(circuit, supported_gates=set(native) | {"cx", "ccx"})
    folder = PhaseFoldingPass()
    folded = folder.run(lowered)
    if folder.stats.rotations_out == folder.stats.rotations_in:
        return circuit
    return HighLevelOptimizePass().run(folded)


def _drop_identity_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove explicit ``id`` gates (they have no decomposition rule)."""
    from quspect.circuit.circuit import QuantumCircuit

    def is_id(instr: CircuitInstruction) -> bool:
        return instr.is_gate and instr.gate is not None and instr.gate.name == "id"

    if not any(is_id(i) for i in circuit):
        return circuit
    return QuantumCircuit._from_instruction_snapshot(
        [i for i in circuit if not is_id(i)], name=circuit.name
    )


def _logical_layouts(
    wires: Sequence[int],
    final: Sequence[int],
    routed_width: int,
    num_logical: int,
    num_physical: int,
) -> tuple[list[int], list[int]]:
    """Logical ``(initial_layout, final_layout)`` from wire-space mappings.

    The first ``routed_width`` wires are the routed logical qubits (later
    wires are routing workspace).  Logical qubits beyond the routed width
    (idle in the compiled circuit) get unused physical qubits and stay put.
    """
    initial = list(wires[:routed_width])
    end = list(final[:routed_width])
    if num_logical > routed_width:
        used = set(wires)
        free = [p for p in range(num_physical) if p not in used]
        extra = free[: num_logical - routed_width]
        initial += extra
        end += extra
    return initial, end


def _estimate(circuit: QuantumCircuit) -> int:
    """:func:`~._mapping.estimate_two_qubit_cost` of a (wire-space) circuit."""
    from quspect.transpiler._mapping import _extract_gates, estimate_two_qubit_cost

    gates, n = _extract_gates(circuit)
    return estimate_two_qubit_cost(gates, max(n, 1))


@dataclass
class _RoutingChoice:
    """Level-2 choice between the approximate and the exact routing."""

    use_exact: bool
    posted: tuple[QuantumCircuit, int, float] | None = None


def _pick_routing(
    routed: QuantumCircuit,
    logical: QuantumCircuit,
    routed_exact: QuantumCircuit,
    logical_exact: QuantumCircuit,
    post: Callable[[QuantumCircuit], tuple[QuantumCircuit, int, float]],
) -> _RoutingChoice:
    """Choose between routing the approximate and the exact logical circuit.

    The pre-routing approximation removes vanishing interactions, which
    changes the routing DAG; SABRE's result on it is not reliably better
    (the post-routing approximation removes those interactions from the
    exact routing too).  The proxy is the *routing overhead* — estimated
    two-qubit count of the routed circuit minus that of its logical input
    (:func:`_estimate`): the post-routing approximation brings the exact
    routing down to roughly the approximate input plus its overhead.  When
    the two overheads differ by at most ``_ROUTING_PROXY_MARGIN`` the proxy
    is not trusted and both routings go through the post-routing pipeline
    (the better ``(two-qubit gates, two-qubit depth, total gates)`` wins,
    the approximate one on ties).
    """
    over_a = _estimate(routed) - _estimate(logical)
    over_x = _estimate(routed_exact) - _estimate(logical_exact)
    if abs(over_a - over_x) > _ROUTING_PROXY_MARGIN * max(over_a, over_x, 1):
        return _RoutingChoice(use_exact=over_x < over_a)
    out_a = post(routed)
    out_x = post(routed_exact)
    if _score(out_x[0]) < _score(out_a[0]):
        return _RoutingChoice(use_exact=True, posted=out_x)
    return _RoutingChoice(use_exact=False, posted=out_a)


def _count_swaps(circuit: QuantumCircuit) -> int:
    return sum(
        1 for i in circuit if i.is_gate and i.gate is not None and i.gate.name == "swap"
    )


def _count_two_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if i.is_gate and len(i.qubits) == 2)


def _depth(circuit: QuantumCircuit, *, two_qubit_only: bool) -> int:
    """Circuit depth; with ``two_qubit_only`` only two-qubit gates count."""
    level: dict[int, int] = {}
    deepest = 0
    for instr in circuit:
        if instr.is_barrier or not instr.qubits:
            continue
        if two_qubit_only and not (instr.is_gate and len(instr.qubits) == 2):
            continue
        lv = 1 + max(level.get(q, 0) for q in instr.qubits)
        for q in instr.qubits:
            level[q] = lv
        deepest = max(deepest, lv)
    return deepest


def _depth_key(circuit: QuantumCircuit) -> tuple[int, int, int]:
    """``(depth, two-qubit depth, total gates)`` of a circuit."""
    gates = sum(1 for i in circuit if i.is_gate)
    return (
        _depth(circuit, two_qubit_only=False),
        _depth(circuit, two_qubit_only=True),
        gates,
    )


def _finish_key(circuit: QuantumCircuit) -> tuple[int, int, int, int]:
    """``(two-qubit gates, depth, two-qubit depth, total gates)`` of a circuit."""
    return (_count_two_qubit(circuit), *_depth_key(circuit))


def _score(circuit: QuantumCircuit) -> tuple[int, int, int]:
    """``(two-qubit gates, two-qubit depth, total gates)`` of a circuit."""
    gates = sum(1 for i in circuit if i.is_gate)
    return _count_two_qubit(circuit), _depth(circuit, two_qubit_only=True), gates


def _to_physical(circuit: QuantumCircuit, wires: Sequence[int]) -> QuantumCircuit:
    """Rename wire ``w`` of a wire-space circuit to physical ``wires[w]``."""
    from quspect.circuit.circuit import QuantumCircuit

    out = QuantumCircuit(name=circuit.name)
    n = circuit.num_qubits
    if n:
        out.compose(circuit, qubits=list(wires[:n]), inplace=True)
    return out
