"""Connectivity-aware Toffoli (CCX) synthesis and three-qubit SABRE routing.

Decomposing a Toffoli into the textbook six-CX circuit *before* routing
hands the router a triangle of interactions; on triangle-free coupling
graphs (line, ring, grid, heavy-hex) every such triangle forces SWAPs.
This module lets :class:`~quspect.transpiler.SABRE` route a ``ccx`` as one
three-qubit unit instead:

* a ``ccx`` node is executable once its three physical qubits form a
  connected subgraph (a path ``a - m - c`` or a triangle), and its routing
  cost is ``min_m (D[m][a] + D[m][c]) - 1`` (``1`` when executable, like
  the distance of an executable two-qubit gate);
* after routing, each Toffoli is synthesized for the realised placement:
  the six-CX circuit (:func:`ccx_triangle_decomposition`) on a triangle, or
  the eight-CX nearest-neighbour circuit (:func:`ccx_path_decomposition`)
  whose CXs act only on the two edges of the path.  Among the exact
  variants of each (controls exchanged, circuit reversed; see
  :func:`ccx_candidates`) the one that merges best with the neighbouring
  gates under :class:`~quspect.transpiler.ConsolidateBlocksPass` is kept,
  which typically saves one CX per Toffoli in adder-like circuits.

Both syntheses are exact (no relative phase).  The eight-CX circuit is the
phase polynomial of ``CCZ = (-1)^{abc}`` evaluated on the seven parities
``a, b, c, a⊕b, b⊕c, a⊕c, a⊕b⊕c`` using CXs on the edges ``(a, m)`` and
``(m, c)`` only; the target is moved into the ``Z`` frame by two
Hadamards, so any of the three qubits may sit in the middle.

Routing itself is done by the shared SABRE core
(:func:`quspect.transpiler._routing.route_dag`), which handles two- and
three-qubit nodes in one loop; three-qubit nodes enter the front and
extended-set terms with the cost above
(:func:`quspect.transpiler._routing.cost3`).
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

# Gate 在运行时导入（而非仅类型检查时）：autodoc 据此把签名中的 Gate 解析为
# 完全限定的 quspect.gates.gate.Gate，避免交叉引用歧义。
from quspect.gates.gate import Gate
from quspect.gates.standard import CX, TDG, H, T
from quspect.synthesis._memo import BoundedCache

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.transpiler._mapping import _MappingGate

__all__ = [
    "ccx_candidates",
    "ccx_path_decomposition",
    "ccx_triangle_decomposition",
    "synthesize_ccx_for_coupling",
]

logger = logging.getLogger(__name__)

_TIE_EPS = 1e-10

#: Gate names routed as three-qubit units by :class:`~quspect.transpiler.SABRE`.
ROUTABLE_THREE_QUBIT_GATES = frozenset({"ccx"})

GateSequence = list[tuple["Gate", tuple[int, ...]]]


# =============================================================================
# Toffoli syntheses
# =============================================================================


def ccx_triangle_decomposition(c1: int, c2: int, t: int) -> GateSequence:
    """Six-CX Toffoli for three pairwise-coupled qubits.

    This is the textbook circuit (Nielsen & Chuang, Fig. 4.9) used by the
    built-in ``ccx`` decomposition rule; it needs a CX on every pair.

    Args:
        c1: First control.
        c2: Second control.
        t: Target.

    Returns:
        ``(gate, qubits)`` pairs over ``{h, t, tdg, cx}``.

    Example:
        >>> ops = ccx_triangle_decomposition(0, 1, 2)
        >>> sum(1 for g, _ in ops if g.name == "cx")
        6
    """
    from quspect.gates.standard import CCX
    from quspect.ir.compilation import _decompose_ccx

    return _decompose_ccx(CCX, (c1, c2, t))


def ccx_path_decomposition(c1: int, c2: int, t: int, middle: int) -> GateSequence:
    """Eight-CX nearest-neighbour Toffoli whose CXs touch only ``middle``.

    The CX gates act on ``(e1, middle)`` and ``(middle, e2)`` only, where
    ``e1, e2`` are the two qubits other than ``middle``, so the circuit runs
    on a coupling path ``e1 - middle - e2`` without SWAPs.  ``middle`` may be
    either control or the target.  The circuit is exactly ``CCX`` (no
    relative or global phase).

    Args:
        c1: First control.
        c2: Second control.
        t: Target.
        middle: One of ``c1``, ``c2``, ``t``: the qubit in the middle of
            the coupling path.

    Returns:
        ``(gate, qubits)`` pairs over ``{h, t, tdg, cx}`` with 8 CXs.

    Raises:
        ValueError: If the qubits are not distinct or ``middle`` is not one
            of them.

    Example:
        >>> ops = ccx_path_decomposition(0, 1, 2, middle=1)
        >>> sorted({qs for g, qs in ops if g.name == "cx"})
        [(0, 1), (1, 2)]
    """
    if len({c1, c2, t}) != 3:
        raise ValueError(f"Toffoli qubits must be distinct, got {(c1, c2, t)}.")
    if middle not in (c1, c2, t):
        raise ValueError(f"middle={middle} is not one of the Toffoli qubits.")
    a, c = (q for q in (c1, c2, t) if q != middle)
    b = middle
    # CCZ = exp(iπ/4 · (a + b + c - a⊕b - a⊕c - b⊕c + a⊕b⊕c)) on bit values;
    # wire states are tracked in the comments as parities of (a, b, c).
    ccz: GateSequence = [
        (T, (a,)),
        (T, (b,)),
        (T, (c,)),
        (CX, (a, b)),  # b = a⊕b
        (TDG, (b,)),
        (CX, (b, c)),  # c = a⊕b⊕c
        (T, (c,)),
        (CX, (a, b)),  # b = b
        (CX, (b, c)),  # c = a⊕c
        (TDG, (c,)),
        (CX, (a, b)),  # b = a⊕b
        (CX, (b, c)),  # c = b⊕c
        (TDG, (c,)),
        (CX, (a, b)),  # b = b
        (CX, (b, c)),  # c = c
    ]
    return [(H, (t,)), *ccz, (H, (t,))]


def _dagger(ops: GateSequence) -> GateSequence:
    """Inverse of a ``{h, t, tdg, cx}`` sequence (reversed, ``t ↔ tdg``)."""
    inverse = {"t": TDG, "tdg": T}
    return [(inverse.get(g.name, g), qs) for g, qs in reversed(ops)]


def ccx_candidates(
    c1: int, c2: int, t: int, adjacent: Callable[[int, int], bool]
) -> list[GateSequence]:
    """All built-in exact Toffoli syntheses that fit the given coupling.

    On a coupling triangle these are the four six-CX circuits obtained from
    :func:`ccx_triangle_decomposition` by exchanging the controls and/or
    reversing the circuit (``CCX`` is self-inverse, so the reversed circuit
    with ``t ↔ tdg`` is again ``CCX``).  On a path they are the four
    eight-CX circuits of :func:`ccx_path_decomposition` around the middle
    qubit (target preferred, then ``c1``), again with exchanged ends and/or
    reversed.  The variants differ only in which CXs sit at the borders,
    which decides what merges with neighbouring gates.

    Args:
        c1: First control.
        c2: Second control.
        t: Target.
        adjacent: Predicate telling whether two qubits are coupled.

    Returns:
        Candidate ``(gate, qubits)`` sequences; the first one is the
        canonical choice.

    Raises:
        ValueError: If the three qubits do not form a connected subgraph.

    Example:
        >>> line = {(0, 1), (1, 2)}
        >>> adj = lambda x, y: (min(x, y), max(x, y)) in line
        >>> cands = ccx_candidates(0, 2, 1, adj)
        >>> len(cands), sum(1 for g, _ in cands[0] if g.name == "cx")
        (4, 8)
    """
    ab, at, bt = adjacent(c1, c2), adjacent(c1, t), adjacent(c2, t)
    bases: list[GateSequence]
    if ab and at and bt:
        bases = [
            ccx_triangle_decomposition(c1, c2, t),
            ccx_triangle_decomposition(c2, c1, t),
        ]
    else:
        for middle, ok in ((t, at and bt), (c1, ab and at), (c2, ab and bt)):
            if ok:
                bases = [
                    ccx_path_decomposition(c1, c2, t, middle),
                    ccx_path_decomposition(c2, c1, t, middle),
                ]
                break
        else:
            raise ValueError(
                f"Toffoli qubits {(c1, c2, t)} are not connected on the coupling graph."
            )
    return [bases[0], _dagger(bases[0]), bases[1], _dagger(bases[1])]


def _context_cost(
    before: Sequence[tuple[Gate, tuple[int, ...]]],
    ops: GateSequence,
    after: Sequence[tuple[Gate, tuple[int, ...]]],
) -> tuple[int, int]:
    """``(two-qubit cost, gate count)`` of ``before + ops + after`` after
    two-qubit block consolidation (``swap`` counts 3).

    The result only depends on the gate sequence up to a renaming of its
    qubits (block collection compares qubits for equality only), so it is
    memoized on the sequence with qubits relabelled in order of first use:
    the same Toffoli neighbourhood recurs in every routing candidate.
    """
    seq = (*before, *ops, *after)
    labels: dict[int, int] = {}
    # repr 区分 0.0 与 -0.0（Gate 的相等比较不区分，但矩阵的零符号不同）
    key = tuple(
        (
            gate,
            tuple(map(repr, gate.params)),
            tuple(labels.setdefault(q, len(labels)) for q in qs),
        )
        for gate, qs in seq
    )
    try:
        cached = _CONTEXT_COST_CACHE.get(key)
    except TypeError:  # unhashable (e.g. array-valued) gate parameter
        return _consolidated_cost(seq)
    if cached is None:
        cached = _consolidated_cost(seq)
        _CONTEXT_COST_CACHE.put(key, cached)
    return cached


_CONTEXT_COST_CACHE: BoundedCache[tuple[object, ...], tuple[int, int]] = BoundedCache()


def _consolidated_cost(
    seq: Sequence[tuple[Gate, tuple[int, ...]]],
) -> tuple[int, int]:
    """Uncached core of :func:`_context_cost`."""
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.transpiler._consolidate import ConsolidateBlocksPass

    qc = QuantumCircuit()
    for gate, qs in seq:
        qc.append(gate, qs)
    out = ConsolidateBlocksPass(fuse_single_qubit=False).run(qc)
    cost = 0
    for ins in out:
        if len(ins.qubits) == 2:
            cost += 3 if ins.gate is not None and ins.gate.name == "swap" else 1
    return cost, len(out)


def synthesize_ccx_for_coupling(
    c1: int,
    c2: int,
    t: int,
    adjacent: Callable[[int, int], bool],
    *,
    before: Sequence[tuple[Gate, tuple[int, ...]]] = (),
    after: Sequence[tuple[Gate, tuple[int, ...]]] = (),
) -> GateSequence:
    """Synthesize a Toffoli for the coupling of its three (physical) qubits.

    Uses the six-CX circuit when all three pairs are coupled and the
    eight-CX nearest-neighbour circuit on a path (see
    :func:`ccx_candidates`).  When neighbouring gates are given, the
    candidate whose two-qubit cost after
    :class:`~quspect.transpiler.ConsolidateBlocksPass` resynthesis of
    ``before + candidate + after`` is lowest is returned (ties: fewer
    gates, then the canonical order), so border CXs merge with adjacent
    gates on the same pair.

    Args:
        c1: First control.
        c2: Second control.
        t: Target.
        adjacent: Predicate telling whether two qubits are coupled.
        before: Gates on (a subset of) the three qubits right before the
            Toffoli; only used to pick the cheapest candidate.
        after: Gates on (a subset of) the three qubits right after it.

    Returns:
        ``(gate, qubits)`` pairs over ``{h, t, tdg, cx}`` whose CXs all act
        on coupled pairs.

    Raises:
        ValueError: If the three qubits do not form a connected subgraph.

    Example:
        >>> line = {(0, 1), (1, 2)}
        >>> adj = lambda x, y: (min(x, y), max(x, y)) in line
        >>> ops = synthesize_ccx_for_coupling(0, 2, 1, adj)
        >>> sum(1 for g, _ in ops if g.name == "cx")
        8
    """
    candidates = ccx_candidates(c1, c2, t, adjacent)
    if not before and not after:
        return candidates[0]
    choice_key = _choice_key(c1, c2, t, adjacent, before, after)
    if choice_key is not None:
        rank = _CCX_CHOICE_CACHE.get(choice_key)
        if rank is not None:
            return candidates[rank]
    best_rank = 0
    best_key: tuple[int, int, int] | None = None
    for rank, ops in enumerate(candidates):
        key = (*_context_cost(before, ops, after), rank)
        if best_key is None or key < best_key:
            best_rank, best_key = rank, key
    if choice_key is not None:
        _CCX_CHOICE_CACHE.put(choice_key, best_rank)
    return candidates[best_rank]


def _choice_key(
    c1: int,
    c2: int,
    t: int,
    adjacent: Callable[[int, int], bool],
    before: Sequence[tuple[Gate, tuple[int, ...]]],
    after: Sequence[tuple[Gate, tuple[int, ...]]],
) -> tuple[object, ...] | None:
    """Memo key of the variant :func:`synthesize_ccx_for_coupling` picks.

    The candidates depend only on which of the three pairs are coupled and
    on the roles of the qubits, and every candidate's :func:`_context_cost`
    only on its gate sequence up to a renaming of the qubits.  So the
    chosen rank is a function of the coupling pattern and of ``before``,
    the triple and ``after`` with the qubits relabelled in order of first
    use (gates compared as in :func:`_context_cost`).  ``None`` when a gate
    parameter is unhashable.
    """
    labels: dict[int, int] = {}

    def side(seq: Sequence[tuple[Gate, tuple[int, ...]]]) -> tuple[object, ...]:
        return tuple(
            (
                gate,
                tuple(map(repr, gate.params)),
                tuple(labels.setdefault(q, len(labels)) for q in qs),
            )
            for gate, qs in seq
        )

    head = side(before)
    roles = tuple(labels.setdefault(q, len(labels)) for q in (c1, c2, t))
    key = (
        adjacent(c1, c2),
        adjacent(c1, t),
        adjacent(c2, t),
        head,
        roles,
        side(after),
    )
    try:
        hash(key)
    except TypeError:  # unhashable (e.g. array-valued) gate parameter
        return None
    return key


_CCX_CHOICE_CACHE: BoundedCache[tuple[object, ...], int] = BoundedCache()


#: Maximum number of neighbouring gates on each side used to pick a variant.
_CONTEXT_LIMIT = 24


def _context_window(
    gates: Sequence[_MappingGate],
    order: Sequence[int],
    qubits: tuple[int, int, int],
    gate_of: Callable[[_MappingGate], Gate | None],
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Gates confined to ``qubits`` adjacent to a Toffoli, walking ``order``.

    A qubit stops collecting once a gate couples it to a qubit outside the
    triple (or is a measurement / three-qubit gate); gates on other qubits
    are skipped (they commute with everything collected).
    """
    return _collect_window((gates[j] for j in order), qubits, gate_of)


def _gate_qubits(g: _MappingGate) -> tuple[int, ...]:
    if g.q3 >= 0:
        return (g.q1, g.q2, g.q3)
    if g.is_two_qubit:
        return (g.q1, g.q2)
    return (g.q1,)


def _collect_window(
    walk: Iterable[_MappingGate],
    qubits: tuple[int, int, int],
    gate_of: Callable[[_MappingGate], Gate | None],
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Body of :func:`_context_window` over the gates of a walk."""
    open_qubits = set(qubits)
    window: list[tuple[Gate, tuple[int, ...]]] = []
    for g in walk:
        if not open_qubits or len(window) >= _CONTEXT_LIMIT:
            break
        qs = _gate_qubits(g)
        hit = open_qubits.intersection(qs)
        if not hit:
            continue
        gate = None if g.q3 >= 0 or g.name == "measure" else gate_of(g)
        if gate is None or len(hit) != len(qs):
            open_qubits.difference_update(qs)
            continue
        window.append((gate, qs))
    return window


def _merged_positions(
    lists: Sequence[list[int]], start: int, stop: int, step: int
) -> Iterator[int]:
    """Distinct entries of sorted ``lists`` in ``range(start, stop, step)``, in order.

    Every list is sorted ascending; ``step`` is ``1`` (ascending walk from
    ``start``) or ``-1`` (descending walk from ``start``).
    """
    heads: list[tuple[list[int], int]] = []
    for lst in lists:
        if step > 0:
            i = bisect_left(lst, start)
            if i < len(lst) and lst[i] < stop:
                heads.append((lst, i))
        else:
            i = bisect_right(lst, start) - 1
            if i >= 0 and lst[i] > stop:
                heads.append((lst, i))
    last = None
    while heads:
        best = 0
        val = heads[0][0][heads[0][1]]
        for h in range(1, len(heads)):
            v = heads[h][0][heads[h][1]]
            if (v < val) if step > 0 else (v > val):
                best, val = h, v
        lst, i = heads[best]
        i += step
        if 0 <= i < len(lst) and ((lst[i] < stop) if step > 0 else (lst[i] > stop)):
            heads[best] = (lst, i)
        else:
            del heads[best]
        if val != last:
            last = val
            yield val


def expand_toffolis(
    physical_gates: list[_MappingGate],
    adj_sets: list[set[int]],
    circuit: QuantumCircuit | None = None,
) -> list[_MappingGate]:
    """Replace routed three-qubit ``ccx`` gates by coupling-aware syntheses.

    Each Toffoli is synthesized with :func:`synthesize_ccx_for_coupling`
    for its physical placement; with ``circuit`` (the routed input, used
    to look up the original gates) the variant is chosen against the
    neighbouring gates on the same three qubits.

    Args:
        physical_gates: Routed gates on physical qubits (``_replay``).
        adj_sets: Adjacency sets of the coupling graph.
        circuit: Optional circuit the ``index`` fields refer to.

    Returns:
        The gate list with every three-qubit gate replaced by one- and
        two-qubit gates carrying explicit ``gate`` objects (and the index
        of the original instruction).
    """
    from quspect.gates.standard import SWAP
    from quspect.transpiler._mapping import _MappingGate

    if not any(g.q3 >= 0 for g in physical_gates):
        return physical_gates

    def adjacent(x: int, y: int) -> bool:
        return y in adj_sets[x]

    def gate_of(g: _MappingGate) -> Gate | None:
        if g.gate is not None:
            return g.gate
        if g.index < 0:
            return SWAP if g.name == "swap" else None
        if circuit is None:
            return None
        return circuit[g.index].gate

    # 每个物理比特上的门在 physical_gates 中的位置（升序）：上下文窗口只需访问作用在
    # 三个比特之一上的门，其余的门在原来的逐门遍历中也只是被跳过
    positions: list[list[int]] = [[] for _ in adj_sets]
    try:
        for j, g in enumerate(physical_gates):
            for q in _gate_qubits(g):
                positions[q].append(j)
    except (IndexError, TypeError):
        positions = []
    # 已展开的 Toffoli：physical_gates 中的位置 → 写入 out 的门
    expanded: dict[int, list[_MappingGate]] = {}

    def walk_before(k: int, triple: tuple[int, int, int]) -> Iterator[_MappingGate]:
        """The gates of ``out`` touching ``triple``, last first."""
        for j in _merged_positions([positions[q] for q in triple], k - 1, -1, -1):
            ops = expanded.get(j)
            if ops is None:
                yield physical_gates[j]
            else:
                yield from reversed(ops)

    out: list[_MappingGate] = []
    n_path = n_triangle = 0
    for k, g in enumerate(physical_gates):
        if g.q3 < 0:
            out.append(g)
            continue
        triple = (g.q1, g.q2, g.q3)
        if circuit is None:
            before: GateSequence = []
            after: GateSequence = []
        elif positions:
            before = _collect_window(walk_before(k, triple), triple, gate_of)[::-1]
            after = _collect_window(
                (
                    physical_gates[j]
                    for j in _merged_positions(
                        [positions[q] for q in triple], k + 1, len(physical_gates), 1
                    )
                ),
                triple,
                gate_of,
            )
        else:
            before = _context_window(out, range(len(out) - 1, -1, -1), triple, gate_of)[
                ::-1
            ]
            after = _context_window(
                physical_gates, range(k + 1, len(physical_gates)), triple, gate_of
            )
        ops = synthesize_ccx_for_coupling(
            g.q1, g.q2, g.q3, adjacent, before=before, after=after
        )
        if sum(1 for gate, _ in ops if gate.name == "cx") == 6:
            n_triangle += 1
        else:
            n_path += 1
        start = len(out)
        for gate, qs in ops:
            if len(qs) == 2:
                out.append(
                    _MappingGate(True, qs[0], qs[1], g.index, gate.name, None, gate)
                )
            else:
                out.append(
                    _MappingGate(False, qs[0], qs[0], g.index, gate.name, None, gate)
                )
        expanded[k] = out[start:]
    logger.debug(
        "Synthesized %d Toffoli(s) on coupling triangles and %d on paths",
        n_triangle,
        n_path,
    )
    return out
