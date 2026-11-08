"""Numerical instantiation search for three-qubit circuits (level 3).

For a circuit on exactly three logical qubits the whole ``8×8`` unitary
``U`` is resynthesized by a QSearch-style search (Davis et al. 2020,
arXiv:2002.04612): circuit *structures* — an initial ``U3`` per qubit,
then CZ gates on coupled pairs, each followed by ``Rz·Ry`` on its two
qubits — are explored best-first by CZ count plus the remaining distance,
and the continuous angles of every structure are fitted numerically
(Levenberg–Marquardt on ``V(θ) − e^{iφ}U`` with an analytic Jacobian).  A
structure is accepted only when the element-wise residual is at most
``1e-12``; the search keeps looking for fewer CZ gates until its budget
(a number of optimizer iterations, not wall-clock time) is spent.

The layout is free: one search covers the targets ``P_end·U·P_startᵀ``
for every output permutation (the final layout absorbs it) and, on a path,
every choice of the logical qubit in the middle.

Relation to the exact-rewrite policy: the angles come from a numerical
fit, but the replacement goes through the same verification as every
other resynthesis — the product of the *emitted* gate matrices must equal
``P·U`` up to global phase with ``max|·| ≤ 1e-9`` element-wise — and it is
used only when it has strictly fewer two-qubit gates.  The numerics run on
the deterministic C++ layer (IEEE operations without contraction,
correctly rounded trigonometry, a fixed integer pseudo-random sequence for
restarts, a total order on the search queue), so the result is
bit-identical on every platform and thread count.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from quspect.circuit._native import flatten_slots
from quspect.transpiler._three_qubit import (
    ThreeQubitBlockPass,
    _Block,
    _block_unitary_in,
    _distance,
    _native,
    _ops_to_instructions,
)

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction

__all__ = ["NumericThreeQubit", "NumericThreeQubitBlockPass", "instantiate_three_qubit"]

#: Element-wise residual the fit must reach.
FIT_TOL = 1e-12
#: Verification budget of the emitted gates (phase-aligned, element-wise).
VERIFY_ATOL = 1e-9
#: Optimizer iterations of one call (≈ 60 µs each).
DEFAULT_BUDGET = 40_000
#: Extra pseudo-random starting points per structure.
RESTARTS = 1
#: Searches run in turn as ``(A* weight of the remaining distance, share of
#: the budget)``: f = CZ count + weight · distance, where the distance is
#: ``√(1 − |tr(U†V)|²/64)``; a later search only looks below the CZ count
#: found before.  (Weights 10–30 and two-stage schedules were tried on the
#: development set; one search with weight 20 found the fewest CZ gates.)
SEARCHES: tuple[tuple[float, float], ...] = ((20.0, 1.0),)


@dataclass
class NumericThreeQubit:
    """A verified replacement found by :func:`instantiate_three_qubit`.

    Attributes:
        instructions: Gates on local wires ``0, 1, 2`` (``rz``, ``ry``,
            ``cz``).
        start: Logical qubit ``i`` starts on local wire ``start[i]``.
        end: Logical qubit ``i`` ends on local wire ``end[i]``.
        num_cz: Number of CZ gates.
    """

    instructions: list[CircuitInstruction]
    start: tuple[int, int, int]
    end: tuple[int, int, int]
    num_cz: int


def _perm_matrix(perm: Sequence[int]) -> np.ndarray:
    """Logical qubit ``i`` (local 0 = high bit) moves to wire ``perm[i]``."""
    p = np.zeros((8, 8))
    for x in range(8):
        bits = [(x >> (2 - i)) & 1 for i in range(3)]
        y = 0
        for i in range(3):
            y |= bits[i] << (2 - perm[i])
        p[y, x] = 1.0
    return p


def _start_perms(coupled: Collection[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Initial placements to search, up to automorphisms of the coupling: on
    a path one per logical qubit in the middle, on a triangle only one.
    ``perm[i]`` is the local wire logical qubit ``i`` starts on."""
    if len(coupled) >= 3:
        return [(0, 1, 2)]
    (a, b), (c, d) = sorted(coupled)
    mid = ({a, b} & {c, d}).pop()
    ends = sorted({a, b, c, d} - {mid})
    out = []
    for m in range(3):
        rest = [i for i in range(3) if i != m]
        perm = [0, 0, 0]
        perm[m] = mid
        perm[rest[0]], perm[rest[1]] = ends[0], ends[1]
        out.append((perm[0], perm[1], perm[2]))
    return out


def instantiate_three_qubit(
    circuit: QuantumCircuit,
    coupled: Collection[tuple[int, int]],
    max_cz: int,
    *,
    permute_output: bool = True,
    fixed_layout: bool = False,
    budget: int = DEFAULT_BUDGET,
    restarts: int = RESTARTS,
) -> NumericThreeQubit | None:
    """Resynthesize a three-qubit circuit with at most ``max_cz`` CZ gates.

    Args:
        circuit: Circuit on three qubits, unitary gates only.
        coupled: Coupled pairs of local wires ``0..2`` (at least two).
        max_cz: Largest accepted number of CZ gates.
        permute_output: Allow the output wires to be permuted.
        fixed_layout: Logical qubit ``i`` starts on local wire ``i`` (a
            block inside a larger circuit); it also ends there unless
            ``permute_output``.
        budget: Optimizer iterations in total.
        restarts: Pseudo-random starting points per structure besides the
            one from the parent structure.

    Returns:
        The verified replacement, or ``None`` if none was found (or the C++
        extension is not available).

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().cx(0, 1).cx(1, 2).cx(0, 1).cx(1, 2)  # = cx(0, 2)
        >>> found = instantiate_three_qubit(qc, {(0, 1), (1, 2)}, 3)
        >>> found.num_cz, found.start[1]  # logical 1 off the middle wire
        (1, 0)
    """
    core = _native()
    if core is None or max_cz < 0 or circuit.num_qubits != 3:
        return None
    pairs: set[tuple[int, int]] = {(min(p), max(p)) for p in coupled if p[0] != p[1]}
    if len(pairs) < 2:
        return None
    gates = [ins for ins in circuit if ins.is_gate]
    u = _block_unitary_in([0, 1, 2], gates)
    # 逻辑比特 i 起于局部线 start[i]、终于 end[i]：V·P_start = P_end·U
    starts = [(0, 1, 2)] if fixed_layout else _start_perms(pairs)
    all_ends: list[tuple[int, int, int]] = [
        (a, b, c) for a, b, c in itertools.permutations(range(3))
    ]
    jobs = [
        (start, end)
        for start in starts
        for end in (all_ends if permute_output else [start])
    ]
    targets = [_perm_matrix(end) @ u @ _perm_matrix(start).T for start, end in jobs]
    edges = sorted(pairs)
    best: NumericThreeQubit | None = None
    bound = max_cz
    for factor, share in SEARCHES:
        if bound < 0:
            break
        found, t, czs, params, _res, _it, _nodes = core._instantiate_three_qubit(  # type: ignore[attr-defined]
            targets, edges, bound, int(budget * share), restarts, FIT_TOL, factor
        )
        if not found:
            continue
        ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
        for q in range(3):
            ops.append(("rz", (q,), (params[3 * q],)))
            ops.append(("ry", (q,), (params[3 * q + 1],)))
            ops.append(("rz", (q,), (params[3 * q + 2],)))
        for k, (a, b) in enumerate(czs):
            p = 9 + 4 * k
            ops += [
                ("cz", (a, b), ()),
                ("ry", (a,), (params[p],)),
                ("rz", (a,), (params[p + 1],)),
                ("ry", (b,), (params[p + 2],)),
                ("rz", (b,), (params[p + 3],)),
            ]
        instrs = _ops_to_instructions(ops, [0, 1, 2])
        if _distance(targets[t], _block_unitary_in([0, 1, 2], instrs)) > VERIFY_ATOL:
            continue
        best = NumericThreeQubit(instrs, jobs[t][0], jobs[t][1], len(czs))
        bound = len(czs) - 1
    return best


class NumericThreeQubitBlockPass(ThreeQubitBlockPass):
    """Numerically resynthesize three-qubit blocks of a routed circuit.

    Blocks are collected as in :class:`ThreeQubitBlockPass` (maximal runs of
    gates within three wires); blocks with at least ``min_cz`` two-qubit
    gates are tried in order of decreasing two-qubit count (ties: earlier
    first) with :func:`instantiate_three_qubit` on the coupled pairs among
    their wires, layout fixed, accepting only strictly fewer CZ gates.  At
    most ``total_budget // block_budget`` blocks are tried, so the work per
    run is bounded and deterministic.

    With ``free_output``, a block that ends the circuit on all its wires may
    leave its wires permuted (the caller folds :attr:`wire_moves` into the
    final layout).

    Args:
        coupled: Coupled wire pairs (``None``: all pairs).
        free_output: Let terminal blocks permute their output wires.
        min_cz: Smallest two-qubit count of a block worth trying.
        block_budget: Optimizer iterations per block.
        total_budget: Optimizer iterations per run.
        restarts: See :func:`instantiate_three_qubit`.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().cz(0, 1).cz(1, 2).cz(0, 1).cz(1, 2).cz(0, 1)
        >>> out = NumericThreeQubitBlockPass(min_cz=3).run(qc)
        >>> sum(1 for i in out if len(i.qubits) == 2) < 5
        True
    """

    def __init__(
        self,
        coupled: Collection[tuple[int, int]] | None = None,
        *,
        free_output: bool = False,
        min_cz: int = 4,
        block_budget: int = 4_000,
        total_budget: int = 40_000,
        restarts: int = RESTARTS,
    ) -> None:
        super().__init__()
        self._coupled: set[tuple[int, int]] | None = (
            None if coupled is None else {(min(p), max(p)) for p in coupled}
        )
        self._free_output = free_output
        #: After :meth:`run`: the state of wire ``w`` ends on wire
        #: ``wire_moves[w]`` (only wires of terminal blocks that moved).
        self.wire_moves: dict[int, int] = {}
        self._min_cz = min_cz
        self._block_budget = block_budget
        self._total_budget = total_budget
        self._restarts = restarts
        self._found: list[_Block] = []

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of ``circuit`` with three-qubit blocks resynthesized."""
        from quspect.circuit.circuit import QuantumCircuit

        self._found = []
        self.wire_moves = {}
        super().run(circuit)  # 只收集块（见 _finalize），不改线路
        instrs = list(circuit.instructions)
        last_use: dict[int, int] = {}
        for idx, ins in enumerate(instrs):
            for q in ins.qubits:
                last_use[q] = idx
        cands = []
        for blk in self._found:
            members = [instrs[i] for i in blk.members]
            n2 = sum(1 for i in members if len(i.qubits) == 2)
            if n2 >= self._min_cz and all(len(i.qubits) <= 2 for i in members):
                cands.append((-n2, blk.members[0], blk))
        cands.sort(key=lambda c: (c[0], c[1]))
        tries = self._total_budget // max(1, self._block_budget)
        slots: list[list[CircuitInstruction] | CircuitInstruction | None] = list(instrs)
        changed = False
        for neg_n2, _, blk in cands[:tries]:
            order = sorted(blk.qubits)
            pos = {q: i for i, q in enumerate(order)}
            pairs = {
                (i, j)
                for i in range(3)
                for j in range(i + 1, 3)
                if self._coupled is None or (order[i], order[j]) in self._coupled
            }
            if len(pairs) < 2:
                continue
            sub = QuantumCircuit()
            for i in blk.members:
                ins = instrs[i]
                assert ins.gate is not None
                sub.append(ins.gate, [pos[q] for q in ins.qubits])
            if sub.num_qubits != 3:
                continue
            self.stats.blocks_examined += 1
            member_idx = set(blk.members)
            terminal = self._free_output and all(
                last_use[q] in member_idx for q in order
            )
            try:
                found = instantiate_three_qubit(
                    sub,
                    pairs,
                    -neg_n2 - 1,
                    permute_output=terminal,
                    fixed_layout=True,
                    budget=self._block_budget,
                    restarts=self._restarts,
                )
            except (TypeError, ValueError):
                continue
            if found is None:
                continue
            for i in blk.members:
                slots[i] = None
            slots[blk.members[-1]] = [
                replace(ins, qubits=tuple(order[q] for q in ins.qubits))
                for ins in found.instructions
            ]
            changed = True
            for i in range(3):
                if found.end[i] != i:
                    self.wire_moves[order[i]] = order[found.end[i]]
            self.stats.blocks_replaced += 1
            self.stats.two_qubit_before += -neg_n2
            self.stats.two_qubit_after += found.num_cz
        if not changed:
            return circuit
        return QuantumCircuit._from_instruction_snapshot(
            flatten_slots(slots), name=circuit.name
        )

    def _finalize(
        self,
        block: _Block,
        instrs: list[CircuitInstruction],
        slots: list[list[CircuitInstruction] | CircuitInstruction | None],
    ) -> None:
        self._found.append(block)
