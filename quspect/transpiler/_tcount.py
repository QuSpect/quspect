"""T-count optimization of Clifford+T circuits.

:class:`TCountOptimizePass` lowers the number of ``T`` / ``T†`` gates, the
dominant cost of fault-tolerant execution.  It works on the *Clifford+T
regions* of a circuit: maximal runs of ``h``, ``s``, ``sdg``, ``t``, ``tdg``,
``x``, ``y``, ``z``, ``cx``, ``cz``, ``swap``, ``ccx`` and ``rz`` / ``p`` /
``u1`` with an angle that is a multiple of ``π/4``.  Any other instruction
(other gates, measurements, barriers, noise channels) ends a region and is
kept in place.

**Method.**  A region is rewritten as ``F · R_m ⋯ R_1``: a Clifford ``F`` after
``π/4`` rotations about Pauli operators, one rotation per ``T`` gate (Gosset
et al. 2014).  The pass then

1. merges rotations about the same Pauli that can be brought together
   (rotation merging; Amy–Maslov–Mosca 2014, Nam et al. 2018);
2. collects mutually commuting rotations into groups.  In a Clifford frame a
   group is a phase polynomial ``Σ_j ±(c_j·x mod 2)`` with one parity per
   rotation, and its parities are reduced by TODD (third-order
   duplicate-and-destroy, Heyfron & Campbell 2018) with best-move selection
   and sideways moves.  Groups are formed in several ways (consecutive runs,
   gathered by commutation, extended by rotations borrowed from neighbouring
   groups) and the best of a small portfolio of orders is kept;
3. optionally (``rm_iterations``, ``tensor_walks``) searches the large groups
   further: a Reed–Muller large-neighbourhood search, started from the TODD
   result and — when the cubic part of the polynomial is trilinear, as in
   finite-field multipliers — from low-rank tensor decompositions found by a
   flip-graph walk;
4. emits the rotations with a CNOT-aware schedule and re-synthesizes ``F``
   from its tableau.

Every replacement of a group is accepted only after an exact check that the
old and new phase polynomials differ by a Clifford (integer arithmetic on the
multilinear form mod 8), and that Clifford is applied.  The output equals the
input up to a global phase and is deterministic for a given ``seed``; a region
is replaced only if its T-count decreases.

The pass minimizes T-count only: the CX count of a rewritten region usually
grows, and connectivity is ignored, so run it before layout and routing.  It
is not part of :func:`~quspect.transpiler.transpile`.

**With ancillas** (``ancillas=True``).  Hadamard gates between the T gates
split the phase polynomials.  Heyfron & Campbell's Hadamard gadget replaces
each of them by a clean ancilla, a measurement and classically controlled
Clifford gates (see :mod:`quspect.transpiler._tcount_ancilla`), so that all T
gates of a region form one phase polynomial and the search above applies to
the whole region.  A region is rewritten this way only when that gives
strictly fewer T gates than the ancilla-free result.  The output then
contains measurements and classically controlled gates
(``InstructionType.CONDITIONAL``): it implements the same unitary on the
circuit's qubits for *every* measurement record, returns every ancilla to
``|0⟩`` and leaves each qubit on its own index.  The ancillas are the qubit
indices from ``circuit.num_qubits`` on and must be in ``|0⟩``; the gadget
measurements write the classical bits from ``circuit.num_clbits`` on.
Published T-counts with Hadamard gadgets are far lower than without, e.g.
``rc_adder_6`` 47 → 37 and ``mod_red_21`` 67 → 51 here.  With gadgets the
default settings give ``adder_8`` 117 (best published 119, FastTODD),
``qcla_adder_10`` 109 (FastTODD 109) and ``ham15-med`` 141 (FastTODD 137,
AlphaTensor-Quantum 156).

**Reference results** (ancilla-free, verified exactly; see ``docs``): on the
standard arithmetic benchmarks of Nam et al. the default settings match or
improve the published ancilla-free T-counts of TODD-type optimizers, e.g.
``mod_mult_55`` 49 → 18, ``hwb6`` 105 → 67, ``mod_red_21`` 119 → 67; with
``tensor_walks`` the GF(2^m) multipliers reach 39 (m = 4), 61–63 (m = 5).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.gates import standard
from quspect.transpiler import _tcount_ancilla as gadget
from quspect.transpiler import _tcount_poly as poly
from quspect.transpiler import _tcount_rotations as rot
from quspect.transpiler._tcount_rotations import CX, SDG, TDG, GateT, H, Pauli, S, T

__all__ = ["TCountOptimizePass", "TCountStats"]

logger = logging.getLogger(__name__)

#: Angles within this distance of a multiple of ``π/4`` are Clifford+T phases.
_ANGLE_ATOL = 1e-12
#: Groups with at least this many parities get the optional heavy search.
_HEAVY_MIN_COLUMNS = 24
#: Groups up to this size get sideways TODD moves when ``plateau`` is automatic.
_PLATEAU_AUTO_COLUMNS = 24
_PLATEAU_AUTO = 100
#: Number of best starting points handed to the Reed–Muller search.
_HEAVY_KEEP = 4

_PHASE_GATES: dict[int, tuple[int, ...]] = {
    0: (),
    1: (T,),
    2: (S,),
    3: (S, T),
    4: (S, S),
    5: (S, S, T),
    6: (SDG,),
    7: (TDG,),
}
_PHASE_NAMES = {"z": 4, "s": 2, "sdg": 6, "t": 1, "tdg": 7}
_ROTATION_NAMES = frozenset({"rz", "p", "u1"})
#: Sideways TODD moves for the phase polynomial of a gadgetized region when
#: ``plateau`` is automatic (they pay off on these large polynomials).
_GADGET_PLATEAU = 200
#: A gadget form with more variables (qubits of the region plus ancillas) than
#: the compiled search handles (256-bit columns) is not tried.
_GADGET_MAX_VARIABLES = poly.NATIVE_MAX_VARIABLES
_CONDITIONAL_GATES = {
    "x": standard.X,
    "z": standard.Z,
    "s": standard.S,
    "sdg": standard.SDG,
}
_OUT_GATES = {
    H: standard.H,
    S: standard.S,
    SDG: standard.SDG,
    T: standard.T,
    TDG: standard.TDG,
    CX: standard.CX,
}


@dataclass
class TCountStats:
    """Counters of the last :meth:`TCountOptimizePass.run`.

    Attributes:
        t_count_in: ``T`` / ``T†`` gates in the Clifford+T regions of the input
            (a ``ccx`` counts 7, ``rz(kπ/4)`` with odd ``k`` counts 1).
        t_count_out: ``T`` / ``T†`` gates in those regions of the output.
        regions: Clifford+T regions with at least two ``T`` gates.
        regions_improved: Regions that were replaced.
        ancillas: Ancilla qubits used (the largest number in one region; the
            same ancillas serve every region).  0 without ``ancillas=True``.
        measurements: Gadget measurements added.
        conditional_gates: Classically controlled gates added.

    Example:
        >>> TCountStats().t_count_in
        0
    """

    t_count_in: int = 0
    t_count_out: int = 0
    regions: int = 0
    regions_improved: int = 0
    ancillas: int = 0
    measurements: int = 0
    conditional_gates: int = 0


def _phase_index(angle: float) -> int | None:
    """``k`` with ``angle ≡ k·π/4 (mod 2π)``, or ``None``."""
    if not math.isfinite(angle):
        return None
    k = round(angle / (0.25 * math.pi))
    if abs(angle - k * 0.25 * math.pi) > _ANGLE_ATOL * max(1.0, abs(angle)):
        return None
    return int(k) % 8


def _translate(ins: CircuitInstruction) -> list[GateT] | None:
    """Clifford+T gate list of a supported gate (on its own qubits), else ``None``."""
    gate = ins.gate
    if not ins.is_gate or gate is None:
        return None
    name, qs = gate.name, ins.qubits
    if len(gate.params) == 0:
        if name == "h":
            return [(H, qs[0], -1)]
        k = _PHASE_NAMES.get(name)
        if k is not None:
            return [(op, qs[0], -1) for op in _PHASE_GATES[k]]
        if name == "cx":
            return [(CX, qs[0], qs[1])]
        if name == "cz":
            return [(H, qs[1], -1), (CX, qs[0], qs[1]), (H, qs[1], -1)]
        if name == "x":
            return [(H, qs[0], -1), (S, qs[0], -1), (S, qs[0], -1), (H, qs[0], -1)]
        if name == "y":  # Y ∝ Z·X
            q = qs[0]
            return [
                (H, q, -1),
                (S, q, -1),
                (S, q, -1),
                (H, q, -1),
                (S, q, -1),
                (S, q, -1),
            ]
        if name == "id":
            return []
        if name == "swap":
            a, b = qs
            return [(CX, a, b), (CX, b, a), (CX, a, b)]
        if name == "ccx":
            a, b, c = qs
            return [
                (H, c, -1),
                (CX, b, c),
                (TDG, c, -1),
                (CX, a, c),
                (T, c, -1),
                (CX, b, c),
                (TDG, c, -1),
                (CX, a, c),
                (T, b, -1),
                (T, c, -1),
                (H, c, -1),
                (CX, a, b),
                (T, a, -1),
                (TDG, b, -1),
                (CX, a, b),
            ]
        return None
    if name in _ROTATION_NAMES and len(gate.params) == 1:
        param = gate.params[0]
        if isinstance(param, Real):
            k = _phase_index(float(param))
            if k is not None:
                return [(op, qs[0], -1) for op in _PHASE_GATES[k]]
    return None


def _cleanup(gates: list[GateT]) -> list[GateT]:
    """Cancel adjacent inverse pairs (``h h``, ``s sdg``, ``cx cx``) repeatedly."""
    inverse = {H: H, S: SDG, SDG: S, T: TDG, TDG: T, CX: CX}
    changed = True
    while changed:
        changed = False
        out: list[GateT | None] = []
        last: dict[int, int] = {}
        for g in gates:
            qs = (g[1],) if g[2] < 0 else (g[1], g[2])
            i = last.get(qs[0], -1)
            prev = out[i] if i >= 0 else None
            if (
                prev is not None
                and prev[0] == inverse[g[0]]
                and prev[1] == g[1]
                and prev[2] == g[2]
                and all(last.get(q, -1) == i for q in qs)
            ):
                out[i] = None
                for q in qs:
                    del last[q]
                changed = True
                continue
            out.append(g)
            for q in qs:
                last[q] = len(out) - 1
        gates = [g for g in out if g is not None]
    return gates


def _commuting_groups(rots: Sequence[Pauli]) -> list[list[Pauli]]:
    """Reorder rotations into commuting groups, exchanging only commuting rotations."""
    groups: list[list[Pauli]] = []
    for r in rots:
        target = len(groups)
        for g in range(len(groups) - 1, -1, -1):
            if all(rot.commute(r, q) for q in groups[g]):
                target = g
            else:
                break
        if target == len(groups):
            groups.append([r])
        else:
            groups[target].append(r)
    return groups


def _runs(rots: Sequence[Pauli]) -> list[list[Pauli]]:
    """Maximal runs of consecutive, mutually commuting rotations (no reordering)."""
    out: list[list[Pauli]] = []
    i = 0
    while i < len(rots):
        j = i + 1
        while j < len(rots) and all(rot.commute(rots[j], rots[t]) for t in range(i, j)):
            j += 1
        out.append(list(rots[i:j]))
        i = j
    return out


def _extended_group(
    groups: list[list[Pauli]], gi: int
) -> tuple[list[Pauli], list[Pauli], list[Pauli]]:
    """Group ``gi`` plus the rotations of other groups that can be commuted next to it.

    Returns ``(before, group, after)``, a reordering of all rotations in which
    the extended group is contiguous.  A rotation is pulled in only if it
    commutes with the whole group and with every rotation it has to pass.
    """
    group = list(groups[gi])
    after_groups: list[list[Pauli]] = []
    passed: list[Pauli] = []
    for gj in range(gi + 1, len(groups)):
        stay = []
        for r in groups[gj]:
            if all(rot.commute(r, q) for q in group) and all(
                rot.commute(r, q) for q in passed
            ):
                group.append(r)
            else:
                stay.append(r)
        after_groups.append(stay)
        passed += stay
    before_groups: list[list[Pauli]] = []
    passed = []
    for gj in range(gi - 1, -1, -1):
        stay = []
        for r in groups[gj]:
            if all(rot.commute(r, q) for q in group) and all(
                rot.commute(r, q) for q in passed
            ):
                group.append(r)
            else:
                stay.append(r)
        before_groups.append(stay)
        passed += stay
    before = [r for g in reversed(before_groups) for r in g]
    after = [r for g in after_groups for r in g]
    return before, group, after


class TCountOptimizePass:
    """Reduce the T-count of the Clifford+T regions of a circuit.

    See the module docstring for the method.  The default settings run
    rotation merging and TODD; ``rm_iterations`` and ``tensor_walks`` enable
    the (much slower) searches for large phase polynomials.

    Args:
        plateau: Sideways TODD moves per group.  ``None`` (default) uses 100
            for groups of at most 24 parities and 0 for larger ones.
        rm_iterations: Steps of the Reed–Muller search on each group with at
            least 24 parities (0 disables it).
        rm_runs: Independent Reed–Muller runs (different seeds, spread over
            the best starting points); they run concurrently in the native
            implementation.
        tensor_walks: Flip-graph walks used to build starting points from
            low-rank decompositions when the cubic part of a group is
            trilinear (0 disables it).
        tensor_steps: Steps per flip-graph walk.
        lookahead: Rotations looked ahead when choosing CX-ladder targets
            during emission (affects CX count only).
        seed: Seed of all randomized searches; the result is a deterministic
            function of the input circuit and the arguments.
        ancillas: Allow Hadamard gadgets: clean ancilla qubits, mid-circuit
            measurements and classically controlled gates (see the module
            docstring).  Off by default: the output of the default pass is a
            unitary circuit on the input's qubits.
        max_ancillas: With ``ancillas``, the most ancillas a region may use
            (``None``: no limit).  A gadget form needing more than 256
            variables (qubits of the region plus its ancillas) is never
            tried.  The search on a gadget form takes seconds up to about 64
            variables and minutes beyond (``ham15-med``, 72–73 variables:
            about 25 minutes); ``max_ancillas`` bounds it.
        native: Use the compiled implementation of the searches when it is
            available (same results as the Python one, much faster).

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2).t(0).t(0)
        >>> out = TCountOptimizePass().run(qc)
        >>> sum(i.gate.name in ("t", "tdg") for i in out)
        0
    """

    def __init__(
        self,
        *,
        plateau: int | None = None,
        rm_iterations: int = 0,
        rm_runs: int = 4,
        tensor_walks: int = 0,
        tensor_steps: int = 200_000,
        lookahead: int = 6,
        seed: int = 0,
        ancillas: bool = False,
        max_ancillas: int | None = None,
        native: bool = True,
    ) -> None:
        if plateau is not None and plateau < 0:
            raise ValueError("plateau must be non-negative.")
        if rm_iterations < 0 or tensor_walks < 0 or tensor_steps < 0 or lookahead < 0:
            raise ValueError(
                "rm_iterations, tensor_walks, tensor_steps and lookahead must be >= 0."
            )
        if rm_runs < 1:
            raise ValueError("rm_runs must be at least 1.")
        if not 0 <= seed < 2**32:
            raise ValueError("seed must be in [0, 2**32).")
        self.plateau = plateau
        self.rm_iterations = rm_iterations
        self.rm_runs = rm_runs
        self.tensor_walks = tensor_walks
        self.tensor_steps = tensor_steps
        self.lookahead = lookahead
        self.seed = seed
        if max_ancillas is not None and max_ancillas < 0:
            raise ValueError("max_ancillas must be non-negative.")
        self.ancillas = ancillas
        self.max_ancillas = max_ancillas
        self.native = native
        self.stats = TCountStats()
        self._first_ancilla = 0
        self._first_clbit = 0
        self._heavy_cache: dict[tuple[tuple[tuple[int, int], ...], int], list[int]] = {}

    # ------------------------------------------------------------------ public
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return an equivalent circuit (up to global phase) with fewer ``T`` gates.

        Args:
            circuit: Circuit to optimize.  Instructions outside the supported
                Clifford+T gate set are kept in place and split the circuit
                into regions that are optimized independently.

        Returns:
            A new circuit; regions whose T-count could not be lowered are
            copied unchanged.  A rewritten region consists of ``h``, ``s``,
            ``sdg``, ``t``, ``tdg`` and ``cx`` gates.

        Example:
            >>> from quspect import QuantumCircuit
            >>> qc = QuantumCircuit()
            >>> for a, b, c in [(0, 1, 2), (1, 2, 3), (0, 1, 2)]:
            ...     qc = qc.ccx(a, b, c)
            >>> p = TCountOptimizePass()
            >>> out = p.run(qc)
            >>> p.stats.t_count_in, p.stats.t_count_out < 21
            (21, True)
        """
        stats = TCountStats()
        self._heavy_cache = {}
        # Ancillas and gadget bits lie behind everything the circuit uses.
        self._first_ancilla = circuit.num_qubits
        self._first_clbit = circuit.num_clbits
        out: list[CircuitInstruction] = []
        region: list[CircuitInstruction] = []
        gates: list[GateT] = []
        for ins in circuit:
            translated = _translate(ins)
            if translated is None:
                out += self._region(region, gates, stats)
                region, gates = [], []
                out.append(ins)
            else:
                region.append(ins)
                gates += translated
        out += self._region(region, gates, stats)
        self.stats = stats
        if stats.t_count_out < stats.t_count_in:
            logger.debug(
                "TCountOptimizePass: T-count %d -> %d",
                stats.t_count_in,
                stats.t_count_out,
            )
        return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)

    # ------------------------------------------------------------------ regions
    def _region(
        self, region: list[CircuitInstruction], gates: list[GateT], stats: TCountStats
    ) -> list[CircuitInstruction]:
        t_in = sum(1 for g in gates if g[0] == T or g[0] == TDG)
        stats.t_count_in += t_in
        if t_in < 2:
            stats.t_count_out += t_in
            return region
        stats.regions += 1
        qubits = sorted(
            {q for g in gates for q in ((g[1],) if g[2] < 0 else (g[1], g[2]))}
        )
        local = {q: i for i, q in enumerate(qubits)}
        n = len(qubits)
        mapped = [(g[0], local[g[1]], local[g[2]] if g[2] >= 0 else -1) for g in gates]
        merged = rot.merge_rotations(*rot.from_gates(mapped, n), n)
        clifford, rots = merged
        if rots:
            clifford, rots = self._optimise(clifford, rots, n)
        if self.ancillas and rots:
            program = self._gadget_program(mapped, merged, (clifford, rots), n)
            if program is not None and program.t_count < min(t_in, len(rots)):
                stats.t_count_out += program.t_count
                stats.regions_improved += 1
                return self._gadget_instructions(program, qubits, n, stats)
        if len(rots) >= t_in:
            stats.t_count_out += t_in
            return region
        new = _cleanup(rot.emit(clifford, rots, n, self.lookahead))
        stats.t_count_out += len(rots)
        stats.regions_improved += 1
        return [
            CircuitInstruction(
                type=InstructionType.GATE,
                gate=_OUT_GATES[g[0]],
                qubits=(qubits[g[1]],) if g[2] < 0 else (qubits[g[1]], qubits[g[2]]),
            )
            for g in new
        ]

    # ------------------------------------------------------------------ ancillas
    def _gadget_reduce(
        self, old: list[tuple[int, int]], base: list[int], k: int
    ) -> list[int]:
        """Column search for the single phase polynomial of a gadgetized region."""
        plateau = _GADGET_PLATEAU if self.plateau is None else self.plateau
        best = poly.todd(base, k, plateau=plateau, seed=self.seed, native=self.native)
        if self.rm_iterations or self.tensor_walks:
            other = self._heavy(old, base, k)
            if len(other) < len(best):
                best = other
        return best

    def _gadget_program(
        self,
        mapped: list[GateT],
        merged: tuple[list[GateT], list[Pauli]],
        optimised: tuple[list[GateT], list[Pauli]],
        n: int,
    ) -> gadget.GadgetProgram | None:
        """Best gadget form over three gate lists: fewest T, then fewest ancillas."""
        starts = [
            mapped,
            gadget.min_h_emit(merged[0], merged[1], n),
            gadget.min_h_emit(optimised[0], optimised[1], n),
        ]
        best: gadget.GadgetProgram | None = None
        for gates in starts:
            needed = gadget.internal_hadamards(gates, n)
            if needed == 0:
                continue
            if self.max_ancillas is not None and needed > self.max_ancillas:
                continue
            if n + needed > _GADGET_MAX_VARIABLES:
                continue
            program = gadget.build_gadget_program(gates, n, self._gadget_reduce)
            if program is not None and (
                best is None
                or (program.t_count, program.ancillas) < (best.t_count, best.ancillas)
            ):
                best = program
        return best

    def _gadget_instructions(
        self,
        program: gadget.GadgetProgram,
        qubits: list[int],
        n: int,
        stats: TCountStats,
    ) -> list[CircuitInstruction]:
        """Circuit instructions of a gadget program on the circuit's qubits and bits."""

        def qubit(wire: int) -> int:
            return qubits[wire] if wire < n else self._first_ancilla + wire - n

        clbit = {w: self._first_clbit + k for k, w in enumerate(program.measured)}
        out: list[CircuitInstruction] = []
        run: list[GateT] = []

        def flush() -> None:
            for g in _cleanup(run):
                out.append(
                    CircuitInstruction(
                        type=InstructionType.GATE,
                        gate=_OUT_GATES[g[0]],
                        qubits=(qubit(g[1]),)
                        if g[2] < 0
                        else (qubit(g[1]), qubit(g[2])),
                    )
                )
            run.clear()

        for op in program.ops:
            if op[0] == gadget.GATE:
                run.append(op[1])  # type: ignore[arg-type]
                continue
            flush()
            if op[0] == gadget.MEASURE_X:
                wire = op[1]
                assert isinstance(wire, int)
                out.append(
                    CircuitInstruction(
                        type=InstructionType.GATE,
                        gate=standard.H,
                        qubits=(qubit(wire),),
                    )
                )
                out.append(
                    CircuitInstruction(
                        type=InstructionType.MEASURE,
                        qubits=(qubit(wire),),
                        clbits=(clbit[wire],),
                    )
                )
                stats.measurements += 1
            else:
                _, control, name, target = op
                assert isinstance(control, int) and isinstance(target, int)
                out.append(
                    CircuitInstruction(
                        type=InstructionType.CONDITIONAL,
                        conditional_gate=_CONDITIONAL_GATES[str(name)],
                        qubits=(qubit(target),),
                        clbits=(clbit[control],),
                        condition=1,
                    )
                )
                stats.conditional_gates += 1
        flush()
        stats.ancillas = max(stats.ancillas, program.num_wires - n)
        return out

    # ------------------------------------------------------------------ groups
    def _todd(
        self, old: Sequence[tuple[int, int]], base: list[int], k: int
    ) -> list[int]:
        plateau = self.plateau
        if plateau is None:
            plateau = _PLATEAU_AUTO if len(base) <= _PLATEAU_AUTO_COLUMNS else 0
        return poly.todd(base, k, plateau=plateau, seed=self.seed, native=self.native)

    def _heavy(
        self, old: Sequence[tuple[int, int]], base: list[int], k: int
    ) -> list[int]:
        """TODD, tensor-decomposition starts and Reed–Muller search on a large group."""
        if len(base) < _HEAVY_MIN_COLUMNS:
            return self._todd(old, base, k)
        key = (tuple(old), k)
        cached = self._heavy_cache.get(key)
        if cached is not None:
            return cached
        pool = [self._todd(old, base, k)]
        if self.tensor_walks:
            starts = poly.tensor_starts(
                old,
                walks=self.tensor_walks,
                steps=self.tensor_steps,
                seed=self.seed,
                native=self.native,
            )
            pool += poly.todd_batch([(s, k, 0, 0) for s in starts], native=self.native)
            pool.sort(key=len)
        if self.rm_iterations:
            keep = min(len(pool), _HEAVY_KEEP)
            problems = [
                (pool[j % keep], k, self.rm_iterations, self.seed * 1000003 + j)
                for j in range(self.rm_runs)
            ]
            pool += poly.rm_search(problems, native=self.native)
        best = min(pool, key=len)
        self._heavy_cache[key] = best
        return best

    def _resynth_group(
        self, group: Sequence[Pauli], n: int, heavy: bool
    ) -> tuple[list[GateT], list[Pauli]] | None:
        """Reduce a commuting group: ``(Clifford first, rotations)`` or ``None``."""
        frame = rot.local_frame(group, n)
        if frame is None:
            gates, used = rot.compress_frame(group, n)
            gates = gates + [(H, q, -1) for q in used]
        else:
            gates, used = frame
        k = len(used)
        used_mask = 0
        for q in used:
            used_mask |= 1 << q
        old: list[tuple[int, int]] = []
        for p in group:
            x, z, sg = rot.conj_circ((p[0], p[1], 1), gates)
            if x or z & ~used_mask or not z:
                return None
            old.append(
                (sum(((z >> q) & 1) << i for i, q in enumerate(used)), p[2] * sg)
            )
        base = poly.normalise(c for c, _ in old)
        new = self._heavy(old, base, k) if heavy else self._todd(old, base, k)
        if len(new) >= len(old):
            return None
        corr = poly.clifford_difference(old, new)
        if corr is None:
            return None
        powers, pairs = corr
        diag: list[GateT] = []
        for i in sorted(powers):
            q = used[i]
            diag += [(op, q, -1) for op in _PHASE_GATES[2 * powers[i]]]
        for i, j in pairs:
            diag += [(H, used[j], -1), (CX, used[i], used[j]), (H, used[j], -1)]
        rots = []
        for c in new:
            z = sum(((c >> i) & 1) << q for i, q in enumerate(used))
            rots.append(rot.conj_circ_inv((0, z, 1), gates))
        return gates + diag + [rot.inv_gate(g) for g in reversed(gates)], rots

    def _replace(
        self,
        clifford: list[GateT],
        before: list[Pauli],
        new: list[Pauli],
        after: list[Pauli],
        c: list[GateT],
        n: int,
    ) -> tuple[list[GateT], list[Pauli]]:
        clifford, rots = rot.push_clifford_left(
            clifford, before + new + after, len(before), c
        )
        if len(clifford) > 40 * n:
            clifford = rot.clifford_circuit(clifford, n)
        return rot.merge_rotations(clifford, rots, n)

    def _group_pass(
        self,
        clifford: list[GateT],
        rots: list[Pauli],
        n: int,
        *,
        gather: bool,
        heavy: bool = False,
        max_rounds: int = 50,
    ) -> tuple[list[GateT], list[Pauli]]:
        """TODD on commuting groups, to a fixed point or ``max_rounds`` rounds."""
        for _ in range(max_rounds):
            groups = _commuting_groups(rots) if gather else _runs(rots)
            rots = [r for g in groups for r in g]
            start = 0
            done = True
            for g in groups:
                if len(g) >= 4:
                    res = self._resynth_group(g, n, heavy)
                    if res is not None:
                        c, new = res
                        clifford, rots = self._replace(
                            clifford, rots[:start], new, rots[start + len(g) :], c, n
                        )
                        done = False
                        break
                start += len(g)
            if done:
                break
        return clifford, rots

    def _extend_pass(
        self, clifford: list[GateT], rots: list[Pauli], n: int, max_rounds: int = 50
    ) -> tuple[list[GateT], list[Pauli]]:
        """TODD on groups extended by commuting rotations of neighbouring groups."""
        for _ in range(max_rounds):
            groups = _commuting_groups(rots)
            done = True
            for gi in range(len(groups)):
                before, group, after = _extended_group(groups, gi)
                if len(group) < 4 or len(group) == len(groups[gi]):
                    continue
                res = self._resynth_group(group, n, False)
                if res is not None:
                    c, new = res
                    clifford, rots = self._replace(clifford, before, new, after, c, n)
                    done = False
                    break
            if done:
                break
        return clifford, rots

    def _optimise(
        self, clifford: list[GateT], rots: list[Pauli], n: int
    ) -> tuple[list[GateT], list[Pauli]]:
        """Portfolio of group orders; the fewest rotations win (ties: the earliest)."""
        best: tuple[list[GateT], list[Pauli]] | None = None
        for steps in (
            ("runs", "groups", "extend"),
            ("groups", "runs", "extend"),
            ("extend", "runs", "extend"),
        ):
            cand = (list(clifford), list(rots))
            for step in steps:
                if step == "extend":
                    cand = self._extend_pass(cand[0], cand[1], n)
                else:
                    cand = self._group_pass(
                        cand[0], cand[1], n, gather=step == "groups"
                    )
            if best is None or len(cand[1]) < len(best[1]):
                best = cand
        assert best is not None
        if self.rm_iterations or self.tensor_walks:
            cand = self._group_pass(
                list(best[0]), list(best[1]), n, gather=True, heavy=True
            )
            if len(cand[1]) < len(best[1]):
                best = cand
        return best
