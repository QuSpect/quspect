"""Coupling-aware optimal resynthesis of CNOT-phase regions after routing.

:class:`PhasePolynomialPass` works on a routed circuit whose two-qubit gates
are ``cz`` / ``cx`` and rewrites small *CNOT-phase regions* with the fewest
CX gates that the region's couplings allow.

**CNOT-phase regions.**  A circuit made of CX gates and diagonal single-qubit
gates acts as ``|x⟩ ↦ e^{i f(x)} |A x ⊕ c⟩``: ``A ∈ GL(n, 2)`` is the linear
reversible map of the CX network, ``c`` collects the ``x`` gates and ``f(x) =
Σ_y θ_y (y·x)`` is its phase polynomial (Amy–Maslov–Mosca 2014; Nam et al.
2018).  A CX network realizes the region iff it implements ``A`` and every
parity ``y`` with ``θ_y ≠ 0 (mod 2π)`` appears on some wire at some moment
(the rotation ``rz(θ_y)`` is placed there).  Diagonal ``cz`` gates contribute
``π/2·(x_a + x_b − x_a⊕x_b)`` and an anti-diagonal single-qubit run is
``x·diag``, so both stay inside a region.

**Orientation.**  The input is in a CZ basis, where ``cx(c, t) = h_t · cz ·
h_t``.  Every ``cz`` is read as a CX in one of the two directions or as a
diagonal gate.  The single-qubit run between two consecutive two-qubit gates
on a wire then becomes ``H^{t'} R H^{t}``, and the region can continue
through the run only if that is *monomial* (diagonal or anti-diagonal).  A
local search over the three readings of each gate maximizes the number of
monomial runs (the cost of a run depends only on the readings of its two
neighbouring gates, so moves are evaluated in ``O(1)``).

**Regions** are grown greedily in circuit order (or from the end of the
circuit) on at most ``max_qubits`` wires: a wire leaves the region at its
first non-monomial run, and a gate touching a wire that has left (or a
two-qubit gate that would exceed the qubit cap) leaves the region as well,
together with its wires, so the region is convex.  For each region with at
least two two-qubit gates a CX network with strictly fewer gates on the
directed couplings among the region's wires is searched (see
``cnot_phase_synth.hpp``): regions of at most 6 wires by A* with the
admissible lower bound ``missing parities + differing rows whose target is
not a missing parity`` (a solution is optimal for the region), regions of 5
or more wires by a layer-wise beam search (heuristic; up to 16 wires: large
linear maps such as all-to-all CX layers, ZZ layers, Toffoli arithmetic on
dense couplings).  Routed SWAPs are linear maps, so they merge with the
surrounding CNOT-phase logic (e.g. a Toffoli next to a SWAP).

The rounds run in the native extension (``_phase_poly_run``, see
``phase_poly.hpp``); the Python implementation below is the reference (same
results) and the fallback without the extension.

**Couplings.**  By default only qubit pairs that already carry a two-qubit
gate in the circuit are used (they are couplings of the routed circuit);
``edges`` adds further allowed pairs.

Every replacement is verified on its emitted gates: up to 6 wires the
``2^k × 2^k`` unitaries of the old and new region must agree up to global
phase with ``max|U − e^{iφ}V| ≤ 1e-9`` (phase-aligned, element-wise);
larger regions must have the same linear map, constants and phase
polynomial recomputed from the emitted gates and, up to 14 wires, agree on
two pseudo-random states; otherwise the region is kept.  The output uses
``cz``, ``h``, ``rz`` and ``x``; merge the single-qubit runs afterwards (e.g.
with :class:`OptimizePass`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType

__all__ = ["PhasePolynomialPass", "PhasePolynomialStats"]

_VERIFY_ATOL = 1e-9
_MONO_ATOL = 1e-10
_ZERO_ANGLE = 1e-12
_TWO_PI = 2.0 * math.pi
# 区域线数：≤ _EXACT_WIRES 用 A*（最优）；≥ _BEAM_MIN_WIRES 用束搜索
# （5–6 根线两者都做，取短的）；最多 _MAX_WIRES 根线
_EXACT_WIRES = 6
_BEAM_MIN_WIRES = 5
_MAX_WIRES = 16
# 验证：≤ _DENSE_WIRES 比较整个酉矩阵；更大的区域做符号验证，
# ≤ _STATE_WIRES 时再做两个随机态的数值验证（与原生实现相同）
_DENSE_WIRES = 6
_STATE_WIRES = 14
_SQRT_HALF = 0.7071067811865476  # 1/√2，正确舍入

# 2×2 复矩阵 [[a, b], [c, d]] 记为 8 个实数
# (a.re, a.im, b.re, b.im, c.re, c.im, d.re, d.im)，用逐次舍入的 Python 浮点运算
# 相乘（不经 BLAS、不会被收缩成 FMA），三角函数取确定性数值层的正确舍入实现：
# 段矩阵与发出的旋转角在所有平台上逐位相同。None 表示不透明的段（不是单项式）。
_M = tuple[float, float, float, float, float, float, float, float]
_H: _M = (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0, _SQRT_HALF, 0.0, -_SQRT_HALF, 0.0)
_I2: _M = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_X: _M = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)


@dataclass
class PhasePolynomialStats:
    """Counters of the last :meth:`PhasePolynomialPass.run`.

    Attributes:
        regions: CNOT-phase regions with at least two two-qubit gates.
        replaced: Regions replaced by a cheaper network.
        saved: Two-qubit gates removed.
        rounds: Rounds run (a round re-reads the improved circuit).
    """

    regions: int = 0
    replaced: int = 0
    saved: int = 0
    rounds: int = 0


@dataclass
class _Run:
    """Single-qubit gates between two-qubit gates on one wire."""

    qubit: int
    gates: list[CircuitInstruction] = field(default_factory=list)
    matrix: _M | None = _I2


# Item kinds of the oriented circuit.
_U, _CX, _CZ = 0, 1, 2


def _gate_ins(gate: Any, qubits: tuple[int, ...]) -> CircuitInstruction:
    return CircuitInstruction(type=InstructionType.GATE, gate=gate, qubits=qubits)


_H_INS: dict[int, CircuitInstruction] = {}


def _h(q: int) -> CircuitInstruction:
    ins = _H_INS.get(q)
    if ins is None:
        from quspect.gates.standard import H

        ins = _H_INS[q] = _gate_ins(H, (q,))
    return ins


def _mm(x: _M | None, y: _M | None) -> _M | None:
    """``x @ y`` (``None`` if either is opaque)."""
    if x is None or y is None:
        return None
    ar, ai, br, bi, cr, ci, dr, di = x
    er, ei, fr, fi, gr, gi, hr, hi = y
    return (
        (ar * er - ai * ei) + (br * gr - bi * gi),
        (ar * ei + ai * er) + (br * gi + bi * gr),
        (ar * fr - ai * fi) + (br * hr - bi * hi),
        (ar * fi + ai * fr) + (br * hi + bi * hr),
        (cr * er - ci * ei) + (dr * gr - di * gi),
        (cr * ei + ci * er) + (dr * gi + di * gr),
        (cr * fr - ci * fi) + (dr * hr - di * hi),
        (cr * fi + ci * fr) + (dr * hi + di * hr),
    )


_ATOL2 = _MONO_ATOL * _MONO_ATOL


def _small(re: float, im: float) -> bool:
    return re * re + im * im <= _ATOL2


def _monomial(m: _M | None) -> int:
    """0: diagonal, 1: anti-diagonal, -1: neither (or opaque)."""
    if m is None:
        return -1
    if _small(m[2], m[3]) and _small(m[4], m[5]):
        return 0
    if _small(m[0], m[1]) and _small(m[6], m[7]):
        return 1
    return -1


def _relative_phase(u: tuple[float, float], v: tuple[float, float]) -> float:
    """``arg(v / u)`` = ``arg(v · conj(u))``, deterministic."""
    re = v[0] * u[0] + v[1] * u[1]
    im = v[1] * u[0] - v[0] * u[1]
    return _det("atan2", im, re)


_DET_CACHE: dict[tuple[str, float, float], float] = {}


def _det(name: str, x: float, y: float = 0.0) -> float:
    """Correctly rounded ``name(x[, y])`` of the deterministic numerics layer."""
    key = (name, x, y)
    hit = _DET_CACHE.get(key)
    if hit is not None:
        return hit
    from quspect._native_loader import load_core

    core = load_core()
    fn = None if core is None else getattr(core, "_detmath_scalar", None)
    if fn is None:  # pragma: no cover - extension missing
        val = math.atan2(x, y) if name == "atan2" else getattr(math, name)(x)
    elif name == "atan2":
        val = float(fn(name, np.array([x]), np.array([y]))[0])
    else:
        val = float(fn(name, np.array([x]))[0])
    if len(_DET_CACHE) > 100_000:
        _DET_CACHE.clear()
    _DET_CACHE[key] = val
    return val


_FIXED: dict[str, _M] = {
    "id": _I2,
    "h": _H,
    "x": _X,
    "y": (0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0),
    "z": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "s": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "sdg": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    "t": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, _SQRT_HALF, _SQRT_HALF),
    "tdg": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, _SQRT_HALF, -_SQRT_HALF),
    "sx": (0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5),
    "sxdg": (0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5),
}


def _gate_m(gate: Any) -> _M | None:
    """Deterministic matrix of a single-qubit gate (``None``: opaque / symbolic)."""
    fixed = _FIXED.get(gate.name)
    if fixed is not None:
        return fixed if not gate.params else None
    params = gate.params
    if not all(isinstance(p, Real) for p in params):
        return None
    vals = [float(p) for p in params]
    name = gate.name
    if name == "rz" and len(vals) == 1:
        c, sn = _det("cos", 0.5 * vals[0]), _det("sin", 0.5 * vals[0])
        return (c, -sn, 0.0, 0.0, 0.0, 0.0, c, sn)
    if name in ("p", "u1") and len(vals) == 1:
        return (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            _det("cos", vals[0]),
            _det("sin", vals[0]),
        )
    if name == "rx" and len(vals) == 1:
        c, sn = _det("cos", 0.5 * vals[0]), _det("sin", 0.5 * vals[0])
        return (c, 0.0, 0.0, -sn, 0.0, -sn, c, 0.0)
    if name == "ry" and len(vals) == 1:
        c, sn = _det("cos", 0.5 * vals[0]), _det("sin", 0.5 * vals[0])
        return (c, 0.0, -sn, 0.0, sn, 0.0, c, 0.0)
    return None


class PhasePolynomialPass:
    """Resynthesize CNOT-phase regions with the fewest CX on the couplings.

    See the module docstring.  The pass never increases the two-qubit gate
    count; when nothing improves, :meth:`run` returns its input object.

    Args:
        max_qubits: Largest number of wires in a region (2..16).  Regions of
            at most 6 wires are solved exactly (A*), larger ones (and 5–6
            wire regions the A* budget does not settle) by a beam search.
        node_budget: A* expansions per region before giving up.
        max_rounds: Maximum number of improve-and-reread rounds.
        settings: Several ``(max_qubits, node_budget[, reverse])``
            partitions to try in every round (the orientation is shared);
            the one with the fewest two-qubit gates is kept.  ``reverse``
            grows the regions from the end of the circuit instead of the
            front.  Overrides ``max_qubits`` / ``node_budget``.
        beam_width: Beam width of the heuristic search for large regions.
        native: Run the rounds in the native extension (same results as
            the Python reference implementation, which is used when the
            extension is missing or ``native=False``).
        edges: Extra allowed qubit pairs (unordered) besides those used by
            two-qubit gates of the circuit.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().cx(0, 1).t(1).cx(0, 1).cx(1, 0).cx(0, 1)
        >>> out = PhasePolynomialPass().run(qc)
        >>> sum(1 for i in out if len(i.qubits) == 2) < 4
        True
    """

    def __init__(
        self,
        *,
        max_qubits: int = 4,
        node_budget: int = 200_000,
        max_rounds: int = 3,
        edges: Iterable[tuple[int, int]] = (),
        settings: Sequence[tuple[int, ...]] = (),
        beam_width: int = 64,
        native: bool = True,
    ) -> None:
        self.settings: tuple[tuple[int, int, bool], ...] = tuple(
            (int(s[0]), int(s[1]), bool(s[2]) if len(s) > 2 else False)
            for s in (tuple(settings) or ((max_qubits, node_budget),))
        )
        if not all(2 <= cap <= _MAX_WIRES for cap, _, _ in self.settings):
            raise ValueError(f"max_qubits must be in 2..{_MAX_WIRES}.")
        self.beam_width = beam_width
        self.native = native
        self.max_rounds = max_rounds
        self.extra_edges: set[tuple[int, int]] = {
            (min(a, b), max(a, b)) for a, b in edges
        }
        self.stats = PhasePolynomialStats()

    # ------------------------------------------------------------------
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return ``circuit`` with cheaper CNOT-phase regions (or ``circuit`` itself).

        Args:
            circuit: A circuit of single-qubit gates and ``cz`` / ``cx``.
                Circuits with other instructions (measurements, barriers,
                other multi-qubit gates, symbolic parameters) are returned
                unchanged.

        Returns:
            The rewritten circuit, equal to ``circuit`` up to global phase.
        """
        self.stats = PhasePolynomialStats()
        if self.native:
            out = self._run_native(circuit)
            if out is not None:
                return out
        current = circuit
        for _ in range(self.max_rounds):
            new = self._round(current)
            self.stats.rounds += 1
            if new is None:
                break
            current = new
        return current

    # ------------------------------------------------------------------
    def _run_native(self, circuit: QuantumCircuit) -> QuantumCircuit | None:
        """All rounds in ``_phase_poly_run``; ``None`` if it is unavailable."""
        from quspect._native_loader import load_core

        core = load_core()
        fn = None if core is None else getattr(core, "_phase_poly_run", None)
        if fn is None:  # pragma: no cover - extension missing
            return None
        kinds: list[int] = []
        qa: list[int] = []
        qb: list[int] = []
        hint: list[int] = []
        index: list[int] = []
        mats: list[_M] = []
        opaque: list[int] = []
        singles: list[CircuitInstruction] = []
        n_two = 0
        for ins in circuit:
            gate = ins.gate
            if not ins.is_gate or gate is None:
                return circuit
            qs = ins.qubits
            if len(qs) == 1:
                m = _gate_m(gate)
                kinds.append(0)
                qa.append(qs[0])
                qb.append(-1)
                hint.append(-1)
                index.append(len(singles))
                singles.append(ins)
                mats.append(_I2 if m is None else m)
                opaque.append(1 if m is None else 0)
            elif len(qs) == 2 and gate.name in ("cz", "cx"):
                a, b = qs
                n_two += 1
                if gate.name == "cx":
                    kinds += [1, 2, 1]
                    qa += [b, a, b]
                    qb += [-1, b, -1]
                    hint += [-1, 0, -1]
                    index += [-1, -1, -1]
                else:
                    kinds.append(2)
                    qa.append(a)
                    qb.append(b)
                    hint.append(-1)
                    index.append(-1)
            else:
                return circuit
        if n_two < 2:
            self.stats.rounds = 1
            return circuit
        res, st = fn(
            kinds,
            qa,
            qb,
            hint,
            index,
            mats,
            opaque,
            sorted(self.extra_edges),
            list(self.settings),
            self.beam_width,
            self.max_rounds,
        )
        self.stats.regions, self.stats.replaced, self.stats.saved, self.stats.rounds = (
            st
        )
        if res is None:
            return circuit
        from quspect.gates.parameterized import RZ
        from quspect.gates.standard import CZ, X

        out: list[CircuitInstruction] = []
        for k, a, b, i, ang in zip(*res, strict=True):
            if k == 0:
                out.append(singles[i])
            elif k == 1:
                out.append(_h(a))
            elif k == 2:
                out.append(_gate_ins(CZ, (a, b)))
            elif k == 3:
                out.append(_gate_ins(RZ.with_params(ang), (a,)))
            else:
                out.append(_gate_ins(X, (a,)))
        return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)

    # ------------------------------------------------------------------
    def _round(self, circuit: QuantumCircuit) -> QuantumCircuit | None:
        # 1. 读入：单比特门（带矩阵）与 cz（cx 记为 h·cz·h）
        items: list[tuple[Any, ...]] = []
        pairs: set[tuple[int, int]] = set(self.extra_edges)
        for ins in circuit:
            gate = ins.gate
            if not ins.is_gate or gate is None:
                return None
            qs = ins.qubits
            if len(qs) == 1:
                items.append((_U, qs[0], ins, _gate_m(gate)))
            elif len(qs) == 2 and gate.name in ("cz", "cx"):
                a, b = qs
                pairs.add((min(a, b), max(a, b)))
                if gate.name == "cx":
                    items.append((_U, b, _h(b), _H))
                    items.append((_CZ, a, b, 0))  # 提示：读作 cx(a→b)
                    items.append((_U, b, _h(b), _H))
                else:
                    items.append((_CZ, a, b, -1))
            else:
                return None
        n_two = sum(1 for it in items if it[0] == _CZ)
        if n_two < 2:
            return None

        # 2. 各线上的单比特段与两比特门的相邻关系
        runs: list[_Run] = []
        gate_qubits: list[tuple[int, int]] = []
        hints: list[int] = []
        seq: list[tuple[int, int]] = []  # (kind, index) in order: kind 0 run, 1 gate
        open_run: dict[int, int] = {}
        # run_prev[g][k] / run_next[g][k]: run before / after gate g on its k-th qubit
        run_prev: list[list[int]] = []
        run_next: list[list[int]] = []
        last_gate: dict[int, tuple[int, int]] = {}
        run_gate_before: list[tuple[int, int] | None] = []
        run_gate_after: list[tuple[int, int] | None] = []

        def run_of(q: int) -> int:
            r = open_run.get(q)
            if r is None:
                r = len(runs)
                runs.append(_Run(q))
                run_gate_before.append(last_gate.get(q))
                run_gate_after.append(None)
                open_run[q] = r
                seq.append((0, r))
            return r

        for it in items:
            if it[0] == _U:
                r = run_of(it[1])
                runs[r].gates.append(it[2])
                runs[r].matrix = _mm(it[3], runs[r].matrix)
                continue
            g = len(gate_qubits)
            a, b = it[1], it[2]
            gate_qubits.append((a, b))
            hints.append(it[3])
            rp = [run_of(a), run_of(b)]
            for k, q in enumerate((a, b)):
                run_gate_after[rp[k]] = (g, k)
                del open_run[q]
                last_gate[q] = (g, k)
            run_prev.append(rp)
            run_next.append([-1, -1])
            seq.append((1, g))
        for r, (gk) in enumerate(run_gate_before):
            if gk is not None:
                run_next[gk[0]][gk[1]] = r

        # 3. 朝向：o[g] ∈ {0: 目标 b, 1: 目标 a, 2: 对角}
        def is_target(o: int, k: int) -> int:
            return 1 if (o == 0 and k == 1) or (o == 1 and k == 0) else 0

        # 每段的 4 种 (前门目标?, 后门目标?) 是否单项式
        mono_table: list[tuple[bool, bool, bool, bool] | None] = []
        for r, run in enumerate(runs):
            if run_gate_before[r] is None or run_gate_after[r] is None:
                mono_table.append(None)  # 线首 / 线尾段：不计
                continue
            m = run.matrix
            hm = _mm(_H, m)
            mh = _mm(m, _H)
            hmh = _mm(hm, _H)
            mono_table.append(
                (
                    _monomial(m) >= 0,
                    _monomial(mh) >= 0,  # 前门目标：段前多一个 H
                    _monomial(hm) >= 0,  # 后门目标：段后多一个 H
                    _monomial(hmh) >= 0,
                )
            )

        orient = [0] * len(gate_qubits)

        def run_cost(r: int) -> int:
            tab = mono_table[r]
            if tab is None:
                return 0
            gb, kb = run_gate_before[r]  # type: ignore[misc]
            ga, ka = run_gate_after[r]  # type: ignore[misc]
            tp = is_target(orient[gb], kb)
            tn = is_target(orient[ga], ka)
            return 0 if tab[tp + 2 * tn] else 1

        def local_cost(g: int) -> int:
            return sum(run_cost(r) for r in (*run_prev[g], *run_next[g]) if r >= 0)

        for g in range(len(gate_qubits)):
            best_o, best_c = 0, None
            for o in ((hints[g],) if hints[g] >= 0 else ()) + (0, 1, 2):
                orient[g] = o
                c = local_cost(g)
                if best_c is None or c < best_c:
                    best_o, best_c = o, c
            orient[g] = best_o
        # 局部搜索：一个门的朝向只影响它两侧的段；某门改变后，只需重看与这些段相邻的门
        neighbours: list[list[int]] = []
        for g in range(len(gate_qubits)):
            near = []
            for r in (*run_prev[g], *run_next[g]):
                if r < 0:
                    continue
                for gk in (run_gate_before[r], run_gate_after[r]):
                    if gk is not None and gk[0] != g:
                        near.append(gk[0])
            neighbours.append(near)
        pending = list(range(len(gate_qubits)))
        queued = [True] * len(gate_qubits)
        steps = 0
        limit = 8 * len(gate_qubits)
        while pending and steps < limit:
            g = pending.pop()
            queued[g] = False
            steps += 1
            cur = orient[g]
            best_c = local_cost(g)
            best_o = cur
            for o in (0, 1, 2):
                if o == cur:
                    continue
                orient[g] = o
                c = local_cost(g)
                if c < best_c:
                    best_o, best_c = o, c
            orient[g] = best_o
            if best_o != cur:
                for h in neighbours[g]:
                    if not queued[h]:
                        queued[h] = True
                        pending.append(h)

        # 4. 定向后的序列：段（含插入的 H）与 cx / cz
        oriented: list[tuple[Any, ...]] = []
        pre_h: dict[int, bool] = {}
        for kind, idx in seq:
            if kind == 0:
                run = runs[idx]
                gates = list(run.gates)
                mat = run.matrix
                if pre_h.pop(run.qubit, False):
                    gates.insert(0, _h(run.qubit))
                    mat = _mm(mat, _H)
                gb = run_gate_after[idx]
                if gb is not None and is_target(orient[gb[0]], gb[1]):
                    gates.append(_h(run.qubit))
                    mat = _mm(_H, mat)
                oriented.append((_U, run.qubit, gates, mat))
                continue
            a, b = gate_qubits[idx]
            o = orient[idx]
            if o == 2:
                oriented.append((_CZ, a, b))
            else:
                c, t = (a, b) if o == 0 else (b, a)
                oriented.append((_CX, c, t))
                if run_next[idx][0 if t == a else 1] >= 0:
                    pre_h[t] = True
                else:  # 线尾：补一个 H
                    oriented.append((_U, t, [_h(t)], _H))

        # 5. 区域与综合：每种划分各做一次，取两比特门最少的
        best: tuple[int, list[CircuitInstruction], int, int] | None = None
        for cap, budget, reverse in self.settings:
            result = self._resynthesize(oriented, pairs, cap, budget, reverse)
            if result is not None and (best is None or result[0] < best[0]):
                best = result
        if best is None:
            return None
        _, out, replaced, saved = best
        self.stats.replaced += replaced
        self.stats.saved += saved
        return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)

    def _resynthesize(
        self,
        oriented: list[tuple[Any, ...]],
        pairs: set[tuple[int, int]],
        cap: int,
        budget: int,
        reverse: bool = False,
    ) -> tuple[int, list[CircuitInstruction], int, int] | None:
        """(two-qubit count, gates, regions replaced, saved) or ``None``."""
        regions = _split_regions(oriented, cap, reverse=reverse)
        problems: list[tuple[Any, ...]] = []
        meta: list[tuple[int, dict[int, float], list[int], int]] = []
        for ridx, (wires, reg) in enumerate(regions):
            if reg is None:
                continue
            info = _phase_polynomial(wires, reg)
            if info is None:
                continue
            rows, consts, terms, n2 = info
            if n2 < 2:
                continue
            self.stats.regions += 1
            local = {q: i for i, q in enumerate(wires)}
            edges = [
                (local[a], local[b])
                for a, b in sorted(pairs)
                if a in local and b in local
            ]
            edges = edges + [(b, a) for a, b in edges]
            k = len(wires)
            base = (k, edges, rows, sorted(terms), n2 - 1, 0)
            # ≤ 6 根线：A*（最优）；5 根线起再加一次束搜索（A* 预算用完时的
            # 后备）；更大：只用束搜索
            if k <= _EXACT_WIRES:
                problems.append((*base, 0, budget))
                meta.append((ridx, terms, consts, n2))
            if k >= _BEAM_MIN_WIRES:
                problems.append((*base, self.beam_width, 0))
                meta.append((ridx, terms, consts, n2))
        if not problems:
            return None
        solutions = _solve(problems)
        best_sol: dict[int, list[tuple[int, int]]] = {}
        for (ridx, _, _, n2), sol in zip(meta, solutions, strict=True):
            if sol is None or len(sol[0]) >= n2:
                continue
            if ridx not in best_sol or len(sol[0]) < len(best_sol[ridx]):
                best_sol[ridx] = list(sol[0])
        info_of = {m[0]: m for m in meta}
        replaced: dict[int, list[tuple[Any, ...]]] = {}
        saved = 0
        for ridx, moves in sorted(best_sol.items()):
            _, terms, consts, n2 = info_of[ridx]
            wires, reg = regions[ridx]
            assert reg is not None
            new = _emit(wires, moves, terms, consts)
            if not _verify(wires, reg, new):
                continue
            replaced[ridx] = new
            saved += n2 - len(moves)
        if not replaced:
            return None

        out: list[CircuitInstruction] = []
        n_two = 0
        from quspect.gates.parameterized import RZ
        from quspect.gates.standard import CZ, X

        def emit(items_: Sequence[tuple[Any, ...]]) -> None:
            nonlocal n_two
            for it in items_:
                if it[0] == _U:
                    out.extend(it[2])
                elif it[0] == _CZ:
                    out.append(_gate_ins(CZ, (it[1], it[2])))
                    n_two += 1
                elif it[0] == _CX:
                    out.append(_h(it[2]))
                    out.append(_gate_ins(CZ, (it[1], it[2])))
                    out.append(_h(it[2]))
                    n_two += 1
                elif it[0] == "rz":
                    out.append(_gate_ins(RZ.with_params(it[2]), (it[1],)))
                elif it[0] == "x":
                    out.append(_gate_ins(X, (it[1],)))

        for ridx, (wires, reg) in enumerate(regions):
            if reg is None:
                emit([wires])  # 单个项
            elif ridx in replaced:
                emit(replaced[ridx])
            else:
                emit(reg)
        return n_two, out, len(replaced), saved


def _split_regions(
    oriented: list[tuple[Any, ...]], cap: int, *, reverse: bool = False
) -> list[tuple[Any, list[tuple[Any, ...]] | None]]:
    """Split into ``(item, None)`` and ``(wires, items)`` (regions), in order.

    ``reverse`` grows the regions from the end of the circuit (the regions
    are convex either way, so reading the result backwards gives a valid
    split in circuit order).
    """
    if reverse:
        back = _split_regions(oriented[::-1], cap)
        return [
            (wires, None if reg is None else reg[::-1]) for wires, reg in back[::-1]
        ]
    out: list[tuple[Any, list[tuple[Any, ...]] | None]] = []
    rest = oriented
    while rest:
        pre, wires, reg, post = _take_region(rest, cap)
        out.extend((it, None) for it in pre)
        if not reg:
            out.extend((it, None) for it in post)
            break
        out.append((wires, reg))
        rest = post
    return out


def _take_region(
    items: list[tuple[Any, ...]], cap: int
) -> tuple[
    list[tuple[Any, ...]], list[int], list[tuple[Any, ...]], list[tuple[Any, ...]]
]:
    """Grow one convex region from the front of ``items``.

    Returns ``(pre, wires, region, post)``: ``pre`` (items on wires the region
    never reaches) go before the region, ``post`` after it, both in order.
    ``region`` is empty if no two-qubit gate could start one.
    """
    wires: list[int] = []
    inside: set[int] = set()
    closed: set[int] = set()
    reg: list[tuple[Any, ...]] = []
    pre: list[tuple[Any, ...]] = []
    post: list[tuple[Any, ...]] = []
    n_open = 0  # 区域内尚未关闭的线数
    for pos, it in enumerate(items):
        qs = (it[1],) if it[0] == _U else (it[1], it[2])
        to_close: tuple[int, ...] = ()
        if closed.intersection(qs):
            post.append(it)
            to_close = qs
        elif it[0] == _U:
            if it[1] not in inside:
                pre.append(it)
            elif _monomial(it[3]) >= 0:
                reg.append(it)
            else:
                post.append(it)
                to_close = qs
        else:
            new = [q for q in qs if q not in inside]
            if len(inside) + len(new) <= cap:
                for q in new:
                    inside.add(q)
                    wires.append(q)
                    n_open += 1
                reg.append(it)
            else:
                post.append(it)
                to_close = qs
        for q in to_close:
            if q not in closed:
                closed.add(q)
                if q in inside:
                    n_open -= 1
        if reg and n_open == 0:
            # 区域的线都已关闭：之后的项与本区域无关，原样留给下一轮
            post.extend(items[pos + 1 :])
            break
    return pre, wires, reg, post


def _phase_polynomial(
    wires: Sequence[int], reg: Sequence[tuple[Any, ...]]
) -> tuple[list[int], list[int], dict[int, float], int] | None:
    """(final rows, constants, {parity: angle}, two-qubit gate count) of a region."""
    local = {q: i for i, q in enumerate(wires)}
    rows = [1 << i for i in range(len(wires))]
    consts = [0] * len(wires)
    terms: dict[int, float] = {}
    n2 = 0

    def add(mask: int, const: int, angle: float) -> None:
        terms[mask] = terms.get(mask, 0.0) + (-angle if const else angle)

    for it in reg:
        if it[0] == "rz":
            i = local[it[1]]
            add(rows[i], consts[i], it[2])
            continue
        if it[0] == "x":
            consts[local[it[1]]] ^= 1
            continue
        if it[0] == _U:
            i = local[it[1]]
            m = it[3]
            kind = _monomial(m)
            if kind == 0:
                add(rows[i], consts[i], _relative_phase((m[0], m[1]), (m[6], m[7])))
            elif kind == 1:  # x · diag(m10, m01)
                add(rows[i], consts[i], _relative_phase((m[4], m[5]), (m[2], m[3])))
                consts[i] ^= 1
            else:
                return None
        elif it[0] == _CX:
            c, t = local[it[1]], local[it[2]]
            rows[t] ^= rows[c]
            consts[t] ^= consts[c]
            n2 += 1
        else:  # 对角 cz：π·va·vb = π/2·(va + vb − va⊕vb)
            a, b = local[it[1]], local[it[2]]
            add(rows[a], consts[a], 0.5 * math.pi)
            add(rows[b], consts[b], 0.5 * math.pi)
            add(rows[a] ^ rows[b], consts[a] ^ consts[b], -0.5 * math.pi)
            n2 += 1
    kept: dict[int, float] = {}
    for mask, angle in terms.items():
        reduced = math.remainder(angle, _TWO_PI)
        if abs(reduced) > _ZERO_ANGLE:
            kept[mask] = reduced
    return rows, consts, kept, n2


#: Solutions by problem: the same regions recur across rounds and across the
#: routing candidates of one compilation (bounded, cleared when full).
_SOLVE_CACHE: dict[tuple[Any, ...], Any] = {}
_SOLVE_CACHE_MAX = 200_000


def _solve(problems: list[tuple[Any, ...]]) -> list[Any]:
    """Solve ``(n, edges, rows, terms, max_cx, free_mask, beam, budget)`` problems.

    Returns ``(moves, perm)`` or ``None`` per problem (see
    ``_cnot_phase_synth_batch``).
    """
    from quspect._native_loader import load_core

    keys = [
        (n, tuple(edges), tuple(rows), tuple(terms), max_cx, free, beam, budget)
        for n, edges, rows, terms, max_cx, free, beam, budget in problems
    ]
    todo = [i for i, k in enumerate(keys) if k not in _SOLVE_CACHE]
    if todo:
        core = load_core()
        fn = None if core is None else getattr(core, "_cnot_phase_synth_batch", None)
        if fn is None:  # pragma: no cover - extension missing
            return [_SOLVE_CACHE.get(k) for k in keys]
        found: list[Any] = fn([problems[i] for i in todo])
        if len(_SOLVE_CACHE) + len(todo) > _SOLVE_CACHE_MAX:
            _SOLVE_CACHE.clear()
        for i, sol in zip(todo, found, strict=True):
            _SOLVE_CACHE[keys[i]] = sol
    return [_SOLVE_CACHE[k] for k in keys]


def _emit(
    wires: Sequence[int],
    moves: Sequence[tuple[int, int]],
    terms: dict[int, float],
    consts: Sequence[int],
) -> list[tuple[Any, ...]]:
    rows = [1 << i for i in range(len(wires))]
    out: list[tuple[Any, ...]] = []
    placed: set[int] = set()

    def place(i: int) -> None:
        m = rows[i]
        if m in terms and m not in placed:
            placed.add(m)
            out.append(("rz", wires[i], terms[m]))

    for i in range(len(wires)):
        place(i)
    for c, t in moves:
        rows[t] ^= rows[c]
        out.append((_CX, wires[c], wires[t]))
        place(t)
    for i, c in enumerate(consts):
        if c:
            out.append(("x", wires[i]))
    return out


def _apply(state: np.ndarray, local: dict[int, int], it: tuple[Any, ...]) -> None:
    """Left-multiply ``state`` (``2^k × 2^k``, qubit i = bit i) by one item."""
    dim = state.shape[0]
    idx = np.arange(dim)
    kind = it[0]
    if kind in (_U, "rz", "x"):
        q = local[it[1]]
        if kind == _U:
            v = it[3]
            m = np.array(
                [
                    [complex(v[0], v[1]), complex(v[2], v[3])],
                    [complex(v[4], v[5]), complex(v[6], v[7])],
                ]
            )
        elif kind == "rz":
            c, sn = _det("cos", 0.5 * it[2]), _det("sin", 0.5 * it[2])
            m = np.array([[complex(c, -sn), 0], [0, complex(c, sn)]])
        else:
            m = np.array([[0, 1], [1, 0]], dtype=complex)
        bit = (idx >> q) & 1
        partner = idx ^ (1 << q)
        new = m[bit, bit][:, None] * state + m[bit, 1 - bit][:, None] * state[partner]
        state[:] = new
    elif kind == _CX:
        c, t = local[it[1]], local[it[2]]
        src = np.where((idx >> c) & 1, idx ^ (1 << t), idx)
        state[:] = state[src]
    else:  # _CZ
        a, b = local[it[1]], local[it[2]]
        sign = np.where(((idx >> a) & 1) & ((idx >> b) & 1), -1.0, 1.0)
        state *= sign[:, None]


def _unitary(wires: Sequence[int], items: Iterable[tuple[Any, ...]]) -> np.ndarray:
    k = len(wires)
    local = {q: i for i, q in enumerate(wires)}
    state = np.eye(1 << k, dtype=complex)
    for it in items:
        _apply(state, local, it)
    return state


def _verify(
    wires: Sequence[int], old: Sequence[tuple[Any, ...]], new: Sequence[tuple[Any, ...]]
) -> bool:
    """Check that the new region equals the old one up to global phase.

    Up to ``_DENSE_WIRES`` wires the two ``2^k × 2^k`` unitaries are compared
    (phase-aligned, element-wise).  Larger regions are checked symbolically
    (the phase polynomial, linear map and constants recomputed from the
    emitted gates must equal those of the old region) and, up to
    ``_STATE_WIRES`` wires, also numerically on two pseudo-random states.
    Uses only exactly rounded arithmetic (squared magnitudes, ``math.sqrt``),
    no platform math functions.
    """
    k = len(wires)
    if k > _DENSE_WIRES:
        if not _same_phase_polynomial(wires, old, new):
            return False
        if k > _STATE_WIRES:
            return True
        rng = np.random.default_rng(20261107)
        start = rng.standard_normal((1 << k, 2)) + 1j * rng.standard_normal((1 << k, 2))
        a = start.copy()
        b = start
        local = {q: i for i, q in enumerate(wires)}
        for it in old:
            _apply(a, local, it)
        for it in new:
            _apply(b, local, it)
        return _phase_aligned_close(a, b)
    a = _unitary(wires, old)
    b = _unitary(wires, new)
    return _phase_aligned_close(a, b)


def _same_phase_polynomial(
    wires: Sequence[int], old: Sequence[tuple[Any, ...]], new: Sequence[tuple[Any, ...]]
) -> bool:
    """Symbolic check: equal linear map, constants and phase polynomial (mod 2π)."""
    a = _phase_polynomial(wires, old)
    b = _phase_polynomial(wires, new)
    if a is None or b is None or a[0] != b[0] or a[1] != b[1]:
        return False
    ta, tb = a[2], b[2]
    if ta.keys() != tb.keys():
        return False
    return all(abs(math.remainder(ta[m] - tb[m], _TWO_PI)) <= _VERIFY_ATOL for m in ta)


def _phase_aligned_close(a: np.ndarray, b: np.ndarray) -> bool:
    """``max|a − e^{iφ} b|² ≤ atol²`` with ``φ`` from the largest entry of ``a``."""
    mag = a.real * a.real + a.imag * a.imag
    k = int(np.argmax(mag))
    i, j = divmod(k, a.shape[1])
    bij = complex(b[i, j])
    if bij.real * bij.real + bij.imag * bij.imag < 1e-12:
        return False
    aij = complex(a[i, j])
    # e^{iφ} = a_ij · conj(b_ij) / |a_ij · conj(b_ij)|
    re = aij.real * bij.real + aij.imag * bij.imag
    im = aij.imag * bij.real - aij.real * bij.imag
    norm = math.sqrt(re * re + im * im)
    phase = complex(re / norm, im / norm)
    diff = a - phase * b
    err = diff.real * diff.real + diff.imag * diff.imag
    return bool(np.max(err) <= _VERIFY_ATOL * _VERIFY_ATOL)
