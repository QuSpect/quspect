"""Frozen copy of the round-12 single-qubit minimization (``a329090``).

Reference implementation for the differential tests in
``test_gauge_differential.py``: the batched / native beam search in
:mod:`quspect.transpiler._gauge` must reproduce
:func:`legacy_minimize_single_qubit_gates` bit for bit (same gates, same
parameters, same qubits, same order).

Copied verbatim from ``quspect/transpiler/_gauge.py``; only the public
function was renamed.  Do not edit.
"""
# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate

__all__ = ["legacy_minimize_single_qubit_gates"]

_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
#: ``V`` with ``V·Z·V† = P`` for each Pauli axis ``P``.
_TO_Z = {
    "z": _I2,
    "x": _H,
    "y": np.array([[1, 0], [0, 1j]], dtype=complex) @ _H,
}
_SWAP_AXIS = {"z": "x", "x": "z"}
_SQRT_HALF = math.sqrt(0.5)
_HALF_PI = math.pi / 2.0
_TWO_PI = 2.0 * math.pi

#: Beam width of the per-wire search.
_BEAM = 4
#: Runs on each side of a reversed CX that are re-optimized.
_WINDOW = 0
#: Orientation local-search sweeps.
_SWEEPS = 2
#: Score of a run: ``_GATE_WEIGHT`` per gate plus ``_RUN_WEIGHT`` if it is
#: not empty (a new run must save more than half a gate).
_GATE_WEIGHT = 2
_RUN_WEIGHT = 1
#: Cost of a run that cannot be expressed (never chosen).
_INF = 10**6


def _rot(axis: str, phi: float) -> np.ndarray:
    """``exp(-i·phi/2·P)`` for the Pauli ``P`` named ``axis``."""
    return math.cos(phi / 2.0) * _I2 - 1j * math.sin(phi / 2.0) * _PAULI[axis]


def _leading_angles(m: np.ndarray, axis: str) -> list[float]:
    """Angles ``a`` with ``m ∝ R_axis(a)·rest`` for which ``rest`` is short.

    The leading rotation of the ``axis``-Euler decompositions of ``m`` (a
    rotation about ``axis`` is moved out whole).
    """
    from quspect.gates.synthesis import _zyz_angles

    v = _TO_Z[axis]
    z = v.conj().T @ m @ v  # m in the frame where ``axis`` is Z
    if abs(z[0, 1]) < 1e-12 and abs(z[1, 0]) < 1e-12:
        return [float(np.angle(z[1, 1] / z[0, 0]))]
    _, phi, _ = _zyz_angles(z)
    # z ∝ RZ(phi)·RY(θ)·RZ(λ) and RZ(phi)·RY(θ) = RZ(phi ± π/2)·RX(±θ)·RZ(∓π/2)
    return [phi, phi + _HALF_PI, phi - _HALF_PI, phi + math.pi]


def _axis_commuting(gate_m: np.ndarray, slot: int) -> str | None:
    """Pauli axis whose rotations on ``slot`` commute with a 2q gate."""
    for name in ("z", "x", "y"):
        p = _PAULI[name]
        op = np.kron(p, _I2) if slot == 0 else np.kron(_I2, p)
        if np.max(np.abs(op @ gate_m - gate_m @ op)) < 1e-12:
            return name
    return None


def _nz(theta: float) -> int:
    """1 if ``RZ(theta)`` is not the identity (synthesis tolerance), else 0."""
    t = math.fmod(theta, _TWO_PI)
    if t <= -math.pi:
        t += _TWO_PI
    elif t > math.pi:
        t -= _TWO_PI
    return 0 if abs(t) < 1e-9 or abs(abs(t) - _TWO_PI) < 1e-9 else 1


class _LengthModel:
    """Length of :func:`~quspect.gates.synthesis.synthesize_one_qubit`.

    For ``rz``-based bases with ``h`` and / or ``sx`` it mirrors the
    synthesis candidates without building or verifying sequences (about
    ten times faster); otherwise it calls the synthesis.  Used only to
    *score* choices: emitted runs are always synthesized and verified.
    """

    def __init__(self, basis: frozenset[str]) -> None:
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.standard import SDG, SX, SXDG, TDG, H, S, T, X, Y, Z

        self.basis = basis
        self.fast = (
            "rz" in basis
            and not ({"u3", "rx", "ry"} & basis)
            and bool({"h", "sx"} & basis)
        )
        self.named_entries = [
            tuple(complex(x) for x in np.asarray(gate_matrix(g), dtype=complex).flat)
            for g in (H, X, Y, Z, S, SDG, T, TDG, SX, SXDG)
            if g.name in basis
        ]
        self.h = "h" in basis
        self.x = "x" in basis
        self.sx = "sx" in basis
        self.sxdg = "sxdg" in basis

    def __call__(self, m: np.ndarray) -> int:
        if not self.fast:
            from quspect.gates.synthesis import synthesize_one_qubit

            seq = synthesize_one_qubit(m, self.basis)
            return _INF if seq is None else len(seq)
        return self._fast(m)

    def _fast(self, m: np.ndarray) -> int:
        u00, u01 = complex(m[0, 0]), complex(m[0, 1])
        u10, u11 = complex(m[1, 0]), complex(m[1, 1])
        a00, a10 = abs(u00), abs(u10)
        if a10 < 1e-9 and abs(u01) < 1e-9:
            if abs(u00 - u11) < 1e-9:
                return 0
            return 1  # a diagonal gate: one rz (or a named gate)
        if (a00 < 1e-9 or abs(a00 - _SQRT_HALF) < 1e-9) and self._named(
            u00, u01, u10, u11
        ):
            return 1
        best = _INF
        if a00 < 1e-9 and abs(u11) < 1e-9:
            tail = _nz(cmath.phase(u10 / u01))
            if self.x:
                best = 1 + tail
            elif self.sx:
                best = 2 + tail
            elif self.h:
                best = 3 + tail
        if self.h and abs(a00 - _SQRT_HALF) < 1e-9:
            best = min(
                best, 1 + _nz(cmath.phase(u10 / u00)) + _nz(cmath.phase(u01 / u00))
            )
        theta, phi, lam = _zyz(u00, u01, u10, u11)
        for c, b, a in (
            (lam - _HALF_PI, theta, phi + _HALF_PI),
            (lam + _HALF_PI, -theta, phi - _HALF_PI),
        ):
            ends = _nz(c) + _nz(a)
            if self.h:
                best = min(best, 2 + ends + _nz(b))
            if self.sx and abs(b - _HALF_PI) < 1e-9:
                best = min(best, 1 + ends)
            if self.sxdg and abs(b + _HALF_PI) < 1e-9:
                best = min(best, 1 + ends)
        if self.sx:
            best = min(best, 2 + _nz(lam) + _nz(theta + math.pi) + _nz(phi + math.pi))
        return best

    def _named(self, u00: complex, u01: complex, u10: complex, u11: complex) -> bool:
        """Whether the matrix is one of the basis's named gates (up to phase)."""
        k = max(range(4), key=lambda i: abs((u00, u01, u10, u11)[i]))
        a = (u00, u01, u10, u11)
        for g in self.named_entries:
            gk = g[k]
            if abs(gk) < 0.5 * abs(a[k]):
                continue
            ph = a[k] / gk
            ph /= abs(ph)
            if all(abs(a[i] - ph * g[i]) < 1e-9 for i in range(4)):
                return True
        return False


def _zyz(
    u00: complex, u01: complex, u10: complex, u11: complex
) -> tuple[float, float, float]:
    """Pure-Python copy of the ZYZ (U3) angle extraction used by synthesis."""
    c = min(abs(u00), 1.0)
    s = min(abs(u10), 1.0)
    theta = 2.0 * math.atan2(s, c)
    if abs(u00) >= 1e-10:
        corr = cmath.exp(-1j * cmath.phase(u00))
        u00, u01, u10, u11 = u00 * corr, u01 * corr, u10 * corr, u11 * corr
    if theta < 1e-10:
        phi, lam = 0.0, cmath.phase(u11)
    elif abs(theta - math.pi) < 1e-10:
        phi, lam = cmath.phase(u10), cmath.phase(-u01)
    elif c >= s:
        total = cmath.phase(u11)
        diff = cmath.phase(u10 * (-u01).conjugate())
        phi = 0.5 * (total + diff)
        lam = 0.5 * (total - diff)
        if (cmath.exp(-1j * phi) * u10).real < 0.0:
            phi += math.pi
            lam += math.pi
    else:
        phi, lam = cmath.phase(u10), cmath.phase(-u01)
    return theta, phi % _TWO_PI, lam % _TWO_PI


class _Run:
    __slots__ = ("gates", "matrix", "native")

    def __init__(self) -> None:
        self.gates: list[Gate] = []
        self.matrix = _I2
        self.native = True


class _Anchor:
    """One qubit of a multi-qubit instruction (or a blocking instruction)."""

    __slots__ = ("axis", "gate")

    def __init__(self, axis: str | None, gate: int) -> None:
        self.axis = axis  # commuting axis in the original orientation
        self.gate = gate  # index into the flippable gates, or -1


def legacy_minimize_single_qubit_gates(
    circuit: QuantumCircuit, one_qubit: frozenset[str]
) -> QuantumCircuit:
    """Minimize single-qubit gates around commuting / reversible 2q gates.

    Rotations are moved through two-qubit gates they commute with (on one
    qubit) and CX gates are reversed (``CX(a, b) = (H⊗H)·CX(b, a)·(H⊗H)``)
    where that lowers the number of single-qubit gates needed in
    ``one_qubit``; changed runs are resynthesized with
    :func:`~quspect.gates.synthesis.synthesize_one_qubit` (verified to
    ``1e-9``, phase-aligned).  The two-qubit gates stay the same gates on
    the same qubit pairs and in the same order.  The result never has more
    single-qubit gates than the input.

    Args:
        circuit: Circuit whose single-qubit gates have known matrices or
            are native (symbolic gates are kept and block movement).
        one_qubit: The target's single-qubit gate names.

    Returns:
        An equivalent circuit (up to global phase).

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().rz(0.3, 0).cx(0, 1).rz(0.2, 0).h(0)
        >>> out = legacy_minimize_single_qubit_gates(qc, frozenset({"h", "rz"}))
        >>> [(i.gate.name, i.qubits) for i in out]
        [('cx', (0, 1)), ('rz', (0,)), ('h', (0,))]
    """
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.synthesis import synthesize_one_qubit

    instrs = circuit.instructions
    mat_cache: dict[Gate, np.ndarray | None] = {}

    def matrix_of(gate: Gate) -> np.ndarray | None:
        try:
            return mat_cache[gate]
        except KeyError:
            pass
        except TypeError:
            return None
        try:
            m: np.ndarray | None = np.asarray(gate_matrix(gate), dtype=complex)
        except (KeyError, ValueError, TypeError):
            m = None
        mat_cache[gate] = m
        return m

    axis_cache: dict[Gate, tuple[str | None, str | None]] = {}

    def axes_of(gate: Gate) -> tuple[str | None, str | None]:
        try:
            return axis_cache[gate]
        except KeyError:
            pass
        except TypeError:
            return (None, None)
        m = matrix_of(gate)
        ax = (
            (None, None)
            if m is None or m.shape != (4, 4)
            else (_axis_commuting(m, 0), _axis_commuting(m, 1))
        )
        axis_cache[gate] = ax
        return ax

    # --- split every wire into runs and anchors ---------------------------
    runs: list[_Run] = []
    current: dict[int, int] = {}
    # per wire: run ids r0..rk and anchors a1..ak (anchors[i] follows runs[i])
    wire_runs: dict[int, list[int]] = {}
    wire_anchors: dict[int, list[_Anchor]] = {}
    instr_runs: dict[int, list[tuple[int, int]]] = {}  # instr → [(q, run id)]
    flip_instr: list[int] = []  # flippable gate → instruction index
    flip_pos: list[tuple[tuple[int, int], tuple[int, int]]] = []  # (q, anchor idx)
    all_wires: list[int] = []

    def new_run(q: int) -> int:
        runs.append(_Run())
        rid = current[q] = len(runs) - 1
        if q not in wire_runs:
            wire_runs[q] = []
            wire_anchors[q] = []
            all_wires.append(q)
        wire_runs[q].append(rid)
        return rid

    original_1q = 0
    for idx, ins in enumerate(instrs):
        gate = ins.gate
        if ins.is_gate and gate is not None and len(ins.qubits) == 1:
            m = matrix_of(gate)
            if m is not None:
                q = ins.qubits[0]
                rid = current[q] if q in current else new_run(q)
                run = runs[rid]
                run.gates.append(gate)
                run.matrix = m @ run.matrix
                if gate.name not in one_qubit:
                    run.native = False
                original_1q += 1
                continue
        qubits = list(dict.fromkeys(ins.qubits))
        if not qubits and ins.is_barrier:
            qubits = list(all_wires)
        axes: tuple[str | None, ...] = (None,) * len(qubits)
        flip = -1
        if ins.is_gate and gate is not None and len(ins.qubits) == 2:
            axes = axes_of(gate)
            if gate.name == "cx" and not gate.params:
                flip = len(flip_instr)
                flip_instr.append(idx)
        entries = []
        pos = []
        for q, ax in zip(qubits, axes, strict=True):
            rid = current[q] if q in current else new_run(q)
            entries.append((q, rid))
            wire_anchors[q].append(_Anchor(ax, flip))
            pos.append((q, len(wire_anchors[q]) - 1))
            new_run(q)
        if flip >= 0:
            flip_pos.append((pos[0], pos[1]))
        instr_runs[idx] = entries

    flipped = [False] * len(flip_instr)
    length = _LengthModel(one_qubit)

    def anchor_axis(a: _Anchor) -> str | None:
        if a.gate >= 0 and flipped[a.gate] and a.axis is not None:
            return _SWAP_AXIS.get(a.axis)
        return a.axis

    def run_base(q: int, i: int) -> tuple[np.ndarray, bool]:
        """Run ``i`` of wire ``q`` with the Hadamards of reversed CX."""
        anchors = wire_anchors[q]
        m = runs[wire_runs[q][i]].matrix
        changed = False
        if i > 0:
            a = anchors[i - 1]
            if a.gate >= 0 and flipped[a.gate]:
                m = m @ _H
                changed = True
        if i < len(anchors):
            a = anchors[i]
            if a.gate >= 0 and flipped[a.gate]:
                m = _H @ m
                changed = True
        return m, changed

    def cost_of(run: _Run, m: np.ndarray, changed: bool) -> int:
        c = length(m)
        if not changed and run.native:
            c = min(c, len(run.gates))
        return _GATE_WEIGHT * c + (_RUN_WEIGHT if c else 0)

    def is_flipped(a: _Anchor) -> bool:
        return a.gate >= 0 and flipped[a.gate]

    memo: dict[tuple[int, bool, bool, int, bool, float], list[tuple[float, int]]] = {}

    def run_options(
        q: int,
        i: int,
        phi_in: float,
        ax_in: str | None,
        ax_out: str | None,
        phi_fixed: float | None,
    ) -> list[tuple[float, int]]:
        """``(angle carried out, run cost)`` candidates of run ``i``."""
        run = runs[wire_runs[q][i]]
        m, changed = run_base(q, i)
        if ax_in is not None and phi_in != 0.0:
            m = m @ _rot(ax_in, phi_in)
            changed = True
        if phi_fixed is not None:
            opts = [phi_fixed]
        elif ax_out is None:
            opts = [0.0]
        else:
            opts = [0.0, *_leading_angles(m, ax_out)]
        result = []
        for phi in opts:
            if phi != 0.0 and ax_out is not None:
                mm = _rot(ax_out, -phi) @ m
                result.append((phi, cost_of(run, mm, True)))
            else:
                result.append((phi, cost_of(run, m, changed)))
        return result

    # per wire: angle carried out of each run (about the next anchor's axis)
    outs: dict[int, list[float]] = {}
    costs: dict[int, list[int]] = {}

    def solve(
        q: int, s: int, e: int, phi_s: float, phi_e: float
    ) -> tuple[int, list[float], list[int]]:
        """Best angles for runs ``s..e`` of wire ``q`` with fixed ends."""
        anchors = wire_anchors[q]
        rids = wire_runs[q]
        # beam entries: (total, angle out, parent index, run cost)
        layers: list[list[tuple[int, float, int, int]]] = []
        prev: list[tuple[int, float, int, int]] = [(0, phi_s, -1, 0)]
        for i in range(s, e + 1):
            rid = rids[i]
            fp = i > 0 and is_flipped(anchors[i - 1])
            fn = i < len(anchors) and is_flipped(anchors[i])
            ax_in = anchor_axis(anchors[i - 1]) if i > 0 else None
            ax_out = anchor_axis(anchors[i]) if i < len(anchors) else None
            last = i == e
            nxt: dict[int, tuple[int, float, int, int]] = {}
            for pi, (total, phi_in, _, _) in enumerate(prev):
                key = (rid, fp, fn, round(phi_in * 1e10), last, phi_e if last else 0.0)
                options = memo.get(key)
                if options is None:
                    options = run_options(
                        q, i, phi_in, ax_in, ax_out, phi_e if last else None
                    )
                    memo[key] = options
                for phi, c in options:
                    akey = round(math.remainder(phi, _TWO_PI) * 1e8)
                    old = nxt.get(akey)
                    if old is None or total + c < old[0]:
                        nxt[akey] = (total + c, phi, pi, c)
            prev = sorted(nxt.values(), key=lambda t: (t[0], abs(t[1])))[:_BEAM]
            layers.append(prev)
        best = 0
        total = layers[-1][0][0]
        angles = [0.0] * (e - s + 1)
        run_costs = [0] * (e - s + 1)
        for k in range(e - s, -1, -1):
            entry = layers[k][best]
            angles[k] = entry[1]
            run_costs[k] = entry[3]
            best = entry[2]
        return total, angles, run_costs

    for q in all_wires:
        k = len(wire_runs[q]) - 1
        _, outs[q], costs[q] = solve(q, 0, k, 0.0, 0.0)

    # --- CX orientation local search --------------------------------------
    def window(q: int, j: int) -> tuple[int, int, float, float]:
        k = len(wire_runs[q]) - 1
        s = max(0, j - _WINDOW)
        e = min(k, j + 1 + _WINDOW)
        phi_s = outs[q][s - 1] if s > 0 else 0.0
        return s, e, phi_s, outs[q][e]

    for _ in range(_SWEEPS):
        improved = False
        for g, ((qa, ja), (qb, jb)) in enumerate(flip_pos):
            wa, wb = window(qa, ja), window(qb, jb)
            old = sum(costs[qa][wa[0] : wa[1] + 1]) + sum(costs[qb][wb[0] : wb[1] + 1])
            flipped[g] = not flipped[g]
            ra = solve(qa, *wa)
            rb = solve(qb, *wb)
            if ra[0] + rb[0] < old:
                for q, (s, e, _, _), (_, ang, cst) in ((qa, wa, ra), (qb, wb, rb)):
                    outs[q][s : e + 1] = ang
                    costs[q][s : e + 1] = cst
                improved = True
            else:
                flipped[g] = not flipped[g]
        if not improved:
            break

    # --- emit -------------------------------------------------------------
    final: dict[int, list[Gate]] = {}
    for q in all_wires:
        anchors = wire_anchors[q]
        for i, rid in enumerate(wire_runs[q]):
            run = runs[rid]
            m, changed = run_base(q, i)
            phi_in = outs[q][i - 1] if i > 0 else 0.0
            phi_out = outs[q][i]
            if phi_in != 0.0:
                ax = anchor_axis(anchors[i - 1])
                assert ax is not None
                m = m @ _rot(ax, phi_in)
                changed = True
            if phi_out != 0.0:
                ax = anchor_axis(anchors[i])
                assert ax is not None
                m = _rot(ax, -phi_out) @ m
                changed = True
            seq = synthesize_one_qubit(m, one_qubit)
            if (
                not changed
                and run.native
                and (seq is None or len(run.gates) <= len(seq))
            ):
                seq = run.gates
            if seq is None:
                if run.gates or changed:
                    raise ValueError(
                        f"Cannot express a single-qubit run on qubit {q} in the "
                        f"target gates {sorted(one_qubit)}."
                    )
                seq = []
            final[rid] = seq
    if sum(len(s) for s in final.values()) > original_1q:
        return circuit

    out: list[CircuitInstruction] = []

    def emit(q: int, rid: int) -> None:
        out.extend(
            CircuitInstruction(type=InstructionType.GATE, gate=g, qubits=(q,))
            for g in final[rid]
        )

    flip_of = {idx: g for g, idx in enumerate(flip_instr)}
    for idx, ins in enumerate(instrs):
        anchored = instr_runs.get(idx)
        if anchored is None:
            continue  # absorbed into a run
        for q, rid in anchored:
            emit(q, rid)
        g = flip_of.get(idx, -1)
        if g >= 0 and flipped[g]:
            ins = CircuitInstruction(
                type=ins.type, gate=ins.gate, qubits=ins.qubits[::-1], label=ins.label
            )
        out.append(ins)
    for q in all_wires:
        emit(q, current[q])
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)
