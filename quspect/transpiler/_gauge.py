"""Single-qubit gate minimization around two-qubit gates (CX, CZ, …).

A rotation about an axis that commutes with a two-qubit gate on one of its
qubits can be moved through that gate: ``RZ`` through the control of a CX
(or either qubit of a CZ), ``RX`` through the target of a CX.  Moving a
rotation from the end of one single-qubit run to the start of the next
changes neither the unitary nor the two-qubit gates, only how the
single-qubit gates are distributed over the runs — and therefore how many
gates each run needs in the target's single-qubit basis.  In addition a CX
can be reversed, ``CX(a, b) = (H⊗H)·CX(b, a)·(H⊗H)``, the Hadamards being
absorbed into the four neighbouring runs.

The rotations commute with a gate *on one qubit*, independently of the
other qubit, so for fixed CX orientations the problem splits into one
chain per wire: runs ``R0, g1, R1, g2, …`` where each ``g`` lets a rotation
about a fixed axis (or none) through.  :func:`minimize_single_qubit_gates`
solves each chain with a small beam search over the rotation angle carried
into the next run (candidates: nothing, or the leading rotation of one of
the run's Euler decompositions about that axis), then improves the CX
orientations by local search (a reversal is kept when re-solving the runs
next to it on both wires, with the angles carried in and out fixed,
scores better), and finally resynthesizes every changed run in the target
basis.  A wire's score is two per single-qubit gate plus one per non-empty
run, so a rotation is only split off into a new run when that saves gates.
Run lengths are scored with a fast model of
:func:`~quspect.gates.synthesis.synthesize_one_qubit`; the emitted runs are
synthesized and verified for real.  This generalizes Qiskit's
``CommutativeCancellation`` + ``Optimize1qGatesDecomposition`` pair.

When the C++ extension is built, the search runs on native kernels
(environment variable ``QUSPECT_GAUGE_BACKEND``: ``auto``, ``native`` or
``python``): all wires advance layer by layer with the 2×2 products batched
in numpy, the orientation sweep evaluates every CX at once against the
state at the start of the sweep and commits the results in gate order
(re-evaluating the few gates next to an accepted reversal), and every
kernel reproduces the arithmetic of the Python search bit for bit, so both
backends return identical circuits.
"""

from __future__ import annotations

import cmath
import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate

__all__ = ["minimize_single_qubit_gates", "orient_cx_gates"]

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

#: Entry tolerance of :func:`~quspect.gates.synthesis.synthesize_one_qubit`
#: (its identity / diagonal tests).
_ATOL_1Q = 1e-9
#: Beam width of the per-wire search.
_BEAM = 4
#: Runs on each side of a reversed CX that are re-optimized.
_WINDOW = 0
#: Orientation local-search sweeps.
_SWEEPS = 2
#: Subgradient iterations of the global CX orientation (:func:`orient_cx_gates`).
_ORIENT_ITERATIONS = 40
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


#: Environment variable selecting the beam-search kernels: ``"auto"``
#: (default: native when the C++ extension provides them and passes its
#: self-check, else Python), ``"native"`` (raise if unavailable) or
#: ``"python"``.  Both give bit-identical results.
GAUGE_BACKEND_ENV = "QUSPECT_GAUGE_BACKEND"
_BACKENDS = ("auto", "native", "python")
_AXIS_CODE = {"z": 0, "x": 1, "y": 2}
#: ``conj(V)`` and ``V`` per axis code (for batched frame changes; ``V†`` is
#: used as the transposed view of ``conj(V)``, as in ``v.conj().T @ m``).
_TO_Z_CONJ = np.stack([_TO_Z[a].conj() for a in ("z", "x", "y")])
_TO_Z_STACK = np.stack([_TO_Z[a] for a in ("z", "x", "y")])
_EMPTY_STACK = np.empty((0, 2, 2), dtype=complex)


def _axis_code(axis: str | None) -> int:
    return -1 if axis is None else _AXIS_CODE[axis]


class _NativeFallbackError(Exception):
    """A native kernel met an input it does not reproduce (use Python)."""


class _Native:
    """C++ kernels of the beam search (``quspect._core_*``).

    Every kernel reproduces the Python reference bit for bit (the CPython /
    numpy complex arithmetic, including the FMA contractions of their C
    builds, is written out explicitly); 2×2 matrix products stay in numpy
    (BLAS), batched over many runs at a time.  Inputs a kernel cannot
    reproduce (non-finite values, division by zero) raise
    :class:`_NativeFallbackError`.
    """

    def __init__(self, core: Any, length: _LengthModel) -> None:
        named = np.array(length.named_entries, dtype=complex).reshape(-1, 2, 2)
        self.core = core
        self.model = core._GaugeLengthModel(
            length.h, length.x, length.sx, length.sxdg, named
        )

    def lengths(self, mats: np.ndarray) -> list[int]:
        out = self.core._gauge_lengths(self.model, mats)
        if (out < 0).any():
            raise _NativeFallbackError
        return [int(x) for x in out]

    def rot(self, axis: str, phi: float) -> np.ndarray:
        return np.asarray(self.core._gauge_rotations([_AXIS_CODE[axis]], [phi])[0])

    def rots(self, axis: str, phis: list[float]) -> np.ndarray:
        return np.asarray(
            self.core._gauge_rotations([_AXIS_CODE[axis]] * len(phis), phis)
        )

    def leading(self, m: np.ndarray, axis: str) -> list[float]:
        v = _TO_Z[axis]
        z = v.conj().T @ m @ v
        counts, angles = self.core._gauge_leading_angles(z.reshape(1, 2, 2))
        if counts[0] == 0:
            raise _NativeFallbackError
        return [float(a) for a in angles[0, : counts[0]]]


_NATIVE_CACHE: dict[frozenset[str], _Native | None] = {}


def _gauge_backend() -> str:
    backend = os.environ.get(GAUGE_BACKEND_ENV, "auto") or "auto"
    if backend not in _BACKENDS:
        raise ValueError(
            f"{GAUGE_BACKEND_ENV} must be one of {', '.join(_BACKENDS)}; "
            f"got {backend!r}"
        )
    return backend


def _native_kernels(length: _LengthModel) -> _Native | None:
    """Native kernels for ``length``'s basis, or ``None`` (not built / not fast).

    The kernels are checked once per basis against the Python reference on
    a fixed sample; any difference disables them (e.g. a platform whose
    CPython or numpy rounds differently from the replicated arithmetic).
    """
    if not length.fast:
        return None
    try:
        return _NATIVE_CACHE[length.basis]
    except KeyError:
        pass
    from quspect._native_loader import load_core

    native: _Native | None = None
    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        core = None
    if core is not None and hasattr(core, "_GaugeBeamSearch"):
        native = _Native(core, length)
        if not _self_check(native, length):  # pragma: no cover - platform specific
            native = None
    _NATIVE_CACHE[length.basis] = native
    return native


def _self_check(native: _Native, length: _LengthModel) -> bool:
    """Whether the native kernels match the Python reference on a sample."""
    rng = np.random.default_rng(12345)
    a = rng.normal(size=(64, 2, 2)) + 1j * rng.normal(size=(64, 2, 2))
    mats = [np.linalg.qr(x)[0] for x in a]
    for axis in ("z", "x", "y"):
        for phi in (0.3, -1.2, math.pi / 2, math.pi, 1e-9):
            mats.append(_rot(axis, phi) @ _H)
            mats.append(_H @ _rot(axis, phi) @ _rot("x", 0.7))
    mats += [_I2, _H, _PAULI["x"], _PAULI["y"], _PAULI["z"]]
    try:
        got = native.lengths(np.array(mats))
        for m, g in zip(mats, got, strict=True):
            if g != length._fast(m):
                return False
        for axis in ("z", "x", "y"):
            for m in mats:
                ref = _leading_angles(m, axis)
                new = native.leading(m, axis)
                if [x.hex() for x in map(float, ref)] != [x.hex() for x in new]:
                    return False
            for phi in (0.0, -0.0, 0.3, -2.5, math.pi, 7.0, 1e-300):
                if native.rot(axis, phi).tobytes() != _rot(axis, phi).tobytes():
                    return False
    except _NativeFallbackError:
        return False
    return True


class _Run:
    __slots__ = ("gates", "instrs", "matrix", "native", "plain")

    def __init__(self) -> None:
        self.gates: list[Gate] = []
        # the source instructions; ``plain``: each equals the instruction the
        # emission would build for its gate (no label, no clbits), so a kept
        # run can reuse them (they are frozen values)
        self.instrs: list[CircuitInstruction] = []
        self.matrix = _I2
        self.native = True
        self.plain = True


class _Anchor:
    """One qubit of a multi-qubit instruction (or a blocking instruction)."""

    __slots__ = ("axis", "gate")

    def __init__(self, axis: str | None, gate: int) -> None:
        self.axis = axis  # commuting axis in the original orientation
        self.gate = gate  # index into the flippable gates, or -1


def minimize_single_qubit_gates(
    circuit: QuantumCircuit,
    one_qubit: frozenset[str],
    *,
    global_orientation: bool = False,
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

    With ``global_orientation`` the CX orientations are first chosen for
    the whole circuit at once (:func:`orient_cx_gates`, a dual
    decomposition over the wires) and the minimization is run from that
    orientation as well; the result with fewer single-qubit gates is
    returned (the plain one on a tie).  The local orientation search alone
    cannot leave a minimum where a whole region has to be reversed
    together (e.g. ZZ interactions emitted in the X frame).

    Args:
        circuit: Circuit whose single-qubit gates have known matrices or
            are native (symbolic gates are kept and block movement).
        one_qubit: The target's single-qubit gate names.
        global_orientation: Also start from the globally chosen CX
            orientation (needs the C++ extension and an ``rz`` basis with
            ``h`` or ``sx``; ignored otherwise).

    Returns:
        An equivalent circuit (up to global phase).

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().rz(0.3, 0).cx(0, 1).rz(0.2, 0).h(0)
        >>> out = minimize_single_qubit_gates(qc, frozenset({"h", "rz"}))
        >>> [(i.gate.name, i.qubits) for i in out]
        [('cx', (0, 1)), ('rz', (0,)), ('h', (0,))]
    """
    out = _minimize_local(circuit, one_qubit)
    if global_orientation:
        oriented = orient_cx_gates(circuit, one_qubit)
        if oriented is not circuit:
            alt = _minimize_local(oriented, one_qubit)
            if _count_one_qubit(alt) < _count_one_qubit(out):
                out = alt
    return out


def _count_one_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit.instructions if i.is_gate and len(i.qubits) == 1)


def _minimize_local(
    circuit: QuantumCircuit, one_qubit: frozenset[str]
) -> QuantumCircuit:
    """:func:`minimize_single_qubit_gates` without the global orientation."""
    backend = _gauge_backend()
    if backend != "python":
        length = _LengthModel(one_qubit)
        native = _native_kernels(length)
        if native is not None:
            try:
                return _minimize(circuit, one_qubit, native)
            except _NativeFallbackError:
                pass
        elif backend == "native" and length.fast:
            raise RuntimeError(
                "The native single-qubit minimization kernels are not available "
                "(build the C++ extension, see quspect._native_loader.BUILD_HINT)."
            )
    return _minimize(circuit, one_qubit, None)


def orient_cx_gates(
    circuit: QuantumCircuit,
    one_qubit: frozenset[str],
    *,
    iterations: int = _ORIENT_ITERATIONS,
) -> QuantumCircuit:
    """Choose the orientation of every CX for the whole circuit at once.

    ``CX(a, b) = (H⊗H)·CX(b, a)·(H⊗H)``: each CX can be reversed, the
    Hadamards joining the four single-qubit runs around it.  With one
    binary variable per CX, the single-qubit cost of a wire depends only on
    the orientations of the CX gates on that wire (and on the rotations
    moved through them: ``RZ`` through a control, ``RX`` through a target),
    so the total is a sum of chain problems in which every CX appears on
    exactly two chains.  The orientations are found by *dual
    decomposition*: every CX variable is duplicated, each wire's chain is
    solved by dynamic programming (orientation × beam of moved rotation
    angles, the run cost of :func:`minimize_single_qubit_gates`) under
    Lagrange multipliers that price the disagreement of the two copies,
    and the multipliers follow a subgradient (Polyak) step.  Every
    iteration turns both copies into a primal orientation; the cheapest
    is kept (never worse than the input orientation).

    The returned circuit has the chosen CX gates reversed with explicit
    ``h`` gates around them (the same unitary); it is meant as the start of
    :func:`minimize_single_qubit_gates`, which merges the Hadamards.

    Args:
        circuit: A circuit of ``cx`` gates and single-qubit gates with
            known matrices.
        one_qubit: The target's single-qubit gate names (the cost model).
        iterations: Maximum number of subgradient iterations.

    Returns:
        The re-oriented circuit, or ``circuit`` itself when nothing is
        reversed or the circuit / basis is not supported (other gates,
        no C++ extension, a basis without ``rz`` and ``h`` / ``sx``).

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().h(0).h(1).cx(0, 1).rx(0.3, 0).cx(0, 1)
        >>> qc = qc.h(0).h(1)
        >>> out = orient_cx_gates(qc, frozenset({"rz", "sx", "x"}))
        >>> [i.qubits for i in out if i.gate.name == "cx"]
        [(1, 0), (1, 0)]
    """
    from quspect.gates._matrix_utils import gate_matrix

    length = _LengthModel(one_qubit)
    if not length.fast or _gauge_backend() == "python":
        return circuit
    native = _native_kernels(length)
    if native is None or not hasattr(native.core, "_cx_orientation"):
        return circuit
    instrs = circuit.instructions
    cx_instr: list[int] = []
    wire_gates: dict[int, list[int]] = {}
    wire_ctrl: dict[int, list[int]] = {}
    wire_runs: dict[int, list[np.ndarray]] = {}
    current: dict[int, np.ndarray] = {}
    for idx, ins in enumerate(instrs):
        if not ins.is_gate or ins.gate is None:
            return circuit
        if len(ins.qubits) == 1:
            try:
                m = np.asarray(gate_matrix(ins.gate), dtype=complex)
            except (KeyError, ValueError, TypeError):
                return circuit
            q = ins.qubits[0]
            current[q] = m @ current[q] if q in current else m
            continue
        if ins.gate.name != "cx" or len(ins.qubits) != 2:
            return circuit
        g = len(cx_instr)
        cx_instr.append(idx)
        for pos, q in enumerate(ins.qubits):
            wire_runs.setdefault(q, []).append(current.pop(q, _I2))
            wire_gates.setdefault(q, []).append(g)
            wire_ctrl.setdefault(q, []).append(1 if pos == 0 else 0)
    if not cx_instr:
        return circuit
    wires = sorted(wire_gates)
    for q in wires:
        wire_runs[q].append(current.pop(q, _I2))
    runs = np.array([m for q in wires for m in wire_runs[q]], dtype=complex)
    if not np.isfinite(runs).all():
        return circuit
    flip, _, _, _, _ = native.core._cx_orientation(
        native.model,
        len(cx_instr),
        [wire_gates[q] for q in wires],
        [wire_ctrl[q] for q in wires],
        runs.reshape(-1, 2, 2),
        _BEAM,
        _GATE_WEIGHT,
        _RUN_WEIGHT,
        iterations,
    )
    if not any(flip):
        return circuit
    return _reverse_cx(circuit, {cx_instr[g] for g, f in enumerate(flip) if f})


def _reverse_cx(circuit: QuantumCircuit, reverse: set[int]) -> QuantumCircuit:
    """``circuit`` with the CX instructions ``reverse`` written as ``H⊗H·CX·H⊗H``."""
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates.standard import H

    out: list[CircuitInstruction] = []
    for idx, ins in enumerate(circuit.instructions):
        if idx not in reverse:
            out.append(ins)
            continue
        hs = [
            CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(q,))
            for q in ins.qubits
        ]
        out += hs
        out.append(
            CircuitInstruction(
                type=ins.type, gate=ins.gate, qubits=ins.qubits[::-1], label=ins.label
            )
        )
        out += hs
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)


def _minimize(
    circuit: QuantumCircuit, one_qubit: frozenset[str], native: _Native | None
) -> QuantumCircuit:
    """Body of :func:`minimize_single_qubit_gates`.

    ``native`` selects the C++ kernels (batched initial beam search, native
    run lengths / leading angles / rotations); ``None`` runs the pure-Python
    reference.  Both produce identical circuits.
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
                run.instrs.append(ins)
                if ins.label or ins.clbits:
                    run.plain = False
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
        if native is not None:
            return run_options_native(q, i, phi_in, ax_in, ax_out, phi_fixed)
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

    def run_options_native(
        q: int,
        i: int,
        phi_in: float,
        ax_in: str | None,
        ax_out: str | None,
        phi_fixed: float | None,
    ) -> list[tuple[float, int]]:
        """:func:`run_options` with the native kernels (same values)."""
        assert native is not None
        run = runs[wire_runs[q][i]]
        m, changed = run_base(q, i)
        if ax_in is not None and phi_in != 0.0:
            m = m @ native.rot(ax_in, phi_in)
            changed = True
        if phi_fixed is not None:
            opts = [phi_fixed]
        elif ax_out is None:
            opts = [0.0]
        else:
            opts = [0.0, *native.leading(m, ax_out)]
        moved = [phi for phi in opts if phi != 0.0 and ax_out is not None]
        mats = [m]
        if moved:
            assert ax_out is not None
            mats.extend(np.matmul(native.rots(ax_out, [-phi for phi in moved]), m))
        lengths = native.lengths(np.array(mats))
        result = []
        j = 0
        for phi in opts:
            if phi != 0.0 and ax_out is not None:
                j += 1
                c = lengths[j]
                ch = True
            else:
                c = lengths[0]
                ch = changed
            if not ch and run.native:
                c = min(c, len(run.gates))
            result.append((phi, _GATE_WEIGHT * c + (_RUN_WEIGHT if c else 0)))
        return result

    # per wire: angle carried out of each run (about the next anchor's axis)
    outs: dict[int, list[float]] = {}
    costs: dict[int, list[int]] = {}

    #: native beam search / memo (set by :func:`solve_all_native`)
    searches: list[Any] = []

    def memo_get(
        key: tuple[int, bool, bool, int, bool, float],
    ) -> list[tuple[float, int]] | None:
        if searches:
            result: list[tuple[float, int]] | None = searches[0].memo_get(*key)
            return result
        return memo.get(key)

    def memo_put(
        key: tuple[int, bool, bool, int, bool, float],
        phi_in: float,
        options: list[tuple[float, int]],
    ) -> None:
        if searches:
            searches[0].memo_put(*key, phi_in, options)
        else:
            memo[key] = options

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
                options = memo_get(key)
                if options is None:
                    options = run_options(
                        q, i, phi_in, ax_in, ax_out, phi_e if last else None
                    )
                    memo_put(key, phi_in, options)
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

    matrices = np.array([run.matrix for run in runs], dtype=complex)
    run_len_of = [len(run.gates) if run.native else -1 for run in runs]

    def native_layer(
        search: Any,
        rows: tuple[list[Any], ...],
        insert_memo: bool,
    ) -> None:
        """One batched layer of the native beam search (see ``GaugeBeamSearch``).

        ``rows``: slots, run ids, fp, fn, ax_in, ax_out (codes, −1 = none),
        last, phi_fixed, run lengths, base changed.  The run matrices get
        the same numpy products as ``run_base`` / ``run_options`` (each
        batched product is the single ``@`` BLAS call).
        """
        slots, rids, fp, fn, ax_in, ax_out, last, phi_fixed, run_len, changed = rows
        try:
            crow, rot_in, rot_in_mats, need_lead = search.prepare(
                slots, rids, fp, fn, ax_in, ax_out, last, phi_fixed, run_len, changed
            )
            m = matrices[np.asarray(rids, dtype=np.intp)[crow]]
            if len(crow) and any(changed):
                fp_rows = np.flatnonzero(np.asarray(fp, dtype=bool)[crow])
                if len(fp_rows):
                    m[fp_rows] = np.matmul(m[fp_rows], _H)
                fn_rows = np.flatnonzero(np.asarray(fn, dtype=bool)[crow])
                if len(fn_rows):
                    m[fn_rows] = np.matmul(_H, m[fn_rows])
            if len(rot_in):
                m[rot_in] = np.matmul(m[rot_in], rot_in_mats)
            if len(need_lead):
                axes = np.asarray(ax_out, dtype=np.intp)[crow[need_lead]]
                z = np.matmul(
                    np.matmul(_TO_Z_CONJ[axes].transpose(0, 2, 1), m[need_lead]),
                    _TO_Z_STACK[axes],
                )
            else:
                z = _EMPTY_STACK
            opt_rows, rot_out = search.options(z)
            search.finish(m, np.matmul(rot_out, m[opt_rows]), insert_memo)
        except ValueError as exc:
            raise _NativeFallbackError from exc

    def solve_all_native() -> None:
        """``solve(q, 0, k, 0.0, 0.0)`` for every wire, all wires at once.

        The wires are independent before any CX is reversed, so the beam
        search advances layer by layer over all wires together: the 2×2
        products run as a few batched numpy calls per layer and everything
        else in C++.  The (native) memo gets the same entries as the
        per-wire search.
        """
        assert native is not None
        try:
            search = native.core._GaugeBeamSearch(
                len(all_wires), _BEAM, _GATE_WEIGHT, _RUN_WEIGHT, native.model
            )
        except ValueError as exc:  # pragma: no cover - defensive
            raise _NativeFallbackError from exc
        searches.append(search)
        try:
            search.set_wires(
                [wire_runs[q] for q in all_wires],
                [[_axis_code(a.axis) for a in wire_anchors[q]] for q in all_wires],
                run_len_of,
            )
            depth = max((len(wire_runs[q]) for q in all_wires), default=0)
            for i in range(depth):
                crid, cax, rot_in, rot_in_mats, need_lead = search.prepare_layer(i)
                m = matrices[crid]
                if len(rot_in):
                    m[rot_in] = np.matmul(m[rot_in], rot_in_mats)
                if len(need_lead):
                    axes = cax[need_lead]
                    z = np.matmul(
                        np.matmul(_TO_Z_CONJ[axes].transpose(0, 2, 1), m[need_lead]),
                        _TO_Z_STACK[axes],
                    )
                else:
                    z = _EMPTY_STACK
                opt_rows, rot_out = search.options(z)
                search.finish(m, np.matmul(rot_out, m[opt_rows]), True)
        except ValueError as exc:
            raise _NativeFallbackError from exc
        for slot, q in enumerate(all_wires):
            _, angles, run_costs = search.backtrack(slot)
            outs[q] = list(angles)
            costs[q] = list(run_costs)

    if native is not None:
        solve_all_native()
    else:
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

    def evaluate(g: int) -> bool:
        """Try reversing gate ``g`` (re-solving its windows); keep it if better."""
        (qa, ja), (qb, jb) = flip_pos[g]
        wa, wb = window(qa, ja), window(qb, jb)
        old = sum(costs[qa][wa[0] : wa[1] + 1]) + sum(costs[qb][wb[0] : wb[1] + 1])
        flipped[g] = not flipped[g]
        ra = solve(qa, *wa)
        rb = solve(qb, *wb)
        if ra[0] + rb[0] < old:
            for q, (s, e, _, _), (_, ang, cst) in ((qa, wa, ra), (qb, wb, rb)):
                outs[q][s : e + 1] = ang
                costs[q][s : e + 1] = cst
            return True
        flipped[g] = not flipped[g]
        return False

    def sweep_native() -> bool:
        """One sweep of :func:`evaluate` over all gates, batched speculatively.

        Every gate is first evaluated against the state at the start of the
        sweep, all windows at once (two runs per window, one native layer
        each; memo entries are held back).  The results are then committed
        in gate order exactly as the sequential sweep would produce them: a
        gate is re-evaluated sequentially when an earlier gate of this
        sweep was reversed next to one of its windows (anchors ``j − 2``,
        ``j − 1`` or ``j + 1`` on either wire, whose reversal changes the
        window's matrices, axes, end angles or costs), or when a memo key
        it computed was meanwhile filled from a different ``phi_in`` (the
        sequential sweep would have used those options).  Reversals are
        rare (well under 1 %), so nearly every gate is committed as is.
        """
        search = searches[0]
        slots: list[int] = []
        chains: list[tuple[int, int, int]] = []  # (gate, wire, anchor)
        for g, ends in enumerate(flip_pos):
            for q, j in ends:
                chains.append((g, q, j))
                slots.append(len(slots))
        search.reset([outs[q][j - 1] if j > 0 else 0.0 for _, q, j in chains])

        def flipped_for(a: _Anchor, g: int) -> bool:
            return a.gate >= 0 and (flipped[a.gate] != (a.gate == g))

        def axis_for(a: _Anchor, g: int) -> int:
            ax = a.axis
            if ax is not None and flipped_for(a, g):
                ax = _SWAP_AXIS.get(ax)
            return -1 if ax is None else _AXIS_CODE[ax]

        for layer in (0, 1):
            cols: tuple[list[Any], ...] = tuple([] for _ in range(10))
            for slot, (g, q, j) in enumerate(chains):
                anchors = wire_anchors[q]
                i = j + layer
                fp = i > 0 and flipped_for(anchors[i - 1], g)
                fn = i < len(anchors) and flipped_for(anchors[i], g)
                rid = wire_runs[q][i]
                cols[0].append(slot)
                cols[1].append(rid)
                cols[2].append(1 if fp else 0)
                cols[3].append(1 if fn else 0)
                cols[4].append(axis_for(anchors[i - 1], g) if i > 0 else -1)
                cols[5].append(axis_for(anchors[i], g) if i < len(anchors) else -1)
                cols[6].append(layer)
                cols[7].append(outs[q][j + 1] if layer else 0.0)
                cols[8].append(run_len_of[rid])
                cols[9].append(1 if fp or fn else 0)
            native_layer(search, cols, False)
        results = [search.backtrack(slot) for slot in slots]

        improved = False
        dirty: set[tuple[int, int]] = set()
        for g, ((qa, ja), (qb, jb)) in enumerate(flip_pos):
            conflict = any(
                (q, a) in dirty
                for q, j in ((qa, ja), (qb, jb))
                for a in (j - 2, j - 1, j + 1)
            )
            if conflict or not search.commit_slots([2 * g, 2 * g + 1]):
                accepted = evaluate(g)
            else:
                ra, rb = results[2 * g], results[2 * g + 1]
                old = sum(costs[qa][ja : ja + 2]) + sum(costs[qb][jb : jb + 2])
                accepted = ra[0] + rb[0] < old
                if accepted:
                    for q, j, (_, ang, cst) in ((qa, ja, ra), (qb, jb, rb)):
                        outs[q][j : j + 2] = list(ang)
                        costs[q][j : j + 2] = list(cst)
                    flipped[g] = not flipped[g]
            if accepted:
                dirty.add((qa, ja))
                dirty.add((qb, jb))
                improved = True
        return improved

    for _ in range(_SWEEPS):
        if searches and _WINDOW == 0:
            improved = sweep_native()
        else:
            improved = False
            for g in range(len(flip_pos)):
                if evaluate(g):
                    improved = True
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
            if (
                not changed
                and run.native
                and (
                    not run.gates
                    or (
                        len(run.gates) == 1
                        and not (abs(m[0, 1]) < _ATOL_1Q and abs(m[1, 0]) < _ATOL_1Q)
                    )
                )
            ):
                # An unchanged native run of at most one gate is kept: the
                # synthesis could only replace it by something strictly
                # shorter, i.e. by nothing, and every candidate it builds for
                # a non-diagonal matrix has at least one gate.
                final[rid] = run.gates
                continue
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
        seq = final[rid]
        run = runs[rid]
        if seq is run.gates and run.plain:
            out.extend(run.instrs)  # equal to the instructions built below
            return
        out.extend(
            CircuitInstruction(type=InstructionType.GATE, gate=g, qubits=(q,))
            for g in seq
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
