"""Target-basis handling of :func:`~quspect.transpiler.transpile`.

The optimizing pipeline (block resynthesis, three-qubit and Toffoli
templates, phase folding, mirror-gate costs, SABRE's cost estimates) works
in a *working basis* built around CZ: the target's single-qubit gates plus
``{cz, h, rz, x}``.  CZ, CX and every other CZ-class entangler differ only
by single-qubit gates, so all two-qubit counts computed there are exact for
any such target.  The last step, :func:`translate_to_target`, rewrites each
CZ into the target's entangler — for CX ``CZ(a, b) = H(b)·CX(a, b)·H(b)``,
the target qubit chosen per gate so that the Hadamards merge best into the
neighbouring single-qubit runs — and resynthesizes every touched
single-qubit run in the target's own single-qubit gates (e.g. ``{rz, sx,
x}`` for IBM devices).  The two-qubit count is therefore identical to the
CZ-target result; single-qubit runs stay at most one Euler sequence long.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from quspect.synthesis.two_qubit import _ENTANGLERS, Entangler, entangler

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate
    from quspect.transpiler._target import CompilationTarget

__all__ = ["BasisPlan", "plan_basis", "translate_to_target"]

#: Gates the internal pipeline needs besides the target's single-qubit gates.
WORKING_CORE = frozenset({"cz", "h", "rz", "x"})

#: A fixed generic single-qubit unitary used for the universality check.
_GENERIC_1Q = np.array(
    [
        [0.6 + 0.48j, -0.4 + 0.5j],
        [0.4 + 0.5j, 0.6 - 0.48j],
    ],
    dtype=complex,
)
_GENERIC_1Q = np.linalg.qr(_GENERIC_1Q)[0]


@dataclass(frozen=True)
class BasisPlan:
    """How :func:`~quspect.transpiler.transpile` reaches a target basis.

    Attributes:
        native: The target's native gates (empty: no basis restriction).
        working: Gates the internal pipeline compiles to.
        entangler: The target entangler that replaces ``cz`` at the end
            (``None`` if ``cz`` is native or no translation is needed).
        one_qubit: The target's single-qubit gate names.

    Example:
        >>> plan = plan_basis({"cx", "rz", "sx", "x"})
        >>> sorted(plan.working), plan.entangler
        (['cz', 'h', 'rz', 'sx', 'x'], 'cx')
    """

    native: frozenset[str]
    working: frozenset[str]
    entangler: str | None
    one_qubit: frozenset[str]

    @property
    def needs_translation(self) -> bool:
        """Whether the working-basis output must be rewritten for the target."""
        return bool(self.native) and self.working != self.native

    def working_target(self, target: CompilationTarget) -> CompilationTarget:
        """Return ``target`` with the working basis as its native gates."""
        if not self.needs_translation:
            return target
        return replace(target, native_gates=set(self.working))


def _one_qubit_names(native: Collection[str]) -> frozenset[str]:
    """Native gate names that are known single-qubit gates."""
    from quspect.gates.parameterized import ALL_PARAMETERIZED
    from quspect.gates.standard import ALL_STANDARD

    known = {g.name for g in (*ALL_STANDARD, *ALL_PARAMETERIZED) if g.num_qubits == 1}
    return frozenset(n for n in native if n in known)


def plan_basis(
    native_gates: Collection[str], circuit: QuantumCircuit | None = None
) -> BasisPlan:
    """Check a target basis and derive the working basis of the pipeline.

    Args:
        native_gates: The target's native gate names.  Empty means "no
            restriction" (nothing is checked or translated).
        circuit: Optional input circuit; a target without entangler is
            accepted when the circuit has no multi-qubit gates.

    Returns:
        The :class:`BasisPlan`.

    Raises:
        ValueError: If the single-qubit gates of the target cannot express
            an arbitrary single-qubit unitary (e.g. ``{h, t}``), or if the
            circuit has multi-qubit gates and the target has no supported
            entangler (``cz``, ``cx`` or one registered with
            :func:`~quspect.synthesis.two_qubit.register_entangler`).

    Example:
        >>> plan_basis({"cz", "h", "rz", "x"}).needs_translation
        False
        >>> plan_basis({"cx", "h", "rz", "x"}).entangler
        'cx'
    """
    from quspect.gates.synthesis import synthesize_one_qubit

    native = frozenset(native_gates)
    if not native:
        return BasisPlan(native, native, None, frozenset())
    one_q = _one_qubit_names(native)
    if synthesize_one_qubit(_GENERIC_1Q, one_q) is None:
        raise ValueError(
            "Target basis cannot express arbitrary single-qubit gates: native "
            f"single-qubit gates {sorted(one_q)}; need e.g. {{rz, h}}, {{rz, sx}}, "
            "{rz, rx}, {rz, ry} or {u3}."
        )
    ent: str | None
    if "cz" in native:
        ent = None
    else:
        ent = next(
            (
                e
                for e in ["cx", *sorted(set(_ENTANGLERS) - {"cx", "cz"})]
                if e in native
            ),
            None,
        )
        if ent is None and circuit is not None and _has_multi_qubit_gates(circuit):
            raise ValueError(
                "Target basis has no supported two-qubit entangler: native gates "
                f"{sorted(native)}; need cz, cx or a gate registered with "
                "register_entangler()."
            )
    if ent is None and "cz" not in native:
        # 没有纠缠门（电路也不需要）：只翻译单比特门
        working = (native | WORKING_CORE) - {"cz"}
    else:
        working = (native - {ent}) | WORKING_CORE
    return BasisPlan(native, frozenset(working), ent, one_q)


def _has_multi_qubit_gates(circuit: QuantumCircuit) -> bool:
    return any(i.is_gate and len(i.qubits) >= 2 for i in circuit)


# =============================================================================
# Final translation
# =============================================================================


class _Run:
    """A pending single-qubit run on one wire."""

    __slots__ = ("dirty", "gates", "matrix")

    def __init__(self) -> None:
        self.gates: list[CircuitInstruction] = []
        self.matrix: np.ndarray = np.eye(2, dtype=complex)
        self.dirty = False


def _is_identity(m: np.ndarray) -> bool:
    return bool(
        abs(m[0, 1]) < 1e-14 and abs(m[1, 0]) < 1e-14 and abs(m[0, 0] - m[1, 1]) < 1e-14
    )


def translate_to_target(circuit: QuantumCircuit, plan: BasisPlan) -> QuantumCircuit:
    """Rewrite a working-basis circuit into the target's native gates.

    Every ``cz`` becomes the plan's entangler with its local dressing (for
    CX a Hadamard on the target qubit, the target picked per gate so that
    the Hadamards cancel or merge best with the adjacent single-qubit runs);
    every single-qubit run that received a dressing or contains a
    non-native gate is resynthesized with
    :func:`~quspect.gates.synthesis.synthesize_one_qubit` in the target's
    single-qubit gates (verified to ``1e-9``, phase-aligned).  Untouched
    native runs are kept as they are.

    Args:
        circuit: Circuit whose gates are all in ``plan.working``.
        plan: The :class:`BasisPlan` of the target.

    Returns:
        An equivalent circuit (up to global phase) in ``plan.native``, with
        the same two-qubit gate count.

    Raises:
        ValueError: If a gate can be neither kept nor translated (e.g. a
            symbolic non-native single-qubit gate).

    Example:
        >>> from quspect import QuantumCircuit
        >>> plan = plan_basis({"cx", "h", "rz", "x"})
        >>> out = translate_to_target(QuantumCircuit().h(1).cz(0, 1).h(1), plan)
        >>> [(i.gate.name, i.qubits) for i in out]
        [('cx', (0, 1))]
    """
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.synthesis import synthesize_one_qubit

    if not plan.needs_translation:
        return circuit
    native = plan.native
    one_q = plan.one_qubit
    ent: Entangler | None = (
        entangler(plan.entangler) if plan.entangler is not None else None
    )
    instrs = circuit.instructions
    out: list[CircuitInstruction] = []
    runs: dict[int, _Run] = {}
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

    def synth_len(m: np.ndarray) -> int:
        seq = synthesize_one_qubit(m, one_q)
        return 10**6 if seq is None else len(seq)

    def flush(q: int) -> None:
        run = runs.pop(q, None)
        if run is None:
            return
        if not run.dirty:
            out.extend(run.gates)
            return
        seq = synthesize_one_qubit(run.matrix, one_q)
        if seq is None:
            raise ValueError(
                f"Cannot express a single-qubit run on qubit {q} in the target "
                f"gates {sorted(one_q)}."
            )
        out.extend(
            CircuitInstruction(type=InstructionType.GATE, gate=g, qubits=(q,))
            for g in seq
        )

    def run_of(q: int) -> _Run:
        run = runs.get(q)
        if run is None:
            run = runs[q] = _Run()
        return run

    # 每条线上的指令位置；cursor[q] 指向当前指令之后的第一个位置
    wire_pos: dict[int, list[int]] = {}
    for i, ins in enumerate(instrs):
        for q in ins.qubits:
            wire_pos.setdefault(q, []).append(i)
    cursor: dict[int, int] = dict.fromkeys(wire_pos, 0)

    def is_translated(ins: CircuitInstruction) -> bool:
        gate = ins.gate
        return (
            ent is not None
            and ins.is_gate
            and gate is not None
            and gate.name == "cz"
            and not gate.params
            and len(ins.qubits) == 2
        )

    def lookahead(q: int) -> tuple[np.ndarray, bool]:
        """Single-qubit gates on ``q`` after the current instruction.

        Returns their matrix and whether the run ends at another ``cz``
        that will be translated (and may dress the run's far end too).
        """
        m = np.eye(2, dtype=complex)
        positions = wire_pos[q]
        for pos in range(cursor[q], len(positions)):
            ins = instrs[positions[pos]]
            if not (ins.is_gate and len(ins.qubits) == 1 and ins.gate is not None):
                return m, is_translated(ins)
            g = matrix_of(ins.gate)
            if g is None:
                return m, False
            m = g @ m
        return m, False

    # 下一个 cz 可能放在一段单比特门末端的修饰门（CX：目标比特上的 H）
    far_ends: list[np.ndarray] = []
    if ent is not None:
        far_ends = [d for d in ent.before if not _is_identity(d)]

    for ins in instrs:
        for q in ins.qubits:
            cursor[q] += 1
        gate = ins.gate
        if ins.is_gate and gate is not None and len(ins.qubits) == 1:
            q = ins.qubits[0]
            m = matrix_of(gate)
            if m is None:
                if gate.name not in native:
                    raise ValueError(
                        f"Gate {gate.name!r} (symbolic or unknown) is not in the "
                        f"target basis {sorted(native)} and cannot be translated."
                    )
                flush(q)
                out.append(ins)
                continue
            run = run_of(q)
            run.gates.append(ins)
            run.matrix = m @ run.matrix
            if gate.name not in native:
                run.dirty = True
            continue
        if is_translated(ins):
            assert ent is not None
            a, b = ins.qubits
            options = [False] if ent.symmetric else [False, True]
            best: tuple[int, bool] | None = None
            if len(options) > 1:
                pre = {q: (runs[q].matrix if q in runs else np.eye(2)) for q in (a, b)}
                post = {q: lookahead(q) for q in (a, b)}
                for rev in options:
                    _, dress = ent.oriented(rev)
                    cost = 0
                    for slot, q in enumerate((a, b)):
                        cost += synth_len(dress.before[slot] @ pre[q])
                        m_post, open_end = post[q]
                        m_post = m_post @ dress.after[slot]
                        # 后一个 cz 也可能把修饰门放到这一段的末端：取两者中较好的
                        ends = [m_post] + [d @ m_post for d in far_ends if open_end]
                        cost += min(synth_len(m) for m in ends)
                    if best is None or cost < best[0]:
                        best = (cost, rev)
            rev = False if best is None else best[1]
            _, dress = ent.oriented(rev)
            for slot, q in enumerate((a, b)):
                if not _is_identity(dress.before[slot]):
                    run = run_of(q)
                    run.matrix = dress.before[slot] @ run.matrix
                    run.dirty = True
                flush(q)
            out.append(
                CircuitInstruction(
                    type=InstructionType.GATE,
                    gate=_gate_named(ent.name),
                    qubits=(b, a) if rev else (a, b),
                )
            )
            for slot, q in enumerate((a, b)):
                if not _is_identity(dress.after[slot]):
                    run = run_of(q)
                    run.matrix = dress.after[slot]
                    run.dirty = True
            continue
        if ins.is_gate and gate is not None and gate.name not in native:
            raise ValueError(
                f"Gate {gate.name!r} on {len(ins.qubits)} qubits is not in the "
                f"target basis {sorted(native)} and cannot be translated."
            )
        if not ins.qubits and ins.is_barrier:
            for q in list(runs):
                flush(q)
        for q in ins.qubits:
            flush(q)
        out.append(ins)
    for q in sorted(runs):
        flush(q)
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)


def _gate_named(name: str) -> Gate:
    from quspect.gates.standard import ALL_STANDARD

    for g in ALL_STANDARD:
        if g.name == name:
            return g
    from quspect.gates.gate import Gate

    return Gate(name=name, num_qubits=2)
