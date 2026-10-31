"""Phase folding: merge diagonal rotations that act on the same parity.

:class:`PhaseFoldingPass` is the rotation-merging step of phase-polynomial
optimization (Amy–Maslov–Mosca "T-par" phase folding; Nam et al. 2018,
arXiv:1710.07345, "rotation merging").  It tracks, for every qubit, the
affine Boolean function ``L(v) ⊕ c`` of *path variables* ``v`` that the
qubit holds in the computational basis:

- every qubit starts with its own variable;
- ``cx`` XORs the control's function into the target, ``x``/``y`` flip the
  constant, ``swap`` exchanges two functions;
- diagonal gates (``cz``, ``cp``, ``crz``, ``rzz``, ``ccz``…) leave the
  functions unchanged;
- any other gate (``h``, ``rx``, ``u3``, the target of a ``ccx``…) gives
  each qubit it changes a fresh variable.

In the sum-over-paths form of the circuit, a diagonal single-qubit
rotation ``p(θ)`` on a qubit holding ``L ⊕ c`` contributes the phase factor
``e^{iθ(L ⊕ c)}``, which only depends on the path.  Two rotations whose
qubits hold the same ``L`` (at different times, possibly on different
qubits) can therefore be merged into one rotation at the position of the
first, and a rotation (merged or not) disappears when its angle is a
multiple of ``2π``.  The rewrite is exact up to a global phase, never adds gates and
never moves or touches a multi-qubit gate.  Typical gains: the
``cx·rz(θ)·cx·rz(−θ)`` remnants of decomposed multi-controlled phases
(e.g. ``crz(4π)`` or ``mcp`` with angles that are multiples of ``2π``),
and T/T† pairs of adjacent Toffoli decompositions.

Run it on a circuit whose two-qubit gates are ``cx`` and diagonal gates
(e.g. after decomposing to ``{cx, h, rz, …}``); a ``cx`` written as
``h·cz·h`` hides the parity structure.  Follow it with a commutation-aware
CX cancellation (:class:`~quspect.transpiler.HighLevelOptimizePass`): once
the rotations between them are gone, CX pairs that share a target or a
control cancel.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType

if TYPE_CHECKING:
    from quspect.gates.gate import Gate

__all__ = ["PhaseFoldingPass", "PhaseFoldingStats"]

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi

#: Fixed diagonal single-qubit gates and their phase ``λ`` (``diag(1, e^{iλ})``
#: up to global phase).
_FIXED_PHASE = {
    "z": math.pi,
    "s": 0.5 * math.pi,
    "sdg": -0.5 * math.pi,
    "t": 0.25 * math.pi,
    "tdg": -0.25 * math.pi,
}
#: Parameterized diagonal single-qubit gates (angle = first parameter).
_ROTATION_1Q = frozenset({"rz", "p", "u1"})
#: Gates that do not change the computational-basis value of any qubit.
_DIAGONAL = frozenset(
    {"id", "cz", "cp", "crz", "cu1", "rzz", "cs", "csdg", "ccz", "cpz"}
)
#: Merged angles within this distance of a multiple of ``2π`` are dropped
#: (the error is linear in the angle: ``|1 - e^{iε}| ≈ ε``).
_ZERO_ATOL = 1e-12


@dataclass
class PhaseFoldingStats:
    """Counters of the last :meth:`PhaseFoldingPass.run`.

    Attributes:
        rotations_in: Diagonal single-qubit rotations seen.
        rotations_out: Rotations left after merging.

    Example:
        >>> PhaseFoldingStats().rotations_in
        0
    """

    rotations_in: int = 0
    rotations_out: int = 0


@dataclass
class _Term:
    slot: int  # output position of the merged rotation
    qubit: int
    sign: int  # +1 if the qubit held L there, -1 if it held L ⊕ 1
    angle: float  # phase of e^{i·angle·L}
    count: int
    original: CircuitInstruction


def _phase_of(gate: Gate) -> float | None:
    """Phase ``λ`` of a diagonal single-qubit gate, or ``None``."""
    name = gate.name
    fixed = _FIXED_PHASE.get(name)
    if fixed is not None:
        return fixed
    if name in _ROTATION_1Q and len(gate.params) == 1:
        param = gate.params[0]
        if isinstance(param, Real):
            return float(param)
    return None


def _is_numeric(gate: Gate) -> bool:
    return all(isinstance(p, Real) for p in gate.params)


class PhaseFoldingPass:
    """Merge diagonal rotations acting on the same parity (phase folding).

    See the module docstring for the algorithm.  The pass is a single linear
    sweep; parities are Python integers used as bit sets over the path
    variables.  Measurements, resets, barriers and noise channels end all
    merging (rotations are never moved across them).

    Merged rotations are emitted as ``rz`` with an angle in ``(−π, π]``; a
    rotation that was not merged with any other is kept unchanged unless it
    is an identity up to global phase (e.g. ``rz(4π)``), which is removed.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().t(0).cx(0, 1).cx(1, 0).t(1).cx(1, 0).t(0)
        >>> [i.gate.name for i in PhaseFoldingPass().run(qc)]
        ['rz', 'cx', 'cx', 't', 'cx']
        >>> qc = QuantumCircuit().cx(0, 2).t(2).cx(1, 2).h(0).cx(1, 2)
        >>> qc = qc.tdg(2)  # qubit 2 holds q0 ⊕ q2 again (old value of q0)
        >>> [i.gate.name for i in PhaseFoldingPass().run(qc)]
        ['cx', 'cx', 'h', 'cx']
    """

    def __init__(self) -> None:
        self.stats = PhaseFoldingStats()

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of ``circuit`` with same-parity rotations merged.

        Args:
            circuit: Circuit to optimize (any instructions; unknown gates are
                handled as basis changes on their qubits).

        Returns:
            A new circuit, equal to ``circuit`` up to global phase, with no
            more gates than the input.

        Example:
            >>> from quspect import QuantumCircuit
            >>> qc = QuantumCircuit().rz(0.3, 0).cx(1, 0).cx(1, 0).rz(-0.3, 0)
            >>> [i.gate.name for i in PhaseFoldingPass().run(qc)]
            ['cx', 'cx']
        """
        stats = PhaseFoldingStats()
        parity: dict[int, int] = {}
        const: dict[int, int] = {}
        next_var = 0
        terms: dict[int, _Term] = {}
        out: list[CircuitInstruction | None] = []

        def wire(q: int) -> int:
            nonlocal next_var
            p = parity.get(q)
            if p is None:
                p = 1 << next_var
                next_var += 1
                parity[q] = p
                const[q] = 0
            return p

        def refresh(q: int) -> None:
            nonlocal next_var
            parity[q] = 1 << next_var
            const[q] = 0
            next_var += 1

        def flush() -> None:
            terms.clear()
            parity.clear()
            const.clear()

        for ins in circuit:
            gate = ins.gate
            if not ins.is_gate or gate is None:
                flush()
                out.append(ins)
                continue
            qubits = ins.qubits
            name = gate.name
            if len(qubits) == 1:
                q = qubits[0]
                phase = _phase_of(gate)
                if phase is not None:
                    stats.rotations_in += 1
                    key = wire(q)
                    sign = -1 if const[q] else 1
                    term = terms.get(key)
                    if term is None:
                        terms[key] = _Term(len(out), q, sign, sign * phase, 1, ins)
                        out.append(ins)
                    else:
                        term.angle += sign * phase
                        term.count += 1
                    continue
                if name in ("x", "y"):
                    wire(q)
                    const[q] ^= 1
                elif name not in _ROTATION_1Q and name != "id":
                    # 符号参数的对角门保持原位，不改变比特的奇偶函数
                    refresh(q)
                out.append(ins)
                continue
            if name == "cx" and len(qubits) == 2:
                c, t = qubits
                parity[t] = wire(t) ^ wire(c)
                const[t] ^= const[c]
            elif name == "swap" and len(qubits) == 2:
                a, b = qubits
                wire(a)
                wire(b)
                parity[a], parity[b] = parity[b], parity[a]
                const[a], const[b] = const[b], const[a]
            elif name in _DIAGONAL and _is_numeric(gate):
                pass
            elif name == "ccx" and len(qubits) == 3:
                refresh(qubits[2])
            else:
                for q in qubits:
                    refresh(q)
            out.append(ins)

        for term in terms.values():
            angle = math.remainder(term.sign * term.angle, _TWO_PI)
            if abs(angle) <= _ZERO_ATOL:
                out[term.slot] = None
                continue
            stats.rotations_out += 1
            if term.count > 1:
                out[term.slot] = _rz(angle, term.qubit)

        self.stats = stats
        if stats.rotations_out < stats.rotations_in:
            logger.debug(
                "PhaseFoldingPass: %d -> %d diagonal rotations",
                stats.rotations_in,
                stats.rotations_out,
            )
        return QuantumCircuit._from_instruction_snapshot(
            [ins for ins in out if ins is not None], name=circuit.name
        )


def _rz(angle: float, qubit: int) -> CircuitInstruction:
    from quspect.gates.parameterized import RZ

    return CircuitInstruction(
        type=InstructionType.GATE, gate=RZ.with_params(angle), qubits=(qubit,)
    )
