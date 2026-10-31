"""High-level (pre-decomposition) optimization of multi-qubit gates.

:class:`HighLevelOptimizePass` cancels pairs of identical self-inverse
multi-qubit gates (``ccx``, ``cx``, ``cz``, ``swap``) *before* they are
decomposed.  Once a Toffoli is lowered to CX and T gates, ``ccx · ccx`` no
longer looks like an identity to peephole passes (twelve CZ remain), while
at this level it is a plain cancellation.

The pair may be separated by gates that commute with it.  Commutation is
decided by a per-qubit rule that is exact (sufficient, never wrong): two
gates commute if, on every qubit they share, both act diagonally in the
same basis — ``Z`` (e.g. a control, ``rz``, ``t``, either qubit of ``cz``)
or ``X`` (e.g. a CX/Toffoli target, ``x``, ``rx``).  For instance
``ccx(0, 1, 2) · t(0) · cx(3, 2) · ccx(0, 1, 2)`` reduces to
``t(0) · cx(3, 2)`` (``t`` acts diagonally on the control, and qubit 2 is
the target of both the Toffoli and the CX).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction

if TYPE_CHECKING:
    from quspect.gates.gate import Gate

__all__ = ["HighLevelOptimizePass", "HighLevelStats"]

logger = logging.getLogger(__name__)

#: Single-qubit gates diagonal in the computational (``Z``) basis.
_Z_1Q = frozenset({"id", "z", "s", "sdg", "t", "tdg", "rz", "p", "u1"})
#: Single-qubit gates diagonal in the ``X`` basis.
_X_1Q = frozenset({"x", "rx", "sx", "sxdg"})
#: Two-qubit gates diagonal on both qubits.
_DIAGONAL_2Q = frozenset({"cz", "cp", "crz", "cu1", "rzz", "cs", "csdg"})
#: Controlled gates whose target operator is ``X``-diagonal.
_X_TARGET_2Q = frozenset({"cx", "crx", "csx"})

#: Self-inverse gates cancelled by the pass, with their operand symmetry:
#: ``"none"`` (order matters), ``"all"`` (fully symmetric), or ``"controls"``
#: (the first two operands may be exchanged).
_SELF_INVERSE = {"ccx": "controls", "cx": "none", "cz": "all", "swap": "all"}

#: Maximum number of gates scanned backwards for a cancellation partner.
_DEFAULT_WINDOW = 256


def basis_on(gate: Gate, qubits: tuple[int, ...], q: int) -> str | None:
    """Basis in which ``gate`` acts diagonally on qubit ``q``, if any.

    Args:
        gate: The gate.
        qubits: Its operands.
        q: One of the operands.

    Returns:
        ``"Z"`` (e.g. ``rz``, ``cz``, a control), ``"X"`` (e.g. ``x``, a
        CX/Toffoli target) or ``None`` (neither, or unknown).

    Example:
        >>> from quspect.gates.standard import CX
        >>> basis_on(CX, (0, 1), 0), basis_on(CX, (0, 1), 1)
        ('Z', 'X')
    """
    name = gate.name
    pos = qubits.index(q)
    if len(qubits) == 1:
        if name in _Z_1Q:
            return "Z"
        if name in _X_1Q:
            return "X"
        return None
    if name in _DIAGONAL_2Q and len(qubits) == 2:
        return "Z"
    if name in _X_TARGET_2Q and len(qubits) == 2:
        return "Z" if pos == 0 else "X"
    if name == "ccx" and len(qubits) == 3:
        return "Z" if pos < 2 else "X"
    return None


def _commute(a: CircuitInstruction, b: CircuitInstruction) -> bool:
    """Sufficient test that two gate instructions commute (see module doc)."""
    shared = set(a.qubits).intersection(b.qubits)
    if not shared:
        return True
    if a.gate is None or b.gate is None:
        return False
    for q in shared:
        basis = basis_on(a.gate, a.qubits, q)
        if basis is None or basis != basis_on(b.gate, b.qubits, q):
            return False
    return True


def _cancels(a: CircuitInstruction, b: CircuitInstruction) -> bool:
    """Whether ``a · b`` is the identity (identical self-inverse gates)."""
    ga, gb = a.gate, b.gate
    if ga is None or gb is None or ga.name != gb.name or ga.params or gb.params:
        return False
    symmetry = _SELF_INVERSE.get(ga.name)
    if symmetry is None or len(a.qubits) != len(b.qubits):
        return False
    if symmetry == "none":
        return a.qubits == b.qubits
    if symmetry == "all":
        return set(a.qubits) == set(b.qubits)
    return set(a.qubits[:2]) == set(b.qubits[:2]) and a.qubits[2:] == b.qubits[2:]


@dataclass
class HighLevelStats:
    """Statistics of the last :meth:`HighLevelOptimizePass.run` call.

    Attributes:
        pairs_cancelled: Number of gate pairs removed, by gate name.
    """

    pairs_cancelled: dict[str, int]


class HighLevelOptimizePass:
    """Cancel self-inverse multi-qubit gate pairs before decomposition.

    For each ``ccx``, ``cx``, ``cz`` or ``swap`` the pass scans backwards
    (at most ``window`` gates) over gates that commute with it — disjoint
    gates, or gates diagonal in the same basis on every shared qubit — and
    removes it together with an identical earlier gate (Toffoli controls
    and the ``cz``/``swap`` operands may appear in either order).  A
    measurement, barrier, channel or non-commuting gate on a shared qubit
    stops the scan.  Removals cascade:
    ``ccx · cx · cx · ccx`` is removed completely.

    The rewrite is exact (the unitary is unchanged, global phase
    included) and never increases the gate count.  Run it before
    :class:`~quspect.transpiler._passes.DecomposePass`.

    Args:
        window: Maximum number of instructions scanned backwards for a
            partner (at least 1).

    Raises:
        ValueError: If ``window`` is smaller than 1.

    Example:
        >>> from quspect import QuantumCircuit
        >>> qc = QuantumCircuit().ccx(0, 1, 2).t(0).cx(3, 2).ccx(1, 0, 2)
        >>> [i.gate.name for i in HighLevelOptimizePass().run(qc)]
        ['t', 'cx']
    """

    def __init__(self, *, window: int = _DEFAULT_WINDOW) -> None:
        if window < 1:
            raise ValueError("window must be at least 1.")
        self.window = window
        self.stats = HighLevelStats(pairs_cancelled={})

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of ``circuit`` with cancelling pairs removed.

        Args:
            circuit: Circuit to optimize (any gates; unknown gates simply
                block the scan on their qubits).

        Returns:
            A new :class:`~quspect.circuit.circuit.QuantumCircuit` (the
            input is not modified).

        Example:
            >>> from quspect import QuantumCircuit
            >>> qc = QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2).h(0)
            >>> len(HighLevelOptimizePass().run(qc))
            1
        """
        out: list[CircuitInstruction | None] = []
        cancelled: dict[str, int] = {}
        for ins in circuit:
            gate = ins.gate
            if ins.is_gate and gate is not None and gate.name in _SELF_INVERSE:
                partner = self._find_partner(out, ins)
                if partner is not None:
                    out[partner] = None
                    cancelled[gate.name] = cancelled.get(gate.name, 0) + 1
                    continue
            out.append(ins)
        self.stats = HighLevelStats(pairs_cancelled=cancelled)
        if cancelled:
            logger.debug("HighLevelOptimizePass cancelled pairs: %s", cancelled)
        result = QuantumCircuit(name=circuit.name)
        result._instructions = [ins for ins in out if ins is not None]
        return result

    def _find_partner(
        self, out: list[CircuitInstruction | None], ins: CircuitInstruction
    ) -> int | None:
        """Index in ``out`` of an earlier gate cancelling ``ins``, if any."""
        qubits = set(ins.qubits)
        scanned = 0
        for j in range(len(out) - 1, -1, -1):
            prev = out[j]
            if prev is None:
                continue
            scanned += 1
            if scanned > self.window:
                return None
            if qubits.isdisjoint(prev.qubits) and not prev.is_barrier:
                continue
            if not prev.is_gate:
                return None
            if _cancels(prev, ins):
                return j
            if not _commute(prev, ins):
                return None
        return None
