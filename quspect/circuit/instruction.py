"""Circuit instructions for quantum circuit construction.

A CircuitInstruction represents a single operation in a quantum circuit:
a gate application, a measurement, a barrier, or a noise channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from quspect.gates.gate import Gate

if TYPE_CHECKING:
    from quspect.noise.channel import NoiseChannel


class InstructionType(Enum):
    """Type of a circuit instruction.

    Attributes:
        GATE: A unitary gate application.
        MEASURE: A projective measurement.
        BARRIER: A barrier preventing optimization reordering.
        CHANNEL: A CPTP quantum noise channel (non-unitary).
    """

    GATE = auto()
    MEASURE = auto()
    BARRIER = auto()
    CHANNEL = auto()


@dataclass(frozen=True)
class CircuitInstruction:
    """A single instruction in a quantum circuit.

    Represents one operation with specified qubit and classical
    bit arguments. Instructions are immutable value objects.

    Note:
        Measurement is represented as an instruction with
        ``type=InstructionType.MEASURE`` rather than as a Gate,
        because measurement is not a unitary operation (architecture
        decision 7.3 — 方案 B).

    Note:
        Noise channels use ``type=InstructionType.CHANNEL`` and the
        ``channel`` field, keeping them distinct from unitary gates
        at the type level.

    Gate and channel instructions must act on distinct qubits: an
    instruction such as ``cx(0, 0)`` is rejected with ``ValueError`` at
    construction time. Measurements may repeat a qubit (measuring the same
    qubit into several classical bits).

    Attributes:
        type: The type of instruction (GATE, MEASURE, BARRIER, or CHANNEL).
        gate: The quantum gate to apply (for GATE type; None otherwise).
        channel: The noise channel to apply (for CHANNEL type; None otherwise).
        qubits: Tuple of qubit indices this instruction involves.
        clbits: Tuple of classical bit indices (for MEASURE type).
        label: Optional human-readable label.

    Example:
        >>> from quspect.gates.standard import H, CX
        >>> instr1 = CircuitInstruction(
        ...     type=InstructionType.GATE, gate=H, qubits=(0,)
        ... )
        >>> instr2 = CircuitInstruction(
        ...     type=InstructionType.GATE, gate=CX, qubits=(0, 1)
        ... )
        >>> instr3 = CircuitInstruction(
        ...     type=InstructionType.MEASURE, qubits=(0, 1), clbits=(0, 1)
        ... )
    """

    type: InstructionType
    qubits: tuple[int, ...]
    clbits: tuple[int, ...] = field(default=())
    gate: Gate | None = None
    channel: NoiseChannel | None = field(default=None, compare=False, hash=False)
    label: str = ""

    def __post_init__(self) -> None:
        """Validate instruction consistency."""
        _validate_bit_indices(self.qubits, "qubit")
        _validate_bit_indices(self.clbits, "clbit")

        if self.type == InstructionType.GATE:
            if self.gate is None:
                raise ValueError("GATE instruction must have a non-None gate.")
            if len(self.qubits) != self.gate.num_qubits:
                raise ValueError(
                    f"Gate {self.gate.name} requires {self.gate.num_qubits} "
                    f"qubits, but {len(self.qubits)} were given."
                )
            if self.channel is not None:
                raise ValueError("GATE instruction should not have a channel.")
            _validate_distinct_qubits(self.qubits, f"Gate {self.gate.name}")
        elif self.type == InstructionType.MEASURE:
            if self.gate is not None:
                raise ValueError("MEASURE instruction should not have a gate.")
            if self.channel is not None:
                raise ValueError("MEASURE instruction should not have a channel.")
            if len(self.qubits) != len(self.clbits):
                raise ValueError(
                    f"MEASURE requires equal number of qubits and clbits, "
                    f"got {len(self.qubits)} qubits and {len(self.clbits)} clbits."
                )
        elif self.type == InstructionType.BARRIER:
            if self.gate is not None:
                raise ValueError("BARRIER instruction should not have a gate.")
            if self.channel is not None:
                raise ValueError("BARRIER instruction should not have a channel.")
            # Barrier with empty qubits means "all active qubits" — valid.
        elif self.type == InstructionType.CHANNEL:
            if self.channel is None:
                raise ValueError("CHANNEL instruction must have a non-None channel.")
            if self.gate is not None:
                raise ValueError("CHANNEL instruction should not have a gate.")
            _validate_distinct_qubits(self.qubits, f"Channel {self.channel.name}")

    def matrix(self) -> np.ndarray | None:
        """Return the unitary matrix of this instruction, if applicable.

        Matrix computation is delegated to
        :func:`~quspect.gates._matrix_utils.gate_matrix`.

        Channel instructions do not have a single unitary matrix; they have
        a Kraus operator decomposition instead, available via
        ``instr.channel.kraus_operators()``.

        Returns:
            The gate matrix for GATE instructions, None otherwise.

        Raises:
            ValueError: If the gate contains symbolic (unbound) parameters
                or an incorrect number of parameters.

        Example:
            >>> from quspect.gates.standard import X
            >>> CircuitInstruction(
            ...     type=InstructionType.GATE, gate=X, qubits=(0,)
            ... ).matrix().shape
            (2, 2)
        """
        if self.type == InstructionType.GATE and self.gate is not None:
            from quspect.gates._matrix_utils import gate_matrix

            return gate_matrix(self.gate)
        return None

    @property
    def is_gate(self) -> bool:
        """Return True if this is a gate instruction."""
        return self.type == InstructionType.GATE

    @property
    def is_measure(self) -> bool:
        """Return True if this is a measurement instruction."""
        return self.type == InstructionType.MEASURE

    @property
    def is_barrier(self) -> bool:
        """Return True if this is a barrier instruction."""
        return self.type == InstructionType.BARRIER

    @property
    def is_channel(self) -> bool:
        """Return True if this instruction is a noise channel."""
        return self.type == InstructionType.CHANNEL


def _validate_bit_indices(indices: tuple[int, ...], kind: str) -> None:
    """Require bit indices to be non-negative integers.

    Keeping this validation on the immutable instruction object covers every
    construction path (named gate helpers, generic append, measurements,
    barriers, and noise channels) and prevents malformed circuits from failing
    later while inferring their size.
    """
    for index in indices:
        if type(index) is int and index >= 0:  # fast path (the common case)
            continue
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"{kind.capitalize()} index must be a non-negative integer, "
                f"got {index!r}."
            )
        if index < 0:
            raise ValueError(
                f"{kind.capitalize()} index must be a non-negative integer, "
                f"got {index}."
            )


def _validate_distinct_qubits(qubits: tuple[int, ...], what: str) -> None:
    """Require the qubits of a gate or channel instruction to be distinct.

    Raises:
        ValueError: If any qubit index appears more than once.
    """
    n = len(qubits)
    if n < 2 or (n == 2 and qubits[0] != qubits[1]):
        return
    if len(set(qubits)) != n:
        raise ValueError(f"{what} requires distinct qubits, got duplicate {qubits}.")
