"""Circuit instructions for quantum circuit construction.

A CircuitInstruction represents a single operation in a quantum circuit:
a gate application, a measurement, a barrier, or a noise channel.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

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


# Module-level aliases: enum members are singletons, so identity tests are
# exact and avoid the class attribute lookup on hot paths.
_GATE = InstructionType.GATE
_MEASURE = InstructionType.MEASURE
_BARRIER = InstructionType.BARRIER
_CHANNEL = InstructionType.CHANNEL


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
    qubit into several classical bits), but one measurement instruction
    writes each classical bit at most once.

    Qubit and classical-bit indices may be any integer-like object (one
    with ``__index__``, such as a numpy integer); they are stored as plain
    ``int``. ``bool`` and ``float`` values are rejected with ``TypeError``,
    negative indices with ``ValueError``.

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
        qubits = self.qubits
        # Fast path for the common case (plain non-negative ints); anything
        # else goes through the full check, which raises the precise error
        # or returns the indices as plain ints (numpy integers, ...).
        for index in qubits:
            if type(index) is not int or index < 0:
                qubits = _normalize_bit_indices(qubits, "qubit")
                object.__setattr__(self, "qubits", qubits)
                break
        for index in self.clbits:
            if type(index) is not int or index < 0:
                object.__setattr__(
                    self, "clbits", _normalize_bit_indices(self.clbits, "clbit")
                )
                break

        kind = self.type
        if kind is _GATE:
            gate = self.gate
            if gate is None:
                raise ValueError("GATE instruction must have a non-None gate.")
            if len(qubits) != gate.num_qubits:
                raise ValueError(
                    f"Gate {gate.name} requires {gate.num_qubits} "
                    f"qubits, but {len(qubits)} were given."
                )
            if self.channel is not None:
                raise ValueError("GATE instruction should not have a channel.")
            n = len(qubits)
            if n > 2 or (n == 2 and qubits[0] == qubits[1]):
                _validate_distinct_qubits(qubits, f"Gate {gate.name}")
        elif kind is _MEASURE:
            if self.gate is not None:
                raise ValueError("MEASURE instruction should not have a gate.")
            if self.channel is not None:
                raise ValueError("MEASURE instruction should not have a channel.")
            if len(qubits) != len(self.clbits):
                raise ValueError(
                    f"MEASURE requires equal number of qubits and clbits, "
                    f"got {len(qubits)} qubits and {len(self.clbits)} clbits."
                )
            if len(qubits) > 1:
                _validate_distinct_clbits(self.clbits)
        elif kind is _BARRIER:
            if self.gate is not None:
                raise ValueError("BARRIER instruction should not have a gate.")
            if self.channel is not None:
                raise ValueError("BARRIER instruction should not have a channel.")
            # Barrier with empty qubits means "all active qubits" — valid.
        elif kind is _CHANNEL:
            if self.channel is None:
                raise ValueError("CHANNEL instruction must have a non-None channel.")
            if self.gate is not None:
                raise ValueError("CHANNEL instruction should not have a gate.")
            _validate_distinct_qubits(qubits, f"Channel {self.channel.name}")

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
        return self.type is _GATE

    @property
    def is_measure(self) -> bool:
        """Return True if this is a measurement instruction."""
        return self.type is _MEASURE

    @property
    def is_barrier(self) -> bool:
        """Return True if this is a barrier instruction."""
        return self.type is _BARRIER

    @property
    def is_channel(self) -> bool:
        """Return True if this instruction is a noise channel."""
        return self.type is _CHANNEL


def _normalize_bit_indices(indices: Sequence[object], kind: str) -> tuple[int, ...]:
    """Return bit indices as a tuple of plain non-negative ``int``.

    Any integer-like index (an object with ``__index__``, e.g. ``np.int64``
    from a loop over ``np.arange``) is accepted and converted, so that every
    later consumer (serialization, hashing, the native extension) sees plain
    ints.  Booleans (``bool`` and ``np.bool_``) and floats are not indices.

    Keeping this validation on the immutable instruction object covers every
    construction path (named gate helpers, generic append, measurements,
    barriers, and noise channels) and prevents malformed circuits from failing
    later while inferring their size.

    Raises:
        TypeError: If an index is not integer-like.
        ValueError: If an index is negative.
    """
    normalized: list[int] = []
    for index in indices:
        if type(index) is not int:
            if isinstance(index, (bool, np.bool_)) or not hasattr(index, "__index__"):
                raise TypeError(
                    f"{kind.capitalize()} index must be a non-negative integer, "
                    f"got {index!r}."
                )
            try:
                index = operator.index(index)
            except TypeError:
                raise TypeError(
                    f"{kind.capitalize()} index must be a non-negative integer, "
                    f"got {index!r}."
                ) from None
        if index < 0:
            raise ValueError(
                f"{kind.capitalize()} index must be a non-negative integer, "
                f"got {index}."
            )
        normalized.append(index)
    return tuple(normalized)


def _validate_distinct_clbits(clbits: tuple[int, ...]) -> None:
    """Require one measurement instruction to write each clbit once.

    ``measure([0, 1], [0, 0])`` would store two outcomes in one classical
    bit and silently keep only the last one.

    Raises:
        ValueError: If a classical bit appears more than once.
    """
    seen: set[int] = set()
    for clbit in clbits:
        if clbit in seen:
            raise ValueError(
                f"MEASURE writes classical bit {clbit} more than once "
                f"(clbits {clbits}); give every measured qubit its own "
                f"classical bit."
            )
        seen.add(clbit)


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


_GATE_BUILDER: Any = None


def _gate_instructions(
    items: Sequence[tuple[Gate, tuple[int, ...]]], label: str = ""
) -> list[CircuitInstruction]:
    """Gate instructions ``(gate, qubits)`` with an optional ``label``.

    The same as ``[CircuitInstruction(type=GATE, gate=g, qubits=qs,
    label=label) for g, qs in items]``.

    With the C++ extension the plain cases (a :class:`Gate` on one or two
    distinct non-negative ``int`` qubits) are built field by field, exactly
    as the dataclass ``__init__`` would set them (its validation cannot fail
    for them); every other item goes through the constructor.
    """
    global _GATE_BUILDER
    if _GATE_BUILDER is None:
        from quspect._native_loader import load_core

        try:
            core = load_core()
        except ImportError:  # pragma: no cover - broken build
            core = None
        builder = None if core is None else getattr(core, "_gate_instructions", None)
        if builder is None:
            _GATE_BUILDER = False
        else:

            def native_builder(
                items: Sequence[tuple[Gate, tuple[int, ...]]], label: str
            ) -> list[CircuitInstruction]:
                return builder(CircuitInstruction, _GATE, Gate, items, label)  # type: ignore[no-any-return]

            _GATE_BUILDER = native_builder
    if _GATE_BUILDER is False:
        return [
            CircuitInstruction(type=_GATE, gate=g, qubits=qs, clbits=(), label=label)
            for g, qs in items
        ]
    return _GATE_BUILDER(items, label)  # type: ignore[no-any-return]
