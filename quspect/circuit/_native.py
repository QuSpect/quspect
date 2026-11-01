"""Bridge between the Python circuit objects and the C++ circuit IR.

The native extension provides a contiguous circuit representation
(``_Circuit``: an array of compact instructions whose per-wire predecessors
form an implicit DAG, see ``quspect_native/ir/circuit.hpp``) and an
OpenQASM 2.0 fast path.  :class:`~quspect.gates.gate.Gate` and
:class:`~quspect.circuit.instruction.CircuitInstruction` stay frozen Python
dataclasses; this module registers their types with the extension so it can
convert between the two representations in bulk.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from quspect._native_loader import load_core

__all__ = ["bit_counts", "instruction_stats", "native_ir", "remap_qubits"]


@functools.cache
def native_ir() -> Any | None:
    """Return the native module with the circuit IR registered, or ``None``.

    The first call registers the Python gate and instruction types (and the
    standard gate singletons / parameterized gate templates) with the
    extension.

    Returns:
        The loaded extension module, or ``None`` when it is not built.

    Example:
        >>> from quspect.circuit._native import native_ir
        >>> native_ir() is None or hasattr(native_ir(), "_Circuit")
        True
    """
    core = load_core()
    if core is None or not hasattr(core, "_Circuit"):
        return None

    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates.gate import Gate
    from quspect.gates.parameterized import ALL_PARAMETERIZED
    from quspect.gates.standard import ALL_STANDARD

    gates = {g.name: g for g in (*ALL_STANDARD, *ALL_PARAMETERIZED)}
    core._register_ir_types(
        Gate,
        CircuitInstruction,
        InstructionType.GATE,
        InstructionType.MEASURE,
        InstructionType.BARRIER,
        InstructionType.CHANNEL,
        gates,
    )
    return core


@functools.cache
def _circuit_helpers() -> Any | None:
    """Return the extension module if it provides the bulk circuit helpers."""
    core = load_core()
    if core is None or not hasattr(core, "_instruction_stats"):
        return None
    return core


def bit_counts(instructions: list[Any]) -> tuple[int, int] | None:
    """``(num_qubits, num_clbits)`` of a plain instruction list, natively.

    Same values as :attr:`QuantumCircuit.num_qubits` / ``num_clbits``;
    ``None`` when the extension is not built or an instruction is not a
    plain :class:`CircuitInstruction` with ``int`` bit tuples (the caller
    then computes the counts itself).
    """
    core = _circuit_helpers()
    if core is None:
        return None
    from quspect.circuit.instruction import CircuitInstruction

    result: tuple[int, int] | None = core._instruction_bit_counts(
        instructions, CircuitInstruction
    )
    return result


def instruction_stats(instructions: list[Any]) -> tuple[int, int, int, int] | None:
    """``(two-qubit gates, depth, two-qubit depth, gates)``, natively.

    Depth counts every instruction with qubits except barriers; the
    two-qubit depth only two-qubit gates.  ``None`` when the native helper
    cannot be used (see :func:`bit_counts`).
    """
    core = _circuit_helpers()
    if core is None:
        return None
    from quspect.circuit.instruction import CircuitInstruction, InstructionType

    result: tuple[int, int, int, int] | None = core._instruction_stats(
        instructions, CircuitInstruction, InstructionType.GATE, InstructionType.BARRIER
    )
    return result


def remap_qubits(instructions: list[Any], mapping: Sequence[int]) -> list[Any] | None:
    """Copies of ``instructions`` with qubit ``q`` renamed to ``mapping[q]``.

    The bulk form of the :meth:`QuantumCircuit.compose` loop (clbits kept);
    ``None`` when the native helper cannot be used, including every case in
    which the loop would raise.
    """
    core = _circuit_helpers()
    if core is None:
        return None
    from quspect.circuit.instruction import CircuitInstruction

    result: list[Any] | None = core._remap_instruction_qubits(
        instructions, CircuitInstruction, mapping
    )
    return result
