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
from typing import Any

from quspect._native_loader import load_core

__all__ = ["native_ir"]


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
