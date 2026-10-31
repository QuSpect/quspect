"""Native (C++) back end of the DAG optimization passes.

Each pass in :mod:`quspect.dag.passes` keeps its Python implementation as
the reference. When the extension module is built, the passes run the same
algorithm in C++ directly on the DAG storage core (same traversal order,
matching rules, DAG edits and floating-point operations, so the resulting
DAG is identical). The native path is used only when

* the DAG is a plain :class:`~quspect.dag.dag_circuit.DAGCircuit` on the
  native core and the pass is the library class itself (subclasses may
  override helper methods, so they keep the Python code);
* debug logging of the pass module is off (the Python code logs every edit);
* the DAG contains nothing the C++ decoder cannot represent exactly
  (non-canonical wire labels, subclassed nodes / instructions, non-float
  numeric parameters, ...). The C++ side checks this before touching the
  DAG and reports it, and the Python implementation then runs instead.

The environment variable ``QUSPECT_DAG_PASS_BACKEND`` selects the back end:
``auto`` (default) or ``python`` (always the reference implementation).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from quspect.dag.dag_circuit import DAGCircuit

DAG_PASS_BACKEND_ENV = "QUSPECT_DAG_PASS_BACKEND"
_BACKENDS = ("auto", "python")


def _load() -> Any | None:
    """Return the extension module when the native passes are available."""
    from quspect.dag import dag_circuit

    core = dag_circuit._native
    if core is None or not hasattr(core, "_register_dag_pass_types"):
        return None
    if not hasattr(DAGCircuit, "_native_gate_cancellation"):
        return None
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates.gate import Gate
    from quspect.gates.parameterized import RX, RY, RZ, P

    core._register_dag_pass_types(
        CircuitInstruction,
        Gate,
        InstructionType.GATE,
        InstructionType.MEASURE,
        InstructionType.BARRIER,
        InstructionType.CHANNEL,
        RX,
        RY,
        RZ,
        P,
    )
    core._enable_fast_instruction_repr(_repr_samples())
    return core


def _repr_samples() -> list[object]:
    """Instructions on which the native repr is checked against ``repr()``."""
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates.gate import Gate
    from quspect.gates.parameterized import RZ, U3
    from quspect.gates.standard import CX, H

    gate = InstructionType.GATE
    angles = (0.0, -0.0, 0.1, -2.5e-17, 1e16, 1e-300, 3.141592653589793)
    samples: list[object] = [
        CircuitInstruction(gate, (0,), gate=H),
        CircuitInstruction(gate, (3, 12), gate=CX, label='it\'s "x"\u00e9'),
        CircuitInstruction(gate, (1,), gate=U3.with_params(*angles[:3])),
        CircuitInstruction(gate, (0,), gate=Gate(name="g\n", num_qubits=1)),
        CircuitInstruction(InstructionType.MEASURE, (0, 1), (1, 0)),
        CircuitInstruction(InstructionType.BARRIER, ()),
    ]
    samples += [
        CircuitInstruction(gate, (7,), gate=RZ.with_params(a))
        for a in (*angles, float("inf"), float("-inf"), float("nan"))
    ]
    return samples


def _backend_from_env() -> str:
    value = os.environ.get(DAG_PASS_BACKEND_ENV, "auto").strip().lower() or "auto"
    if value not in _BACKENDS:
        raise ValueError(
            f"{DAG_PASS_BACKEND_ENV} must be one of {', '.join(_BACKENDS)}, "
            f"got {value!r}."
        )
    return value


_core: Any | None = _load()
_enabled: bool = _core is not None and _backend_from_env() == "auto"


def native_core() -> Any | None:
    """Return the extension module providing the native passes (or ``None``)."""
    return _core


def use_native(dag: object, logger: logging.Logger) -> bool:
    """Return whether a pass should try its native implementation on ``dag``."""
    return (
        _enabled and type(dag) is DAGCircuit and not logger.isEnabledFor(logging.DEBUG)
    )


def enabled() -> bool:
    """Return whether the native passes are active."""
    return _enabled


@contextmanager
def python_backend() -> Iterator[None]:
    """Run the DAG passes on their Python reference implementation (tests)."""
    global _enabled
    old = _enabled
    _enabled = False
    try:
        yield
    finally:
        _enabled = old
