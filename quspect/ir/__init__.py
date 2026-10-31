"""IR (Intermediate Representation) module for QuSpect.

This module provides exporters to convert QuSpect circuits into
standard and custom intermediate representations:

- :class:`QASM2Exporter` — export to OpenQASM 2.0 (hardware-facing)
- :mod:`quspect.ir.qsal` — QuSpect Abstract Language (native IR)

The IR module sits at the **export boundary** of QuSpect, converting
internal representations (:class:`~quspect.circuit.circuit.QuantumCircuit`,
:class:`~quspect.operators.qubit_operator.QubitOperator`) into formats
consumable by external tools and backends.
"""

from __future__ import annotations

from quspect.ir.compilation import (
    GateDecompositionRule,
    clear_extra_rules,
    decompose_circuit,
    decompose_deep,
    decompose_gate,
    get_rules_for,
    register_decomposition,
)
from quspect.ir.errors import (
    IRDeserializationError,
    IRSerializationError,
    ParameterizedGateError,
    QSALVersionError,
    UnsupportedGateError,
)
from quspect.ir.qasm2 import (
    QASM2ExportConfig,
    QASM2Exporter,
    QASM2ExportMode,
)
from quspect.ir.qasm_importer import (
    QASM2Importer,
    QASM2ParseError,
    qasm2_loads,
)
from quspect.ir.qsal import (
    QSALDocument,
    dump,
    dumps,
    load,
    loads,
)

__all__ = [
    # OpenQASM 2.0
    "QASM2ExportConfig",
    "QASM2Exporter",
    "QASM2ExportMode",
    "QASM2Importer",
    "QASM2ParseError",
    "qasm2_loads",
    # Gate compilation
    "GateDecompositionRule",
    "clear_extra_rules",
    "decompose_circuit",
    "decompose_deep",
    "decompose_gate",
    "get_rules_for",
    "register_decomposition",
    # QSAL
    "QSALDocument",
    "dump",
    "dumps",
    "load",
    "loads",
    # Errors
    "IRDeserializationError",
    "IRSerializationError",
    "ParameterizedGateError",
    "QSALVersionError",
    "UnsupportedGateError",
]
