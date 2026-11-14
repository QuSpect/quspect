"""Quantum gate definitions for QuSpect."""

from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.gate import Gate
from quspect.gates.parameterized import ALL_PARAMETERIZED
from quspect.gates.registry import GateDefinition, gate_definition, register_gate
from quspect.gates.standard import ALL_STANDARD

__all__ = [
    "Gate",
    "ALL_STANDARD",
    "ALL_PARAMETERIZED",
    "GateDefinition",
    "gate_definition",
    "gate_matrix",
    "register_gate",
]
