"""Pre-defined parameterized quantum gate *templates*.

Each gate template is a :class:`Gate` instance **without parameters**.
To create a concrete parameterized gate, call :meth:`Gate.with_params`
with the required numeric values.

Example:
    >>> from quspect.gates.parameterized import RX
    >>> RX
    Gate(name='rx', num_qubits=1)
    >>> rx_gate = RX.with_params(0.5)
    >>> rx_gate
    Gate(name='rx', num_qubits=1, params=(0.5,))
"""

from __future__ import annotations

from quspect.gates.gate import Gate

# =============================================================================
# Single-qubit parameterized gate templates
# =============================================================================

RX = Gate(name="rx", num_qubits=1, label="RX")
"""RX rotation gate template. Requires 1 parameter: ``theta``."""

RY = Gate(name="ry", num_qubits=1, label="RY")
"""RY rotation gate template. Requires 1 parameter: ``theta``."""

RZ = Gate(name="rz", num_qubits=1, label="RZ")
"""RZ rotation gate template. Requires 1 parameter: ``theta``."""

P = Gate(name="p", num_qubits=1, label="P")
"""Phase gate template. Requires 1 parameter: ``theta``."""

U1 = Gate(name="u1", num_qubits=1, label="U1")
"""U1 gate template. Requires 1 parameter: ``lambda``."""

U2 = Gate(name="u2", num_qubits=1, label="U2")
"""U2 gate template. Requires 2 parameters: ``phi``, ``lambda``."""

U3 = Gate(name="u3", num_qubits=1, label="U3")
"""U3 gate template. Requires 3 parameters: ``theta``, ``phi``, ``lambda``."""

# =============================================================================
# Two-qubit parameterized gate templates
# =============================================================================

CRX = Gate(name="crx", num_qubits=2, label="CRX")
"""Controlled-RX gate template. Requires 1 parameter: ``theta``."""

CRY = Gate(name="cry", num_qubits=2, label="CRY")
"""Controlled-RY gate template. Requires 1 parameter: ``theta``."""

CRZ = Gate(name="crz", num_qubits=2, label="CRZ")
"""Controlled-RZ gate template. Requires 1 parameter: ``theta``."""

CP = Gate(name="cp", num_qubits=2, label="CP")
"""Controlled-Phase gate template. Requires 1 parameter: ``theta``."""

CU3 = Gate(name="cu3", num_qubits=2, label="CU3")
"""Controlled-U3 gate template. Requires 3 params: ``theta``, ``phi``, ``lambda``."""

# =============================================================================
# Convenience
# =============================================================================

ALL_PARAMETERIZED: list[Gate] = [
    RX,
    RY,
    RZ,
    P,
    U1,
    U2,
    U3,
    CRX,
    CRY,
    CRZ,
    CP,
    CU3,
]
"""All pre-defined parameterized gate template instances."""
