"""Pre-defined standard (fixed-parameter) quantum gates.

Each gate is a module-level singleton :class:`Gate` instance. Standard
gates have no free parameters — their unitary matrices are constants.

Example:
    >>> from quspect.gates.standard import H, X, CX
    >>> H
    Gate(name='h', num_qubits=1)
    >>> CX
    Gate(name='cx', num_qubits=2)
"""

from __future__ import annotations

from quspect.gates.gate import Gate

# =============================================================================
# Single-qubit gates
# =============================================================================

I = Gate(name="id", num_qubits=1, label="I")  # noqa: E741
"""Identity gate. Leaves the qubit state unchanged."""

H = Gate(name="h", num_qubits=1, label="H")
"""Hadamard gate. Creates equal superposition from computational basis states."""

X = Gate(name="x", num_qubits=1, label="X")
"""Pauli-X gate (quantum NOT). Flips |0⟩ ↔ |1⟩."""

Y = Gate(name="y", num_qubits=1, label="Y")
"""Pauli-Y gate. Combination of X and Z with a phase factor."""

Z = Gate(name="z", num_qubits=1, label="Z")
"""Pauli-Z gate. Flips the phase of |1⟩."""

S = Gate(name="s", num_qubits=1, label="S")
"""Phase gate (π/2). Adds a phase of i to |1⟩."""

SDG = Gate(name="sdg", num_qubits=1, label="S†")
"""Conjugate transpose of S gate. Adds a phase of -i to |1⟩."""

T = Gate(name="t", num_qubits=1, label="T")
"""T gate (π/4). Adds a phase of exp(iπ/4) to |1⟩."""

TDG = Gate(name="tdg", num_qubits=1, label="T†")
"""Conjugate transpose of T gate. Adds a phase of exp(-iπ/4) to |1⟩."""

SX = Gate(name="sx", num_qubits=1, label="√X")
"""Square root of X: ``SX = ½[[1+i, 1-i], [1-i, 1+i]] = e^{iπ/4}·RX(π/2)``.

``SX · SX = X`` exactly. Native on IBM devices (basis ``{cx, rz, sx, x}``).
"""

SXDG = Gate(name="sxdg", num_qubits=1, label="√X†")
"""Conjugate transpose of :data:`SX` (``e^{-iπ/4}·RX(-π/2)``)."""

# =============================================================================
# Two-qubit gates
# =============================================================================

CX = Gate(name="cx", num_qubits=2, label="CX")
"""Controlled-NOT (CNOT) gate. Flips target if control is |1⟩."""

CZ = Gate(name="cz", num_qubits=2, label="CZ")
"""Controlled-Z gate. Applies Z to target if control is |1⟩."""

SWAP = Gate(name="swap", num_qubits=2, label="SWAP")
"""SWAP gate. Exchanges the states of two qubits."""

# =============================================================================
# Three-qubit gates
# =============================================================================

CCX = Gate(name="ccx", num_qubits=3, label="CCX")
"""Toffoli gate (CCNOT). Flips target if both controls are |1⟩."""

# =============================================================================
# Convenience: all standard gates in a list
# =============================================================================

ALL_STANDARD: list[Gate] = [
    I,
    H,
    X,
    Y,
    Z,
    S,
    SDG,
    T,
    TDG,
    SX,
    SXDG,
    CX,
    CZ,
    SWAP,
    CCX,
]
"""All pre-defined standard gate instances."""
