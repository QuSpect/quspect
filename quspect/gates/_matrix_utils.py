"""Matrix computation for quantum gates.

This module provides the :func:`gate_matrix` function that computes
the unitary matrix for any :class:`Gate` instance. Standard gates
use pre-computed constant matrices; parameterized gates compute
their matrices from gate parameters at call time.

When the C++ extension is built, parameterized matrices come from its
deterministic implementation (correctly rounded sin / cos, fixed-order
complex products), so they are bit-identical on every platform; the Python
formulas below are the fallback (same formulas, platform libm).
"""

from __future__ import annotations

import functools
import math
from numbers import Real
from typing import Any

import numpy as np

from quspect.gates.gate import Gate

# =============================================================================
# Constants
# =============================================================================

I_MATRIX: np.ndarray = np.array([[1, 0], [0, 1]], dtype=complex)
X_MATRIX: np.ndarray = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MATRIX: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MATRIX: np.ndarray = np.array([[1, 0], [0, -1]], dtype=complex)
H_MATRIX: np.ndarray = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S_MATRIX: np.ndarray = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG_MATRIX: np.ndarray = np.array([[1, 0], [0, -1j]], dtype=complex)
# cos(π/4)、sin(π/4)（π/4 取 math.pi / 4）的正确舍入值，写成常数而不调用平台 libm：
# 与确定性数值层的 p(π/4) 逐位相同，且在所有平台上一致。
_COS_PI4 = float.fromhex("0x1.6a09e667f3bcdp-1")
_SIN_PI4 = float.fromhex("0x1.6a09e667f3bccp-1")
T_MATRIX: np.ndarray = np.array([[1, 0], [0, _COS_PI4 + 1j * _SIN_PI4]], dtype=complex)
TDG_MATRIX: np.ndarray = np.array(
    [[1, 0], [0, _COS_PI4 - 1j * _SIN_PI4]], dtype=complex
)
SX_MATRIX: np.ndarray = (
    np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2
)
SXDG_MATRIX: np.ndarray = (
    np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex) / 2
)

CNOT_MATRIX: np.ndarray = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)
CZ_MATRIX: np.ndarray = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ],
    dtype=complex,
)
SWAP_MATRIX: np.ndarray = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)
CCX_MATRIX: np.ndarray = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=complex,
)

# =============================================================================
# Standard gate matrix lookup
# =============================================================================

_STANDARD_MATRICES: dict[str, np.ndarray] = {
    "id": I_MATRIX,
    "x": X_MATRIX,
    "y": Y_MATRIX,
    "z": Z_MATRIX,
    "h": H_MATRIX,
    "s": S_MATRIX,
    "sdg": SDG_MATRIX,
    "t": T_MATRIX,
    "tdg": TDG_MATRIX,
    "sx": SX_MATRIX,
    "sxdg": SXDG_MATRIX,
    "cx": CNOT_MATRIX,
    "cz": CZ_MATRIX,
    "swap": SWAP_MATRIX,
    "ccx": CCX_MATRIX,
}


# =============================================================================
# Public API
# =============================================================================


def gate_matrix(gate: Gate) -> np.ndarray:
    """Return the unitary matrix for a gate.

    Standard gates return pre-computed constant matrices. Parameterized
    gates compute their matrix from ``gate.params`` on each call. Every
    parameter must be a concrete real number (or a constant
    :class:`~quspect.circuit.parameter.ParameterExpression`) and the number
    of parameters must match the gate exactly.

    Args:
        gate: The gate to compute the matrix for.

    Returns:
        A ``(2^n, 2^n)`` complex numpy array representing the unitary
        transformation, where ``n = gate.num_qubits``.

    Raises:
        KeyError: If ``gate.name`` is not a recognized gate.
        ValueError: If the gate requires parameters but none are provided,
            if the parameter count is incorrect (too few or too many), or
            if any parameter is symbolic / not a real number.

    Example:
        >>> from quspect.gates.parameterized import RX
        >>> gate_matrix(RX.with_params(0.0)).shape
        (2, 2)
    """
    if gate.name in _STANDARD_MATRICES:
        if gate.params:
            raise ValueError(
                f"Gate '{gate.name}' takes no parameters, got {len(gate.params)}"
            )
        return _STANDARD_MATRICES[gate.name].copy()

    return _parameterized_matrix(gate)


#: Number of parameters required by each known parameterized gate.
_PARAM_COUNTS: dict[str, int] = {
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "p": 1,
    "u1": 1,
    "u2": 2,
    "u3": 3,
    "crx": 1,
    "cry": 1,
    "crz": 1,
    "cp": 1,
    "cu3": 3,
    "rxx": 1,
    "ryy": 1,
    "rzz": 1,
}


def _concrete_param(gate: Gate, value: object) -> float:
    """Convert one gate parameter to ``float`` or raise ``ValueError``.

    Symbolic parameters (``Parameter`` or non-constant
    ``ParameterExpression``) cannot produce a numeric matrix.
    """
    if isinstance(value, Real):
        return float(value)
    if not isinstance(value, (str, bytes, complex)):
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    raise ValueError(
        f"Gate '{gate.name}' has a non-numeric or symbolic parameter "
        f"{value!r}; bind all parameters before computing its matrix."
    )


def _parameterized_matrix(gate: Gate) -> np.ndarray:
    """Compute the matrix for a parameterized gate."""
    name = gate.name
    if name not in _PARAM_COUNTS:
        raise KeyError(f"Unknown gate name: '{gate.name}'")

    if not gate.params:
        raise ValueError(
            f"Gate '{gate.name}' requires parameters, but none were provided"
        )
    expected = _PARAM_COUNTS[name]
    if len(gate.params) != expected:
        raise ValueError(
            f"Gate '{name}' requires {expected} parameter"
            f"{'s' if expected != 1 else ''}, got {len(gate.params)}"
        )
    params = [_concrete_param(gate, value) for value in gate.params]

    native = _native_gate_matrix()
    if native is not None:
        m = native(name, params)
        if m is not None:
            return m  # type: ignore[no-any-return]

    if name == "rx":
        return _rx_matrix(params[0])
    if name == "ry":
        return _ry_matrix(params[0])
    if name == "rz":
        return _rz_matrix(params[0])
    if name == "p":
        return _phase_matrix(params[0])
    if name == "u1":
        return _u1_matrix(params[0])
    if name == "u2":
        return _u2_matrix(params[0], params[1])
    if name == "u3":
        return _u3_matrix(params[0], params[1], params[2])
    if name == "crx":
        return _crx_matrix(params[0])
    if name == "cry":
        return _cry_matrix(params[0])
    if name == "crz":
        return _crz_matrix(params[0])
    if name == "cp":
        return _cp_matrix(params[0])
    if name in ("rxx", "ryy", "rzz"):
        return _pauli_pair_matrix(name[1], params[0])
    return _cu3_matrix(params[0], params[1], params[2])


@functools.cache
def _native_gate_matrix() -> Any | None:
    """The extension module's ``_gate_matrix`` (``None`` without it).

    Honors ``QUSPECT_SYNTHESIS_BACKEND=python`` (pure-Python numerics).
    """
    import os

    if os.environ.get("QUSPECT_SYNTHESIS_BACKEND", "auto") == "python":
        return None
    from quspect._native_loader import load_core

    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        return None
    return getattr(core, "_gate_matrix", None) if core is not None else None


# =============================================================================
# Single-qubit parameterized gate matrices
# =============================================================================


def _rx_matrix(theta: float) -> np.ndarray:
    """Rotation around X axis: exp(-i * θ/2 * X)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry_matrix(theta: float) -> np.ndarray:
    """Rotation around Y axis: exp(-i * θ/2 * Y)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz_matrix(theta: float) -> np.ndarray:
    """Rotation around Z axis: exp(-i * θ/2 * Z)."""
    e_pos = math.cos(theta / 2) - 1j * math.sin(theta / 2)
    e_neg = math.cos(theta / 2) + 1j * math.sin(theta / 2)
    return np.array([[e_pos, 0], [0, e_neg]], dtype=complex)


def _phase_matrix(theta: float) -> np.ndarray:
    """Phase gate: diag(1, exp(iθ))."""
    return np.array(
        [[1, 0], [0, math.cos(theta) + 1j * math.sin(theta)]], dtype=complex
    )


def _u1_matrix(lam: float) -> np.ndarray:
    """U1 gate: diag(1, exp(iλ)). Same as Phase gate."""
    return _phase_matrix(lam)


def _u2_matrix(phi: float, lam: float) -> np.ndarray:
    """U2 gate: U3(π/2, φ, λ)."""
    return _u3_matrix(math.pi / 2, phi, lam)


def _u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Universal single-qubit gate U3(θ, φ, λ)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    e_ip = math.cos(phi) + 1j * math.sin(phi)
    e_il = math.cos(lam) + 1j * math.sin(lam)
    return np.array(
        [
            [c, -s * e_il],
            [s * e_ip, c * e_ip * e_il],
        ],
        dtype=complex,
    )


# =============================================================================
# Two-qubit parameterized gate matrices
# =============================================================================


def _crx_matrix(theta: float) -> np.ndarray:
    """Controlled-RX gate."""
    rx = _rx_matrix(theta)
    result = np.eye(4, dtype=complex)
    result[2:, 2:] = rx
    return result


def _cry_matrix(theta: float) -> np.ndarray:
    """Controlled-RY gate."""
    ry = _ry_matrix(theta)
    result = np.eye(4, dtype=complex)
    result[2:, 2:] = ry
    return result


def _crz_matrix(theta: float) -> np.ndarray:
    """Controlled-RZ gate."""
    rz = _rz_matrix(theta)
    result = np.eye(4, dtype=complex)
    result[2:, 2:] = rz
    return result


def _cp_matrix(theta: float) -> np.ndarray:
    """Controlled-Phase gate: diag(1, 1, 1, exp(iθ))."""
    result = np.eye(4, dtype=complex)
    result[3, 3] = math.cos(theta) + 1j * math.sin(theta)
    return result


def _cu3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Controlled-U3 gate."""
    u3 = _u3_matrix(theta, phi, lam)
    result = np.eye(4, dtype=complex)
    result[2:, 2:] = u3
    return result


def _pauli_pair_matrix(axis: str, theta: float) -> np.ndarray:
    """Two-qubit Pauli rotation ``exp(-i * θ/2 * P⊗P)`` for ``P`` = X, Y or Z.

    Equals ``cos(θ/2) * I - i * sin(θ/2) * P⊗P``; the matrix is symmetric in
    the two qubits, so the operand order does not matter.
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if axis == "z":
        # Z⊗Z = diag(1, -1, -1, 1)
        even = c - 1j * s
        odd = c + 1j * s
        return np.diag(np.array([even, odd, odd, even], dtype=complex))
    # X⊗X and Y⊗Y flip both qubits; Y⊗Y has -1 on the |00>↔|11> corners.
    corner = -1j * s if axis == "x" else 1j * s
    flip = -1j * s
    return np.array(
        [
            [c, 0, 0, corner],
            [0, c, flip, 0],
            [0, flip, c, 0],
            [corner, 0, 0, c],
        ],
        dtype=complex,
    )
