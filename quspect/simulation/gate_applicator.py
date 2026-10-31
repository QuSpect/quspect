"""Gate application via the C++ native backend.

This module provides :func:`apply_gate` and :func:`apply_gate_out_of_place`
which delegate to the optimized C++ ``apply_dense_unitary`` kernel.
"""

from __future__ import annotations

import numpy as np

from quspect.native import _core, _ensure_core


def _validate_state(state: np.ndarray) -> int:
    """Validate a state vector for in-place gate application.

    The C++ kernel can only update a writeable, C-contiguous, one-dimensional
    ``complex128`` buffer in place; any other array would be silently copied
    by the binding, so the caller's array would never change.

    Args:
        state: Candidate state vector.

    Returns:
        The number of qubits ``n`` such that ``len(state) == 2**n``.

    Raises:
        TypeError: If ``state`` is not a ``complex128`` numpy array.
        ValueError: If ``state`` is not one-dimensional, not C-contiguous,
            not writeable, or its length is not a positive power of two.
    """
    if not isinstance(state, np.ndarray) or state.dtype != np.complex128:
        dtype = getattr(state, "dtype", type(state).__name__)
        raise TypeError(
            f"State vector must be a numpy complex128 array, got {dtype}. "
            f"Convert it with np.ascontiguousarray(state, dtype=np.complex128)."
        )
    if state.ndim != 1:
        raise ValueError(
            f"State vector must be one-dimensional, got shape {state.shape}"
        )
    if not state.flags.c_contiguous:
        raise ValueError(
            "State vector must be C-contiguous for in-place gate application."
        )
    if not state.flags.writeable:
        raise ValueError("State vector must be writeable.")
    dim = int(state.shape[0])
    if dim < 1 or dim & (dim - 1):
        raise ValueError(f"State vector length must be a power of two, got {dim}")
    return dim.bit_length() - 1


def apply_gate(
    state: np.ndarray,
    matrix: np.ndarray,
    qubits: tuple[int, ...],
) -> None:
    """Apply a k-qubit unitary to a state vector in-place via C++ backend.

    The state vector is reshaped into an n-dimensional tensor, the gate
    qubits are transposed to the front, and the matrix is applied as a
    single matrix-vector product using BLAS-accelerated operations.

    Args:
        state: Writeable, C-contiguous ``complex128`` numpy array of shape
            ``(2^n,)``. It is updated in place.
        matrix: Complex unitary matrix of shape ``(2^k, 2^k)`` where
            ``k = len(qubits)``.
        qubits: Tuple of qubit indices to apply the gate to. Length must
            match the matrix dimension.

    Raises:
        TypeError: If ``state`` is not a ``complex128`` array.
        ValueError: If dimensions are inconsistent, the state length is not
            a power of two, the state is not C-contiguous / writeable, or
            the qubit indices are out of range or duplicated.

    Example:
        >>> import numpy as np
        >>> state = np.array([1, 0], dtype=np.complex128)
        >>> apply_gate(state, np.array([[0, 1], [1, 0]], dtype=complex), (0,))
        >>> state
        array([0.+0.j, 1.+0.j])
    """
    _ensure_core()
    n = _validate_state(state)
    k = len(qubits)
    expected_dim = 1 << k
    if matrix.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Matrix shape {matrix.shape} inconsistent with "
            f"{k} qubits (expected ({expected_dim}, {expected_dim}))"
        )
    if max(qubits, default=-1) >= n:
        raise ValueError(f"Qubit index {max(qubits)} out of range for {n}-qubit state")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"Duplicate qubit indices: {qubits}")
    unitary = np.ascontiguousarray(matrix, dtype=np.complex128)
    _core.apply_dense_unitary(state, unitary, np.array(qubits, dtype=np.uint32))


def apply_gate_out_of_place(
    state: np.ndarray,
    matrix: np.ndarray,
    qubits: tuple[int, ...],
) -> np.ndarray:
    """Apply a gate and return a new state vector without modifying the input.

    The input may have any numeric dtype or memory layout; it is copied into
    a fresh C-contiguous ``complex128`` array before the gate is applied.

    Args:
        state: Numeric numpy array of shape ``(2^n,)``.
        matrix: Complex unitary matrix of shape ``(2^k, 2^k)``.
        qubits: Tuple of qubit indices to apply the gate to.

    Returns:
        A new ``complex128`` numpy array with the gate applied.

    Raises:
        ValueError: As in :func:`apply_gate`.

    Example:
        >>> import numpy as np
        >>> state = np.array([1, 0], dtype=np.complex64)
        >>> x = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> apply_gate_out_of_place(state, x, (0,))
        array([0.+0.j, 1.+0.j])
    """
    result = np.array(state, dtype=np.complex128, order="C", copy=True)
    apply_gate(result, matrix, qubits)
    return result
