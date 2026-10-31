"""Quantum Fourier transform circuit construction."""

from __future__ import annotations

import math

from quspect.circuit import QuantumCircuit


def build_qft_circuit(
    num_qubits: int,
    *,
    inverse: bool = False,
    include_swaps: bool = True,
) -> QuantumCircuit:
    """Build an exact quantum Fourier transform circuit.

    The returned circuit uses QuSpect's little-endian qubit convention and
    decomposes the transform into Hadamard, controlled-phase, and optional
    SWAP gates. For ``N = 2**num_qubits``, the forward circuit implements
    the positive-sign discrete Fourier transform
    ``|x> -> (1 / sqrt(N)) sum_y exp(+2*pi*i*x*y/N) |y>``.

    When ``include_swaps=False``, the forward circuit omits the final SWAP
    gates and leaves its output in bit-reversed order. Setting ``inverse=True``
    returns the adjoint of the complete gate sequence selected by
    ``include_swaps``; in particular, disabling swaps also omits them from the
    sequence whose adjoint is returned.

    Args:
        num_qubits: Number of qubits in the transform. Must be a positive
            integer and cannot be a boolean.
        inverse: If True, return the inverse quantum Fourier transform.
            Must be a boolean.
        include_swaps: If True, append the final swaps that restore natural
            output bit order. Must be a boolean.

    Returns:
        A circuit named ``"qft"`` for the forward transform or ``"iqft"``
        for the inverse transform.

    Raises:
        TypeError: If ``num_qubits`` is not an integer, or if ``inverse`` or
            ``include_swaps`` is not a boolean.
        ValueError: If ``num_qubits`` is not positive.

    Example:
        >>> qft = build_qft_circuit(3)
        >>> qft.name
        'qft'
        >>> qft.num_qubits
        3
        >>> len(qft)
        7
    """
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a boolean")
    if not isinstance(include_swaps, bool):
        raise TypeError("include_swaps must be a boolean")

    circuit = QuantumCircuit(name="qft")
    for target in range(num_qubits - 1, -1, -1):
        circuit.h(target)
        for control in range(target - 1, -1, -1):
            distance = target - control
            circuit.cp(math.pi / (1 << distance), control, target)

    if include_swaps:
        for lower in range(num_qubits // 2):
            circuit.swap(lower, num_qubits - 1 - lower)

    if inverse:
        inverse_circuit = circuit.dagger()
        inverse_circuit.name = "iqft"
        return inverse_circuit
    return circuit
