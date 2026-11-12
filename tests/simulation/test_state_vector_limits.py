"""A state vector that cannot be allocated is reported as such."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect.simulation import Simulator


@pytest.mark.parametrize("num_qubits", [56, 59, 60, 63, 64, 100])
def test_impossible_state_vector_is_a_clear_memory_error(num_qubits: int) -> None:
    """2^56 amplitudes and more exceed any address space: nothing is allocated."""
    top = num_qubits - 1
    message = f"state vector of {num_qubits} qubits"
    with pytest.warns(ResourceWarning), pytest.raises(MemoryError, match=message):
        Simulator().run(QuantumCircuit().h(top).measure([top], [0]), shots=1)
    with pytest.raises(MemoryError, match=message) as excinfo:
        Simulator().statevector(QuantumCircuit().h(top))
    assert f"2^{num_qubits} amplitudes" in str(excinfo.value)
    assert str(excinfo.value) != "std::bad_alloc"
