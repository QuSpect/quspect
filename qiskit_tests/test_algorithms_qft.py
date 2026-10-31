"""Cross-validate ``build_qft_circuit`` against qiskit's QFT.

Established convention (all comparisons exact, no global phase, no
``reverse_qargs``):

* ``build_qft_circuit(n)`` == ``qiskit.circuit.library.QFTGate(n)`` ==
  the positive-sign DFT ``F[y, x] = exp(+2 pi i x y / N) / sqrt(N)`` on
  little-endian indices (both libraries are little-endian and both include
  the final swaps by default).
* ``build_qft_circuit(n, include_swaps=False)`` == qiskit
  ``QFT(n, do_swaps=False)`` == ``R @ F`` where ``R`` is the qubit-order
  reversal (the SWAP layer), i.e. output in bit-reversed order.
* ``build_qft_circuit(n, inverse=True)`` == ``QFTGate(n).inverse()`` ==
  ``F^dagger``; with ``include_swaps=False`` it is ``(R @ F)^dagger``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import qiskit
from qiskit.circuit.library import QFT, QFTGate
from qiskit.quantum_info import Operator, Statevector

from qiskit_tests._sim_helpers import bit_reversal_qiskit
from qiskit_tests.helpers import (
    assert_states_equal,
    assert_unitary_equal,
    quspect_circuit_unitary,
)
from quspect import QuantumCircuit
from quspect.algorithms import build_qft_circuit
from quspect.simulation import Simulator

SIZES = [1, 2, 3, 4, 5]


def _dft(n: int) -> np.ndarray:
    dim = 1 << n
    idx = np.arange(dim)
    return np.exp(2j * np.pi * np.outer(idx, idx) / dim) / math.sqrt(dim)


def _qft_unitary(n: int, **kwargs: bool) -> np.ndarray:
    return quspect_circuit_unitary(build_qft_circuit(n, **kwargs), num_qubits=n)


def _legacy_qft(n: int, **kwargs: bool) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return np.asarray(Operator(QFT(n, **kwargs)).data)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("n", SIZES)
def test_forward_qft_equals_qiskit_qftgate(n: int) -> None:
    """Forward QFT == qiskit ``QFTGate(n)`` == +sign DFT, no phase."""
    u = _qft_unitary(n)
    assert_unitary_equal(u, Operator(QFTGate(n)).data, atol=1e-10)
    assert_unitary_equal(u, _dft(n), atol=1e-10)
    assert_unitary_equal(u, _legacy_qft(n), atol=1e-10)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("n", SIZES)
def test_qft_without_swaps_equals_qiskit_do_swaps_false(n: int) -> None:
    """``include_swaps=False`` == ``QFT(n, do_swaps=False)`` == R @ F."""
    u = _qft_unitary(n, include_swaps=False)
    assert_unitary_equal(u, _legacy_qft(n, do_swaps=False), atol=1e-10)
    qc = qiskit.QuantumCircuit(n)
    qc.append(QFTGate(n), range(n))
    qc.compose(bit_reversal_qiskit(n), inplace=True)
    assert_unitary_equal(u, Operator(qc).data, atol=1e-10)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("n", SIZES)
def test_inverse_qft_equals_qiskit_inverse(n: int) -> None:
    """``inverse=True`` == ``QFTGate(n).inverse()`` == F^dagger."""
    u = _qft_unitary(n, inverse=True)
    assert_unitary_equal(u, Operator(QFTGate(n).inverse()).data, atol=1e-10)
    assert_unitary_equal(u, _dft(n).conj().T, atol=1e-10)
    assert_unitary_equal(u, _legacy_qft(n, inverse=True), atol=1e-10)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("n", SIZES)
def test_inverse_without_swaps(n: int) -> None:
    """``inverse=True, include_swaps=False`` is the adjoint of the no-swap QFT."""
    u = _qft_unitary(n, inverse=True, include_swaps=False)
    fwd = _qft_unitary(n, include_swaps=False)
    assert_unitary_equal(u, fwd.conj().T, atol=1e-10)
    assert_unitary_equal(u, _legacy_qft(n, inverse=True, do_swaps=False), atol=1e-10)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize(("n", "x"), [(3, 5), (4, 6), (5, 19)])
def test_qft_on_basis_state(n: int, x: int, simulator: Simulator) -> None:
    """QFT|x> amplitudes are exp(2 pi i x y / N) / sqrt(N) (little-endian)."""
    circ = QuantumCircuit()
    for q in range(n):
        if (x >> q) & 1:
            circ.x(q)
    circ.id(n - 1)
    circ.compose(build_qft_circuit(n), list(range(n)), inplace=True)
    sv = simulator.statevector(circ)
    ref = Statevector.from_int(x, 1 << n).evolve(QFTGate(n))
    assert_states_equal(sv, ref, up_to_phase=False, atol=1e-10)
    np.testing.assert_allclose(sv, _dft(n)[:, x], atol=1e-10)


@pytest.mark.parametrize("n", SIZES)
def test_qft_structure(n: int) -> None:
    """Names and gate counts: n H, n(n-1)/2 CP, n//2 SWAP."""
    fwd = build_qft_circuit(n)
    inv = build_qft_circuit(n, inverse=True)
    assert fwd.name == "qft"
    assert inv.name == "iqft"
    assert fwd.num_qubits == n
    names = [i.gate.name for i in fwd if i.gate is not None]
    assert names.count("h") == n
    assert names.count("cp") == n * (n - 1) // 2
    assert names.count("swap") == n // 2
    assert len(build_qft_circuit(n, include_swaps=False)) == len(fwd) - n // 2
    assert len(inv) == len(fwd)


@pytest.mark.parametrize(
    ("args", "kwargs", "exc"),
    [
        ((0,), {}, ValueError),
        ((-2,), {}, ValueError),
        ((2.0,), {}, TypeError),
        ((True,), {}, TypeError),
        ((2,), {"inverse": 1}, TypeError),
        ((2,), {"include_swaps": "yes"}, TypeError),
    ],
)
def test_qft_invalid_arguments(
    args: tuple[object, ...], kwargs: dict[str, object], exc: type[Exception]
) -> None:
    """Non-positive / non-int sizes and non-bool flags are rejected."""
    with pytest.raises(exc):
        build_qft_circuit(*args, **kwargs)  # type: ignore[arg-type]
