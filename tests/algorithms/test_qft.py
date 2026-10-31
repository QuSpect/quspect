"""Tests for the quantum Fourier transform circuit builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect import QuantumCircuit, Simulator
from quspect.algorithms import build_qft_circuit


def _prepare_basis_state(num_qubits: int, basis_state: int) -> QuantumCircuit:
    circuit = QuantumCircuit()
    for qubit in range(num_qubits):
        if basis_state & (1 << qubit):
            circuit.x(qubit)
    return circuit


def _explicit_dft_column(
    num_qubits: int,
    basis_state: int,
    *,
    inverse: bool,
) -> np.ndarray:
    dimension = 1 << num_qubits
    output_indices = np.arange(dimension)
    sign = -1 if inverse else 1
    return np.exp(
        sign * 2j * np.pi * basis_state * output_indices / dimension
    ) / math.sqrt(dimension)


def _reverse_bits(value: int, width: int) -> int:
    result = 0
    for index in range(width):
        result |= ((value >> index) & 1) << (width - 1 - index)
    return result


def test_three_qubit_qft_has_exact_gate_structure() -> None:
    circuit = build_qft_circuit(3)

    assert circuit.name == "qft"
    assert circuit.num_qubits == 3
    assert [
        (instruction.gate.name, instruction.qubits, instruction.gate.params)
        for instruction in circuit.instructions
        if instruction.gate is not None
    ] == [
        ("h", (2,), ()),
        ("cp", (1, 2), (math.pi / 2,)),
        ("cp", (0, 2), (math.pi / 4,)),
        ("h", (1,), ()),
        ("cp", (0, 1), (math.pi / 2,)),
        ("h", (0,), ()),
        ("swap", (0, 2), ()),
    ]


@pytest.mark.parametrize("num_qubits", range(1, 7))
@pytest.mark.parametrize("include_swaps", [False, True])
def test_qft_gate_count(num_qubits: int, include_swaps: bool) -> None:
    circuit = build_qft_circuit(num_qubits, include_swaps=include_swaps)
    expected = num_qubits + num_qubits * (num_qubits - 1) // 2
    if include_swaps:
        expected += num_qubits // 2

    assert len(circuit) == expected


@pytest.mark.parametrize("num_qubits", range(1, 5))
@pytest.mark.parametrize("inverse", [False, True])
def test_qft_matches_explicit_dft(num_qubits: int, inverse: bool) -> None:
    qft = build_qft_circuit(num_qubits, inverse=inverse)
    simulator = Simulator(seed=7)

    for basis_state in range(1 << num_qubits):
        prepared = _prepare_basis_state(num_qubits, basis_state)
        circuit = prepared.compose(qft, qubits=range(num_qubits))
        actual = simulator.statevector(circuit)
        expected = _explicit_dft_column(
            num_qubits,
            basis_state,
            inverse=inverse,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_inverse_is_complete_adjoint_and_round_trips_state() -> None:
    qft = build_qft_circuit(3)
    iqft = build_qft_circuit(3, inverse=True)
    expected_inverse = qft.dagger()

    assert iqft.name == "iqft"
    assert iqft.instructions == expected_inverse.instructions

    prepared = QuantumCircuit().h(0).ry(0.37, 1).x(2).cx(1, 2)
    simulator = Simulator(seed=11)
    expected_state = simulator.statevector(prepared)
    transformed = prepared.compose(qft, qubits=range(3))
    round_tripped = transformed.compose(iqft, qubits=range(3))
    actual_state = simulator.statevector(round_tripped)

    np.testing.assert_allclose(actual_state, expected_state, atol=1e-12)


@pytest.mark.parametrize("basis_state", range(8))
def test_qft_without_swaps_has_bit_reversed_output(basis_state: int) -> None:
    num_qubits = 3
    qft = build_qft_circuit(num_qubits, include_swaps=False)
    prepared = _prepare_basis_state(num_qubits, basis_state)
    circuit = prepared.compose(qft, qubits=range(num_qubits))

    actual = Simulator(seed=13).statevector(circuit)
    natural_order = _explicit_dft_column(
        num_qubits,
        basis_state,
        inverse=False,
    )
    bit_reversed_order = natural_order[
        [_reverse_bits(index, num_qubits) for index in range(1 << num_qubits)]
    ]

    np.testing.assert_allclose(actual, bit_reversed_order, atol=1e-12)


@pytest.mark.parametrize(
    "num_qubits",
    [True, False, 1.0, "3", None, np.int64(3)],
)
def test_num_qubits_rejects_non_integer_values(num_qubits: object) -> None:
    with pytest.raises(TypeError, match="num_qubits"):
        build_qft_circuit(num_qubits)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1, -10])
def test_num_qubits_must_be_positive(num_qubits: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_qft_circuit(num_qubits)


@pytest.mark.parametrize("inverse", [0, 1, None, "false"])
def test_inverse_requires_strict_boolean(inverse: object) -> None:
    with pytest.raises(TypeError, match="inverse"):
        build_qft_circuit(3, inverse=inverse)  # type: ignore[arg-type]


@pytest.mark.parametrize("include_swaps", [0, 1, None, "true"])
def test_include_swaps_requires_strict_boolean(include_swaps: object) -> None:
    with pytest.raises(TypeError, match="include_swaps"):
        build_qft_circuit(3, include_swaps=include_swaps)  # type: ignore[arg-type]


def test_qft_composes_onto_non_contiguous_qubits() -> None:
    qft = build_qft_circuit(3)
    composed = QuantumCircuit().compose(qft, qubits=[1, 3, 5])

    assert composed.num_qubits == 6
    assert composed.active_qubits == {1, 3, 5}
    assert [instruction.qubits for instruction in composed.instructions] == [
        (5,),
        (3, 5),
        (1, 5),
        (3,),
        (1, 3),
        (1,),
        (1, 5),
    ]
