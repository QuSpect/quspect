"""Circuit construction validates its inputs at the boundary."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter
from quspect.gates.standard import CX, H
from quspect.ir.qsal import dumps
from quspect.noise.channel import PauliChannel
from quspect.simulation import Simulator

# ---------------------------------------------------------------------------
# Integer-like bit indices (anything with __index__, e.g. numpy integers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index", [np.int64(2), np.int32(2), np.uint8(2), np.intp(2), np.array(2)]
)
def test_numpy_integer_qubit_is_stored_as_int(index: object) -> None:
    circuit = QuantumCircuit().h(index)  # type: ignore[arg-type]
    assert circuit[0].qubits == (2,)
    assert type(circuit[0].qubits[0]) is int
    assert circuit[0] == QuantumCircuit().h(2)[0]
    assert hash(circuit[0]) == hash(QuantumCircuit().h(2)[0])


def test_loop_over_numpy_range_builds_the_same_circuit() -> None:
    built = QuantumCircuit()
    for qubit in np.arange(3):
        built.h(qubit)
    for control, target in zip(np.arange(2), np.arange(1, 3), strict=True):
        built.cx(control, target)
    built.rz(0.25, np.int64(1)).ccx(np.int8(0), np.int16(1), np.int32(2))
    built.barrier(*np.arange(3))
    built.depolarize(0.1, np.int64(0))
    built.measure(np.arange(3), np.arange(3)[::-1])

    plain = QuantumCircuit().h(0).h(1).h(2).cx(0, 1).cx(1, 2)
    plain.rz(0.25, 1).ccx(0, 1, 2).barrier(0, 1, 2)
    plain.depolarize(0.1, 0)
    plain.measure([0, 1, 2], [2, 1, 0])

    assert built.instructions == plain.instructions
    assert (built.num_qubits, built.num_clbits) == (3, 3)
    for instruction in built.instructions:
        assert all(type(q) is int for q in instruction.qubits)
        assert all(type(c) is int for c in instruction.clbits)


def test_numpy_indices_serialize_and_simulate() -> None:
    circuit = QuantumCircuit()
    for qubit in np.arange(2):
        circuit.x(qubit)
    circuit.measure(np.arange(2), np.arange(2))
    assert json.loads(circuit.to_json())["instructions"][0]["qubits"] == [0]
    assert json.loads(dumps(circuit))
    assert Simulator(seed=1).run(circuit, shots=20).counts == {"11": 20}
    composed = QuantumCircuit().compose(QuantumCircuit().cx(0, 1), np.array([3, 1]))
    assert composed[0].qubits == (3, 1)
    assert type(composed[0].qubits[0]) is int


def test_instruction_constructor_normalizes_indices() -> None:
    instruction = CircuitInstruction(
        InstructionType.MEASURE, (np.int64(1), 0), (np.int64(0), np.int8(3))
    )
    assert instruction.qubits == (1, 0) and instruction.clbits == (0, 3)
    assert all(type(bit) is int for bit in instruction.qubits + instruction.clbits)
    with pytest.raises(ValueError, match="distinct"):
        CircuitInstruction(InstructionType.GATE, (np.int64(1), 1), gate=CX)


@pytest.mark.parametrize(
    "index", [True, np.True_, 0.0, 1.5, np.float64(1.0), "1", None, 1 + 0j]
)
def test_non_integer_qubit_is_a_type_error(index: object) -> None:
    with pytest.raises(TypeError, match="Qubit index must be a non-negative integer"):
        QuantumCircuit().h(index)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Clbit index must be a non-negative integer"):
        QuantumCircuit().measure([0], [index])  # type: ignore[list-item]


@pytest.mark.parametrize("index", [-1, np.int64(-3)])
def test_negative_index_is_a_value_error(index: object) -> None:
    with pytest.raises(ValueError, match="Qubit index must be a non-negative integer"):
        QuantumCircuit().append(H, (index,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Clbit index must be a non-negative integer"):
        QuantumCircuit().measure([0], [index])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# One measurement instruction writes each classical bit once
# ---------------------------------------------------------------------------


def test_measure_refuses_two_qubits_into_one_clbit() -> None:
    with pytest.raises(ValueError, match="classical bit 0 more than once"):
        QuantumCircuit().measure([0, 1], [0, 0])
    with pytest.raises(ValueError, match="classical bit 2 more than once"):
        QuantumCircuit().measure([0, 1, 2], [2, 1, np.int64(2)])


def test_measure_still_allows_repeated_qubits_and_later_overwrites() -> None:
    copied = QuantumCircuit().x(0).measure([0, 0], [0, 1])
    assert Simulator(seed=1).run(copied, shots=10).counts == {"11": 10}
    overwritten = QuantumCircuit().x(1).measure([0], [0]).measure([1], [0])
    assert Simulator(seed=1).run(overwritten, shots=10).counts == {"1": 10}


# ---------------------------------------------------------------------------
# Non-finite angles and probabilities
# ---------------------------------------------------------------------------

_NON_FINITE = [math.nan, math.inf, -math.inf]


@pytest.mark.parametrize("value", _NON_FINITE)
def test_simulator_names_a_non_finite_gate_angle(value: float) -> None:
    """The simulators report the gate, not a native JSON parse error."""
    circuit = QuantumCircuit().h(0).rx(value, 0)
    with pytest.raises(ValueError, match="Gate 'rx' has a non-finite parameter"):
        Simulator().statevector(circuit)
    with pytest.raises(ValueError, match="Gate 'rx' has a non-finite parameter"):
        Simulator().run(circuit.copy().measure_all(), shots=4)

    theta = Parameter("theta")
    bound = QuantumCircuit().u3(0.1, theta, 0.3, 1).bind_parameters({theta: value})
    with pytest.raises(ValueError, match="Gate 'u3' has a non-finite parameter"):
        Simulator().statevector(bound)


@pytest.mark.parametrize("probabilities", [(math.nan, 0, 0), (0, math.nan, 0)])
def test_pauli_channel_refuses_nan(probabilities: tuple[float, float, float]) -> None:
    px, py, pz = probabilities
    with pytest.raises(ValueError, match="Pauli channel probabilities"):
        QuantumCircuit().pauli_error(px, py, pz, 0)
    with pytest.raises(ValueError, match="Pauli channel probabilities"):
        PauliChannel(name="pauli", num_qubits=1, px=px, py=py, pz=pz)


def test_pauli_channel_bounds_are_unchanged() -> None:
    assert len(QuantumCircuit().pauli_error(0.2, 0.3, 0.5, 0)) == 1
    with pytest.raises(ValueError, match="non-negative"):
        QuantumCircuit().pauli_error(-0.1, 0, 0, 0)
    with pytest.raises(ValueError, match="must not exceed 1.0"):
        QuantumCircuit().pauli_error(0.5, 0.5, 0.1, 0)
