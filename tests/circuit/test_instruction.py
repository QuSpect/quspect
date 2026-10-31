"""Tests for CircuitInstruction."""

from __future__ import annotations

import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter
from quspect.gates.gate import Gate
from quspect.gates.standard import CCX, CX, H, X
from quspect.noise.channel import DepolarizingChannel


class TestGateInstruction:
    """Test gate-type instructions."""

    def test_create_gate_instruction(self) -> None:
        """Test creating a gate instruction."""
        instr = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
        assert instr.type == InstructionType.GATE
        assert instr.gate is H
        assert instr.qubits == (0,)
        assert instr.clbits == ()
        assert instr.is_gate is True
        assert instr.is_measure is False
        assert instr.is_barrier is False

    def test_create_two_qubit_gate(self) -> None:
        """Test creating a two-qubit gate instruction."""
        instr = CircuitInstruction(type=InstructionType.GATE, gate=CX, qubits=(0, 1))
        assert instr.gate is CX
        assert instr.qubits == (0, 1)

    def test_gate_instruction_matrix(self) -> None:
        """Test matrix() access on gate instruction."""
        instr = CircuitInstruction(type=InstructionType.GATE, gate=X, qubits=(0,))
        import numpy as np

        m = instr.matrix()
        assert m is not None
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(m, expected)

    def test_qubit_count_mismatch(self) -> None:
        """Test that qubit count must match gate."""
        with pytest.raises(ValueError, match="requires"):
            CircuitInstruction(type=InstructionType.GATE, gate=CX, qubits=(0,))

    def test_gate_is_none_raises(self) -> None:
        """Test that GATE type with None gate raises."""
        with pytest.raises(ValueError, match="non-None gate"):
            CircuitInstruction(
                type=InstructionType.GATE,
                gate=None,
                qubits=(0,),
            )


class TestMeasureInstruction:
    """Test measurement-type instructions."""

    def test_create_measure_instruction(self) -> None:
        """Test creating a measurement instruction."""
        instr = CircuitInstruction(
            type=InstructionType.MEASURE, qubits=(0, 1), clbits=(0, 1)
        )
        assert instr.type == InstructionType.MEASURE
        assert instr.gate is None
        assert instr.qubits == (0, 1)
        assert instr.clbits == (0, 1)
        assert instr.is_gate is False
        assert instr.is_measure is True
        assert instr.is_barrier is False

    def test_measure_matrix_is_none(self) -> None:
        """Test matrix() returns None for measurement."""
        instr = CircuitInstruction(
            type=InstructionType.MEASURE, qubits=(0,), clbits=(0,)
        )
        assert instr.matrix() is None

    def test_measure_qubits_clbits_mismatch(self) -> None:
        """Test that qubits and clbits must have same length."""
        with pytest.raises(ValueError, match="equal number"):
            CircuitInstruction(
                type=InstructionType.MEASURE,
                qubits=(0, 1),
                clbits=(0,),
            )

    def test_measure_with_gate_raises(self) -> None:
        """Test that MEASURE with gate raises."""
        with pytest.raises(ValueError, match="should not have a gate"):
            CircuitInstruction(
                type=InstructionType.MEASURE,
                gate=H,
                qubits=(0,),
                clbits=(0,),
            )


class TestBarrierInstruction:
    """Test barrier-type instructions."""

    def test_create_barrier_instruction(self) -> None:
        """Test creating a barrier instruction."""
        instr = CircuitInstruction(type=InstructionType.BARRIER, qubits=(0, 1, 2))
        assert instr.type == InstructionType.BARRIER
        assert instr.gate is None
        assert instr.is_barrier is True
        assert instr.is_gate is False
        assert instr.is_measure is False

    def test_barrier_empty_qubits(self) -> None:
        """Test barrier with empty qubits (means all)."""
        instr = CircuitInstruction(type=InstructionType.BARRIER, qubits=())
        assert instr.qubits == ()

    def test_barrier_with_gate_raises(self) -> None:
        """Test that BARRIER with gate raises."""
        with pytest.raises(ValueError, match="should not have a gate"):
            CircuitInstruction(
                type=InstructionType.BARRIER,
                gate=H,
                qubits=(0,),
            )


class TestInstructionFrozen:
    """Test immutability."""

    def test_is_frozen(self) -> None:
        """Test instruction is immutable."""
        instr = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            instr.qubits = (1,)  # type: ignore[misc]

    def test_hashable(self) -> None:
        """Test instructions are hashable."""
        instr1 = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
        instr2 = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
        assert hash(instr1) == hash(instr2)
        s = {instr1, instr2}
        assert len(s) == 1


class TestDistinctQubitValidation:
    """Regression tests for B05/B06: duplicate qubits and symbolic matrices."""

    @pytest.mark.parametrize(
        ("gate", "qubits"), [(CX, (0, 0)), (CCX, (0, 1, 0)), (CCX, (2, 2, 2))]
    )
    def test_gate_with_duplicate_qubits_rejected(
        self, gate: Gate, qubits: tuple[int, ...]
    ) -> None:
        """Multi-qubit gates must act on distinct qubits."""
        with pytest.raises(ValueError, match="distinct"):
            CircuitInstruction(type=InstructionType.GATE, gate=gate, qubits=qubits)

    def test_fluent_cx_with_duplicate_qubits_rejected(self) -> None:
        """``cx(0, 0)`` raises at construction time."""
        with pytest.raises(ValueError, match="distinct"):
            QuantumCircuit().cx(0, 0)

    def test_channel_with_duplicate_qubits_rejected(self) -> None:
        """Multi-qubit channels must act on distinct qubits."""
        channel = DepolarizingChannel(name="depolarizing", num_qubits=2, p=0.1)
        with pytest.raises(ValueError, match="distinct"):
            CircuitInstruction(
                type=InstructionType.CHANNEL, channel=channel, qubits=(1, 1)
            )

    def test_measure_may_repeat_qubit(self) -> None:
        """Measuring one qubit into several clbits stays allowed."""
        instr = CircuitInstruction(
            type=InstructionType.MEASURE, qubits=(0, 0), clbits=(0, 1)
        )
        assert instr.qubits == (0, 0)

    def test_matrix_with_symbolic_parameter_raises_value_error(self) -> None:
        """``matrix()`` raises the documented ValueError for symbolic gates."""
        theta = Parameter("theta")
        instr = QuantumCircuit().rx(theta, 0)[0]
        with pytest.raises(ValueError, match="symbolic"):
            instr.matrix()
        with pytest.raises(ValueError, match="symbolic"):
            QuantumCircuit().rx(theta + 1, 0)[0].matrix()

    def test_matrix_with_constant_expression_works(self) -> None:
        """A constant ParameterExpression is treated as a number."""
        theta = Parameter("theta")
        instr = QuantumCircuit().rx(theta - theta + 0.5, 0)[0]
        matrix = instr.matrix()
        assert matrix is not None
        assert matrix.shape == (2, 2)
