"""Tests for the native C++ circuit IR bridge (``QuantumCircuit._to_native``)."""

from __future__ import annotations

import dataclasses

import pytest

from quspect import QuantumCircuit
from quspect.circuit._native import native_ir
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter
from quspect.gates.gate import Gate
from quspect.gates.standard import CX, H

pytestmark = pytest.mark.skipif(native_ir() is None, reason="native core not built")


def _sample() -> QuantumCircuit:
    theta = Parameter("theta")
    qc = QuantumCircuit(name="s")
    qc.h(0).cx(0, 1).rz(0.25, 1).rx(theta, 2).u3(0.1, 0.2, 0.3, 0)
    qc.append(Gate(name="custom", num_qubits=2, params=(1.0,)), (2, 0))
    qc.barrier().depolarize(0.1, 1).measure([0, 1], [1, 0])
    return qc


def test_round_trip_keeps_instruction_objects() -> None:
    qc = _sample()
    native = qc._to_native()
    assert len(native) == len(qc)
    assert native.num_qubits == qc.num_qubits
    assert native.num_clbits == qc.num_clbits
    back = QuantumCircuit._from_native(native, name="s")
    assert back.instructions == qc.instructions
    assert all(a is b for a, b in zip(back, qc, strict=True))
    assert back.name == "s"


def test_implicit_dag_queries() -> None:
    qc = QuantumCircuit().h(0).cx(0, 1).h(1).x(2).cz(1, 2)
    native = qc._to_native()
    assert native.predecessors(1) == [0]
    assert native.predecessors(4) == [2, 3]
    assert native.successors(1) == [2]
    assert native.successors(0) == [1]
    assert native.qubit_chain(1) == [1, 2, 4]
    assert native.depth() == 4
    assert [native.gate_name(i) for i in range(len(native))] == [
        "h",
        "cx",
        "h",
        "x",
        "cz",
    ]
    with pytest.raises(IndexError):
        native.predecessors(99)


def test_qasm_fast_path_builds_ordinary_dataclasses() -> None:
    core = native_ir()
    assert core is not None
    from quspect.ir.qasm_importer import _QELIB1_DEFINITIONS

    parsed = core._qasm2_parse(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[1];'
        "h q[0]; cx q[0],q[1]; rz(pi/2) q[1]; measure q[1] -> c[0];",
        _QELIB1_DEFINITIONS,
    )
    ins = parsed.to_instructions()
    assert ins[0].gate is H
    assert ins[1] == CircuitInstruction(
        type=InstructionType.GATE, gate=CX, qubits=(0, 1)
    )
    rz = ins[2]
    assert dataclasses.is_dataclass(rz) and dataclasses.is_dataclass(rz.gate)
    assert rz.gate == Gate(name="rz", num_qubits=1, params=(1.5707963267948966,))
    assert rz.gate.label == "RZ"
    assert (
        repr(rz.gate)
        == "Gate(name='rz', num_qubits=1, params=(1.5707963267948966,), label='RZ')"
    )
    assert hash(rz) == hash(
        CircuitInstruction(type=InstructionType.GATE, gate=rz.gate, qubits=(1,))
    )
    assert dataclasses.replace(rz, qubits=(0,)).qubits == (0,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rz.qubits = (0,)  # type: ignore[misc]
    assert ins[3].type is InstructionType.MEASURE
    assert (ins[3].qubits, ins[3].clbits, ins[3].gate, ins[3].label) == (
        (1,),
        (0,),
        None,
        "",
    )
    assert (
        core._qasm2_parse("OPENQASM 2.0; qreg q[1]; reset q[0];", _QELIB1_DEFINITIONS)
        is None
    )
