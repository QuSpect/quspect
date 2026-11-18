"""Classically controlled gates (``InstructionType.CONDITIONAL``) across the stack.

The randomized tests compare against :mod:`tests._dynamic`, an independent
exact semantics of circuits with measurements and classical control.
"""

from __future__ import annotations

import dataclasses
import math
import pickle

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter
from quspect.gates.standard import CCX, CX, H, X, Z
from quspect.ir.errors import UnsupportedGateError
from quspect.ir.qasm2 import QASM2ExportConfig, QASM2Exporter, QASM2ExportMode
from quspect.ir.qasm_importer import QASM2Importer
from quspect.simulation import Simulator
from quspect.simulation.density_simulator import DensityMatrixSimulator
from quspect.transpiler import (
    CompilationTarget,
    ConsolidateBlocksPass,
    HighLevelOptimizePass,
    PhaseFoldingPass,
    TCountOptimizePass,
    ThreeQubitBlockPass,
    transpile,
)
from quspect.transpiler._conditional import (
    expand_conditional_gates,
    translate_conditional_gates,
)
from quspect.transpiler._passes import OptimizePass
from quspect.transpiler._phase_poly import PhasePolynomialPass
from tests._dynamic import (
    assert_compiled_behaviour,
    assert_same_behaviour,
    distribution,
    random_dynamic_circuit,
)


def _teleport(theta: float) -> QuantumCircuit:
    """Teleport ``ry(theta)|0>`` from qubit 0 to qubit 2 and undo it there."""
    qc = QuantumCircuit().ry(theta, 0).h(1).cx(1, 2).cx(0, 1).h(0)
    qc.measure([0, 1], [0, 1])
    with qc.if_test(1, 1):
        qc.x(2)
    with qc.if_test(0, 1):
        qc.z(2)
    return qc.ry(-theta, 2).measure([2], [2])


def _fields(ins: CircuitInstruction) -> tuple[object, ...]:
    return (
        ins.type,
        ins.qubits,
        ins.clbits,
        ins.gate,
        ins.conditional_gate,
        ins.condition,
    )


# ---------------------------------------------------------------- the instruction
def test_conditional_instruction_fields_and_predicates() -> None:
    ins = CircuitInstruction(
        type=InstructionType.CONDITIONAL,
        conditional_gate=X,
        qubits=(2,),
        clbits=(0, 3),
        condition=np.int64(2),
    )
    assert ins.is_conditional and not ins.is_gate and not ins.is_measure
    assert ins.gate is None and ins.conditional_gate is X
    assert ins.condition == 2 and type(ins.condition) is int
    assert ins.matrix() is None
    assert ins == dataclasses.replace(ins) and hash(ins) == hash(
        dataclasses.replace(ins)
    )
    assert pickle.loads(pickle.dumps(ins)) == ins
    other = dataclasses.replace(ins, condition=1)
    assert other != ins and repr(other) != repr(ins)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"conditional_gate": None, "clbits": (0,), "condition": 1}, ValueError),
        (
            {"gate": X, "conditional_gate": X, "clbits": (0,), "condition": 1},
            ValueError,
        ),
        ({"conditional_gate": X, "clbits": (), "condition": 0}, ValueError),
        ({"conditional_gate": X, "clbits": (0, 0), "condition": 0}, ValueError),
        ({"conditional_gate": X, "clbits": (0,), "condition": 2}, ValueError),
        ({"conditional_gate": X, "clbits": (0,), "condition": -1}, ValueError),
        ({"conditional_gate": X, "clbits": (0,), "condition": None}, TypeError),
        ({"conditional_gate": X, "clbits": (0,), "condition": True}, TypeError),
        ({"conditional_gate": X, "clbits": (0,), "condition": 1.0}, TypeError),
        ({"conditional_gate": CX, "clbits": (0,), "condition": 1}, ValueError),
    ],
)
def test_conditional_instruction_validation(
    kwargs: dict[str, object], error: type
) -> None:
    with pytest.raises(error):
        CircuitInstruction(type=InstructionType.CONDITIONAL, qubits=(1,), **kwargs)  # type: ignore[arg-type]


def test_other_instruction_types_take_no_condition() -> None:
    with pytest.raises(ValueError, match="Only CONDITIONAL"):
        CircuitInstruction(type=InstructionType.GATE, gate=X, qubits=(0,), condition=1)
    with pytest.raises(ValueError, match="Only CONDITIONAL"):
        CircuitInstruction(
            type=InstructionType.MEASURE, qubits=(0,), clbits=(0,), conditional_gate=X
        )
    with pytest.raises(ValueError, match="distinct qubits"):
        CircuitInstruction(
            type=InstructionType.CONDITIONAL,
            conditional_gate=CX,
            qubits=(1, 1),
            clbits=(0,),
            condition=1,
        )


# ---------------------------------------------------------------- building circuits
def test_if_test_block_and_append_condition() -> None:
    qc = QuantumCircuit().h(0).measure([0], [0])
    with qc.if_test(0) as inner:
        assert inner is qc
        qc.x(1).cx(1, 2)
    qc.append(X, [3], condition=([0, 4], 2)).h(1)
    assert [_fields(i)[0].name for i in qc] == [
        "GATE",
        "MEASURE",
        "CONDITIONAL",
        "CONDITIONAL",
        "CONDITIONAL",
        "GATE",
    ]
    assert _fields(qc[2]) == (InstructionType.CONDITIONAL, (1,), (0,), None, X, 1)
    assert _fields(qc[3])[1:3] == ((1, 2), (0,)) and qc[3].conditional_gate is CX
    assert qc[4].clbits == (0, 4) and qc[4].condition == 2
    assert qc.num_qubits == 4
    assert qc.num_clbits == 5  # the bits read count
    assert qc.active_clbits == {0, 4}
    assert qc._condition is None


def test_if_test_refuses_what_cannot_be_conditioned() -> None:
    qc = QuantumCircuit().h(0)
    with pytest.raises(ValueError, match="measurement"), qc.if_test(0):
        qc.measure([0], [0])
    with pytest.raises(ValueError, match="barrier"), qc.if_test(0):
        qc.barrier(0)
    with pytest.raises(ValueError, match="noise channel"), qc.if_test(0):
        qc.bit_flip(0.1, 0)
    with pytest.raises(ValueError, match="compose"), qc.if_test(0):
        qc.compose(QuantumCircuit().h(0), [0])
    with pytest.raises(ValueError, match="nest"), qc.if_test(0), qc.if_test(1):
        pass
    with pytest.raises(ValueError, match="nest"), qc.if_test(0):
        qc.append(X, [0], condition=(1, 1))
    with pytest.raises(ValueError, match="clbits"):
        qc.append(X, [0], [1], condition=(1, 1))
    with pytest.raises(ValueError, match="does not fit"), qc.if_test(0, 2):
        qc.x(0)
    # a failed block leaves the circuit usable and unconditional
    assert qc._condition is None
    assert len(qc.x(0)) == 2 and qc[1].is_gate


def test_compose_bind_json_qsal_keep_conditions() -> None:
    from quspect.ir.qsal import _deserialize_circuit, _serialize_circuit

    theta = Parameter("theta")
    qc = QuantumCircuit().h(0).measure([0], [0])
    with qc.if_test([0, 1], 1):
        qc.rz(theta, 1).cx(0, 1)
    assert qc.parameters == {theta}
    bound = qc.bind_parameters({theta: 0.25})
    assert bound[2].is_conditional and bound[2].conditional_gate.params == (0.25,)
    assert (
        bound[2].clbits == (0, 1) and bound[2].condition == 1 and bound[2].gate is None
    )

    # qubit remap (native fast path) and clbit remap (Python path)
    moved = QuantumCircuit().compose(bound, [3, 5])
    assert [_fields(i) for i in moved][2:] == [
        (InstructionType.CONDITIONAL, (5,), (0, 1), None, bound[2].conditional_gate, 1),
        (InstructionType.CONDITIONAL, (3, 5), (0, 1), None, CX, 1),
    ]
    assert list(vars(moved[2])) == list(vars(bound[2]))
    renamed = QuantumCircuit().compose(bound, [0, 1], clbits=[4, 2])
    assert renamed[1].clbits == (4,) and renamed[2].clbits == (4, 2)
    assert renamed[2].condition == 1

    for restored in (
        QuantumCircuit.from_json(bound.to_json()),
        _deserialize_circuit(_serialize_circuit(bound)),
    ):
        assert [_fields(i) for i in restored] == [_fields(i) for i in bound]
    symbolic = _deserialize_circuit(_serialize_circuit(qc))
    assert symbolic[2].conditional_gate.params[0].name == "theta"


def test_qsal_document_of_a_circuit_without_conditions_is_unchanged() -> None:
    from quspect.ir.qsal import _serialize_circuit

    doc = _serialize_circuit(QuantumCircuit().h(0).measure([0], [0]))
    assert all("condition" not in ins for ins in doc["instructions"])


def test_dagger_refuses_classical_control() -> None:
    qc = QuantumCircuit().h(0)
    qc.append(X, [0], condition=(0, 1))
    with pytest.raises(ValueError, match="classically controlled"):
        qc.dagger()


def test_native_builders_write_the_same_fields_as_the_constructor() -> None:
    qc = QuantumCircuit().h(0).cx(0, 1)
    built = QuantumCircuit().compose(qc, [1, 0])  # native remap
    for ins in built:
        assert list(vars(ins)) == [
            f.name for f in dataclasses.fields(CircuitInstruction)
        ]
        assert ins.conditional_gate is None and ins.condition is None


# ---------------------------------------------------------------- simulation
@pytest.mark.parametrize("theta", [0.3, 1.7, 2.9])
def test_teleportation_with_feed_forward(theta: float) -> None:
    qc = _teleport(theta)
    exact = distribution(qc)
    assert all(record[2] == 0 for record in exact)  # qubit 2 is back in |0>
    for sim in (Simulator(seed=11), DensityMatrixSimulator(seed=11)):
        counts = sim.run(qc, shots=600).counts
        assert sum(n for key, n in counts.items() if key[2] == "1") == 0
        assert sum(counts.values()) == 600


def test_without_the_corrections_teleportation_fails() -> None:
    qc = QuantumCircuit().ry(1.7, 0).h(1).cx(1, 2).cx(0, 1).h(0)
    qc.measure([0, 1], [0, 1]).ry(-1.7, 2).measure([2], [2])
    counts = Simulator(seed=11).run(qc, shots=600).counts
    assert sum(n for key, n in counts.items() if key[2] == "1") > 100


def test_multi_bit_conditions_and_unwritten_bits() -> None:
    qc = QuantumCircuit().h(0).h(1).measure([0, 1], [0, 1])
    qc.append(X, [2], condition=([0, 1], 2))  # c0 = 0 and c1 = 1
    qc.append(X, [3], condition=(5, 0))  # clbit 5 is never written: reads 0
    qc.append(X, [4], condition=(5, 1))  # never true
    qc.measure([2, 3, 4], [2, 3, 4])
    for sim in (Simulator(seed=5), DensityMatrixSimulator(seed=5)):
        result = sim.run(qc, shots=400)
        assert result.num_clbits == 6
        seen = {key for key, n in result.counts.items() if n}
        assert seen == {"000100", "100100", "011100", "110100"}


@pytest.mark.parametrize("seed", range(25))
def test_simulators_match_the_reference_distribution(seed: int) -> None:
    qc = random_dynamic_circuit(seed)
    if not any(i.is_measure for i in qc):
        qc.measure([0], [0])
    exact = distribution(qc)
    shots = 3000
    for sim in (Simulator(seed=seed), DensityMatrixSimulator(seed=seed)):
        counts = sim.run(qc, shots=shots).counts
        for key, n in counts.items():
            record = tuple(int(ch) for ch in key)
            assert abs(n / shots - exact.get(record, 0.0)) < 0.06, (type(sim), key)


def test_conditional_gate_before_any_measurement_reads_zero() -> None:
    qc = QuantumCircuit()
    qc.append(X, [0], condition=(0, 0))  # applied: the bit reads 0
    qc.append(X, [1], condition=(0, 1))  # not applied
    qc.measure([0, 1], [0, 1])
    for sim in (Simulator(seed=1), DensityMatrixSimulator(seed=1)):
        counts = sim.run(qc, shots=50).counts
        assert {k for k, n in counts.items() if n} == {"10"}


def test_state_queries_refuse_classical_control() -> None:
    qc = QuantumCircuit().h(0).measure([0], [0])
    qc.append(X, [1], condition=(0, 1))
    with pytest.raises(ValueError, match="classically controlled"):
        Simulator().statevector(qc)
    with pytest.raises(ValueError, match="classically controlled"):
        DensityMatrixSimulator().density_matrix(qc)
    theta = Parameter("t")
    sym = QuantumCircuit().h(0).measure([0], [0])
    with sym.if_test(0):
        sym.rz(theta, 0)
    with pytest.raises(ValueError, match="unbound symbolic"):
        Simulator().run(sym)


def test_registered_gate_under_a_condition_runs_as_its_decomposition() -> None:
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.registry import _REGISTRY, register_gate

    saved = dict(_REGISTRY)
    try:
        bell = register_gate(
            "cc_bell_test",
            gate_matrix(CX) @ np.kron(gate_matrix(H), np.eye(2)),
            [(H, [0]), (CX, [0, 1])],
        )
        qc = QuantumCircuit().x(0).measure([0], [0])
        qc.append(bell, [1, 2], condition=(0, 1))
        qc.measure([1, 2], [1, 2])
        counts = Simulator(seed=2).run(qc, shots=400).counts
        assert {k for k, n in counts.items() if n} == {"100", "111"}
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


# ---------------------------------------------------------------- OpenQASM 2
def test_qasm_export_splits_classical_registers_and_round_trips() -> None:
    qc = QuantumCircuit().h(0).measure([0, 1, 2], [0, 1, 4])
    qc.append(X, [1], condition=(0, 1))
    qc.append(CCX, [0, 1, 2], condition=([1, 2], 2))
    qc.rz(0.5, 2).append(Z, [0], condition=(0, 0))
    text = QASM2Exporter().export(qc)
    lines = text.splitlines()
    assert lines[3:7] == ["creg c0[1];", "creg c1[2];", "creg c2[2];", ""]
    assert "measure q[2] -> c2[1];" in lines
    assert "if(c0==1) x q[1];" in lines
    assert "if(c1==2) ccx q[0],q[1],q[2];" in lines
    assert "if(c0==0) z q[0];" in lines
    back = QASM2Importer().imports(text)
    # the importer writes one measure instruction per qubit; the rest is identical
    assert [_fields(i) for i in back if not i.is_measure] == [
        _fields(i) for i in qc if not i.is_measure
    ]
    assert_same_behaviour(qc, back)


def test_qasm_export_keeps_one_register_when_a_condition_reads_all_bits() -> None:
    qc = QuantumCircuit().h(0).measure([0, 1], [0, 1])
    qc.append(X, [2], condition=([0, 1], 3))
    lines = QASM2Exporter().export(qc).splitlines()
    assert "creg c[2];" in lines and "if(c==3) x q[2];" in lines
    plain = QASM2Exporter().export(QuantumCircuit().h(0).measure([0], [3]))
    assert "creg c[4];" in plain.splitlines()


@pytest.mark.parametrize(
    "bits", [(1, 0), (0, 2), (2, 1)], ids=["reversed", "gap", "reversed-high"]
)
def test_qasm_export_refuses_conditions_that_are_not_a_register(
    bits: tuple[int, ...],
) -> None:
    qc = QuantumCircuit().h(0).measure([0, 1, 2], [0, 1, 2])
    qc.append(X, [0], condition=(bits, 1))
    with pytest.raises(UnsupportedGateError, match="consecutive"):
        QASM2Exporter().export(qc)


def test_qasm_export_refuses_overlapping_condition_groups() -> None:
    qc = QuantumCircuit().h(0).measure([0, 1, 2], [0, 1, 2])
    qc.append(X, [0], condition=([0, 1], 1)).append(X, [0], condition=([1, 2], 1))
    with pytest.raises(UnsupportedGateError, match="overlapping"):
        QASM2Exporter().export(qc)


def test_qasm_export_modes_lower_conditional_gates_under_their_condition() -> None:
    qc = QuantumCircuit().h(0).measure([0, 1], [0, 1])
    with qc.if_test([0, 1], 3):
        qc.ryy(0.3, 0, 1)
    with pytest.raises(UnsupportedGateError, match="ryy"):
        QASM2Exporter().export(qc)
    for mode in (QASM2ExportMode.DECOMPOSE, QASM2ExportMode.EXTENDED):
        text = QASM2Exporter(QASM2ExportConfig(mode=mode)).export(qc)
        body = [line for line in text.splitlines() if line.startswith("if(")]
        assert body and all(line.startswith("if(c==3) ") for line in body)
        assert "ryy" not in text
        assert_same_behaviour(qc, QASM2Importer().imports(text))


@pytest.mark.parametrize("seed", range(15))
def test_qasm_round_trip_of_random_dynamic_circuits(seed: int) -> None:
    qc = random_dynamic_circuit(seed, max_size=14)
    try:
        text = QASM2Exporter(QASM2ExportConfig(mode=QASM2ExportMode.DECOMPOSE)).export(
            qc
        )
    except UnsupportedGateError as err:
        assert err.gate_name == "if"  # a condition that is not a register
        return
    assert_same_behaviour(qc, QASM2Importer().imports(text))


# ---------------------------------------------------------------- DAG and passes
@pytest.mark.parametrize("seed", range(30))
def test_dag_round_trip_keeps_every_instruction(seed: int) -> None:
    from quspect.dag.dag_to_circuit import DAGToCircuitCompiler

    qc = random_dynamic_circuit(seed)
    dag = qc.to_dag()
    back = DAGToCircuitCompiler().compile(dag)
    assert sorted(map(repr, back)) == sorted(map(repr, qc))
    assert_same_behaviour(qc, back)
    labelled = QuantumCircuit()
    labelled._instructions = [dataclasses.replace(i, label="L") for i in qc]
    back = DAGToCircuitCompiler().compile(labelled.to_dag())
    assert sum(i.is_conditional for i in back) == sum(i.is_conditional for i in qc)
    nodes = dag.nodes() if callable(dag.nodes) else dag.nodes
    assert any(n.is_conditional for n in nodes) == any(i.is_conditional for i in qc)


_PASSES = {
    "highlevel": lambda c: HighLevelOptimizePass().run(c),
    "phase_fold": lambda c: PhaseFoldingPass().run(c),
    "phase_poly": lambda c: PhasePolynomialPass().run(c),
    "consolidate": lambda c: ConsolidateBlocksPass().run(c),
    "three_qubit": lambda c: ThreeQubitBlockPass().run(c),
    "tcount": lambda c: TCountOptimizePass().run(c),
    "optimize": lambda c: OptimizePass().run(c),
    "optimize_cz": lambda c: OptimizePass(basis_gates={"cz", "h", "rz", "x"}).run(c),
    "optimize_cx": lambda c: OptimizePass(basis_gates={"cx", "sx", "rz", "x"}).run(c),
}


@pytest.mark.parametrize("name", sorted(_PASSES))
def test_passes_treat_conditional_gates_as_fences(name: str) -> None:
    for seed in range(25):
        qc = random_dynamic_circuit(seed)
        out = _PASSES[name](qc)
        # independent instructions may be reordered; none is changed or lost
        assert sorted(repr(i) for i in out if i.is_conditional) == sorted(
            repr(i) for i in qc if i.is_conditional
        ), (name, seed)
        assert_same_behaviour(qc, out)


def test_single_qubit_fusion_does_not_absorb_a_conditional_gate() -> None:
    # h · (if c: s) · h must not become one unconditional gate
    qc = QuantumCircuit().h(0)
    qc.append(Z, [0], condition=(0, 1))
    qc.h(0)
    out = OptimizePass(basis_gates={"cz", "h", "rz", "x"}).run(qc)
    assert sum(i.is_conditional for i in out) == 1
    assert_same_behaviour(qc, out)


def test_a_gate_does_not_cancel_across_a_conditional_gate() -> None:
    qc = QuantumCircuit().h(0)
    qc.append(X, [0], condition=(0, 0))
    qc.h(0)
    for run in _PASSES.values():
        out = run(qc)
        assert sum(i.is_conditional for i in out) == 1
        assert_same_behaviour(qc, out)


# ---------------------------------------------------------------- compiling to a target
def test_expand_conditional_gates_leaves_single_qubit_conditions_only() -> None:
    qc = QuantumCircuit().h(0).h(1).measure([0], [0])
    qc.append(CX, [0, 1], condition=(0, 1))
    out = expand_conditional_gates(qc)
    assert [
        (i.type.name, (i.gate or i.conditional_gate).name, i.qubits) for i in out[3:]
    ] == [
        ("CONDITIONAL", "h", (1,)),
        ("GATE", "cx", (0, 1)),
        ("CONDITIONAL", "sdg", (1,)),
        ("GATE", "cx", (0, 1)),
        ("CONDITIONAL", "s", (0,)),
        ("CONDITIONAL", "s", (1,)),
        ("CONDITIONAL", "h", (1,)),
    ]
    assert_same_behaviour(qc, out)
    same = QuantumCircuit().h(0)
    same.append(X, [0], condition=(0, 1))
    assert expand_conditional_gates(same) is same


@pytest.mark.parametrize("gate", ["swap", "cz", "ccx", "rzz", "cp", "rxx", "ryy"])
def test_expanded_multi_qubit_conditional_gate_behaves_the_same(gate: str) -> None:
    qc = QuantumCircuit().h(0).ry(0.4, 1).rx(1.1, 2).cx(0, 2).measure([0], [0])
    with qc.if_test(0):
        if gate == "ccx":
            qc.ccx(1, 2, 0)
        elif gate in ("rzz", "cp", "rxx", "ryy"):
            getattr(qc, gate)(0.7, 1, 2)
        else:
            getattr(qc, gate)(2, 1)
    out = expand_conditional_gates(qc)
    assert all(len(i.qubits) == 1 for i in out if i.is_conditional)
    assert_same_behaviour(qc, out)


def test_translate_conditional_gates_merges_a_run_and_keeps_native_runs() -> None:
    qc = QuantumCircuit().measure([0], [0])
    with qc.if_test(0):
        qc.h(1).t(1).h(1)
    with qc.if_test(0, 0):
        qc.x(1)
    native = {"cz", "h", "rz", "x"}
    out = translate_conditional_gates(qc, native)
    assert all(
        (i.gate or i.conditional_gate).name in native for i in out if not i.is_measure
    )
    assert out[-1] == qc[-1]  # the native run is kept as it is
    assert {i.condition for i in out if i.is_conditional} == {0, 1}
    assert_same_behaviour(qc, out)
    assert translate_conditional_gates(qc, set()) is qc
    two = QuantumCircuit()
    two.append(CX, [0, 1], condition=(0, 1))
    with pytest.raises(ValueError, match="not expanded"):
        translate_conditional_gates(two, native)
    sym = QuantumCircuit()
    sym.append(
        dataclasses.replace(H, name="ry", params=(Parameter("a"),)),
        [0],
        condition=(0, 1),
    )
    with pytest.raises(ValueError, match="classically controlled"):
        translate_conditional_gates(sym, native)


_TARGETS = {
    "line5-cz": CompilationTarget.from_coupling_map(
        5, [(0, 1), (1, 2), (2, 3), (3, 4)], native_gates={"cz", "h", "rz", "x"}
    ),
    "full4-cz": CompilationTarget.from_coupling_map(
        4,
        [(a, b) for a in range(4) for b in range(a)],
        native_gates={"cz", "h", "rz", "x"},
    ),
    "line4-cx": CompilationTarget.from_coupling_map(
        4, [(0, 1), (1, 2), (2, 3)], native_gates={"cx", "rz", "sx", "x"}
    ),
}


def _check_transpiled(
    qc: QuantumCircuit, target: CompilationTarget, level: int
) -> None:
    res = transpile(qc, target, optimization_level=level, seed=1)
    edges = {tuple(sorted(e)) for e in target.coupling_graph}
    for ins in res.circuit:
        gate = ins.gate if ins.gate is not None else ins.conditional_gate
        if gate is None:
            continue
        assert gate.name in target.native_gates, (ins.type.name, gate.name)
        if ins.is_conditional:
            assert len(ins.qubits) == 1
        elif len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in edges
    assert_compiled_behaviour(
        qc, res.circuit, res.final_layout[: qc.num_qubits], target.num_qubits
    )
    wires = res.wire_circuit
    assert sum(i.is_conditional for i in wires) == sum(
        i.is_conditional for i in res.circuit
    )


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_transpile_keeps_the_behaviour_of_dynamic_circuits(
    target: str, level: int
) -> None:
    for seed in range(8):
        _check_transpiled(random_dynamic_circuit(seed), _TARGETS[target], level)


@pytest.mark.slow
@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_transpile_keeps_the_behaviour_of_dynamic_circuits_more_seeds(
    target: str, level: int
) -> None:
    for seed in range(8, 60):
        _check_transpiled(random_dynamic_circuit(seed), _TARGETS[target], level)


@pytest.mark.parametrize("level", [1, 3])
def test_transpile_teleportation(level: int) -> None:
    _check_transpiled(_teleport(0.9), _TARGETS["line5-cz"], level)
    _check_transpiled(_teleport(0.9), _TARGETS["line4-cx"], level)


def test_transpile_of_a_circuit_without_conditions_is_unchanged() -> None:
    qc = QuantumCircuit().h(0).cx(0, 2).t(2).cx(1, 2).measure([0, 1, 2], [0, 1, 2])
    target = _TARGETS["line5-cz"]
    a = transpile(qc, target, optimization_level=2, seed=3)
    b = transpile(qc, target, optimization_level=2, seed=3)
    assert [_fields(i) for i in a.circuit] == [_fields(i) for i in b.circuit]
    assert not any(i.is_conditional for i in a.circuit)


def test_router_refuses_a_multi_qubit_conditional_gate() -> None:
    from quspect.transpiler import SABRE

    target = _TARGETS["line5-cz"]
    qc = QuantumCircuit().h(0).cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 4).measure([0], [0])
    with qc.if_test(0):
        qc.cx(0, 4)
    with pytest.raises(ValueError, match="Cannot route the classically controlled"):
        SABRE(target).solve(qc, seed=1)
    routed, initial, _ = SABRE(target).solve(expand_conditional_gates(qc), seed=1)
    for ins in routed:
        if ins.is_gate and len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in target.coupling_graph
        assert not ins.is_conditional or len(ins.qubits) == 1
    assert len(initial) == qc.num_qubits


def test_decompose_lowers_conditional_gates_under_their_condition() -> None:
    from quspect.ir.compilation import decompose_deep
    from quspect.transpiler._passes import DecomposePass

    native = {"cz", "h", "rz", "x"}
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (1, 2)], native_gates=native
    )
    qc = QuantumCircuit().ry(0.3, 0).h(1).measure([0], [0])
    with qc.if_test(0):
        qc.ccx(0, 1, 2).rx(0.3, 1).s(2)
    qc.rx(0.3, 1)
    out = DecomposePass(target).run(qc)
    names = {(i.gate or i.conditional_gate).name for i in out if not i.is_measure}
    assert names <= native
    assert all(i.condition == 1 and i.clbits == (0,) for i in out if i.is_conditional)
    assert_same_behaviour(qc, out)
    # the rules alone cannot express a conditional ``s``: strict mode says so
    with pytest.raises(ValueError, match="classically controlled"):
        decompose_deep(qc, supported_gates=native, strict=True)
    lowered = decompose_deep(qc, supported_gates=native)
    assert "ccx" not in {i.conditional_gate.name for i in lowered if i.is_conditional}
    assert_same_behaviour(qc, lowered)
    # a conditional gate with neither a rule nor a matrix is reported, not kept
    odd = QuantumCircuit()
    odd.append(dataclasses.replace(H, name="mystery"), [0], condition=(0, 1))
    with pytest.raises(ValueError, match="mystery"):
        DecomposePass(target).run(odd)


def test_transpile_errors_for_conditional_gates_are_value_errors() -> None:
    target = CompilationTarget.from_coupling_map(
        2, [(0, 1)], native_gates={"cz", "h", "rz", "x"}
    )
    qc = QuantumCircuit().h(0).measure([0], [0])
    qc.append(dataclasses.replace(H, name="mystery"), [1], condition=(0, 1))
    with pytest.raises(ValueError, match="mystery"):
        transpile(qc, target, seed=1)


def test_transpile_keeps_native_symbolic_conditional_gates() -> None:
    target = CompilationTarget.from_coupling_map(
        2, [(0, 1)], native_gates={"cz", "h", "rz", "x"}
    )
    theta = Parameter("theta")
    qc = QuantumCircuit().h(0).measure([0], [0])
    with qc.if_test(0):
        qc.rz(theta, 1).s(1)
        qc.crz(theta, 0, 1)
    res = transpile(qc, target, seed=1)
    assert res.circuit.parameters == {theta}
    assert all(
        (i.gate or i.conditional_gate).name in target.native_gates
        for i in res.circuit
        if not i.is_measure
    )
    bound = res.circuit.bind_parameters({theta: 0.37})
    assert_compiled_behaviour(
        qc.bind_parameters({theta: 0.37}), bound, res.final_layout[:2], 2
    )
    # a symbolic gate that is not native is lowered by the rules, condition kept
    sym = QuantumCircuit().h(1).measure([0], [0])
    with sym.if_test(0, 0):
        sym.ry(theta, 1)
    res = transpile(sym, target, seed=1)
    assert all(
        (i.gate or i.conditional_gate).name in target.native_gates
        for i in res.circuit
        if not i.is_measure
    )
    assert_compiled_behaviour(
        sym.bind_parameters({theta: 1.2}),
        res.circuit.bind_parameters({theta: 1.2}),
        res.final_layout[:2],
        2,
    )


def test_phase_polynomial_extraction_refuses_conditional_gates() -> None:
    from quspect.dag.phase_polynomial import PhasePolynomialCompiler

    qc = QuantumCircuit().t(0).measure([1], [0])
    with qc.if_test(0):
        qc.rz(0.5, 0)
    with pytest.raises(ValueError, match="classically controlled"):
        PhasePolynomialCompiler.extract(qc.to_dag())


def test_copies_made_inside_a_block_have_no_open_block() -> None:
    import copy

    qc = QuantumCircuit().h(0)
    with qc.if_test(0):
        clones = [copy.deepcopy(qc), pickle.loads(pickle.dumps(qc)), qc.copy()]
    for clone in clones:
        assert clone._condition is None
        assert clone.x(1)[-1].is_gate


def test_if_test_validates_its_condition_before_any_gate() -> None:
    qc = QuantumCircuit()
    with pytest.raises(ValueError, match="does not fit"), qc.if_test(0, 5):
        pass
    with pytest.raises(ValueError, match="more than once"), qc.if_test([1, 1], 0):
        pass
    with pytest.raises(TypeError), qc.if_test(0, True):
        pass
    assert qc._condition is None and len(qc) == 0
    with qc.if_test(np.int64(2), np.int64(1)):
        qc.x(0)
    assert qc[0].clbits == (2,) and type(qc[0].condition) is int


def test_native_executor_rejects_a_condition_on_a_plain_gate() -> None:
    import json

    from quspect.ir import qsal
    from quspect.native import NativeExecutor, load_core

    if load_core() is None:
        pytest.skip("native extension not built")
    qc = QuantumCircuit().h(0).measure([0], [0])
    with qc.if_test(0):
        qc.x(1)
    qc.measure([1], [1])
    doc = json.loads(qsal.dumps(qc))
    doc["circuit"]["instructions"][2]["type"] = "GATE"  # still carries "condition"
    with pytest.raises(Exception, match="only CONDITIONAL"):
        NativeExecutor(seed=1).run(json.dumps(doc), shots=10)
    doc = json.loads(qsal.dumps(qc))
    doc["circuit"]["instructions"][2]["clbits"] = [0, 0]
    with pytest.raises(Exception, match="more than once"):
        NativeExecutor(seed=1).run(json.dumps(doc), shots=10)


# ---------------------------------------------------------------- consumers that refuse
def test_cloud_backends_refuse_classical_control() -> None:
    from quspect.backends._compile import compile_for_device
    from quspect.backends._exceptions import TranslationError
    from quspect.backends.logicalqubit._translator import LQCircuitTranslator
    from quspect.backends.originq._translator import translate as originq_translate
    from quspect.backends.tianyan._translator import translate as tianyan_translate
    from quspect.ir.qsal import dumps

    qc = QuantumCircuit().h(0).measure([0], [0])
    qc.append(X, [1], condition=(0, 1))
    qc.measure([1], [1])
    with pytest.raises(TranslationError, match="classically controlled"):
        originq_translate(qc)
    with pytest.raises(TranslationError, match="classically controlled"):
        tianyan_translate(qc)
    with pytest.raises(ValueError, match="classically controlled"):
        compile_for_device(_TARGETS["line5-cz"], qc)
    try:
        import lqcloud  # noqa: F401
    except ImportError:
        return  # optional dependency of the LogicalQubit backend
    with pytest.raises(TranslationError, match="classically controlled"):
        LQCircuitTranslator().translate(dumps(qc))


def test_variational_objectives_and_statevector_validation_refuse_it() -> None:
    from quspect.transpiler.validation import _unitary_view

    qc = QuantumCircuit().h(0)
    qc.append(X, [0], condition=(0, 1))
    with pytest.raises(ValueError, match="classically controlled"):
        _unitary_view(qc)
    with pytest.raises(ValueError, match="classically controlled"):
        _unitary_view(
            QuantumCircuit().h(0).measure([0], [0]).append(X, [0], condition=(0, 1))
        )


def test_noise_model_leaves_conditional_gates_without_gate_noise() -> None:
    from quspect.noise.adder import NoiseModelAdder
    from quspect.noise.noise_model import NoiseModel

    channel = QuantumCircuit().bit_flip(0.1, 0)[0].channel
    assert channel is not None
    model = NoiseModel().add_all_qubit_gate_error(channel)
    qc = QuantumCircuit().h(0).measure([0], [0])
    qc.append(X, [1], condition=(0, 1))
    noisy = NoiseModelAdder(model).apply(qc)
    kinds = [i.type.name for i in noisy]
    assert kinds == ["GATE", "CHANNEL", "MEASURE", "CONDITIONAL"]
    assert noisy[-1] == qc[-1]
    counts = Simulator(seed=4).run(noisy.measure([1], [1]), shots=300).counts
    assert sum(counts.values()) == 300


# ---------------------------------------------------------------- drawing
def test_text_drawing_shows_the_condition_bits() -> None:
    qc = QuantumCircuit().h(0).measure([0, 1], [0, 1])
    with qc.if_test([0, 1], 1):
        qc.rz(0.5, 0)
    with qc.if_test(1):
        qc.x(2)
    art = qc.draw().splitlines()
    assert (
        art
        == [
            "q0: --[H]----[M]---[RZ(0.5)]--------",
            "q1: --------[M]--------|------------",
            "q2: ---------|---------|------[X]---",
            "c0: =========0========(1)======|====",
            "c1: =========1========(0)=====(1)===",
        ]
        or len({len(line) for line in art}) == 1
    )
    assert "(1)" in art[3] and "(0)" in art[4] and "(1)" in art[4]
    assert "[RZ(0.5)]" in art[0] and "[X]" in art[2]
    column = art[0].index("[RZ(0.5)]") + len("[RZ(0.5)]") // 2
    assert art[1][column] == "|" and art[2][column] == "|"


def test_svg_drawing_reserves_the_classical_lanes_and_marks_the_bits() -> None:
    from quspect.visualization.svg_drawer import (
        _build_instruction_layers,
        _instruction_lane_span,
    )

    qc = QuantumCircuit().h(0).measure([0, 1], [0, 1])
    with qc.if_test([0, 1], 1):
        qc.rz(0.5, 0)
    qc.h(2)
    svg = str(qc.svg())
    assert "<title>rz if (c0, c1) == (1, 0)</title>" in svg
    assert svg.count('r="5.5"') == 2
    conditional = qc[2]
    assert _instruction_lane_span(conditional, 3) == (0, 1, 2, 3, 4)
    layers = _build_instruction_layers(qc.instructions, 3, 5)
    # h(2) lies inside the connector's lanes: it cannot share the column
    column_of = {id(i): k for k, layer in enumerate(layers) for i in layer}
    assert column_of[id(qc[3])] != column_of[id(conditional)]
    QuantumCircuit().x(0).svg()  # unchanged for plain circuits
    assert math.isfinite(len(svg))
