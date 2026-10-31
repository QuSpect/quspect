"""Cross-validate quspect's DAG IR (``quspect.dag``) against qiskit.

Covers :class:`DAGNodeType`, :class:`DAGNode`, :class:`DAGEdge`,
:class:`DAGCircuit`, :class:`CircuitCompiler` and
:class:`DAGToCircuitCompiler`.

The topology of ``circ.to_dag()`` is compared with
``qiskit.converters.circuit_to_dag(to_qiskit(circ))``: node / edge counts,
depth, per-wire operation sequences and per-node predecessor / successor
sets.  Round trips ``circuit -> DAG -> circuit`` must preserve the unitary
exactly (no phase freedom) and the instruction list.

Node id convention of :class:`CircuitCompiler` (verified below): INPUT nodes
for qubits then clbits get ids ``0 .. nq+nc-1``, the ``k``-th instruction gets
id ``nq + nc + k``, OUTPUT nodes come last.  qiskit assigns increasing
``_node_id`` to op nodes in insertion order, so sorting qiskit op nodes by
``_node_id`` recovers the instruction index as well.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest
import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

import quspect
from qiskit_tests.helpers import (
    assert_unitary_equal,
    qiskit_circuit_unitary,
    qiskit_probabilities,
    quspect_circuit_unitary,
    random_circuit_pair,
    to_qiskit,
)
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.dag import (
    CircuitCompiler,
    DAGCircuit,
    DAGEdge,
    DAGNode,
    DAGNodeType,
    DAGToCircuitCompiler,
)
from quspect.gates.standard import H

pytestmark = pytest.mark.qiskit_ref


# =============================================================================
# Helpers
# =============================================================================


def _random_mixed_circuit(n: int, depth: int, seed: int) -> quspect.QuantumCircuit:
    """Random unitary circuit plus barriers and single-qubit measurements."""
    circ, _ = random_circuit_pair(n, depth, seed)
    rng = np.random.default_rng(seed + 10_000)
    out = quspect.QuantumCircuit()
    for instr in circ.instructions:
        assert instr.gate is not None
        out.append(instr.gate, instr.qubits)
        r = rng.random()
        if r < 0.08:
            qs = sorted(int(q) for q in rng.choice(n, size=2, replace=False))
            out.barrier(*qs)
    clbits = [int(c) for c in rng.permutation(n)]
    for q in range(n):
        out.measure([q], [clbits[q]])
    return out


def _quspect_op_index(dag: DAGCircuit) -> dict[int, int]:
    """Map quspect op node id -> instruction index."""
    offset = dag.num_qubits + dag.num_clbits
    return {n.node_id: n.node_id - offset for n in dag.operation_nodes()}


def _qiskit_op_index(qdag: qiskit.dagcircuit.DAGCircuit) -> dict[int, int]:
    """Map qiskit op node id -> instruction index (insertion order)."""
    ids = sorted(n._node_id for n in qdag.op_nodes())
    return {nid: k for k, nid in enumerate(ids)}


def _wire_sequences(
    circ: quspect.QuantumCircuit,
) -> dict[str, list[CircuitInstruction]]:
    """Per-wire (qubit and clbit) ordered instruction lists of a circuit."""
    seqs: dict[str, list[CircuitInstruction]] = {}
    for instr in circ.instructions:
        for q in instr.qubits:
            seqs.setdefault(f"q{q}", []).append(instr)
        for c in instr.clbits:
            seqs.setdefault(f"c{c}", []).append(instr)
    return seqs


def _op_name(instr: CircuitInstruction) -> str:
    """qiskit-compatible operation name of a quspect instruction."""
    if instr.gate is not None:
        return {"u1": "p", "u3": "u", "u2": "u", "cu3": "cu"}.get(
            instr.gate.name, instr.gate.name
        )
    return "measure" if instr.is_measure else "barrier"


# =============================================================================
# DAGNodeType / DAGNode / DAGEdge
# =============================================================================


def test_dag_node_type_members() -> None:
    """DAGNodeType has exactly INPUT, OUTPUT, OPERATION."""
    assert [m.name for m in DAGNodeType] == ["INPUT", "OUTPUT", "OPERATION"]


def test_dag_node_predicates_and_repr() -> None:
    """is_* predicates and repr for every node kind."""
    gate_instr = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(1,))
    meas = CircuitInstruction(type=InstructionType.MEASURE, qubits=(0,), clbits=(0,))
    bar = CircuitInstruction(type=InstructionType.BARRIER, qubits=(0, 1))
    n_in = DAGNode(0, DAGNodeType.INPUT, qubits={0})
    n_out = DAGNode(1, DAGNodeType.OUTPUT, qubits={0})
    n_g = DAGNode(2, DAGNodeType.OPERATION, instruction=gate_instr, qubits={1})
    n_m = DAGNode(3, DAGNodeType.OPERATION, instruction=meas, qubits={0}, clbits={0})
    n_b = DAGNode(4, DAGNodeType.OPERATION, instruction=bar, qubits={0, 1})
    n_none = DAGNode(5, DAGNodeType.OPERATION)

    assert (n_in.is_input, n_in.is_output, n_in.is_operation) == (True, False, False)
    assert (n_out.is_input, n_out.is_output) == (False, True)
    assert n_g.is_operation and n_g.is_gate and not n_g.is_measure
    assert n_m.is_measure and not n_m.is_gate and not n_m.is_barrier
    assert n_b.is_barrier and not n_b.is_gate
    assert not (n_none.is_gate or n_none.is_measure or n_none.is_barrier)
    assert not n_in.is_gate

    assert repr(n_in) == "DAGNode(id=0, INPUT, qubits=[q0])"
    assert repr(n_out) == "DAGNode(id=1, OUTPUT, qubits=[q0])"
    assert repr(n_g) == "DAGNode(id=2, OP[h], qubits=[q1])"
    assert repr(n_m) == "DAGNode(id=3, OP[MEASURE], qubits=[q0])"
    assert repr(n_b) == "DAGNode(id=4, OP[BARRIER], qubits=[q0, q1])"
    assert repr(n_none) == "DAGNode(id=5, OP[?], qubits=[])"
    assert n_none.metadata == {} and n_none.label == ""


def test_dag_edge_defaults() -> None:
    """DAGEdge stores (source, target, wire) with empty default wire."""
    e = DAGEdge(1, 2)
    assert (e.source_id, e.target_id, e.wire) == (1, 2, "")
    assert DAGEdge(1, 2, "q0") == DAGEdge(1, 2, "q0")


# =============================================================================
# DAGCircuit API
# =============================================================================


def test_empty_circuit_compiles_to_empty_dag() -> None:
    """An empty circuit gives a DAG with no nodes, depth 0."""
    dag = CircuitCompiler.compile(quspect.QuantumCircuit())
    assert dag.num_nodes() == 0 and dag.num_edges() == 0
    assert dag.topological_order() == []
    assert dag.layers() == [[]]
    assert dag.critical_path_length() == 0
    assert dag.qubit_wires() == [] and dag.clbit_wires() == []


def test_add_get_remove_node_errors() -> None:
    """Duplicate ids raise ValueError; unknown ids raise KeyError."""
    dag = DAGCircuit()
    dag.add_node(DAGNode(0, DAGNodeType.INPUT, qubits={0}))
    with pytest.raises(ValueError, match="already exists"):
        dag.add_node(DAGNode(0, DAGNodeType.OUTPUT, qubits={0}))
    with pytest.raises(KeyError):
        dag.get_node(42)
    with pytest.raises(KeyError):
        dag.remove_node(42)


def test_manual_dag_construction_and_removal() -> None:
    """Hand-built DAG: adjacency, removal rebuilds predecessor/successor lists."""
    dag = DAGCircuit()
    dag.set_qubit_count(1)
    dag.set_clbit_count(0)
    instr = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
    dag.add_node(DAGNode(0, DAGNodeType.INPUT, qubits={0}))
    dag.add_node(DAGNode(1, DAGNodeType.OPERATION, instruction=instr, qubits={0}))
    dag.add_node(DAGNode(2, DAGNodeType.OUTPUT, qubits={0}))
    dag.add_edge(DAGEdge(0, 1, "q0"))
    dag.add_edge(DAGEdge(1, 2, "q0"))
    assert dag.successors(0) == [1] and dag.predecessors(2) == [1]
    assert dag.critical_path_length() == 1
    dag.remove_node(1)
    assert dag.num_edges() == 0
    assert dag.successors(0) == [] and dag.predecessors(2) == []
    assert dag.successors(99) == [] and dag.predecessors(99) == []
    assert dag.critical_path_length() == 0


def test_cycle_detection() -> None:
    """topological_order raises RuntimeError on a cyclic graph."""
    dag = DAGCircuit()
    dag.add_node(DAGNode(0, DAGNodeType.OPERATION))
    dag.add_node(DAGNode(1, DAGNodeType.OPERATION))
    dag.add_edge(DAGEdge(0, 1, "q0"))
    dag.add_edge(DAGEdge(1, 0, "q0"))
    with pytest.raises(RuntimeError, match="Cycle"):
        dag.topological_order()


def test_compiler_node_ids_and_boundaries() -> None:
    """INPUT ids first (qubits then clbits), ops in order, OUTPUT ids last."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 2).measure([2], [1])
    dag = circ.to_dag()
    nq, nc = 3, 2
    assert (dag.num_qubits, dag.num_clbits) == (nq, nc)
    inputs = sorted(dag.input_nodes(), key=lambda n: n.node_id)
    outputs = sorted(dag.output_nodes(), key=lambda n: n.node_id)
    assert [n.node_id for n in inputs] == list(range(nq + nc))
    assert [n.label for n in inputs] == ["IN:q0", "IN:q1", "IN:q2", "IN:c0", "IN:c1"]
    assert [n.label for n in outputs] == [
        "OUT:q0",
        "OUT:q1",
        "OUT:q2",
        "OUT:c0",
        "OUT:c1",
    ]
    ops = dag.operation_nodes()
    assert [n.node_id for n in ops] == [nq + nc + k for k in range(3)]
    assert [n.instruction for n in ops] == list(circ.instructions)
    assert dag.nodes == sorted(dag.nodes, key=lambda n: n.node_id)
    assert dag.get_node(nq + nc + 1).qubits == {0, 2}
    assert dag.nodes_on_wire("c1") == [1 + nq, nq + nc + 2, nq + nc + 3 + nq + 1]
    assert dag.nodes_on_wire("q1") == [1, nq + nc + 3 + 1]
    assert dag.qubit_wires() == [0, 1, 2] and dag.clbit_wires() == [0, 1]


def test_summary_contents() -> None:
    """summary() reports node/edge counts and critical path."""
    dag = quspect.QuantumCircuit().h(0).cx(0, 1).measure([0, 1], [0, 1]).to_dag()
    text = dag.summary()
    assert text.splitlines()[0] == f"DAGCircuit: {dag.num_nodes()} nodes, 11 edges"
    assert "Qubits: 2" in text and "Clbits: 2" in text
    assert "OPERATION nodes: 3" in text
    assert "Critical path length: 3" in text
    assert "Depth (layers): 5" in text


# =============================================================================
# Topology vs qiskit circuit_to_dag
# =============================================================================

_TOPO_CASES = [(n, s) for n in range(2, 6) for s in range(4)]


@pytest.mark.parametrize(("n", "seed"), _TOPO_CASES)
def test_dag_topology_matches_qiskit(n: int, seed: int) -> None:
    """Node/edge counts, depth, layers, wire sequences and adjacency == qiskit."""
    circ = _random_mixed_circuit(n, 25, 100 * n + seed)
    dag = circ.to_dag()
    qdag = circuit_to_dag(to_qiskit(circ))
    # Measurements are single-qubit, so qiskit has one op node per instruction.
    assert len(dag.operation_nodes()) == qdag.size() == len(circ.instructions)
    assert dag.num_nodes() == len(list(qdag.nodes()))
    assert dag.num_edges() == len(list(qdag.edges()))
    assert (dag.num_qubits, dag.num_clbits) == (qdag.num_qubits(), qdag.num_clbits())
    assert dag.critical_path_length() == qdag.depth()
    assert len(dag.layers()) - 2 == len(list(qdag.layers()))

    qidx = _qiskit_op_index(qdag)
    midx = _quspect_op_index(dag)
    instrs = circ.instructions
    qc = to_qiskit(circ)
    for q in range(n):
        mine = [midx[nid] for nid in dag.nodes_on_wire(f"q{q}") if nid in midx]
        theirs = [
            qidx[node._node_id]
            for node in qdag.nodes_on_wire(qc.qubits[q], only_ops=True)
        ]
        assert mine == theirs
        assert [_op_name(instrs[k]) for k in mine] == [
            node.name for node in qdag.nodes_on_wire(qc.qubits[q], only_ops=True)
        ]
    for c in range(dag.num_clbits):
        mine = [midx[nid] for nid in dag.nodes_on_wire(f"c{c}") if nid in midx]
        theirs = [
            qidx[node._node_id]
            for node in qdag.nodes_on_wire(qc.clbits[c], only_ops=True)
        ]
        assert mine == theirs

    qnodes = {qidx[nd._node_id]: nd for nd in qdag.op_nodes()}
    for nid, k in midx.items():
        preds = {midx[p] for p in dag.predecessors(nid) if p in midx}
        succs = {midx[s] for s in dag.successors(nid) if s in midx}
        qpreds = {
            qidx[p._node_id]
            for p in qdag.predecessors(qnodes[k])
            if isinstance(p, DAGOpNode)
        }
        qsuccs = {
            qidx[s._node_id]
            for s in qdag.successors(qnodes[k])
            if isinstance(s, DAGOpNode)
        }
        assert preds == qpreds, f"predecessors of instruction {k}"
        assert succs == qsuccs, f"successors of instruction {k}"


@pytest.mark.parametrize("seed", range(3))
def test_topological_order_respects_wire_order(seed: int) -> None:
    """Every edge goes forward in topological order; layers are antichains."""
    dag = _random_mixed_circuit(4, 30, seed).to_dag()
    pos = {nid: i for i, nid in enumerate(dag.topological_order())}
    assert all(pos[e.source_id] < pos[e.target_id] for e in dag.edges)
    layer_of = {nid: d for d, layer in enumerate(dag.layers()) for nid in layer}
    assert all(layer_of[e.source_id] < layer_of[e.target_id] for e in dag.edges)
    for layer in dag.layers():
        assert len(layer) == len(set(layer))


# =============================================================================
# Round trip circuit -> DAG -> circuit
# =============================================================================


@pytest.mark.parametrize(("n", "seed"), [(n, s) for n in range(1, 6) for s in range(3)])
def test_roundtrip_preserves_unitary_exactly(n: int, seed: int) -> None:
    """DAGToCircuitCompiler(to_dag(c)) has the same instructions and unitary."""
    circ, qc = random_circuit_pair(n, 30, 7 * n + seed)
    back = DAGToCircuitCompiler.compile(circ.to_dag())
    # Kahn's order may interleave independent wires differently, but every
    # wire sees the same instruction sequence.
    assert sorted(map(repr, back.instructions)) == sorted(map(repr, circ.instructions))
    assert _wire_sequences(back) == _wire_sequences(circ)
    assert back.num_qubits == circ.num_qubits
    assert_unitary_equal(quspect_circuit_unitary(back), qiskit_circuit_unitary(qc))


@pytest.mark.parametrize("seed", range(3))
def test_roundtrip_with_measure_and_barrier(seed: int) -> None:
    """Measurements (incl. permuted clbits) and barriers survive the round trip."""
    circ = _random_mixed_circuit(3, 15, seed)
    circ.measure([0, 1], [1, 0])  # multi-qubit measure instruction
    back = DAGToCircuitCompiler.compile(circ.to_dag(), name="rt")
    assert back.name == "rt"
    assert _wire_sequences(back) == _wire_sequences(circ)
    assert len(back.instructions) == len(circ.instructions)
    assert (back.num_qubits, back.num_clbits) == (circ.num_qubits, circ.num_clbits)


def test_roundtrip_preserves_measurement_distribution() -> None:
    """Exact measurement distribution unchanged by the round trip."""
    circ, _ = random_circuit_pair(4, 30, 5)
    circ.measure([0, 1, 2, 3], [2, 0, 3, 1])
    back = DAGToCircuitCompiler.compile(CircuitCompiler.compile(circ))
    p0, p1 = qiskit_probabilities(circ), qiskit_probabilities(back)
    assert p0.keys() == p1.keys()
    for k in p0:
        assert p1[k] == pytest.approx(p0[k], abs=1e-12)


def test_roundtrip_is_topological_reordering_of_commuting_ops() -> None:
    """Gates on disjoint wires may be reordered but the unitary is unchanged."""
    circ = quspect.QuantumCircuit().h(0).x(1).cx(0, 1).rz(0.3, 2).swap(1, 2)
    back = DAGToCircuitCompiler.compile(circ.to_dag())
    assert _wire_sequences(back) == _wire_sequences(circ)
    assert_unitary_equal(quspect_circuit_unitary(back), qiskit_circuit_unitary(circ))


def test_dag_to_circuit_skips_invalid_nodes(caplog: pytest.LogCaptureFixture) -> None:
    """Nodes without an instruction / gate are skipped with a warning."""
    dag = quspect.QuantumCircuit().h(0).x(0).to_dag()
    dag.add_node(DAGNode(100, DAGNodeType.OPERATION, qubits={0}))
    broken = CircuitInstruction(type=InstructionType.GATE, gate=H, qubits=(0,))
    object.__setattr__(broken, "gate", None)
    dag.add_node(DAGNode(101, DAGNodeType.OPERATION, instruction=broken, qubits={0}))
    with caplog.at_level(logging.WARNING, logger="quspect.dag.dag_to_circuit"):
        back = DAGToCircuitCompiler.compile(dag)
    assert [i.gate.name for i in back.instructions if i.gate] == ["h", "x"]
    assert "no instruction" in caplog.text and "no gate" in caplog.text


def test_roundtrip_preserves_noise_channels() -> None:
    """A depolarizing channel must survive circuit -> DAG -> circuit."""
    circ = quspect.QuantumCircuit().h(0).depolarize(0.1, 0).cx(0, 1)
    back = DAGToCircuitCompiler.compile(circ.to_dag())
    assert len(back.instructions) == 3
    assert any(i.is_channel for i in back.instructions)


def test_dag_node_instruction_is_shared_not_copied() -> None:
    """Op nodes reference the very instruction objects of the circuit."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1)
    dag = circ.to_dag()
    for node, instr in zip(dag.operation_nodes(), circ.instructions, strict=True):
        assert node.instruction is instr
        assert dataclasses.is_dataclass(node)
