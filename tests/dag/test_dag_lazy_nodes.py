"""Node objects of the native DAG core are created on first access.

The C++ core keeps a node's kind and instruction and builds the
:class:`~quspect.dag.dag_node.DAGNode` only when Python code asks for it
(``get_node``, ``_nodes``, ``operation_nodes``, copies, ...).  These tests
check that every observable — node fields and set iteration order, node
identity, the ``_nodes`` mapping, edge order, pass results, circuit
reconstruction — is the same as with the eagerly built reference DAG, and
that operations which do not need node objects work on the records.
"""

from __future__ import annotations

import copy
import pickle
import random
from typing import Any

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.dag.dag_circuit import DAGCircuit, _PyDAGCircuit, _PyDagCore
from quspect.dag.dag_node import DAGNode
from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
from quspect.dag.passes import (
    CommutationAnalysis,
    EliminateDiagonalBeforeMeasurement,
    GateCancellation,
    PassManager,
    RemoveIdentityGates,
    RotationMerging,
    SingleQubitGateFusion,
)
from quspect.gates.standard import CX, X

pytestmark = pytest.mark.skipif(
    issubclass(DAGCircuit, _PyDagCore),
    reason="native DAG core not built (DAGCircuit uses the Python core)",
)


def _random_circuit(rng: random.Random, n: int, m: int) -> QuantumCircuit:
    qc = QuantumCircuit()
    for _ in range(m):
        r = rng.random()
        a = rng.randrange(n)
        b = (a + 1 + rng.randrange(n - 1)) % n
        if r < 0.3:
            qc.cx(a, b)
        elif r < 0.4:
            qc.cz(a, b)
        elif r < 0.55:
            qc.rz(rng.choice([0.5, -0.5, 0.25]), a)
        elif r < 0.65:
            qc.h(a)
        elif r < 0.75:
            qc.x(a)
        elif r < 0.85:
            qc.t(a)
        elif r < 0.9:
            qc.barrier(a, b)
        elif r < 0.95:
            qc.measure([a], [rng.randrange(3)])
        elif n >= 3:
            c = next(q for q in range(n) if q not in (a, b))
            qc.ccx(a, b, c)
    return qc


def _pair(circuit: QuantumCircuit) -> tuple[DAGCircuit, _PyDAGCircuit]:
    native = circuit.to_dag()
    ref = _PyDAGCircuit()
    ref.set_qubit_count(circuit.num_qubits)
    ref.set_clbit_count(circuit.num_clbits)
    ref._build_from_instructions(
        circuit.instructions, circuit.num_qubits, circuit.num_clbits
    )
    return native, ref


def _node_fields(node: DAGNode) -> tuple[Any, ...]:
    return (
        node.node_id,
        node.node_type,
        node.instruction,
        list(node.qubits),  # set iteration order matters (edge order)
        list(node.clbits),
        node.label,
        node.metadata,
    )


def _structure(dag: Any) -> dict[str, Any]:
    """Observables that do not create node objects on the native core."""
    wires = [f"q{q}" for q in range(dag.num_qubits)]
    wires += [f"c{c}" for c in range(dag.num_clbits)]
    return {
        "keys": list(dag._nodes),
        "len": len(dag._nodes),
        "edges": dag._edge_triples(),
        "topo": dag.topological_order(),
        "wires": {w: dag.nodes_on_wire(w) for w in wires},
        "seq": dag._wire_sequences(None),
        "version": dag.modification_count,
        "counts": (dag.num_nodes(), dag.num_edges()),
    }


def _nodes(dag: Any) -> list[tuple[Any, ...]]:
    return [_node_fields(dag.get_node(i)) for i in list(dag._nodes)]


@pytest.mark.parametrize("seed", range(16))
def test_lazy_nodes_match_eager_reference(seed: int) -> None:
    rng = random.Random(seed)
    circuit = _random_circuit(rng, rng.randrange(2, 12), rng.randrange(1, 80))
    native, ref = _pair(circuit)
    assert _structure(native) == _structure(ref)
    # Access in a random order: every node equals the eagerly built one.
    ids = list(ref._nodes)
    rng.shuffle(ids)
    for i in ids:
        assert _node_fields(native.get_node(i)) == _node_fields(ref.get_node(i))
        assert native.get_node(i) == ref.get_node(i)
    assert _nodes(native) == _nodes(ref)


def test_two_qubit_set_order_matches_python_sets() -> None:
    # Pairs whose slots in an 8-entry set table collide or not, both orders.
    qc = QuantumCircuit()
    pairs = [(a, b) for a in range(20) for b in range(20) if a != b]
    for a, b in pairs:
        qc.cx(a, b)
    native, ref = _pair(qc)
    assert native._edge_triples() == ref._edge_triples()
    for node in native.operation_nodes():
        assert list(node.qubits) == list(set(node.instruction.qubits))


def test_node_identity_and_mutation_are_kept() -> None:
    dag = QuantumCircuit().h(0).cx(0, 1).to_dag()
    ops = dag.operation_nodes()
    assert ops[0] is dag.get_node(ops[0].node_id)
    assert dag._nodes[ops[1].node_id] is ops[1]
    ops[1].metadata["tag"] = 1
    assert dag.get_node(ops[1].node_id).metadata == {"tag": 1}
    assert dag.operation_nodes()[1] is ops[1]
    assert [n.node_id for n in dag.input_nodes()] == [0, 1]
    assert [n.label for n in dag.output_nodes()] == ["OUT:q0", "OUT:q1"]


def test_node_mapping_follows_dict_semantics() -> None:
    dag = QuantumCircuit().h(0).measure([0], [0]).to_dag()
    nodes = dag._nodes
    keys = list(nodes)
    assert keys == list(nodes.keys()) == sorted(keys)
    assert len(nodes) == dag.num_nodes()
    assert 1.0 in nodes and True in nodes and np.int64(2) in nodes
    assert nodes[True] is nodes[1] is nodes[np.int64(1)] is nodes[1.0]
    assert 1.5 not in nodes and "1" not in nodes and 10**30 not in nodes
    assert nodes.get(99) is None and nodes.get(99, "x") == "x"
    assert nodes.get(0) is dag.get_node(0)
    assert [k for k, _ in nodes.items()] == keys
    assert [v.node_id for v in nodes.values()] == keys
    for key in (99, "q", (1, 2), 1.5):
        with pytest.raises(KeyError) as info:
            dag.get_node(key)  # type: ignore[arg-type]
        assert info.value.args == (key,)
    with pytest.raises(TypeError):
        _ = [] in nodes
    with pytest.raises(TypeError):
        dag.get_node([])  # type: ignore[arg-type]
    assert dag.has_node(0) and not dag.has_node(99)


def test_mapping_order_after_remove_and_readd() -> None:
    native, ref = _pair(QuantumCircuit().h(0).x(0).cx(0, 1))
    for dag in (native, ref):
        node = dag.get_node(4)
        dag.remove_node(4)
        dag.add_node(node)
    assert list(native._nodes) == list(ref._nodes)
    assert native.nodes == ref.nodes


def test_substitute_on_lazy_nodes() -> None:
    native, ref = _pair(QuantumCircuit().h(0).h(1).cx(0, 1))
    same = CircuitInstruction(InstructionType.GATE, (0,), gate=X)
    other = CircuitInstruction(InstructionType.GATE, (1,), gate=X)
    two = CircuitInstruction(InstructionType.GATE, (0, 1), gate=CX)
    for dag in (native, ref):
        ops = [i for i in dag.topological_order() if i >= 4]
        dag.substitute_node_instruction(ops[0], same)
        # Different qubits: the node keeps the sets of its original instruction.
        dag.substitute_node_instruction(ops[1], other if ops[1] != 5 else two)
    assert _structure(native) == _structure(ref)
    assert _nodes(native) == _nodes(ref)
    with pytest.raises(KeyError):
        native.substitute_node_instruction(99, same)
    with pytest.raises(TypeError):
        native.substitute_node_instruction(4, "x")  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", range(6))
def test_remove_and_reconnect_on_lazy_nodes(seed: int) -> None:
    rng = random.Random(seed)
    circuit = _random_circuit(rng, 4, 40)
    native, ref = _pair(circuit)
    ops = [i for i in ref.topological_order() if ref.get_node(i).is_operation]
    removed = rng.sample(ops, min(10, len(ops)))
    for i in removed:
        native.remove_node_and_reconnect(i)
        ref.remove_node_and_reconnect(i)
        assert _structure(native) == _structure(ref)
    assert _nodes(native) == _nodes(ref)
    with pytest.raises(KeyError):
        native.remove_node_and_reconnect(removed[0] if removed else 10**6)


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_passes_on_lazy_dag(seed: int, materialize: bool) -> None:
    rng = random.Random(50 + seed)
    circuit = _random_circuit(rng, rng.randrange(2, 7), rng.randrange(20, 120))
    native, ref = _pair(circuit)
    if materialize:
        _ = native.nodes

    def pipeline() -> PassManager:
        return PassManager(
            [
                GateCancellation(),
                CommutationAnalysis(),
                RotationMerging(),
                SingleQubitGateFusion(basis_gates=frozenset({"cz", "rz", "sx", "x"})),
                EliminateDiagonalBeforeMeasurement(),
                RemoveIdentityGates(),
            ]
        )

    out_native = pipeline().run(native, max_iterations=50)
    out_ref = pipeline().run(ref, max_iterations=50)
    assert _structure(out_native) == _structure(out_ref)
    a = DAGToCircuitCompiler.compile(out_native)
    b = DAGToCircuitCompiler.compile(out_ref)
    assert a.instructions == b.instructions
    assert _nodes(out_native) == _nodes(out_ref)


def test_dag_to_circuit_fast_path_and_fallback() -> None:
    qc = QuantumCircuit().h(0).cx(0, 1).barrier(0, 1).measure([0, 1], [0, 1])
    dag = qc.to_dag()
    fast = dag._operation_instructions()
    assert fast is not None and tuple(fast) == qc.instructions
    out = DAGToCircuitCompiler.compile(dag)
    assert out.instructions == qc.instructions
    assert all(a is b for a, b in zip(out.instructions, qc.instructions, strict=True))
    # A labelled instruction takes the per-node path (rebuilt without label).
    labelled = QuantumCircuit()
    labelled._instructions.append(
        CircuitInstruction(InstructionType.GATE, (0,), gate=X, label="mark")
    )
    dag = labelled.to_dag()
    assert dag._operation_instructions() is None
    assert (
        DAGToCircuitCompiler.compile(dag).instructions
        == QuantumCircuit().x(0).instructions
    )


def test_copy_and_pickle_of_lazy_dag() -> None:
    rng = random.Random(7)
    circuit = _random_circuit(rng, 5, 60)
    lazy = circuit.to_dag()
    eager = circuit.to_dag()
    _ = eager.nodes
    for clone in (copy.deepcopy(lazy), pickle.loads(pickle.dumps(lazy))):
        assert _structure(clone) == _structure(eager)
        assert _nodes(clone) == _nodes(eager)
    assert _nodes(lazy) == _nodes(eager)


def test_nonstandard_instructions_build_nodes_eagerly() -> None:
    class Sub(CircuitInstruction):
        pass

    qc = QuantumCircuit()
    qc._instructions.append(Sub(InstructionType.GATE, (0,), gate=X))
    qc.cx(0, 1)
    native, ref = _pair(qc)
    assert _structure(native) == _structure(ref)
    assert _nodes(native) == _nodes(ref)
    assert type(native.get_node(2).instruction) is Sub
