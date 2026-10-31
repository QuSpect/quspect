"""Differential tests: the C++ DAG storage core against the Python reference.

:class:`~quspect.dag.dag_circuit.DAGCircuit` derives from the C++ ``_DagCore``
when the native extension is built and from ``_PyDagCore`` otherwise.  Both
must behave identically — topological order, edge order, wire walks,
incremental topological numbering, transaction rollback, modification
counter and exceptions — because optimization passes (and therefore
compilation results) depend on every one of them.  These tests drive both
implementations with the same random operation sequences and with the full
optimization pass pipeline and compare every observable.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from quspect import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.dag.dag_circuit import DAGCircuit, _PyDAGCircuit, _PyDagCore
from quspect.dag.dag_node import DAGEdge, DAGNode, DAGNodeType
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
from quspect.gates.standard import X

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
        if r < 0.25:
            qc.cx(a, b)
        elif r < 0.35:
            qc.cz(a, b)
        elif r < 0.5:
            qc.rz(rng.choice([0.5, -0.5, 0.25, 3.14159]), a)
        elif r < 0.6:
            qc.h(a)
        elif r < 0.7:
            qc.x(a)
        elif r < 0.78:
            qc.sx(a)
        elif r < 0.85:
            qc.t(a)
        elif r < 0.9:
            qc.tdg(a)
        elif r < 0.95:
            qc.rx(rng.choice([0.3, -0.3]), a)
        elif r < 0.98:
            qc.barrier(a, b)
        elif n >= 3:
            c = next(q for q in range(n) if q not in (a, b))
            qc.ccx(a, b, c)
    if rng.random() < 0.5:
        qc.measure_all()
    return qc


def _build_pair(circuit: QuantumCircuit) -> tuple[DAGCircuit, _PyDAGCircuit]:
    native = circuit.to_dag()
    ref = _PyDAGCircuit()
    ref.set_qubit_count(circuit.num_qubits)
    ref.set_clbit_count(circuit.num_clbits)
    ref._build_from_instructions(
        circuit.instructions, circuit.num_qubits, circuit.num_clbits
    )
    return native, ref


def _snapshot(dag: Any) -> dict[str, Any]:
    wires = [f"q{q}" for q in range(dag.num_qubits)]
    wires += [f"c{c}" for c in range(dag.num_clbits)]
    ids = [n.node_id for n in dag.nodes]
    return {
        "ids": ids,
        "dict_order": list(dag._nodes),
        "nodes": [repr(n) for n in dag.nodes],
        "edges": dag.edges,
        "topo": _call(dag, "topological_order"),
        "ops": _call(dag, "operation_nodes"),
        "wires": {w: _call(dag, "nodes_on_wire", w) for w in wires},
        "layers": _call(dag, "layers"),
        "critical": _call(dag, "critical_path_length"),
        "wire_sequences": _call(dag, "_wire_sequences", None),
        "wire_sequences_cached": _call(dag, "_wire_sequences", {}),
        "version": dag.modification_count,
        "counts": (dag.num_nodes(), dag.num_edges(), dag.num_qubits, dag.num_clbits),
        "adj": {
            i: (
                dag.predecessors(i),
                dag.successors(i),
                dag.out_edges_with_wires(i),
                dag.in_edges_with_wires(i),
                [dag.wire_predecessor(i, w) for w in wires],
                [dag.wire_successor(i, w) for w in wires],
            )
            for i in ids
        },
    }


def _call(dag: Any, name: str, *args: Any, **kw: Any) -> tuple[str, Any]:
    try:
        result = getattr(dag, name)(*args, **kw)
    except Exception as exc:  # noqa: BLE001 - compare exception type and text
        return type(exc).__name__, str(exc)
    if isinstance(result, list) and result and isinstance(result[0], DAGNode):
        result = [n.node_id for n in result]
    return "ok", result


def _same(a: Any, b: Any) -> None:
    sa, sb = _snapshot(a), _snapshot(b)
    for key in sa:
        assert sa[key] == sb[key], key


def _swap_on_wire(dag: Any, a: int, wire: str) -> tuple[str, Any]:
    """Swap node ``a`` with its successor on ``wire`` (commutation-pass style)."""
    b = dag.wire_successor(a, wire)
    pred = dag.wire_predecessor(a, wire)
    if b is None or pred is None or not dag.get_node(b).is_operation:
        return "skip", None
    succ = dag.wire_successor(b, wire)
    if succ is None:
        return "skip", None
    dag.ensure_topological_positions()
    sp = dag.begin_transaction()
    for s, t in ((pred, a), (a, b), (b, succ)):
        dag.remove_edge(s, t, wire)
    for s, t in ((pred, b), (b, a), (a, succ)):
        dag.add_edge(DAGEdge(s, t, wire), defer_order_check=True)
    ok = dag.reverse_adjacent_edge_order(a, b)
    if not ok:
        dag.rollback(sp)
    dag.end_transaction()
    return "ok", ok


@pytest.mark.parametrize("seed", range(12))
def test_random_operations_match_python_core(seed: int) -> None:
    rng = random.Random(seed)
    circuit = _random_circuit(rng, rng.randrange(2, 6), rng.randrange(5, 40))
    native, ref = _build_pair(circuit)
    _same(native, ref)
    next_id = 10_000
    for _ in range(60):
        ops = [n.node_id for n in ref.operation_nodes()]
        wires = [f"q{q}" for q in range(ref.num_qubits)]
        r = rng.random()
        if r < 0.3 and ops:
            a = rng.choice(ops)
            w = rng.choice(wires)
            assert _swap_on_wire(native, a, w) == _swap_on_wire(ref, a, w)
        elif r < 0.45 and ops:
            a = rng.choice(ops)
            assert _call(native, "remove_node_and_reconnect", a) == _call(
                ref, "remove_node_and_reconnect", a
            )
        elif r < 0.55 and ops:
            a = rng.choice(ops)
            instr = CircuitInstruction(type=InstructionType.GATE, gate=X, qubits=(0,))
            node_a, node_r = native.get_node(a), ref.get_node(a)
            if node_a.instruction is not None and node_a.instruction.qubits == (0,):
                native.substitute_node_instruction(a, instr)
                ref.substitute_node_instruction(a, instr)
                assert node_r.instruction == node_a.instruction
        elif r < 0.65:
            # add an operation node spliced after a random node on q0
            q = rng.randrange(ref.num_qubits)
            w = f"q{q}"
            anchor = rng.choice(ref.nodes_on_wire(w)[:-1])
            for dag in (native, ref):
                succ = dag.wire_successor(anchor, w)
                dag.add_node(
                    DAGNode(
                        node_id=next_id,
                        node_type=DAGNodeType.OPERATION,
                        instruction=CircuitInstruction(
                            type=InstructionType.GATE, gate=X, qubits=(q,)
                        ),
                        qubits={q},
                    )
                )
                dag.remove_edge(anchor, succ, w)
                dag.add_edge(DAGEdge(anchor, next_id, w))
                dag.add_edge(DAGEdge(next_id, succ, w))
            next_id += 1
        elif r < 0.7:
            assert _call(native, "add_node", DAGNode(0, DAGNodeType.INPUT)) == _call(
                ref, "add_node", DAGNode(0, DAGNodeType.INPUT)
            )
        elif r < 0.75:
            bad = rng.randrange(1_000_000, 2_000_000)
            assert _call(native, "remove_node", bad) == _call(ref, "remove_node", bad)
            assert _call(native, "remove_edge", 0, bad, "q0") == _call(
                ref, "remove_edge", 0, bad, "q0"
            )
            assert _call(native, "rollback", 0) == _call(ref, "rollback", 0)
        elif r < 0.85 and ops:
            # transaction with mixed edits, then roll back everything
            a = rng.choice(ops)
            for dag in (native, ref):
                sp = dag.begin_transaction()
                dag.ensure_topological_positions()
                for tgt, w in list(dag.out_edges_with_wires(a)):
                    dag.remove_edge(a, tgt, w)
                    dag.add_edge(DAGEdge(a, tgt, w))
                for src, w in list(dag.in_edges_with_wires(a)):
                    dag.remove_edge(src, a, w)
                dag.rollback(sp)
                dag.end_transaction()
        else:
            for dag in (native, ref):
                dag.ensure_topological_positions()
        _same(native, ref)


@pytest.mark.parametrize("seed", range(10))
def test_optimization_pipeline_matches_python_core(seed: int) -> None:
    rng = random.Random(100 + seed)
    circuit = _random_circuit(rng, rng.randrange(2, 7), rng.randrange(20, 120))
    native, ref = _build_pair(circuit)

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
    _same(out_native, out_ref)
    a = DAGToCircuitCompiler.compile(out_native)
    b = DAGToCircuitCompiler.compile(out_ref)
    assert a.instructions == b.instructions


def test_non_canonical_wire_labels() -> None:
    """Labels other than ``q<i>`` / ``c<i>`` keep the Python semantics."""
    native, ref = _build_pair(QuantumCircuit().h(0).cx(0, 1).h(1))
    for dag in (native, ref):
        dag.add_edge(DAGEdge(2, 3, "anc"))
        dag.add_edge(DAGEdge(3, 4, "q01"))
    _same(native, ref)
    for label in ("anc", "q01", "q", "zz", "c007"):
        assert _call(native, "nodes_on_wire", label) == _call(
            ref, "nodes_on_wire", label
        )
        assert native.wire_successor(2, label) == ref.wire_successor(2, label)
        assert native.has_edge(2, 3, label) == ref.has_edge(2, 3, label)
    assert _call(native, "remove_edge", 3, 4, "q01") == _call(
        ref, "remove_edge", 3, 4, "q01"
    )
    assert _call(native, "remove_edge", 3, 4, "q01") == _call(
        ref, "remove_edge", 3, 4, "q01"
    )
    _same(native, ref)


def test_cycle_and_dangling_edges() -> None:
    native, ref = _build_pair(QuantumCircuit().h(0).x(0))
    for dag in (native, ref):
        dag.add_edge(DAGEdge(3, 2, "q0"))
    assert _call(native, "topological_order") == _call(ref, "topological_order")
    assert _call(native, "reverse_adjacent_edge_order", 2, 3) == _call(
        ref, "reverse_adjacent_edge_order", 2, 3
    )
    native2, ref2 = _build_pair(QuantumCircuit().h(0))
    for dag in (native2, ref2):
        dag.add_edge(DAGEdge(1, 99, "q0"))
    assert _call(native2, "topological_order") == _call(ref2, "topological_order")
    assert _call(native2, "layers") == _call(ref2, "layers")


@pytest.mark.parametrize("make", [DAGCircuit, _PyDAGCircuit])
@pytest.mark.parametrize("seed", range(4))
def test_copy_and_pickle_preserve_state(make: Any, seed: int) -> None:
    import copy
    import pickle

    rng = random.Random(500 + seed)
    circuit = _random_circuit(rng, 4, 30)
    dag = make()
    dag.set_qubit_count(circuit.num_qubits)
    dag.set_clbit_count(circuit.num_clbits)
    dag._build_from_instructions(
        circuit.instructions, circuit.num_qubits, circuit.num_clbits
    )
    dag.add_edge(DAGEdge(0, dag.num_nodes() - 1, "anc"))
    dag.ensure_topological_positions()
    ops = [n.node_id for n in dag.operation_nodes()]
    dag.remove_node_and_reconnect(ops[0])
    sp = dag.begin_transaction()  # copies carry the open transaction
    _swap_on_wire(dag, ops[1], f"q{dag.get_node(ops[1]).instruction.qubits[0]}")
    dag.tag = "user attribute"

    for clone in (
        copy.deepcopy(dag),
        pickle.loads(pickle.dumps(dag)),
        copy.copy(dag),
    ):
        assert type(clone) is type(dag)
        assert clone.tag == "user attribute"
        _same(clone, dag)
        assert clone.get_node(ops[1]) == dag.get_node(ops[1])
        assert clone.has_node(ops[1]) and not clone.has_node(ops[0])
    deep = copy.deepcopy(dag)
    assert deep.get_node(ops[1]) is not dag.get_node(ops[1])
    # the copy rolls back independently and identically
    before = _snapshot(dag)
    deep.rollback(sp)
    deep.end_transaction()
    assert _snapshot(dag) == before
    dag.rollback(sp)
    dag.end_transaction()
    _same(deep, dag)
