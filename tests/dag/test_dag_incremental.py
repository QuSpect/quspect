"""Tests for the incremental DAGCircuit indexes (wire walks, O(degree) edits)."""

from __future__ import annotations

import random
import time

import pytest

from quspect import QuantumCircuit
from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode, DAGNodeType


def _random_circuit(seed: int, n: int = 4, depth: int = 60) -> QuantumCircuit:
    rng = random.Random(seed)
    c = QuantumCircuit()
    for _ in range(depth):
        k = rng.random()
        if k < 0.4:
            getattr(c, rng.choice(["h", "x", "s", "t", "z"]))(rng.randrange(n))
        elif k < 0.6:
            c.rz(rng.uniform(-3, 3), rng.randrange(n))
        elif k < 0.95:
            a, b = rng.sample(range(n), 2)
            c.cx(a, b)
        else:
            c.barrier(*rng.sample(range(n), 2))
    c.measure(list(range(n)), list(range(n)))
    return c


def _topo_filter(dag: DAGCircuit, wire: str) -> list[int]:
    idx = int(wire[1:])
    result = []
    for nid in dag.topological_order():
        node = dag.get_node(nid)
        bits = node.qubits if wire[0] == "q" else node.clbits
        if idx in bits:
            result.append(nid)
    return result


@pytest.mark.parametrize("seed", range(10))
def test_wire_walk_matches_topological_filter(seed: int) -> None:
    """The O(wire) walk returns exactly the topological-scan result."""
    dag = _random_circuit(seed).to_dag()
    wires = [f"q{q}" for q in range(dag.num_qubits)]
    wires += [f"c{c}" for c in range(dag.num_clbits)]
    for wire in wires:
        assert dag.nodes_on_wire(wire) == _topo_filter(dag, wire)


def test_remove_node_does_not_reconnect() -> None:
    """remove_node keeps its historical semantics (no splicing)."""
    dag = QuantumCircuit().h(0).x(0).to_dag()
    h, x = (n.node_id for n in dag.operation_nodes())
    dag.remove_node(h)
    assert dag.wire_predecessor(x, "q0") is None
    assert dag.num_edges() == 1


def test_remove_node_and_reconnect_two_qubit() -> None:
    """Splicing out a CX reconnects both wires."""
    dag = QuantumCircuit().h(0).h(1).cx(0, 1).x(0).x(1).to_dag()
    ops = dag.operation_nodes()
    cx = next(n for n in ops if n.instruction and n.instruction.gate.name == "cx")
    dag.remove_node_and_reconnect(cx.node_id)
    names = [n.instruction.gate.name for n in dag.operation_nodes()]
    assert sorted(names) == ["h", "h", "x", "x"]
    for wire in ("q0", "q1"):
        seq = dag.nodes_on_wire(wire)
        assert len(seq) == 4  # IN, h, x, OUT


def test_rollback_restores_exact_edge_order() -> None:
    """A transaction rollback restores the previous edge list exactly."""
    dag = _random_circuit(3).to_dag()
    before = [(e.source_id, e.target_id, e.wire) for e in dag.edges]
    order_before = dag.topological_order()
    sp = dag.begin_transaction()
    for e in list(dag.edges)[:10]:
        dag.remove_edge(e.source_id, e.target_id, e.wire)
    dag.add_edge(DAGEdge(0, 1, "q9"))
    dag.rollback(sp)
    dag.end_transaction()
    assert [(e.source_id, e.target_id, e.wire) for e in dag.edges] == before
    assert dag.topological_order() == order_before


def test_reverse_adjacent_edge_order_detects_cycle() -> None:
    """cx(0,1) cx(1,2)-path cx(0,2): reversing would create a cycle."""
    dag = QuantumCircuit().cx(0, 1).cx(1, 2).cx(0, 2).to_dag()
    dag.ensure_topological_positions()
    a, _, b = (n.node_id for n in dag.operation_nodes())
    # a and b are adjacent on q0; a -> (q1) -> mid -> (q2) -> b also exists.
    sp = dag.begin_transaction()
    pred = dag.wire_predecessor(a, "q0")
    succ = dag.wire_successor(b, "q0")
    assert pred is not None and succ is not None
    for s, t in ((pred, a), (a, b), (b, succ)):
        dag.remove_edge(s, t, "q0")
    for s, t in ((pred, b), (b, a), (a, succ)):
        dag.add_edge(DAGEdge(s, t, "q0"), defer_order_check=True)
    assert dag.reverse_adjacent_edge_order(a, b) is False
    dag.rollback(sp)
    dag.end_transaction()
    dag.topological_order()  # still acyclic


def test_nodes_on_wire_falls_back_for_non_chain_wires() -> None:
    """A hand-built diamond on one wire is still handled (topological scan)."""
    dag = DAGCircuit()
    dag.set_qubit_count(1)
    dag.add_node(DAGNode(node_id=0, node_type=DAGNodeType.INPUT, qubits={0}))
    for nid in (1, 2, 3):
        dag.add_node(DAGNode(node_id=nid, node_type=DAGNodeType.OPERATION, qubits={0}))
    dag.add_node(DAGNode(node_id=4, node_type=DAGNodeType.OUTPUT, qubits={0}))
    for s, t in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
        dag.add_edge(DAGEdge(s, t, "q0"))
    assert dag.nodes_on_wire("q0") == [0, 1, 2, 3, 4]


def test_modification_count_increases() -> None:
    """Structural edits bump the modification counter."""
    dag = QuantumCircuit().h(0).h(0).to_dag()
    c0 = dag.modification_count
    dag.remove_node_and_reconnect(dag.operation_nodes()[0].node_id)
    assert dag.modification_count > c0


def test_optimize_pass_scales_linearly() -> None:
    """OptimizePass on ~6k gates finishes quickly (was O(N^2): ~40 s)."""
    from quspect.transpiler._passes import OptimizePass

    rng = random.Random(0)
    n = 20
    c = QuantumCircuit()
    for _ in range(2000):
        a, b = rng.sample(range(n), 2)
        c.h(b).cz(a, b).h(b)
        c.rz(rng.uniform(-1, 1), a)
    t0 = time.perf_counter()
    OptimizePass().run(c)
    assert time.perf_counter() - t0 < 15.0
