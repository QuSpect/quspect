"""Native (C++) DAG passes against their Python reference implementation.

Every pass is run on two copies of the same DAG -- once natively, once with
the Python implementation -- and the complete storage state (nodes and their
instructions, edge sequence numbers, wire bookkeeping, version counter,
incremental topological numbering, journal) must be identical.
"""

from __future__ import annotations

import copy
import logging
import math
import random

import pytest

from quspect import Parameter, QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.dag.compiler import CircuitCompiler
from quspect.dag.dag_circuit import DAGCircuit
from quspect.dag.dag_node import DAGEdge, DAGNode, DAGNodeType
from quspect.dag.passes import (
    CommutationAnalysis,
    EliminateDiagonalBeforeMeasurement,
    GateCancellation,
    PassManager,
    RemoveIdentityGates,
    RotationMerging,
    SingleQubitGateFusion,
    _native,
)
from quspect.dag.passes.base import (
    _native_repr_cache,
    _signature,
    _signatures_differ,
    _wire_sequences,
)
from quspect.gates.parameterized import RZ

pytestmark = pytest.mark.skipif(
    not _native.enabled(), reason="native DAG passes are not available"
)

_BASIS = frozenset({"cz", "h", "rz", "x", "sx", "s", "sdg", "t", "tdg", "y", "z"})


def _state(dag: DAGCircuit) -> tuple[object, ...]:
    st = dag._core_getstate()  # type: ignore[attr-defined]
    nodes = [(n[0], n[1], tuple(n[2]), repr(n[3].instruction)) for n in st[1]]
    return (nodes, *st[2:])


def _pipeline(basis: frozenset[str] | None = _BASIS) -> list[object]:
    return [
        GateCancellation(),
        CommutationAnalysis(),
        RotationMerging(),
        SingleQubitGateFusion(basis_gates=basis),
        EliminateDiagonalBeforeMeasurement(),
        RemoveIdentityGates(),
    ]


def _random_circuit(seed: int, n: int, m: int) -> QuantumCircuit:
    r = random.Random(seed)
    qc = QuantumCircuit()
    angles = [0.0, -0.0, math.pi, -math.pi / 2, 2 * math.pi, 1e-11, 0.3, -0.3]
    for _ in range(m):
        k = r.random()
        q = r.randrange(n)
        q2 = (q + 1 + r.randrange(n - 1)) % n
        a = r.choice(angles) if r.random() < 0.5 else r.uniform(-4, 4)
        if k < 0.15:
            qc.cx(q, q2)
        elif k < 0.22:
            qc.cz(q, q2)
        elif k < 0.3:
            qc.h(q)
        elif k < 0.45:
            qc.rz(a, q)
        elif k < 0.5:
            qc.rx(a, q)
        elif k < 0.55:
            qc.x(q)
        elif k < 0.6:
            qc.t(q)
        elif k < 0.65:
            qc.tdg(q)
        elif k < 0.7:
            qc.s(q)
        elif k < 0.73:
            qc.p(a, q)
        elif k < 0.76:
            qc.u3(a, 0.0, -a, q)
        elif k < 0.8:
            qc.z(q)
        elif k < 0.84:
            qc.ry(a, q)
        elif k < 0.87:
            qc.swap(q, q2)
        else:
            qc.sdg(q)
    qc.measure(list(range(n)), list(range(n)))
    return qc


def _assert_parity(dag: DAGCircuit, basis: frozenset[str] | None = _BASIS) -> None:
    native, python = copy.deepcopy(dag), copy.deepcopy(dag)
    pn, pp = _pipeline(basis), _pipeline(basis)
    for _ in range(4):
        for a, b in zip(pn, pp, strict=True):
            a.run(native)  # type: ignore[attr-defined]
            with _native.python_backend():
                b.run(python)  # type: ignore[attr-defined]
            assert _state(native) == _state(python), type(a).__name__


@pytest.mark.parametrize("seed", range(24))
def test_random_circuits_match_python(seed: int) -> None:
    qc = _random_circuit(seed, n=2 + seed % 5, m=20 + 7 * seed)
    _assert_parity(CircuitCompiler().compile(qc))


@pytest.mark.parametrize("seed", range(4))
def test_u3_fusion_mode_unchanged(seed: int) -> None:
    # Without a basis the fusion pass keeps its Python implementation.
    _assert_parity(CircuitCompiler().compile(_random_circuit(seed, 3, 60)), None)


def test_pass_manager_result_matches_python() -> None:
    qc = _random_circuit(99, 5, 300)
    dag_n = CircuitCompiler().compile(qc)
    dag_p = copy.deepcopy(dag_n)
    pm_n = PassManager(_pipeline())  # type: ignore[arg-type]
    pm_p = PassManager(_pipeline())  # type: ignore[arg-type]
    pm_n.run(dag_n)
    with _native.python_backend():
        pm_p.run(dag_p)
    assert _state(dag_n) == _state(dag_p)
    assert [
        (s.pass_name, s.nodes_before, s.nodes_after, s.changed)
        for s in pm_n.stats_history
    ] == [
        (s.pass_name, s.nodes_before, s.nodes_after, s.changed)
        for s in pm_p.stats_history
    ]


def test_symbolic_parameters_are_kept() -> None:
    theta = Parameter("theta")
    qc = QuantumCircuit().rz(theta, 0).rz(0.5, 0).rz(-0.5, 0).rx(theta, 1).rx(theta, 1)
    _assert_parity(CircuitCompiler().compile(qc))


def test_infinite_angle_raises_like_python() -> None:
    qc = QuantumCircuit().h(0).rz(math.inf, 0)
    dag = CircuitCompiler().compile(qc)
    with pytest.raises(ValueError):
        RemoveIdentityGates().run(dag)


def test_noncanonical_wire_falls_back() -> None:
    dag = CircuitCompiler().compile(QuantumCircuit().h(0).h(0))
    dag.add_edge(DAGEdge(source_id=0, target_id=1, wire="extra"))
    assert dag._native_gate_cancellation() is False  # type: ignore[attr-defined]
    GateCancellation().run(dag)


def test_subclassed_nodes_fall_back() -> None:
    class Node(DAGNode):
        pass

    dag = CircuitCompiler().compile(QuantumCircuit().h(0))
    dag.add_node(Node(node_id=99, node_type=DAGNodeType.OPERATION))
    assert dag._native_commutation(5) is False  # type: ignore[attr-defined]


def test_debug_logging_uses_python(caplog: pytest.LogCaptureFixture) -> None:
    dag = CircuitCompiler().compile(QuantumCircuit().h(0).h(0))
    with caplog.at_level(logging.DEBUG, logger="quspect.dag.passes.gate_cancellation"):
        GateCancellation().run(dag)
    assert any("Cancelled" in r.getMessage() for r in caplog.records)


def test_subclassed_pass_uses_python() -> None:
    calls: list[int] = []

    class Counting(GateCancellation):
        def _cancel_on_wire(self, dag, wire, removed):  # type: ignore[no-untyped-def]
            calls.append(1)
            return super()._cancel_on_wire(dag, wire, removed)

    Counting().run(CircuitCompiler().compile(QuantumCircuit().h(0).h(0)))
    assert calls


def test_signature_matches_wire_sequences() -> None:
    qc = _random_circuit(5, 4, 120)
    dag = CircuitCompiler().compile(qc)
    sig = _signature(dag, {}, _native_repr_cache())
    ref = _wire_sequences(dag, {})
    assert not isinstance(sig, dict)
    assert sig.to_dict() == ref
    assert not _signatures_differ(sig, ref)
    GateCancellation().run(dag)
    assert _signatures_differ(_signature(dag, {}, _native_repr_cache()), ref) == (
        _wire_sequences(dag, {}) != ref
    )


def test_native_repr_is_exact() -> None:
    cache = _native_repr_cache()
    instrs = [
        CircuitInstruction(InstructionType.GATE, (0,), gate=RZ.with_params(x))
        for x in (0.1, -0.0, 1e-320, 123456789.125, math.nan)
    ]
    qc = QuantumCircuit()
    for ins in instrs:
        qc.append(ins.gate, ins.qubits)  # type: ignore[arg-type]
    dag = CircuitCompiler().compile(qc)
    assert dag._wire_signature(cache).to_dict() == _wire_sequences(dag, {})  # type: ignore[attr-defined]


def test_fusion_cache_is_used() -> None:
    fusion = SingleQubitGateFusion(basis_gates=_BASIS)
    for _ in range(2):
        dag = CircuitCompiler().compile(QuantumCircuit().h(0).t(0).h(0).t(0).h(0))
        fusion.run(dag)
    assert len(fusion._native_run_cache) == 1
    assert len(fusion._run_cache) == 1


def test_backend_env_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_native.DAG_PASS_BACKEND_ENV, "bogus")
    with pytest.raises(ValueError, match="QUSPECT_DAG_PASS_BACKEND"):
        _native._backend_from_env()
    monkeypatch.setenv(_native.DAG_PASS_BACKEND_ENV, "python")
    assert _native._backend_from_env() == "python"
