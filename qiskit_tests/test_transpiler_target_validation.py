"""Tests for ``CompilationTarget``, the transpiler passes and ``validation``.

* :class:`CompilationTarget`: ``from_coupling_map`` / ``from_backend`` /
  ``induced_subgraph`` -- compared with qiskit ``CouplingMap`` distances and
  connectivity.
* ``quspect.transpiler._passes``: ``DecomposePass``, ``LayoutRoutingPass``,
  ``OptimizePass`` and ``TranspilerPipeline`` -- semantic checks with qiskit
  ``Operator``.
* ``quspect.transpiler.validation``: every branch of
  :func:`generate_random_circuit` and :func:`check_compilation_consistency`,
  with the QASM/qiskit statevector cross-checked against
  ``Statevector(to_qiskit(circ))``.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import quspect
from qiskit_tests._dag_helpers import (
    grid_edges,
    line_edges,
    permutation_matrix,
    quspect_gate_counts,
    ring_edges,
    routed_permutation,
    target_from_edges,
    two_qubit_operands,
)
from qiskit_tests.helpers import (
    assert_states_equal,
    assert_unitary_equal,
    qiskit_circuit_unitary,
    random_circuit_pair,
    to_qiskit,
)
from quspect.circuit.parameter import Parameter
from quspect.transpiler import (
    SABRE,
    CompilationConsistencyResult,
    CompilationTarget,
    check_compilation_consistency,
    generate_random_circuit,
)
from quspect.transpiler._mapping import _floyd_warshall
from quspect.transpiler._passes import (
    DecomposePass,
    LayoutRoutingPass,
    OptimizePass,
    TranspilerPipeline,
)
from quspect.transpiler.validation import _state_fidelity

pytestmark = pytest.mark.qiskit_ref


# =============================================================================
# CompilationTarget
# =============================================================================


@pytest.mark.parametrize(
    ("n", "edges"),
    [
        (5, line_edges(5)),
        (6, ring_edges(6)),
        (9, grid_edges(3, 3)),
        (4, [(0, 1), (1, 0), (2, 2), (3, 1)]),
    ],
)
def test_from_coupling_map_matches_qiskit_coupling_map(
    n: int, edges: list[tuple[int, int]]
) -> None:
    """Undirected normalised edges, adjacency and distances == qiskit."""
    target = CompilationTarget.from_coupling_map(n, edges, native_gates={"cx"})
    assert target.num_qubits == n and target.native_gates == {"cx"}
    expected = {(min(a, b), max(a, b)) for a, b in edges if a != b}
    assert target.coupling_graph == expected
    for q in range(n):
        assert sorted(target.adjacency[q]) == sorted(
            {b for a, b in expected if a == q} | {a for a, b in expected if b == q}
        )
    cmap = CouplingMap([list(e) for e in expected] + [[b, a] for a, b in expected])
    dist = _floyd_warshall(target.adjacency, n)
    if sorted(cmap.physical_qubits) == list(range(n)) and cmap.is_connected():
        for a in range(n):
            for b in range(n):
                assert dist[a][b] == int(cmap.distance(a, b))
    else:
        assert max(max(row) for row in dist) >= 10**9
    assert CompilationTarget.from_coupling_map(2, []).native_gates == set()


def test_floyd_warshall_disconnected_is_infinite() -> None:
    """Disconnected pairs keep the INF sentinel."""
    target = target_from_edges(4, [(0, 1), (2, 3)])
    dist = _floyd_warshall(target.adjacency, 4)
    assert dist[0][1] == 1 and dist[0][2] >= 10**9


def test_from_backend_reads_topology_and_native_gates() -> None:
    """from_backend normalises coupling_map pairs and copies native gates."""
    backend: Any = SimpleNamespace(
        config={
            "topology": {"coupling_map": [[0, 1], (2, 1), [1, 0], [3, 3], [5], "x"]}
        },
        num_qubits=4,
        native_gate_set={"cz", "rx"},
    )
    target = CompilationTarget.from_backend(backend)
    assert target.num_qubits == 4
    assert target.coupling_graph == {(0, 1), (1, 2)}
    assert sorted(target.adjacency[1]) == [0, 2] and target.adjacency[3] == []
    assert target.native_gates == {"cz", "rx"}


@pytest.mark.parametrize(
    "backend",
    [
        SimpleNamespace(num_qubits=2),
        SimpleNamespace(config="bogus", num_qubits=2),
        SimpleNamespace(config={"topology": []}, num_qubits=2),
        SimpleNamespace(config={"topology": {"coupling_map": "0-1"}}, num_qubits=2),
    ],
)
def test_from_backend_tolerates_missing_or_malformed_config(backend: Any) -> None:
    """Missing / malformed config gives an edgeless target, no crash."""
    target = CompilationTarget.from_backend(backend)
    assert target.coupling_graph == set() and target.adjacency == [[], []]
    assert target.native_gates == set()


def test_induced_subgraph_dense_relabelling() -> None:
    """induced_subgraph keeps only internal edges and relabels densely."""
    target = CompilationTarget.from_coupling_map(
        6, grid_edges(2, 3), native_gates={"cx", "h"}
    )
    dense, mapping = target.induced_subgraph([4, 1, 2, 5])
    assert mapping == [4, 1, 2, 5]
    # physical edges among {1,2,4,5}: (1,2), (1,4), (2,5), (4,5)
    assert dense.coupling_graph == {(1, 2), (0, 1), (2, 3), (0, 3)}
    assert dense.native_gates == {"cx", "h"} and dense.num_qubits == 4
    dense.native_gates.add("x")
    assert target.native_gates == {"cx", "h"}


@pytest.mark.parametrize(
    ("qubits", "exc", "match"),
    [
        ({0, 1}, TypeError, "list or tuple"),
        ([], ValueError, "empty"),
        ([0, "1"], TypeError, "integer"),
        ([0, False], TypeError, "integer"),
        ([1, 1], ValueError, "duplicates"),
        ([0, 9], ValueError, "outside"),
    ],
)
def test_induced_subgraph_validation(
    qubits: object, exc: type[Exception], match: str
) -> None:
    """Invalid qubit selections raise the documented errors."""
    target = target_from_edges(3, line_edges(3))
    with pytest.raises(exc, match=match):
        target.induced_subgraph(qubits)  # type: ignore[arg-type]


def test_induced_subgraph_route_and_map_back() -> None:
    """Route on a dense subgraph, map wires back to device qubits."""
    device = target_from_edges(9, grid_edges(3, 3))
    dense, to_phys = device.induced_subgraph([0, 1, 2, 5, 4])
    circ, _ = random_circuit_pair(4, 25, 2, gate_set=["h", "rz", "cx", "cz"])
    routed, initial, final = SABRE(dense).solve(circ, seed=3)
    for a, b in two_qubit_operands(routed):
        pa, pb = to_phys[initial[a]], to_phys[initial[b]]
        assert (min(pa, pb), max(pa, pb)) in device.coupling_graph
    m = len(initial)
    assert_unitary_equal(
        qiskit_circuit_unitary(routed, num_qubits=m),
        permutation_matrix(routed_permutation(initial, final))
        @ qiskit_circuit_unitary(circ, num_qubits=m),
    )


# =============================================================================
# Transpiler passes
# =============================================================================


def test_decompose_pass_to_native_set() -> None:
    """DecomposePass lowers to the native set; unitary kept up to phase."""
    target = CompilationTarget.from_coupling_map(
        3, line_edges(3), native_gates={"h", "rz", "cz", "rx", "ry", "t", "tdg", "cx"}
    )
    circ = quspect.QuantumCircuit().ccx(0, 1, 2).swap(0, 2).cp(0.4, 1, 0).u3(1, 2, 3, 2)
    out = DecomposePass(target).run(circ)
    assert set(quspect_gate_counts(out)) <= target.native_gates
    assert_unitary_equal(
        qiskit_circuit_unitary(out, num_qubits=3),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


def test_decompose_pass_without_native_gates_is_identity() -> None:
    """Empty native set: the circuit object is returned unchanged."""
    circ = quspect.QuantumCircuit().ccx(0, 1, 2)
    target = target_from_edges(3, line_edges(3))
    assert DecomposePass(target).run(circ) is circ


def test_layout_routing_pass_wraps_sabre() -> None:
    """LayoutRoutingPass == SABRE.solve with the same seed / mapping."""
    target = target_from_edges(5, ring_edges(5))
    circ, _ = random_circuit_pair(4, 30, 6, gate_set=["h", "cx", "rz", "cz"])
    lr = LayoutRoutingPass(target, iter_num=2, seed=7)
    assert lr.target is target
    routed, initial, final = lr.run(circ)
    ref = SABRE(target).solve(circ, iter_num=2, seed=7)
    assert (initial, final) == (ref[1], ref[2])
    assert [repr(i) for i in routed.instructions] == [
        repr(i) for i in ref[0].instructions
    ]
    fixed = LayoutRoutingPass(target, initial_mapping=[4, 3, 2, 1]).run(circ)
    assert fixed[1][:4] == [4, 3, 2, 1]


@pytest.mark.parametrize("seed", range(4))
def test_optimize_pass_preserves_unitary(seed: int) -> None:
    """OptimizePass output == input up to global phase, never larger."""
    circ, _ = random_circuit_pair(4, 40, 50 + seed)
    out = OptimizePass().run(circ)
    assert len(out.instructions) <= len(circ.instructions)
    assert_unitary_equal(
        qiskit_circuit_unitary(out, num_qubits=4),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


def test_optimize_pass_reduces_known_patterns() -> None:
    """Inverse pairs and rotation chains are removed by OptimizePass."""
    circ = quspect.QuantumCircuit().h(0).h(0).rz(0.2, 1).rz(0.3, 1).cx(0, 1).cx(0, 1)
    out = OptimizePass().run(circ)
    assert sum(quspect_gate_counts(out).values()) == 1


def test_optimize_pass_commutation_cancellation_bug() -> None:
    """cx(0,1) rz(0) cx(0,1) should optimize to a single rz."""
    circ = quspect.QuantumCircuit().cx(0, 1).rz(0.3, 0).cx(0, 1)
    out = OptimizePass().run(circ)
    assert sum(quspect_gate_counts(out).values()) == 1


def test_transpiler_pipeline_end_to_end() -> None:
    """Decompose -> route pipeline returns the routing triple, equivalent."""
    target = CompilationTarget.from_coupling_map(
        4, line_edges(4), native_gates={"h", "t", "tdg", "cx", "rz", "ry", "rx"}
    )
    circ = quspect.QuantumCircuit().h(0).ccx(0, 3, 2).ry(0.4, 1).cx(3, 1)
    pipeline = TranspilerPipeline(
        [DecomposePass(target), LayoutRoutingPass(target, seed=5)]
    )
    routed, initial, final = pipeline.run(circ)
    for a, b in two_qubit_operands(routed):
        assert (min(initial[a], initial[b]), max(initial[a], initial[b])) in (
            target.coupling_graph
        )
    m = len(initial)
    assert_unitary_equal(
        qiskit_circuit_unitary(routed, num_qubits=m),
        permutation_matrix(routed_permutation(initial, final))
        @ qiskit_circuit_unitary(circ, num_qubits=m),
        up_to_phase=True,
    )
    assert TranspilerPipeline([]).run(circ) is circ


# =============================================================================
# generate_random_circuit
# =============================================================================

_SINGLE = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p", "u3"}
_TWO = {"cx", "cz", "swap", "cp", "crx", "cry", "crz"}


@pytest.mark.parametrize(("n", "depth", "seed"), [(1, 3, 0), (3, 5, 1), (5, 8, 2)])
def test_generate_random_circuit_structure(n: int, depth: int, seed: int) -> None:
    """Each layer has one 1q gate per qubit; 2q gates from the documented set."""
    circ = generate_random_circuit(n, depth, seed=seed, two_qubit_probability=0.7)
    counts = quspect_gate_counts(circ)
    assert set(counts) <= _SINGLE | _TWO
    assert sum(v for k, v in counts.items() if k in _SINGLE) == n * depth
    assert circ.num_qubits == n
    assert circ.name == f"random_{n}q_depth{depth}"
    assert not any(i.is_measure or i.is_barrier for i in circ.instructions)
    # The generated circuit is valid input for qiskit and matches it.
    sv = quspect.simulation.Simulator().statevector(circ)
    assert_states_equal(sv, Statevector(to_qiskit(circ)), up_to_phase=False)


def test_generate_random_circuit_determinism_and_options() -> None:
    """Seed reproducibility, name, two-qubit and barrier probabilities."""
    a = generate_random_circuit(4, 6, seed=3)
    b = generate_random_circuit(4, 6, seed=3)
    assert [repr(i) for i in a.instructions] == [repr(i) for i in b.instructions]
    c = generate_random_circuit(4, 6, seed=4)
    assert [repr(i) for i in a.instructions] != [repr(i) for i in c.instructions]
    none2q = generate_random_circuit(4, 6, seed=3, two_qubit_probability=0.0)
    assert not set(quspect_gate_counts(none2q)) & _TWO
    all2q = generate_random_circuit(4, 6, seed=3, two_qubit_probability=1.0)
    assert sum(v for k, v in quspect_gate_counts(all2q).items() if k in _TWO) == 12
    barriers = generate_random_circuit(3, 4, seed=1, barrier_probability=1.0, name="x")
    assert barriers.name == "x"
    assert sum(i.is_barrier for i in barriers.instructions) == 4
    unseeded = generate_random_circuit(2, 2)
    assert unseeded.num_qubits == 2


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"num_qubits": 0}, ValueError),
        ({"num_qubits": True}, TypeError),
        ({"num_qubits": 2.0}, TypeError),
        ({"depth": 0}, ValueError),
        ({"depth": "3"}, TypeError),
        ({"two_qubit_probability": 1.5}, ValueError),
        ({"two_qubit_probability": -0.1}, ValueError),
        ({"two_qubit_probability": math.nan}, ValueError),
        ({"two_qubit_probability": True}, TypeError),
        ({"barrier_probability": "0.5"}, TypeError),
        ({"barrier_probability": math.inf}, ValueError),
    ],
)
def test_generate_random_circuit_validation(
    kwargs: dict[str, Any], exc: type[Exception]
) -> None:
    """Invalid sizes / probabilities raise TypeError or ValueError."""
    args: dict[str, Any] = {"num_qubits": 2, "depth": 2, **kwargs}
    with pytest.raises(exc):
        generate_random_circuit(**args)


# =============================================================================
# check_compilation_consistency
# =============================================================================


@pytest.mark.parametrize("seed", range(4))
def test_consistency_default_compiler(seed: int) -> None:
    """Default OptimizePass: both checks pass; fields cross-checked with qiskit."""
    circ = generate_random_circuit(4, 6, seed=seed, barrier_probability=0.3)
    res = check_compilation_consistency(circ)
    assert isinstance(res, CompilationConsistencyResult)
    assert (
        res.consistent and res.quspect_qiskit_consistent and res.compilation_consistent
    )
    assert res.quspect_qiskit_fidelity == pytest.approx(1.0, abs=1e-9)
    assert res.compilation_fidelity == pytest.approx(1.0, abs=1e-9)
    ref = Statevector(to_qiskit(circ)).data
    assert_states_equal(res.qiskit_statevector, ref)
    assert_states_equal(res.quspect_statevector, ref)
    assert_states_equal(res.compiled_statevector, ref)
    assert res.original_qasm.startswith("OPENQASM 2.0;")
    assert res.original_circuit.num_qubits == 4


def test_consistency_detects_broken_compiler() -> None:
    """A compiler that drops a gate is flagged; consistent is False."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).ry(0.7, 1)

    def drop_last(c: quspect.QuantumCircuit) -> quspect.QuantumCircuit:
        out = quspect.QuantumCircuit()
        for instr in c.instructions[:-1]:
            assert instr.gate is not None
            out.append(instr.gate, instr.qubits)
        return out

    res = check_compilation_consistency(circ, compiler=drop_last)
    assert res.quspect_qiskit_consistent and not res.compilation_consistent
    assert not res.consistent
    assert res.compilation_fidelity < 0.99


def test_consistency_atol_controls_verdict() -> None:
    """A tiny rotation error passes with a loose atol, fails with a tight one."""
    circ = quspect.QuantumCircuit().ry(0.5, 0).cx(0, 1)

    def perturb(c: quspect.QuantumCircuit) -> quspect.QuantumCircuit:
        return c.copy().ry(1e-3, 0)

    assert not check_compilation_consistency(circ, compiler=perturb).consistent
    assert check_compilation_consistency(circ, compiler=perturb, atol=1e-3).consistent
    assert check_compilation_consistency(
        circ, compiler=perturb, atol=0
    ).quspect_qiskit_consistent


def test_consistency_with_sabre_compiler_needs_permutation() -> None:
    """Routing permutes qubits, so the plain state check can fail (expected)."""
    target = target_from_edges(3, line_edges(3))
    circ = quspect.QuantumCircuit().x(0).cx(0, 1).cx(0, 2).cx(1, 2)

    def route(c: quspect.QuantumCircuit) -> quspect.QuantumCircuit:
        return SABRE(target).solve(c, initial_mapping=[0, 1, 2])[0]

    res = check_compilation_consistency(circ, compiler=route)
    routed, initial, final = SABRE(target).solve(circ, initial_mapping=[0, 1, 2])
    sigma = routed_permutation(initial, final)
    permuted = permutation_matrix(sigma) @ res.quspect_statevector
    assert_states_equal(res.compiled_statevector, permuted)
    assert res.compilation_consistent == bool(sigma == sorted(sigma))


def test_consistency_measurement_only_circuit_is_padded() -> None:
    """A measurement-only circuit becomes an identity on the same width."""
    res = check_compilation_consistency(quspect.QuantumCircuit().measure([1], [0]))
    assert res.consistent and res.quspect_statevector.shape == (4,)


def test_consistency_strips_terminal_measurements_and_keeps_width() -> None:
    """Terminal measurements / trailing barriers are dropped; width kept."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).measure([0, 1, 2], [0, 1, 2])
    circ.barrier()
    res = check_compilation_consistency(circ)
    assert res.consistent
    assert res.original_circuit.num_qubits == 3
    assert not any(i.is_measure for i in res.original_circuit.instructions)
    assert res.quspect_statevector.shape == (8,)


def test_consistency_width_mismatch_gives_zero_fidelity() -> None:
    """A compiler that shrinks the register yields fidelity 0."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1)
    res = check_compilation_consistency(
        circ, compiler=lambda c: quspect.QuantumCircuit().h(0)
    )
    assert res.compilation_fidelity == 0.0 and not res.compilation_consistent


def test_state_fidelity_edge_cases() -> None:
    """_state_fidelity: zero norm -> 0, unnormalised inputs are normalised."""
    zero = np.zeros(2, dtype=complex)
    one = np.array([1, 0], dtype=complex)
    assert _state_fidelity(zero, one) == 0.0
    assert _state_fidelity(2 * one, one) == pytest.approx(1.0)
    assert _state_fidelity(one, np.ones(4, dtype=complex)) == 0.0


@pytest.mark.parametrize(
    ("circ_factory", "kwargs", "exc", "match"),
    [
        (lambda: "not a circuit", {}, TypeError, "QuantumCircuit"),
        (lambda: quspect.QuantumCircuit().h(0), {"atol": True}, TypeError, "atol"),
        (lambda: quspect.QuantumCircuit().h(0), {"atol": "1"}, TypeError, "atol"),
        (lambda: quspect.QuantumCircuit().h(0), {"atol": -1.0}, ValueError, "atol"),
        (lambda: quspect.QuantumCircuit().h(0), {"atol": math.nan}, ValueError, "atol"),
        (
            lambda: quspect.QuantumCircuit().rx(Parameter("a"), 0),
            {},
            ValueError,
            "unbound",
        ),
        (
            lambda: quspect.QuantumCircuit().h(0).measure([0], [0]).x(0),
            {},
            ValueError,
            "mid-circuit",
        ),
        (
            lambda: quspect.QuantumCircuit().h(0).depolarize(0.1, 0),
            {},
            ValueError,
            "noise",
        ),
        (
            lambda: quspect.QuantumCircuit(),
            {},
            ValueError,
            "at least one",
        ),
        (
            lambda: quspect.QuantumCircuit().h(0),
            {"compiler": 42},
            TypeError,
            "callable",
        ),
        (
            lambda: quspect.QuantumCircuit().h(0),
            {"compiler": lambda c: "nope"},
            TypeError,
            "must return QuantumCircuit",
        ),
    ],
)
def test_consistency_input_validation(
    circ_factory: Any, kwargs: dict[str, Any], exc: type[Exception], match: str
) -> None:
    """Every documented error branch."""
    with pytest.raises(exc, match=match):
        check_compilation_consistency(circ_factory(), **kwargs)


def test_consistency_requires_qiskit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without qiskit importable an ImportError with install hint is raised."""
    monkeypatch.setitem(sys.modules, "qiskit", None)
    with pytest.raises(ImportError, match="Qiskit is required"):
        check_compilation_consistency(quspect.QuantumCircuit().h(0))
