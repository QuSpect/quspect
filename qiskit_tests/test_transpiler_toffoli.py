"""Cross-validate Toffoli-aware routing and HighLevelOptimizePass with qiskit.

* Every built-in Toffoli synthesis equals qiskit's ``ccx`` exactly (no
  global phase, element-wise ``atol = 1e-9``).
* SABRE routes ``ccx`` as a three-qubit unit: the routed circuit is on the
  coupling graph and equals ``P_sigma @ U`` (unitary for at most 5 wires,
  random-input statevectors otherwise) on line, ring, grid and heavy-hex
  graphs.
* The Cuccaro adder pipeline (``HighLevelOptimizePass`` → ``DecomposePass``
  keeping ``ccx`` → SABRE → lowering + optimisation) stays equivalent and
  beats the decompose-before-routing pipeline on triangle-free graphs.
"""

from __future__ import annotations

import itertools
import random

import pytest

import quspect
from qiskit_tests._dag_helpers import (
    grid_edges,
    heavy_hex_edges,
    line_edges,
    ring_edges,
)
from qiskit_tests.helpers import assert_unitary_equal, qiskit_circuit_unitary
from qiskit_tests.test_transpiler_sabre import (
    _assert_on_coupling_edges,
    _assert_routed_equivalent,
)
from quspect.transpiler import (
    SABRE,
    CompilationTarget,
    ConsolidateBlocksPass,
    HighLevelOptimizePass,
)
from quspect.transpiler._passes import DecomposePass, OptimizePass
from quspect.transpiler._toffoli import ccx_candidates, ccx_path_decomposition

pytestmark = pytest.mark.qiskit_ref

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}

_HH_N, _HH_EDGES = heavy_hex_edges()
TOPOLOGIES: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "line5": (5, line_edges(5)),
    "ring6": (6, ring_edges(6)),
    "grid3x3": (9, grid_edges(3, 3)),
    "heavyhex13": (_HH_N, _HH_EDGES),
    "full5": (5, list(itertools.combinations(range(5), 2))),
}


def _circuit(ops: list[tuple[quspect.Gate, tuple[int, ...]]]) -> quspect.QuantumCircuit:
    qc = quspect.QuantumCircuit()
    for gate, qs in ops:
        qc.append(gate, qs)
    return qc


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_path_templates_match_qiskit_ccx(perm: tuple[int, ...]) -> None:
    c1, c2, t = perm
    ref = qiskit_circuit_unitary(quspect.QuantumCircuit().ccx(c1, c2, t))
    for middle in perm:
        ops = ccx_path_decomposition(c1, c2, t, middle)
        got = qiskit_circuit_unitary(_circuit(ops), num_qubits=3)
        assert_unitary_equal(got, ref)


@pytest.mark.parametrize(
    "edges", [[(0, 1), (1, 2), (0, 2)], [(0, 1), (1, 2)], [(0, 2), (2, 1)]]
)
def test_candidates_match_qiskit_ccx(edges: list[tuple[int, int]]) -> None:
    edge_set = {frozenset(e) for e in edges}
    ref = qiskit_circuit_unitary(quspect.QuantumCircuit().ccx(0, 1, 2))
    for ops in ccx_candidates(0, 1, 2, lambda x, y: frozenset((x, y)) in edge_set):
        got = qiskit_circuit_unitary(_circuit(ops), num_qubits=3)
        assert_unitary_equal(got, ref, up_to_phase=True)


def _random_ccx_circuit(n: int, depth: int, seed: int) -> quspect.QuantumCircuit:
    rng = random.Random(seed)
    qc = quspect.QuantumCircuit()
    for _ in range(depth):
        qs = rng.sample(range(n), 3)
        r = rng.random()
        if r < 0.45:
            qc.ccx(*qs)
        elif r < 0.7:
            qc.cx(qs[0], qs[1])
        else:
            qc.h(qs[0]).t(qs[1]).ry(rng.uniform(-3, 3), qs[2])
    return qc


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("topo", list(TOPOLOGIES))
def test_sabre_ccx_routing_equivalent(topo: str, seed: int) -> None:
    n_phys, edges = TOPOLOGIES[topo]
    target = CompilationTarget.from_coupling_map(n_phys, edges)
    circ = _random_ccx_circuit(min(n_phys, 7), 18, seed)
    routed, initial, final = SABRE(target).solve(circ, seed=seed)
    assert all(len(i.qubits) <= 2 for i in routed)
    _assert_on_coupling_edges(routed, initial, target)
    _assert_routed_equivalent(circ, routed, initial, final, seed=seed)


def _cuccaro(n: int) -> quspect.QuantumCircuit:
    k = (n - 2) // 2
    a = [2 * i + 1 for i in range(k)]
    b = [2 * i + 2 for i in range(k)]
    cin, cout = 0, 2 * k + 1
    qc = quspect.QuantumCircuit()
    for q in a[1::2] + b[::2]:
        qc.x(q)

    def maj(x: int, y: int, z: int) -> None:
        qc.cx(z, y).cx(z, x).ccx(x, y, z)

    def uma(x: int, y: int, z: int) -> None:
        qc.ccx(x, y, z).cx(z, x).cx(x, y)

    maj(cin, b[0], a[0])
    for i in range(1, k):
        maj(a[i - 1], b[i], a[i])
    qc.cx(a[k - 1], cout)
    for i in range(k - 1, 0, -1):
        uma(a[i - 1], b[i], a[i])
    uma(cin, b[0], a[0])
    return qc


def _compile(
    circ: quspect.QuantumCircuit,
    target: CompilationTarget,
    seed: int,
    *,
    toffoli_routing: bool,
) -> tuple[quspect.QuantumCircuit, list[int], list[int]]:
    decompose = DecomposePass(target)
    consolidate = ConsolidateBlocksPass(target)
    if toffoli_routing:
        c = DecomposePass(target, keep={"ccx"}).run(HighLevelOptimizePass().run(circ))
    else:
        c = decompose.run(circ)
    routed, initial, final = SABRE(target).solve(consolidate.run(c), seed=seed)
    c = consolidate.run(decompose.run(routed))
    c = decompose.run(OptimizePass(basis_gates=BASIS).run(c))
    return decompose.run(consolidate.run(c)), initial, final


def _n2q(circ: quspect.QuantumCircuit) -> int:
    return sum(1 for i in circ if len(i.qubits) == 2)


@pytest.mark.parametrize("topo", ["grid3x3", "heavyhex13", "full5"])
def test_cuccaro_pipeline_equivalent_and_cheaper(topo: str) -> None:
    n_phys, edges = TOPOLOGIES[topo]
    target = CompilationTarget.from_coupling_map(n_phys, edges, native_gates=BASIS)
    circ = _cuccaro(min(n_phys, 8) // 2 * 2)
    new, initial, final = _compile(circ, target, seed=0, toffoli_routing=True)
    old, _, _ = _compile(circ, target, seed=0, toffoli_routing=False)
    assert {i.gate.name for i in new if i.gate is not None} <= BASIS
    _assert_on_coupling_edges(new, initial, target)
    _assert_routed_equivalent(circ, new, initial, final, up_to_phase=True)
    assert _n2q(new) < _n2q(old)


@pytest.mark.parametrize("seed", range(10))
def test_highlevel_pass_matches_qiskit_unitary(seed: int) -> None:
    rng = random.Random(seed)
    circ = quspect.QuantumCircuit()
    for _ in range(40):
        qs = rng.sample(range(4), 3)
        kind = rng.choice(["ccx", "ccx", "ccx", "cx", "cz", "swap", "t", "x", "rz"])
        if kind == "ccx":
            circ.ccx(*qs)
        elif kind in ("cx", "cz", "swap"):
            getattr(circ, kind)(qs[0], qs[1])
        elif kind == "rz":
            circ.rz(rng.uniform(-3, 3), qs[0])
        else:
            getattr(circ, kind)(qs[0])
    out = HighLevelOptimizePass().run(circ)
    assert_unitary_equal(
        qiskit_circuit_unitary(out, num_qubits=4),
        qiskit_circuit_unitary(circ, num_qubits=4),
    )


def test_highlevel_pass_ccx_pair_compiles_to_nothing() -> None:
    target = CompilationTarget.from_coupling_map(3, line_edges(3), native_gates=BASIS)
    circ = quspect.QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2)
    out = DecomposePass(target).run(HighLevelOptimizePass().run(circ))
    assert len(out) == 0
