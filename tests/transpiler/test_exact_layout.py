"""Tests for VF2++ exact layout and permutation elision in SABRE."""

from __future__ import annotations

import random
import time

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget
from quspect.transpiler._mapping import _elide_trailing_swaps, _MappingGate
from quspect.transpiler._vf2 import find_subgraph_monomorphism

# ---------------------------------------------------------------------------
# Coupling graphs
# ---------------------------------------------------------------------------


def _line(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _ring(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def _grid(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


def _heavy_hex(rows: int, width: int) -> tuple[int, list[tuple[int, int]]]:
    """Heavy-hex style lattice: chains joined by degree-2 bridge qubits."""
    edges = []
    for r in range(rows):
        edges += [(r * width + c, r * width + c + 1) for c in range(width - 1)]
    n = rows * width
    for r in range(rows - 1):
        for c in range(0 if r % 2 == 0 else 2, width, 4):
            edges += [(r * width + c, n), (n, (r + 1) * width + c)]
            n += 1
    return n, edges


def _adjacency(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


_HH_N, _HH_EDGES = _heavy_hex(4, 9)
TOPOLOGIES: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "line": (12, _line(12)),
    "ring": (12, _ring(12)),
    "grid": (12, _grid(3, 4)),
    "heavy_hex": (_HH_N, _HH_EDGES),
}


def _assert_monomorphism(
    mapping: list[int], edges: list[tuple[int, int]], adj: list[list[int]]
) -> None:
    images = [p for p in mapping if p >= 0]
    assert len(set(images)) == len(images)
    for a, b in edges:
        assert mapping[b] in adj[mapping[a]]


def _swaps(circuit: QuantumCircuit) -> int:
    return sum(
        1 for i in circuit if i.is_gate and i.gate is not None and i.gate.name == "swap"
    )


def _assert_on_edges(
    routed: QuantumCircuit, initial: list[int], target: CompilationTarget
) -> None:
    for instr in routed:
        if instr.is_gate and len(instr.qubits) == 2:
            a, b = (initial[w] for w in instr.qubits)
            assert b in target.adjacency[a]


# ---------------------------------------------------------------------------
# find_subgraph_monomorphism
# ---------------------------------------------------------------------------


class TestMonomorphism:
    @pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
    def test_chain_embeds(self, topo: str) -> None:
        n, edges = TOPOLOGIES[topo]
        adj = _adjacency(n, edges)
        chain = _line(10)
        mapping = find_subgraph_monomorphism(10, chain, adj)
        assert mapping is not None
        _assert_monomorphism(mapping, chain, adj)

    def test_monomorphism_not_induced(self) -> None:
        # A path on three nodes embeds into a triangle (an induced
        # subgraph test would reject it because of the extra edge).
        triangle = _adjacency(3, [(0, 1), (1, 2), (0, 2)])
        mapping = find_subgraph_monomorphism(3, [(0, 1), (1, 2)], triangle)
        assert mapping is not None
        _assert_monomorphism(mapping, [(0, 1), (1, 2)], triangle)

    def test_isolated_pattern_nodes_unmapped(self) -> None:
        mapping = find_subgraph_monomorphism(4, [(1, 2)], _adjacency(3, _line(3)))
        assert mapping is not None
        assert mapping[0] == mapping[3] == -1
        assert {mapping[1], mapping[2]} in ({0, 1}, {1, 2})

    def test_no_edges(self) -> None:
        assert find_subgraph_monomorphism(2, [], _adjacency(2, _line(2))) == [-1, -1]

    @pytest.mark.parametrize("seed", range(5))
    def test_random_relabelled_subgraph_found(self, seed: int) -> None:
        rng = random.Random(seed)
        for n, edges in (TOPOLOGIES["grid"], TOPOLOGIES["heavy_hex"]):
            adj = _adjacency(n, edges)
            # Random connected subgraph grown by BFS, randomly relabelled.
            start = rng.randrange(n)
            nodes = [start]
            seen = {start}
            while len(nodes) < min(9, n):
                frontier = [z for u in nodes for z in adj[u] if z not in seen]
                pick = rng.choice(frontier)
                seen.add(pick)
                nodes.append(pick)
            label = {p: i for i, p in enumerate(rng.sample(nodes, len(nodes)))}
            sub = [
                (label[a], label[b])
                for a, b in edges
                if a in seen and b in seen and rng.random() < 0.8
            ]
            mapping = find_subgraph_monomorphism(len(nodes), sub, adj)
            assert mapping is not None
            _assert_monomorphism(mapping, sub, adj)

    def test_quick_rejections(self) -> None:
        line = _adjacency(6, _line(6))
        star = [(0, i) for i in range(1, 4)]  # degree 3 > line degree 2
        assert find_subgraph_monomorphism(4, star, line) is None
        assert find_subgraph_monomorphism(7, _line(7), line) is None  # too large
        grid = _adjacency(9, _grid(3, 3))
        petersen = [(i, (i + 1) % 5) for i in range(5)]
        petersen += [(i, i + 5) for i in range(5)]
        petersen += [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
        start = time.perf_counter()
        assert (
            find_subgraph_monomorphism(10, petersen, _adjacency(12, _grid(3, 4)))
            is None
        )
        assert find_subgraph_monomorphism(3, [(0, 1), (1, 2), (0, 2)], grid) is None
        assert time.perf_counter() - start < 0.1

    def test_exhaustive_negative(self) -> None:
        # Odd ring into a non-bipartite but too sparse target.
        target = _adjacency(7, [*_ring(5), (0, 5), (5, 6)])
        assert find_subgraph_monomorphism(7, _ring(7), target) is None

    def test_state_budget_respected(self) -> None:
        # A long chain in a large lattice that needs many states.
        n, edges = _heavy_hex(6, 13)
        adj = _adjacency(n, edges)
        start = time.perf_counter()
        assert find_subgraph_monomorphism(n, _line(n), adj, max_states=500) is None
        assert time.perf_counter() - start < 1.0

    def test_time_limit_respected(self) -> None:
        n, edges = _heavy_hex(6, 13)
        adj = _adjacency(n, edges)
        start = time.perf_counter()
        result = find_subgraph_monomorphism(
            n, _ring(n - 1), adj, max_states=10**9, time_limit=0.05
        )
        assert result is None
        assert time.perf_counter() - start < 1.0

    def test_deterministic(self) -> None:
        n, edges = TOPOLOGIES["heavy_hex"]
        adj = _adjacency(n, edges)
        first = find_subgraph_monomorphism(20, _line(20), adj)
        assert first is not None
        assert find_subgraph_monomorphism(20, _line(20), adj) == first

    def test_invalid_arguments(self) -> None:
        adj = _adjacency(3, _line(3))
        with pytest.raises(ValueError, match="self loop"):
            find_subgraph_monomorphism(2, [(1, 1)], adj)
        with pytest.raises(ValueError, match="out of range"):
            find_subgraph_monomorphism(2, [(0, 2)], adj)
        with pytest.raises(ValueError, match="max_states"):
            find_subgraph_monomorphism(2, [(0, 1)], adj, max_states=0)


# ---------------------------------------------------------------------------
# SABRE exact layout
# ---------------------------------------------------------------------------


def _ghz(n: int, perm: list[int]) -> QuantumCircuit:
    circ = QuantumCircuit().h(perm[0])
    for i in range(n - 1):
        circ.cx(perm[i], perm[i + 1])
    return circ


def _heisenberg(n: int, perm: list[int]) -> QuantumCircuit:
    circ = QuantumCircuit()
    for q in range(n):
        circ.x(q)
    for _ in range(2):
        for i in range(n - 1):
            a, b = perm[i], perm[i + 1]
            circ.cx(a, b).rz(0.3, b).cx(a, b).h(a).h(b).cz(a, b).h(a).h(b)
    return circ


def _star(leaves: int) -> QuantumCircuit:
    circ = QuantumCircuit().h(0)
    for q in range(1, leaves + 1):
        circ.cx(q, 0)
    return circ


class TestExactLayout:
    @pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
    @pytest.mark.parametrize("kind", ["ghz", "heisenberg", "star"])
    def test_embeddable_circuits_need_no_swap(self, topo: str, kind: str) -> None:
        n, edges = TOPOLOGIES[topo]
        target = CompilationTarget.from_coupling_map(n, edges)
        perm = random.Random(7).sample(range(10), 10)  # trivial layout fails
        circ = {
            "ghz": lambda: _ghz(10, perm),
            "heisenberg": lambda: _heisenberg(10, perm),
            "star": lambda: _star(2 if topo in ("line", "ring") else 3),
        }[kind]()
        routed, initial, final = SABRE(target).solve(circ, seed=3, layout_trials=1)
        assert _swaps(routed) == 0
        assert final == initial
        _assert_on_edges(routed, initial, target)

    def test_exact_layout_beats_disabled_search(self) -> None:
        n, edges = TOPOLOGIES["heavy_hex"]
        target = CompilationTarget.from_coupling_map(n, edges)
        circ = _ghz(20, random.Random(1).sample(range(20), 20))
        routed, _, _ = SABRE(target).solve(circ, seed=0, layout_trials=1)
        baseline, _, _ = SABRE(target).solve(
            circ, seed=0, layout_trials=1, exact_layout=False
        )
        assert _swaps(routed) == 0 < _swaps(baseline)

    def test_non_embeddable_falls_back(self) -> None:
        target = CompilationTarget.from_coupling_map(5, _line(5))
        circ = QuantumCircuit()
        for a in range(4):
            for b in range(a + 1, 4):
                circ.cz(a, b)
        routed, initial, _ = SABRE(target).solve(circ, seed=0)
        assert _swaps(routed) > 0
        _assert_on_edges(routed, initial, target)

    def test_deterministic(self) -> None:
        n, edges = TOPOLOGIES["grid"]
        target = CompilationTarget.from_coupling_map(n, edges)
        circ = _heisenberg(10, random.Random(5).sample(range(10), 10))
        first = SABRE(target).solve(circ, seed=11)
        second = SABRE(target).solve(circ, seed=11)
        assert first[1:] == second[1:]
        assert first[0].instructions == second[0].instructions

    def test_trivial_layout_kept_when_exact(self) -> None:
        target = CompilationTarget.from_coupling_map(4, _line(4))
        _, initial, _ = SABRE(target).solve(_ghz(4, [0, 1, 2, 3]), seed=0)
        assert initial == [0, 1, 2, 3]

    def test_initial_mapping_disables_exact_layout(self) -> None:
        target = CompilationTarget.from_coupling_map(3, _line(3))
        circ = QuantumCircuit().cx(0, 1).cx(0, 2)
        routed, initial, _ = SABRE(target).solve(circ, initial_mapping=[0, 1, 2])
        assert initial[:3] == [0, 1, 2]
        assert _swaps(routed) == 1

    def test_invalid_budget(self) -> None:
        target = CompilationTarget.from_coupling_map(2, _line(2))
        with pytest.raises(ValueError, match="vf2_max_states"):
            SABRE(target).solve(QuantumCircuit().cx(0, 1), vf2_max_states=0)


# ---------------------------------------------------------------------------
# Permutation elision (structure; equivalence is checked in qiskit_tests)
# ---------------------------------------------------------------------------


class TestElidePermutations:
    def test_input_swap_becomes_final_permutation(self) -> None:
        target = CompilationTarget.from_coupling_map(3, _line(3))
        circ = QuantumCircuit().h(0).cx(0, 1).swap(0, 1).x(0)
        routed, initial, final = SABRE(target).solve(circ, seed=0)
        assert [i.gate.name for i in routed if i.gate is not None] == ["h", "cx", "x"]
        # x acted on the old wire of logical 1, where logical 0's state is.
        assert routed.instructions[-1].qubits == (1,)
        assert final[:2] == [initial[1], initial[0]]

    def test_disabled_keeps_swap(self) -> None:
        target = CompilationTarget.from_coupling_map(3, _line(3))
        circ = QuantumCircuit().h(0).swap(0, 1)
        routed, initial, final = SABRE(target).solve(
            circ, seed=0, elide_permutations=False
        )
        assert _swaps(routed) == 1
        assert final == initial  # the input SWAP is an ordinary gate here

    def test_decomposed_swap_elided_with_native_residual(self) -> None:
        natives = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}
        target = CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates=natives)
        circ = QuantumCircuit().t(0)
        circ.h(1).cz(0, 1).h(1).h(0).cz(0, 1).h(0).h(1).cz(0, 1).h(1)
        routed, initial, final = SABRE(target).solve(circ, seed=0)
        assert _swaps(routed) == 0
        assert not any(len(i.qubits) == 2 for i in routed)
        assert {i.gate.name for i in routed if i.gate is not None} <= natives
        assert final[:2] == [initial[1], initial[0]]

    def test_measurement_redirected(self) -> None:
        target = CompilationTarget.from_coupling_map(2, [(0, 1)])
        circ = QuantumCircuit().x(0).swap(0, 1).measure([0, 1], [0, 1])
        routed, _, _ = SABRE(target).solve(circ, seed=0)
        measures = [(i.qubits, i.clbits) for i in routed if i.is_measure]
        assert sorted(measures, key=lambda m: m[1]) == [((1,), (0,)), ((0,), (1,))]

    def test_trailing_routing_swap_dropped(self) -> None:
        # cx(0,1); SWAP(1,2); measure(2); SWAP(0,3); h(3) on physical qubits.
        gates = [
            _MappingGate(True, 0, 1, 0, "cx"),
            _MappingGate(True, 1, 2, -1, "swap"),
            _MappingGate(False, 2, 2, 1, "measure", clbit=0),
            _MappingGate(True, 0, 3, -1, "swap"),
            _MappingGate(False, 3, 3, 2, "h"),
        ]
        out, initial, final = _elide_trailing_swaps(gates, [0, 1, 2], 3)
        assert [(g.name, g.q1, g.q2) for g in out] == [
            ("cx", 0, 1),
            ("measure", 1, 1),  # redirected: the state never left qubit 1
            ("swap", 0, 3),  # followed by h(3): kept
            ("h", 3, 3),
        ]
        assert initial == [0, 1, 2, 3]
        assert final == [3, 1, 2, 0]

    def test_trailing_swap_chain_on_workspace(self) -> None:
        gates = [
            _MappingGate(True, 0, 1, -1, "swap"),
            _MappingGate(True, 1, 2, -1, "swap"),
        ]
        out, initial, final = _elide_trailing_swaps(gates, [0], 1)
        assert out == [] and initial == [0] and final == [0]
