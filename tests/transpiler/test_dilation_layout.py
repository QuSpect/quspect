"""Low-dilation initial layouts (``quspect.transpiler._embedding``) and their use."""

from __future__ import annotations

import math
import random

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget, transpile
from quspect.transpiler import _embedding as emb
from quspect.transpiler._mapping import _bfs_distances

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}

# IBM heavy-hex lattice of distance 5 (57 qubits; its longest simple path has 49
# vertices, so a 50-qubit chain cannot be embedded without dilation).
_HEAVY_HEX_D5 = [
    (0, 25), (0, 37), (1, 38), (2, 39), (3, 40), (5, 25), (5, 41), (6, 42),
    (7, 43), (8, 44), (9, 30), (10, 31), (10, 45), (11, 46), (12, 47), (13, 48),
    (14, 30), (15, 31), (15, 49), (16, 50), (17, 51), (18, 52), (19, 36),
    (20, 53), (21, 54), (22, 55), (23, 56), (24, 36), (1, 37), (2, 38),
    (26, 38), (3, 39), (4, 40), (27, 40), (6, 41), (28, 41), (7, 42), (26, 42),
    (8, 43), (29, 43), (9, 44), (27, 44), (11, 45), (28, 45), (12, 46),
    (32, 46), (13, 47), (29, 47), (14, 48), (33, 48), (16, 49), (34, 49),
    (17, 50), (32, 50), (18, 51), (35, 51), (19, 52), (33, 52), (21, 53),
    (34, 53), (22, 54), (23, 55), (35, 55), (24, 56),
]  # fmt: skip


def _graph(
    n: int, edges: list[tuple[int, int]]
) -> tuple[list[list[int]], list[list[int]]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    adj = [sorted(r) for r in adj]
    return adj, _bfs_distances(adj, n)


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                out.append((q, q + 1))
            if r + 1 < rows:
                out.append((q, q + cols))
    return out


def _dilation(
    edges: list[tuple[int, int, int]], dist: list[list[int]], pos: list[int]
) -> int:
    return sum(w * (dist[pos[a]][pos[b]] - 1) for a, b, w in edges)


def test_chain_embeds_in_grid_without_dilation() -> None:
    adj, dist = _graph(20, _grid_edges(4, 5))
    chain = [(i, i + 1, 2) for i in range(19)]
    cost, pos = emb.dilation_layouts(20, chain, adj, dist, seed=3)[0]
    assert cost == 0 == _dilation(chain, dist, pos)
    assert sorted(pos) == list(range(20))


def test_chain_on_heavy_hex_needs_one_unit() -> None:
    adj, dist = _graph(57, _HEAVY_HEX_D5)
    chain = [(i, i + 1, 1) for i in range(49)]
    cost, pos = emb.dilation_layouts(50, chain, adj, dist, seed=0)[0]
    assert cost == 1 == _dilation(chain, dist, pos)


def test_idle_qubits_unplaced_and_results_sorted() -> None:
    adj, dist = _graph(9, _grid_edges(3, 3))
    edges = [
        (0, 1, 2),
        (0, 2, 1),
        (0, 3, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 4),
        (3, 5, 1),
    ]
    found = emb.dilation_layouts(6, edges, adj, dist, seed=1, count=3)
    assert found
    for i, (cost, pos) in enumerate(found):
        assert pos[4] == -1
        assert cost == _dilation(edges, dist, pos)
        if i:
            assert found[i - 1][0] <= cost and found[i - 1][1] != pos
    assert emb.dilation_layouts(3, [], adj, dist, seed=0) == []


def test_native_matches_python_reference() -> None:
    from quspect._native_loader import load_core

    core = load_core()
    if core is None or not hasattr(core, "_dilation_layouts"):
        pytest.skip("native extension not built")
    rng = random.Random(7)
    graphs = [_graph(57, _HEAVY_HEX_D5), _graph(20, _grid_edges(4, 5))]
    for _ in range(25):
        adj, dist = rng.choice(graphs)
        n = rng.randint(2, len(adj))
        weights: dict[tuple[int, int], int] = {}
        for _ in range(rng.randint(1, 3 * n)):
            a, b = sorted(rng.sample(range(n), 2))
            weights[a, b] = weights.get((a, b), 0) + rng.randint(1, 4)
        edges = [(a, b, w) for (a, b), w in sorted(weights.items())]
        seed = rng.getrandbits(64)
        ref = emb.dilation_layouts_python(n, edges, adj, dist, seed, 3)
        got = [
            (c, list(p))
            for c, p in core._dilation_layouts(n, edges, adj, dist, seed, 3)
        ]
        assert got == ref


def test_python_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    adj, dist = _graph(20, _grid_edges(4, 5))
    chain = [(i, i + 1, 1) for i in range(12)]
    native = emb.dilation_layouts(13, chain, adj, dist, seed=5, count=2)
    monkeypatch.setenv("QUSPECT_ROUTING_BACKEND", "python")
    assert emb._native_core() is None
    assert emb.dilation_layouts(13, chain, adj, dist, seed=5, count=2) == native


def _heavy_hex_target() -> CompilationTarget:
    return CompilationTarget.from_coupling_map(57, _HEAVY_HEX_D5, native_gates=BASIS)


def test_sabre_dilation_starts_route_chain_with_one_swap() -> None:
    ghz = QuantumCircuit().h(0)
    for i in range(49):
        ghz.cx(i, i + 1)
    target = _heavy_hex_target()
    plain, _, _ = SABRE(target).solve(ghz, seed=0)
    routed, _, _ = SABRE(target, dilation_layouts=1).solve(ghz, seed=0)

    def swaps(c: QuantumCircuit) -> int:
        return sum(
            1 for i in c if i.is_gate and i.gate is not None and i.gate.name == "swap"
        )

    assert swaps(routed) == 1 < swaps(plain)
    with pytest.raises(ValueError, match="dilation_layouts"):
        SABRE(target, dilation_layouts=-1)


@pytest.mark.parametrize("level", [2, 3])
def test_transpile_chain_circuits_on_heavy_hex(level: int) -> None:
    target = _heavy_hex_target()
    ghz = QuantumCircuit().h(0)
    for i in range(49):
        ghz.cx(i, i + 1)
    res = transpile(ghz, target, optimization_level=level, seed=0)
    assert res.stats.two_qubit_gates <= 51  # 49 + one SWAP merged with a CX
    # 一维 Heisenberg 链（XX + YY + ZZ，2 个 Trotter 步）：每步每对一个 3-CZ 块
    heis = QuantumCircuit()
    for _ in range(2):
        for i in range(49):
            j = i + 1
            heis.h(i).h(j).cx(i, j).rz(0.2, j).cx(i, j).h(i).h(j)
            heis.rx(math.pi / 2, i).rx(math.pi / 2, j).cx(i, j).rz(0.2, j).cx(i, j)
            heis.rx(-math.pi / 2, i).rx(-math.pi / 2, j).cx(i, j).rz(0.2, j).cx(i, j)
    res = transpile(heis, target, optimization_level=level, seed=0)
    assert res.stats.two_qubit_gates <= 2 * 49 * 3 + 9


def test_level2_exact_routing_choice_is_equivalent() -> None:
    """Level 2 may route the unapproximated circuit; results stay correct."""
    import numpy as np

    from tests.transpiler.test_transpile import _assert_equivalent, _embed, _simulate

    n = 7
    circ = QuantumCircuit()
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            circ.cp(math.pi / 2 ** (3 * (j - i) + 4), j, i)
    edges = [(i, i + 1) for i in range(n - 1)]
    target = CompilationTarget.from_coupling_map(n, edges, native_gates=BASIS)
    level2 = transpile(circ, target, optimization_level=2, seed=0)
    assert level2.stats.selected in ("default", "exact-routing")
    assert level2.stats.approximation_error < 1e-4
    psi = np.random.default_rng(0).normal(size=2**n) + 0j
    psi /= np.linalg.norm(psi)
    expected = _embed(_simulate(circ, psi, n), level2.final_layout, n)
    got = _simulate(level2.circuit, _embed(psi, level2.initial_layout, n), n)
    assert abs(np.vdot(expected, got)) > 1 - 1e-9
    level3 = transpile(circ, target, optimization_level=3, seed=0)
    assert level3.stats.two_qubit_gates <= level2.stats.two_qubit_gates
    exact = transpile(
        circ, target, optimization_level=2, seed=0, approximation_degree=None
    )
    assert exact.stats.selected == "default"
    _assert_equivalent(circ, exact, target)
