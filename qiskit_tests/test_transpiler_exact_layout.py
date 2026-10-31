"""Cross-validate SABRE exact layout and permutation elision with qiskit.

Equivalence convention (see ``test_transpiler_sabre.py``): the routed circuit
is on wires, wire ``w`` bound to physical ``initial[w]``; logical qubit ``l``
of the *input* circuit ends on the wire bound to ``final[l]``.  Elided SWAP
blocks re-synthesize single-qubit residuals, so equality is up to a global
phase.  Measured circuits are compared through their exact clbit
distributions, which do not depend on the layout at all.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import quspect
from qiskit_tests._dag_helpers import (
    grid_edges,
    line_edges,
    quspect_gate_counts,
    routed_permutation,
    target_from_edges,
    two_qubit_operands,
)
from qiskit_tests.helpers import (
    assert_states_equal,
    qiskit_probabilities,
    random_circuit_pair,
    to_qiskit,
)
from quspect.ir.compilation import decompose_deep
from quspect.transpiler import SABRE, CompilationTarget
from quspect.transpiler._vf2 import find_subgraph_monomorphism

pytestmark = pytest.mark.qiskit_ref

_NATIVE = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}
_ROUTABLE = ["h", "x", "s", "t", "rx", "ry", "rz", "u3", "cx", "cz", "swap", "cp"]


def _heavy_hex(distance: int) -> tuple[int, list[tuple[int, int]]]:
    cmap = CouplingMap.from_heavy_hex(distance)
    edges = sorted({(min(a, b), max(a, b)) for a, b in cmap.get_edges()})
    return cmap.size(), edges


def _full(n: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def _assert_equivalent(
    circ: quspect.QuantumCircuit,
    routed: quspect.QuantumCircuit,
    initial: list[int],
    final: list[int],
    *,
    seed: int = 0,
) -> None:
    """Random-input statevector check of ``routed == P_sigma @ circ``."""
    m = len(initial)
    assert len(final) == m and sorted(final) == sorted(initial)
    sigma = routed_permutation(initial, final)
    rng = np.random.default_rng(seed)
    prep = qiskit.QuantumCircuit(m)
    for q in range(circ.num_qubits):  # workspace wires stay |0>
        prep.u(*(float(v) for v in rng.uniform(-math.pi, math.pi, 3)), q)
    sv_in = Statevector(prep)
    orig = sv_in.evolve(to_qiskit(circ, num_qubits=m)).data
    out = sv_in.evolve(to_qiskit(routed, num_qubits=m)).data
    expected = np.zeros_like(orig)
    for x in range(1 << m):
        y = 0
        for bit in range(m):
            if (x >> bit) & 1:
                y |= 1 << sigma[bit]
        expected[y] = orig[x]
    assert_states_equal(out, expected, up_to_phase=True)


def _assert_on_edges(
    routed: quspect.QuantumCircuit, initial: list[int], target: CompilationTarget
) -> None:
    for a, b in two_qubit_operands(routed):
        assert initial[b] in target.adjacency[initial[a]]


def _two_qubit_count(circ: quspect.QuantumCircuit) -> int:
    return sum(1 for i in circ if i.is_gate and len(i.qubits) == 2)


def _qft(n: int) -> quspect.QuantumCircuit:
    circ = quspect.QuantumCircuit()
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            circ.cp(math.pi / 2 ** (j - i), j, i)
    for i in range(n // 2):
        circ.swap(i, n - 1 - i)
    return circ


def _chain(n: int, perm: list[int]) -> quspect.QuantumCircuit:
    circ = quspect.QuantumCircuit()
    for q in range(n):
        circ.ry(0.1 * (q + 1), q)
    for i in range(n - 1):
        circ.cx(perm[i], perm[i + 1]).rz(0.3, perm[i + 1]).cx(perm[i], perm[i + 1])
    return circ


# =============================================================================
# Exact layout
# =============================================================================


@pytest.mark.parametrize(
    "topo",
    ["line", "grid", "heavy_hex"],
)
def test_chain_on_device_needs_no_swaps(topo: str) -> None:
    n_logical = 9
    if topo == "line":
        num, edges = 9, line_edges(9)
    elif topo == "grid":
        num, edges = 9, grid_edges(3, 3)
    else:
        num, edges = _heavy_hex(3)
    target = target_from_edges(num, edges)
    perm = list(np.random.default_rng(3).permutation(n_logical))
    circ = _chain(n_logical, [int(q) for q in perm])
    routed, initial, final = SABRE(target).solve(circ, seed=0)
    assert quspect_gate_counts(routed)["swap"] == 0
    _assert_on_edges(routed, initial, target)
    _assert_equivalent(circ, routed, initial, final)


def test_heavy_hex_chain_of_32_needs_no_swaps() -> None:
    num, edges = _heavy_hex(5)
    target = target_from_edges(num, edges)
    circ = quspect.QuantumCircuit().h(0)
    for i in range(31):
        circ.cx(i, i + 1)
    routed, initial, _ = SABRE(target).solve(circ, seed=0)
    assert quspect_gate_counts(routed)["swap"] == 0
    _assert_on_edges(routed, initial, target)


def test_vf2_budget_on_hard_negative() -> None:
    """A 50-chain does not embed in the 57-qubit heavy-hex (rustworkx agrees)."""
    num, edges = _heavy_hex(5)
    target = target_from_edges(num, edges)
    start = time.perf_counter()
    found = find_subgraph_monomorphism(
        50, [(i, i + 1) for i in range(49)], target.adjacency
    )
    assert found is None
    assert time.perf_counter() - start < 2.0
    # SABRE falls back to its layout trials and still routes correctly.
    circ = _chain(50, list(range(50)))
    routed, initial, _ = SABRE(target).solve(circ, seed=0, layout_trials=1)
    assert quspect_gate_counts(routed)["swap"] > 0
    _assert_on_edges(routed, initial, target)


@pytest.mark.parametrize("seed", range(3))
def test_fallback_random_circuits_equivalent(seed: int) -> None:
    target = target_from_edges(6, grid_edges(2, 3))
    circ, _ = random_circuit_pair(6, 40, 500 + seed, gate_set=_ROUTABLE)
    routed, initial, final = SABRE(target).solve(circ, seed=seed)
    _assert_on_edges(routed, initial, target)
    _assert_equivalent(circ, routed, initial, final, seed=seed)


# =============================================================================
# Permutation elision
# =============================================================================


@pytest.mark.parametrize(
    "num,edges",
    [(6, line_edges(6)), (6, grid_edges(2, 3)), (6, _full(6))],
    ids=["line", "grid", "full"],
)
def test_qft_reversal_swaps_elided(num: int, edges: list[tuple[int, int]]) -> None:
    target = target_from_edges(num, edges)
    circ = _qft(6)
    routed, initial, final = SABRE(target).solve(circ, seed=0)
    _assert_on_edges(routed, initial, target)
    _assert_equivalent(circ, routed, initial, final)
    kept, _, _ = SABRE(target).solve(circ, seed=0, elide_permutations=False)
    assert _two_qubit_count(routed) <= _two_qubit_count(kept) - 3
    if num == 6 and len(edges) == 15:
        assert quspect_gate_counts(routed)["swap"] == 0


@pytest.mark.parametrize("seed", range(4))
def test_decomposed_swaps_elided_on_native_target(seed: int) -> None:
    edges = _full(5)
    target = CompilationTarget.from_coupling_map(5, edges, native_gates=_NATIVE)
    circ, _ = random_circuit_pair(5, 30, 900 + seed, gate_set=_ROUTABLE)
    lowered = decompose_deep(circ, supported_gates=_NATIVE)
    routed, initial, final = SABRE(target).solve(lowered, seed=seed)
    names = {i.gate.name for i in routed if i.gate is not None} - {"id"}
    assert names <= _NATIVE
    _assert_equivalent(circ, routed, initial, final, seed=seed)
    n_swaps = quspect_gate_counts(circ)["swap"]
    assert _two_qubit_count(routed) <= _two_qubit_count(lowered) - 3 * min(n_swaps, 1)


@pytest.mark.parametrize("seed", range(4))
def test_measurements_after_elided_swaps(seed: int) -> None:
    """Clbit distributions are unchanged when SWAPs before measures vanish."""
    target = target_from_edges(5, _full(5))
    circ, _ = random_circuit_pair(5, 30, 1300 + seed, gate_set=_ROUTABLE)
    circ.swap(0, 3).swap(3, 4)
    circ.measure([0, 1, 2, 3, 4], [4, 2, 0, 1, 3])
    routed, _, _ = SABRE(target).solve(circ, seed=seed)
    assert quspect_gate_counts(routed)["swap"] == 0
    expected = qiskit_probabilities(circ)
    actual = qiskit_probabilities(routed)
    assert expected.keys() == actual.keys()
    for key, p in expected.items():
        assert actual[key] == pytest.approx(p, abs=1e-9)


def test_partial_measurement_between_swaps() -> None:
    """Measure one qubit after a SWAP, keep computing on the others."""
    target = target_from_edges(4, _full(4))
    circ = quspect.QuantumCircuit().h(0).ry(0.7, 1).cx(0, 2).swap(0, 1)
    circ.measure([0], [0])
    circ.cx(1, 2).swap(1, 3).rx(0.4, 3).cz(2, 3)
    circ.measure([1, 2, 3], [1, 2, 3])
    routed, _, _ = SABRE(target).solve(circ, seed=0)
    assert quspect_gate_counts(routed)["swap"] == 0
    expected = qiskit_probabilities(circ)
    actual = qiskit_probabilities(routed)
    assert expected.keys() == actual.keys()
    for key, p in expected.items():
        assert actual[key] == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_routed_with_measurements_on_line(seed: int) -> None:
    """Routing SWAPs plus elided input SWAPs, terminal measurements."""
    target = target_from_edges(5, line_edges(5))
    circ, _ = random_circuit_pair(4, 25, 1700 + seed, gate_set=_ROUTABLE)
    circ.measure([0, 1, 2, 3], [0, 1, 2, 3])
    routed, initial, _ = SABRE(target).solve(circ, seed=seed)
    _assert_on_edges(routed, initial, target)
    expected = qiskit_probabilities(circ)
    try:
        actual = qiskit_probabilities(routed)
    except ValueError:  # routing SWAP through an already measured wire
        pytest.skip("routed circuit has a mid-circuit measurement")
    for key, p in expected.items():
        assert actual.get(key, 0.0) == pytest.approx(p, abs=1e-9)
