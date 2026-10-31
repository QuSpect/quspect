"""Cross-validate ``quspect.transpiler.SABRE`` routing against qiskit.

Layout convention (pinned by ``test_layout_convention_*`` below)
----------------------------------------------------------------
``SABRE(target).solve(circ)`` returns ``(routed, initial, final)`` where

* ``initial[l]`` and ``final[l]`` map **logical -> physical** qubits
  (``pi[logical] = physical``), before and after routing;
* ``routed`` is written on *wires* ``w = 0..m-1``: wire ``w`` is statically
  bound to physical qubit ``initial[w]``.  It is therefore **not** in physical
  index space; the physical circuit is obtained by relabelling ``w ->
  initial[w]``;
* ``m = len(initial)`` may exceed ``circ.num_qubits``: when SABRE swaps
  through an unused physical qubit it appends a *workspace* wire (initially
  ``|0>``) to both mappings.

Equivalence: logical qubit ``l`` starts on wire ``l`` and ends on wire
``sigma[l] = initial.index(final[l])``, so
``U_routed == P_sigma @ (U_original (x) I_workspace)`` exactly (no global
phase), with ``P_sigma`` the little-endian qubit permutation (see
:func:`qiskit_tests._dag_helpers.permutation_matrix`).

The target's coupling graph is **undirected** (``CompilationTarget`` stores
``(min, max)`` pairs), so "on an edge" means ``{initial[a], initial[b]}`` is a
coupling edge in either orientation; SABRE does not fix CX direction.
"""

from __future__ import annotations

import math
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

import numpy as np
import pytest
import qiskit
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import quspect
from qiskit_tests._dag_helpers import (
    grid_edges,
    heavy_hex_edges,
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
    qiskit_probabilities,
    random_circuit_pair,
    to_qiskit,
)
from quspect.ir.compilation import decompose_deep
from quspect.transpiler import SABRE, CompilationTarget

pytestmark = pytest.mark.qiskit_ref

_ROUTABLE = [
    "h", "x", "s", "t", "rx", "ry", "rz", "u3",
    "cx", "cz", "swap", "crz", "cp", "cu3",
]  # fmt: skip

_HH_N, _HH_EDGES = heavy_hex_edges()
TOPOLOGIES: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "line5": (5, line_edges(5)),
    "line7": (7, line_edges(7)),
    "ring5": (5, ring_edges(5)),
    "ring6": (6, ring_edges(6)),
    "grid2x3": (6, grid_edges(2, 3)),
    "grid3x3": (9, grid_edges(3, 3)),
    "heavyhex13": (_HH_N, _HH_EDGES),
}


# =============================================================================
# Helpers
# =============================================================================


def _assert_on_coupling_edges(
    routed: quspect.QuantumCircuit, initial: list[int], target: CompilationTarget
) -> None:
    """Every 2q gate acts on wires bound to coupled physical qubits."""
    for a, b in two_qubit_operands(routed):
        pa, pb = initial[a], initial[b]
        assert (min(pa, pb), max(pa, pb)) in target.coupling_graph, (a, b, pa, pb)
        assert pb in target.adjacency[pa]


def _assert_routed_equivalent(
    circ: quspect.QuantumCircuit,
    routed: quspect.QuantumCircuit,
    initial: list[int],
    final: list[int],
    *,
    seed: int = 0,
    up_to_phase: bool = False,
) -> None:
    """Unitary (m <= 5) or random-input statevector equivalence with P_sigma.

    ``up_to_phase`` allows a global phase: re-synthesized single-qubit
    residuals of elided SWAP blocks do not keep it.
    """
    m = len(initial)
    assert len(final) == m and m >= circ.num_qubits
    assert sorted(final) == sorted(initial)
    sigma = routed_permutation(initial, final)
    if m <= 5:
        u = qiskit_circuit_unitary(circ, num_qubits=m)
        r = qiskit_circuit_unitary(routed, num_qubits=m)
        assert_unitary_equal(r, permutation_matrix(sigma) @ u, up_to_phase=up_to_phase)
        return
    rng = np.random.default_rng(seed)
    for _ in range(2):
        prep = qiskit.QuantumCircuit(m)
        for q in range(circ.num_qubits):  # workspace wires stay |0>
            prep.u(*(float(v) for v in rng.uniform(-math.pi, math.pi, 3)), q)
        sv_in = Statevector(prep)
        orig = sv_in.evolve(to_qiskit(circ, num_qubits=m)).data
        out = sv_in.evolve(to_qiskit(routed, num_qubits=m)).data
        perm = np.zeros_like(orig)
        for x in range(1 << m):
            y = 0
            for bit in range(m):
                if (x >> bit) & 1:
                    y |= 1 << sigma[bit]
            perm[y] = orig[x]
        assert_states_equal(out, perm, up_to_phase=up_to_phase)


def _swap_count(circ: quspect.QuantumCircuit) -> int:
    return quspect_gate_counts(circ)["swap"]


def _qiskit_swaps(
    circ: quspect.QuantumCircuit, edges: list[tuple[int, int]], seed: int
) -> int:
    """SWAPs inserted by qiskit sabre layout+routing (optimization_level=0)."""
    qc = to_qiskit(circ)
    cmap = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    out = transpile(
        qc,
        coupling_map=cmap,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed,
        optimization_level=0,
    )
    return int(out.count_ops().get("swap", 0)) - int(qc.count_ops().get("swap", 0))


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Raise TimeoutError if the block runs longer than ``seconds`` (Unix)."""

    def handler(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"exceeded {seconds} s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# =============================================================================
# Layout convention
# =============================================================================


def test_layout_convention_fixed_mapping_example() -> None:
    """cx(0,1) on line 0-1-2 from layout [0, 2]: initial/final are logical->physical.

    Logical 0 sits on physical 0 and logical 1 on physical 2 (not adjacent).
    SABRE borrows physical 1 as workspace wire 2, swaps wires (0, 2) (physical
    0 <-> 1), then applies cx on wires (2, 1) = physical (1, 2).  Afterwards
    logical 0 lives on physical 1: final == [1, 2, 0].
    """
    target = target_from_edges(3, line_edges(3))
    circ = quspect.QuantumCircuit().cx(0, 1)
    routed, initial, final = SABRE(target).solve(circ, initial_mapping=[0, 2])
    assert initial == [0, 2, 1]
    assert final == [1, 2, 0]
    ops = [(i.gate.name, i.qubits) for i in routed.instructions if i.gate]
    assert ops == [("swap", (0, 2)), ("cx", (2, 1))]
    _assert_on_coupling_edges(routed, initial, target)
    _assert_routed_equivalent(circ, routed, initial, final)


def test_layout_convention_physical_relabelling() -> None:
    """Relabelling wire w -> initial[w] gives the physical circuit.

    In physical space logical l starts on initial[l] and ends on final[l]:
    U_phys = P_final @ (U_orig embedded via P_initial) @ P_initial^dagger.
    """
    target = target_from_edges(5, line_edges(5))
    circ, _ = random_circuit_pair(4, 25, 3, gate_set=_ROUTABLE)
    routed, initial, final = SABRE(target).solve(circ, seed=11)
    m = len(initial)
    # Physical circuit on all 5 physical qubits.
    phys = quspect.QuantumCircuit().id(4)
    for instr in routed.instructions:
        assert instr.gate is not None
        phys.append(instr.gate, tuple(initial[w] for w in instr.qubits))
    free = [p for p in range(5) if p not in initial]
    pi0 = list(initial) + free  # wire -> physical, completed to 5 qubits
    pif = list(final) + free
    u_log = qiskit_circuit_unitary(circ, num_qubits=5)
    expected = permutation_matrix(pif) @ u_log @ permutation_matrix(pi0).conj().T
    assert_unitary_equal(qiskit_circuit_unitary(phys, num_qubits=5), expected)
    assert m <= 5
    # The inverse reading (physical -> logical) is NOT what is returned.
    inv = [pi0.index(p) for p in range(5)]
    if inv != pi0:
        wrong = permutation_matrix(pif) @ u_log @ permutation_matrix(inv).conj().T
        assert not np.allclose(qiskit_circuit_unitary(phys, num_qubits=5), wrong)


def test_fixed_initial_mapping_is_kept() -> None:
    """A supplied initial_mapping is used verbatim (prefix of the result)."""
    target = target_from_edges(5, ring_edges(5))
    circ, _ = random_circuit_pair(4, 20, 8, gate_set=_ROUTABLE)
    given = [3, 0, 4, 1]
    routed, initial, final = SABRE(target).solve(circ, initial_mapping=given)
    assert initial[:4] == given
    assert given == [3, 0, 4, 1]  # caller's list not mutated
    _assert_on_coupling_edges(routed, initial, target)
    _assert_routed_equivalent(circ, routed, initial, final)


# =============================================================================
# Routing correctness on many topologies
# =============================================================================

_ROUTE_CASES = [
    (topo, n, seed) for topo in TOPOLOGIES for n in (2, 3, 4, 5) for seed in range(3)
]


@pytest.mark.parametrize(("topo", "n", "seed"), _ROUTE_CASES)
def test_sabre_routing_valid_and_equivalent(topo: str, n: int, seed: int) -> None:
    """All 2q gates on coupling edges; routed == P_sigma @ original."""
    num_phys, edges = TOPOLOGIES[topo]
    target = target_from_edges(num_phys, edges)
    circ, _ = random_circuit_pair(n, 30, 100 * n + seed, gate_set=_ROUTABLE)
    # Default: input SWAPs and trailing routing SWAPs are elided into ``final``.
    routed, initial, final = SABRE(target).solve(circ, seed=seed)
    _assert_on_coupling_edges(routed, initial, target)
    _assert_routed_equivalent(circ, routed, initial, final, seed=seed, up_to_phase=True)
    # Pure routing: non-swap gates are preserved one-to-one.
    routed, initial, final = SABRE(target).solve(
        circ, seed=seed, elide_permutations=False
    )
    assert len(set(initial)) == len(initial)
    assert all(0 <= p < num_phys for p in initial)
    _assert_on_coupling_edges(routed, initial, target)
    _assert_routed_equivalent(circ, routed, initial, final, seed=seed)
    before = quspect_gate_counts(circ)
    after = quspect_gate_counts(routed)
    added = after["swap"] - before["swap"]
    assert added >= 0
    before["swap"] = after["swap"] = 0
    assert +before == +after


@pytest.mark.parametrize("iter_num", [0, 1, 5])
def test_sabre_iter_num_variants(iter_num: int) -> None:
    """iter_num=0 (random layout only) and larger values all route correctly."""
    target = target_from_edges(6, grid_edges(2, 3))
    circ, _ = random_circuit_pair(5, 40, 21, gate_set=_ROUTABLE)
    routed, initial, final = SABRE(target).solve(circ, iter_num=iter_num, seed=4)
    _assert_on_coupling_edges(routed, initial, target)
    _assert_routed_equivalent(circ, routed, initial, final)


def test_sabre_is_deterministic_for_a_seed() -> None:
    """Same seed -> identical routed circuit and mappings."""
    target = target_from_edges(7, line_edges(7))
    circ, _ = random_circuit_pair(5, 40, 5, gate_set=_ROUTABLE)
    a = SABRE(target).solve(circ, seed=99)
    b = SABRE(target).solve(circ, seed=99)
    assert a[1:] == b[1:]
    assert [repr(i) for i in a[0].instructions] == [repr(i) for i in b[0].instructions]


def test_sabre_no_swaps_when_already_mapped() -> None:
    """A nearest-neighbour circuit on the trivial layout needs no SWAP."""
    target = target_from_edges(4, line_edges(4))
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).cx(1, 2).cx(2, 3).cz(3, 2)
    routed, initial, final = SABRE(target).solve(circ, initial_mapping=[0, 1, 2, 3])
    assert _swap_count(routed) == 0
    assert initial == final == [0, 1, 2, 3]
    assert [repr(i) for i in routed.instructions] == [
        repr(i) for i in circ.instructions
    ]


def test_sabre_all_to_all_needs_no_swaps() -> None:
    """On a complete graph any circuit is routed without SWAPs."""
    edges = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    target = target_from_edges(5, edges)
    circ, _ = random_circuit_pair(5, 40, 1, gate_set=_ROUTABLE)
    routed, initial, final = SABRE(target).solve(circ, seed=0, elide_permutations=False)
    assert _swap_count(routed) == _swap_count(circ)
    _assert_routed_equivalent(circ, routed, initial, final)
    # With elision the input SWAPs become part of the output permutation.
    routed, initial, final = SABRE(target).solve(circ, seed=0)
    assert _swap_count(circ) > 0 and _swap_count(routed) == 0
    _assert_routed_equivalent(circ, routed, initial, final, up_to_phase=True)


def test_sabre_trivial_circuits() -> None:
    """Empty and single-qubit circuits route to themselves."""
    target = target_from_edges(3, line_edges(3))
    routed, initial, final = SABRE(target).solve(quspect.QuantumCircuit(), seed=0)
    assert routed.instructions == () and initial == final == []
    circ = quspect.QuantumCircuit().h(0).t(0)
    routed, initial, final = SABRE(target).solve(circ, seed=0)
    assert routed.instructions == circ.instructions and len(initial) == 1


def test_sabre_directed_coupling_is_treated_as_undirected() -> None:
    """A (1, 0)-only map is stored as {(0, 1)}; cx(0, 1) is kept as is."""
    target = CompilationTarget.from_coupling_map(2, [(1, 0)])
    assert target.coupling_graph == {(0, 1)}
    circ = quspect.QuantumCircuit().cx(0, 1)
    routed, initial, _ = SABRE(target).solve(circ, initial_mapping=[0, 1])
    assert two_qubit_operands(routed) == [(0, 1)]  # direction not corrected
    _assert_on_coupling_edges(routed, initial, target)


# =============================================================================
# Measurements and barriers
# =============================================================================


@pytest.mark.parametrize(
    ("topo", "seed"), [(t, s) for t in ("line5", "grid2x3", "ring5") for s in range(3)]
)
def test_sabre_measurement_distribution_preserved(topo: str, seed: int) -> None:
    """Terminal measurements (permuted clbits) keep the exact distribution."""
    num_phys, edges = TOPOLOGIES[topo]
    target = target_from_edges(num_phys, edges)
    circ, _ = random_circuit_pair(4, 30, 300 + seed, gate_set=_ROUTABLE)
    clbits = [int(c) for c in np.random.default_rng(seed).permutation(4)]
    circ.measure([0, 1, 2, 3], clbits)
    routed, initial, _ = SABRE(target).solve(circ, seed=seed)
    _assert_on_coupling_edges(routed, initial, target)
    measures = [i for i in routed.instructions if i.is_measure]
    assert sorted(c for i in measures for c in i.clbits) == sorted(clbits)
    # Measurements stay terminal: nothing but measurements after the first.
    first = next(k for k, i in enumerate(routed.instructions) if i.is_measure)
    assert all(i.is_measure for i in routed.instructions[first:])
    p_in, p_out = qiskit_probabilities(circ), qiskit_probabilities(routed)
    for k in set(p_in) | set(p_out):
        assert p_out.get(k, 0.0) == pytest.approx(p_in.get(k, 0.0), abs=1e-9)


def test_sabre_measure_all_partial_measure() -> None:
    """measure on a subset + measure_all-style multi-qubit instruction."""
    target = target_from_edges(4, line_edges(4))
    circ = quspect.QuantumCircuit().h(0).cx(0, 3).cx(1, 3).ry(0.4, 2).cx(2, 0)
    circ.measure([3, 0], [0, 1])
    routed, _, _ = SABRE(target).solve(circ, seed=2)
    assert qiskit_probabilities(routed) == pytest.approx(qiskit_probabilities(circ))


def test_sabre_mid_circuit_measurement_fence() -> None:
    """A mid-circuit measurement is a scheduling fence: gate order kept."""
    target = target_from_edges(3, line_edges(3))
    circ = quspect.QuantumCircuit().h(0).cx(0, 2).measure([0], [0]).x(1).cx(1, 2)
    circ.measure([2], [1])
    routed, initial, _ = SABRE(target).solve(circ, seed=0)
    _assert_on_coupling_edges(routed, initial, target)
    kinds = [
        "m" if i.is_measure else i.gate.name
        for i in routed.instructions
        if i.gate or i.is_measure
    ]
    first_m = kinds.index("m")
    assert "x" not in kinds[:first_m]
    assert kinds[-1] == "m"


def test_sabre_drops_barriers_but_keeps_semantics() -> None:
    """Barriers are not forwarded by SABRE (observation); unitary unaffected."""
    target = target_from_edges(4, line_edges(4))
    circ = quspect.QuantumCircuit().h(0).barrier().cx(0, 3).barrier(1, 2).cx(1, 2)
    routed, initial, final = SABRE(target).solve(circ, seed=1)
    assert not any(i.is_barrier for i in routed.instructions)
    _assert_routed_equivalent(circ, routed, initial, final)


# =============================================================================
# CCX lowering before routing (commit daed7bf)
# =============================================================================


def test_sabre_rejects_three_qubit_gates() -> None:
    """Three-qubit gates other than CCX must be decomposed before routing."""
    from quspect.gates.gate import Gate

    target = target_from_edges(3, line_edges(3))
    circ = quspect.QuantumCircuit().append(Gate(name="cswap", num_qubits=3), (0, 1, 2))
    with pytest.raises(ValueError, match=r"3-qubit gate 'cswap'.*decompose"):
        SABRE(target).solve(circ, seed=0)


@pytest.mark.parametrize("seed", range(3))
def test_ccx_lowered_then_routed_is_equivalent(seed: int) -> None:
    """decompose_deep(ccx -> {h,t,tdg,cx}) then SABRE on a line: equivalent."""
    target = target_from_edges(4, line_edges(4))
    circ = quspect.QuantumCircuit().h(0).ccx(0, 3, 1).ry(0.3, 2).ccx(2, 1, 3).cx(0, 3)
    lowered = decompose_deep(
        circ, supported_gates={"h", "t", "tdg", "cx", "ry", "s", "sdg"}
    )
    assert not any(len(i.qubits) > 2 for i in lowered.instructions)
    assert_unitary_equal(
        qiskit_circuit_unitary(lowered, num_qubits=4),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )
    routed, initial, final = SABRE(target).solve(lowered, seed=seed)
    _assert_on_coupling_edges(routed, initial, target)
    u = qiskit_circuit_unitary(circ, num_qubits=len(initial))
    r = qiskit_circuit_unitary(routed, num_qubits=len(initial))
    sigma = routed_permutation(initial, final)
    assert_unitary_equal(r, permutation_matrix(sigma) @ u, up_to_phase=True)


# =============================================================================
# Errors
# =============================================================================


def test_sabre_too_many_logical_qubits() -> None:
    """More logical than physical qubits raises ValueError."""
    target = target_from_edges(2, [(0, 1)])
    circ = quspect.QuantumCircuit().h(2)
    with pytest.raises(ValueError, match="only 2 physical"):
        SABRE(target).solve(circ, seed=0)


@pytest.mark.parametrize(
    ("mapping", "exc", "match"),
    [
        ({0: 0, 1: 1}, TypeError, "list or tuple"),
        ([0], ValueError, "length"),
        ([0, 1.0], TypeError, "integer"),
        ([0, True], TypeError, "integer"),
        ([1, 1], ValueError, "duplicates"),
        ([0, 7], ValueError, "outside"),
        ([-1, 0], ValueError, "outside"),
    ],
)
def test_sabre_initial_mapping_validation(
    mapping: object, exc: type[Exception], match: str
) -> None:
    """Every invalid initial_mapping is rejected with a clear error."""
    target = target_from_edges(3, line_edges(3))
    circ = quspect.QuantumCircuit().cx(0, 1)
    with pytest.raises(exc, match=match):
        SABRE(target).solve(circ, initial_mapping=mapping)  # type: ignore[arg-type]


def test_sabre_isolated_qubits_raise_value_error() -> None:
    """Both operands on isolated physical qubits: no SWAP candidate -> ValueError."""
    target = target_from_edges(4, [(0, 1)])
    circ = quspect.QuantumCircuit().cx(0, 1)
    with pytest.raises(ValueError, match="disconnected"):
        SABRE(target).solve(circ, initial_mapping=[2, 3])


def test_sabre_disconnected_components_fail_fast_bug() -> None:
    """cx between two connected components should raise ValueError quickly."""
    target = target_from_edges(4, [(0, 1), (2, 3)])
    circ = quspect.QuantumCircuit().cx(0, 1)
    with _time_limit(2.0), pytest.raises(ValueError):
        SABRE(target).solve(circ, initial_mapping=[0, 2])


# =============================================================================
# SWAP count vs qiskit
# =============================================================================

_SWAP_CASES = [
    (topo, n, seed)
    for topo in ("line5", "ring6", "grid3x3", "heavyhex13")
    for n in (4, 5)
    for seed in range(3)
]
_SWAP_TABLE: dict[str, list[int]] = {}


@pytest.mark.parametrize(("topo", "n", "seed"), _SWAP_CASES)
def test_swap_count_vs_qiskit(topo: str, n: int, seed: int) -> None:
    """quspect SWAPs <= 3 * qiskit + 3 (soft check, numbers recorded)."""
    num_phys, edges = TOPOLOGIES[topo]
    target = target_from_edges(num_phys, edges)
    circ, _ = random_circuit_pair(n, 40, 7000 + 10 * n + seed, gate_set=_ROUTABLE)
    routed, _, _ = SABRE(target).solve(circ, seed=seed)
    mine = _swap_count(routed) - _swap_count(circ)
    ref = _qiskit_swaps(circ, edges, seed)
    acc = _SWAP_TABLE.setdefault(topo, [0, 0, 0])
    acc[0] += mine
    acc[1] += ref
    acc[2] += 1
    assert mine <= 3 * ref + 3, f"quspect {mine} vs qiskit {ref}"


def test_swap_count_table(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the aggregated SWAP comparison (informational, run with -s)."""
    if not _SWAP_TABLE:
        for topo, n, seed in _SWAP_CASES:
            test_swap_count_vs_qiskit(topo, n, seed)
    with capsys.disabled():
        print("\n[SABRE swap comparison] topology: quspect / qiskit (cases)")
        for topo, (mine, ref, k) in _SWAP_TABLE.items():
            print(f"  {topo:<11} {mine:>4} / {ref:>4}  ({k})")
    total_mine = sum(v[0] for v in _SWAP_TABLE.values())
    total_ref = sum(v[1] for v in _SWAP_TABLE.values())
    assert total_mine <= 2 * total_ref + 5
