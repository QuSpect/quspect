"""Tests for :mod:`quspect.hamsim` (product formulas and Pauli-network synthesis)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from quspect._native_loader import load_core
from quspect.circuit.circuit import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.hamsim import (
    commutator_bound,
    compile_evolution,
    connected_active_set,
    cycle_of_length,
    ladder_circuit,
    long_path,
    merge_commuting,
    order_terms,
    pauli_terms,
    product_formula,
    select_term_order,
    state_error,
    synthesize_network,
)
from quspect.operators.qubit_operator import QubitOperator
from quspect.transpiler import CompilationTarget

native_only = pytest.mark.skipif(
    load_core() is None, reason="extension module not built"
)

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

GATES = {"cz", "h", "rz", "s", "sdg", "x", "y", "z"}

#: Fast compile options for the term-order selection tests (it compiles every
#: candidate formula in full).
FAST = {"network_configs": [(16, 64, 512, 850, 1, 1024)], "ladder_level": 1}


def circuit_unitary(circuit: QuantumCircuit, num_qubits: int) -> np.ndarray:
    """Unitary of ``circuit`` with qubit 0 as the most significant bit."""
    dim = 2**num_qubits
    u = np.eye(dim, dtype=complex).reshape([2] * (2 * num_qubits))
    for ins in circuit.instructions:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
        u = np.tensordot(g, u, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        u = np.moveaxis(u, list(range(k)), list(ins.qubits))
    return u.reshape(dim, dim)


def pauli_matrix(letters: str, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Dense Pauli on ``n`` qubits, qubit 0 most significant (as circuit_unitary)."""
    ops = ["I"] * n
    for c, q in zip(letters, qubits, strict=True):
        ops[q] = c
    out = np.ones((1, 1), dtype=complex)
    for c in ops:
        out = np.kron(out, _PAULI[c])
    return out


def rotations_unitary(rotations, n: int) -> np.ndarray:
    u = np.eye(2**n, dtype=complex)
    for p, q, th in rotations:
        u = expm(-1j * th * pauli_matrix(p, q, n)) @ u
    return u


def hamiltonian_unitary(terms, t: float, n: int) -> np.ndarray:
    h = sum(c * pauli_matrix(p, q, n) for p, q, c in terms)
    return expm(-1j * t * h)


def embed(psi: np.ndarray, layout: list[int], big: int) -> np.ndarray:
    """Logical state (qubit 0 most significant) → physical, logical q on layout[q]."""
    n = len(layout)
    out = np.zeros(2**big, dtype=complex)
    for idx, amp in enumerate(psi):
        phys = 0
        for q in range(n):
            if (idx >> (n - 1 - q)) & 1:
                phys |= 1 << (big - 1 - layout[q])
        out[phys] = amp
    return out


def layout_error(
    circuit: QuantumCircuit, big: int, want: np.ndarray, initial, final, seed: int = 3
) -> float:
    """Worst phase-aligned error of the physical circuit over random logical states."""
    u = (
        circuit_unitary(circuit, big)
        if circuit.instructions
        else np.eye(2**big, dtype=complex)
    )
    n = int(round(np.log2(want.shape[0])))
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(3):
        psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi /= np.linalg.norm(psi)
        got = u @ embed(psi, list(initial), big)
        ref = embed(want @ psi, list(final), big)
        ov = np.vdot(ref, got)
        worst = max(worst, float(np.linalg.norm(got - (ov / abs(ov)) * ref)))
    return worst


def heisenberg(n: int, periodic: bool = False):
    terms = []
    bonds = [(i, i + 1) for i in range(n - 1)] + ([(0, n - 1)] if periodic else [])
    for p in "XYZ":
        for a, b in bonds:
            terms.append((p + p, (a, b), 1.0))
    return terms


def hubbard_like():
    # JW hopping strings + density terms on 5 qubits
    return [
        ("XZX", (0, 1, 2), -0.5),
        ("YZY", (0, 1, 2), -0.5),
        ("XZZX", (1, 2, 3, 4), -0.5),
        ("YZZY", (1, 2, 3, 4), -0.5),
        ("ZZ", (0, 1), 0.7),
        ("ZZ", (2, 3), 0.7),
        ("Z", (4,), 0.3),
        ("XX", (3, 4), 0.2),
    ]


def line(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


# ---------------------------------------------------------------- terms


def test_pauli_terms_from_triples_and_operator() -> None:
    n, terms = pauli_terms([("XZ", [2, 0], 0.5), ("II", [0, 1], 3.0), ("Y", [1], 0.0)])
    assert n == 3
    assert terms == [("XZ", (2, 0), 0.5)]
    op = QubitOperator({((0, "X"), (2, "Z")): 0.25, ((1, "Y"),): -1.0})
    n2, t2 = pauli_terms(op)
    assert n2 == 3
    assert sorted(t2) == sorted([("XZ", (0, 2), 0.25), ("Y", (1,), -1.0)])


def test_pauli_terms_sparse_pauli_op() -> None:
    qi = pytest.importorskip("qiskit.quantum_info")
    op = qi.SparsePauliOp.from_sparse_list(
        [("XY", [0, 3], 0.5), ("Z", [1], -2.0)], num_qubits=5
    )
    n, terms = pauli_terms(op)
    assert n == 5
    assert terms == [("XY", (0, 3), 0.5), ("Z", (1,), -2.0)]


@pytest.mark.parametrize(
    ("bad", "exc"),
    [
        ([("XX", (0, 0), 1.0)], ValueError),
        ([("X", (0,), 1.0 + 0.5j)], ValueError),
        ([("Q", (0,), 1.0)], ValueError),
        ("XX", TypeError),
    ],
)
def test_pauli_terms_rejects(bad, exc) -> None:
    with pytest.raises(exc):
        pauli_terms(bad)
    with pytest.raises(ValueError):
        pauli_terms([("X", (3,), 1.0)], num_qubits=2)


# ---------------------------------------------------------------- formulas


def test_product_formula_orders_and_merging() -> None:
    terms = [("ZZ", (0, 1), 1.0), ("X", (0,), 0.5), ("X", (1,), 0.5)]
    assert len(product_formula(terms, 1.0, 3, 1)) == 9
    # Strang: 2m − 1 per step, step boundaries merge
    assert len(product_formula(terms, 1.0, 1, 2)) == 5
    assert len(product_formula(terms, 1.0, 2, 2)) == 9
    s4 = product_formula(terms, 1.0, 1, 4)
    assert len(s4) == 5 * 4 + 1
    # the angles add up to c_j t for every term
    for p, q, c in terms:
        total = sum(th for pp, qq, th in s4 if (pp, qq) == (p, q))
        assert total == pytest.approx(c * 1.0)
    with pytest.raises(ValueError):
        product_formula(terms, 1.0, 1, 3)
    with pytest.raises(ValueError):
        product_formula(terms, 1.0, 0, 2)


def test_order_terms_support_groups_bonds() -> None:
    terms = heisenberg(6, periodic=True)
    grouped = order_terms(terms, "support")
    assert sorted(grouped) == sorted(terms)
    assert order_terms(terms, "input") == terms
    # XX, YY, ZZ of one bond are adjacent
    keys = [tuple(sorted(q)) for _, q, _ in grouped]
    for i in range(0, len(keys), 3):
        assert keys[i] == keys[i + 1] == keys[i + 2]
    # bonds of a colour class are disjoint
    with pytest.raises(ValueError):
        order_terms(terms, "random")


def test_state_error_resolves_errors_far_below_1e_8() -> None:
    """The distance is computed directly, not as sqrt(2 - 2|<a|b>|).

    That expression cannot resolve anything below about 2e-8 in double
    precision and turns a norm drift of 1e-12 after many rotations into an
    "error" of 1e-6.
    """
    exact = [("ZZ", (0, 1), 1.0), ("Z", (0,), 0.5)]  # commuting: no Trotter error
    assert state_error(2, exact, 1.0, product_formula(exact, 1.0, 1, 1)) < 1e-12

    terms = [("XX", (0, 1), 0.7), ("Z", (0,), 0.5), ("Y", (1,), 0.3)]
    errors = {
        steps: state_error(2, terms, 1.0, product_formula(terms, 1.0, steps, 2))
        for steps in (10, 100, 1000, 10000)
    }
    assert errors[10] == pytest.approx(9.7366e-4, rel=1e-4)
    # second order: 10x the steps divides the error by 100, down to 1e-9
    for steps in (10, 100, 1000):
        ratio = errors[steps] / errors[10 * steps]
        assert ratio == pytest.approx(100.0, rel=0.02), (steps, errors)
    assert errors[10000] < 2e-9


def test_state_error_and_bound_converge() -> None:
    terms = heisenberg(4)
    errs = [
        state_error(4, terms, 1.0, product_formula(terms, 1.0, r, 2)) for r in (2, 4, 8)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.25)  # second order
    u = rotations_unitary(product_formula(terms, 1.0, 4, 2), 4)
    v = hamiltonian_unitary(terms, 1.0, 4)
    op_err = np.linalg.norm(u - v, 2)
    bound = commutator_bound(4, terms, 1.0, 4, 2)
    assert bound >= op_err
    assert commutator_bound(4, terms, 1.0, 8, 2) == pytest.approx(bound / 4)
    assert commutator_bound(4, terms, 1.0, 4, 1) > 0.0


@pytest.mark.parametrize("order", [1, 2])
def test_commutator_bound_depends_on_the_magnitude_of_time(order: int) -> None:
    """The bound is a norm: backward evolution is bounded as forward evolution."""
    terms = heisenberg(4)
    for steps in (1, 3):
        forward = commutator_bound(4, terms, 1.0, steps, order)
        backward = commutator_bound(4, terms, -1.0, steps, order)
        assert backward == forward > 0.0
    assert commutator_bound(4, terms, 0.0, 1, order) == 0.0
    assert commutator_bound(4, terms, -0.0, 1, order) == 0.0


def test_commutator_bound_holds_for_random_hamiltonians_and_signed_times() -> None:
    """Property check: the bound is non-negative, even in ``t`` and at least the
    true operator-norm error of the formula (dense reference)."""
    rng = np.random.default_rng(20261109)
    n = 3
    for _ in range(30):
        terms = []
        for _ in range(int(rng.integers(2, 7))):
            k = int(rng.integers(1, n + 1))
            qubits = tuple(int(q) for q in rng.choice(n, size=k, replace=False))
            letters = "".join(rng.choice(list("XYZ"), size=k))
            terms.append((letters, qubits, float(rng.normal())))
        t = float(rng.uniform(0.05, 1.5)) * (-1.0 if rng.random() < 0.7 else 1.0)
        steps = int(rng.integers(1, 5))
        exact = hamiltonian_unitary(terms, t, n)
        for order in (1, 2):
            bound = commutator_bound(n, terms, t, steps, order)
            assert bound >= 0.0
            assert bound == commutator_bound(n, terms, -t, steps, order)
            u = rotations_unitary(product_formula(terms, t, steps, order), n)
            assert bound >= np.linalg.norm(u - exact, 2) - 1e-9


# ---------------------------------------------------------------- native synthesis


@native_only
def test_merge_commuting_is_exact() -> None:
    terms = hubbard_like()
    rots = product_formula(terms, 0.8, 2, 2)
    merged = merge_commuting(rots, 5)
    assert len(merged) < len(rots)
    np.testing.assert_allclose(
        rotations_unitary(merged, 5), rotations_unitary(rots, 5), atol=1e-10
    )


@native_only
@pytest.mark.parametrize("finalize", [True, False])
def test_synthesize_network_exact_on_line(finalize: bool) -> None:
    terms = hubbard_like()
    rots = merge_commuting(product_formula(terms, 0.7, 1, 2), 5)
    net = synthesize_network(rots, 5, line(5), list(range(5)), finalize=finalize)
    for ins in net.circuit.instructions:
        if len(ins.qubits) == 2:
            a, b = ins.qubits
            assert abs(a - b) == 1
    assert net.num_cx == sum(
        1 for ins in net.circuit.instructions if ins.gate.name == "cx"
    )
    u = circuit_unitary(net.circuit, 5)
    want = rotations_unitary(rots, 5)
    if finalize:
        ov = np.vdot(want.ravel(), u.ravel())
        np.testing.assert_allclose(u, (ov / abs(ov)) * want, atol=1e-9)
    else:
        assert net.finalize_cx == 0


@native_only
def test_synthesize_network_with_layout_and_connector() -> None:
    # 3 logical qubits on a 2x3 grid, layout leaves gaps → connectors in the active set
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    terms = [("XX", (0, 1), 1.0), ("ZYZ", (0, 1, 2), 0.4), ("YY", (1, 2), -0.3)]
    rots = product_formula(terms, 0.5, 1, 2)
    layout = [0, 5, 2]
    active = connected_active_set(6, edges, layout)
    assert set(layout) <= set(active)
    net = synthesize_network(rots, 6, edges, layout, active=active)
    eset = {tuple(sorted(e)) for e in edges}
    for ins in net.circuit.instructions:
        if len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in eset
    big = 1 + max(max(ins.qubits) for ins in net.circuit.instructions)
    err = layout_error(net.circuit, big, rotations_unitary(rots, 3), layout, layout)
    assert err < 1e-9


@native_only
def test_synthesize_network_deterministic() -> None:
    rots = product_formula(heisenberg(5), 1.0, 2, 4)
    a = synthesize_network(rots, 5, line(5), list(range(5)))
    b = synthesize_network(rots, 5, line(5), list(range(5)))
    assert a.num_cx == b.num_cx
    assert [(i.gate.name, i.qubits) for i in a.circuit.instructions] == [
        (i.gate.name, i.qubits) for i in b.circuit.instructions
    ]


@native_only
@pytest.mark.parametrize("block_mode", [0, 1])
def test_synthesize_network_permutation_and_blocks(block_mode: int) -> None:
    # final frame reduced to a qubit permutation; blocks / mirror SWAPs on the line
    rots = merge_commuting(
        product_formula(heisenberg(5) + hubbard_like(), 0.4, 2, 2), 5
    )
    net = synthesize_network(
        rots,
        5,
        line(5),
        list(range(5)),
        block_mode=block_mode,
        block_bias=600,
        finalize_permutation=True,
    )
    assert sorted(net.final_permutation) == list(range(5))
    if block_mode:
        assert net.blocks > 0
    for ins in net.circuit.instructions:
        if len(ins.qubits) == 2:
            assert abs(ins.qubits[0] - ins.qubits[1]) == 1
    err = layout_error(
        net.circuit,
        5,
        rotations_unitary(rots, 5),
        list(range(5)),
        net.final_permutation,
    )
    assert err < 1e-9


@native_only
def test_synthesize_network_cx_limit() -> None:
    rots = product_formula(hubbard_like(), 0.7, 2, 2)
    ref = synthesize_network(rots, 5, line(5), list(range(5)))
    assert not ref.aborted
    cut = synthesize_network(rots, 5, line(5), list(range(5)), cx_limit=ref.num_cx // 3)
    assert cut.aborted and not cut.circuit.instructions


def test_ladder_circuit_matches_rotations() -> None:
    rots = product_formula(hubbard_like(), 0.3, 1, 1)
    u = circuit_unitary(ladder_circuit(rots, 5), 5)
    want = rotations_unitary(rots, 5)
    ov = np.vdot(want.ravel(), u.ravel())
    np.testing.assert_allclose(u, (ov / abs(ov)) * want, atol=1e-9)


# ---------------------------------------------------------------- compile_evolution


@native_only
@pytest.mark.parametrize("term_order", ["input", "support"])
def test_compile_evolution_line_target(term_order: str) -> None:
    terms = hubbard_like()
    target = CompilationTarget.from_coupling_map(6, line(6), native_gates=GATES)
    res = compile_evolution(terms, 0.6, target, steps=2, order=2, term_order=term_order)
    assert res.steps == 2 and res.order == 2 and res.term_order == term_order
    assert set(res.candidates) == {"network", "ladder"}
    assert res.two_qubit_gates == min(res.candidates.values())
    names = {i.gate.name for i in res.circuit.instructions}
    assert names <= GATES
    for ins in res.circuit.instructions:
        if len(ins.qubits) == 2:
            assert abs(ins.qubits[0] - ins.qubits[1]) == 1
    big = max(6, 1 + max(max(i.qubits) for i in res.circuit.instructions))
    want = rotations_unitary(res.rotations, 5)
    assert (
        layout_error(res.circuit, big, want, res.initial_layout, res.final_layout)
        < 1e-8
    )
    if term_order == "input":
        np.testing.assert_allclose(
            want, rotations_unitary(product_formula(terms, 0.6, 2, 2), 5), atol=1e-12
        )


def _assert_evolution_on_device(res, n: int, target: CompilationTarget) -> None:
    """The compiled circuit is the rotation sequence, on couplings of ``target``."""
    for layout in (res.initial_layout, res.final_layout):
        assert len(layout) == len(set(layout)) == n
    for ins in res.circuit.instructions:
        assert ins.gate.name in target.native_gates
        if len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in target.coupling_graph
    want = rotations_unitary(res.rotations, n)
    big = target.num_qubits
    assert (
        layout_error(res.circuit, big, want, res.initial_layout, res.final_layout)
        < 1e-8
    )


@native_only
@pytest.mark.parametrize(
    "terms",
    [
        [("ZZ", (1, 2), 1.0), ("X", (2,), 1.0)],  # qubit 0 idle
        [("ZZ", (0, 3), 1.0), ("X", (0,), 0.5), ("X", (3,), 0.5)],  # 1, 2 idle
        [("XX", (0, 2), 0.4), ("ZZ", (2, 4), 0.7), ("Y", (4,), 0.3)],  # 1, 3 idle
    ],
)
@pytest.mark.parametrize("shape", ["grid", "line"])
def test_compile_evolution_on_device_with_idle_qubit_index(terms, shape: str) -> None:
    """A Hamiltonian may skip a qubit index (the network candidate used to fail
    with "initial_mapping length ... must equal the circuit qubit count")."""
    edges = GRID_2X3 if shape == "grid" else line(5)
    size = 6 if shape == "grid" else 5
    target = CompilationTarget.from_coupling_map(size, edges, native_gates=GATES)
    n = 1 + max(max(q) for _, q, _ in terms)
    for methods in (("network", "ladder"), ("network",), ("ladder",)):
        res = compile_evolution(terms, 1.0, target, steps=2, order=2, methods=methods)
        assert set(res.candidates) == set(methods)
        _assert_evolution_on_device(res, n, target)


@native_only
def test_compile_evolution_random_hamiltonians_with_idle_indices() -> None:
    """Property check: random Hamiltonians that leave qubit indices unused."""
    rng = np.random.default_rng(20261109)
    targets = [
        CompilationTarget.from_coupling_map(6, GRID_2X3, native_gates=GATES),
        CompilationTarget.from_coupling_map(6, line(6), native_gates=GATES),
        CompilationTarget.from_coupling_map(
            5, [(0, 1), (0, 2), (0, 3), (0, 4)], native_gates=GATES
        ),
    ]
    for case in range(18):
        target = targets[case % len(targets)]
        n = int(rng.integers(3, target.num_qubits + 1))
        active = sorted(
            int(q)
            for q in rng.choice(n - 1, size=int(rng.integers(1, n - 1)), replace=False)
        ) + [n - 1]
        terms = []
        for _ in range(int(rng.integers(2, 6))):
            k = int(rng.integers(1, min(3, len(active)) + 1))
            qubits = tuple(int(q) for q in rng.choice(active, size=k, replace=False))
            letters = "".join(rng.choice(list("XYZ"), size=k))
            terms.append((letters, qubits, float(rng.uniform(0.2, 1.0))))
        terms.append(("Z", (n - 1,), 0.5))  # the Hamiltonian has n qubits
        assert len({q for _, qs, _ in terms for q in qs}) < n
        res = compile_evolution(
            terms, 0.7, target, steps=int(rng.integers(1, 3)), order=2, **FAST
        )
        _assert_evolution_on_device(res, n, target)


@native_only
@pytest.mark.parametrize("on_line", [False, True])
def test_compile_evolution_network_beats_ladder_on_jw_strings(on_line: bool) -> None:
    terms = hubbard_like()
    target = (
        CompilationTarget.from_coupling_map(6, line(6), native_gates=GATES)
        if on_line
        else None
    )
    res = compile_evolution(terms, 1.0, target, steps=3, order=2, methods=("network",))
    assert res.method == "network"
    lad = compile_evolution(terms, 1.0, target, steps=3, order=2, methods=("ladder",))
    assert res.two_qubit_gates < lad.two_qubit_gates
    both = compile_evolution(terms, 1.0, target, steps=3, order=2)
    assert both.two_qubit_gates == res.two_qubit_gates


@native_only
def test_compile_evolution_eps_meets_target() -> None:
    terms = [
        ("ZZ", (0, 1), 1.0),
        ("ZZ", (1, 2), 1.0),
        ("X", (0,), 0.7),
        ("X", (1,), 0.7),
        ("X", (2,), 0.7),
    ]
    res = compile_evolution(terms, 1.0, eps=1e-3, order=2)
    assert res.error_metric == "state"
    assert res.error_estimate is not None and res.error_estimate <= 1e-3
    worse = state_error(3, terms, 1.0, product_formula(terms, 1.0, res.steps - 1, 2))
    assert worse > 1e-3
    big = max(3, 1 + max(max(i.qubits) for i in res.circuit.instructions))
    exact = hamiltonian_unitary(terms, 1.0, 3)
    assert (
        layout_error(res.circuit, big, exact, res.initial_layout, res.final_layout)
        <= 1.5e-3
    )
    bres = compile_evolution(
        terms, 1.0, eps=1e-2, order=2, error_metric="bound", methods=("ladder",)
    )
    assert bres.error_metric == "bound" and bres.steps >= 1


@pytest.mark.parametrize(
    ("order", "metric"),
    [(1, "bound"), (2, "bound"), (1, "state"), (2, "state"), (4, "state")],
)
def test_compile_evolution_eps_meets_target_for_negative_time(
    order: int, metric: str
) -> None:
    """``eps`` is met for ``t < 0`` as for ``t > 0`` (the order-2 bound used to be
    negative there, so one step "met" every ``eps``)."""
    terms = heisenberg(4)
    eps = 2e-2
    kwargs = {"eps": eps, "order": order, "error_metric": metric}
    fast = {"methods": ("ladder",), "optimization_level": 1}
    back = compile_evolution(terms, -1.0, **kwargs, **fast)
    assert back.error_metric == metric
    assert back.error_estimate is not None and 0.0 <= back.error_estimate <= eps
    if metric == "bound":
        fwd = compile_evolution(terms, 1.0, **kwargs, **fast)
        assert (back.steps, back.error_estimate) == (fwd.steps, fwd.error_estimate)
    # the formula really is that close to exp(-i H t), and so is the circuit
    exact = hamiltonian_unitary(terms, -1.0, 4)
    u = rotations_unitary(back.rotations, 4)
    ov = np.trace(exact.conj().T @ u)
    assert np.linalg.norm(u - ov / abs(ov) * exact, 2) <= (
        eps if metric == "bound" else 3 * eps
    )
    big = max(4, 1 + max(max(i.qubits) for i in back.circuit.instructions))
    assert (
        layout_error(back.circuit, big, exact, back.initial_layout, back.final_layout)
        <= 1.5 * eps
    )


def test_select_term_order_bound_metric_with_negative_time() -> None:
    terms = heisenberg(4, periodic=True) + [
        ("Z", (q,), 0.3 * (q + 1)) for q in range(4)
    ]
    kwargs = {"error_metric": "bound", "compile_options": FAST}
    back = select_term_order(terms, -0.5, 5e-2, 2, **kwargs)
    fwd = select_term_order(terms, 0.5, 5e-2, 2, **kwargs)
    assert {k: v[:2] for k, v in back.table.items()} == {
        k: v[:2] for k, v in fwd.table.items()
    }
    assert all(0.0 < e <= 5e-2 for _, e, _ in back.table.values())
    assert back.steps > 1


@native_only
def test_select_term_order_and_auto() -> None:
    terms = heisenberg(4, periodic=True) + [
        ("Z", (q,), 0.3 * (q + 1)) for q in range(4)
    ]
    ch = select_term_order(terms, 1.0, 1e-3, 2, compile_options=FAST)
    assert set(ch.table) == {"input", "support"}
    assert ch.term_order in ch.table and ch.error_metric == "state"
    steps, err, cost = ch.table[ch.term_order]
    assert ch.steps == steps and err <= 1e-3
    assert cost == min(c for _, _, c in ch.table.values())
    for name, (r, e, _) in ch.table.items():
        assert e == state_error(
            4,
            order_terms(terms, name),
            1.0,
            product_formula(order_terms(terms, name), 1.0, r, 2),
        )
    res = compile_evolution(terms, 1.0, eps=1e-3, order=2, term_order="auto", **FAST)
    assert res.term_order == ch.term_order and res.steps == ch.steps
    assert res.term_order_table == ch.table
    big = max(4, 1 + max(max(i.qubits) for i in res.circuit.instructions))
    exact = hamiltonian_unitary(terms, 1.0, 4)
    assert (
        layout_error(res.circuit, big, exact, res.initial_layout, res.final_layout)
        <= 1.5e-3
    )
    with pytest.raises(ValueError):
        compile_evolution(terms, 1.0, steps=2, term_order="auto")
    # the cost is estimated on the given target
    chain = CompilationTarget.from_coupling_map(4, line(4), native_gates=GATES)
    on_line = select_term_order(terms, 1.0, 1e-3, 2, target=chain, compile_options=FAST)
    assert {k: v[:2] for k, v in on_line.table.items()} == {
        k: v[:2] for k, v in ch.table.items()
    }
    lres = compile_evolution(
        terms, 1.0, chain, eps=1e-3, order=2, term_order="auto", **FAST
    )
    assert lres.term_order == on_line.term_order
    assert lres.term_order_table == on_line.table


@native_only
def test_compile_evolution_fixed_layout_and_config() -> None:
    # the selected network layout and setting reproduce the same circuit
    terms = hubbard_like()
    target = CompilationTarget.from_coupling_map(6, line(6), native_gates=GATES)
    res = compile_evolution(terms, 0.8, target, steps=2, order=2, methods=("network",))
    assert res.network_layout is not None and res.network_config is not None
    again = compile_evolution(
        terms,
        0.8,
        target,
        steps=2,
        order=2,
        methods=("network",),
        layouts=[res.network_layout],
        network_configs=[res.network_config],
    )
    assert again.two_qubit_gates == res.two_qubit_gates
    assert again.initial_layout == res.initial_layout
    assert again.final_layout == res.final_layout
    old = compile_evolution(
        terms, 0.8, target, steps=2, order=2, finalize_permutation=False
    )
    want = rotations_unitary(old.rotations, 5)
    big = max(6, 1 + max(max(i.qubits) for i in old.circuit.instructions))
    assert (
        layout_error(old.circuit, big, want, old.initial_layout, old.final_layout)
        < 1e-8
    )


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


def test_long_path_grid_and_tree() -> None:
    edges = _grid_edges(3, 4)
    path = long_path(12, edges)
    assert sorted(path) == list(range(12))  # Hamiltonian path of the grid
    es = {frozenset(e) for e in edges}
    assert all(frozenset(p) in es for p in zip(path, path[1:], strict=False))
    assert long_path(12, edges) == path  # deterministic
    star = [(0, 1), (0, 2), (0, 3)]
    assert len(long_path(4, star)) == 3
    assert long_path(3, []) == [0]


@native_only
def test_compile_evolution_path_subgraph() -> None:
    # a Jordan–Wigner chain on a grid: the network may run on a long path of it
    terms = hubbard_like()
    edges = _grid_edges(2, 4)
    grid = CompilationTarget.from_coupling_map(8, edges, native_gates=GATES)
    res = compile_evolution(
        terms,
        0.7,
        grid,
        steps=3,
        order=2,
        methods=("network",),
        network_graphs=("path",),
    )
    assert res.method == "network" and res.network_graph == "path"
    path = long_path(8, edges)
    on_path = {frozenset(e) for e in zip(path, path[1:], strict=False)}
    for ins in res.circuit.instructions:
        if len(ins.qubits) == 2:
            assert frozenset(ins.qubits) in on_path
    want = rotations_unitary(res.rotations, 5)
    assert (
        layout_error(res.circuit, 8, want, res.initial_layout, res.final_layout) < 1e-8
    )
    both = compile_evolution(terms, 0.7, grid, steps=3, order=2, methods=("network",))
    dev = compile_evolution(
        terms,
        0.7,
        grid,
        steps=3,
        order=2,
        methods=("network",),
        network_graphs=("device",),
    )
    assert both.two_qubit_gates <= min(dev.two_qubit_gates, res.two_qubit_gates)
    again = compile_evolution(
        terms,
        0.7,
        grid,
        steps=3,
        order=2,
        methods=("network",),
        layouts=[res.network_layout],
        network_configs=[res.network_config],
        network_graphs=[res.network_graph],
    )
    assert again.two_qubit_gates == res.two_qubit_gates
    # all-to-all connectivity has no path variant
    full = compile_evolution(terms, 0.7, steps=1, order=2, network_graphs=("path",))
    assert full.method == "ladder" or full.network_graph == "device"
    with pytest.raises(ValueError):
        compile_evolution(terms, 0.7, grid, steps=1, network_graphs=("tree",))


@native_only
def test_default_portfolio_extends_within_budget() -> None:
    from quspect.hamsim._compile import NETWORK_CONFIGS

    terms = hubbard_like()
    base = compile_evolution(
        terms,
        0.8,
        steps=2,
        order=2,
        methods=("network",),
        network_configs=NETWORK_CONFIGS,
    )
    ext = compile_evolution(terms, 0.8, steps=2, order=2, methods=("network",))
    assert ext.two_qubit_gates <= base.two_qubit_gates
    none = compile_evolution(
        terms,
        0.8,
        steps=2,
        order=2,
        methods=("network",),
        network_configs=None,
    )
    assert none.two_qubit_gates == ext.two_qubit_gates


def test_cycle_of_length_grid() -> None:
    edges = _grid_edges(2, 4)
    es = {frozenset(e) for e in edges}
    for k in (4, 6, 8):
        cyc = cycle_of_length(8, edges, k)
        assert len(cyc) == len(set(cyc)) == k
        assert all(frozenset((cyc[i], cyc[(i + 1) % k])) in es for i in range(k))
        assert cycle_of_length(8, edges, k) == cyc  # deterministic
    assert cycle_of_length(8, edges, 5) == []  # odd cycle on a bipartite graph
    assert cycle_of_length(8, edges, 9) == []
    assert cycle_of_length(4, [(0, 1), (1, 2), (2, 3)], 4) == []  # a path


@native_only
def test_compile_evolution_cycle_subgraph() -> None:
    # a periodic chain on a grid: the network may run on a cycle of n qubits
    terms = heisenberg(6, periodic=True)
    edges = _grid_edges(2, 4)
    grid = CompilationTarget.from_coupling_map(8, edges, native_gates=GATES)
    res = compile_evolution(
        terms,
        0.6,
        grid,
        steps=2,
        order=2,
        methods=("network",),
        network_graphs=("cycle",),
    )
    assert res.network_graph == "cycle"
    cyc = cycle_of_length(8, edges, 6)
    ring = {frozenset((cyc[i], cyc[(i + 1) % 6])) for i in range(6)}
    for ins in res.circuit.instructions:
        if len(ins.qubits) == 2:
            assert frozenset(ins.qubits) in ring
    want = rotations_unitary(res.rotations, 6)
    assert (
        layout_error(res.circuit, 8, want, res.initial_layout, res.final_layout) < 1e-8
    )
    # without a wrap-around term the cycle is not used
    open_chain = compile_evolution(
        heisenberg(6),
        0.6,
        grid,
        steps=2,
        order=2,
        methods=("network",),
        network_graphs=("cycle",),
    )
    assert open_chain.network_graph == "device"


@native_only
def test_compile_evolution_block_steps(monkeypatch) -> None:
    from quspect.hamsim import _compile

    terms = hubbard_like()
    chain = CompilationTarget.from_coupling_map(6, line(6), native_gates=GATES)
    res = compile_evolution(terms, 0.9, chain, steps=3, order=2, block_steps=1)
    assert res.block_steps == 1 and res.method == "network" and res.steps == 3
    assert res.rotations == product_formula(terms, 0.9, 3, 2)
    want = rotations_unitary(res.rotations, 5)
    big = max(6, 1 + max(max(i.qubits) for i in res.circuit.instructions))
    assert (
        layout_error(res.circuit, big, want, res.initial_layout, res.final_layout)
        < 1e-8
    )
    assert res.two_qubit_gates == sum(
        1 for ins in res.circuit.instructions if len(ins.qubits) == 2
    )
    # large repeated blocks: the block is transpiled once and repeated
    monkeypatch.setattr(_compile, "REPEAT_TRANSPILE_CX", 0)
    once = compile_evolution(terms, 0.9, chain, steps=3, order=2, block_steps=1)
    assert once.block_steps == 1
    big = max(6, 1 + max(max(i.qubits) for i in once.circuit.instructions))
    assert (
        layout_error(once.circuit, big, want, once.initial_layout, once.final_layout)
        < 1e-8
    )
    assert once.two_qubit_gates == sum(
        1 for ins in once.circuit.instructions if len(ins.qubits) == 2
    )
    assert once.two_qubit_gates % 3 == 0
    monkeypatch.undo()
    plain = compile_evolution(terms, 0.9, chain, steps=3, order=2)
    never = compile_evolution(terms, 0.9, chain, steps=3, order=2, block_steps=0)
    assert plain.block_steps is None and never.block_steps is None
    assert plain.two_qubit_gates == never.two_qubit_gates
    with pytest.raises(ValueError):
        compile_evolution(terms, 0.9, chain, steps=3, order=2, block_steps=2)
    # automatic: a formula whose single network run exceeds its share of the
    # budget is compiled as a repeated one-step block
    monkeypatch.setattr(_compile, "PROBE_DIVISOR", 2)
    auto = compile_evolution(terms, 0.9, chain, steps=4, order=2, block_steps=None)
    assert auto.block_steps is None
    # (a full run of the four-step formula needs about 140 000 work units, a
    # run of the one-step block about 20 000)
    monkeypatch.setattr(_compile, "WORK_BUDGET", 160_000)
    small = compile_evolution(terms, 0.9, chain, steps=4, order=2)
    assert small.block_steps == 1
    want = rotations_unitary(small.rotations, 5)
    big = max(6, 1 + max(max(i.qubits) for i in small.circuit.instructions))
    assert (
        layout_error(small.circuit, big, want, small.initial_layout, small.final_layout)
        < 1e-8
    )
    # without the probe: the first full run stops at half of the budget
    # (deterministic work limit) and the formula is compiled as blocks too
    monkeypatch.setattr(_compile, "PROBE_DIVISOR", 1000)
    monkeypatch.setattr(_compile, "WORK_BUDGET", 240_000)
    capped = compile_evolution(terms, 0.9, chain, steps=4, order=2)
    assert capped.block_steps == 1
    want = rotations_unitary(capped.rotations, 5)
    big = max(6, 1 + max(max(i.qubits) for i in capped.circuit.instructions))
    assert (
        layout_error(
            capped.circuit, big, want, capped.initial_layout, capped.final_layout
        )
        < 1e-8
    )
    # a one-step formula has no blocks: a run above the limit leaves the
    # ladder, or (network only) one run in full
    monkeypatch.setattr(_compile, "WORK_BUDGET", 2_000)
    lad = compile_evolution(terms, 0.9, chain, steps=1, order=2)
    assert lad.method == "ladder"
    net = compile_evolution(terms, 0.9, chain, steps=1, order=2, methods=("network",))
    assert net.method == "network"


@native_only
def test_default_portfolio_screens_within_budget(monkeypatch) -> None:
    from quspect.hamsim import _compile

    terms = hubbard_like()
    grid = CompilationTarget.from_coupling_map(8, _grid_edges(2, 4), native_gates=GATES)
    calls: list[int] = []
    real = _compile.synthesize_network

    def counted(rotations, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        calls.append(len(rotations))
        return real(rotations, *args, **kwargs)

    monkeypatch.setattr(_compile, "synthesize_network", counted)
    wide = compile_evolution(terms, 0.7, grid, steps=3, order=2, block_steps=0)
    n_wide = sum(calls)  # rotations processed
    calls.clear()
    # a small budget: stage 1 is screened on prefixes, stage 2 is cut short
    monkeypatch.setattr(_compile, "EXHAUSTIVE_WORK", 0)
    monkeypatch.setattr(_compile, "WORK_BUDGET", 20_000_000)
    tight = compile_evolution(terms, 0.7, grid, steps=3, order=2, block_steps=0)
    assert sum(calls) < n_wide
    full_len = max(calls)
    assert any(c < full_len for c in calls)  # prefix runs
    want = rotations_unitary(tight.rotations, 5)
    assert (
        layout_error(tight.circuit, 8, want, tight.initial_layout, tight.final_layout)
        < 1e-8
    )
    again = compile_evolution(terms, 0.7, grid, steps=3, order=2, block_steps=0)
    assert again.two_qubit_gates == tight.two_qubit_gates  # deterministic
    assert again.initial_layout == tight.initial_layout
    assert wide.two_qubit_gates <= int(1.2 * tight.two_qubit_gates)


@native_only
def test_select_term_order_compiles_candidates() -> None:
    terms = heisenberg(4, periodic=True) + [
        ("Z", (q,), 0.3 * (q + 1)) for q in range(4)
    ]
    chain = CompilationTarget.from_coupling_map(4, line(4), native_gates=GATES)
    ch = select_term_order(terms, 1.0, 1e-3, 2, target=chain, compile_options=FAST)
    assert ch.cost == "compile"
    for name, (r, _, c) in ch.table.items():
        got = ch.results[name]
        assert got.steps == r and got.term_order == name
        assert c == got.two_qubit_gates
        fixed = compile_evolution(
            terms, 1.0, chain, steps=r, order=2, term_order=name, **FAST
        )
        assert fixed.two_qubit_gates == c  # same options, same compilation
    best = min(ch.table, key=lambda k: (ch.table[k][2], list(ch.table).index(k)))
    assert ch.term_order == best
    res = compile_evolution(
        terms, 1.0, chain, eps=1e-3, order=2, term_order="auto", **FAST
    )
    assert res.two_qubit_gates == ch.table[ch.term_order][2]
    assert res.error_estimate == ch.error_estimate
    est = select_term_order(terms, 1.0, 1e-3, 2, target=chain, cost="estimate")
    assert est.cost == "estimate" and not est.results
    assert {k: v[:2] for k, v in est.table.items()} == {
        k: v[:2] for k, v in ch.table.items()
    }
    with pytest.raises(ValueError):
        select_term_order(terms, 1.0, 1e-3, 2, cost="exact")
    with pytest.raises(ValueError):
        select_term_order(terms, 1.0, 1e-3, 2, compile_options={"window": 3})


@native_only
def test_select_term_order_extrapolates_long_formulas(monkeypatch) -> None:
    from quspect.hamsim import _compile

    monkeypatch.setattr(_compile, "SELECT_FULL_CX", 10)
    terms = heisenberg(4) + [("Z", (q,), 0.2) for q in range(4)]
    ch = select_term_order(terms, 3.0, 1e-4, 2)
    for name, (r, _, c) in ch.table.items():
        assert r > 4
        two, four = ch.results[name]
        assert (two.steps, four.steps) == (2, 4)
        assert c == _compile._extrapolate(two.two_qubit_gates, four.two_qubit_gates, r)
        assert two.method == four.method
    assert _compile._extrapolate(10, 20, 6) == 30
    assert _compile._extrapolate(10, 21, 7) == 38  # 10 + 5 * 11 / 2 = 37.5


def test_compile_evolution_rejects_bad_arguments() -> None:
    terms = [("ZZ", (0, 1), 1.0)]
    small = CompilationTarget.from_coupling_map(1, [], native_gates=GATES)
    with pytest.raises(ValueError):
        compile_evolution(terms, 1.0, small, steps=1)
    with pytest.raises(ValueError):
        compile_evolution(terms, 1.0)
    with pytest.raises(ValueError):
        compile_evolution(terms, 1.0, steps=1, order=3)
    with pytest.raises(ValueError):
        compile_evolution(terms, 1.0, steps=1, methods=("magic",))
    with pytest.raises(ValueError):
        compile_evolution(terms, 1.0, eps=1e-3, order=4, error_metric="bound")


# ---------------------------------------------------------------- golden digests

GRID_2X3 = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]


def _golden_cases():
    full = None
    line6 = CompilationTarget.from_coupling_map(6, line(6), native_gates=GATES)
    grid = CompilationTarget.from_coupling_map(6, GRID_2X3, native_gates=GATES)
    grid8 = CompilationTarget.from_coupling_map(
        8, _grid_edges(2, 4), native_gates=GATES
    )
    return {
        "hubbard_full_o2": (hubbard_like(), 0.9, full, 2, 2, "input"),
        "hubbard_line_o4": (hubbard_like(), 0.9, line6, 1, 4, "input"),
        "heisenberg_grid_o2": (
            heisenberg(6, periodic=True),
            1.0,
            grid,
            2,
            2,
            "support",
        ),
        "hubbard_grid8_o2": (hubbard_like(), 0.7, grid8, 3, 2, "input"),
    }


def golden_digests() -> dict[str, str]:
    """``{case: sha256 prefix}`` of gates, qubits, parameter bits and layouts."""
    import hashlib

    out = {}
    for name, (terms, t, target, steps, order, term_order) in _golden_cases().items():
        res = compile_evolution(
            terms, t, target, steps=steps, order=order, term_order=term_order
        )
        gates = ";".join(
            ins.gate.name
            + "".join(f",{q}" for q in ins.qubits)
            + "".join(f":{float(p).hex()}" for p in ins.gate.params)
            for ins in res.circuit
        )
        rec = f"{res.method} {res.initial_layout} {res.final_layout} {gates}"
        out[name] = hashlib.sha256(rec.encode()).hexdigest()[:16]
    return out


# 由 ``python tests/hamsim/test_hamsim.py`` 生成。
GOLDEN: dict[str, str] = {
    "hubbard_full_o2": "79e0ebf6030b9d14",
    "hubbard_line_o4": "43357cacfad3e2b6",
    "heisenberg_grid_o2": "b398768aca1185ed",
    "hubbard_grid8_o2": "3d32dc83bcb60154",
}


@native_only
def test_compile_evolution_golden_digests() -> None:
    """Every platform and build produces exactly these compilations."""
    assert golden_digests() == GOLDEN


if __name__ == "__main__":
    import pprint

    pprint.pprint(golden_digests(), sort_dicts=False)
