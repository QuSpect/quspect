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
    ladder_circuit,
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


@native_only
def test_select_term_order_and_auto() -> None:
    terms = heisenberg(4, periodic=True) + [
        ("Z", (q,), 0.3 * (q + 1)) for q in range(4)
    ]
    ch = select_term_order(terms, 1.0, 1e-3, 2)
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
    res = compile_evolution(terms, 1.0, eps=1e-3, order=2, term_order="auto")
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
    on_line = select_term_order(terms, 1.0, 1e-3, 2, target=chain)
    assert {k: v[:2] for k, v in on_line.table.items()} == {
        k: v[:2] for k, v in ch.table.items()
    }
    lres = compile_evolution(terms, 1.0, chain, eps=1e-3, order=2, term_order="auto")
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
    "hubbard_line_o4": "fa2fcb3342db5661",
    "heisenberg_grid_o2": "b398768aca1185ed",
}


@native_only
def test_compile_evolution_golden_digests() -> None:
    """Every platform and build produces exactly these compilations."""
    assert golden_digests() == GOLDEN


if __name__ == "__main__":
    import pprint

    pprint.pprint(golden_digests(), sort_dicts=False)
