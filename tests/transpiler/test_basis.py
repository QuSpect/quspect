"""Target-basis independence of the transpiler (CZ, CX and IBM bases)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import CompilationTarget, ConsolidateBlocksPass, transpile
from quspect.transpiler._basis import plan_basis, translate_to_target
from quspect.transpiler._passes import DecomposePass

_ONE = ("h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z")
BASES: dict[str, frozenset[str]] = {
    "cz": frozenset({"cz", *_ONE}),
    "cx": frozenset({"cx", *_ONE}),
    "ibm": frozenset({"cx", "rz", "sx", "x"}),
    "cz_sx": frozenset({"cz", "rz", "sx", "x"}),
}


def _line(n: int, basis: str) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASES[basis])
    )


def _grid(rows: int, cols: int, basis: str) -> CompilationTarget:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return CompilationTarget.from_coupling_map(
        rows * cols, edges, native_gates=set(BASES[basis])
    )


def _mixed_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    """Random circuit with cx, cz, cp, ccx, swap and assorted 1q gates."""
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(depth):
        a, b, c = rng.sample(range(n), 3)
        r = rng.randrange(10)
        if r == 0:
            qc.cx(a, b)
        elif r == 1:
            qc.cz(a, b)
        elif r == 2:
            qc.cp(rng.uniform(-math.pi, math.pi), a, b)
        elif r == 3:
            qc.ccx(a, b, c)
        elif r == 4:
            qc.swap(a, b)
        elif r == 5:
            qc.h(a)
        elif r == 6:
            qc.rz(rng.uniform(-math.pi, math.pi), a)
        elif r == 7:
            qc.t(a).sx(b)
        elif r == 8:
            qc.ry(rng.uniform(-math.pi, math.pi), a)
        else:
            qc.crz(rng.uniform(-math.pi, math.pi), a, b)
    return qc


def _embed(state: np.ndarray, layout: list[int], num_physical: int) -> np.ndarray:
    n = len(layout)
    free = [p for p in range(num_physical) if p not in layout]
    zeros = np.zeros(2 ** (num_physical - n), dtype=complex)
    zeros[0] = 1.0
    tensor = np.kron(state, zeros).reshape([2] * num_physical)
    order = [
        layout.index(p) if p in layout else n + free.index(p)
        for p in range(num_physical)
    ]
    return np.transpose(tensor, order).reshape(-1)


def _simulate(circuit: QuantumCircuit, state: np.ndarray, n: int) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    psi = state.reshape([2] * n)
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
        psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(ins.qubits))
    return psi.reshape(-1)


def _assert_valid(
    circ: QuantumCircuit, result: object, target: CompilationTarget
) -> None:
    from quspect.transpiler import TranspileResult

    assert isinstance(result, TranspileResult)
    for instr in result.circuit:
        assert instr.gate is not None
        assert instr.gate.name in target.native_gates, instr.gate.name
        if len(instr.qubits) == 2:
            assert tuple(sorted(instr.qubits)) in target.coupling_graph
    n, big_n = circ.num_qubits, target.num_qubits
    rng = np.random.default_rng(0)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    expected = _embed(_simulate(circ, psi, n), result.final_layout, big_n)
    got = _simulate(result.circuit, _embed(psi, result.initial_layout, big_n), big_n)
    assert abs(abs(np.vdot(expected, got)) - 1.0) < 1e-9


def _two_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if i.is_gate and len(i.qubits) == 2)


# ---------------------------------------------------------------------------
# Equivalence and basis membership, every level and basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", sorted(BASES))
@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_transpile_equivalent_in_every_basis(basis: str, level: int, seed: int) -> None:
    target = _grid(2, 3, basis)
    circ = _mixed_circuit(6, 30, seed)
    result = transpile(circ, target, optimization_level=level, seed=seed)
    _assert_valid(circ, result, target)
    assert result.stats.two_qubit_gates == _two_qubit(result.circuit)


@pytest.mark.parametrize("basis", ["cx", "ibm"])
def test_cz_only_input_has_no_cz_leak(basis: str) -> None:
    """Inputs made of cz gates (hidden-shift-like) are fully translated."""
    circ = QuantumCircuit()
    for q in range(5):
        circ.h(q)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]:
        circ.cz(a, b)
    for q in range(5):
        circ.h(q)
    target = _line(5, basis)
    for level in range(4):
        result = transpile(circ, target, optimization_level=level, seed=3)
        _assert_valid(circ, result, target)
        assert "cz" not in {i.gate.name for i in result.circuit if i.gate}


# ---------------------------------------------------------------------------
# CX / CZ parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cx_basis_two_qubit_count_equals_cz_basis(level: int, seed: int) -> None:
    """CX and CZ differ by local gates only: identical routing and 2q counts."""
    circ = _mixed_circuit(8, 40, seed)
    counts = {}
    for basis in ("cz", "cx"):
        result = transpile(
            circ, _grid(2, 4, basis), optimization_level=level, seed=seed
        )
        counts[basis] = (
            result.stats.two_qubit_gates,
            result.initial_layout,
            result.final_layout,
        )
    # same routing; the final target-basis optimization may only remove gates
    assert counts["cx"][1:] == counts["cz"][1:]
    assert counts["cz"][0] - 2 <= counts["cx"][0] <= counts["cz"][0]


def test_cx_basis_single_qubit_overhead_is_small() -> None:
    """Folding the Hadamards keeps the CX output about as short as the CZ one."""
    circ = _mixed_circuit(8, 60, 7)
    totals = {}
    for basis in ("cz", "cx"):
        result = transpile(circ, _grid(2, 4, basis), optimization_level=2, seed=7)
        totals[basis] = result.stats.total_gates
    assert totals["cx"] <= 1.1 * totals["cz"]


# ---------------------------------------------------------------------------
# Basis planning and translation
# ---------------------------------------------------------------------------


def test_plan_basis_working_sets() -> None:
    assert not plan_basis(BASES["cz"]).needs_translation
    cx = plan_basis(BASES["cx"])
    assert cx.entangler == "cx" and cx.working == BASES["cz"]
    ibm = plan_basis(BASES["ibm"])
    assert ibm.entangler == "cx"
    assert ibm.working == frozenset({"cz", "h", "rz", "sx", "x"})
    assert not plan_basis(set()).needs_translation


@pytest.mark.parametrize(
    "native",
    [{"cx", "h", "t"}, {"cz", "h", "s", "x"}, {"cx", "x", "y", "z"}],
)
def test_plan_basis_rejects_non_universal_single_qubit_sets(native: set[str]) -> None:
    with pytest.raises(ValueError, match="single-qubit"):
        plan_basis(native)


def test_transpile_rejects_target_without_entangler() -> None:
    target = CompilationTarget.from_coupling_map(
        2, [(0, 1)], native_gates={"h", "rz", "x"}
    )
    with pytest.raises(ValueError, match="entangler"):
        transpile(QuantumCircuit().cx(0, 1), target)
    # single-qubit circuits are fine
    res = transpile(QuantumCircuit().h(0).t(0), target)
    assert {i.gate.name for i in res.circuit if i.gate} <= {"h", "rz", "x"}


def test_translate_folds_hadamards() -> None:
    plan = plan_basis(BASES["cx"])
    out = translate_to_target(QuantumCircuit().h(0).cz(0, 1).h(0), plan)
    assert [(i.gate.name, i.qubits) for i in out if i.gate] == [("cx", (1, 0))]
    out = translate_to_target(QuantumCircuit().cz(0, 1), plan)
    assert [i.gate.name for i in out if i.gate].count("cx") == 1
    assert len(out.instructions) == 3


def test_translate_ibm_single_qubit_runs() -> None:
    plan = plan_basis(BASES["ibm"])
    circ = QuantumCircuit().h(0).s(0).t(1).cz(0, 1).h(1).x(0)
    out = translate_to_target(circ, plan)
    assert {i.gate.name for i in out if i.gate} <= BASES["ibm"]
    _assert_unitary_equal(circ, out, 2)


def _unitary(circ: QuantumCircuit, n: int) -> np.ndarray:
    cols = []
    for k in range(2**n):
        e = np.zeros(2**n, dtype=complex)
        e[k] = 1.0
        cols.append(_simulate(circ, e, n))
    return np.array(cols).T


def _assert_unitary_equal(a: QuantumCircuit, b: QuantumCircuit, n: int) -> None:
    ua, ub = _unitary(a, n), _unitary(b, n)
    k = int(np.argmax(np.abs(ua)))
    phase = ua.flat[k] / ub.flat[k]
    assert np.max(np.abs(ua - phase * ub)) < 1e-9


# ---------------------------------------------------------------------------
# DecomposePass and ConsolidateBlocksPass on non-CZ targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["cx", "ibm", "cz_sx"])
def test_decompose_pass_reaches_every_basis(basis: str) -> None:
    circ = (
        QuantumCircuit()
        .h(0)
        .cz(0, 1)
        .cx(1, 2)
        .cp(0.3, 0, 2)
        .ccx(0, 1, 2)
        .swap(1, 2)
        .rx(0.4, 1)
        .ry(-0.7, 2)
        .u3(0.1, 0.2, 0.3, 0)
        .sx(1)
        .sxdg(2)
        .y(0)
    )
    target = _line(3, basis)
    out = DecomposePass(target).run(circ)
    assert {i.gate.name for i in out if i.gate} <= BASES[basis]
    _assert_unitary_equal(circ, out, 3)


def test_decompose_pass_raises_for_inexpressible_gates() -> None:
    target = CompilationTarget.from_coupling_map(
        2, [(0, 1)], native_gates={"cx", "h", "t"}
    )
    with pytest.raises(ValueError, match="Cannot express"):
        DecomposePass(target).run(QuantumCircuit().rz(0.3, 0))
    no_ent = CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates={"h", "rz"})
    with pytest.raises(ValueError, match="neither cx nor cz"):
        DecomposePass(no_ent).run(QuantumCircuit().cp(0.2, 0, 1))


def test_cz_to_cx_rule() -> None:
    from quspect.gates.standard import CZ
    from quspect.ir.compilation import decompose_gate

    seq = decompose_gate(CZ, (0, 1))
    assert [g.name for g, _ in seq] == ["h", "cx", "h"]


@pytest.mark.parametrize("basis", ["cx", "ibm"])
def test_consolidate_emits_target_entangler(basis: str) -> None:
    rng = random.Random(4)
    circ = QuantumCircuit()
    for _ in range(12):
        circ.cx(0, 1).rz(rng.uniform(-3, 3), 0).ry(rng.uniform(-3, 3), 1)
    target = _line(2, basis)
    lowered = DecomposePass(target).run(circ)
    out = ConsolidateBlocksPass(target).run(lowered)
    assert {i.gate.name for i in out if i.gate} <= BASES[basis]
    assert _two_qubit(out) <= 3
    _assert_unitary_equal(circ, out, 2)
