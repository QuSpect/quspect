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
    if level == 3:
        # 第 3 级最后按目标门集里的完成结果取优（再编译一遍的输入是目标门集的
        # 物理线路，深度并列时的取舍也看单比特门），两种门集可以选到不同的路由
        assert abs(counts["cx"][0] - counts["cz"][0]) <= 2
        return
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


#: Universal single-qubit sets without a Z rotation (``rz`` / ``p`` / ``u1``),
#: and ``{u2, u1}``: each used to be refused or to fail on some runs.
_NO_RZ_BASES: dict[str, frozenset[str]] = {
    "u3": frozenset({"u3"}),
    "rx_ry": frozenset({"rx", "ry"}),
    "u2_u1": frozenset({"u2", "u1"}),
    "u2": frozenset({"u2"}),
}


@pytest.mark.parametrize("one_qubit", sorted(_NO_RZ_BASES))
def test_plan_basis_accepts_universal_sets_without_rz(one_qubit: str) -> None:
    native = _NO_RZ_BASES[one_qubit] | {"cz"}
    plan = plan_basis(native)
    assert plan.one_qubit == _NO_RZ_BASES[one_qubit]


@pytest.mark.parametrize("gate", ["t", "s", "z", "h", "sx"])
def test_u3_only_target_compiles_single_gates(gate: str) -> None:
    """``{u3}`` is universal: a diagonal run is ``u3(0, 0, λ)``."""
    target = CompilationTarget.from_coupling_map(1, [], native_gates={"u3"})
    circ = getattr(QuantumCircuit(), gate)(0)
    for level in range(4):
        result = transpile(circ, target, optimization_level=level)
        assert [i.gate.name for i in result.circuit if i.gate] == ["u3"]
        _assert_valid(circ, result, target)


def _qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit()
    for t in range(n):
        qc.h(t)
        for c in range(t + 1, n):
            qc.cp(math.pi / 2 ** (c - t), c, t)
    return qc


@pytest.mark.parametrize("one_qubit", sorted(_NO_RZ_BASES))
@pytest.mark.parametrize("entangler", ["cz", "cx"])
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_transpile_equivalent_without_rz(
    one_qubit: str, entangler: str, level: int
) -> None:
    native = set(_NO_RZ_BASES[one_qubit]) | {entangler}
    line = CompilationTarget.from_coupling_map(
        4, [(0, 1), (1, 2), (2, 3)], native_gates=native
    )
    for circ in (
        QuantumCircuit().cp(0.3, 0, 1),
        QuantumCircuit().h(0).cx(0, 1),
        _qft(4),
        _mixed_circuit(4, 20, level),
    ):
        result = transpile(circ, line, optimization_level=level, seed=0)
        _assert_valid(circ, result, line)


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


def _working_circuit(
    seed: int, working: frozenset[str], n: int, size: int
) -> QuantumCircuit:
    """Random circuit in a plan's working gates plus measurements / barriers."""
    from quspect.circuit.parameter import Parameter

    rng = random.Random(seed)
    one = sorted(g for g in working if g not in ("cz", "rz"))
    theta = Parameter("t")
    qc = QuantumCircuit()
    for _ in range(size):
        a, b = rng.sample(range(n), 2)
        r = rng.randrange(12)
        if r < 4:
            qc.cz(a, b)
        elif r < 6:
            qc.rz(rng.choice([rng.uniform(-4, 4), math.pi / 2, 0.0, -0.0]), a)
        elif r < 9:
            getattr(qc, rng.choice(one))(a)
        elif r == 9:
            qc.measure([a], [b])
        elif r == 10:
            qc.barrier(a, b) if rng.random() < 0.7 else qc.barrier()
        elif "rz" in working:
            qc.rz(theta, a)  # symbolic native gate: kept as is
    return qc


@pytest.mark.parametrize("basis", ["cx", "ibm", "cz_sx"])
@pytest.mark.parametrize("seed", range(10))
def test_native_translation_matches_python(
    basis: str, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The C++ translate_to_target follows the Python body instruction by instruction.

    The C++ products run on the deterministic numerics layer and the Python
    body on numpy, so rotation angles may differ in the last bits (not bit
    for bit); gates, operands and the instructions reused from the input are
    identical.
    """
    from quspect.transpiler import _basis

    plan = plan_basis(BASES[basis])
    qc = _working_circuit(seed, plan.working, 2 + seed % 5, 40 + 25 * seed)
    monkeypatch.setenv(_basis.TRANSLATE_DRIVER_ENV, "python")
    ref = translate_to_target(qc, plan)
    monkeypatch.setenv(_basis.TRANSLATE_DRIVER_ENV, "auto")
    if _basis._translate_native(qc, plan) is None:
        pytest.skip("native translation not available")
    got = translate_to_target(qc, plan)

    def key(ins: object) -> tuple[object, ...]:
        gate = ins.gate  # type: ignore[attr-defined]
        name = None if gate is None else gate.name
        return (ins.type, ins.qubits, ins.clbits, name)  # type: ignore[attr-defined]

    assert [key(i) for i in got.instructions] == [key(i) for i in ref.instructions]
    for a, b in zip(got.instructions, ref.instructions, strict=True):
        if a.gate is None or not a.gate.params:
            continue
        for x, y in zip(a.gate.params, b.gate.params, strict=True):
            if isinstance(x, float):
                assert x == pytest.approx(y, abs=1e-12)
            else:
                assert x == y
    reused = {id(i) for i in qc.instructions}
    assert [id(i) in reused for i in got.instructions] == [
        id(i) in reused for i in ref.instructions
    ]


def test_native_translation_leaves_errors_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quspect.circuit.parameter import Parameter
    from quspect.transpiler import _basis

    plan = plan_basis(BASES["ibm"])
    qc = QuantumCircuit().h(0).cz(0, 1).ry(Parameter("p"), 1)
    assert _basis._translate_native(qc, plan) is None
    for driver in ("python", "auto"):
        monkeypatch.setenv(_basis.TRANSLATE_DRIVER_ENV, driver)
        with pytest.raises(ValueError, match="not in the target basis"):
            translate_to_target(qc, plan)
    monkeypatch.setenv(_basis.TRANSLATE_DRIVER_ENV, "fast")
    with pytest.raises(ValueError, match=_basis.TRANSLATE_DRIVER_ENV):
        translate_to_target(QuantumCircuit().h(0).cz(0, 1), plan)
