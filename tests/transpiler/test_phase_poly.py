"""Tests for :class:`quspect.transpiler._phase_poly.PhasePolynomialPass`."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler._phase_poly import PhasePolynomialPass


def _unitary(circuit: QuantumCircuit, n: int) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    psi = np.eye(2**n, dtype=complex).reshape([2] * n + [2**n])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = np.asarray(gate_matrix(ins.gate), dtype=complex).reshape([2] * (2 * k))
        psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(ins.qubits))
    return psi.reshape(2**n, 2**n)


def _phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    overlap = complex(np.vdot(b, a))
    return float(np.max(np.abs(a - overlap / abs(overlap) * b)))


def _two_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if len(i.qubits) == 2)


def _pairs(circuit: QuantumCircuit) -> set[tuple[int, int]]:
    return {(min(i.qubits), max(i.qubits)) for i in circuit if len(i.qubits) == 2}


def _cz_form(circuit: QuantumCircuit) -> QuantumCircuit:
    """Rewrite every cx as h·cz·h (the working basis of the pipeline)."""
    out = QuantumCircuit()
    for ins in circuit:
        assert ins.gate is not None
        if ins.gate.name == "cx":
            c, t = ins.qubits
            out.h(t).cz(c, t).h(t)
        else:
            out.append(ins.gate, list(ins.qubits))
    return out


def _check(circuit: QuantumCircuit, n: int, **kwargs: object) -> QuantumCircuit:
    out = PhasePolynomialPass(**kwargs).run(circuit)  # type: ignore[arg-type]
    assert _phase_distance(_unitary(out, n), _unitary(circuit, n)) < 1e-9
    assert _two_qubit(out) <= _two_qubit(circuit)
    assert _pairs(out) <= _pairs(circuit)
    assert {i.gate.name for i in out if i.gate is not None} <= set(
        {i.gate.name for i in circuit if i.gate is not None} | {"cz", "h", "rz", "x"}
    )
    return out


def test_native_search_is_optimal_for_ccz() -> None:
    from quspect._native_loader import load_core

    core = load_core()
    full = [(a, b) for a in range(3) for b in range(3) if a != b]
    line = [(0, 1), (1, 0), (1, 2), (2, 1)]
    terms = [1, 2, 4, 3, 5, 6, 7]
    out = core._cnot_phase_synth_many(
        [(3, full, [1, 2, 4], terms, 12), (3, line, [1, 2, 4], terms, 12)], 100_000
    )
    assert len(out[0]) == 6
    assert len(out[1]) == 8
    # 上限之内无解
    assert core._cnot_phase_synth_many([(3, full, [1, 2, 4], terms, 5)], 100_000) == [
        None
    ]


def test_toffoli_followed_by_cx_on_a_line() -> None:
    # 线 0-1-2 上的 Toffoli（目标 2）与随后的 cx(0, 1)：逐门综合要 9 个 CZ
    qc = QuantumCircuit().ccx(0, 1, 2).cx(0, 1)
    routed = transpile(
        qc,
        CompilationTarget.from_coupling_map(
            3, [(0, 1), (1, 2)], native_gates={"cz", "h", "rz", "x"}
        ),
        optimization_level=2,
        seed=0,
    ).wire_circuit
    out = _check(routed, 3, edges=[(0, 1), (1, 2)])
    assert _two_qubit(out) <= 8


def test_swap_merges_with_phase_logic() -> None:
    # cx 链 + T + SWAP：SWAP 的 3 个 CX 与相位网络合并
    qc = QuantumCircuit().cx(0, 1).t(1).cx(0, 1).cx(0, 1).cx(1, 0).cx(0, 1)
    out = _check(_cz_form(qc), 2)
    assert _two_qubit(out) < 5


def test_returns_input_when_nothing_improves() -> None:
    qc = _cz_form(QuantumCircuit().cx(0, 1).ry(0.3, 0).ry(0.3, 1).cx(0, 1))
    assert PhasePolynomialPass().run(qc) is qc
    measured = QuantumCircuit().cx(0, 1).cx(0, 1)
    measured.measure([0], [0])
    assert PhasePolynomialPass().run(measured) is measured


@pytest.mark.parametrize("seed", range(12))
def test_random_cnot_phase_circuits(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.choice([3, 4, 5])
    edges = [(i, i + 1) for i in range(n - 1)]
    qc = QuantumCircuit()
    for _ in range(rng.randint(6, 24)):
        r = rng.random()
        if r < 0.45:
            a, b = rng.choice(edges)
            if rng.random() < 0.5:
                a, b = b, a
            qc.cx(a, b)
        elif r < 0.8:
            qc.rz(
                rng.choice([math.pi / 4, -math.pi / 4, math.pi / 2, 0.37]),
                rng.randrange(n),
            )
        elif r < 0.9:
            qc.x(rng.randrange(n))
        else:
            qc.h(rng.randrange(n))
    for q in range(n):  # 所有比特都出现
        qc.rz(0.1 * (q + 1), q)
    for max_qubits in (3, 4, 5):
        _check(_cz_form(qc), n, max_qubits=max_qubits)


def test_level3_uses_the_pass_and_level2_does_not() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).cx(0, 1).ccx(1, 2, 3).cx(2, 3)
    target = CompilationTarget.from_coupling_map(
        4, [(0, 1), (1, 2), (2, 3)], native_gates={"cz", "h", "rz", "x", "s", "t"}
    )
    o2 = transpile(qc, target, optimization_level=2, seed=0)
    o3 = transpile(qc, target, optimization_level=3, seed=0)
    for res in (o2, o3):
        n = 4
        u = _unitary(res.wire_circuit, n)
        assert u.shape == (16, 16)
    assert o3.stats.two_qubit_gates <= o2.stats.two_qubit_gates


# ---------------------------------------------------------------------------
# 21.2：大区域（束搜索）、自由线、原生整轮流程、两个起点


def _replay(n: int, moves: list[tuple[int, int]]) -> tuple[list[int], set[int]]:
    rows = [1 << i for i in range(n)]
    seen = set(rows)
    for c, t in moves:
        rows[t] ^= rows[c]
        seen.add(rows[t])
    return rows, seen


def test_beam_search_finds_short_networks() -> None:
    from quspect._native_loader import load_core

    core = load_core()
    n = 12
    full = [(a, b) for a in range(n) for b in range(n) if a != b]
    # 全连接纠缠层（所有 i < j 的 cx(i, j)）的线性映射：n - 1 个 CX 即可
    rows = [1 << i for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            rows[j] ^= rows[i]
    ((moves, perm),) = core._cnot_phase_synth_batch([(n, full, rows, [], 65, 0, 64, 0)])
    assert len(moves) == n - 1
    assert _replay(n, moves)[0] == rows
    assert perm == list(range(n))
    # 全部两体奇偶（ZZ 层）、恒等映射：远少于逐项的 2·C(n, 2) 个 CX
    n = 8
    full = [(a, b) for a in range(n) for b in range(n) if a != b]
    req = [(1 << i) | (1 << j) for i in range(n) for j in range(i + 1, n)]
    ((moves, _),) = core._cnot_phase_synth_batch(
        [(n, full, [1 << i for i in range(n)], req, 2 * len(req) - 1, 0, 64, 0)]
    )
    end, seen = _replay(n, moves)
    assert end == [1 << i for i in range(n)]
    assert set(req) <= seen
    assert len(moves) <= len(req) + n
    # 上限之内无解
    line = [(0, 1), (1, 0), (1, 2), (2, 1)]
    assert core._cnot_phase_synth_batch(
        [(3, line, [1, 2, 4], [3, 6], 3, 0, 64, 0)]
    ) == [None]


def test_free_wires_absorb_a_permutation() -> None:
    from quspect._native_loader import load_core

    core = load_core()
    full = [(0, 1), (1, 0)]
    # SWAP：固定线要 3 个 CX；两根线都自由时 0 个（输出置换）
    ((moves, perm),) = core._cnot_phase_synth_batch(
        [(2, full, [2, 1], [], 3, 0, 0, 10_000)]
    )
    assert len(moves) == 3
    assert perm == [0, 1]
    for beam in (0, 64):
        ((moves, perm),) = core._cnot_phase_synth_batch(
            [(2, full, [2, 1], [], 3, 3, beam, 10_000)]
        )
        assert moves == []
        assert perm == [1, 0]


def _same_gates(a: QuantumCircuit, b: QuantumCircuit) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        assert x.gate is not None and y.gate is not None
        if x.gate.name != y.gate.name or x.qubits != y.qubits:
            return False
        for p, q in zip(x.gate.params, y.gate.params, strict=True):
            if abs(math.remainder(float(p) - float(q), 2 * math.pi)) > 1e-12:
                return False
    return True


@pytest.mark.parametrize("seed", range(8))
def test_native_rounds_match_the_python_reference(seed: int) -> None:
    rng = random.Random(100 + seed)
    n = rng.choice([4, 5, 7, 9])
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
    qc = QuantumCircuit()
    for _ in range(rng.randint(20, 60)):
        r = rng.random()
        if r < 0.5:
            a, b = rng.choice(edges)
            if rng.random() < 0.5:
                a, b = b, a
            qc.cx(a, b)
        elif r < 0.8:
            qc.rz(rng.choice([math.pi / 4, math.pi / 2, 0.37, -1.1]), rng.randrange(n))
        elif r < 0.9:
            qc.h(rng.randrange(n))
        else:
            qc.x(rng.randrange(n))
    for q in range(n):
        qc.rz(0.1 * (q + 1), q)
    settings = ((4, 50_000), (5, 5_000), (16, 5_000), (16, 5_000, True))
    circ = _cz_form(qc)
    native = PhasePolynomialPass(settings=settings, native=True).run(circ)
    python = PhasePolynomialPass(settings=settings, native=False).run(circ)
    assert _same_gates(native, python)
    assert _phase_distance(_unitary(native, n), _unitary(circ, n)) < 1e-9


def test_large_region_on_a_line() -> None:
    # 8 根线上的全连接 CX 层（路由后只剩线上的比特对）：束搜索给出 7 个 CX
    n = 8
    qc = QuantumCircuit()
    for i in range(n):
        for j in range(i + 1, n):
            qc.cx(i, j)
    full = [(i, j) for i in range(n) for j in range(i + 1, n)]
    out = _check(_cz_form(qc), n, settings=((16, 1_000),), edges=full)
    assert _two_qubit(out) == n - 1


def test_reverse_partition_is_valid() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).cx(0, 1).ccx(1, 2, 3).cx(2, 3).cx(3, 0)
    routed = transpile(
        qc,
        CompilationTarget.from_coupling_map(
            4, [(0, 1), (1, 2), (2, 3)], native_gates={"cz", "h", "rz", "x"}
        ),
        optimization_level=2,
        seed=0,
    ).wire_circuit
    for settings in (((4, 20_000, True),), ((16, 2_000, True), (5, 2_000))):
        _check(routed, 4, settings=settings)


def test_level3_second_start_resynthesizes_before_routing() -> None:
    # RealAmplitudes 式的全连接纠缠层：路由前把它化为 CX 梯子，线上不需要 SWAP
    n = 6
    qc = QuantumCircuit()
    for layer in range(2):
        for q in range(n):
            qc.ry(0.3 + 0.1 * q + layer, q)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
    target = CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates={"cz", "h", "rz", "x", "ry"}
    )
    o2 = transpile(qc, target, optimization_level=2, seed=0)
    o3 = transpile(qc, target, optimization_level=3, seed=0)
    assert o3.stats.two_qubit_gates <= 2 * (n - 1) < o2.stats.two_qubit_gates
