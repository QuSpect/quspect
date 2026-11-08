"""Tests for the numerical three-qubit instantiation search (level 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect._native_loader import load_core
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler._instantiate import (
    NumericThreeQubitBlockPass,
    _perm_matrix,
    instantiate_three_qubit,
)
from quspect.transpiler._three_qubit import _block_unitary_in, _distance

pytestmark = pytest.mark.skipif(
    load_core() is None, reason="extension module not built"
)

_BASIS = {"cz", "rz", "sx", "x"}


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


def _layout_matrix(layout: list[int], n_phys: int, n_log: int) -> np.ndarray:
    """|x⟩ (logical, qubit 0 = high bit) ↦ physical basis state, other
    physical qubits |0⟩ (an isometry)."""
    m = np.zeros((2**n_phys, 2**n_log))
    for x in range(2**n_log):
        y = 0
        for i in range(n_log):
            if (x >> (n_log - 1 - i)) & 1:
                y |= 1 << (n_phys - 1 - layout[i])
        m[y, x] = 1.0
    return m


def _assert_equivalent(circuit: QuantumCircuit, res: object, n_phys: int) -> None:
    n = circuit.num_qubits
    u = _unitary(circuit, n)
    v = _unitary(res.circuit, n_phys) if res.circuit.num_qubits == n_phys else None  # type: ignore[attr-defined]
    if v is None:
        padded = QuantumCircuit()
        for ins in res.circuit:  # type: ignore[attr-defined]
            padded.append(ins.gate, list(ins.qubits))
        padded.id(n_phys - 1)
        v = _unitary(padded, n_phys)
    a = _layout_matrix(res.initial_layout, n_phys, n)  # type: ignore[attr-defined]
    b = _layout_matrix(res.final_layout, n_phys, n)  # type: ignore[attr-defined]
    got = v @ a
    want = b @ u
    overlap = complex(np.vdot(want, got))
    assert abs(overlap) > 0
    assert np.max(np.abs(got - overlap / abs(overlap) * want)) < 1e-8


def _miller() -> QuantumCircuit:
    # RevLib miller_11 的骨架：两个 Toffoli 夹着 CX
    qc = QuantumCircuit()
    qc.cx(2, 1).ccx(0, 1, 2).cx(1, 0).ccx(2, 0, 1).cx(0, 2).t(1).h(0).ccx(1, 2, 0)
    return qc


def test_finds_cx_between_ends_with_free_layout() -> None:
    qc = QuantumCircuit().cx(0, 1).cx(1, 2).cx(0, 1).cx(1, 2)  # = cx(0, 2)
    found = instantiate_three_qubit(qc, {(0, 1), (1, 2)}, 5)
    assert found is not None and found.num_cz == 1
    # 发出的门等于 P_end · U · P_startᵀ
    u = _block_unitary_in([0, 1, 2], list(qc))
    v = _block_unitary_in([0, 1, 2], found.instructions)
    target = _perm_matrix(found.end) @ u @ _perm_matrix(found.start).T
    assert _distance(target, v) < 1e-9
    for ins in found.instructions:
        if len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in {(0, 1), (1, 2)}


def test_fixed_layout_and_bound() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2)
    assert instantiate_three_qubit(qc, {(0, 1), (1, 2), (0, 2)}, 4) is None
    found = instantiate_three_qubit(
        qc, {(0, 1), (1, 2), (0, 2)}, 8, fixed_layout=True, permute_output=False
    )
    assert found is not None and 6 <= found.num_cz <= 8
    assert found.start == found.end == (0, 1, 2)
    u = _block_unitary_in([0, 1, 2], list(qc))
    assert _distance(u, _block_unitary_in([0, 1, 2], found.instructions)) < 1e-9


def test_search_is_deterministic() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).cx(0, 1)  # 半加器
    a = instantiate_three_qubit(qc, {(0, 1), (1, 2)}, 8, budget=10000)
    b = instantiate_three_qubit(qc, {(0, 1), (1, 2)}, 8, budget=10000)
    assert a is not None and b is not None
    assert a.start == b.start and a.end == b.end
    pa = [
        (i.gate.name, i.qubits, tuple(map(float.hex, i.gate.params)))
        for i in a.instructions
    ]  # type: ignore[union-attr]
    pb = [
        (i.gate.name, i.qubits, tuple(map(float.hex, i.gate.params)))
        for i in b.instructions
    ]  # type: ignore[union-attr]
    assert pa == pb


@pytest.mark.parametrize("edges", [[(0, 1), (1, 2)], [(0, 1), (1, 2), (0, 2)]])
def test_level3_three_qubit_circuit(edges: list[tuple[int, int]]) -> None:
    qc = _miller()
    target = CompilationTarget.from_coupling_map(3, edges, native_gates=_BASIS)
    o2 = transpile(qc, target, optimization_level=2, seed=0)
    o3 = transpile(qc, target, optimization_level=3, seed=0)
    assert o3.stats.selected == "numeric3q"
    assert o3.stats.two_qubit_gates < o2.stats.two_qubit_gates
    _assert_equivalent(qc, o3, 3)
    assert {i.gate.name for i in o3.circuit} <= _BASIS  # type: ignore[union-attr]


def test_level3_three_qubit_on_larger_device() -> None:
    qc = _miller()
    target = CompilationTarget.from_coupling_map(
        5, [(0, 1), (1, 2), (2, 3), (3, 4)], native_gates=_BASIS
    )
    o3 = transpile(qc, target, optimization_level=3, seed=0)
    _assert_equivalent(qc, o3, 5)
    for ins in o3.circuit:
        if len(ins.qubits) == 2:
            assert abs(ins.qubits[0] - ins.qubits[1]) == 1


def test_block_pass_reduces_redundant_block() -> None:
    qc = QuantumCircuit().h(3).cz(3, 0)
    for _ in range(3):
        qc.cz(0, 1).rx(0.3, 1).cz(1, 2).ry(0.2, 0).cz(0, 1)
    out = NumericThreeQubitBlockPass({(0, 1), (1, 2), (0, 3)}, min_cz=4).run(qc)
    n_in = sum(1 for i in qc if len(i.qubits) == 2)
    n_out = sum(1 for i in out if len(i.qubits) == 2)
    assert n_out < n_in
    assert all(
        tuple(sorted(i.qubits)) in {(0, 1), (1, 2), (0, 3)}
        for i in out
        if len(i.qubits) == 2
    )
    u, v = _unitary(qc, 4), _unitary(out, 4)
    overlap = complex(np.vdot(u, v))
    assert np.max(np.abs(v - overlap / abs(overlap) * u)) < 1e-9


def test_repass_composes_layouts() -> None:
    # QAOA 式的环上 ZZ 层：第二遍编译（输入 = 路由后的物理线路）的布局复合正确
    from quspect.transpiler._transpile import _recompile

    n = 6
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for i in range(n):
        for j in (i + 1, i + 3):
            if j < n:
                qc.cx(i, j).rz(0.4 + 0.1 * i, j).cx(i, j)
    for q in range(n):
        qc.rx(0.7, q)
    target = CompilationTarget.from_coupling_map(
        n, [(i, (i + 1) % n) for i in range(n)], native_gates=_BASIS
    )
    res = transpile(qc, target, optimization_level=3, seed=1)
    _assert_equivalent(qc, res, n)
    kwargs = {
        "optimization_level": 3,
        "seed": 1,
        "initial_layout": None,
        "iter_num": 3,
        "layout_trials": None,
        "swap_trials": None,
        "max_evaluations": None,
        "work_limit": None,
        "time_limit": None,
        "exact_layout": True,
        "elide_permutations": True,
        "tolerance": 0.0,
        "depth_weight": 0.0,
    }
    again = _recompile(res, qc, target, kwargs)
    assert again is not None
    _assert_equivalent(qc, again, n)
    assert sorted(again.circuit.active_qubits) == sorted(
        again.wire_layout[w] for w in again.wire_circuit.active_qubits
    )


def test_numeric_disabled_below_level3() -> None:
    qc = _miller()
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (1, 2)], native_gates=_BASIS
    )
    assert (
        transpile(qc, target, optimization_level=2, seed=0).stats.selected
        != "numeric3q"
    )
    assert math.isfinite(
        transpile(qc, target, optimization_level=1, seed=0).stats.seconds
    )


def test_block_pass_free_output_on_terminal_block() -> None:
    qc = QuantumCircuit().h(3).cz(3, 0).h(0).barrier(0, 3)
    qc.cx(0, 1).cx(1, 0).cx(0, 1).cz(1, 2).rz(0.3, 2)  # 末端块 = cz(1, 2)·swap(0, 1)
    coupled = {(0, 1), (1, 2), (0, 2), (0, 3)}
    fixed = NumericThreeQubitBlockPass(coupled, min_cz=4).run(qc)
    blk = NumericThreeQubitBlockPass(coupled, free_output=True, min_cz=4)
    out = blk.run(qc)
    n2 = [
        sum(1 for i in c if i.is_gate and len(i.qubits) == 2) for c in (qc, fixed, out)
    ]
    assert n2[2] < n2[1] <= n2[0]
    assert n2[2] == 2  # cz(3, 0) + 一个 CZ
    assert blk.wire_moves and set(blk.wire_moves) == set(blk.wire_moves.values())
    # out 之后把线 a 的状态搬到 wire_moves[a]，等于原线路
    perm = list(range(4))
    for a, b in blk.wire_moves.items():
        perm[a] = b
    p = np.zeros((16, 16))
    for x in range(16):
        y = 0
        for q in range(4):
            if (x >> (3 - q)) & 1:
                y |= 1 << (3 - perm[q])
        p[y, x] = 1.0

    def gates(c: QuantumCircuit) -> QuantumCircuit:
        g = QuantumCircuit()
        for ins in c:
            if ins.is_gate:
                g.append(ins.gate, list(ins.qubits))  # type: ignore[arg-type]
        return g

    u, v = _unitary(gates(qc), 4), _unitary(gates(out), 4)
    want = p @ u
    overlap = complex(np.vdot(want, v))
    assert np.max(np.abs(v - overlap / abs(overlap) * want)) < 1e-9


@pytest.mark.parametrize("mode", ["inside", "end"])
def test_level3_numeric_blocks_equivalent(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quspect.transpiler._transpile as transpile_module

    monkeypatch.setattr(transpile_module, "_NUMERIC_BLOCKS_MODE", mode)
    qc = QuantumCircuit()
    for _ in range(2):
        qc.ccx(0, 1, 2).cx(2, 3).ccx(1, 2, 3).h(0).cx(3, 0).t(1)
    target = CompilationTarget.from_coupling_map(
        4, [(0, 1), (1, 2), (2, 3)], native_gates=_BASIS
    )
    res = transpile(qc, target, optimization_level=3, seed=0)
    _assert_equivalent(qc, res, 4)
    o2 = transpile(qc, target, optimization_level=2, seed=0)
    assert res.stats.two_qubit_gates <= o2.stats.two_qubit_gates


@pytest.mark.parametrize("seed", range(4))
def test_level3_random_circuits_equivalent(seed: int) -> None:
    # 三比特块（含末端块的输出置换）、再编译与普通门级候选一起作用后仍等价
    import random

    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(14):
        a, b, c = rng.sample(range(5), 3)
        kind = rng.randrange(4)
        if kind == 0:
            qc.ccx(a, b, c)
        elif kind == 1:
            qc.cx(a, b).rz(rng.uniform(-3, 3), b).cx(a, b)
        elif kind == 2:
            qc.h(a).cx(a, c).t(c)
        else:
            qc.swap(a, b).cz(b, c)
    target = CompilationTarget.from_coupling_map(
        5, [(0, 1), (1, 2), (2, 3), (3, 4)], native_gates=_BASIS
    )
    res = transpile(qc, target, optimization_level=3, seed=seed)
    _assert_equivalent(qc, res, 5)
