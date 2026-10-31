"""Depth-oriented reordering of commuting gates (``schedule_for_depth``)."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler._schedule import (
    _X,
    _Z,
    _axes,
    _best_order,
    _matrix_axes,
    schedule_for_depth,
)
from quspect.transpiler._transpile import _depth


def _unitary(circuit: QuantumCircuit, n: int) -> np.ndarray:
    u = np.eye(2**n, dtype=complex).reshape([2] * n + [2**n])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = np.asarray(gate_matrix(ins.gate)).reshape([2] * (2 * k))
        u = np.tensordot(g, u, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        u = np.moveaxis(u, list(range(k)), list(ins.qubits))
    return u.reshape(2**n, 2**n)


def _random_circuit(seed: int, n: int = 4, size: int = 60) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(size):
        a, b = rng.sample(range(n), 2)
        r = rng.randrange(10)
        if r < 2:
            qc.cx(a, b)
        elif r < 4:
            qc.cz(a, b)
        elif r < 6:
            qc.rz(rng.uniform(-3, 3), a)
        elif r == 6:
            qc.h(a)
        elif r == 7:
            qc.sx(a)
        elif r == 8:
            qc.x(a)
        else:
            qc.cp(rng.uniform(-3, 3), a, b)
    return qc


def _depths(c: QuantumCircuit) -> tuple[int, int]:
    return _depth(c, two_qubit_only=False), _depth(c, two_qubit_only=True)


@pytest.mark.parametrize("seed", range(12))
def test_random_circuits_stay_equivalent_and_never_deeper(seed: int) -> None:
    qc = _random_circuit(seed)
    out = schedule_for_depth(qc)
    assert sorted(map(id, out.instructions)) == sorted(map(id, qc.instructions))
    np.testing.assert_allclose(_unitary(out, 4), _unitary(qc, 4), atol=1e-10)
    d0, d20 = _depths(qc)
    d, d2 = _depths(out)
    assert d2 <= d20
    assert (d, d2) <= (d0, d20)


def test_native_and_python_orders_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuits = [_random_circuit(s, n=6, size=150) for s in range(8)]
    native = [list(schedule_for_depth(c)) for c in circuits]
    monkeypatch.setenv("QUSPECT_ROUTING_BACKEND", "python")
    python = [list(schedule_for_depth(c)) for c in circuits]
    assert native == python


def test_commuting_cz_ring_is_edge_coloured() -> None:
    qc = QuantumCircuit().cz(0, 1).cz(1, 2).cz(2, 3).cz(3, 0)
    out = schedule_for_depth(qc)
    assert _depths(qc) == (4, 4)
    assert _depths(out) == (2, 2)


def test_rotation_moves_through_cx_control() -> None:
    qc = QuantumCircuit().cx(1, 2).cx(0, 1).rz(0.3, 0).cx(0, 3)
    out = schedule_for_depth(qc)
    assert [(i.gate.name, i.qubits) for i in out if i.gate] == [
        ("cx", (1, 2)),
        ("cx", (0, 3)),
        ("cx", (0, 1)),
        ("rz", (0,)),
    ]
    assert _depth(out, two_qubit_only=False) == 3


def test_non_commuting_order_is_kept() -> None:
    qc = QuantumCircuit().cx(0, 1).h(1).cx(1, 2).h(2).cx(2, 0)
    assert schedule_for_depth(qc) is qc


def test_two_qubit_depth_is_never_traded_for_total_depth() -> None:
    # long single-qubit chains on the hub: the total-depth schedule would
    # interleave them with the CX of other qubits
    rng = random.Random(3)
    qc = QuantumCircuit()
    for _ in range(4):
        for t in range(1, 5):
            qc.cx(0, t)
            for _ in range(3):
                qc.rz(rng.uniform(-1, 1), t).sx(t)
    out = schedule_for_depth(qc)
    assert _depth(out, two_qubit_only=True) <= _depth(qc, two_qubit_only=True)
    np.testing.assert_allclose(_unitary(out, 5), _unitary(qc, 5), atol=1e-10)


def test_measurements_and_barriers_are_fences() -> None:
    qc = QuantumCircuit().cz(0, 1).rz(0.2, 0).barrier().cz(0, 2).cz(1, 2)
    qc.measure([0], [0]).measure([1], [0]).rz(0.1, 2)
    out = schedule_for_depth(qc)
    names = [(i.type.name, i.qubits) for i in out]
    # nothing crosses the barrier
    k = next(i for i, ins in enumerate(out) if ins.is_barrier)
    assert {q for ins in list(out)[:k] for q in ins.qubits} == {0, 1}
    # the two measurements into the same classical bit keep their order
    meas = [ins.qubits for ins in out if ins.is_measure]
    assert meas == [(0,), (1,)]
    assert len(names) == len(qc.instructions)


def test_short_circuits_are_returned_unchanged() -> None:
    qc = QuantumCircuit().cz(0, 1).cz(1, 2)
    assert schedule_for_depth(qc) is qc


def test_gate_axes_from_names_and_matrices() -> None:
    qc = QuantumCircuit().cx(0, 1).cp(0.3, 0, 1).rx(0.2, 0).ry(0.1, 0).h(0)
    cache: dict[Any, tuple[int, ...]] = {}
    assert [_axes(i, cache) for i in qc] == [(_Z, _X), (_Z, _Z), (_X,), (2,), (-1,)]
    # a gate known only by its matrix: CRX commutes about Z (control), X (target)
    crx = QuantumCircuit().crx(0.4, 0, 1).instructions[0]
    assert crx.gate is not None
    assert _matrix_axes(crx.gate, 2) == (_Z, _X)
    assert _axes(crx, cache) == (_Z, _X)
    assert _axes(crx, cache) == (_Z, _X)  # cached


def test_reference_scheduler_on_explicit_problem() -> None:
    # cz ring as an explicit problem: two layers
    res = [[0, 1], [1, 2], [2, 3], [3, 0]]
    order = _best_order(res, [[_Z, _Z]] * 4, [True] * 4, [True] * 4)
    assert order == [0, 2, 1, 3]


def test_transpile_reorders_for_depth() -> None:
    target = CompilationTarget.from_coupling_map(
        8,
        [(i, (i + 1) % 8) for i in range(8)],
        native_gates={"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"},
    )
    qc = QuantumCircuit()
    for layer in range(3):
        for q in range(8):
            qc.ry(0.1 * (q + layer), q).rz(0.2 * q, q)
        for q in range(8):
            qc.cx(q, (q + 1) % 8)
    for level in (1, 2, 3):
        res = transpile(qc, target, optimization_level=level, seed=0)
        assert res.stats.depth == _depth(res.wire_circuit, two_qubit_only=False)
        assert math.isfinite(res.stats.seconds)
    o2 = transpile(qc, target, optimization_level=2, seed=0)
    # the circular entangling ladder: same gates, much shallower than the
    # gate order the post-routing passes leave behind
    assert o2.stats.depth < 3 * 8 * 3


def test_zz_rotations_commute_as_blocks() -> None:
    zz = QuantumCircuit().cx(0, 1).rz(0.3, 1).cx(0, 1)
    zz.cx(1, 2).rz(0.2, 2).cx(1, 2).cx(3, 2).rz(0.1, 2).cx(3, 2)
    # gate by gate nothing commutes (CX targets against CX controls)
    assert _depths(schedule_for_depth(zz, blocks=False)) == (9, 6)
    out = schedule_for_depth(zz)
    assert _depths(out) == (6, 4)
    np.testing.assert_allclose(_unitary(out, 4), _unitary(zz, 4), atol=1e-10)


@pytest.mark.parametrize("seed", range(20))
def test_block_partition_is_a_valid_order(seed: int) -> None:
    from quspect.transpiler._schedule import _blocks

    qc = _random_circuit(seed, n=4, size=40)
    instrs = qc.instructions
    nodes = _blocks(instrs)
    assert sorted(i for n in nodes for i in n) == list(range(len(instrs)))
    for members in nodes:
        assert members == sorted(members)
        qubits = {q for i in members for q in instrs[i].qubits}
        assert len(qubits) <= 2
    flat = QuantumCircuit._from_instruction_snapshot(
        [instrs[i] for n in nodes for i in n]
    )
    np.testing.assert_allclose(_unitary(flat, 4), _unitary(qc, 4), atol=1e-10)
