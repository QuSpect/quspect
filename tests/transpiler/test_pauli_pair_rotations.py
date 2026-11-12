"""``rxx``/``ryy``/``rzz`` through the transpiler.

They are lowered exactly (``rzz = cx·rz·cx``), recognised as diagonal in the
``X``/``Y``/``Z`` basis by the commutation rules, kept whole by the
commutation-aware routing, and compiled to every supported target basis.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates.parameterized import RXX, RYY, RZZ
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler._highlevel import HighLevelOptimizePass, basis_on
from quspect.transpiler._passes import DecomposePass
from tests.transpiler.test_transpile import _embed, _simulate

BASES = {
    "cz": {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"},
    "cx": {"cx", "rz", "sx", "x"},
    "rx": {"cz", "rx", "rz"},
}


def _ising_trotter(n: int, steps: int, seed: int) -> QuantumCircuit:
    """XYZ-Heisenberg / transverse-field Trotter steps on a ring."""
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for _ in range(steps):
        for i in range(n):
            j = (i + 1) % n
            qc.rxx(rng.uniform(-1, 1), i, j)
            qc.ryy(rng.uniform(-1, 1), j, i)
            qc.rzz(rng.uniform(-1, 1), i, j)
        for q in range(n):
            qc.rx(rng.uniform(-1, 1), q)
    return qc


def _qaoa_rzz(n: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    edges = [(i, (i + 1) % n) for i in range(n)] + [
        (i, i + n // 2) for i in range(n // 2)
    ]
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for _ in range(2):
        gamma, beta = rng.uniform(0, math.pi), rng.uniform(0, math.pi)
        for u, v in edges:
            qc.rzz(2 * gamma, u, v)
        for q in range(n):
            qc.rx(2 * beta, q)
    return qc


def _line(n: int, basis: str) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASES[basis])
    )


def _assert_equivalent(circ: QuantumCircuit, result, target, basis: str) -> None:  # type: ignore[no-untyped-def]
    for instr in result.circuit:
        assert instr.gate is not None
        assert instr.gate.name in BASES[basis]
        if len(instr.qubits) == 2:
            assert tuple(sorted(instr.qubits)) in target.coupling_graph
    n, big_n = circ.num_qubits, target.num_qubits
    rng = np.random.default_rng(1)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    expected = _embed(_simulate(circ, psi, n), result.final_layout, big_n)
    got = _simulate(result.circuit, _embed(psi, result.initial_layout, big_n), big_n)
    assert abs(abs(np.vdot(expected, got)) - 1.0) < 1e-9


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("basis", sorted(BASES))
def test_trotter_circuit_transpiles_exactly(level: int, basis: str) -> None:
    circ = _ising_trotter(4, steps=2, seed=level)
    target = _line(5, basis)
    result = transpile(circ, target, optimization_level=level, seed=0)
    _assert_equivalent(circ, result, target, basis)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_qaoa_rzz_matches_its_cx_form(level: int) -> None:
    """Native rzz never compiles worse than the cx·rz·cx it replaces."""
    circ = _qaoa_rzz(6, seed=4)
    lowered = DecomposePass(
        CompilationTarget(num_qubits=0, native_gates={"cx", "rz", "h", "rx"})
    ).run(circ)
    target = _line(6, "cz")
    native = transpile(circ, target, optimization_level=level, seed=0)
    expanded = transpile(lowered, target, optimization_level=level, seed=0)
    _assert_equivalent(circ, native, target, "cz")

    def two_qubit(c: QuantumCircuit) -> int:
        return sum(1 for i in c if i.is_gate and len(i.qubits) == 2)

    assert two_qubit(native.circuit) <= two_qubit(expanded.circuit)


def test_decompose_pass_lowers_to_the_target() -> None:
    circ = QuantumCircuit().rxx(0.3, 0, 1).ryy(-0.4, 1, 2).rzz(0.5, 2, 0)
    target = CompilationTarget(num_qubits=0, native_gates={"cx", "rz", "sx", "x"})
    out = DecomposePass(target).run(circ)
    assert {i.gate.name for i in out if i.gate} <= {"cx", "rz", "sx", "x"}
    assert sum(1 for i in out if i.gate and i.gate.name == "cx") == 6


@pytest.mark.parametrize(("gate", "expected"), [(RXX, "X"), (RYY, "Y"), (RZZ, "Z")])
def test_basis_on_both_qubits(gate, expected: str) -> None:  # type: ignore[no-untyped-def]
    g = gate.with_params(0.3)
    assert basis_on(g, (3, 1), 3) == expected
    assert basis_on(g, (3, 1), 1) == expected


def test_highlevel_pass_cancels_cx_across_commuting_rzz() -> None:
    """cx(0,1) · rzz(0,2) · cx(0,1): rzz is Z-diagonal on the control."""
    circ = QuantumCircuit().cx(0, 1).rzz(0.4, 0, 2).cx(0, 1)
    out = HighLevelOptimizePass().run(circ)
    assert [(i.gate.name, i.qubits) for i in out if i.gate] == [("rzz", (0, 2))]


def test_highlevel_pass_does_not_commute_across_other_bases() -> None:
    circ = QuantumCircuit().cx(0, 1).rxx(0.4, 0, 2).cx(0, 1)
    out = HighLevelOptimizePass().run(circ)
    assert [i.gate.name for i in out if i.gate] == ["cx", "rxx", "cx"]


def test_cz_target_entangler_count() -> None:
    """Each Pauli rotation costs two CZ in isolation."""
    for build in (
        lambda c: c.rxx(0.3, 0, 1),
        lambda c: c.ryy(0.3, 0, 1),
        lambda c: c.rzz(0.3, 0, 1),
    ):
        circ = build(QuantumCircuit())
        result = transpile(circ, _line(2, "cz"), optimization_level=2, seed=0)
        assert sum(1 for i in result.circuit if i.gate and i.gate.name == "cz") == 2
