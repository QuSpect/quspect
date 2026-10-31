"""Cross-validate :func:`quspect.transpiler.transpile` against qiskit statevectors.

The physical output circuit (``TranspileResult.circuit``) applied to a
random product state placed on ``initial_layout`` must equal the input
circuit applied to the same state with logical qubit ``q`` read on physical
``final_layout[q]`` (unused physical qubits stay in ``|0⟩``), up to global
phase.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import Statevector

import quspect
from qiskit_tests._dag_helpers import grid_edges, heavy_hex_edges, line_edges
from qiskit_tests.helpers import random_circuit_pair, to_qiskit
from quspect.transpiler import CompilationTarget, TranspileResult, transpile

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _targets() -> dict[str, CompilationTarget]:
    hh_n, hh_edges = heavy_hex_edges()
    return {
        "line8": CompilationTarget.from_coupling_map(
            8, line_edges(8), native_gates=set(BASIS)
        ),
        "grid3x3": CompilationTarget.from_coupling_map(
            9, grid_edges(3, 3), native_gates=set(BASIS)
        ),
        "heavyhex13": CompilationTarget.from_coupling_map(
            hh_n, hh_edges, native_gates=set(BASIS)
        ),
    }


TARGETS = _targets()


def _qft(n: int) -> quspect.QuantumCircuit:
    qc = quspect.QuantumCircuit()
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(math.pi / 2 ** (j - i), j, i)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def _qaoa(n: int, seed: int) -> quspect.QuantumCircuit:
    rng = random.Random(seed)
    qc = quspect.QuantumCircuit()
    for q in range(n):
        qc.h(q)
    edges = {tuple(sorted(rng.sample(range(n), 2))) for _ in range(2 * n)}
    for _ in range(2):
        for a, b in sorted(edges):
            qc.cx(a, b).rz(rng.uniform(0, math.pi), b).cx(a, b)
        for q in range(n):
            qc.rx(rng.uniform(0, math.pi), q)
    return qc


def _random(n: int, seed: int) -> quspect.QuantumCircuit:
    circ, _ = random_circuit_pair(
        n,
        6 * n,
        seed,
        gate_set=["h", "x", "s", "t", "rx", "ry", "rz", "cx", "cz", "cp", "swap"],
    )
    return circ


CIRCUITS = {
    "qft6": _qft(6),
    "qaoa7": _qaoa(7, 3),
    "random7": _random(7, 11),
    "ghz8": quspect.QuantumCircuit()
    .h(0)
    .cx(0, 1)
    .cx(1, 2)
    .cx(2, 3)
    .cx(3, 4)
    .cx(4, 5)
    .cx(5, 6)
    .cx(6, 7),
}


def _fidelity(
    logical: quspect.QuantumCircuit,
    result: TranspileResult,
    num_physical: int,
    seed: int = 7,
) -> float:
    n = logical.num_qubits
    rng = random.Random(seed)
    angles = [tuple(rng.uniform(0, 2 * math.pi) for _ in range(3)) for _ in range(n)]

    prep = qiskit.QuantumCircuit(n)
    for q, (a, b, c) in enumerate(angles):
        prep.u(a, b, c, q)
    ideal_logical = prep.compose(to_qiskit(logical, num_qubits=n))
    ideal = qiskit.QuantumCircuit(num_physical).compose(
        ideal_logical, qubits=result.final_layout
    )

    actual = qiskit.QuantumCircuit(num_physical)
    for q, (a, b, c) in enumerate(angles):
        actual.u(a, b, c, result.initial_layout[q])
    actual.compose(to_qiskit(result.circuit, num_qubits=num_physical), inplace=True)
    sv_ideal = Statevector(ideal).data
    sv_actual = Statevector(actual).data
    return float(abs(np.vdot(sv_ideal, sv_actual)) ** 2)


def _check_valid(result: TranspileResult, target: CompilationTarget) -> None:
    for instr in result.circuit:
        if not instr.is_gate:
            continue
        assert instr.gate is not None
        assert instr.gate.name in target.native_gates
        if len(instr.qubits) == 2:
            a, b = sorted(instr.qubits)
            assert (a, b) in target.coupling_graph
    assert len(set(result.initial_layout)) == len(result.initial_layout)
    assert len(set(result.final_layout)) == len(result.final_layout)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("target_name", sorted(TARGETS))
@pytest.mark.parametrize("circuit_name", sorted(CIRCUITS))
def test_transpile_equivalent_and_valid(
    level: int, target_name: str, circuit_name: str
) -> None:
    target = TARGETS[target_name]
    circ = CIRCUITS[circuit_name]
    result = transpile(circ, target, optimization_level=level, seed=5)
    _check_valid(result, target)
    assert len(result.initial_layout) == circ.num_qubits
    assert _fidelity(circ, result, target.num_qubits) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("circuit_name", sorted(CIRCUITS))
def test_wire_circuit_matches_physical_circuit(circuit_name: str) -> None:
    target = TARGETS["grid3x3"]
    result = transpile(CIRCUITS[circuit_name], target, optimization_level=3, seed=2)
    relabelled = [
        (i.gate, tuple(result.wire_layout[q] for q in i.qubits))
        for i in result.wire_circuit
    ]
    assert relabelled == [(i.gate, i.qubits) for i in result.circuit]
    n = CIRCUITS[circuit_name].num_qubits
    assert result.initial_layout == result.wire_layout[:n]


def test_transpile_with_measurements_and_fixed_layout() -> None:
    target = TARGETS["line8"]
    circ = quspect.QuantumCircuit().h(0).cx(0, 3).cx(1, 2).cx(0, 2).measure_all()
    layout = [4, 1, 6, 2]
    for level in (2, 3):
        result = transpile(
            circ, target, optimization_level=level, seed=1, initial_layout=layout
        )
        _check_valid(result, target)
        assert result.initial_layout == layout
        assert sum(1 for i in result.circuit if i.is_measure) >= 1


def test_idle_trailing_qubit_gets_a_layout() -> None:
    target = TARGETS["grid3x3"]
    circ = quspect.QuantumCircuit().h(0).cx(0, 2).cx(1, 2).id(4)
    for level in (0, 2, 3):
        result = transpile(circ, target, optimization_level=level, seed=0)
        _check_valid(result, target)
        assert len(result.initial_layout) == 5
        assert _fidelity(circ, result, target.num_qubits) == pytest.approx(
            1.0, abs=1e-8
        )
