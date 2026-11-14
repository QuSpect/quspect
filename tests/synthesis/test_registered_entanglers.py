"""Circuits compiled for an entangler added with ``register_entangler``.

Known issue: the compiled circuits could not be simulated ("unknown standard
gate") nor exported ("No decomposition rule"), because only the synthesis
table knew the gate.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from quspect import QuantumCircuit, Simulator
from quspect.gates import gate_definition, gate_matrix, registry
from quspect.gates.parameterized import RXX
from quspect.gates.standard import CX
from quspect.ir import QASM2Exporter, dumps, qasm2_loads
from quspect.ir.compilation import _lower_registered_gates
from quspect.simulation import DensityMatrixSimulator
from quspect.synthesis import register_entangler, two_qubit
from quspect.transpiler import (
    CompilationTarget,
    check_compilation_consistency,
    transpile,
)
from quspect.transpiler._passes import DecomposePass

#: The cross-resonance gate of the issue's repro (not symmetric).
ECR = np.array(
    [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]], dtype=complex
) / math.sqrt(2)
#: The Mølmer–Sørensen gate exp(-iπ/4 X⊗X) of trapped ions (symmetric).
MS = gate_matrix(RXX.with_params(math.pi / 2))
ENTANGLERS = {"ecr": ECR, "ms": MS}


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    gates = dict(registry._REGISTRY)
    entanglers = dict(two_qubit._ENTANGLERS)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(gates)
    two_qubit._ENTANGLERS.clear()
    two_qubit._ENTANGLERS.update(entanglers)


def _statevector(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(Simulator().statevector(circuit))


def _density(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(DensityMatrixSimulator().density_matrix(circuit)._data)


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = np.vdot(b, a)
    return bool(
        abs(abs(overlap) - 1) < 1e-9
        and np.allclose(a, overlap / abs(overlap) * b, atol=1e-9)
    )


def _random_circuit(seed: int, num_qubits: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit()
    for qubit in range(num_qubits):
        circuit.ry(float(rng.uniform(-3, 3)), qubit)
    for _ in range(6):
        a, b = (int(q) for q in rng.choice(num_qubits, 2, replace=False))
        circuit.cx(a, b).rz(float(rng.uniform(-3, 3)), b).h(a)
    return circuit


def test_the_issue_repro_simulates_and_exports() -> None:
    register_entangler("ecr", ECR)
    target = CompilationTarget.from_coupling_map(
        2, [(0, 1)], native_gates={"ecr", "rz", "sx", "x"}
    )
    source = QuantumCircuit().h(0).cx(0, 1)

    result = transpile(source, target)

    assert "ecr" in {i.gate.name for i in result.circuit if i.is_gate}
    assert _equal_up_to_phase(_statevector(result.circuit), _statevector(source))
    qasm = QASM2Exporter().export(result.circuit)
    assert "ecr" not in qasm
    assert "cz q[" in qasm
    assert _equal_up_to_phase(_statevector(qasm2_loads(qasm)), _statevector(source))


@pytest.mark.parametrize("name", sorted(ENTANGLERS))
def test_registering_an_entangler_registers_its_cz_dressing(name: str) -> None:
    register_entangler(name, ENTANGLERS[name])

    definition = gate_definition(name)
    assert definition is not None
    assert np.allclose(gate_matrix(definition.gate), ENTANGLERS[name])
    gates = [gate.name for gate, _ in definition.decomposition]
    assert gates.count("cz") == 1
    assert set(gates) == {"cz", "u3"}
    cz = gates.index("cz")
    assert definition.decomposition[cz][1] == (0, 1)
    assert len(gates[:cz]) <= 2
    assert len(gates[cz + 1 :]) <= 2


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("num_qubits", [2, 3])
@pytest.mark.parametrize("name", sorted(ENTANGLERS))
def test_compiled_circuits_simulate_lower_and_export(
    name: str, num_qubits: int, seed: int
) -> None:
    register_entangler(name, ENTANGLERS[name])
    edges = [(q, q + 1) for q in range(num_qubits - 1)]
    target = CompilationTarget.from_coupling_map(
        num_qubits, edges, native_gates={name, "rz", "sx", "x"}
    )
    source = _random_circuit(seed, num_qubits)
    if num_qubits == 3:
        source.cx(0, 2)  # needs routing on the line

    check = check_compilation_consistency(
        source, compiler=lambda circuit: transpile(circuit, target)
    )

    assert check.compilation_consistent
    compiled = check.compiled_circuit
    assert name in {i.gate.name for i in compiled if i.is_gate}
    reference = _statevector(compiled)
    # DecomposePass lowers it for a target without it ...
    other = CompilationTarget.from_coupling_map(
        num_qubits, edges, native_gates={"cz", "rz", "sx", "x"}
    )
    lowered = DecomposePass(other).run(compiled)
    assert {i.gate.name for i in lowered if i.is_gate} <= {"cz", "rz", "sx", "x"}
    assert _equal_up_to_phase(_statevector(lowered), reference)
    # ... the exporter writes it decomposed ...
    qasm = QASM2Exporter().export(compiled)
    assert name not in {line.split(" ")[0].split("(")[0] for line in qasm.splitlines()}
    assert _equal_up_to_phase(_statevector(qasm2_loads(qasm)), reference)
    # ... and the density-matrix simulator, which applies gate_matrix,
    # agrees with the state-vector simulator, which runs the decomposition.
    assert np.allclose(
        _density(compiled), np.outer(reference, reference.conj()), atol=1e-9
    )


@pytest.mark.parametrize("name", sorted(ENTANGLERS))
def test_both_orientations_simulate_as_the_matrix(name: str) -> None:
    register_entangler(name, ENTANGLERS[name])
    definition = gate_definition(name)
    assert definition is not None
    for qubits in ([0, 1], [1, 0], [2, 0]):
        circuit = QuantumCircuit().h(0).ry(0.7, 1).rx(0.2, 2)
        circuit.append(definition.gate, qubits).t(0)
        # The state-vector simulator runs the decomposition, the
        # density-matrix simulator applies the registered matrix.
        state = _statevector(circuit)
        assert np.allclose(_density(circuit), np.outer(state, state.conj()), atol=1e-9)
        counts = Simulator(seed=9).run(circuit.copy().measure_all(), shots=64).counts
        assert sum(counts.values()) == 64


def test_circuits_without_registered_gates_do_not_change() -> None:
    """Registering entanglers leaves every other circuit byte-identical."""
    circuits = [
        _random_circuit(11, 3).measure_all(),
        QuantumCircuit().h(0).cx(0, 1).ccx(0, 1, 2).rzz(0.3, 1, 2).swap(0, 2),
    ]
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (1, 2)], native_gates={"cz", "rz", "sx", "x"}
    )

    def outputs(circuit: QuantumCircuit) -> tuple[object, ...]:
        unitary = circuit.copy()
        unitary._instructions = [i for i in unitary._instructions if i.is_gate]
        return (
            dumps(circuit),
            QASM2Exporter().export(circuit),
            _statevector(unitary).tobytes(),
            Simulator(seed=5).run(circuit, shots=100).counts,
            transpile(circuit, target).circuit.instructions,
        )

    before = [outputs(circuit) for circuit in circuits]
    for name, matrix in ENTANGLERS.items():
        register_entangler(name, matrix)
    after = [outputs(circuit) for circuit in circuits]

    assert after == before
    for circuit in circuits:
        assert _lower_registered_gates(circuit) is circuit


def test_builtin_entanglers_are_not_registered_as_gates() -> None:
    register_entangler("cx", gate_matrix(CX))

    assert gate_definition("cx") is None
    assert "cx" not in registry._REGISTRY
