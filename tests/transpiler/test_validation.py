"""Tests for differential compiler validation helpers."""

from __future__ import annotations

import dataclasses
import random

import pytest

pytest.importorskip("qiskit")

from quspect import QuantumCircuit
from quspect.transpiler import (
    CompilationTarget,
    TranspileResult,
    check_compilation_consistency,
    generate_random_circuit,
    transpile,
)


def _bernstein_vazirani(num_qubits: int) -> QuantumCircuit:
    """Return BV with an all-ones secret; the last qubit is the ancilla."""
    ancilla = num_qubits - 1
    circuit = QuantumCircuit(name=f"bv{num_qubits}")
    circuit.x(ancilla)
    for qubit in range(num_qubits):
        circuit.h(qubit)
    for qubit in range(ancilla):
        circuit.cx(qubit, ancilla)
    for qubit in range(ancilla):
        circuit.h(qubit)
    return circuit


def _line_target(num_qubits: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        num_qubits,
        [(qubit, qubit + 1) for qubit in range(num_qubits - 1)],
        native_gates={"cz", "rz", "h", "x"},
    )


def test_random_circuit_is_reproducible_and_uses_requested_width() -> None:
    first = generate_random_circuit(3, 5, seed=123)
    second = generate_random_circuit(3, 5, seed=123)
    different = generate_random_circuit(3, 5, seed=456)

    assert first.instructions == second.instructions
    assert first.instructions != different.instructions
    assert first.num_qubits == 3
    assert len(first.instructions) >= 15


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"num_qubits": 0, "depth": 1}, ValueError),
        ({"num_qubits": 1, "depth": 0}, ValueError),
        ({"num_qubits": 1, "depth": 1, "two_qubit_probability": 1.1}, ValueError),
        ({"num_qubits": 1, "depth": 1, "barrier_probability": -0.1}, ValueError),
    ],
)
def test_random_circuit_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        generate_random_circuit(**kwargs)  # type: ignore[arg-type]


def test_consistent_circuit_passes_both_checks() -> None:
    circuit = QuantumCircuit(name="bell")
    circuit.h(0).cx(0, 1)

    result = check_compilation_consistency(circuit)

    assert result.quspect_qiskit_consistent is True
    assert result.compilation_consistent is True
    assert result.consistent is True
    assert result.quspect_qiskit_fidelity == pytest.approx(1.0)
    assert result.compilation_fidelity == pytest.approx(1.0)


def test_terminal_measurement_is_removed_for_statevector_checks() -> None:
    circuit = QuantumCircuit().h(0).measure_all()

    result = check_compilation_consistency(circuit)

    assert all(not instruction.is_measure for instruction in result.original_circuit)
    assert result.consistent is True


def test_x_h_x_fusion_regression_is_consistent() -> None:
    circuit = QuantumCircuit(name="single_qubit_fusion_regression")
    circuit.x(0).h(0).x(0)
    circuit.barrier(0)
    circuit.h(0)

    result = check_compilation_consistency(circuit)

    assert result.quspect_qiskit_consistent is True
    assert result.quspect_qiskit_fidelity == pytest.approx(1.0)
    assert result.compilation_consistent is True
    assert result.compilation_fidelity == pytest.approx(1.0)
    assert result.consistent is True


def test_random_circuit_qasm_agrees_with_qiskit() -> None:
    circuit = generate_random_circuit(
        3,
        4,
        seed=7,
        two_qubit_probability=1.0,
        barrier_probability=0.25,
    )

    result = check_compilation_consistency(circuit, compiler=lambda item: item.copy())

    assert result.quspect_qiskit_consistent is True
    assert result.compilation_consistent is True


def test_one_hundred_random_circuits_preserve_state() -> None:
    """A deterministic random corpus must survive QASM and compilation."""
    master_rng = random.Random(20260729)

    for index in range(100):
        num_qubits = master_rng.randint(1, 5)
        depth = master_rng.randint(1, 10)
        circuit_seed = master_rng.randrange(2**63)
        circuit = generate_random_circuit(
            num_qubits,
            depth,
            seed=circuit_seed,
            two_qubit_probability=0.35,
            barrier_probability=0.05,
            name=f"random_validation_{index:03d}",
        )

        result = check_compilation_consistency(circuit)
        context = (
            f"index={index}, num_qubits={num_qubits}, depth={depth}, "
            f"seed={circuit_seed}, qasm_fidelity="
            f"{result.quspect_qiskit_fidelity}, compilation_fidelity="
            f"{result.compilation_fidelity}"
        )

        assert result.quspect_qiskit_consistent, context
        assert result.compilation_consistent, context


def test_mid_circuit_measurement_is_rejected() -> None:
    circuit = QuantumCircuit().h(0).measure([0], [0]).x(0)

    with pytest.raises(ValueError, match="mid-circuit"):
        check_compilation_consistency(circuit)


def test_compiler_must_return_circuit() -> None:
    circuit = QuantumCircuit().h(0)

    with pytest.raises(TypeError, match="must return QuantumCircuit"):
        check_compilation_consistency(circuit, compiler=lambda _item: None)  # type: ignore[arg-type,return-value]


@pytest.mark.parametrize("seed", range(5))
def test_routed_transpile_result_is_consistent(seed: int) -> None:
    """BV-10 on a chain ends permuted; the check follows ``final_layout``."""
    circuit = _bernstein_vazirani(10)
    target = _line_target(10)

    result = check_compilation_consistency(
        circuit,
        compiler=lambda item: transpile(item, target, optimization_level=3, seed=seed),
    )

    # Every seed ends with the ancilla moved, so a check that ignored the
    # layout would see fidelity 0.25 here.
    assert result.final_layout is not None
    assert result.final_layout != tuple(range(10))
    assert result.compilation_fidelity == pytest.approx(1.0)
    assert result.compilation_consistent is True
    assert result.consistent is True


def test_transpile_result_on_wider_target_is_consistent() -> None:
    """Physical qubits outside the layout must stay in ``|0>``."""
    circuit = QuantumCircuit().x(0).h(1).cx(0, 2).cx(1, 2)
    target = _line_target(5)

    result = check_compilation_consistency(
        circuit,
        compiler=lambda item: transpile(
            item, target, optimization_level=3, seed=0, initial_layout=[4, 0, 2]
        ),
    )

    assert result.final_layout is not None
    assert max(result.final_layout) >= circuit.num_qubits
    assert result.compiled_statevector.shape == (2**5,)
    assert result.compilation_consistent is True


def test_wrong_final_layout_is_flagged() -> None:
    """Swapping the ancilla's and qubit 0's final places breaks the check."""
    circuit = _bernstein_vazirani(4)
    target = _line_target(4)

    def mislabelled(item: QuantumCircuit) -> TranspileResult:
        compiled = transpile(item, target, optimization_level=3, seed=0)
        layout = list(compiled.final_layout)
        layout[0], layout[-1] = layout[-1], layout[0]
        return dataclasses.replace(compiled, final_layout=layout)

    result = check_compilation_consistency(circuit, compiler=mislabelled)

    assert result.compilation_consistent is False
    assert result.compilation_fidelity == pytest.approx(0.25)


@pytest.mark.parametrize(
    "final_layout",
    [[0, 1], [0, 1, 1], [0, 1, -1]],
    ids=["short", "repeated", "negative"],
)
def test_malformed_final_layout_is_rejected(final_layout: list[int]) -> None:
    circuit = QuantumCircuit().h(0).cx(0, 1).cx(1, 2)
    target = _line_target(3)

    def malformed(item: QuantumCircuit) -> TranspileResult:
        compiled = transpile(item, target, seed=0)
        return dataclasses.replace(compiled, final_layout=final_layout)

    with pytest.raises(ValueError, match="final_layout"):
        check_compilation_consistency(circuit, compiler=malformed)
