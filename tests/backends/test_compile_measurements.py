"""Shared device compilation: terminal measurements stay terminal.

Offline; the compiled circuits are checked exactly against the simulator.
"""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit, Simulator
from quspect.backends._compile import (
    compile_for_device,
    defer_terminal_measurements,
)
from quspect.transpiler._target import CompilationTarget
from tests.backends._measurement_cases import (
    SPELLINGS,
    measurements_are_last,
    random_measured_circuits,
    used_after_measurement,
)
from tests.backends._originq_fakes import (
    assert_same_distribution,
    quspect_exact_distribution,
)

LINE = [(i, i + 1) for i in range(6)]
GRID = [(i, i + 1) for i in range(16) if i % 4 < 3] + [(i, i + 4) for i in range(12)]
CZ_BASIS = {"cz", "h", "rx", "ry", "rz"}


def _shape(circuit: QuantumCircuit) -> list[tuple]:
    return [
        (
            "measure" if ins.is_measure else "barrier" if ins.is_barrier else "gate",
            ins.gate.name if ins.gate is not None else None,
            ins.qubits,
            ins.clbits,
        )
        for ins in circuit.instructions
    ]


# ---------------------------------------------------------------------------
# defer_terminal_measurements
# ---------------------------------------------------------------------------


def test_defer_returns_the_same_circuit_when_measurements_are_already_last() -> None:
    circuit = QuantumCircuit().h(0).cx(0, 1).measure_all()
    assert defer_terminal_measurements(circuit) is circuit

    per_qubit = QuantumCircuit().h(0).cx(0, 1)
    per_qubit.measure([1], [0]).barrier().measure([0], [1])
    assert defer_terminal_measurements(per_qubit) is per_qubit


def test_defer_moves_interleaved_terminal_measurements_behind_the_gates() -> None:
    circuit = QuantumCircuit(name="interleaved")
    circuit.h(0).measure([0], [1]).h(1).cx(1, 2).measure([2], [0]).x(1)
    circuit.measure([1], [2])

    deferred = defer_terminal_measurements(circuit)

    assert deferred.name == "interleaved"
    assert _shape(deferred) == [
        ("gate", "h", (0,), ()),
        ("gate", "h", (1,), ()),
        ("gate", "cx", (1, 2), ()),
        ("gate", "x", (1,), ()),
        ("measure", None, (0,), (1,)),
        ("measure", None, (2,), (0,)),
        ("measure", None, (1,), (2,)),
    ]
    # The source circuit is not modified.
    assert circuit.instructions[1].is_measure


def test_defer_keeps_mid_circuit_measurements_in_place() -> None:
    circuit = QuantumCircuit()
    circuit.x(0).measure([0], [0]).x(0).h(1).measure([1], [1]).x(2)
    circuit.measure([0], [2])

    deferred = defer_terminal_measurements(circuit)

    # measure(0 -> c0) is followed by x(0): it stays before that gate.
    assert _shape(deferred) == [
        ("gate", "x", (0,), ()),
        ("measure", None, (0,), (0,)),
        ("gate", "x", (0,), ()),
        ("gate", "h", (1,), ()),
        ("gate", "x", (2,), ()),
        ("measure", None, (1,), (1,)),
        ("measure", None, (0,), (2,)),
    ]


def test_defer_never_moves_a_measurement_past_a_later_write_of_its_clbit() -> None:
    circuit = QuantumCircuit()
    circuit.x(0).measure([0], [0])  # terminal on qubit 0 ...
    circuit.measure([1], [0]).x(1)  # ... but c0 is overwritten mid-circuit
    assert defer_terminal_measurements(circuit) is circuit
    assert Simulator(seed=3).run(circuit, shots=8).counts == {"0": 8}


def test_defer_treats_barriers_as_transparent_and_instructions_as_units() -> None:
    circuit = QuantumCircuit()
    circuit.h(0).measure([0], [0]).barrier(0, 1).h(1).measure([1, 2], [1, 2]).x(2)
    circuit.measure([3], [3])

    deferred = defer_terminal_measurements(circuit)

    assert _shape(deferred) == [
        ("gate", "h", (0,), ()),
        ("barrier", None, (0, 1), ()),
        ("gate", "h", (1,), ()),
        # One qubit of this instruction is used again: it stays whole.
        ("measure", None, (1, 2), (1, 2)),
        ("gate", "x", (2,), ()),
        ("measure", None, (0,), (0,)),
        ("measure", None, (3,), (3,)),
    ]


# ---------------------------------------------------------------------------
# compile_for_device
# ---------------------------------------------------------------------------


def test_swap_is_not_routed_through_a_measured_wire() -> None:
    """Regression: per-qubit measurements became mid-circuit after routing.

    The optimizer emits ``measure(2)`` before ``cx(0, 1)``; routing that gate
    on a line then swapped through the measured wire and measured it again.
    """
    circuit = QuantumCircuit()
    circuit.h(0).ry(0.4, 1).ry(1.1, 2)
    circuit.cx(2, 1).cx(0, 2).cx(0, 1).cx(2, 1)
    for qubit in range(3):
        circuit.measure([qubit], [qubit])
    target = CompilationTarget.from_coupling_map(5, LINE[:4])

    routed, layout = compile_for_device(target, circuit)

    assert layout is not None
    assert any(i.is_gate and i.gate.name == "swap" for i in routed.instructions)
    assert used_after_measurement(routed) == []
    assert all(i.is_measure for i in routed.instructions[-3:])
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize(
    ("num_qubits", "edges", "native"),
    [
        (7, LINE, None),
        (7, LINE, CZ_BASIS),
        (16, GRID, None),
        (16, GRID, CZ_BASIS),
    ],
    ids=["line", "line-cz", "grid", "grid-cz"],
)
def test_terminal_measurements_stay_terminal_for_random_circuits(
    spelling: str,
    num_qubits: int,
    edges: list[tuple[int, int]],
    native: set[str] | None,
) -> None:
    target = CompilationTarget.from_coupling_map(num_qubits, edges, native_gates=native)
    swaps = 0
    for circuit in random_measured_circuits(spelling, count=40):
        routed, layout = compile_for_device(target, circuit)

        assert layout is not None
        assert used_after_measurement(routed) == [], _shape(circuit)
        assert measurements_are_last(routed)
        assert sorted(c for ins in routed.instructions for c in ins.clbits) == sorted(
            c for ins in circuit.instructions for c in ins.clbits
        )
        assert_same_distribution(
            quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
        )
        swaps += sum(
            1 for ins in routed.instructions if ins.is_gate and ins.gate.name == "swap"
        )
    assert swaps > 0  # the sample must exercise routing


def test_mid_circuit_measurement_keeps_its_order_through_compilation() -> None:
    circuit = QuantumCircuit()
    circuit.x(0).measure([0], [0]).x(0).x(2).cx(0, 2)
    circuit.measure([1], [1]).measure([2], [2]).measure([0], [3])
    target = CompilationTarget.from_coupling_map(5, LINE[:4])

    routed, _ = compile_for_device(target, circuit)

    # The input uses qubit 0 after measuring it, so the output does too ...
    assert used_after_measurement(routed) != []
    kinds = [ins.is_measure for ins in routed.instructions]
    assert kinds.index(True) < len(kinds) - 3
    # ... and the three terminal measurements are issued last.
    assert all(ins.is_measure for ins in routed.instructions[-3:])
    expected = Simulator(seed=5).run(circuit, shots=16).counts
    assert expected == {"1010": 16}
    assert Simulator(seed=5).run(routed, shots=16).counts == expected


def test_unrouted_target_also_issues_terminal_measurements_last() -> None:
    """Without a coupling graph nothing is routed, but readout stays aligned."""
    circuit = QuantumCircuit()
    circuit.h(0).cx(2, 1).cx(0, 2).cx(0, 1).cx(2, 1)
    for qubit in range(3):
        circuit.measure([qubit], [qubit])
    target = CompilationTarget(num_qubits=3)

    compiled, layout = compile_for_device(target, circuit)

    assert layout is None
    assert all(ins.is_measure for ins in compiled.instructions[-3:])
    assert not any(ins.is_measure for ins in compiled.instructions[:-3])
    assert_same_distribution(
        quspect_exact_distribution(compiled), quspect_exact_distribution(circuit)
    )
