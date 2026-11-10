"""Offline tests for :func:`quspect.backends._compile.compile_for_device`."""

from __future__ import annotations

import random

import pytest

from quspect import QuantumCircuit, Simulator
from quspect.backends._compile import compile_for_device
from quspect.transpiler import CompilationTarget

NATIVE = {"cz", "rz", "sx", "x"}


def _line(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(NATIVE)
    )


def _assert_routed(
    circuit: QuantumCircuit,
    routed: QuantumCircuit,
    layout: list[int] | None,
    target: CompilationTarget,
) -> None:
    """Wire ``w`` sits on ``layout[w]``, two-qubit gates lie on couplings and the
    measured bits agree with the input (the circuits used here are classical)."""
    assert layout is not None
    assert len(layout) == len(set(layout)) >= routed.num_qubits
    for instruction in routed:
        if instruction.is_gate and len(instruction.qubits) == 2:
            a, b = (layout[w] for w in instruction.qubits)
            assert (min(a, b), max(a, b)) in target.coupling_graph
    expected = Simulator(seed=3).run(circuit, shots=16).counts
    assert Simulator(seed=3).run(routed, shots=16).counts == expected
    assert len(expected) == 1


@pytest.mark.parametrize(
    "circuit",
    [
        QuantumCircuit().x(0).x(1).x(1).measure([0], [0]),
        QuantumCircuit().x(0).cx(0, 1).cx(0, 1).measure([0], [0]),
        QuantumCircuit().x(0).swap(0, 1).swap(0, 1).measure([0], [0]),
        QuantumCircuit().x(0).cx(0, 1).x(3).x(3).measure([0, 1], [1, 0]),
    ],
)
def test_initial_layout_when_the_top_qubit_ends_up_without_gates(
    circuit: QuantumCircuit,
) -> None:
    # The optimization removes every gate of the highest logical qubit, which
    # makes the circuit narrower than the requested layout.
    target = _line(5)
    requested = [3, 2, 0, 4][: circuit.num_qubits]
    routed, layout = compile_for_device(target, circuit, initial_layout=requested)
    assert layout is not None
    assert layout[0] == requested[0]
    _assert_routed(circuit, routed, layout, target)


def test_initial_layout_with_idle_top_qubit_inside_a_qubit_pool() -> None:
    target = _line(12)
    circuit = QuantumCircuit().x(0).cx(0, 1).x(2).x(2).measure([0, 1], [0, 1])
    routed, layout = compile_for_device(
        target, circuit, initial_layout=[9, 7, 8], physical_qubits=[7, 8, 9]
    )
    assert layout is not None
    assert layout[:2] == [9, 7]
    assert set(layout) <= {7, 8, 9}
    _assert_routed(circuit, routed, layout, target)


def test_measured_top_qubit_keeps_its_requested_qubit() -> None:
    target = _line(4)
    circuit = QuantumCircuit().x(0).x(1).x(1).measure_all()
    routed, layout = compile_for_device(target, circuit, initial_layout=[3, 1])
    assert layout is not None
    assert layout[:2] == [3, 1]
    _assert_routed(circuit, routed, layout, target)


def test_initial_layout_is_checked_against_the_input_width() -> None:
    target = _line(4)
    idle_top = QuantumCircuit().x(0).x(1).x(1).measure([0], [0])
    # one entry per logical qubit of the input, whatever the optimization removes
    for bad in ([3], [3, 2, 1]):
        with pytest.raises(ValueError, match="initial_mapping length"):
            compile_for_device(target, idle_top, initial_layout=bad)
    with pytest.raises(ValueError, match="duplicates"):
        compile_for_device(target, idle_top, initial_layout=[3, 3])
    with pytest.raises(ValueError, match="outside the target range"):
        compile_for_device(target, idle_top, initial_layout=[3, 4])
    with pytest.raises(ValueError, match="outside physical_qubits"):
        compile_for_device(
            target, idle_top, initial_layout=[3, 0], physical_qubits=[2, 3]
        )


def test_initial_layout_without_a_coupling_graph() -> None:
    target = CompilationTarget.from_coupling_map(3, [], native_gates=set(NATIVE))
    circuit = QuantumCircuit().x(0).x(1).x(1).measure([0], [0])
    routed, layout = compile_for_device(target, circuit, initial_layout=[2, 0])
    assert layout == [2]
    assert routed.num_qubits == 1
    with pytest.raises(ValueError, match="initial_layout length"):
        compile_for_device(target, circuit, initial_layout=[2])


def test_random_classical_circuits_with_idle_qubits_and_layouts() -> None:
    """Property check: X/CX circuits whose top qubits go idle, random layouts."""
    rng = random.Random(20261109)
    target = _line(6)
    for _ in range(40):
        n = rng.randint(2, 5)
        busy = rng.randint(1, n - 1)  # qubits busy .. n-1 end up without gates
        circuit = QuantumCircuit()
        for q in range(busy):
            if rng.random() < 0.6:
                circuit.x(q)
        for _ in range(rng.randint(0, 6) if busy > 1 else 0):
            circuit.cx(*rng.sample(range(busy), 2))
        for q in range(busy, n):
            if rng.random() < 0.5:
                circuit.x(q).x(q)
            else:
                other = rng.randrange(busy)
                circuit.cx(other, q).cx(other, q)
        measured = list(range(busy))
        circuit.measure(measured, rng.sample(measured, busy))
        assert circuit.num_qubits == n
        requested = rng.sample(range(target.num_qubits), n)
        routed, layout = compile_for_device(target, circuit, initial_layout=requested)
        assert layout is not None
        assert layout[:busy] == requested[:busy]
        _assert_routed(circuit, routed, layout, target)
