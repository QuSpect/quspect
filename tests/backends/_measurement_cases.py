"""Random circuits in several measurement spellings, shared by backend tests.

Every spelling measures only terminally, so a correct device compilation
must give the same distribution for each and leave no gate on a measured
wire.  ``"per qubit"`` (one ``measure`` instruction per qubit) is the form
that used to become mid-circuit after routing; a ``"subset"`` leaves some
qubits unmeasured, which lets the optimizer move even a single measure
instruction ahead of gates on the other qubits.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from quspect import QuantumCircuit

#: The three ways to measure every qubit.
FULL_SPELLINGS = ("measure_all", "one instruction", "per qubit")
SPELLINGS = (*FULL_SPELLINGS, "subset", "subset per qubit")


def used_after_measurement(circuit: QuantumCircuit) -> list[int]:
    """Indices of instructions that act on an already measured wire."""
    measured: set[int] = set()
    offending: list[int] = []
    for index, instruction in enumerate(circuit.instructions):
        if instruction.is_barrier:
            continue
        if measured.intersection(instruction.qubits):
            offending.append(index)
        if instruction.is_measure:
            measured.update(instruction.qubits)
    return offending


def measurements_are_last(circuit: QuantumCircuit) -> bool:
    """Whether nothing but measurements follows the first measurement."""
    kinds = [instruction.is_measure for instruction in circuit.instructions]
    return True in kinds and all(kinds[kinds.index(True) :])


def _random_gates(rng: random.Random, max_qubits: int) -> tuple[int, list[tuple]]:
    n = rng.randint(3, max_qubits)
    ops: list[tuple] = []
    for _ in range(rng.randint(4, 24)):
        r = rng.random()
        if r < 0.35:
            ops.append((rng.choice(["h", "t", "x", "s"]), rng.randrange(n)))
        elif r < 0.5:
            ops.append(("ry", round(rng.uniform(-3, 3), 3), rng.randrange(n)))
        else:
            ops.append((rng.choice(["cx", "cz"]), *rng.sample(range(n), 2)))
    return n, ops


def _measure(
    circuit: QuantumCircuit, n: int, spelling: str, rng: random.Random
) -> QuantumCircuit:
    if spelling == "measure_all":
        return circuit.measure_all()
    if spelling == "one instruction":
        return circuit.measure(list(range(n)), list(range(n)))
    if spelling == "per qubit":
        for qubit in range(n):
            circuit.measure([qubit], [qubit])
        return circuit
    qubits = rng.sample(range(n), rng.randint(1, n))
    clbits = rng.sample(range(n), len(qubits))
    if spelling == "subset":
        return circuit.measure(qubits, clbits)
    if spelling == "subset per qubit":
        for qubit, clbit in zip(qubits, clbits, strict=True):
            circuit.measure([qubit], [clbit])
        return circuit
    raise ValueError(f"unknown measurement spelling {spelling!r}")


def random_measured_circuits(
    spelling: str, *, count: int, seed: int = 20261109, max_qubits: int = 6
) -> Iterator[QuantumCircuit]:
    """Yield ``count`` random circuits measured in the given spelling.

    The gates depend only on ``seed``, so the full spellings of one seed
    measure the same circuits.
    """
    gate_rng = random.Random(seed)
    measure_rng = random.Random(seed + 1)
    for _ in range(count):
        n, ops = _random_gates(gate_rng, max_qubits)
        circuit = QuantumCircuit()
        for name, *args in ops:
            getattr(circuit, name)(*args)
        # Every wire carries a gate, so each spelling measures n qubits.
        for qubit in range(n):
            circuit.rz(0.25, qubit)
        yield _measure(circuit, n, spelling, measure_rng)
