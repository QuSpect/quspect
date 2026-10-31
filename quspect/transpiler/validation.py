"""Differential validation helpers for circuit compilation.

This module compares QuSpect statevectors with an independent Qiskit
interpretation of exported OpenQASM 2, and checks whether a compiler preserves
the input circuit's state up to global phase.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from quspect.circuit.circuit import QuantumCircuit
from quspect.ir.qasm2 import QASM2Exporter
from quspect.simulation.simulator import Simulator

if TYPE_CHECKING:
    from quspect.circuit.instruction import CircuitInstruction


@dataclass(frozen=True)
class CompilationConsistencyResult:
    """Result of an independent compiler-consistency check.

    Attributes:
        original_circuit: Measurement-free circuit used by every statevector
            path.
        compiled_circuit: Circuit returned by the selected compiler.
        original_qasm: OpenQASM 2 exported from ``original_circuit``.
        quspect_statevector: Original state computed by QuSpect.
        qiskit_statevector: Original-QASM state computed by Qiskit.
        compiled_statevector: Compiled-circuit state computed by QuSpect.
        quspect_qiskit_fidelity: Fidelity between the QuSpect and Qiskit
            interpretations of the original circuit.
        compilation_fidelity: Fidelity between the original and compiled
            QuSpect states.
        quspect_qiskit_consistent: Whether the first fidelity is within the
            requested tolerance of one.
        compilation_consistent: Whether compilation preserved the state within
            the requested tolerance.
    """

    original_circuit: QuantumCircuit
    compiled_circuit: QuantumCircuit
    original_qasm: str
    quspect_statevector: NDArray[np.complex128]
    qiskit_statevector: NDArray[np.complex128]
    compiled_statevector: NDArray[np.complex128]
    quspect_qiskit_fidelity: float
    compilation_fidelity: float
    quspect_qiskit_consistent: bool
    compilation_consistent: bool

    @property
    def consistent(self) -> bool:
        """Return whether both independent checks passed."""
        return self.quspect_qiskit_consistent and self.compilation_consistent


def generate_random_circuit(
    num_qubits: int,
    depth: int,
    *,
    seed: int | None = None,
    two_qubit_probability: float = 0.35,
    barrier_probability: float = 0.0,
    name: str = "",
) -> QuantumCircuit:
    """Generate a reproducible random unitary circuit.

    Every layer applies one random single-qubit gate to every qubit, then
    randomly applies two-qubit gates to disjoint shuffled pairs. The selected
    gate set is supported by QuSpect's OpenQASM 2 exporter and Qiskit importer,
    making the output suitable for differential compiler tests.

    Args:
        num_qubits: Positive number of active qubits.
        depth: Positive number of generated layers.
        seed: Optional deterministic random seed.
        two_qubit_probability: Probability of adding a two-qubit gate for each
            disjoint pair in a layer.
        barrier_probability: Probability of adding a full-width barrier after
            each layer.
        name: Optional circuit name.

    Returns:
        A measurement-free :class:`QuantumCircuit`.
    """
    _validate_positive_int(num_qubits, "num_qubits")
    _validate_positive_int(depth, "depth")
    _validate_probability(two_qubit_probability, "two_qubit_probability")
    _validate_probability(barrier_probability, "barrier_probability")

    rng = random.Random(seed)
    circuit_name = name or f"random_{num_qubits}q_depth{depth}"
    circuit = QuantumCircuit(name=circuit_name)

    single_qubit_gates: tuple[Callable[[int], QuantumCircuit], ...] = (
        circuit.h,
        circuit.x,
        circuit.y,
        circuit.z,
        circuit.s,
        circuit.sdg,
        circuit.t,
        circuit.tdg,
        lambda qubit: circuit.rx(rng.uniform(-math.pi, math.pi), qubit),
        lambda qubit: circuit.ry(rng.uniform(-math.pi, math.pi), qubit),
        lambda qubit: circuit.rz(rng.uniform(-math.pi, math.pi), qubit),
        lambda qubit: circuit.p(rng.uniform(-math.pi, math.pi), qubit),
        lambda qubit: circuit.u3(
            rng.uniform(0.0, math.pi),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
            qubit,
        ),
    )

    for _ in range(depth):
        for qubit in range(num_qubits):
            rng.choice(single_qubit_gates)(qubit)

        shuffled = list(range(num_qubits))
        rng.shuffle(shuffled)
        for pair_index in range(0, num_qubits - 1, 2):
            if rng.random() >= two_qubit_probability:
                continue
            left, right = shuffled[pair_index : pair_index + 2]
            gate = rng.choice(("cx", "cz", "swap", "cp", "crx", "cry", "crz"))
            if gate in {"cx", "cz", "swap"}:
                getattr(circuit, gate)(left, right)
            else:
                angle = rng.uniform(-math.pi, math.pi)
                getattr(circuit, gate)(angle, left, right)

        if rng.random() < barrier_probability:
            circuit.barrier(*range(num_qubits))

    return circuit


def check_compilation_consistency(
    circuit: QuantumCircuit,
    *,
    compiler: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    atol: float = 1e-8,
) -> CompilationConsistencyResult:
    """Check QASM interoperability and compiler state preservation.

    Two independent checks are performed:

    1. Export the original circuit to OpenQASM 2, load it in Qiskit, and
       compare Qiskit's statevector with QuSpect's original statevector.
    2. Compile the original circuit and compare QuSpect's compiled statevector
       with its pre-compilation statevector.

    State fidelity is used instead of element-wise equality, so physically
    irrelevant global phase differences pass. Terminal measurements are
    removed before all statevector paths. Mid-circuit measurements and noise
    channels are rejected because they do not define a single pure-state
    evolution.

    Args:
        circuit: Bound QuSpect circuit to validate.
        compiler: Optional compiler callable. It receives a circuit copy and
            must return a :class:`QuantumCircuit`. Defaults to
            :class:`~quspect.transpiler._passes.OptimizePass`.
        atol: Absolute tolerance applied to ``1 - fidelity``.

    Returns:
        A :class:`CompilationConsistencyResult` containing both booleans,
        fidelities, circuits, QASM, and statevectors.

    Raises:
        ImportError: If Qiskit is not installed.
        TypeError: If arguments or compiler output have invalid types.
        ValueError: If the circuit is empty, parameterized, non-unitary, or
            contains a mid-circuit measurement.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"circuit must be QuantumCircuit, got {type(circuit).__name__}."
        )
    if isinstance(atol, bool) or not isinstance(atol, Real):
        raise TypeError("atol must be a non-negative finite real number.")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be a non-negative finite real number.")
    if circuit.parameters:
        names = sorted(parameter.name for parameter in circuit.parameters)
        raise ValueError(f"Circuit contains unbound parameter(s): {names}.")

    try:
        from qiskit import qasm2  # type: ignore[import-untyped]
        from qiskit.quantum_info import (  # type: ignore[import-untyped]
            Statevector as QiskitStatevector,
        )
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for compiler validation. Install it with "
            "`pip install 'quspect[validation]'`."
        ) from exc

    original = _unitary_view(circuit)
    simulator = Simulator()
    quspect_state = np.asarray(simulator.statevector(original), dtype=complex)

    original_qasm = QASM2Exporter().export(original)
    qiskit_circuit = qasm2.loads(
        original_qasm,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    qiskit_state = np.asarray(
        QiskitStatevector.from_instruction(qiskit_circuit).data,
        dtype=complex,
    )

    if compiler is None:
        from quspect.transpiler._passes import OptimizePass

        compiler = OptimizePass().run
    if not callable(compiler):
        raise TypeError("compiler must be callable.")

    compiled = compiler(original.copy())
    if not isinstance(compiled, QuantumCircuit):
        raise TypeError(
            f"compiler must return QuantumCircuit, got {type(compiled).__name__}."
        )
    compiled = _unitary_view(compiled)
    compiled_state = np.asarray(simulator.statevector(compiled), dtype=complex)

    qasm_fidelity = _state_fidelity(quspect_state, qiskit_state)
    compilation_fidelity = _state_fidelity(quspect_state, compiled_state)

    return CompilationConsistencyResult(
        original_circuit=original,
        compiled_circuit=compiled,
        original_qasm=original_qasm,
        quspect_statevector=quspect_state,
        qiskit_statevector=qiskit_state,
        compiled_statevector=compiled_state,
        quspect_qiskit_fidelity=qasm_fidelity,
        compilation_fidelity=compilation_fidelity,
        quspect_qiskit_consistent=(1.0 - qasm_fidelity <= tolerance),
        compilation_consistent=(1.0 - compilation_fidelity <= tolerance),
    )


def _unitary_view(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a measurement-free view while preserving the circuit width."""
    instructions: list[CircuitInstruction] = []
    measurement_started = False

    for instruction in circuit.instructions:
        if instruction.is_channel:
            raise ValueError(
                "Compiler statevector validation does not support noise channels."
            )
        if instruction.is_measure:
            measurement_started = True
            continue
        if measurement_started:
            if instruction.is_gate:
                raise ValueError(
                    "Compiler statevector validation does not support "
                    "mid-circuit measurements followed by gates."
                )
            # Barriers after terminal measurement do not affect the unitary.
            continue
        instructions.append(instruction)

    unitary = QuantumCircuit._from_instruction_snapshot(
        instructions,
        name=circuit.name,
    )
    if circuit.num_qubits > 0 and unitary.num_qubits < circuit.num_qubits:
        unitary.id(circuit.num_qubits - 1)
    if len(unitary) == 0:
        raise ValueError("Circuit must contain at least one unitary instruction.")
    return unitary


def _state_fidelity(
    left: NDArray[np.complex128],
    right: NDArray[np.complex128],
) -> float:
    """Return pure-state fidelity, or zero for incompatible dimensions."""
    if left.shape != right.shape:
        return 0.0
    left_norm = float(np.vdot(left, left).real)
    right_norm = float(np.vdot(right, right).real)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    fidelity = float(abs(np.vdot(left, right)) ** 2 / (left_norm * right_norm))
    return min(1.0, max(0.0, fidelity))


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_probability(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number in [0, 1].")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
