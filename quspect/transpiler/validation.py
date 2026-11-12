"""Differential validation helpers for circuit compilation.

This module compares QuSpect statevectors with an independent Qiskit
interpretation of exported OpenQASM 2, and checks whether a compiler preserves
the input circuit's state up to global phase, following the final qubit layout
when the compiler reports one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from quspect.circuit.circuit import QuantumCircuit
from quspect.ir.qasm2 import QASM2Exporter
from quspect.simulation.simulator import Simulator
from quspect.transpiler._transpile import TranspileResult

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
        compiled_statevector: Compiled-circuit state computed by QuSpect, on
            the compiled circuit's own qubits.  When the compiler returns a
            :class:`~quspect.transpiler.TranspileResult` these are the
            physical qubits it occupies (``compiled_qubits``), not the whole
            device: bit ``k`` of a basis index is physical qubit
            ``compiled_qubits[k]``.
        quspect_qiskit_fidelity: Fidelity between the QuSpect and Qiskit
            interpretations of the original circuit.
        compilation_fidelity: Fidelity between the original QuSpect state,
            with logical qubit ``q`` moved to ``final_layout[q]``, and the
            compiled QuSpect state.
        quspect_qiskit_consistent: Whether the first fidelity is within the
            requested tolerance of one.
        compilation_consistent: Whether compilation preserved the state within
            the requested tolerance.
        final_layout: ``final_layout[logical] = physical`` taken from a
            :class:`~quspect.transpiler.TranspileResult`, or ``None`` when the
            compiler returned a bare circuit, whose qubit ``q`` is then
            compared with original qubit ``q``.
        compiled_qubits: The physical qubits of ``compiled_statevector`` in
            ascending order: every qubit the compiled circuit acts on and
            every qubit of ``final_layout``.  Physical qubits it leaves out
            are untouched and in ``|0>``, so a small circuit on a large
            device stays a small state.  ``None`` for a bare circuit.
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
    final_layout: tuple[int, ...] | None = None
    compiled_qubits: tuple[int, ...] | None = None

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
    compiler: Callable[[QuantumCircuit], QuantumCircuit | TranspileResult]
    | None = None,
    atol: float = 1e-8,
) -> CompilationConsistencyResult:
    """Check QASM interoperability and compiler state preservation.

    Two independent checks are performed:

    1. Export the original circuit to OpenQASM 2, load it in Qiskit, and
       compare Qiskit's statevector with QuSpect's original statevector.
    2. Compile the original circuit and compare QuSpect's compiled statevector
       with its pre-compilation statevector.

    Routing moves logical qubits, so a compiler that maps to a device should
    return the whole :class:`~quspect.transpiler.TranspileResult`, for
    example ``compiler=lambda c: transpile(c, target, seed=0)``. The original
    state is then compared with logical qubit ``q`` on physical qubit
    ``final_layout[q]`` and every other physical qubit in ``|0>``. Only the
    physical qubits the result occupies are simulated (the ones its circuit
    acts on and the ones of ``final_layout``; see ``compiled_qubits`` of the
    result), so the cost follows the circuit, not the size of the device or
    the highest physical index. Both
    states start from ``|0...0>``, which every initial layout prepares
    alike, so ``initial_layout`` does not enter this check. A bare
    :class:`QuantumCircuit` is compared qubit for qubit, so passing only
    ``transpile(...).circuit`` reports a correct but permuted output as
    inconsistent.

    State fidelity is used instead of element-wise equality, so physically
    irrelevant global phase differences pass. Terminal measurements are
    removed before all statevector paths. Mid-circuit measurements and noise
    channels are rejected because they do not define a single pure-state
    evolution.

    Args:
        circuit: Bound QuSpect circuit to validate.
        compiler: Optional compiler callable. It receives a circuit copy and
            must return a :class:`QuantumCircuit` or a
            :class:`~quspect.transpiler.TranspileResult`. Defaults to
            :class:`~quspect.transpiler._passes.OptimizePass`.
        atol: Absolute tolerance applied to ``1 - fidelity``.

    Returns:
        A :class:`CompilationConsistencyResult` containing both booleans,
        fidelities, circuits, QASM, statevectors, and the final layout used.

    Raises:
        ImportError: If Qiskit is not installed.
        TypeError: If arguments or compiler output have invalid types.
        ValueError: If the circuit is empty, parameterized, non-unitary, or
            contains a mid-circuit measurement, or if a returned
            ``final_layout`` does not place every logical qubit on its own
            physical qubit.
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
    final_layout: tuple[int, ...] | None = None
    if isinstance(compiled, TranspileResult):
        final_layout = _checked_final_layout(compiled.final_layout, original.num_qubits)
        compiled = compiled.circuit
    if not isinstance(compiled, QuantumCircuit):
        raise TypeError(
            "compiler must return QuantumCircuit or TranspileResult, "
            f"got {type(compiled).__name__}."
        )

    expected_state = quspect_state
    compiled_qubits: tuple[int, ...] | None = None
    if final_layout is None:
        compiled = _unitary_view(compiled)
        simulated = compiled
    else:
        width = max(compiled.num_qubits, max(final_layout, default=-1) + 1)
        compiled = _unitary_view(compiled, num_qubits=width)
        # A device qubit the result never touches stays in |0> on both sides:
        # simulate the occupied ones only (2**occupied amplitudes instead of
        # 2**(highest physical index + 1)).
        simulated, compiled_qubits = _occupied_view(compiled, final_layout)
        dense = {physical: k for k, physical in enumerate(compiled_qubits)}
        expected_state = _place_state(
            quspect_state,
            tuple(dense[physical] for physical in final_layout),
            len(compiled_qubits),
        )
    compiled_state = np.asarray(simulator.statevector(simulated), dtype=complex)

    qasm_fidelity = _state_fidelity(quspect_state, qiskit_state)
    compilation_fidelity = _state_fidelity(expected_state, compiled_state)

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
        final_layout=final_layout,
        compiled_qubits=compiled_qubits,
    )


def _checked_final_layout(layout: Sequence[int], num_qubits: int) -> tuple[int, ...]:
    """Return ``layout`` once it places each logical qubit on its own qubit."""
    checked = tuple(layout)
    if (
        len(checked) != num_qubits
        or not all(
            isinstance(physical, Integral)
            and not isinstance(physical, bool)
            and physical >= 0
            for physical in checked
        )
        or len(set(checked)) != len(checked)
    ):
        raise ValueError(
            f"final_layout must place each of the {num_qubits} logical qubits "
            f"on a distinct non-negative physical qubit, got {list(checked)}."
        )
    return tuple(int(physical) for physical in checked)


def _occupied_view(
    circuit: QuantumCircuit, layout: tuple[int, ...]
) -> tuple[QuantumCircuit, tuple[int, ...]]:
    """``circuit`` on its occupied qubits only, renumbered in ascending order.

    Occupied are the qubits of its instructions and those of ``layout``.
    Returns the renumbered circuit, exactly ``len(occupied)`` wide, and the
    occupied qubits (new index ``k`` is old qubit ``occupied[k]``).
    """
    occupied = sorted(
        {qubit for instruction in circuit.instructions for qubit in instruction.qubits}
        | set(layout)
    )
    dense = {qubit: k for k, qubit in enumerate(occupied)}
    view = QuantumCircuit._from_instruction_snapshot(
        [
            replace(
                instruction,
                qubits=tuple(dense[qubit] for qubit in instruction.qubits),
            )
            for instruction in circuit.instructions
        ],
        name=circuit.name,
    )
    if view.num_qubits < len(occupied):
        view.id(len(occupied) - 1)
    return view, tuple(occupied)


def _place_state(
    state: NDArray[np.complex128],
    layout: tuple[int, ...],
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Move qubit ``q`` of ``state`` to qubit ``layout[q]`` of a wider register.

    Qubits of the ``num_qubits``-wide register outside ``layout`` are
    ``|0>``. Basis indices are little-endian: bit ``q`` is qubit ``q``.
    """
    indices = np.arange(state.size)
    placed_indices = np.zeros_like(indices)
    for logical, physical in enumerate(layout):
        placed_indices |= ((indices >> logical) & 1) << physical
    placed = np.zeros(1 << num_qubits, dtype=complex)
    placed[placed_indices] = state
    return placed


def _unitary_view(circuit: QuantumCircuit, num_qubits: int = 0) -> QuantumCircuit:
    """Return a measurement-free view at least ``num_qubits`` wide.

    The view is never narrower than ``circuit`` itself.
    """
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
    width = max(circuit.num_qubits, num_qubits)
    if width > 0 and unitary.num_qubits < width:
        unitary.id(width - 1)
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
