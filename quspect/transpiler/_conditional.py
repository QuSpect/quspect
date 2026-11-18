"""Classically controlled gates in the compiler.

A classically controlled gate (``InstructionType.CONDITIONAL``) is applied on
some measurement records only, so no pass may merge it with its neighbours,
cancel it or move a gate across it.  The passes get this for free: to them it
is a non-gate instruction, i.e. a fence on its qubits, like a measurement.
What remains is to make the conditional gates themselves fit a target:

- :func:`expand_conditional_gates` (before layout and routing) rewrites every
  conditional gate into *single-qubit* conditional gates and *unconditional*
  ``cx`` gates.  A conditional gate on several qubits would otherwise have to
  be routed as a unit, which the routers cannot do for a fence.  The rewrite
  uses ``CZ = S⊗S · exp(iπ/4 · Z⊗Z)`` up to a phase and
  ``exp(iπ/4 · Z⊗Z) = CX · (I⊗S†) · CX``::

      if c: cz(a, b)   ==   cx(a, b); if c: sdg(b); cx(a, b); if c: s(a); if c: s(b)

  With the condition false the two ``cx`` cancel; with it true the sequence
  is ``cz`` up to a global phase, which is a phase of that record only and
  cannot be observed.  The price is two always-executed ``cx`` per
  conditional two-qubit gate.
- :func:`translate_conditional_gates` (after everything else) writes each run
  of conditional single-qubit gates in the target's single-qubit gates.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.gates.gate import Gate
from quspect.gates.standard import CX, SDG, H, S

__all__: list[str] = []


def _one_qubit_names() -> set[str]:
    from quspect.gates import parameterized, standard

    names = {g.name for g in standard.ALL_STANDARD if g.num_qubits == 1}
    for value in vars(parameterized).values():
        if isinstance(value, Gate) and value.num_qubits == 1:
            names.add(value.name)
    return names


def _name(ins: CircuitInstruction) -> str:
    return ins.conditional_gate.name if ins.conditional_gate is not None else "?"


def has_conditional_gates(circuit: QuantumCircuit) -> bool:
    return any(ins.is_conditional for ins in circuit)


def expand_conditional_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Rewrite conditional gates into conditional one-qubit gates and plain ``cx``.

    Args:
        circuit: Any circuit.

    Returns:
        A circuit with the same behaviour on every measurement record in
        which each CONDITIONAL instruction acts on one qubit.  A circuit
        that already has this form is returned itself.

    Raises:
        ValueError: If a conditional gate has no decomposition into ``cx``
            and single-qubit gates.
    """
    from quspect.ir.compilation import _lower_registered_gates, decompose_deep

    circuit = _lower_registered_gates(circuit)
    if not any(ins.is_conditional and len(ins.qubits) > 1 for ins in circuit):
        return circuit
    supported = _one_qubit_names() | {"cx"}
    out: list[CircuitInstruction] = []

    def conditional(gate: Gate, qubit: int, like: CircuitInstruction) -> None:
        out.append(
            CircuitInstruction(
                type=InstructionType.CONDITIONAL,
                conditional_gate=gate,
                qubits=(qubit,),
                clbits=like.clbits,
                label=like.label,
                condition=like.condition,
            )
        )

    for ins in circuit:
        gate = ins.conditional_gate
        if gate is None or len(ins.qubits) == 1:
            out.append(ins)
            continue
        single = QuantumCircuit()
        single.append(gate, range(gate.num_qubits))
        lowered = decompose_deep(single, supported_gates=supported)
        for piece in lowered:
            if piece.gate is None or piece.gate.name not in supported:
                raise ValueError(
                    f"Cannot compile the classically controlled gate "
                    f"{gate.name!r} on qubits {ins.qubits}: it has no "
                    f"decomposition into cx and single-qubit gates."
                )
        for piece in lowered:
            assert piece.gate is not None
            qubits = tuple(ins.qubits[q] for q in piece.qubits)
            if len(qubits) == 1:
                conditional(piece.gate, qubits[0], ins)
                continue
            a, b = qubits
            # if c: cx(a, b)  ==  if c: h(b); [if c: cz(a, b)]; if c: h(b)
            conditional(H, b, ins)
            out.append(
                CircuitInstruction(type=InstructionType.GATE, gate=CX, qubits=(a, b))
            )
            conditional(SDG, b, ins)
            out.append(
                CircuitInstruction(type=InstructionType.GATE, gate=CX, qubits=(a, b))
            )
            conditional(S, a, ins)
            conditional(S, b, ins)
            conditional(H, b, ins)
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)


def translate_conditional_gates(
    circuit: QuantumCircuit, native_gates: Collection[str]
) -> QuantumCircuit:
    """Write conditional single-qubit gates in the target's single-qubit gates.

    Consecutive conditional gates on the same qubit under the same condition
    are one conditional unitary; a run that holds a non-native gate is
    resynthesized (up to a global phase, which is unobservable for a
    conditional gate as well: it multiplies one record's state).  Runs of
    native gates are kept as they are.

    Args:
        circuit: Circuit whose conditional gates act on one qubit each.
        native_gates: The target's native gate names (empty: any gate).

    Returns:
        The circuit with every conditional gate native.

    Raises:
        ValueError: If a conditional gate acts on several qubits, has
            symbolic parameters and is not native, or cannot be expressed in
            the native single-qubit gates.
    """
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.synthesis import synthesize_one_qubit

    native = frozenset(native_gates)
    if not native or not has_conditional_gates(circuit):
        return circuit
    instructions = circuit.instructions
    out: list[CircuitInstruction] = []
    i = 0
    while i < len(instructions):
        ins = instructions[i]
        if not ins.is_conditional:
            out.append(ins)
            i += 1
            continue
        if len(ins.qubits) != 1:
            raise ValueError(
                f"Conditional gate {_name(ins)!r} on qubits "
                f"{ins.qubits} was not expanded to single-qubit gates."
            )
        j = i + 1
        while (
            j < len(instructions)
            and instructions[j].is_conditional
            and instructions[j].qubits == ins.qubits
            and instructions[j].clbits == ins.clbits
            and instructions[j].condition == ins.condition
            and instructions[j].label == ins.label
        ):
            j += 1
        run = instructions[i:j]
        i = j
        if all(_name(r) in native for r in run):
            out.extend(run)
            continue
        matrix = np.eye(2, dtype=complex)
        for r in run:
            assert r.conditional_gate is not None
            try:
                step = np.asarray(gate_matrix(r.conditional_gate), dtype=complex)
            except ValueError as err:
                raise ValueError(
                    f"Cannot express the classically controlled gate "
                    f"{_name(r)!r} on qubit {r.qubits[0]} in the target "
                    f"gates: {err}"
                ) from err
            matrix = step @ matrix
        sequence = synthesize_one_qubit(matrix, native)
        if sequence is None:
            raise ValueError(
                f"Cannot express a classically controlled single-qubit gate on "
                f"qubit {ins.qubits[0]} in the target gates {sorted(native)}."
            )
        out.extend(
            CircuitInstruction(
                type=InstructionType.CONDITIONAL,
                conditional_gate=g,
                qubits=ins.qubits,
                clbits=ins.clbits,
                label=ins.label,
                condition=ins.condition,
            )
            for g in sequence
        )
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)
