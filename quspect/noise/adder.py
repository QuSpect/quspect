"""Noise model adder — applies NoiseModel rules to transform circuits.

The :class:`NoiseModelAdder` walks through a clean :class:`QuantumCircuit`
and inserts noise channels at locations matching the rules defined in a
:class:`NoiseModel`.  This implements the circuit-transform paradigm:
    noise_model → adder → noisy_circuit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.noise.noise_model import NoiseModel, NoiseRule


class NoiseModelAdder:
    """Rule engine that applies a NoiseModel to a QuantumCircuit.

    For each gate instruction in the original circuit, the adder checks
    all gate rules and inserts matching noise channels before or after the
    instruction (depending on ``rule.before_gate``).  For each MEASURE
    instruction, rules with ``on_measure=True`` (e.g. readout errors) are
    checked and their channels are inserted right before the measurement.

    The channel placement follows :meth:`NoiseRule.channel_placements`: a
    channel whose width equals the number of target qubits is inserted once
    on all of them, while a 1-qubit channel matched to a multi-qubit
    instruction is inserted once on each (filtered) target qubit.

    Barriers and existing channels are passed through unchanged.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.noise import NoiseModel, NoiseModelAdder
        >>> from quspect.noise.channel import DepolarizingChannel
        >>> model = NoiseModel()
        >>> _ = model.add_all_qubit_gate_error(
        ...     DepolarizingChannel(name="dp", num_qubits=1, p=0.01),
        ... )
        >>> clean_circ = QuantumCircuit().h(0).cx(0, 1)
        >>> noisy_circ = NoiseModelAdder(model).apply(clean_circ)
        >>> [i.qubits for i in noisy_circ.instructions if i.is_channel]
        [(0,), (0,), (1,)]
    """

    def __init__(self, model: NoiseModel) -> None:
        """Initialize from a NoiseModel.

        Args:
            model: The noise model whose rules will be applied.
        """
        self._model = model

    def apply(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transform a clean circuit into a noisy one.

        For each gate instruction:
        1. Check all gate rules for a match.
        2. Insert matching channels before (if ``before_gate=True``)
           or after (default) the gate.

        For each measurement, matching ``on_measure`` rules insert their
        channels before the measurement.  Barriers and existing channels
        are passed through unchanged.

        Args:
            circuit: The original clean quantum circuit.

        Returns:
            A new circuit with noise channels inserted.

        Raises:
            ValueError: If a rule's channel width matches neither the
                number of target qubits nor 1.

        Example:
            >>> from quspect import QuantumCircuit
            >>> from quspect.noise import depolarizing_model
            >>> circ = QuantumCircuit().h(0).cx(0, 1).measure_all()
            >>> model = depolarizing_model(0.001, 0.01)
            >>> noisy = NoiseModelAdder(model).apply(circ)
            >>> len(noisy) > len(circ)
            True
        """
        noisy_circuit = circuit.copy(name=circuit.name)
        new_instructions: list[CircuitInstruction] = []

        for instr in circuit.instructions:
            if instr.is_measure:
                for rule in self._model:
                    if rule.matches_measure(instr.qubits):
                        new_instructions.extend(_channel_instrs(rule, instr.qubits))
                new_instructions.append(instr)
                continue

            if not instr.is_gate or instr.gate is None:
                # Pass through barriers and existing channels unchanged
                new_instructions.append(instr)
                continue

            gate = instr.gate
            qubits = instr.qubits
            before: list[CircuitInstruction] = []
            after: list[CircuitInstruction] = []
            for rule in self._model:
                if rule.on_measure or not rule.matches_gate(gate, qubits):
                    continue
                target = before if rule.before_gate else after
                target.extend(_channel_instrs(rule, qubits))

            new_instructions.extend(before)
            new_instructions.append(instr)
            new_instructions.extend(after)

        noisy_circuit._instructions = new_instructions
        return noisy_circuit


def _channel_instrs(
    rule: NoiseRule, qubits: tuple[int, ...]
) -> list[CircuitInstruction]:
    """Build the channel instructions a matched rule inserts.

    Args:
        rule: The matched rule.
        qubits: Qubits of the matched instruction.

    Returns:
        One CHANNEL instruction per placement of ``rule.channel``.
    """
    from quspect.circuit.instruction import CircuitInstruction, InstructionType

    return [
        CircuitInstruction(
            type=InstructionType.CHANNEL,
            channel=rule.channel,
            qubits=tgt,
            label=rule.label if rule.label else "",
        )
        for tgt in rule.channel_placements(qubits)
    ]
