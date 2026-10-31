"""Text-based circuit visualization.

This module provides a lightweight ASCII drawer for :class:`QuantumCircuit`.
The layout follows a lane-and-column model inspired by richer circuit
renderers, but keeps the output terminal-friendly and dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quspect.circuit.parameter import Parameter

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate


_MIN_CELL_WIDTH = 7
_QUANTUM_FILL = "-"
_CLASSICAL_FILL = "="
_CONTROLLED_TARGET_LABELS = {
    "cx": "X",
    "cz": "Z",
    "crx": "RX",
    "cry": "RY",
    "crz": "RZ",
    "cp": "P",
    "cu3": "U3",
    "ccx": "X",
}

_CHANNEL_LABELS: dict[str, str] = {
    "bit_flip": "BF",
    "phase_flip": "PF",
    "bit_phase_flip": "BPF",
    "pauli": "PC",
    "depolarizing": "DP",
    "amplitude_damping": "AD",
    "phase_damping": "PD",
    "generalized_amplitude_damping": "GAD",
    "gen_amp_damp": "GAD",
    "kraus": "KC",
    "readout_error": "RO",
}


def draw_text_circuit(circuit: QuantumCircuit) -> str:
    """Render a quantum circuit as a multi-line ASCII diagram.

    Args:
        circuit: The circuit to render.

    Returns:
        A terminal-friendly multi-line string.
    """
    if len(circuit) == 0:
        return "(empty circuit)"

    num_qubits = max(circuit.num_qubits, 1)
    num_clbits = circuit.num_clbits
    total_lanes = num_qubits + num_clbits
    labels = _lane_labels(num_qubits, num_clbits)
    prefixes = _lane_prefixes(labels)
    columns = [
        _build_column(instr, num_qubits, num_clbits) for instr in circuit.instructions
    ]

    rendered_lines: list[str] = []
    for lane_index, prefix in enumerate(prefixes):
        cells = [column[lane_index] for column in columns]
        rendered_lines.append(prefix + "".join(cells))

    if total_lanes == 0:
        return "(empty circuit)"
    return "\n".join(rendered_lines)


def _build_column(
    instruction: CircuitInstruction,
    num_qubits: int,
    num_clbits: int,
) -> list[str]:
    total_lanes = num_qubits + num_clbits
    cell_width = _instruction_width(instruction)
    column = [_empty_quantum_cell(cell_width) for _ in range(num_qubits)]
    column.extend(_empty_classical_cell(cell_width) for _ in range(num_clbits))

    if instruction.is_gate and instruction.gate is not None:
        _draw_gate(column, instruction, cell_width)
    elif instruction.is_measure:
        _draw_measurement(column, instruction, num_qubits, cell_width)
    elif instruction.is_barrier:
        _draw_barrier(column, instruction, num_qubits, cell_width)
    elif instruction.is_channel:
        _draw_channel(column, instruction, cell_width)

    if len(column) != total_lanes:
        raise ValueError("Internal drawer error: unexpected lane count.")
    return column


def _draw_gate(
    column: list[str], instruction: CircuitInstruction, cell_width: int
) -> None:
    gate = instruction.gate
    assert gate is not None
    qubits = instruction.qubits
    name = gate.name.lower()

    if len(qubits) == 1:
        column[qubits[0]] = _boxed_cell(_gate_label(gate), cell_width, _QUANTUM_FILL)
        return

    if name == "swap" and len(qubits) == 2:
        for qubit in qubits:
            column[qubit] = _centered_cell("x", cell_width, _QUANTUM_FILL)
        _draw_vertical_wire(column, min(qubits), max(qubits))
        return

    if name in _CONTROLLED_TARGET_LABELS:
        controls = qubits[:-1]
        target = qubits[-1]
        for control in controls:
            column[control] = _centered_cell("o", cell_width, _QUANTUM_FILL)
        column[target] = _boxed_cell(
            _controlled_target_label(gate),
            cell_width,
            _QUANTUM_FILL,
        )
        _draw_vertical_wire(column, min(qubits), max(qubits))
        return

    label = _gate_label(gate)
    for qubit in qubits:
        column[qubit] = _boxed_cell(label, cell_width, _QUANTUM_FILL)
    _draw_vertical_wire(column, min(qubits), max(qubits))


def _draw_measurement(
    column: list[str],
    instruction: CircuitInstruction,
    num_qubits: int,
    cell_width: int,
) -> None:
    for qubit, clbit in zip(instruction.qubits, instruction.clbits, strict=True):
        quantum_lane = qubit
        classical_lane = num_qubits + clbit
        column[quantum_lane] = _boxed_cell("M", cell_width, _QUANTUM_FILL)
        column[classical_lane] = _centered_cell(str(clbit), cell_width, _CLASSICAL_FILL)
        _draw_vertical_wire(column, quantum_lane, classical_lane)


def _draw_barrier(
    column: list[str],
    instruction: CircuitInstruction,
    num_qubits: int,
    cell_width: int,
) -> None:
    lanes = instruction.qubits or tuple(range(num_qubits))
    for lane in lanes:
        column[lane] = _centered_cell("|", cell_width, _QUANTUM_FILL)


def _draw_channel(
    column: list[str], instruction: CircuitInstruction, cell_width: int
) -> None:
    """Draw a noise channel using double-line box characters.

    Noise channels use ╡...╞ double-line borders to visually
    distinguish them from unitary gate boxes [...].
    """
    channel = instruction.channel
    assert channel is not None
    qubits = instruction.qubits
    label = _channel_label(channel)

    # Double-line box for noise channels
    boxed = f"╡{label}╞"
    cell = _centered_cell(boxed, cell_width, _QUANTUM_FILL)

    if len(qubits) == 1:
        column[qubits[0]] = cell
        return

    for qubit in qubits:
        column[qubit] = cell
    _draw_vertical_wire(column, min(qubits), max(qubits))


def _draw_vertical_wire(column: list[str], start_lane: int, end_lane: int) -> None:
    low = min(start_lane, end_lane)
    high = max(start_lane, end_lane)
    for lane in range(low + 1, high):
        fill_char = (
            _CLASSICAL_FILL
            if column[lane].startswith(_CLASSICAL_FILL)
            else _QUANTUM_FILL
        )
        column[lane] = _overlay_center(column[lane], "|", fill_char)


def _lane_labels(num_qubits: int, num_clbits: int) -> list[str]:
    labels = [f"q{i}" for i in range(num_qubits)]
    labels.extend(f"c{i}" for i in range(num_clbits))
    return labels


def _lane_prefixes(labels: list[str]) -> list[str]:
    label_width = max(len(label) for label in labels)
    return [f"{label.rjust(label_width)}: " for label in labels]


def _instruction_width(instruction: CircuitInstruction) -> int:
    labels: list[str] = []
    if instruction.is_gate and instruction.gate is not None:
        labels.append(_gate_label(instruction.gate))
        if instruction.gate.name.lower() in _CONTROLLED_TARGET_LABELS:
            labels.append(_controlled_target_label(instruction.gate))
        labels.append("o")
        labels.append("x")
    elif instruction.is_measure:
        labels.append("M")
        labels.extend(str(bit) for bit in instruction.clbits)
    elif instruction.is_barrier:
        labels.append("|")
    elif instruction.is_channel:
        if instruction.channel is not None:
            labels.append(_channel_label(instruction.channel))

    width = _MIN_CELL_WIDTH
    for label in labels:
        width = max(width, len(f"[{label}]") + 2)
    if width % 2 == 0:
        width += 1
    return width


def _gate_label(gate: Gate) -> str:
    base = gate.label or gate.name.upper()
    if not gate.params:
        return base
    formatted_params = ", ".join(_format_param(param) for param in gate.params)
    return f"{base}({formatted_params})"


def _controlled_target_label(gate: Gate) -> str:
    """Return the label drawn in the target box of a controlled gate.

    The base symbol comes from ``_CONTROLLED_TARGET_LABELS`` (``"RZ"`` for
    ``crz``, ``"P"`` for ``cp``, ...).  Gate parameters are appended in the
    same format as for uncontrolled gates, e.g. ``"RZ(0.3)"``, so that
    controlled rotations with different angles are distinguishable.
    """
    base = _CONTROLLED_TARGET_LABELS[gate.name.lower()]
    if not gate.params:
        return base
    formatted_params = ", ".join(_format_param(param) for param in gate.params)
    return f"{base}({formatted_params})"


def _format_param(param: Any) -> str:
    if isinstance(param, float):
        return f"{param:g}"
    if isinstance(param, Parameter):
        return param.name
    return str(param)


def _channel_label(channel: Any) -> str:
    """Format a noise channel for display in a circuit diagram.

    Args:
        channel: A NoiseChannel instance.

    Returns:
        A short label string for the channel.
    """
    name = channel.name.lower() if hasattr(channel, "name") else ""
    base = _CHANNEL_LABELS.get(name, name.upper()[:3])
    # Extract params based on channel type
    if name == "pauli":
        px = getattr(channel, "px", 0)
        py = getattr(channel, "py", 0)
        pz = getattr(channel, "pz", 0)
        return f"{base}({px:g},{py:g},{pz:g})"
    elif name in ("depolarizing", "bit_flip", "phase_flip", "bit_phase_flip"):
        return f"{base}({getattr(channel, 'p', 0):g})"
    elif name in ("amplitude_damping", "phase_damping"):
        return f"{base}({getattr(channel, 'gamma', 0):g})"
    elif name in ("generalized_amplitude_damping", "gen_amp_damp"):
        gamma = getattr(channel, "gamma", 0)
        p_thermal = getattr(channel, "p_thermal", 0)
        return f"{base}({gamma:g},{p_thermal:g})"
    elif name == "kraus":
        return f"{base}({getattr(channel, 'name', 'kraus')})"
    elif channel.label:
        return str(channel.label)[:8]
    return base


def _boxed_cell(label: str, width: int, fill_char: str) -> str:
    return _centered_cell(f"[{label}]", width, fill_char)


def _centered_cell(content: str, width: int, fill_char: str) -> str:
    if width < len(content):
        raise ValueError("Cell width is smaller than content width.")
    padding = width - len(content)
    left = padding // 2
    right = padding - left
    return f"{fill_char * left}{content}{fill_char * right}"


def _overlay_center(cell: str, symbol: str, fill_char: str) -> str:
    chars = list(cell)
    center = len(chars) // 2
    if chars[center] == fill_char:
        chars[center] = symbol
    return "".join(chars)


def _empty_quantum_cell(width: int) -> str:
    return _QUANTUM_FILL * width


def _empty_classical_cell(width: int) -> str:
    return _CLASSICAL_FILL * width
