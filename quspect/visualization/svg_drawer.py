"""SVG-based circuit visualization.

This module provides a lightweight SVG backend for :class:`QuantumCircuit`.
It packs independent instructions into shared visual columns while producing
notebook-friendly scalable output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from quspect.visualization.text_drawer import (
    _CONTROLLED_TARGET_LABELS,
    _controlled_target_label,
    _gate_label,
    _instruction_width,
    _lane_labels,
)

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate


@dataclass(frozen=True)
class SvgStyle:
    """Color palette for SVG circuit rendering."""

    background: str
    wire: str
    classical_wire: str
    gate_fill: str
    gate_stroke: str
    text: str
    control: str
    barrier: str
    measure_fill: str
    channel_fill: str
    channel_stroke: str

    def as_dict(self) -> dict[str, str]:
        """Return this style as a plain mapping."""
        return {
            "background": self.background,
            "wire": self.wire,
            "classical_wire": self.classical_wire,
            "gate_fill": self.gate_fill,
            "gate_stroke": self.gate_stroke,
            "text": self.text,
            "control": self.control,
            "barrier": self.barrier,
            "measure_fill": self.measure_fill,
            "channel_fill": self.channel_fill,
            "channel_stroke": self.channel_stroke,
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, str]) -> SvgStyle:
        """Construct a style from a mapping with strict key validation."""
        required = {
            "background",
            "wire",
            "classical_wire",
            "gate_fill",
            "gate_stroke",
            "text",
            "control",
            "barrier",
            "measure_fill",
            "channel_fill",
            "channel_stroke",
        }
        missing = sorted(required - set(mapping))
        if missing:
            raise ValueError(f"Missing SVG style keys: {missing}")
        return cls(
            background=mapping["background"],
            wire=mapping["wire"],
            classical_wire=mapping["classical_wire"],
            gate_fill=mapping["gate_fill"],
            gate_stroke=mapping["gate_stroke"],
            text=mapping["text"],
            control=mapping["control"],
            barrier=mapping["barrier"],
            measure_fill=mapping["measure_fill"],
            channel_fill=mapping["channel_fill"],
            channel_stroke=mapping["channel_stroke"],
        )


_SVG_THEME_DATA: dict[str, dict[str, str]] = {
    "official": {
        "background": "#ffffff",
        "wire": "#9aa3b2",
        "classical_wire": "#7c8798",
        "gate_fill": "#f8faff",
        "gate_stroke": "#536dfe",
        "text": "#20283a",
        "control": "#536dfe",
        "barrier": "#9aa3b2",
        "measure_fill": "#fff4f4",
        "channel_fill": "#fff7ed",
        "channel_stroke": "#c77532",
    },
    "light": {
        "background": "#f7f9fd",
        "wire": "#180835",
        "classical_wire": "#667085",
        "gate_fill": "#ffffff",
        "gate_stroke": "#e9b645",
        "text": "#180835",
        "control": "#5b5ce2",
        "barrier": "#8a744d",
        "measure_fill": "#fff8f6",
        "channel_fill": "#fff3e8",
        "channel_stroke": "#c77532",
    },
    "dark": {
        "background": "#180835",
        "wire": "#d8d8d8",
        "classical_wire": "#b8adc8",
        "gate_fill": "#180835",
        "gate_stroke": "#e9b645",
        "text": "#f4f1fb",
        "control": "#66c31a",
        "barrier": "#8a744d",
        "measure_fill": "#211040",
        "channel_fill": "#2a153c",
        "channel_stroke": "#c77532",
    },
}
_SVG_THEMES: dict[str, SvgStyle] = {
    name: SvgStyle.from_mapping(style) for name, style in _SVG_THEME_DATA.items()
}

_TOP_MARGIN = 34
_BOTTOM_MARGIN = 34
_LEFT_MARGIN = 20
_RIGHT_MARGIN = 20
_LABEL_WIDTH = 50
_LANE_GAP = 56
_COLUMN_PADDING = 26
_GATE_HEIGHT = 36
_WIRE_STROKE = 1.6
_GATE_RADIUS = 7
_FONT_FAMILY = "Arial, Helvetica, sans-serif"


@dataclass(frozen=True)
class CircuitSVG:
    """SVG representation of a circuit.

    The object is both string-convertible and Jupyter-renderable via
    ``_repr_svg_``.
    """

    content: str

    def __str__(self) -> str:
        """Return the raw SVG string."""
        return self.content

    def _repr_svg_(self) -> str:
        """Return SVG content for Jupyter frontends."""
        return self.content

    def to_file(self, filename: str | Path) -> None:
        """Write the SVG content to disk.

        Args:
            filename: Destination path.
        """
        Path(filename).write_text(self.content, encoding="utf-8")


def draw_svg_circuit(
    circuit: QuantumCircuit,
    *,
    style: str | dict[str, str] | SvgStyle | None = None,
    width: int | float | None = None,
    scale: float | None = None,
) -> CircuitSVG:
    """Render a quantum circuit as SVG.

    Args:
        circuit: The circuit to render.
        style: Either a named theme or a custom theme dictionary.
        width: Optional minimum SVG width in pixels.
        scale: Optional positive scale factor applied to the whole figure.

    Returns:
        A notebook-friendly SVG wrapper.

    Raises:
        TypeError: If ``width`` or ``scale`` have invalid types.
        ValueError: If ``style`` name is unsupported or numeric values are invalid.
    """
    theme = _resolve_theme(style)
    theme_map = theme.as_dict()
    min_width = _validate_width(width)
    scale_factor = _validate_scale(scale)

    if len(circuit) == 0:
        return _empty_svg(theme_map, scale_factor, min_width)

    num_qubits = max(circuit.num_qubits, 1)
    num_clbits = circuit.num_clbits
    labels = _lane_labels(num_qubits, num_clbits)
    lane_count = len(labels)
    lane_ys: list[float] = []
    for idx in range(lane_count):
        lane_ys.append(float(_TOP_MARGIN + idx * _LANE_GAP))

    instruction_layers = _build_instruction_layers(
        circuit.instructions,
        num_qubits,
        lane_count,
    )
    cursor_x = float(_LEFT_MARGIN + _LABEL_WIDTH)
    columns: list[tuple[float, float, tuple[CircuitInstruction, ...]]] = []
    for layer in instruction_layers:
        column_width = max(
            max(64.0, _instruction_width(instruction) * 9.5) for instruction in layer
        )
        center_x = cursor_x + column_width / 2
        columns.append((cursor_x, center_x, layer))
        cursor_x += column_width + _COLUMN_PADDING

    content_width = cursor_x - _COLUMN_PADDING + _RIGHT_MARGIN
    base_width = (
        max(content_width, min_width) if min_width is not None else content_width
    )
    base_height = (
        float(_TOP_MARGIN + _BOTTOM_MARGIN + (lane_count - 1) * _LANE_GAP)
        if lane_count
        else 120.0
    )

    svg_width = base_width * scale_factor
    svg_height = base_height * scale_factor

    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width:.1f}" '
        f'height="{svg_height:.1f}" '
        f'viewBox="0 0 {base_width:.1f} {base_height:.1f}">'
    )
    background_rect = (
        f'<rect x="0" y="0" width="{base_width:.1f}" '
        f'height="{base_height:.1f}" fill="{theme_map["background"]}" />'
    )
    defs = (
        "<defs>"
        '<filter id="quspect_gate_shadow" x="-20%" y="-30%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="1.2" stdDeviation="1.2" flood-opacity="0.14"/>'
        "</filter>"
        "</defs>"
    )
    parts: list[str] = [svg_open, defs, background_rect, "<g>"]

    wire_start = _LEFT_MARGIN + _LABEL_WIDTH - 6
    wire_end = base_width - _RIGHT_MARGIN
    for lane_index, label in enumerate(labels):
        y = lane_ys[lane_index]
        is_classical = lane_index >= num_qubits
        stroke = theme_map["classical_wire"] if is_classical else theme_map["wire"]
        dash = ' stroke-dasharray="7 5"' if is_classical else ""
        parts.append(
            f'<text x="{_LEFT_MARGIN:.1f}" y="{y + 5:.1f}" fill="{theme_map["text"]}" '
            f'font-family="{_FONT_FAMILY}" font-size="14" font-weight="600">'
            f"{escape(label)}:</text>"
        )
        wire_line = (
            f'<line x1="{wire_start:.1f}" y1="{y:.1f}" '
            f'x2="{wire_end:.1f}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="{_WIRE_STROKE}" '
            f'stroke-linecap="round" opacity="0.9"{dash} />'
        )
        parts.append(wire_line)

    for _, center_x, layer in columns:
        for instruction in layer:
            rendered = _render_instruction(
                instruction,
                center_x,
                lane_ys,
                num_qubits,
                theme_map,
            )
            parts.extend(rendered)

    parts.append("</g></svg>")
    return CircuitSVG("".join(parts))


def _build_instruction_layers(
    instructions: Sequence[CircuitInstruction],
    num_qubits: int,
    lane_count: int,
) -> tuple[tuple[CircuitInstruction, ...], ...]:
    """Pack independent instructions into shared visual columns.

    Each lane tracks the next column available to it.  Multi-lane instructions
    reserve every lane between their endpoints so that a connector cannot pass
    through an unrelated gate.  This follows the lane-frontier layout used by
    MindQuantum's SVG ``GateContainer`` while normalizing each frontier to an
    exact column, which keeps parallel gates vertically aligned.
    """
    next_layer = [0] * lane_count
    layers: list[list[CircuitInstruction]] = []

    for instruction in instructions:
        occupied_lanes = _instruction_lane_span(instruction, num_qubits)
        layer_index = max(
            (next_layer[lane] for lane in occupied_lanes),
            default=0,
        )
        while len(layers) <= layer_index:
            layers.append([])
        layers[layer_index].append(instruction)

        for lane in occupied_lanes:
            next_layer[lane] = layer_index + 1

    return tuple(tuple(layer) for layer in layers)


def _instruction_lane_span(
    instruction: CircuitInstruction,
    num_qubits: int,
) -> tuple[int, ...]:
    """Return the contiguous visual lane span occupied by an instruction."""
    quantum_lanes = instruction.qubits
    if instruction.is_barrier and not quantum_lanes:
        quantum_lanes = tuple(range(num_qubits))

    lanes = list(quantum_lanes)
    if instruction.is_measure:
        lanes.extend(num_qubits + clbit for clbit in instruction.clbits)
    if not lanes:
        return ()
    return tuple(range(min(lanes), max(lanes) + 1))


def _render_instruction(
    instruction: CircuitInstruction,
    center_x: float,
    lane_ys: Sequence[float],
    num_qubits: int,
    theme: dict[str, str],
) -> list[str]:
    parts: list[str] = []
    if instruction.is_gate and instruction.gate is not None:
        rendered = _render_gate(
            instruction.gate,
            instruction.qubits,
            center_x,
            lane_ys,
            theme,
        )
        parts.extend(rendered)
    elif instruction.is_measure:
        parts.extend(_render_measure(instruction, center_x, lane_ys, num_qubits, theme))
    elif instruction.is_barrier:
        parts.extend(_render_barrier(instruction, center_x, lane_ys, num_qubits, theme))
    elif instruction.is_channel:
        parts.extend(_render_channel(instruction, center_x, lane_ys, theme))
    return parts


def _render_gate(
    gate: Gate,
    qubits: tuple[int, ...],
    center_x: float,
    lane_ys: Sequence[float],
    theme: dict[str, str],
) -> list[str]:
    name = gate.name.lower()
    parts: list[str] = []

    if len(qubits) == 1:
        if name == "measure":
            parts.extend(_measure_box(center_x, lane_ys[qubits[0]], theme))
            return parts
        parts.extend(
            _gate_box(
                center_x,
                lane_ys[qubits[0]],
                _gate_label(gate),
                theme,
                gate.name == "measure",
            )
        )
        return parts

    low_y = lane_ys[min(qubits)]
    high_y = lane_ys[max(qubits)]

    if name == "swap" and len(qubits) == 2:
        connector = (
            f'<line x1="{center_x:.1f}" y1="{low_y:.1f}" '
            f'x2="{center_x:.1f}" y2="{high_y:.1f}" '
            f'stroke="{theme["control"]}" stroke-width="2.2" '
            f'stroke-linecap="round" />'
        )
        parts.append(connector)
        for qubit in qubits:
            parts.extend(_swap_mark(center_x, lane_ys[qubit], theme))
        return parts

    if name in _CONTROLLED_TARGET_LABELS:
        connector = (
            f'<line x1="{center_x:.1f}" y1="{low_y:.1f}" '
            f'x2="{center_x:.1f}" y2="{high_y:.1f}" '
            f'stroke="{theme["control"]}" stroke-width="2.4" '
            f'stroke-linecap="round" />'
        )
        parts.append(connector)
        controls = qubits[:-1]
        target = qubits[-1]
        for control in controls:
            control_circle = (
                f'<circle cx="{center_x:.1f}" cy="{lane_ys[control]:.1f}" '
                f'r="5.5" fill="{theme["control"]}" />'
            )
            parts.append(control_circle)
        if name in {"cx", "ccx"}:
            parts.extend(_target_plus(center_x, lane_ys[target], theme))
        else:
            parts.extend(
                _gate_box(
                    center_x,
                    lane_ys[target],
                    _controlled_target_label(gate),
                    theme,
                    False,
                )
            )
        return parts

    connector = (
        f'<line x1="{center_x:.1f}" y1="{low_y:.1f}" '
        f'x2="{center_x:.1f}" y2="{high_y:.1f}" '
        f'stroke="{theme["control"]}" stroke-width="2.2" '
        f'stroke-linecap="round" />'
    )
    parts.append(connector)
    label = _gate_label(gate)
    for qubit in qubits:
        parts.extend(_gate_box(center_x, lane_ys[qubit], label, theme, False))
    return parts


def _render_measure(
    instruction: CircuitInstruction,
    center_x: float,
    lane_ys: Sequence[float],
    num_qubits: int,
    theme: dict[str, str],
) -> list[str]:
    parts: list[str] = []
    for qubit, clbit in zip(instruction.qubits, instruction.clbits, strict=True):
        qy = lane_ys[qubit]
        cy = lane_ys[num_qubits + clbit]
        parts.extend(_measure_box(center_x, qy, theme))
        classical_wire = (
            f'<line x1="{center_x:.1f}" y1="{qy + 16:.1f}" '
            f'x2="{center_x:.1f}" y2="{cy - 8:.1f}" '
            f'stroke="{theme["classical_wire"]}" '
            f'stroke-width="{_WIRE_STROKE}" stroke-dasharray="5 4" '
            f'stroke-linecap="round" />'
        )
        classical_circle = (
            f'<circle cx="{center_x:.1f}" cy="{cy:.1f}" r="8" '
            f'fill="{theme["background"]}" '
            f'stroke="{theme["classical_wire"]}" stroke-width="1.5" />'
        )
        classical_label = (
            f'<text x="{center_x:.1f}" y="{cy + 4:.1f}" text-anchor="middle" '
            f'fill="{theme["text"]}" font-family="{_FONT_FAMILY}" '
            f'font-size="11">{clbit}</text>'
        )
        parts.append(classical_wire)
        parts.append(classical_circle)
        parts.append(classical_label)
    return parts


def _render_channel(
    instruction: CircuitInstruction,
    center_x: float,
    lane_ys: Sequence[float],
    theme: dict[str, str],
) -> list[str]:
    """Render a noise channel as a dashed rectangle with light-red fill."""
    from quspect.visualization.text_drawer import _channel_label

    channel = instruction.channel
    assert channel is not None
    qubits = instruction.qubits
    label = _channel_label(channel)
    parts: list[str] = []

    box_width = max(44.0, 20.0 + len(label) * 9.0)
    box_x = center_x - box_width / 2

    if len(qubits) > 1:
        low_y = lane_ys[min(qubits)]
        high_y = lane_ys[max(qubits)]
        connector = (
            f'<line x1="{center_x:.1f}" y1="{low_y:.1f}" '
            f'x2="{center_x:.1f}" y2="{high_y:.1f}" '
            f'stroke="{theme["channel_stroke"]}" stroke-width="2.0" '
            f'stroke-linecap="round" />'
        )
        parts.append(connector)

    for qubit in qubits:
        y = lane_ys[qubit]
        box_y = y - _GATE_HEIGHT / 2
        safe_label = escape(label)
        # Dashed border for noise channels
        chan_rect = (
            f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_width:.1f}" '
            f'height="{_GATE_HEIGHT:.1f}" rx="{_GATE_RADIUS}" '
            f'fill="{theme["channel_fill"]}" '
            f'stroke="{theme["channel_stroke"]}" stroke-width="1.8" '
            f'stroke-dasharray="5 3" filter="url(#quspect_gate_shadow)" />'
        )
        chan_text = (
            f'<text x="{center_x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'fill="{theme["text"]}" font-family="{_FONT_FAMILY}" '
            f'font-size="11" font-weight="600">{safe_label}</text>'
        )
        parts.append(chan_rect)
        parts.append(chan_text)

    return parts


def _render_barrier(
    instruction: CircuitInstruction,
    center_x: float,
    lane_ys: Sequence[float],
    num_qubits: int,
    theme: dict[str, str],
) -> list[str]:
    lanes = instruction.qubits or tuple(range(num_qubits or 1))
    parts: list[str] = []
    low_y = lane_ys[min(lanes)] - _GATE_HEIGHT / 2
    high_y = lane_ys[max(lanes)] + _GATE_HEIGHT / 2
    barrier_rect = (
        f'<rect x="{center_x - 5:.1f}" y="{low_y:.1f}" width="10" '
        f'height="{high_y - low_y:.1f}" fill="{theme["barrier"]}" '
        f'fill-opacity="0.38" rx="3" />'
    )
    parts.append(barrier_rect)
    return parts


def _gate_box(
    center_x: float,
    center_y: float,
    label: str,
    theme: dict[str, str],
    is_measure: bool,
) -> list[str]:
    safe_label = escape(label)
    box_width = max(36.0, 16.0 + len(label) * 8.5)
    box_x = center_x - box_width / 2
    box_y = center_y - _GATE_HEIGHT / 2
    fill = theme["measure_fill"] if is_measure else theme["gate_fill"]
    gate_rect = (
        f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_width:.1f}" '
        f'height="{_GATE_HEIGHT:.1f}" rx="{_GATE_RADIUS}" fill="{fill}" '
        f'stroke="{theme["gate_stroke"]}" stroke-width="2.0" '
        f'filter="url(#quspect_gate_shadow)" />'
    )
    gate_text = (
        f'<text x="{center_x:.1f}" y="{center_y + 4:.1f}" text-anchor="middle" '
        f'fill="{theme["text"]}" font-family="{_FONT_FAMILY}" '
        f'font-size="12" font-weight="700">{safe_label}</text>'
    )
    return [gate_rect, gate_text]


def _measure_box(center_x: float, center_y: float, theme: dict[str, str]) -> list[str]:
    box_width = 40.0
    box_x = center_x - box_width / 2
    box_y = center_y - _GATE_HEIGHT / 2
    icon = theme["channel_stroke"]
    return [
        (
            f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_width:.1f}" '
            f'height="{_GATE_HEIGHT:.1f}" rx="{_GATE_RADIUS}" '
            f'fill="{theme["measure_fill"]}" stroke="{icon}" stroke-width="2.0" '
            f'filter="url(#quspect_gate_shadow)" />'
        ),
        (
            f'<path d="M {center_x - 10:.1f} {center_y + 5:.1f} '
            f'A 10 10 0 0 1 {center_x + 10:.1f} {center_y + 5:.1f}" '
            f'fill="none" stroke="{icon}" stroke-width="2.2" stroke-linecap="round" />'
        ),
        (
            f'<line x1="{center_x:.1f}" y1="{center_y + 4:.1f}" '
            f'x2="{center_x + 7:.1f}" y2="{center_y - 7:.1f}" '
            f'stroke="{icon}" stroke-width="2.2" stroke-linecap="round" />'
        ),
        f'<circle cx="{center_x:.1f}" cy="{center_y + 6:.1f}" r="2.2" fill="{icon}" />',
    ]


def _target_plus(center_x: float, center_y: float, theme: dict[str, str]) -> list[str]:
    r = 15.0
    return [
        (
            f'<circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="{r:.1f}" '
            f'fill="{theme["background"]}" stroke="{theme["control"]}" '
            f'stroke-width="2.4" filter="url(#quspect_gate_shadow)" />'
        ),
        (
            f'<line x1="{center_x - 8:.1f}" y1="{center_y:.1f}" '
            f'x2="{center_x + 8:.1f}" y2="{center_y:.1f}" '
            f'stroke="{theme["control"]}" stroke-width="2.4" stroke-linecap="round" />'
        ),
        (
            f'<line x1="{center_x:.1f}" y1="{center_y - 8:.1f}" '
            f'x2="{center_x:.1f}" y2="{center_y + 8:.1f}" '
            f'stroke="{theme["control"]}" stroke-width="2.4" stroke-linecap="round" />'
        ),
    ]


def _swap_mark(center_x: float, center_y: float, theme: dict[str, str]) -> list[str]:
    size = 8.0
    return [
        (
            f'<line x1="{center_x - size:.1f}" y1="{center_y - size:.1f}" '
            f'x2="{center_x + size:.1f}" y2="{center_y + size:.1f}" '
            f'stroke="{theme["control"]}" stroke-width="2.4" stroke-linecap="round" />'
        ),
        (
            f'<line x1="{center_x - size:.1f}" y1="{center_y + size:.1f}" '
            f'x2="{center_x + size:.1f}" y2="{center_y - size:.1f}" '
            f'stroke="{theme["control"]}" stroke-width="2.4" stroke-linecap="round" />'
        ),
    ]


def _resolve_theme(style: str | dict[str, str] | SvgStyle | None) -> SvgStyle:
    if style is None:
        return _SVG_THEMES["official"]
    if isinstance(style, SvgStyle):
        return style
    if isinstance(style, str):
        if style not in _SVG_THEMES:
            supported_styles = sorted(_SVG_THEMES)
            raise ValueError(
                f"Unsupported SVG style {style!r}. Supported styles: {supported_styles}"
            )
        return _SVG_THEMES[style]
    return SvgStyle.from_mapping(style)


def _validate_width(width: int | float | None) -> float | None:
    if width is None:
        return None
    if not isinstance(width, (int, float)):
        raise TypeError(f"width must be int or float, got {type(width)}")
    if width <= 0:
        raise ValueError("width must be positive")
    return float(width)


def _validate_scale(scale: float | None) -> float:
    if scale is None:
        return 1.0
    if not isinstance(scale, (int, float)):
        raise TypeError(f"scale must be int or float, got {type(scale)}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    return float(scale)


def _empty_svg(
    theme: dict[str, str],
    scale_factor: float,
    min_width: float | None,
) -> CircuitSVG:
    base_width = max(min_width or 220.0, 220.0)
    base_height = 72.0
    svg_width = base_width * scale_factor
    svg_height = base_height * scale_factor
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width:.1f}" '
        f'height="{svg_height:.1f}" '
        f'viewBox="0 0 {base_width:.1f} {base_height:.1f}">'
        f'<rect x="0" y="0" width="{base_width:.1f}" '
        f'height="{base_height:.1f}" fill="{theme["background"]}" />'
        f'<text x="20" y="42" fill="{theme["text"]}" '
        f'font-family="{_FONT_FAMILY}" font-size="16" font-weight="600">'
        "(empty circuit)</text>"
        "</svg>"
    )
    return CircuitSVG(content)
