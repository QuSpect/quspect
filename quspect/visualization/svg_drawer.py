"""SVG-based circuit visualization.

The SVG drawing speaks the same language as the text drawing of
:mod:`quspect.visualization.text_drawer`, with finer lines and one accent
colour:

* a gate is a rounded card on its wire; a gate on several qubits is one
  card, and a wire it does not act on runs through it dashed;
* a dot is a control, a circled cross the target of CX / Toffoli, a small
  cross a swap;
* a measurement names the classical bit it writes (``M→c0``); there are no
  classical wires;
* a classically controlled gate carries its condition (``c0=1``) set into
  the top edge of its card, or on a small flag above its topmost symbol;
* a noise channel is a card with a dashed outline, a barrier a dotted
  vertical line with its column to itself.

Two line weights carry the hierarchy: what the circuit does (cards,
connectors, symbols) is drawn in ink, the scaffolding (wires, barriers) in a
thinner grey.

Measurements and conditions, the two places where classical bits appear, are
drawn in the accent colour (the ``classical_wire`` entry of a style).  A
style may also give a colour to each family of gates (Pauli, Hadamard-like,
phase, rotation, two-qubit interaction): it outlines their cards and tints
their fill.  Everything else uses the ink of the theme.

The document is kept small and cheap to render: no filters, shared
attributes are set once on a group, and all wires, all connectors, all
control dots and all symbol strokes are one ``<path>`` each.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from quspect.visualization.text_drawer import (
    _ALL_CONTROL_GATES,
    _CONTROLLED_TARGET_LABELS,
    _SYMMETRIC_GATES,
    _XOR_TARGET_GATES,
    _channel_label,
    _condition_text,
    _controlled_target_label,
    _gate_label,
    _pack_layers,
)

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate


# Families of gates that a style may give a colour of their own.
_FAMILY_GATES: dict[str, frozenset[str]] = {
    "pauli": frozenset({"id", "x", "y", "z"}),
    "hadamard": frozenset({"h", "sx", "sxdg"}),
    "phase": frozenset({"s", "sdg", "t", "tdg", "p", "u1", "rz", "cp", "crz"}),
    "rotation": frozenset({"rx", "ry", "u2", "u3", "crx", "cry", "cu3"}),
    "interaction": frozenset({"rxx", "ryy", "rzz"}),
}
_FAMILY_OF_GATE = {
    name: family for family, names in _FAMILY_GATES.items() for name in names
}
_REQUIRED_STYLE_KEYS = (
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
)
_FAMILY_STYLE_KEYS = tuple(f"{family}_gate" for family in _FAMILY_GATES)


@dataclass(frozen=True)
class SvgStyle:
    """Color palette for SVG circuit rendering.

    Attributes:
        background: Fill of the page.
        wire: Qubit wires.
        classical_wire: Accent for everything classical: the outline and
            label of measurements, and the conditions of classically
            controlled gates.
        gate_fill: Fill of gate cards.
        gate_stroke: Outline of gate cards.
        text: Gate labels and qubit labels.
        control: Control dots, targets, swap marks and the lines joining them.
        barrier: Barriers.
        measure_fill: Fill of measurement cards.
        channel_fill: Fill of noise-channel cards.
        channel_stroke: Dashed outline of noise-channel cards.
        pauli_gate: Colour of the cards of X, Y, Z and I.
        hadamard_gate: Colour of the cards of H, √X and √X†.
        phase_gate: Colour of the cards of S, T, their inverses, P, U1, RZ
            and of the targets of CP and CRZ.
        rotation_gate: Colour of the cards of RX, RY, U2, U3 and of the
            targets of CRX, CRY and CU3.
        interaction_gate: Colour of the cards of RXX, RYY and RZZ.

    A family colour outlines the cards of its gates and tints their fill.
    Left empty (the default), the family is drawn like any other gate, with
    ``gate_fill`` and ``gate_stroke``.
    """

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
    pauli_gate: str = ""
    hadamard_gate: str = ""
    phase_gate: str = ""
    rotation_gate: str = ""
    interaction_gate: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return this style as a plain mapping (family colours only if set)."""
        mapping = {key: getattr(self, key) for key in _REQUIRED_STYLE_KEYS}
        for key in _FAMILY_STYLE_KEYS:
            if getattr(self, key):
                mapping[key] = getattr(self, key)
        return mapping

    @classmethod
    def from_mapping(cls, mapping: dict[str, str]) -> SvgStyle:
        """Construct a style from a mapping with strict key validation.

        The eleven basic colours are required; the family colours
        (``pauli_gate``, ``hadamard_gate``, ...) are optional.
        """
        missing = sorted(set(_REQUIRED_STYLE_KEYS) - set(mapping))
        if missing:
            raise ValueError(f"Missing SVG style keys: {missing}")
        unknown = sorted(
            set(mapping) - set(_REQUIRED_STYLE_KEYS) - set(_FAMILY_STYLE_KEYS)
        )
        if unknown:
            raise ValueError(f"Unknown SVG style keys: {unknown}")
        return cls(**{key: mapping[key] for key in mapping})

    def family_colour(self, gate_name: str) -> str:
        """Colour of the family ``gate_name`` belongs to ("" for none)."""
        family = _FAMILY_OF_GATE.get(gate_name.lower())
        return getattr(self, f"{family}_gate") if family else ""


# Each theme is one ink (labels, controls, noise outlines), a grey for the
# scaffolding (wires, barriers), one accent for what is classical, and one
# colour per family of gates.  "mono" is "official" without family colours.
_MONO: dict[str, str] = {
    "background": "#ffffff",
    "wire": "#8a93a6",
    "classical_wire": "#3350d6",
    "gate_fill": "#ffffff",
    "gate_stroke": "#1f2433",
    "text": "#1f2433",
    "control": "#1f2433",
    "barrier": "#8a93a6",
    "measure_fill": "#f4f6ff",
    "channel_fill": "#ffffff",
    "channel_stroke": "#1f2433",
}
_SVG_THEME_DATA: dict[str, dict[str, str]] = {
    "official": {
        **_MONO,
        "pauli_gate": "#c2255c",
        "hadamard_gate": "#d9480f",
        "phase_gate": "#0c8599",
        "rotation_gate": "#7048e8",
        "interaction_gate": "#2b8a3e",
    },
    "mono": dict(_MONO),
    "light": {
        "background": "#f8f6fc",
        "wire": "#8d84a3",
        "classical_wire": "#8f5300",
        "gate_fill": "#ffffff",
        "gate_stroke": "#180835",
        "text": "#180835",
        "control": "#180835",
        "barrier": "#8d84a3",
        "measure_fill": "#fdf3dc",
        "channel_fill": "#ffffff",
        "channel_stroke": "#180835",
        "pauli_gate": "#c2255c",
        "hadamard_gate": "#1864ab",
        "phase_gate": "#0b7285",
        "rotation_gate": "#6741d9",
        "interaction_gate": "#2b8a3e",
    },
    "dark": {
        "background": "#180835",
        "wire": "#8d84a3",
        "classical_wire": "#e9b645",
        "gate_fill": "#231045",
        "gate_stroke": "#cfc6e6",
        "text": "#f4f1fb",
        "control": "#cfc6e6",
        "barrier": "#8d84a3",
        "measure_fill": "#311d37",
        "channel_fill": "#231045",
        "channel_stroke": "#cfc6e6",
        "pauli_gate": "#ff8fab",
        "hadamard_gate": "#74c0fc",
        "phase_gate": "#63e6be",
        "rotation_gate": "#b197fc",
        "interaction_gate": "#8ce99a",
    },
}
_SVG_THEMES: dict[str, SvgStyle] = {
    name: SvgStyle.from_mapping(style) for name, style in _SVG_THEME_DATA.items()
}

# Geometry, in user units (pixels at scale 1).
_PITCH = 44.0  # distance between two wires
_FIRST_WIRE = 32.0  # y of the first wire
_BELOW_LAST_WIRE = 30.0
_MARGIN = 16.0  # left and right of the drawing
_CARD_HEIGHT = 28.0
_CARD_RADIUS = 5.0
_CARD_PADDING = 9.0  # between a card's side and its label
_GAP = 16.0  # wire between two columns (as much as between two cards of a column)
_SLIM_GAP = 8.0  # ... between two columns of bare symbols
_SYMBOL_WIDTH = 20.0
_DOT_RADIUS = 4.0
_TARGET_RADIUS = 7.5
_SWAP_ARM = 5.5
# A barrier is a row of dots.  With 10 dots per wire pitch, starting 4.5
# steps above a wire, every wire passes midway between two dots.
_BARRIER_STEP = _PITCH / 10
_BARRIER_REACH = 4.5 * _BARRIER_STEP  # above and below a wire it covers
_FLAG_HEIGHT = 13.0  # from a wire up to the condition of a bare symbol
# Two line weights: ink (cards, connectors, symbols) and scaffolding (wires).
_INK_STROKE = "1.4"
_FINE_STROKE = "1"

_FONT_FAMILY = (
    "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
    "'DejaVu Sans Mono', 'Liberation Mono', monospace"
)
_FONT_SIZE = 13.0
_SMALL_FONT_SIZE = 11.0
# Every font of the stack is monospaced with an advance of 0.6 em, which is
# what lets cards be sized without measuring text.
_CHAR_WIDTH = 0.6 * _FONT_SIZE
_SMALL_CHAR_WIDTH = 0.6 * _SMALL_FONT_SIZE
_BASELINE_SHIFT = 0.34 * _FONT_SIZE  # from a text's middle down to its baseline
_SMALL_BASELINE_SHIFT = 0.34 * _SMALL_FONT_SIZE


def _n(value: float) -> str:
    """A coordinate with at most one decimal and no trailing zero."""
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


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


# ---------------------------------------------------------------------------
# Sheet: markup collected by kind, so that each kind is written once
# ---------------------------------------------------------------------------


@dataclass
class _Sheet:
    """Pieces of the drawing, grouped by what they share.

    Path data (``d`` fragments) of one list ends up in a single ``<path>``;
    elements of one list end up in a single ``<g>`` that carries their common
    attributes.  The lists are written in the order of the fields, from the
    back of the drawing to the front.
    """

    barriers: list[str] = field(default_factory=list)  # path data
    connectors: list[str] = field(default_factory=list)  # path data
    # <rect> by kind: "gate", "measure", "channel", or the colour of a family
    cards: dict[str, list[str]] = field(default_factory=dict)
    through: list[str] = field(default_factory=list)  # path data
    targets: list[str] = field(default_factory=list)  # <circle>
    dots: list[str] = field(default_factory=list)  # path data
    marks: list[str] = field(default_factory=list)  # path data
    swaps: list[str] = field(default_factory=list)  # path data
    flags: list[str] = field(default_factory=list)  # path data
    page_cuts: list[str] = field(default_factory=list)  # <rect>
    card_cuts: dict[str, list[str]] = field(default_factory=dict)  # as cards
    labels: list[str] = field(default_factory=list)  # <text>
    measure_labels: list[str] = field(default_factory=list)  # <text>
    numbers: list[str] = field(default_factory=list)  # <text>
    captions: list[str] = field(default_factory=list)  # <text>


@dataclass(frozen=True)
class _Shape:
    """One drawable item of a column.

    Attributes:
        lanes: Qubit lanes the shape occupies.
        width: Width the shape needs.
        emit: Adds the shape to a sheet; called with the sheet and the x of
            the shape's left side.
        clbits: Classical bits the shape reads or writes.
        writes: Whether it writes them (a measurement) or reads them.
        slim: Whether the shape is bare symbols.
    """

    lanes: tuple[int, ...]
    width: float
    emit: Callable[[_Sheet, float], None]
    clbits: tuple[int, ...] = ()
    writes: bool = False
    slim: bool = False


@dataclass(frozen=True)
class _Caption:
    """The condition of a classically controlled gate."""

    text: str  # "c0=1"
    title: str  # "rz if (c0, c1) == (1, 0)", shown as a tooltip

    @property
    def width(self) -> float:
        return len(self.text) * _SMALL_CHAR_WIDTH

    def markup(self, x: float, middle: float) -> str:
        """The caption as a ``<text>`` starting at ``x``, centred on ``middle``."""
        return (
            f'<text x="{_n(x)}" y="{_n(middle + _SMALL_BASELINE_SHIFT)}">'
            f"{escape(self.text)}<title>{escape(self.title)}</title></text>"
        )


def _wire_y(qubit: int) -> float:
    return _FIRST_WIRE + qubit * _PITCH


def _span(lanes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(range(min(lanes), max(lanes) + 1))


def _text(x: float, middle: float, content: str, shift: float) -> str:
    return f'<text x="{_n(x)}" y="{_n(middle + shift)}">{escape(content)}</text>'


# A caption interrupts the top edge of its card: the edge resumes
# _CAPTION_MARGIN before and after the text, and the gap starts
# _CAPTION_INSET from the card's side, clear of the rounded corner.
_CAPTION_MARGIN = 3.0
_CAPTION_INSET = _CARD_RADIUS + 3.0


def _card_width(label: str, caption: _Caption | None = None) -> float:
    width = max(_CARD_HEIGHT, len(label) * _CHAR_WIDTH + 2 * _CARD_PADDING)
    if caption is not None:
        # the caption starts after the corner and leaves a piece of edge
        width = max(width, caption.width + 2 * _CAPTION_MARGIN + 2 * _CAPTION_INSET)
    return width


def _emit_card(
    sheet: _Sheet,
    left: float,
    width: float,
    qubits: tuple[int, ...],
    label: str,
    *,
    kind: str = "gate",
    numbered: bool = False,
    caption: _Caption | None = None,
) -> None:
    """Add a rounded card over ``qubits`` to the sheet.

    Args:
        sheet: Sheet to draw on.
        left: x of the card's left side.
        width: Width of the card.
        qubits: Operands, in the order the gate takes them.  The card covers
            every wire from the first to the last of them; a wire in between
            that is not an operand runs through, dashed.
        label: Text shown in the card.
        kind: ``"gate"``, ``"measure"``, ``"channel"``, or the colour of the
            gate's family.
        numbered: Whether to print each operand's position where its wire
            enters the card.
        caption: Condition set into the top edge.
    """
    top = _wire_y(min(qubits)) - _CARD_HEIGHT / 2
    bottom = _wire_y(max(qubits)) + _CARD_HEIGHT / 2
    rect = (
        f'<rect x="{_n(left)}" y="{_n(top)}" width="{_n(width)}" '
        f'height="{_n(bottom - top)}" rx="{_n(_CARD_RADIUS)}"/>'
    )
    sheet.cards.setdefault(kind, []).append(rect)

    operands = set(qubits)
    for qubit in range(min(qubits) + 1, max(qubits)):
        if qubit not in operands:
            sheet.through.append(f"M{_n(left)} {_n(_wire_y(qubit))}h{_n(width)}")
    if numbered:
        for position, qubit in enumerate(qubits):
            sheet.numbers.append(
                _text(left + 4, _wire_y(qubit), str(position), _SMALL_BASELINE_SHIFT)
            )

    middle = (top + bottom) / 2
    lanes_covered = max(qubits) - min(qubits) + 1
    middle_qubit = (min(qubits) + max(qubits)) // 2
    if lanes_covered % 2 == 1 and middle_qubit not in operands:
        middle -= _PITCH / 2  # keep the label off a wire that only runs through
    labels = sheet.measure_labels if kind == "measure" else sheet.labels
    labels.append(_text(left + width / 2, middle, label, _BASELINE_SHIFT))

    if caption is not None:
        # Erase the edge under the caption: its outer half with the colour of
        # the page, its inner half with the colour of the card.
        gap_left = _n(left + _CAPTION_INSET)
        gap_width = _n(caption.width + 2 * _CAPTION_MARGIN)
        sheet.page_cuts.append(
            f'<rect x="{gap_left}" y="{_n(top - 0.8)}" width="{gap_width}" '
            f'height="0.8"/>'
        )
        sheet.card_cuts.setdefault(kind, []).append(
            f'<rect x="{gap_left}" y="{_n(top)}" width="{gap_width}" height="0.8"/>'
        )
        sheet.captions.append(
            caption.markup(left + _CAPTION_INSET + _CAPTION_MARGIN, top)
        )


def _card_shape(
    qubits: tuple[int, ...],
    label: str,
    *,
    kind: str = "gate",
    numbered: bool = False,
    caption: _Caption | None = None,
) -> _Shape:
    """A labelled card over ``qubits`` (one card even when they are not adjacent)."""
    numbered = numbered and len(qubits) > 1
    width = _card_width(label, caption)
    if numbered:
        width += 2 * (len(str(len(qubits) - 1)) * _SMALL_CHAR_WIDTH + 2)

    def emit(sheet: _Sheet, left: float) -> None:
        _emit_card(
            sheet,
            left,
            width,
            qubits,
            label,
            kind=kind,
            numbered=numbered,
            caption=caption,
        )

    return _Shape(lanes=_span(qubits), width=width, emit=emit)


def _emit_flag(sheet: _Sheet, x: float, y: float, reach: float, cap: _Caption) -> None:
    """A condition on a pole standing on the symbol at ``(x, y)``."""
    top = y - _FLAG_HEIGHT
    sheet.flags.append(f"M{_n(x)} {_n(y - reach)}V{_n(top)}h5")
    sheet.captions.append(cap.markup(x + 8, top))


def _flag_width(caption: _Caption) -> float:
    """Room a flag takes to the right of its symbol's centre."""
    return 8 + caption.width + 2


_SYMBOL_REACH = {"dot": _DOT_RADIUS, "target": _TARGET_RADIUS, "swap": _SWAP_ARM}


def _emit_symbol(sheet: _Sheet, kind: str, x: float, y: float) -> None:
    if kind == "dot":
        # a zero-length stroke with round caps is a disc as wide as the stroke
        sheet.dots.append(f"M{_n(x)} {_n(y)}h0")
    elif kind == "target":
        r = _TARGET_RADIUS
        sheet.targets.append(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{_n(r)}"/>')
        sheet.marks.append(
            f"M{_n(x - r)} {_n(y)}h{_n(2 * r)}M{_n(x)} {_n(y - r)}v{_n(2 * r)}"
        )
    else:
        a = _SWAP_ARM
        sheet.swaps.append(
            f"M{_n(x - a)} {_n(y - a)}l{_n(2 * a)} {_n(2 * a)}"
            f"M{_n(x - a)} {_n(y + a)}l{_n(2 * a)} -{_n(2 * a)}"
        )


def _symbol_shape(marks: dict[int, str], *, caption: _Caption | None = None) -> _Shape:
    """Symbols on several qubits joined by one vertical line.

    A caption flies as a flag from the topmost symbol.
    """
    top, bottom = min(marks), max(marks)
    half = _SYMBOL_WIDTH / 2
    width = _SYMBOL_WIDTH
    if caption is not None:
        width = half + max(half, _flag_width(caption))

    def emit(sheet: _Sheet, left: float) -> None:
        x = left + half
        sheet.connectors.append(f"M{_n(x)} {_n(_wire_y(top))}V{_n(_wire_y(bottom))}")
        for qubit, kind in marks.items():
            _emit_symbol(sheet, kind, x, _wire_y(qubit))
        if caption is not None:
            _emit_flag(sheet, x, _wire_y(top), _SYMBOL_REACH[marks[top]], caption)

    return _Shape(
        lanes=_span(list(marks)), width=width, emit=emit, slim=caption is None
    )


def _controlled_card_shape(
    controls: tuple[int, ...],
    target: int,
    label: str,
    *,
    kind: str = "gate",
    caption: _Caption | None = None,
) -> _Shape:
    """Control dots joined to a labelled card on the target.

    A caption goes on whichever is topmost: into the top edge of the card,
    or on a flag above the first control.
    """
    card_on_top = target < min(controls)
    card_caption = caption if card_on_top else None
    card_width = _card_width(label, card_caption)
    width = card_width
    if caption is not None and not card_on_top:
        width = max(width, card_width / 2 + _flag_width(caption))

    def emit(sheet: _Sheet, left: float) -> None:
        x = left + card_width / 2
        lanes = [target, *controls]
        sheet.connectors.append(
            f"M{_n(x)} {_n(_wire_y(min(lanes)))}V{_n(_wire_y(max(lanes)))}"
        )
        _emit_card(
            sheet, left, card_width, (target,), label, kind=kind, caption=card_caption
        )
        for control in controls:
            _emit_symbol(sheet, "dot", x, _wire_y(control))
        if caption is not None and not card_on_top:
            _emit_flag(sheet, x, _wire_y(min(controls)), _DOT_RADIUS, caption)

    return _Shape(lanes=_span([target, *controls]), width=width, emit=emit)


def _gate_shape(
    gate: Gate,
    qubits: tuple[int, ...],
    theme: SvgStyle,
    caption: _Caption | None = None,
) -> _Shape:
    name = gate.name.lower()
    kind = theme.family_colour(name) or "gate"
    if len(qubits) == 1:
        return _card_shape(qubits, _gate_label(gate), kind=kind, caption=caption)
    if name == "swap" and len(qubits) == 2:
        return _symbol_shape(dict.fromkeys(qubits, "swap"), caption=caption)
    if name in _ALL_CONTROL_GATES:
        return _symbol_shape(dict.fromkeys(qubits, "dot"), caption=caption)
    if name in _XOR_TARGET_GATES:
        marks = dict.fromkeys(qubits[:-1], "dot")
        marks[qubits[-1]] = "target"
        return _symbol_shape(marks, caption=caption)
    if name in _CONTROLLED_TARGET_LABELS:
        return _controlled_card_shape(
            qubits[:-1],
            qubits[-1],
            _controlled_target_label(gate),
            kind=kind,
            caption=caption,
        )
    return _card_shape(
        qubits,
        _gate_label(gate),
        kind=kind,
        numbered=name not in _SYMMETRIC_GATES,
        caption=caption,
    )


def _barrier_shape(qubits: tuple[int, ...]) -> _Shape:
    """A dotted vertical line over the qubits of the barrier.

    It has its column to itself from its first to its last qubit.  It stops
    well short of the wire of a qubit in between that it does not cover, so
    that it reads as two barriers there, not as one.
    """
    covered = sorted(set(qubits))
    lanes = _span(covered)
    runs: list[tuple[int, int]] = []  # first and last qubit of each stretch
    for qubit in covered:
        if runs and runs[-1][1] == qubit - 1:
            runs[-1] = (runs[-1][0], qubit)
        else:
            runs.append((qubit, qubit))

    def emit(sheet: _Sheet, left: float) -> None:
        for first, last in runs:
            top = _wire_y(first) - _BARRIER_REACH
            bottom = _wire_y(last) + _BARRIER_REACH
            sheet.barriers.append(f"M{_n(left + 1)} {_n(top)}V{_n(bottom)}")

    return _Shape(lanes=lanes, width=2.0, emit=emit)


def _condition_title(instruction: CircuitInstruction) -> str:
    """The condition in words: ``rz if (c0, c1) == (1, 0)``."""
    gate = instruction.conditional_gate
    assert gate is not None and instruction.condition is not None
    bits = ", ".join(f"c{clbit}" for clbit in instruction.clbits)
    values = ", ".join(
        str((instruction.condition >> position) & 1)
        for position in range(len(instruction.clbits))
    )
    return f"{gate.name} if ({bits}) == ({values})"


def _shapes(
    instruction: CircuitInstruction, num_qubits: int, theme: SvgStyle
) -> list[_Shape]:
    if instruction.is_gate and instruction.gate is not None:
        return [_gate_shape(instruction.gate, instruction.qubits, theme)]
    if instruction.is_measure:
        shapes = []
        for qubit, clbit in zip(instruction.qubits, instruction.clbits, strict=True):
            card = _card_shape((qubit,), f"M→c{clbit}", kind="measure")
            shapes.append(
                _Shape(card.lanes, card.width, card.emit, clbits=(clbit,), writes=True)
            )
        return shapes
    if instruction.is_barrier:
        return [_barrier_shape(instruction.qubits or tuple(range(num_qubits)))]
    if instruction.is_channel and instruction.channel is not None:
        label = _channel_label(instruction.channel)
        return [_card_shape(instruction.qubits, label, kind="channel")]
    if instruction.is_conditional and instruction.conditional_gate is not None:
        assert instruction.condition is not None
        caption = _Caption(
            _condition_text(instruction.clbits, instruction.condition),
            _condition_title(instruction),
        )
        gate = _gate_shape(
            instruction.conditional_gate, instruction.qubits, theme, caption
        )
        return [_Shape(gate.lanes, gate.width, gate.emit, instruction.clbits)]
    return []


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def _path(data: list[str], attributes: str) -> str:
    return f'<path d="{"".join(data)}" {attributes}/>' if data else ""


def _group(elements: list[str], attributes: str) -> str:
    return f"<g {attributes}>{''.join(elements)}</g>" if elements else ""


def _mix(base: str, colour: str, share: float) -> str:
    """``base`` with ``share`` of ``colour`` mixed in (both ``#rrggbb``).

    Colours in another notation cannot be mixed: ``base`` is returned as is.
    """
    if not (len(base) == len(colour) == 7 and base[0] == colour[0] == "#"):
        return base
    try:
        channels = [
            round(
                int(base[i : i + 2], 16) * (1 - share)
                + int(colour[i : i + 2], 16) * share
            )
            for i in (1, 3, 5)
        ]
    except ValueError:
        return base
    return "#" + "".join(f"{channel:02x}" for channel in channels)


_FAMILY_TINT = 0.13  # share of a family's colour in the fill of its cards


def _card_fill(theme: SvgStyle, kind: str) -> str:
    if kind == "gate":
        return theme.gate_fill
    if kind == "measure":
        return theme.measure_fill
    if kind == "channel":
        return theme.channel_fill
    return _mix(theme.gate_fill, kind, _FAMILY_TINT)


def _card_attributes(theme: SvgStyle, kind: str) -> str:
    """Attributes shared by the cards of one kind."""
    fill = _card_fill(theme, kind)
    if kind == "channel":
        return (
            f'fill="{fill}" stroke="{theme.channel_stroke}" '
            f'stroke-width="1.2" stroke-dasharray="5 3.5"'
        )
    stroke = {"gate": theme.gate_stroke, "measure": theme.classical_wire}.get(
        kind, kind
    )
    return f'fill="{fill}" stroke="{stroke}" stroke-width="{_INK_STROKE}"'


def draw_svg_circuit(
    circuit: QuantumCircuit,
    *,
    style: str | dict[str, str] | SvgStyle | None = None,
    width: int | float | None = None,
    scale: float | None = None,
) -> CircuitSVG:
    """Render a quantum circuit as SVG.

    The drawing follows the text drawing of
    :func:`~quspect.visualization.draw_text_circuit`: rounded cards on the
    qubit wires, measurements that name the classical bit they write, and
    conditions written on the gates they control.

    Args:
        circuit: The circuit to render.
        style: Either a named theme (``"official"``, ``"light"``, ``"dark"``,
            or ``"mono"`` for ``"official"`` without a colour per family of
            gates) or a custom theme dictionary or :class:`SvgStyle`.
        width: Optional minimum SVG width in pixels.
        scale: Optional positive scale factor applied to the whole figure.

    Returns:
        A notebook-friendly SVG wrapper.

    Raises:
        TypeError: If ``width`` or ``scale`` have invalid types.
        ValueError: If ``style`` name is unsupported or numeric values are invalid.
    """
    theme = _resolve_theme(style)
    min_width = _validate_width(width)
    scale_factor = _validate_scale(scale)

    if len(circuit) == 0:
        return _empty_svg(theme, scale_factor, min_width)

    num_qubits = max(circuit.num_qubits, 1)
    groups = [
        _shapes(instruction, num_qubits, theme) for instruction in circuit.instructions
    ]
    layers = _pack_layers(groups, num_qubits, circuit.num_clbits)

    label_width = len(f"q{num_qubits - 1}") * _CHAR_WIDTH
    wire_start = _MARGIN + label_width + 8
    sheet = _Sheet()
    cursor = wire_start + _GAP
    previous_slim = False
    for index, layer in enumerate(layers):
        slim = all(shape.slim for shape in layer)
        if index and slim and previous_slim:
            cursor -= _GAP - _SLIM_GAP
        layer_width = max(shape.width for shape in layer)
        for shape in layer:
            shape.emit(sheet, cursor + (layer_width - shape.width) / 2)
        cursor += layer_width + _GAP
        previous_slim = slim

    base_width = cursor + _MARGIN
    if min_width is not None:
        base_width = max(base_width, min_width)
    wire_end = base_width - _MARGIN
    base_height = _FIRST_WIRE + (num_qubits - 1) * _PITCH + _BELOW_LAST_WIRE

    wires = [
        f"M{_n(wire_start)} {_n(_wire_y(qubit))}H{_n(wire_end)}"
        for qubit in range(num_qubits)
    ]
    qubit_labels = [
        _text(_MARGIN, _wire_y(qubit), f"q{qubit}", _BASELINE_SHIFT)
        for qubit in range(num_qubits)
    ]
    ink = f'stroke-width="{_INK_STROKE}"'
    fine = f'stroke-width="{_FINE_STROKE}"'
    small = f'font-size="{_n(_SMALL_FONT_SIZE)}"'
    accent = theme.classical_wire
    body = [
        _path(
            sheet.barriers,
            f'stroke="{theme.barrier}" stroke-width="1.6" '
            f'stroke-dasharray="0 {_n(_BARRIER_STEP)}"',
        ),
        _path(wires, f'stroke="{theme.wire}" {fine}'),
        _path(sheet.connectors, f'stroke="{theme.control}" {ink}'),
        *(
            _group(rects, _card_attributes(theme, kind))
            for kind, rects in sheet.cards.items()
        ),
        _path(sheet.through, f'stroke="{theme.wire}" {fine} stroke-dasharray="2 4"'),
        _group(
            sheet.targets,
            f'fill="{theme.background}" stroke="{theme.control}" {ink}',
        ),
        _path(
            sheet.dots,
            f'stroke="{theme.control}" stroke-width="{_n(2 * _DOT_RADIUS)}"',
        ),
        _path(sheet.marks, f'stroke="{theme.control}" {ink}'),
        _path(sheet.swaps, f'stroke="{theme.control}" stroke-width="1.6"'),
        _path(sheet.flags, f'stroke="{accent}" stroke-width="1.2"'),
        _group(sheet.page_cuts, f'fill="{theme.background}"'),
        *(
            _group(rects, f'fill="{_card_fill(theme, kind)}"')
            for kind, rects in sheet.card_cuts.items()
        ),
        # qubit labels and operand numbers: the ink, held back
        _group(qubit_labels, f'fill="{theme.text}" fill-opacity="0.7"'),
        _group(sheet.labels, f'fill="{theme.text}" text-anchor="middle"'),
        _group(sheet.measure_labels, f'fill="{accent}" text-anchor="middle"'),
        _group(sheet.numbers, f'fill="{theme.text}" fill-opacity="0.7" {small}'),
        _group(sheet.captions, f'fill="{accent}" {small}'),
    ]
    content = (
        f"{_svg_open(base_width, base_height, scale_factor)}"
        f'<rect width="{_n(base_width)}" height="{_n(base_height)}" '
        f'fill="{theme.background}"/>'
        f'<g fill="none" stroke-linecap="round" stroke-linejoin="round" '
        f'font-family="{_FONT_FAMILY}" font-size="{_n(_FONT_SIZE)}" '
        f'font-weight="500">'
        f"{''.join(body)}</g></svg>"
    )
    return CircuitSVG(content)


def _svg_open(base_width: float, base_height: float, scale_factor: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_n(base_width * scale_factor)}" '
        f'height="{_n(base_height * scale_factor)}" '
        f'viewBox="0 0 {_n(base_width)} {_n(base_height)}">'
    )


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
    theme: SvgStyle,
    scale_factor: float,
    min_width: float | None,
) -> CircuitSVG:
    base_width = max(min_width or 220.0, 220.0)
    base_height = 72.0
    content = (
        f"{_svg_open(base_width, base_height, scale_factor)}"
        f'<rect width="{_n(base_width)}" height="{_n(base_height)}" '
        f'fill="{theme.background}"/>'
        f'<text x="{_n(_MARGIN)}" y="{_n(base_height / 2 + _BASELINE_SHIFT)}" '
        f'fill="{theme.wire}" font-family="{_FONT_FAMILY}" '
        f'font-size="{_n(_FONT_SIZE)}">(empty circuit)</text>'
        "</svg>"
    )
    return CircuitSVG(content)
