"""Text-based circuit visualization.

This module draws a :class:`QuantumCircuit` as rounded cards on wires, with
Unicode box-drawing characters and no third-party dependency::

         ╭───╮       ╭──────╮
    q0 ──┤ H ├───●───┤ M→c0 ├───────────────
         ╰───╯   │   ╰──────╯
                 │   ╭──────╮  ╭─ c0=1 ──╮
    q1 ──────────⊕───┤ M→c1 ├──┤    X    ├──
                     ╰──────╯  ╰─────────╯

What the drawing shows:

* a gate is a card on its wire; a gate on several qubits is one card, and a
  wire it does not act on runs through it dashed (``┄``);
* ``●`` is a control, ``⊕`` the target of CX / Toffoli, ``╳`` a swap;
* a measurement names the classical bit it writes (``M→c0``); there are no
  classical wires;
* a classically controlled gate carries its condition as ``╭─ c0=1``: in the
  top edge of its card, or as a flag on its topmost symbol;
* a noise channel is a card with a dashed outline (``╌``, ``╎``);
* a barrier is a dashed vertical line (``┆``) with its column to itself.

Each family of dashes has one meaning: ``╌ ╎`` outline noise, ``┆`` is a
barrier, ``┄`` is a wire seen faintly (behind a card, or going on in another
part of a folded drawing).

How a drawing is built:

1. Every instruction becomes one or more *elements* (a card, a control with
   its target, a measurement, a barrier, ...).  An element knows which lanes
   it occupies, how wide it is and how to paint itself.
2. Elements are packed into *layers*: each one goes into the earliest column
   in which all the lanes it occupies are still free, so gates on disjoint
   qubits share a column.
3. The layers are painted onto a canvas.  Lines are stored as segments, and
   the segments that meet in a cell merge into one box-drawing character:
   that is what joins a control line to the edge of a card (``┴``) or lets it
   cross a wire (``┼``).
4. The canvas is cut between layers into parts that fit the display width.
   Each part is printed with its own lane labels; rows it does not need
   are left out, so an idle qubit takes one line.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from quspect.circuit.parameter import Parameter

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate


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

# Gates drawn with symbols instead of a labelled target card.
_XOR_TARGET_GATES = frozenset({"cx", "ccx"})
_ALL_CONTROL_GATES = frozenset({"cz"})
# Multi-qubit gates that act the same on every operand: their card needs no
# operand numbers.
_SYMMETRIC_GATES = frozenset({"rxx", "ryy", "rzz"})

_CONTROL = "●"
_XOR_TARGET = "⊕"
_SWAP_MARK = "╳"
_BARRIER_MARK = "┆"
_PASS_THROUGH = "┄"
_WRITES = "→"
_FADE = "┄"  # a wire that goes on in another part of the drawing

_DEFAULT_WIDTH = 80
_JUPYTER_DEFAULT_WIDTH = 115
_SYMBOL_WIDTH = 3
# Columns of bare wire between two layers and at both ends of the drawing.
_GAP = 2


# ---------------------------------------------------------------------------
# Canvas: cells hold line segments that merge into box-drawing characters
# ---------------------------------------------------------------------------

# Segments of a cell: does a line leave it upwards, downwards, to the left,
# to the right?
_Segments = tuple[bool, bool, bool, bool]

_NO_SEGMENTS: _Segments = (False, False, False, False)
_GLYPHS: dict[_Segments, str] = {
    (False, False, False, False): " ",
    (False, False, True, True): "─",
    (True, True, False, False): "│",
    (False, True, False, True): "┌",
    (False, True, True, False): "┐",
    (True, False, False, True): "└",
    (True, False, True, False): "┘",
    (True, True, False, True): "├",
    (True, True, True, False): "┤",
    (False, True, True, True): "┬",
    (True, False, True, True): "┴",
    (True, True, True, True): "┼",
    (True, False, False, False): "╵",
    (False, True, False, False): "╷",
    (False, False, True, False): "╴",
    (False, False, False, True): "╶",
}

# How much a cell counts when deciding whether its row is worth a line.
_BLANK = 0  # nothing, or a mark that can be left out (barrier off the wires)
_UPRIGHT = 1  # a vertical line passing through
_SOLID = 2  # an edge, a label, a junction

_WIRE: _Segments = (False, False, True, True)
_VERTICAL: _Segments = (True, True, False, False)
_CROSSING: _Segments = (True, True, True, True)
_WIRE_ENTERS: _Segments = (True, True, True, False)
_WIRE_LEAVES: _Segments = (True, True, False, True)


class _Canvas:
    """A grid of cells holding either line segments or a literal character."""

    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._segments: list[list[_Segments]] = [
            [_NO_SEGMENTS] * num_cols for _ in range(num_rows)
        ]
        self._text: list[list[str | None]] = [
            [None] * num_cols for _ in range(num_rows)
        ]
        self._weight: list[list[int]] = [[_SOLID] * num_cols for _ in range(num_rows)]

    def set(self, row: int, col: int, segments: _Segments) -> None:
        """Replace whatever the cell holds with ``segments``."""
        self._segments[row][col] = segments
        self._text[row][col] = None

    def add(self, row: int, col: int, segments: _Segments) -> None:
        """Merge ``segments`` into the cell."""
        old = self._segments[row][col]
        self._segments[row][col] = (
            old[0] or segments[0],
            old[1] or segments[1],
            old[2] or segments[2],
            old[3] or segments[3],
        )

    def put(self, row: int, col: int, text: str, *, weight: int = _SOLID) -> None:
        """Write literal characters starting at the cell; they hide segments.

        ``weight`` says how much the characters count when deciding whether
        their row is worth a line (see :meth:`row_weight`).
        """
        for offset, char in enumerate(text):
            self._text[row][col + offset] = char
            self._weight[row][col + offset] = _BLANK if char == " " else weight

    def wire(self, row: int) -> None:
        """Draw a wire along the whole row."""
        for col in range(self.num_cols):
            self.set(row, col, _WIRE)

    def vline(self, col: int, top: int, bottom: int) -> None:
        """Draw a vertical line from row ``top`` to row ``bottom``.

        The ends only reach into their cells, so a line that stops on the
        edge of a card forms a T-junction rather than a crossing.
        """
        for row in range(top, bottom + 1):
            self.add(row, col, (row > top, row < bottom, False, False))

    def row_weight(self, row: int, start: int, stop: int) -> int:
        """What ``row`` shows in the given columns.

        Returns:
            ``_SOLID`` if it shows an edge, a label or a junction;
            ``_UPRIGHT`` if it only shows vertical lines passing through;
            ``_BLANK`` if it shows nothing, or only barrier marks.
        """
        segments = self._segments[row]
        text = self._text[row]
        heaviest = _BLANK
        for col in range(start, stop):
            if text[col] is not None:
                weight = self._weight[row][col]
            elif segments[col] == _NO_SEGMENTS:
                weight = _BLANK
            elif segments[col] == _VERTICAL:
                weight = _UPRIGHT
            else:
                return _SOLID
            heaviest = max(heaviest, weight)
        return heaviest

    def render_row(self, row: int, start: int, stop: int) -> str:
        """Characters of ``row`` for columns ``start`` to ``stop`` (exclusive)."""
        segments = self._segments[row]
        text = self._text[row]
        return "".join(
            text[col] or _GLYPHS[segments[col]] for col in range(start, stop)
        )


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Element:
    """One drawable item of a layer.

    Attributes:
        lanes: Qubit lanes the element occupies.  Nothing else is drawn on
            these lanes in the same layer.
        width: Number of columns the element needs.
        paint: Draws the element; called with the canvas and the element's
            leftmost column.
        clbits: Classical bits the element reads or writes.  They are not
            drawn, but an element that reads a bit stays to the right of the
            earlier ones that write it, and one that writes a bit to the
            right of every earlier one that uses it.
        writes: Whether the element writes ``clbits`` (a measurement) rather
            than reads them (a classically controlled gate).
        slim: Whether the element is bare symbols with a column of wire on
            each side.  Two layers of such elements stand closer together.
    """

    lanes: tuple[int, ...]
    width: int
    paint: Callable[[_Canvas, int], None]
    clbits: tuple[int, ...] = ()
    writes: bool = False
    slim: bool = False


# Each qubit lane is three rows high: the top edge of its cards, its wire and
# the bottom edge of its cards.
_LANE_HEIGHT = 3


def _top_row(qubit: int) -> int:
    return _LANE_HEIGHT * qubit


def _wire_row(qubit: int) -> int:
    return _LANE_HEIGHT * qubit + 1


def _bottom_row(qubit: int) -> int:
    return _LANE_HEIGHT * qubit + 2


def _is_wire_row(row: int) -> bool:
    return row % _LANE_HEIGHT == 1


def _span(lanes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Every lane from the lowest to the highest of ``lanes``."""
    return tuple(range(min(lanes), max(lanes) + 1))


def _label_row(qubits: tuple[int, ...]) -> int:
    """Row of a card's label: the middle of the card.

    When the middle is the wire of a qubit the card does not act on, the
    label moves up to the row under the wire above, where the label of a
    card over an even number of lanes is too.
    """
    row = (_top_row(min(qubits)) + _bottom_row(max(qubits))) // 2
    if _is_wire_row(row) and row not in {_wire_row(qubit) for qubit in qubits}:
        row -= 2
    return row


def _card_width(label: str, index_width: int = 0, caption: str = "") -> int:
    """Width of a card, outline included.

    The label has one blank on each side, plus the room the operand numbers
    take at the left and as much at the right, so that it stays centred.  A
    captioned card is at least as wide as ``╭─ caption ─╮``, and one column
    wider when its label could not be centred otherwise.
    """
    inner = len(label) + 2 * (index_width + 1)
    if caption:
        inner = max(inner, len(caption) + 4)
        inner += (inner - len(label)) % 2
    return inner + 2


def _paint_card(
    canvas: _Canvas,
    left: int,
    width: int,
    qubits: tuple[int, ...],
    label: str,
    *,
    numbered: bool = False,
    dashed: bool = False,
    caption: str = "",
) -> None:
    """Paint a rounded card over ``qubits`` with its label in the middle.

    Args:
        canvas: Canvas to paint on.
        left: Leftmost column of the card.
        width: Total width, outline included.
        qubits: Operands, in the order the gate takes them.  The card covers
            every lane from the first to the last of them; the wire of a
            lane in between that is not an operand runs through, dashed.
        label: Text shown in the card.
        numbered: Whether to print each operand's position where its wire
            enters the card, like the pin numbers of a chip.
        dashed: Whether to draw the outline dashed (noise channels).
        caption: Text set into the top edge, at its left (the condition of a
            classically controlled gate).
    """
    right = left + width - 1
    top = _top_row(min(qubits))
    bottom = _bottom_row(max(qubits))
    operand_rows = {_wire_row(qubit): position for position, qubit in enumerate(qubits)}

    if dashed:
        canvas.put(top, left, "╭" + "╌" * (width - 2) + "╮")
        canvas.put(bottom, left, "╰" + "╌" * (width - 2) + "╯")
    else:
        for col in range(left + 1, right):
            canvas.set(top, col, _WIRE)
            canvas.set(bottom, col, _WIRE)
        canvas.put(top, left, "╭")
        canvas.put(top, right, "╮")
        canvas.put(bottom, left, "╰")
        canvas.put(bottom, right, "╯")
    if caption:
        canvas.put(top, left + 2, f" {caption} ", weight=_SOLID)

    for row in range(top + 1, bottom):
        if row in operand_rows:
            canvas.put(row, left + 1, " " * (width - 2))
            canvas.set(row, left, _WIRE_ENTERS)
            canvas.set(row, right, _WIRE_LEAVES)
        elif _is_wire_row(row):
            canvas.put(row, left + 1, _PASS_THROUGH * (width - 2))
            canvas.set(row, left, _CROSSING)
            canvas.set(row, right, _CROSSING)
        else:
            canvas.put(row, left + 1, " " * (width - 2))
            if dashed:
                canvas.put(row, left, "╎", weight=_UPRIGHT)
                canvas.put(row, right, "╎", weight=_UPRIGHT)
            else:
                canvas.set(row, left, _VERTICAL)
                canvas.set(row, right, _VERTICAL)

    if numbered:
        index_width = len(str(len(qubits) - 1))
        for row, position in operand_rows.items():
            canvas.put(row, left + 1, str(position).rjust(index_width))
    canvas.put(_label_row(qubits), left + (width - len(label)) // 2, label)


def _card_element(
    qubits: tuple[int, ...],
    label: str,
    *,
    numbered: bool = False,
    dashed: bool = False,
    caption: str = "",
) -> _Element:
    """A labelled card over ``qubits`` (one card even when they are not adjacent)."""
    numbered = numbered and len(qubits) > 1
    index_width = len(str(len(qubits) - 1)) if numbered else 0
    width = _card_width(label, index_width, caption)

    def paint(canvas: _Canvas, left: int) -> None:
        _paint_card(
            canvas,
            left,
            width,
            qubits,
            label,
            numbered=numbered,
            dashed=dashed,
            caption=caption,
        )

    return _Element(lanes=_span(qubits), width=width, paint=paint)


def _flag(caption: str) -> str:
    """A caption on a pole: ``╭─ c0=1``, the pole standing on a symbol."""
    return f"╭─ {caption}"


def _symbol_element(marks: dict[int, str], *, caption: str = "") -> _Element:
    """Symbols on several qubits joined by one vertical line.

    A caption flies as a flag from the topmost symbol.
    """
    rows = {_wire_row(qubit): mark for qubit, mark in marks.items()}
    top, bottom = min(rows), max(rows)
    # one column of wire, the symbol, then one column or the flag
    width = 1 + len(_flag(caption)) if caption else _SYMBOL_WIDTH

    def paint(canvas: _Canvas, left: int) -> None:
        center = left + 1
        canvas.vline(center, top, bottom)
        for row, mark in rows.items():
            canvas.put(row, center, mark)
        if caption:
            canvas.put(top - 1, center, _flag(caption))

    return _Element(
        lanes=_span(list(marks)), width=width, paint=paint, slim=not caption
    )


def _controlled_card_element(
    controls: tuple[int, ...], target: int, label: str, *, caption: str = ""
) -> _Element:
    """Control dots joined to a labelled card on the target.

    A caption goes on whichever is topmost: into the top edge of the card,
    or as a flag on the first control.
    """
    card_on_top = target < min(controls)
    card_caption = caption if card_on_top else ""
    card_width = _card_width(label, caption=card_caption)
    center_offset = card_width // 2
    width = card_width
    if caption and not card_on_top:
        width = max(width, center_offset + len(_flag(caption)))

    def paint(canvas: _Canvas, left: int) -> None:
        center = left + center_offset
        _paint_card(canvas, left, card_width, (target,), label, caption=card_caption)
        for control in controls:
            if control < target:
                canvas.vline(center, _wire_row(control), _top_row(target))
            else:
                canvas.vline(center, _bottom_row(target), _wire_row(control))
            canvas.put(_wire_row(control), center, _CONTROL)
        if caption and not card_on_top:
            canvas.put(_top_row(min(controls)), center, _flag(caption))

    return _Element(lanes=_span([target, *controls]), width=width, paint=paint)


def _gate_element(gate: Gate, qubits: tuple[int, ...], caption: str = "") -> _Element:
    name = gate.name.lower()
    if len(qubits) == 1:
        return _card_element(qubits, _gate_label(gate), caption=caption)
    if name == "swap" and len(qubits) == 2:
        return _symbol_element(dict.fromkeys(qubits, _SWAP_MARK), caption=caption)
    if name in _ALL_CONTROL_GATES:
        return _symbol_element(dict.fromkeys(qubits, _CONTROL), caption=caption)
    if name in _XOR_TARGET_GATES:
        marks = dict.fromkeys(qubits[:-1], _CONTROL)
        marks[qubits[-1]] = _XOR_TARGET
        return _symbol_element(marks, caption=caption)
    if name in _CONTROLLED_TARGET_LABELS:
        return _controlled_card_element(
            qubits[:-1], qubits[-1], _controlled_target_label(gate), caption=caption
        )
    return _card_element(
        qubits,
        _gate_label(gate),
        numbered=name not in _SYMMETRIC_GATES,
        caption=caption,
    )


def _condition_text(clbits: tuple[int, ...], value: int) -> str:
    """``c0=1`` for one bit, ``c0=1 c1=0`` for several."""
    return " ".join(
        f"c{clbit}={(value >> position) & 1}" for position, clbit in enumerate(clbits)
    )


def _barrier_element(qubits: tuple[int, ...]) -> _Element:
    """A dashed vertical line from the first to the last qubit of the barrier.

    Like a bar line it has its column to itself.  The wire of a qubit in
    between that the barrier does not cover passes in front of it.
    """
    covered = set(qubits)
    lanes = _span(list(covered))
    rows = [
        row
        for row in range(_top_row(lanes[0]), _bottom_row(lanes[-1]) + 1)
        if not _is_wire_row(row) or row // _LANE_HEIGHT in covered
    ]

    def paint(canvas: _Canvas, left: int) -> None:
        for row in rows:
            # no row is kept just because a barrier passes
            canvas.put(row, left, _BARRIER_MARK, weight=_BLANK)

    return _Element(lanes=lanes, width=1, paint=paint)


def _elements(instruction: CircuitInstruction, num_qubits: int) -> list[_Element]:
    if instruction.is_gate and instruction.gate is not None:
        return [_gate_element(instruction.gate, instruction.qubits)]
    if instruction.is_measure:
        elements = []
        for qubit, clbit in zip(instruction.qubits, instruction.clbits, strict=True):
            card = _card_element((qubit,), f"M{_WRITES}c{clbit}")
            elements.append(
                _Element(
                    card.lanes, card.width, card.paint, clbits=(clbit,), writes=True
                )
            )
        return elements
    if instruction.is_barrier:
        return [_barrier_element(instruction.qubits or tuple(range(num_qubits)))]
    if instruction.is_channel and instruction.channel is not None:
        label = _channel_label(instruction.channel)
        return [_card_element(instruction.qubits, label, dashed=True)]
    if instruction.is_conditional and instruction.conditional_gate is not None:
        assert instruction.condition is not None
        gate = _gate_element(
            instruction.conditional_gate,
            instruction.qubits,
            caption=_condition_text(instruction.clbits, instruction.condition),
        )
        return [_Element(gate.lanes, gate.width, gate.paint, instruction.clbits)]
    return []


# ---------------------------------------------------------------------------
# Layout and rendering
# ---------------------------------------------------------------------------


class _Placeable(Protocol):
    """What the packing needs to know about a drawable item."""

    @property
    def lanes(self) -> tuple[int, ...]: ...

    @property
    def clbits(self) -> tuple[int, ...]: ...

    @property
    def writes(self) -> bool: ...


_Item = TypeVar("_Item", bound=_Placeable)


def _pack_layers(
    groups: list[list[_Item]], num_qubits: int, num_clbits: int
) -> list[list[_Item]]:
    """Put each element in the earliest layer where its lanes are free.

    Elements keep their relative order on every qubit they share, and a
    classical bit is never read left of where it was written before, so the
    drawing reads as the same circuit.  The elements of one
    group (the cards of one measurement instruction) go into one layer, the
    earliest that is free for all of them, so that they line up.

    The SVG drawer packs its shapes with this function too, so both drawings
    of a circuit have the same columns.

    Args:
        groups: The elements of each instruction, in circuit order.
        num_qubits: Number of qubit lanes.
        num_clbits: Number of classical bits.

    Returns:
        The elements of each layer, from left to right.
    """
    layers: list[list[_Item]] = []
    lane_layer = [-1] * num_qubits
    written = [-1] * num_clbits  # last layer that writes each classical bit
    read = [-1] * num_clbits  # last layer that reads it

    def earliest(element: _Item) -> int:
        layer = max(lane_layer[lane] for lane in element.lanes)
        for clbit in element.clbits:
            layer = max(layer, written[clbit])
            if element.writes:
                layer = max(layer, read[clbit])
        return layer + 1

    for group in groups:
        together = max((earliest(element) for element in group), default=0)
        for element in group:
            # later than the others only when the group uses a lane twice
            layer = max(together, earliest(element))
            while layer >= len(layers):
                layers.append([])
            layers[layer].append(element)
            for lane in element.lanes:
                lane_layer[lane] = layer
            for clbit in element.clbits:
                if element.writes:
                    written[clbit] = layer
                else:
                    read[clbit] = max(read[clbit], layer)
    return layers


def _layer_starts(widths: list[int], slim: list[bool]) -> list[int]:
    """First column of every layer.

    ``_GAP`` columns of bare wire come before the first layer and between
    two layers, except between two layers of bare symbols: these bring a
    column of wire on each side, so one more column is enough.
    """
    starts: list[int] = []
    cursor = _GAP
    for index, width in enumerate(widths):
        starts.append(cursor)
        cursor += width + _GAP
        if slim[index] and index + 1 < len(widths) and slim[index + 1]:
            cursor -= 1
    return starts


def _run_width(starts: list[int], widths: list[int], first: int, stop: int) -> int:
    """Columns taken by the layers ``first`` to ``stop - 1`` drawn as one part."""
    return starts[stop - 1] + widths[stop - 1] - starts[first] + 2 * _GAP


def _chunk_layers(
    starts: list[int], widths: list[int], available: int
) -> list[tuple[int, int]]:
    """Split the layers into runs ``(first, last + 1)`` that fit ``available``.

    Each run is filled as far as it goes.  A layer wider than ``available``
    gets a run of its own.
    """
    chunks: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(widths)):
        if _run_width(starts, widths, start, index + 1) > available:
            chunks.append((start, index))
            start = index
    chunks.append((start, len(widths)))
    return chunks


def _balanced_chunks(
    starts: list[int], widths: list[int], available: int
) -> list[tuple[int, int]]:
    """Split the layers into as few runs as fit, of about the same width.

    Filling each run to the brim leaves a short last one (often a single
    column under a full-width part).  Instead, with ``n`` the smallest number
    of runs that fit, the cut after run ``i`` is put where the layers so far
    come closest to ``i / n`` of the whole drawing.
    """
    greedy = _chunk_layers(starts, widths, available)
    count = len(greedy)
    if count == 1:
        return greedy
    # where a cut before each layer falls, the end of the drawing included
    cuts = [start - starts[0] for start in starts]
    cuts.append(cuts[-1] + widths[-1] + _GAP)
    chunks: list[tuple[int, int]] = []
    first = 0
    for number in range(1, count):
        target = cuts[-1] * number / count
        stop = first + 1
        # Take one more layer while that brings the cut closer to the target,
        # the part still fits and enough layers remain for the other parts.
        while (
            stop < len(widths) - (count - number)
            and abs(cuts[stop + 1] - target) <= abs(cuts[stop] - target)
            and _run_width(starts, widths, first, stop + 1) <= available
        ):
            stop += 1
        chunks.append((first, stop))
        first = stop
    chunks.append((first, len(widths)))
    fits = all(
        stop - first == 1 or _run_width(starts, widths, first, stop) <= available
        for first, stop in chunks
    )
    return chunks if fits else greedy


def _rows_to_print(canvas: _Canvas, start: int, stop: int) -> set[int]:
    """Rows between the wires that a part of the drawing needs.

    Between two wires lie the bottom-edge row of the upper lane and the
    top-edge row of the lower lane.  A row is printed when it shows an edge
    or a label.  Of two rows that only show vertical lines passing through
    one is enough, and next to a row that is printed anyway such a row is
    not needed at all: the lines stay connected.  Empty rows are left out.
    """
    weights = [canvas.row_weight(row, start, stop) for row in range(canvas.num_rows)]
    rows = {row for row, weight in enumerate(weights) if weight == _SOLID}
    for upper in range(_LANE_HEIGHT - 1, canvas.num_rows - 1, _LANE_HEIGHT):
        lower = upper + 1  # top-edge row of the next lane
        if weights[upper] == _UPRIGHT and weights[lower] != _SOLID:
            rows.add(upper)
    return rows


def draw_text_circuit(
    circuit: QuantumCircuit,
    *,
    width: int | None = None,
    fold: bool = True,
) -> str:
    """Render a quantum circuit as a multi-line Unicode diagram.

    Gates on disjoint qubits share a column.  A circuit wider than the
    display is cut between columns into parts of about the same width, drawn
    one below the other, each with its own qubit labels; a wire that goes on
    in the next part fades out (``┄``) and fades in again there.

    Args:
        circuit: The circuit to render.
        width: Number of columns available.  ``None`` uses the width of the
            terminal (see :func:`terminal_width`).
        fold: Whether to continue a wide circuit on further lines.  With
            ``False`` every qubit is a single line, however long.

    Returns:
        A terminal-friendly multi-line string.
    """
    if len(circuit) == 0:
        return "(empty circuit)"

    num_qubits = max(circuit.num_qubits, 1)
    groups = [
        _elements(instruction, num_qubits) for instruction in circuit.instructions
    ]
    layers = _pack_layers(groups, num_qubits, circuit.num_clbits)
    widths = [max(element.width for element in layer) for layer in layers]
    slim = [all(element.slim for element in layer) for layer in layers]
    starts = _layer_starts(widths, slim)
    num_cols = starts[-1] + widths[-1] + _GAP if widths else 2 * _GAP

    canvas = _Canvas(_LANE_HEIGHT * num_qubits, num_cols)
    for qubit in range(num_qubits):
        canvas.wire(_wire_row(qubit))
    for layer, start, layer_width in zip(layers, starts, widths, strict=True):
        for element in layer:
            element.paint(canvas, start + (layer_width - element.width) // 2)

    label_width = len(f"q{num_qubits - 1}")
    blank_prefix = " " * (label_width + 1)

    if fold and len(widths) > 1:
        display = terminal_width() if width is None else width
        # one column each for the marks at both ends of a wire
        available = display - len(blank_prefix) - 2
        chunks = _balanced_chunks(starts, widths, available)
    else:
        chunks = [(0, len(widths))]

    parts: list[str] = []
    for number, (first, stop) in enumerate(chunks):
        start_col = starts[first] - _GAP if widths else 0
        stop_col = starts[stop - 1] + widths[stop - 1] + _GAP if widths else num_cols
        lead = _FADE if number > 0 else ""
        tail = _FADE if number < len(chunks) - 1 else ""
        printed = _rows_to_print(canvas, start_col, stop_col)
        lines: list[str] = []
        for row in range(canvas.num_rows):
            body = canvas.render_row(row, start_col, stop_col)
            if _is_wire_row(row):
                label = f"q{row // _LANE_HEIGHT}".rjust(label_width)
                lines.append(f"{label} {lead}{body}{tail}")
            elif row in printed:
                lines.append((blank_prefix + " " * len(lead) + body).rstrip())
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Display width
# ---------------------------------------------------------------------------


def _in_jupyter() -> bool:
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        shell = ipython.get_ipython()
    except Exception:
        return False
    return type(shell).__name__ == "ZMQInteractiveShell"


def terminal_width(default: int = _DEFAULT_WIDTH) -> int:
    """Number of columns of the display the drawing will be printed on.

    Follows the rules of the ``rich`` library:

    * the ``COLUMNS`` environment variable wins when it holds a positive
      integer;
    * in a Jupyter kernel the width is ``JUPYTER_COLUMNS``, or 115;
    * otherwise the first of standard input, standard output and standard
      error that is attached to a terminal gives its width (on Windows only
      the two output streams are asked);
    * a dumb terminal (``TERM=dumb`` / ``TERM=unknown``), or no terminal at
      all (output piped to a file, a test run), gives ``default``.

    Args:
        default: Width used when no terminal can be asked.

    Returns:
        The width in columns.
    """
    columns = os.environ.get("COLUMNS", "")
    if columns.isdigit() and int(columns) > 0:
        return int(columns)

    if _in_jupyter():
        jupyter_columns = os.environ.get("JUPYTER_COLUMNS", "")
        if jupyter_columns.isdigit() and int(jupyter_columns) > 0:
            return int(jupyter_columns)
        return _JUPYTER_DEFAULT_WIDTH

    if os.environ.get("TERM", "").lower() in ("dumb", "unknown"):
        return default

    descriptors = (1, 2) if sys.platform == "win32" else (0, 1, 2)
    for descriptor in descriptors:
        try:
            size = os.get_terminal_size(descriptor)
        except (AttributeError, ValueError, OSError):
            continue
        if size.columns > 0:
            return size.columns
    return default


# ---------------------------------------------------------------------------
# Labels (shared with the SVG drawer)
# ---------------------------------------------------------------------------


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


_PI_DENOMINATORS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64)
_PI_MAX_MULTIPLE = 64


def _format_number(value: float) -> str:
    """A number as a multiple of π when it is one, else with 4 digits.

    ``0.7853981633974483`` is shown as ``π/4`` and ``-4.71238898038469`` as
    ``-3π/2``; ``0.3`` stays ``0.3``.
    """
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    for denominator in _PI_DENOMINATORS:
        scaled = value / math.pi * denominator
        numerator = round(scaled)
        if (
            numerator != 0
            and abs(numerator) <= _PI_MAX_MULTIPLE * denominator
            and math.isclose(scaled, numerator, rel_tol=1e-9, abs_tol=1e-12)
        ):
            sign = "-" if numerator < 0 else ""
            multiple = "" if abs(numerator) == 1 else str(abs(numerator))
            fraction = "" if denominator == 1 else f"/{denominator}"
            return f"{sign}{multiple}π{fraction}"
    return f"{value:.4g}"


def _format_param(param: Any) -> str:
    if isinstance(param, bool):
        return str(param)
    if isinstance(param, float):
        return _format_number(param)
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
