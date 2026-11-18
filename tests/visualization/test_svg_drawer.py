"""Tests for SVG circuit rendering."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import pytest

from quspect import QuantumCircuit
from quspect.gates.gate import Gate
from quspect.noise.channel import DepolarizingChannel
from quspect.visualization import (
    SvgStyle,
    draw_svg_circuit,
    draw_text_circuit,
    svg_drawer,
)

NS = "{http://www.w3.org/2000/svg}"

# One colour per role, so that every element can be told apart by its colour.
STYLE = {
    "background": "#000001",
    "wire": "#000002",
    "classical_wire": "#000003",
    "gate_fill": "#000004",
    "gate_stroke": "#000005",
    "text": "#000006",
    "control": "#000007",
    "barrier": "#000008",
    "measure_fill": "#000009",
    "channel_fill": "#00000a",
    "channel_stroke": "#00000b",
}


@dataclass
class Drawing:
    """What a rendered circuit shows, read back from its SVG."""

    root: ET.Element
    gate_cards: list[ET.Element] = field(default_factory=list)
    measure_cards: list[ET.Element] = field(default_factory=list)
    channel_cards: list[ET.Element] = field(default_factory=list)
    targets: list[tuple[float, float]] = field(default_factory=list)
    dots: list[tuple[float, float]] = field(default_factory=list)
    connectors: list[tuple[float, float, float]] = field(default_factory=list)
    swap_marks: int = 0
    wires: list[float] = field(default_factory=list)
    through: list[float] = field(default_factory=list)
    barrier_pieces: list[tuple[float, float, float]] = field(default_factory=list)
    flags: int = 0
    qubit_labels: list[str] = field(default_factory=list)
    labels: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    measure_labels: dict[str, tuple[float, float]] = field(default_factory=dict)
    numbers: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]


def draw(circuit: QuantumCircuit, **options: object) -> Drawing:
    """Render ``circuit`` with :data:`STYLE` and read the drawing back."""
    options.setdefault("style", STYLE)
    root = ET.fromstring(str(draw_svg_circuit(circuit, **options)))  # type: ignore[arg-type]
    drawing = Drawing(root)
    (body,) = root.findall(f"{NS}g")
    for element in body:
        fill, stroke = element.get("fill"), element.get("stroke")
        dash = element.get("stroke-dasharray")
        if element.tag == f"{NS}path":
            data = element.get("d", "")
            pieces = [piece for piece in data.split("M") if piece]
            if stroke == STYLE["barrier"]:
                for piece in pieces:
                    x, top, bottom = _numbers(piece)
                    drawing.barrier_pieces.append((x, top, bottom))
            elif stroke == STYLE["wire"] and dash:
                drawing.through = [_numbers(piece)[1] for piece in pieces]
            elif stroke == STYLE["wire"]:
                drawing.wires = [_numbers(piece)[1] for piece in pieces]
            elif stroke == STYLE["classical_wire"]:
                drawing.flags = len(pieces)
            elif data.endswith("h0"):
                drawing.dots = [tuple(_numbers(piece)[:2]) for piece in pieces]  # type: ignore[misc]
            elif "l" in data or "h" in data:
                drawing.swap_marks += data.count("l") // 2
            else:
                drawing.connectors = [tuple(_numbers(piece)) for piece in pieces]  # type: ignore[misc]
            continue
        rects = element.findall(f"{NS}rect")
        circles = element.findall(f"{NS}circle")
        texts = element.findall(f"{NS}text")
        if rects and fill == STYLE["gate_fill"]:
            drawing.gate_cards = rects
        elif rects and fill == STYLE["measure_fill"]:
            assert stroke == STYLE["classical_wire"]
            drawing.measure_cards = rects
        elif rects and fill == STYLE["channel_fill"]:
            assert stroke == STYLE["channel_stroke"] and dash
            drawing.channel_cards = rects
        elif circles:
            drawing.targets = [
                (float(c.get("cx", 0)), float(c.get("cy", 0))) for c in circles
            ]
        for text in texts:
            content = text.text or ""
            where = (float(text.get("x", 0)), float(text.get("y", 0)))
            small = element.get("font-size") is not None
            held_back = element.get("fill-opacity") is not None
            if held_back and small:
                drawing.numbers.append(content)
            elif held_back:
                drawing.qubit_labels.append(content)
            elif fill == STYLE["text"]:
                drawing.labels.setdefault(content, []).append(where)
            elif small:
                drawing.captions.append(content)
                drawing.titles.extend(t.text or "" for t in text.findall(f"{NS}title"))
            else:
                drawing.measure_labels[content] = where
    return drawing


def _sample_circuit() -> QuantumCircuit:
    circ = QuantumCircuit()
    circ.h(0).cx(0, 1).swap(1, 2).barrier(0, 1, 2).measure([0, 1], [0, 1])
    return circ


# ------------------------------------------------------------ gate families
FAMILY_STYLE = {
    **STYLE,
    "pauli_gate": "#100000",
    "hadamard_gate": "#200000",
    "phase_gate": "#300000",
    "rotation_gate": "#400000",
    "interaction_gate": "#500000",
}


def _card_strokes(circuit: QuantumCircuit, style: object) -> dict[str, int]:
    """Number of cards per outline colour."""
    root = ET.fromstring(str(draw_svg_circuit(circuit, style=style)))  # type: ignore[arg-type]
    (body,) = root.findall(f"{NS}g")
    return {
        group.get("stroke", ""): len(group.findall(f"{NS}rect"))
        for group in body.findall(f"{NS}g")
        if group.get("stroke") is not None and group.findall(f"{NS}rect")
    }


def test_each_family_of_gates_has_its_colour() -> None:
    circuit = QuantumCircuit().x(0).y(0).h(0).sx(0).s(0).t(0).rz(0.1, 0).rx(0.1, 0)
    circuit.u3(0.1, 0.2, 0.3, 0).rzz(0.1, 0, 1).crz(0.1, 0, 1).cry(0.1, 0, 1)
    circuit.append(Gate(name="custom", num_qubits=1), (0,))
    assert _card_strokes(circuit, FAMILY_STYLE) == {
        "#100000": 2,  # X, Y
        "#200000": 2,  # H, √X
        "#300000": 4,  # S, T, RZ and the target of CRZ
        "#400000": 3,  # RX, U3 and the target of CRY
        "#500000": 1,  # RZZ
        STYLE["gate_stroke"]: 1,  # a gate of no family
    }


def test_family_colour_tints_the_card() -> None:
    style = {**STYLE, "gate_fill": "#ffffff", "pauli_gate": "#ff0000"}
    svg = str(draw_svg_circuit(QuantumCircuit().x(0), style=style))
    (group,) = [
        element
        for element in ET.fromstring(svg).iter(f"{NS}g")
        if element.get("stroke") == "#ff0000"
    ]
    fill = group.get("fill", "")
    red, green, blue = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    assert red == 255 and 200 < green == blue < 250  # white with a little red


def test_style_without_family_colours_is_monochrome() -> None:
    circuit = QuantumCircuit().x(0).h(0).rz(0.1, 0).rzz(0.1, 0, 1)
    assert _card_strokes(circuit, STYLE) == {STYLE["gate_stroke"]: 4}
    assert _card_strokes(circuit, "mono") == {
        svg_drawer._SVG_THEME_DATA["mono"]["gate_stroke"]: 4
    }
    assert len(_card_strokes(circuit, "official")) == 4


def test_style_round_trips_with_and_without_family_colours() -> None:
    assert SvgStyle.from_mapping(STYLE).as_dict() == STYLE
    assert SvgStyle.from_mapping(FAMILY_STYLE).as_dict() == FAMILY_STYLE
    with pytest.raises(ValueError, match=r"Unknown SVG style keys: \['pauli'\]"):
        SvgStyle.from_mapping({**STYLE, "pauli": "#ff0000"})


def test_family_colour_that_cannot_be_mixed_still_outlines() -> None:
    style = {**STYLE, "pauli_gate": "crimson"}
    svg = str(draw_svg_circuit(QuantumCircuit().x(0), style=style))
    assert f'fill="{STYLE["gate_fill"]}" stroke="crimson"' in svg


# ---------------------------------------------------------------- document
@pytest.mark.parametrize("style", ["official", "mono", "light", "dark"])
def test_svg_circuit_is_valid_xml(style: str) -> None:
    ET.fromstring(str(_sample_circuit().svg(style=style)))


def test_document_is_lean() -> None:
    """No filters, and one element per kind of line however many gates."""
    circ = QuantumCircuit()
    for _ in range(50):
        circ.h(0).cx(0, 1).cz(1, 2).swap(0, 2)
    svg = str(circ.svg())
    root = ET.fromstring(svg)
    assert "filter" not in svg and "<defs" not in svg
    # wires, connectors, dots, target crosses, swap crosses
    assert len(root.findall(f".//{NS}path")) == 5
    assert len(root.findall(f".//{NS}line")) == 0
    # shared attributes are written once, on a group
    assert svg.count("font-family") == 1
    assert svg.count("stroke-width") < 10


def test_svg_scale_changes_outer_size_without_internal_transform() -> None:
    svg = str(_sample_circuit().svg(width=320, scale=2))
    root = ET.fromstring(svg)
    view_box_width = float(root.attrib["viewBox"].split()[2])
    outer_width = float(root.attrib["width"])

    assert math.isclose(outer_width, view_box_width * 2)
    assert 'transform="scale' not in svg


def test_svg_accepts_custom_style_object() -> None:
    svg = draw_svg_circuit(_sample_circuit(), style=SvgStyle.from_mapping(STYLE))
    assert str(svg) == str(draw_svg_circuit(_sample_circuit(), style=STYLE))
    for colour in STYLE.values():
        if colour not in (STYLE["channel_fill"], STYLE["channel_stroke"]):
            assert colour in str(svg)


# ------------------------------------------------------------------- gates
def test_single_qubit_gates_are_cards_on_their_wires() -> None:
    drawing = draw(QuantumCircuit().h(0).rx(math.pi / 2, 1))
    assert drawing.qubit_labels == ["q0", "q1"]
    assert len(drawing.gate_cards) == 2
    (h,) = drawing.labels["H"]
    (rx,) = drawing.labels["RX(π/2)"]
    assert h[0] == rx[0]  # one column, both centred in it
    assert drawing.wires[1] - drawing.wires[0] == svg_drawer._PITCH
    assert rx[1] - h[1] == pytest.approx(svg_drawer._PITCH)


def test_cards_are_rounded() -> None:
    drawing = draw(QuantumCircuit().h(0))
    assert float(drawing.gate_cards[0].get("rx", 0)) == svg_drawer._CARD_RADIUS


def test_cx_and_toffoli_use_dots_and_a_circled_plus() -> None:
    drawing = draw(QuantumCircuit().cx(0, 1).ccx(0, 2, 1))
    assert len(drawing.dots) == 3 and len(drawing.targets) == 2
    assert not drawing.gate_cards
    wires = drawing.wires
    assert {y for _, y in drawing.targets} == {wires[1]}
    first, second = drawing.connectors
    assert (first[1], first[2]) == (wires[0], wires[1])
    assert (second[1], second[2]) == (wires[0], wires[2])


def test_cz_is_two_dots_and_swap_two_crosses() -> None:
    drawing = draw(QuantumCircuit().cz(0, 1).swap(0, 2))
    assert len(drawing.dots) == 2 and drawing.swap_marks == 2
    assert not drawing.targets and not drawing.gate_cards


def test_controlled_rotation_targets_show_angles() -> None:
    """Controlled-rotation target cards include their parameters (B82)."""
    circuit = QuantumCircuit()
    circuit.crz(0.3, 0, 1).crx(1.2, 0, 1).cp(0.5, 1, 0).cu3(0.1, 0.2, 0.3, 0, 1)
    drawing = draw(circuit)
    for label in ("RZ(0.3)", "RX(1.2)", "P(0.5)", "U3(0.1, 0.2, 0.3)"):
        assert label in drawing.labels
    assert len(drawing.dots) == 4 and len(drawing.gate_cards) == 4


def test_multi_qubit_gate_is_one_card() -> None:
    """One card and one label; the wire it skips runs through, dashed."""
    drawing = draw(QuantumCircuit().rzz(math.pi / 4, 0, 2))
    (card,) = drawing.gate_cards
    assert list(drawing.labels) == ["RZZ(π/4)"]
    wires = drawing.wires
    assert float(card.get("height", 0)) == wires[2] - wires[0] + svg_drawer._CARD_HEIGHT
    assert drawing.through == [wires[1]]
    assert not drawing.numbers  # symmetric: operands are not numbered
    # the label keeps off the wire that runs through
    assert drawing.labels["RZZ(π/4)"][0][1] < wires[1]


def test_generic_gate_numbers_its_operands() -> None:
    circuit = QuantumCircuit()
    circuit.append(Gate(name="foo", num_qubits=3), (2, 0, 1))
    drawing = draw(circuit)
    assert len(drawing.gate_cards) == 1 and sorted(drawing.numbers) == ["0", "1", "2"]


def test_gate_labels_are_escaped() -> None:
    circuit = QuantumCircuit()
    circuit.append(Gate(name="weird", num_qubits=1, label="<a&b>"), (0,))
    assert "&lt;a&amp;b&gt;" in str(draw_svg_circuit(circuit))
    assert "<a&b>" in draw(circuit).labels


# ----------------------------------------------- measurement and conditions
def test_measurement_names_its_classical_bit() -> None:
    """No classical wires: a tinted card with the bit written."""
    drawing = draw(QuantumCircuit().h(0).measure([0, 1], [1, 0]))
    assert drawing.qubit_labels == ["q0", "q1"] and len(drawing.wires) == 2
    assert len(drawing.measure_cards) == 2
    wires = drawing.wires
    assert drawing.measure_labels["M→c1"][1] == pytest.approx(drawing.labels["H"][0][1])
    assert drawing.measure_labels["M→c0"][1] > wires[1]
    # one instruction: its cards line up
    assert drawing.measure_labels["M→c1"][0] == drawing.measure_labels["M→c0"][0]


def test_condition_is_set_into_the_card() -> None:
    circuit = QuantumCircuit().measure([0, 1], [0, 1])
    with circuit.if_test([0, 1], 1):
        circuit.rz(0.5, 0)
    drawing = draw(circuit)
    assert drawing.captions == ["c0=1 c1=0"]
    assert drawing.titles == ["rz if (c0, c1) == (1, 0)"]
    assert drawing.flags == 0
    # the conditional gate is right of the measurements it reads
    assert drawing.labels["RZ(0.5)"][0][0] > drawing.measure_labels["M→c0"][0]


def test_condition_of_a_gate_without_a_card_is_a_flag() -> None:
    circuit = QuantumCircuit().measure([0], [0])
    with circuit.if_test(0, 0):
        circuit.cx(1, 2)
    with circuit.if_test(0):
        circuit.crz(0.5, 1, 2)  # control on top: the flag stands on it
    drawing = draw(circuit)
    assert drawing.captions == ["c0=0", "c0=1"]
    assert drawing.flags == 2


# ---------------------------------------------------- barriers and channels
def test_barrier_spans_its_qubits_and_skips_the_others() -> None:
    drawing = draw(QuantumCircuit().h(0).h(1).h(2).barrier(0, 2).barrier(1))
    wires = drawing.wires
    first_top, first_bottom, lone = drawing.barrier_pieces
    assert first_top[1] < wires[0] < first_top[2] < wires[1]
    assert wires[1] < first_bottom[1] < wires[2] < first_bottom[2]
    assert first_top[0] == first_bottom[0] < lone[0]  # its own column, then the next
    assert lone[1] < wires[1] < lone[2]


def test_noise_channel_is_a_dashed_card() -> None:
    circuit = QuantumCircuit().depolarize(0.1, 0)
    circuit.append_channel(
        DepolarizingChannel(name="depolarizing", num_qubits=2, p=0.2), [0, 2]
    )
    drawing = draw(circuit)
    assert len(drawing.channel_cards) == 2 and not drawing.gate_cards
    assert set(drawing.labels) == {"DP(0.1)", "DP(0.2)"}
    assert drawing.through == [drawing.wires[1]]


# ------------------------------------------------------------------ layout
def test_svg_packs_independent_gates_into_one_column() -> None:
    drawing = draw(QuantumCircuit().h(0).h(1).h(2))
    assert len({x for x, _ in drawing.labels["H"]}) == 1
    assert len({y for _, y in drawing.labels["H"]}) == 3


def test_svg_keeps_shared_qubit_dependencies_in_later_columns() -> None:
    drawing = draw(QuantumCircuit().h(0).x(1).z(0))
    assert drawing.labels["H"][0][0] == drawing.labels["X"][0][0]
    assert drawing.labels["Z"][0][0] > drawing.labels["H"][0][0]


def test_svg_reserves_intermediate_lanes_for_multiqubit_connectors() -> None:
    drawing = draw(QuantumCircuit().cx(0, 2).h(1))
    assert drawing.labels["H"][0][0] > drawing.dots[0][0]


def test_svg_packs_disjoint_multiqubit_and_single_qubit_gates() -> None:
    drawing = draw(QuantumCircuit().cx(0, 1).h(2))
    assert drawing.labels["H"][0][0] == drawing.dots[0][0]


def test_measurement_does_not_block_other_qubits() -> None:
    """Without classical wires a measurement only occupies its own qubit."""
    drawing = draw(QuantumCircuit().measure([0], [0]).h(1))
    assert drawing.labels["H"][0][0] < drawing.measure_labels["M→c0"][0] + 20


def test_columns_are_those_of_the_text_drawing() -> None:
    """Both drawers pack with the same function: same left-to-right order."""
    circuit = QuantumCircuit().h(0).cx(0, 2).x(1).rzz(0.5, 1, 2).measure_all()
    with circuit.if_test(0):
        circuit.t(1)
    drawing = draw(circuit)
    marks = [
        ("H", drawing.labels["H"][0][0]),
        ("●", drawing.dots[0][0]),
        ("X", drawing.labels["X"][0][0]),
        ("RZZ(0.5)", drawing.labels["RZZ(0.5)"][0][0]),
        ("M→c0", drawing.measure_labels["M→c0"][0]),
        ("T", drawing.labels["T"][0][0]),
    ]
    text = draw_text_circuit(circuit, fold=False)
    columns = {
        mark: min(line.index(mark) for line in text.splitlines() if mark in line)
        for mark, _ in marks
    }
    by_svg = [mark for mark, _ in sorted(marks, key=lambda item: item[1])]
    by_text = sorted(columns, key=lambda mark: columns[mark])
    assert by_svg == by_text


def test_two_columns_of_bare_symbols_stand_closer() -> None:
    cards = draw(QuantumCircuit().h(0).h(0).h(0))
    xs = sorted(x for x, _ in cards.labels["H"])
    symbols = draw(QuantumCircuit().cz(0, 1).cz(0, 1).cz(0, 1))
    dot_xs = sorted({x for x, _ in symbols.dots})
    card_gap = xs[1] - xs[0] - svg_drawer._CARD_HEIGHT
    symbol_gap = dot_xs[1] - dot_xs[0] - svg_drawer._SYMBOL_WIDTH
    assert card_gap == pytest.approx(svg_drawer._GAP)
    assert symbol_gap == pytest.approx(svg_drawer._SLIM_GAP)
