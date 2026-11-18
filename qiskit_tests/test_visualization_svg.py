"""Tests for the SVG circuit drawer (:mod:`quspect.visualization.svg_drawer`).

qiskit has no comparable SVG output, so the SVG is parsed with
``xml.etree`` and checked structurally: well-formed XML, element counts per
instruction kind, style colours and size options.  What each kind of
instruction looks like is pinned in ``tests/visualization/test_svg_drawer.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import quspect
from qiskit_tests._ir_helpers import all_channel_examples
from qiskit_tests.helpers import random_circuit_pair
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.gates.gate import Gate
from quspect.visualization import CircuitSVG, SvgStyle, draw_svg_circuit
from quspect.visualization.svg_drawer import (
    _FAMILY_STYLE_KEYS,
    _FAMILY_TINT,
    _SVG_THEME_DATA,
    _mix,
)

NS = "{http://www.w3.org/2000/svg}"
XOR_TARGET = {"cx", "ccx"}
CONTROLLED_CARD = {"crx", "cry", "crz", "cp", "cu3"}

# One colour per role, so that every element can be told apart by its colour.
CUSTOM_STYLE = {
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


def _parse(svg: CircuitSVG | str) -> ET.Element:
    """Parse SVG text into an ElementTree root (asserts well-formedness)."""
    root = ET.fromstring(str(svg))
    assert root.tag == f"{NS}svg"
    return root


def _all(root: ET.Element, tag: str) -> list[ET.Element]:
    """All descendants with the given SVG tag."""
    return list(root.iter(f"{NS}{tag}"))


def _rects_filled(root: ET.Element, colour: str) -> list[ET.Element]:
    """Cards of the group filled with ``colour``."""
    return [
        rect
        for group in _all(root, "g")
        if group.get("fill") == colour and group.get("stroke") is not None
        for rect in group.findall(f"{NS}rect")
    ]


def _path_data(root: ET.Element, **attributes: str | None) -> str:
    """``d`` of the path with the given attributes ("" when there is none)."""
    for path in _all(root, "path"):
        if all(
            path.get(key.replace("_", "-")) == value
            for key, value in attributes.items()
        ):
            return path.get("d", "")
    return ""


def _expected_counts(circ: quspect.QuantumCircuit) -> dict[str, int]:
    """Expected number of gate cards, dots, targets, swap marks, measures, barriers."""
    counts = dict.fromkeys(("card", "dot", "target", "swap", "measure", "barrier"), 0)
    for instr in circ.instructions:
        if instr.is_gate and instr.gate is not None:
            name = instr.gate.name
            if name == "swap":
                counts["swap"] += 2
            elif name == "cz":
                counts["dot"] += 2
            elif name in XOR_TARGET:
                counts["dot"] += len(instr.qubits) - 1
                counts["target"] += 1
            elif name in CONTROLLED_CARD:
                counts["dot"] += 1
                counts["card"] += 1
            else:
                counts["card"] += 1  # one card, however many qubits
        elif instr.is_measure:
            counts["measure"] += len(instr.qubits)
        elif instr.is_barrier:
            counts["barrier"] += 1
    return counts


def _observed_counts(root: ET.Element) -> dict[str, int]:
    """Count the same features in a parsed SVG drawn with ``CUSTOM_STYLE``."""
    control = CUSTOM_STYLE["control"]
    marks = [
        path.get("d", "")
        for path in _all(root, "path")
        if path.get("stroke") == control and "l" in path.get("d", "")
    ]
    dots = [
        path.get("d", "")
        for path in _all(root, "path")
        if path.get("stroke") == control and path.get("d", "").endswith("h0")
    ]
    return {
        "card": len(_rects_filled(root, CUSTOM_STYLE["gate_fill"])),
        "dot": sum(d.count("h0") for d in dots),
        "target": len(_all(root, "circle")),
        "swap": sum(d.count("l") for d in marks) // 2,
        "measure": len(_rects_filled(root, CUSTOM_STYLE["measure_fill"])),
        "barrier": _path_data(root, stroke=CUSTOM_STYLE["barrier"]).count("M"),
    }


# =============================================================================
# Structure
# =============================================================================


@pytest.mark.parametrize("seed", range(10))
def test_random_circuit_element_counts(seed: int) -> None:
    """Cards, dots, targets, swaps, measures and barriers match the circuit."""
    circ, _ = random_circuit_pair(4, 25, seed)
    circ.barrier(0, 1)
    circ.measure([0, 2], [1, 0])
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    assert _observed_counts(root) == _expected_counts(circ)
    # One label and one wire per qubit; classical bits have neither.
    labels = [
        text.text
        for group in _all(root, "g")
        if group.get("fill-opacity") is not None and group.get("font-size") is None
        for text in group.findall(f"{NS}text")
    ]
    assert labels == ["q0", "q1", "q2", "q3"]
    wires = _path_data(root, stroke=CUSTOM_STYLE["wire"], stroke_dasharray=None)
    assert wires.count("M") == 4
    texts = [text.text for text in _all(root, "text")]
    assert "M→c1" in texts and "M→c0" in texts


def test_base_document_structure() -> None:
    """Root attributes, a background and one top group; no filters."""
    circ = quspect.QuantumCircuit()
    circ.h(0)
    root = _parse(draw_svg_circuit(circ))
    assert not _all(root, "filter") and not _all(root, "defs")
    assert len(root.findall(f"{NS}g")) == 1
    bg = _all(root, "rect")[0]
    assert bg.get("fill") == _SVG_THEME_DATA["official"]["background"]
    w, h = root.get("width", ""), root.get("height", "")
    assert root.get("viewBox") == f"0 0 {w} {h}"
    assert bg.get("width") == w and bg.get("height") == h


def test_channel_rendering() -> None:
    """Every channel is one dashed card with its short label."""
    circ = quspect.QuantumCircuit()
    for ch, nq in all_channel_examples():
        circ.append_channel(ch, [0, 2][:nq] if nq == 2 else [0])
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    cards = _rects_filled(root, CUSTOM_STYLE["channel_fill"])
    assert len(cards) == len(all_channel_examples())
    (group,) = [
        g for g in _all(root, "g") if g.get("stroke") == CUSTOM_STYLE["channel_stroke"]
    ]
    assert group.get("stroke-dasharray")
    texts = [t.text for t in _all(root, "text")]
    assert "BF(0.11)" in texts and "GAD(0.25,0.4)" in texts
    # the wire of q1 runs through the two-qubit channel on q0 and q2
    through = [p for p in _all(root, "path") if p.get("stroke-dasharray") == "2 4"]
    assert len(through) == 1 and through[0].get("d", "").count("M") == 1


def test_barrier_rendering_partial_and_full() -> None:
    """A barrier is one dotted line over its qubits (all of them when bare)."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1).h(2).barrier(1, 2)
    circ._instructions.append(  # qubit-less barrier (means "all qubits")
        CircuitInstruction(type=InstructionType.BARRIER, qubits=())
    )
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    data = _path_data(root, stroke=CUSTOM_STYLE["barrier"])
    pieces = [piece for piece in data.split("M") if piece]
    spans = []
    for piece in pieces:
        start, end = piece.split("V")
        spans.append(float(end) - float(start.split()[1]))
    assert len(spans) == 2 and spans[1] - spans[0] == pytest.approx(44.0)


# =============================================================================
# Styles and sizing
# =============================================================================


@pytest.mark.parametrize("name", ["official", "mono", "light", "dark"])
def test_named_themes(name: str) -> None:
    """Each named theme colours background, wires, cards and measurements."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="custom", num_qubits=1), (0,))
    circ.measure([0], [0])
    root = _parse(draw_svg_circuit(circ, style=name))
    theme = _SVG_THEME_DATA[name]
    assert _all(root, "rect")[0].get("fill") == theme["background"]
    assert _path_data(root, stroke=theme["wire"])
    assert len(_rects_filled(root, theme["gate_fill"])) == 1
    (measure,) = [g for g in _all(root, "g") if g.get("fill") == theme["measure_fill"]]
    assert measure.get("stroke") == theme["classical_wire"]


@pytest.mark.parametrize("name", ["official", "mono", "light", "dark"])
def test_theme_text_is_readable(name: str) -> None:
    """Labels contrast with what they are written on (WCAG AA for text)."""

    def luminance(colour: str) -> float:
        channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def contrast(first: str, second: str) -> float:
        high, low = sorted((luminance(first), luminance(second)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    theme = _SVG_THEME_DATA[name]
    assert contrast(theme["text"], theme["gate_fill"]) >= 4.5
    assert contrast(theme["text"], theme["channel_fill"]) >= 4.5
    assert contrast(theme["classical_wire"], theme["measure_fill"]) >= 4.5
    assert contrast(theme["classical_wire"], theme["background"]) >= 4.5
    assert contrast(theme["gate_stroke"], theme["background"]) >= 4.5
    assert contrast(theme["wire"], theme["background"]) >= 3
    # family colours: visible as an outline, and their tint keeps labels readable
    for key in _FAMILY_STYLE_KEYS:
        if key in theme:
            assert contrast(theme[key], theme["background"]) >= 3
            tinted = _mix(theme["gate_fill"], theme[key], _FAMILY_TINT)
            assert contrast(theme["text"], tinted) >= 7


def test_default_style_is_official() -> None:
    """``style=None`` equals ``style='official'``."""
    circ = quspect.QuantumCircuit()
    circ.h(0)
    assert str(draw_svg_circuit(circ)) == str(draw_svg_circuit(circ, style="official"))


def test_custom_dict_and_svgstyle_instance_are_equivalent() -> None:
    """A dict style and the equivalent SvgStyle render the same SVG."""
    circ, _ = random_circuit_pair(3, 10, 1)
    style = SvgStyle.from_mapping(CUSTOM_STYLE)
    assert style.as_dict() == CUSTOM_STYLE
    a = str(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    b = str(draw_svg_circuit(circ, style=style))
    assert a == b
    used = {v for el in _parse(a).iter() for v in el.attrib.values()}
    for key in ("background", "wire", "gate_fill", "gate_stroke", "text", "control"):
        assert CUSTOM_STYLE[key] in used


def test_style_errors() -> None:
    """Unknown theme names and incomplete dicts raise ValueError."""
    circ = quspect.QuantumCircuit()
    circ.h(0)
    with pytest.raises(ValueError, match="Unsupported SVG style"):
        draw_svg_circuit(circ, style="neon")
    partial = dict(CUSTOM_STYLE)
    del partial["wire"]
    with pytest.raises(ValueError, match=r"Missing SVG style keys: \['wire'\]"):
        draw_svg_circuit(circ, style=partial)


def test_width_and_scale() -> None:
    """``width`` is a minimum base width; ``scale`` multiplies the outer size."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1)
    base = _parse(draw_svg_circuit(circ))
    bw, bh = float(base.get("width", "0")), float(base.get("height", "0"))
    wide = _parse(draw_svg_circuit(circ, width=bw + 100))
    assert float(wide.get("width", "0")) == pytest.approx(bw + 100, abs=0.06)
    narrow = _parse(draw_svg_circuit(circ, width=10))
    assert float(narrow.get("width", "0")) == pytest.approx(bw, abs=0.06)
    scaled = _parse(draw_svg_circuit(circ, scale=2))
    assert float(scaled.get("width", "0")) == pytest.approx(2 * bw, abs=0.11)
    assert float(scaled.get("height", "0")) == pytest.approx(2 * bh, abs=0.11)
    assert scaled.get("viewBox") == base.get("viewBox")
    # Height grows by one wire pitch per qubit.
    three = quspect.QuantumCircuit()
    three.h(0).cx(0, 2)
    assert float(_parse(draw_svg_circuit(three)).get("height", "0")) == bh + 44
    # A wider page lengthens the wires, not the gaps between gates.
    wire = _SVG_THEME_DATA["official"]["wire"]
    assert _path_data(wide, stroke=wire).count("M") == 2
    assert str(draw_svg_circuit(circ)).count("<text") == str(
        draw_svg_circuit(circ, width=bw + 100)
    ).count("<text")


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"width": "100"}, TypeError),
        ({"width": 0}, ValueError),
        ({"width": -5.0}, ValueError),
        ({"scale": "2"}, TypeError),
        ({"scale": 0}, ValueError),
        ({"scale": -1.5}, ValueError),
    ],
)
def test_invalid_width_scale(kwargs: dict[str, object], exc: type[Exception]) -> None:
    """Invalid sizing arguments raise TypeError / ValueError."""
    circ = quspect.QuantumCircuit()
    circ.h(0)
    with pytest.raises(exc):
        draw_svg_circuit(circ, **kwargs)  # type: ignore[arg-type]


def test_empty_circuit_svg() -> None:
    """An empty circuit renders a placeholder with min width 220."""
    empty = quspect.QuantumCircuit()
    root = _parse(draw_svg_circuit(empty, style="dark", scale=2))
    assert [t.text for t in _all(root, "text")] == ["(empty circuit)"]
    assert root.get("viewBox") == "0 0 220 72"
    assert root.get("width") == "440"
    assert _all(root, "rect")[0].get("fill") == _SVG_THEME_DATA["dark"]["background"]
    wide = _parse(draw_svg_circuit(empty, width=500))
    assert wide.get("viewBox") == "0 0 500 72"


# =============================================================================
# CircuitSVG wrapper
# =============================================================================


def test_circuit_svg_wrapper_and_file(tmp_path: Path) -> None:
    """``str``, ``_repr_svg_``, ``to_file`` and ``QuantumCircuit.svg`` agree."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1).measure([1], [0])
    svg = draw_svg_circuit(circ, style="light", width=400, scale=1.5)
    assert isinstance(svg, CircuitSVG)
    assert str(svg) == svg.content == svg._repr_svg_()
    out = tmp_path / "c.svg"
    svg.to_file(out)
    assert out.read_text(encoding="utf-8") == svg.content
    assert str(circ.svg(style="light", width=400, scale=1.5)) == svg.content
    _parse(out.read_text(encoding="utf-8"))
    assert np.isfinite(float(_parse(svg).get("width", "nan")))
