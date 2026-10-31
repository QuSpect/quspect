"""Tests for the SVG circuit drawer (:mod:`quspect.visualization.svg_drawer`).

qiskit has no comparable SVG output, so the SVG is parsed with
``xml.etree`` and checked structurally: well-formed XML, element counts per
instruction kind, column packing, style colours and size options.
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
from quspect.visualization.svg_drawer import _SVG_THEME_DATA

NS = "{http://www.w3.org/2000/svg}"
CONTROLLED = {"cx", "cz", "crx", "cry", "crz", "cp", "cu3", "ccx"}

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


def _expected_counts(circ: quspect.QuantumCircuit) -> dict[str, int]:
    """Expected number of gate boxes, controls, targets, measures, barriers."""
    counts = {"gate_box": 0, "control": 0, "plus": 0, "measure": 0, "barrier": 0}
    counts["swap_lines"] = 0
    for instr in circ.instructions:
        if instr.is_gate and instr.gate is not None:
            name = instr.gate.name
            if len(instr.qubits) == 1:
                counts["gate_box"] += 1
            elif name == "swap":
                counts["swap_lines"] += 4
            elif name in CONTROLLED:
                counts["control"] += len(instr.qubits) - 1
                if name in {"cx", "ccx"}:
                    counts["plus"] += 1
                else:
                    counts["gate_box"] += 1
            else:
                counts["gate_box"] += len(instr.qubits)
        elif instr.is_measure:
            counts["measure"] += len(instr.qubits)
        elif instr.is_barrier:
            counts["barrier"] += 1
    return counts


def _observed_counts(root: ET.Element, style: dict[str, str]) -> dict[str, int]:
    """Count the same features in a parsed SVG."""
    rects = _all(root, "rect")
    circles = _all(root, "circle")
    lines = _all(root, "line")
    return {
        "gate_box": sum(r.get("fill") == style["gate_fill"] for r in rects),
        "control": sum(c.get("r") == "5.5" for c in circles),
        "plus": sum(c.get("r") == "15.0" for c in circles),
        "measure": sum(
            r.get("fill") == style["measure_fill"] and r.get("width") == "40.0"
            for r in rects
        ),
        "barrier": sum(r.get("fill-opacity") == "0.38" for r in rects),
        "swap_lines": sum(
            ln.get("stroke-width") == "2.4"
            and ln.get("x1") != ln.get("x2")
            and ln.get("y1") != ln.get("y2")
            for ln in lines
        ),
    }


def _text_x(root: ET.Element, content: str) -> list[float]:
    """x coordinates of ``<text>`` elements whose text equals ``content``."""
    return [float(t.get("x", "nan")) for t in _all(root, "text") if t.text == content]


# =============================================================================
# Structure
# =============================================================================


@pytest.mark.parametrize("seed", range(10))
def test_random_circuit_element_counts(seed: int) -> None:
    """Gate boxes, controls, targets, measures and barriers match the circuit."""
    circ, _ = random_circuit_pair(4, 25, seed)
    circ.barrier(0, 1)
    circ.measure([0, 2], [1, 0])
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    assert _observed_counts(root, CUSTOM_STYLE) == _expected_counts(circ)
    # Lane labels: one per qubit and clbit.
    labels = [t.text for t in _all(root, "text") if t.text and t.text.endswith(":")]
    assert labels == ["q0:", "q1:", "q2:", "q3:", "c0:", "c1:"]
    # Dashed wires are exactly the classical lanes.
    dashed = [
        ln
        for ln in _all(root, "line")
        if ln.get("stroke-dasharray") == "7 5"
        and ln.get("stroke") == CUSTOM_STYLE["classical_wire"]
    ]
    assert len(dashed) == 2


def test_base_document_structure() -> None:
    """Root attributes, defs/filter, background and one ``<g>`` group."""
    circ = quspect.QuantumCircuit()
    circ.h(0)
    root = _parse(draw_svg_circuit(circ))
    assert len(_all(root, "filter")) == 1
    assert _all(root, "filter")[0].get("id") == "quspect_gate_shadow"
    assert len(_all(root, "g")) == 1
    bg = _all(root, "rect")[0]
    assert bg.get("fill") == _SVG_THEME_DATA["official"]["background"]
    w, h = float(root.get("width", "0")), float(root.get("height", "0"))
    assert root.get("viewBox") == f"0 0 {w:.1f} {h:.1f}"


def test_gate_label_text_and_escaping() -> None:
    """Gate labels appear as text and are XML-escaped."""
    circ = quspect.QuantumCircuit()
    circ.rx(0.5, 0)
    circ.append(Gate(name="weird", num_qubits=1, label="<a&b>"), (1,))
    root = _parse(draw_svg_circuit(circ))
    texts = [t.text for t in _all(root, "text")]
    assert "RX(0.5)" in texts
    assert "<a&b>" in texts
    assert "&lt;a&amp;b&gt;" in str(draw_svg_circuit(circ))


@pytest.mark.parametrize("name", sorted(CONTROLLED))
def test_controlled_gate_rendering(name: str) -> None:
    """Controls are dots on the control lanes; target is a plus or a box."""
    circ = quspect.QuantumCircuit()
    nparams = {"cu3": 3, "cx": 0, "cz": 0, "ccx": 0}.get(name, 1)
    qubits = [2, 0, 1] if name == "ccx" else [2, 0]
    getattr(circ, name)(*[0.3] * nparams, *qubits)
    root = _parse(draw_svg_circuit(circ))
    lane_y = {q: 34.0 + 56.0 * q for q in range(3)}
    dots = [c for c in _all(root, "circle") if c.get("r") == "5.5"]
    assert sorted(float(c.get("cy", "0")) for c in dots) == sorted(
        lane_y[q] for q in qubits[:-1]
    )
    if name in {"cx", "ccx"}:
        plus = [c for c in _all(root, "circle") if c.get("r") == "15.0"]
        assert [float(c.get("cy", "0")) for c in plus] == [lane_y[qubits[-1]]]
    else:
        box_texts = [t for t in _all(root, "text") if t.get("font-size") == "12"]
        assert len(box_texts) == 1
        assert float(box_texts[0].get("y", "0")) == lane_y[qubits[-1]] + 4


def test_controlled_rotation_label_shows_angle() -> None:
    """The CRZ target box should display the rotation angle."""
    circ = quspect.QuantumCircuit()
    circ.crz(0.3, 0, 1)
    texts = [t.text or "" for t in _all(_parse(draw_svg_circuit(circ)), "text")]
    assert any("0.3" in t for t in texts)


def test_swap_and_generic_multi_qubit_gate() -> None:
    """SWAP draws two crosses; a custom 2-qubit gate draws a box per operand."""
    circ = quspect.QuantumCircuit()
    circ.swap(0, 2)
    circ.append(Gate(name="rzz", num_qubits=2, params=(0.5,)), (0, 1))
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    obs = _observed_counts(root, CUSTOM_STYLE)
    assert obs["swap_lines"] == 4
    assert obs["gate_box"] == 2
    assert [t.text for t in _all(root, "text")].count("RZZ(0.5)") == 2


def test_measure_rendering() -> None:
    """Each measured pair: meter box on the qubit, numbered circle on the clbit."""
    circ = quspect.QuantumCircuit()
    circ.h(0).measure([0, 1], [1, 0])
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    assert len(_all(root, "path")) == 2
    clbit_circles = [c for c in _all(root, "circle") if c.get("r") == "8"]
    assert len(clbit_circles) == 2
    numbers = [t.text for t in _all(root, "text") if t.get("font-size") == "11"]
    assert sorted(numbers) == ["0", "1"]


def test_gate_named_measure_uses_meter_box() -> None:
    """A single-qubit Gate named ``measure`` is drawn as a meter box."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="measure", num_qubits=1), (0,))
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    assert len(_all(root, "path")) == 1
    assert _observed_counts(root, CUSTOM_STYLE)["measure"] == 1


def test_barrier_rendering_partial_and_full() -> None:
    """Barrier rectangles span exactly the listed lanes (all for empty)."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1).h(2).barrier(1, 2)
    circ._instructions.append(  # qubit-less barrier (means "all qubits")
        CircuitInstruction(type=InstructionType.BARRIER, qubits=())
    )
    root = _parse(draw_svg_circuit(circ))
    bars = [r for r in _all(root, "rect") if r.get("fill-opacity") == "0.38"]
    heights = sorted(float(r.get("height", "0")) for r in bars)
    assert heights == [56.0 + 36.0, 2 * 56.0 + 36.0]


def test_channel_rendering() -> None:
    """Channels are dashed boxes per qubit; multi-qubit ones get a connector."""
    circ = quspect.QuantumCircuit()
    for ch, nq in all_channel_examples():
        circ.append_channel(ch, [0, 2][:nq] if nq == 2 else [0])
    root = _parse(draw_svg_circuit(circ, style=CUSTOM_STYLE))
    chan_rects = [r for r in _all(root, "rect") if r.get("stroke-dasharray") == "5 3"]
    assert len(chan_rects) == len(all_channel_examples()) + 1  # 2q depolarizing
    assert all(r.get("fill") == CUSTOM_STYLE["channel_fill"] for r in chan_rects)
    connectors = [
        ln
        for ln in _all(root, "line")
        if ln.get("stroke") == CUSTOM_STYLE["channel_stroke"]
        and ln.get("stroke-width") == "2.0"
    ]
    assert len(connectors) == 1
    texts = [t.text for t in _all(root, "text")]
    assert "BF(0.11)" in texts and "GAD(0.25,0.4)" in texts


# =============================================================================
# Column packing
# =============================================================================


def test_independent_gates_share_a_column() -> None:
    """Gates on disjoint lanes are vertically aligned; dependent ones are not."""
    circ = quspect.QuantumCircuit()
    circ.h(0).x(1).y(0)
    root = _parse(draw_svg_circuit(circ))
    (hx,), (xx,), (yx,) = _text_x(root, "H"), _text_x(root, "X"), _text_x(root, "Y")
    assert hx == xx
    assert yx > hx


def test_multi_lane_instruction_blocks_lanes_in_between() -> None:
    """A CX(0, 2) reserves lane 1, so a later gate on q1 moves right."""
    circ = quspect.QuantumCircuit()
    circ.cx(0, 2).h(1)
    root = _parse(draw_svg_circuit(circ))
    plus = next(c for c in _all(root, "circle") if c.get("r") == "15.0")
    (hx,) = _text_x(root, "H")
    assert hx > float(plus.get("cx", "0"))


def test_measure_reserves_classical_lanes() -> None:
    """A measure into c0 blocks the qubit lanes below the measured qubit."""
    circ = quspect.QuantumCircuit()
    circ.measure([0], [0]).h(1)
    root = _parse(draw_svg_circuit(circ))
    meter = next(p for p in _all(root, "path"))
    (hx,) = _text_x(root, "H")
    meter_center = float(meter.get("d", "M 0").split()[1]) + 10
    assert hx > meter_center


# =============================================================================
# Styles and sizing
# =============================================================================


@pytest.mark.parametrize("name", ["official", "light", "dark"])
def test_named_themes(name: str) -> None:
    """Each named theme colours background, wires and gate boxes."""
    circ = quspect.QuantumCircuit()
    circ.h(0).measure([0], [0])
    root = _parse(draw_svg_circuit(circ, style=name))
    theme = _SVG_THEME_DATA[name]
    assert _all(root, "rect")[0].get("fill") == theme["background"]
    strokes = {ln.get("stroke") for ln in _all(root, "line")}
    assert theme["wire"] in strokes and theme["classical_wire"] in strokes
    assert any(r.get("fill") == theme["gate_fill"] for r in _all(root, "rect"))


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
    # Height grows with the number of lanes.
    assert bh == 34 + 34 + 56


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
    assert root.get("viewBox") == "0 0 220.0 72.0"
    assert root.get("width") == "440.0"
    assert _all(root, "rect")[0].get("fill") == _SVG_THEME_DATA["dark"]["background"]
    wide = _parse(draw_svg_circuit(empty, width=500))
    assert wide.get("viewBox") == "0 0 500.0 72.0"


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
