"""Tests for SVG circuit rendering."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import isclose

from quspect import QuantumCircuit
from quspect.visualization import SvgStyle, draw_svg_circuit


def _sample_circuit() -> QuantumCircuit:
    circ = QuantumCircuit()
    circ.h(0).cx(0, 1).swap(1, 2).barrier(0, 1, 2).measure([0, 1], [0, 1])
    return circ


def test_svg_circuit_renders_polished_gate_shapes() -> None:
    svg = str(_sample_circuit().svg(style="official"))

    assert 'id="quspect_gate_shadow"' in svg
    assert 'font-family="Arial, Helvetica, sans-serif"' in svg
    assert 'stroke-linecap="round"' in svg
    assert "A 10 10 0 0 1" in svg
    assert 'r="15.0"' in svg


def test_svg_circuit_is_valid_xml() -> None:
    ET.fromstring(str(_sample_circuit().svg(style="dark")))


def test_svg_scale_changes_outer_size_without_internal_transform() -> None:
    svg = str(_sample_circuit().svg(width=320, scale=2))
    root = ET.fromstring(svg)
    view_box_width = float(root.attrib["viewBox"].split()[2])
    outer_width = float(root.attrib["width"])

    assert isclose(outer_width, view_box_width * 2)
    assert 'transform="scale' not in svg


def test_svg_accepts_custom_style_mapping() -> None:
    style = SvgStyle(
        background="#ffffff",
        wire="#111111",
        classical_wire="#555555",
        gate_fill="#fafafa",
        gate_stroke="#222222",
        text="#000000",
        control="#333333",
        barrier="#777777",
        measure_fill="#eeeeee",
        channel_fill="#dddddd",
        channel_stroke="#444444",
    )

    svg = draw_svg_circuit(_sample_circuit(), style=style)

    assert "#222222" in str(svg)


def test_svg_packs_independent_gates_into_one_column() -> None:
    circ = QuantumCircuit().h(0).h(1).h(2)
    root = ET.fromstring(str(circ.svg()))

    h_labels = [element for element in root.iter() if element.text == "H"]

    assert len(h_labels) == 3
    assert len({label.attrib["x"] for label in h_labels}) == 1
    assert len({label.attrib["y"] for label in h_labels}) == 3


def test_svg_keeps_shared_qubit_dependencies_in_later_columns() -> None:
    circ = QuantumCircuit().h(0).x(1).z(0)
    root = ET.fromstring(str(circ.svg()))
    label_x = {
        element.text: float(element.attrib["x"])
        for element in root.iter()
        if element.text in {"H", "X", "Z"}
    }

    assert label_x["H"] == label_x["X"]
    assert label_x["Z"] > label_x["H"]


def test_svg_reserves_intermediate_lanes_for_multiqubit_connectors() -> None:
    circ = QuantumCircuit().cx(0, 2).h(1)
    root = ET.fromstring(str(circ.svg()))
    h_label = next(element for element in root.iter() if element.text == "H")
    control = next(element for element in root.iter() if element.tag.endswith("circle"))

    assert float(h_label.attrib["x"]) > float(control.attrib["cx"])


def test_svg_packs_disjoint_multiqubit_and_single_qubit_gates() -> None:
    circ = QuantumCircuit().cx(0, 1).h(2)
    root = ET.fromstring(str(circ.svg()))
    h_label = next(element for element in root.iter() if element.text == "H")
    control = next(element for element in root.iter() if element.tag.endswith("circle"))

    assert float(h_label.attrib["x"]) == float(control.attrib["cx"])


def test_svg_reserves_measurement_connector_span() -> None:
    circ = QuantumCircuit().measure([0], [0]).h(1)
    root = ET.fromstring(str(circ.svg()))
    h_label = next(element for element in root.iter() if element.text == "H")
    measure_icon = next(
        element
        for element in root.iter()
        if element.tag.endswith("circle") and element.attrib.get("r") == "2.2"
    )

    assert float(h_label.attrib["x"]) > float(measure_icon.attrib["cx"])


def test_controlled_rotation_targets_show_angles() -> None:
    """Controlled-rotation target boxes include their parameters (B82)."""
    circuit = QuantumCircuit()
    circuit.crz(0.3, 0, 1).crx(1.2, 0, 1).cp(0.5, 1, 0).cu3(0.1, 0.2, 0.3, 0, 1)
    texts = [
        element.text
        for element in ET.fromstring(str(draw_svg_circuit(circuit))).iter()
        if element.tag.endswith("text")
    ]
    for label in ("RZ(0.3)", "RX(1.2)", "P(0.5)", "U3(0.1, 0.2, 0.3)"):
        assert label in texts
