"""Tests for :func:`quspect.visualization.draw_text_circuit`.

The structural test compares, wire by wire, the left-to-right sequence of
gate symbols drawn by quspect with the sequence drawn by qiskit's
``QuantumCircuit.draw("text")`` for the same circuit (converted with
:func:`qiskit_tests.helpers.to_qiskit`).  The remaining tests pin the lane
layout, labels, control/target placement and every instruction branch.
Exact drawings are pinned in ``tests/visualization/test_text_drawer.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import quspect
from qiskit_tests._ir_helpers import all_channel_examples
from qiskit_tests.helpers import random_circuit_pair, to_qiskit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter
from quspect.gates.gate import Gate
from quspect.noise.channel import DepolarizingChannel, KrausChannel, NoiseChannel
from quspect.visualization import draw_text_circuit

_QUSPECT_TOKEN = re.compile(r"┤\s*([^├┤]*?)\s*├|(●)|(╳)|(⊕)")
_QISKIT_TOKEN = re.compile(r"┤\s*([^├]*?)\s*├|(■)|(X)")
_CANON = {"U1": "P", "U2": "U", "U3": "U", "I": "ID", "S†": "SDG", "T†": "TDG"}
_WIRE_LINE = re.compile(r"\s*(q\d+) (.*)")

# cz / cp are drawn symmetrically by qiskit (two dots), so they are excluded.
_STRUCT_GATES = [
    "id",
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u1",
    "u2",
    "u3",
    "cx",
    "swap",
    "ccx",
    "crx",
    "cry",
    "crz",
    "cu3",
]


def _canon(label: str) -> str:
    """Canonical gate symbol: base name, upper case, qiskit aliases merged."""
    if label.startswith("M→"):  # quspect names the bit a measurement writes
        return "M"
    base = label.split("(")[0].strip().upper()
    return _CANON.get(base, base)


def _wires(circ: quspect.QuantumCircuit) -> dict[str, str]:
    """Wire rows of the unfolded drawing by qubit label (``q0``, ``q1``, ...)."""
    wires: dict[str, str] = {}
    for line in draw_text_circuit(circ, fold=False).splitlines():
        match = _WIRE_LINE.fullmatch(line)
        if match is not None:
            wires[match.group(1)] = line
    return wires


def _rows(circ: quspect.QuantumCircuit) -> list[str]:
    """All rows of the unfolded drawing, padded to one length."""
    lines = draw_text_circuit(circ, fold=False).splitlines()
    width = max(len(line) for line in lines)
    return [line.ljust(width) for line in lines]


def _column(circ: quspect.QuantumCircuit, col: int) -> str:
    """Characters of column ``col`` from the top row to the bottom row."""
    return "".join(row[col] for row in _rows(circ))


def _tokens(wire: str) -> list[str]:
    """Gate symbols of one wire, left to right (``o`` control, ``x`` swap)."""
    toks = []
    for m in _QUSPECT_TOKEN.finditer(wire):
        if m.group(1) is not None:
            toks.append(_canon(m.group(1)))
        elif m.group(2) is not None:
            toks.append("o")
        elif m.group(3) is not None:
            toks.append("x")
        else:
            toks.append("X")  # the ⊕ target of cx / ccx
    return toks


def _qiskit_wire_tokens(text: str, num_qubits: int) -> list[list[str]]:
    """Same as :func:`_tokens`, per qubit, for qiskit's text drawing."""
    out: list[list[str]] = []
    for q in range(num_qubits):
        line = next(ln for ln in text.splitlines() if ln.startswith(f"q_{q}: "))
        body = line.split(": ", 1)[1]
        toks = []
        for m in _QISKIT_TOKEN.finditer(body):
            if m.group(1) is not None:
                toks.append(_canon(m.group(1)))
            elif m.group(2) is not None:
                toks.append("o")
            else:
                toks.append("x")
        out.append(toks)
    return out


# =============================================================================
# Structure versus qiskit's text drawer
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(10))
def test_wire_gate_order_matches_qiskit_drawer(seed: int) -> None:
    """Each wire shows the same gate/control/swap sequence as qiskit's drawer."""
    nq = 3 + seed % 2
    circ, _ = random_circuit_pair(nq, 18, seed, gate_set=_STRUCT_GATES)
    circ.measure(list(range(nq)), list(range(nq)))
    qiskit_text = str(to_qiskit(circ).draw("text", fold=-1))
    wires = _wires(circ)
    ours = [_tokens(wires[f"q{q}"]) for q in range(nq)]
    theirs = _qiskit_wire_tokens(qiskit_text, nq)
    assert ours == theirs


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("width", [40, 80])
def test_folded_drawing_has_the_same_wires(seed: int, width: int) -> None:
    """Folding cuts between columns: each wire still reads the same."""
    nq = 3 + seed % 2
    circ, _ = random_circuit_pair(nq, 18, seed, gate_set=_STRUCT_GATES)
    circ.measure(list(range(nq)), list(range(nq)))
    folded: dict[str, list[str]] = {}
    text = draw_text_circuit(circ, width=width)
    for line in text.splitlines():
        match = _WIRE_LINE.fullmatch(line)
        if match is not None:
            folded.setdefault(match.group(1), []).extend(_tokens(match.group(2)))
    assert folded == {label: _tokens(wire) for label, wire in _wires(circ).items()}
    assert max(len(line) for line in text.splitlines()) <= width


# =============================================================================
# Lane layout
# =============================================================================


def test_empty_circuit() -> None:
    """An empty circuit renders a placeholder string."""
    assert draw_text_circuit(quspect.QuantumCircuit()) == "(empty circuit)"


@pytest.mark.parametrize(("nq", "nc"), [(1, 0), (3, 0), (2, 2), (11, 3)])
def test_lane_count_labels_and_alignment(nq: int, nc: int) -> None:
    """One wire per qubit with right-aligned labels; no classical wires."""
    circ = quspect.QuantumCircuit()
    circ.h(nq - 1)
    if nc:
        circ.measure([0], [nc - 1])
    lines = draw_text_circuit(circ, fold=False).splitlines()
    width = len(f"q{nq - 1}")
    labelled = [ln for ln in lines if ln[:width].strip()]
    assert [ln[: width + 1] for ln in labelled] == [
        f"q{i}".rjust(width) + " " for i in range(nq)
    ]
    assert len({len(ln) for ln in labelled}) == 1
    # only the two gates need rows above and below their wire
    assert len(lines) == nq + (4 if nc else 2)
    if nc:
        assert f"┤ M→c{nc - 1} ├" in labelled[0]


def test_draw_method_delegates() -> None:
    """``QuantumCircuit.draw`` returns the same text."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1)
    assert circ.draw() == draw_text_circuit(circ)


def test_disjoint_gates_share_a_column() -> None:
    """Instructions are packed: two gates on different qubits get one column."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1)
    wires = _wires(circ)
    assert wires["q0"].index("┤ H ├") == wires["q1"].index("┤ H ├")


def test_gates_on_one_qubit_get_successive_columns() -> None:
    """Gates sharing a qubit keep their order from left to right."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1).h(1)
    wires = _wires(circ)
    assert wires["q0"].index("┤ H ├") < wires["q0"].index("●")
    assert wires["q1"].index("⊕") < wires["q1"].index("┤ H ├")


# =============================================================================
# Labels and parameters
# =============================================================================


def test_single_qubit_labels_on_correct_lines() -> None:
    """Gate labels (with their parameters) appear only on their own qubit line."""
    theta = Parameter("theta")
    circ = quspect.QuantumCircuit()
    circ.h(0).rx(0.5, 1).u3(1.0, 2.25, -0.125, 2).rz(theta, 0).ry(2 * theta + 1, 1)
    wires = _wires(circ)
    q0, q1, q2 = wires["q0"], wires["q1"], wires["q2"]
    assert "┤ H ├" in q0 and "┤ RZ(theta) ├" in q0
    assert "┤ RX(0.5) ├" in q1
    assert "┤ U3(1, 2.25, -0.125) ├" in q2
    expr_label = re.search(r"┤ RY\(([^├]*)\) ├", q1)
    assert expr_label is not None and "theta" in expr_label.group(1)
    for label, line in (("┤ H ├", q0), ("┤ RX(0.5) ├", q1)):
        others = [ln for ln in (q0, q1, q2) if ln is not line]
        assert all(label not in ln for ln in others)


def test_angles_that_are_multiples_of_pi_are_shown_as_such() -> None:
    """``π/2`` rather than ``1.5708``; other values keep four digits."""
    circ = quspect.QuantumCircuit()
    circ.rx(np.pi / 2, 0).rz(-3 * np.pi / 4, 0).ry(0.123456, 0)
    q0 = _wires(circ)["q0"]
    assert "┤ RX(π/2) ├" in q0 and "┤ RZ(-3π/4) ├" in q0 and "┤ RY(0.1235) ├" in q0


def test_custom_gate_label_is_used() -> None:
    """A Gate ``label`` overrides the upper-cased name."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="rx", num_qubits=1, params=(0.25,), label="Drive"), (0,))
    assert "┤ Drive(0.25) ├" in _wires(circ)["q0"]


def test_wide_labels_widen_the_column() -> None:
    """A column is as wide as its widest card; narrower items are centred in it."""
    circ = quspect.QuantumCircuit()
    circ.u3(0.123456, 1.5, 2.75, 0).h(1)
    wires = _wires(circ)
    wide = "┤ U3(0.1235, 1.5, 2.75) ├"
    assert wires["q0"] == f"q0 ──{wide}──"
    assert len(wires["q1"]) == len(wires["q0"])
    offset = wires["q1"].index("┤ H ├") - wires["q0"].index(wide)
    assert offset == (len(wide) - len("┤ H ├")) // 2


@pytest.mark.parametrize("name", ["crx", "cry", "crz", "cp", "cu3"])
def test_controlled_rotation_shows_angle(name: str) -> None:
    """Controlled rotations display their parameters like qiskit does."""
    circ = quspect.QuantumCircuit()
    params = [0.3, 0.4, 0.5][: 3 if name == "cu3" else 1]
    getattr(circ, name)(*params, 0, 1)
    assert "0.3" in _wires(circ)["q1"]


# =============================================================================
# Multi-qubit symbols
# =============================================================================


@pytest.mark.parametrize(
    ("name", "target_label"),
    [("crx", "RX"), ("cry", "RY"), ("crz", "RZ"), ("cp", "P"), ("cu3", "U3")],
)
def test_controlled_gate_layout(name: str, target_label: str) -> None:
    """Control ``●`` under the target card; the line between crosses wires as ``┼``."""
    circ = quspect.QuantumCircuit()
    params = {"cu3": (0.1, 0.2, 0.3)}.get(name, (0.7,))
    getattr(circ, name)(*params, 2, 0)
    wires = _wires(circ)
    card = f"┤ {target_label}(" + ", ".join(f"{p:g}" for p in params) + ") ├"
    assert card in wires["q0"]
    col = wires["q2"].index("●")
    assert wires["q2"].count("●") == 1
    assert wires["q0"].index(card) < col < wires["q0"].index(card) + len(card)
    # from the bottom edge of the card down to the control: one unbroken line
    assert _column(circ, col)[2:] == "┬┼│●"


def test_cx_layout() -> None:
    """CX: ``●`` on the control, ``⊕`` on the target, a line between them."""
    circ = quspect.QuantumCircuit()
    circ.cx(2, 0)
    assert _column(circ, _wires(circ)["q0"].index("⊕")) == "⊕│┼│●"


def test_cz_layout() -> None:
    """CZ is symmetric: a dot on both qubits."""
    circ = quspect.QuantumCircuit()
    circ.cz(2, 0)
    assert _column(circ, _wires(circ)["q0"].index("●")) == "●│┼│●"


def test_toffoli_layout() -> None:
    """CCX: two controls and a ``⊕`` target in one column."""
    circ = quspect.QuantumCircuit()
    circ.ccx(0, 3, 1)
    assert _column(circ, _wires(circ)["q0"].index("●")) == "●│⊕│┼│●"


def test_swap_layout() -> None:
    """SWAP: ``╳`` on both qubits and a vertical line between them."""
    circ = quspect.QuantumCircuit()
    circ.swap(2, 0)
    assert _column(circ, _wires(circ)["q0"].index("╳")) == "╳│┼│╳"


def test_generic_multi_qubit_gate() -> None:
    """A non-controlled multi-qubit gate is one card entered by its operands."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="rzz", num_qubits=2, params=(0.5,)), (0, 2))
    rows = _rows(circ)
    assert sum("RZZ(0.5)" in row for row in rows) == 1
    wires = _wires(circ)
    left = wires["q0"].index("┤")
    right = wires["q0"].index("├")
    assert wires["q2"][left] == "┤" and wires["q2"][right] == "├"
    # the wire of q1 runs through the card
    assert wires["q1"][left] == "┼" and wires["q1"][right] == "┼"
    assert set(wires["q1"][left + 1 : right]) == {"┄"}
    assert _column(circ, left) == "╭┤│┼│┤╰"


def test_generic_gate_operand_numbers() -> None:
    """Operands of a gate that is not symmetric are numbered in the card."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="foo", num_qubits=2), (1, 0))
    wires = _wires(circ)
    assert "┤1 " in wires["q0"] and "┤0 " in wires["q1"]


# =============================================================================
# Measurement, barrier, channels
# =============================================================================


def test_measurement_names_its_classical_bit() -> None:
    """A measurement is a card on the qubit that names the bit it writes."""
    circ = quspect.QuantumCircuit()
    circ.h(0).measure([0], [1])
    assert _wires(circ)["q0"] == "q0 ──┤ H ├──┤ M→c1 ├──"


def test_multi_measure_one_card_each() -> None:
    """Each pair of a multi-qubit measure has its own card, in one column."""
    circ = quspect.QuantumCircuit()
    circ.x(1).measure([0, 2], [1, 0])
    wires = _wires(circ)
    assert wires["q0"].index("┤ M→c1 ├") == wires["q2"].index("┤ M→c0 ├")
    assert "M" not in wires["q1"]


def test_barrier_partial_and_full() -> None:
    """A barrier marks only its qubits; an empty-qubit barrier marks all."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1).h(2).barrier(0, 2)
    circ._instructions.append(  # qubit-less barrier (means "all qubits")
        CircuitInstruction(type=InstructionType.BARRIER, qubits=())
    )
    wires = _wires(circ)
    marks = {q: [i for i, ch in enumerate(wires[q]) if ch == "┆"] for q in wires}
    assert len(marks["q0"]) == 2
    assert len(marks["q1"]) == 1
    assert len(marks["q2"]) == 2
    assert marks["q1"][0] == marks["q0"][1]
    # the full barrier is one unbroken line; the wire of q1 passes in front
    # of the partial one
    assert set(_column(circ, marks["q1"][0])) == {"┆"}
    assert _column(circ, marks["q0"][0]) == "┆┆┆┆─┆┆┆┆"


_EXPECTED_CHANNEL_LABELS = {
    "bit_flip": "BF(0.11)",
    "phase_flip": "PF(0.12)",
    "bit_phase_flip": "BPF(0.13)",
    "depolarizing": "DP(0.14)",
    "pauli": "PC(0.1,0.05,0.02)",
    "amplitude_damping": "AD(0.2)",
    "phase_damping": "PD(0.3)",
    "generalized_amplitude_damping": "GAD(0.25,0.4)",
    "readout": "REA",
}


@pytest.mark.parametrize(
    "index", range(len(all_channel_examples())), ids=lambda i: str(i)
)
def test_channel_labels(index: int) -> None:
    """Each channel type has its short label in a card with a dashed outline."""
    ch, nq = all_channel_examples()[index]
    circ = quspect.QuantumCircuit()
    circ.append_channel(ch, list(range(nq)))
    rows = _rows(circ)
    label = "DP(0.05)" if nq == 2 else _EXPECTED_CHANNEL_LABELS[ch.name]
    assert rows[0].strip() == "╭" + "╌" * (len(label) + 2) + "╮"
    assert rows[-1].strip() == "╰" + "╌" * (len(label) + 2) + "╯"
    if nq == 1:
        assert f"┤ {label} ├" in rows[1]
    else:
        assert f"╎ {label} ╎" in rows[2]


def test_multi_qubit_channel_spans_the_gap() -> None:
    """A 2-qubit channel on non-adjacent qubits is one card over the gap."""
    circ = quspect.QuantumCircuit()
    circ.append_channel(
        DepolarizingChannel(name="depolarizing", num_qubits=2, p=0.2), [0, 2]
    )
    rows = _rows(circ)
    assert sum("DP(0.2)" in row for row in rows) == 1
    wires = _wires(circ)
    left = wires["q0"].index("┤")
    assert wires["q2"][left] == "┤"
    assert wires["q1"][left] == "┼" and "┄" in wires["q1"]
    assert _column(circ, left) == "╭┤╎┼╎┤╰"


def test_kraus_and_custom_label_channels() -> None:
    """``kraus`` named channels show their name; unknown names use the label."""
    eye = np.eye(2, dtype=complex)
    circ = quspect.QuantumCircuit()
    circ.append_channel(KrausChannel(name="kraus", num_qubits=1, kraus_ops=[eye]), [0])

    class Custom(NoiseChannel):
        def kraus_operators(self) -> list[np.ndarray]:
            return [eye]

    circ.append_channel(Custom(name="leak", num_qubits=1, label="leakage-model"), [0])
    circ.append_channel(Custom(name="leak", num_qubits=1), [0])
    line = _wires(circ)["q0"]
    assert "┤ KC(kraus) ├" in line
    assert "┤ leakage- ├" in line  # label truncated to 8 characters
    assert "┤ LEA ├" in line
