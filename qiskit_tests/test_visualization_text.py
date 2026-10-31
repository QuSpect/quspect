"""Tests for :func:`quspect.visualization.draw_text_circuit`.

The structural test compares, wire by wire, the left-to-right sequence of
gate symbols drawn by quspect with the sequence drawn by qiskit's
``QuantumCircuit.draw("text")`` for the same circuit (converted with
:func:`qiskit_tests.helpers.to_qiskit`).  The remaining tests pin the lane
layout, labels, control/target placement and every instruction branch.
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

_QUSPECT_TOKEN = re.compile(r"\[([^\]]*)\]|(?<=-)([ox])(?=-)")
_QISKIT_TOKEN = re.compile(r"┤\s*([^├]*?)\s*├|(■)|(X)")
_CANON = {"U1": "P", "U2": "U", "U3": "U", "I": "ID", "S†": "SDG", "T†": "TDG"}

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
    base = label.split("(")[0].strip().upper()
    return _CANON.get(base, base)


def _quspect_wire_tokens(text: str, num_qubits: int) -> list[list[str]]:
    """Per qubit line: gate symbols left to right (``o`` control, ``x`` swap)."""
    lines = text.splitlines()[:num_qubits]
    out: list[list[str]] = []
    for line in lines:
        body = line.split(": ", 1)[1]
        toks = []
        for m in _QUSPECT_TOKEN.finditer(body):
            toks.append(_canon(m.group(1)) if m.group(1) is not None else m.group(2))
        out.append(toks)
    return out


def _qiskit_wire_tokens(text: str, num_qubits: int) -> list[list[str]]:
    """Same as :func:`_quspect_wire_tokens` for qiskit's text drawing."""
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


def _columns_of(line: str, symbol: str) -> list[int]:
    """Character indices of ``symbol`` in ``line``."""
    return [i for i, ch in enumerate(line) if ch == symbol]


def _lines(circ: quspect.QuantumCircuit) -> list[str]:
    """Rendered lines of ``circ``."""
    return draw_text_circuit(circ).splitlines()


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
    ours = _quspect_wire_tokens(draw_text_circuit(circ), nq)
    theirs = _qiskit_wire_tokens(qiskit_text, nq)
    assert ours == theirs


# =============================================================================
# Lane layout
# =============================================================================


def test_empty_circuit() -> None:
    """An empty circuit renders a placeholder string."""
    assert draw_text_circuit(quspect.QuantumCircuit()) == "(empty circuit)"


@pytest.mark.parametrize(("nq", "nc"), [(1, 0), (3, 0), (2, 2), (11, 3)])
def test_lane_count_labels_and_alignment(nq: int, nc: int) -> None:
    """One line per qubit then per clbit; right-aligned labels; equal widths."""
    circ = quspect.QuantumCircuit()
    circ.h(nq - 1)
    if nc:
        circ.measure([0], [nc - 1])
    lines = _lines(circ)
    assert len(lines) == nq + nc
    width = len(f"q{nq - 1}") if nq + nc else 0
    width = max(width, len(f"c{nc - 1}") if nc else 0)
    expected = [f"q{i}".rjust(width) + ": " for i in range(nq)]
    expected += [f"c{i}".rjust(width) + ": " for i in range(nc)]
    assert [ln[: width + 2] for ln in lines] == expected
    assert len({len(ln) for ln in lines}) == 1
    for ln in lines[nq:]:
        assert set(ln[width + 2 :]) <= set("=|0123456789")


def test_draw_method_delegates() -> None:
    """``QuantumCircuit.draw`` returns the same text."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1)
    assert circ.draw() == draw_text_circuit(circ)


def test_one_column_per_instruction() -> None:
    """Instructions are not packed: two gates on different qubits get two columns."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1)
    q0, q1 = _lines(circ)
    assert q0.index("[H]") < q1.index("[H]")


# =============================================================================
# Labels and parameters
# =============================================================================


def test_single_qubit_labels_on_correct_lines() -> None:
    """Gate labels (with ``:g`` params) appear only on their own qubit line."""
    theta = Parameter("theta")
    circ = quspect.QuantumCircuit()
    circ.h(0).rx(0.5, 1).u3(1.0, 2.25, -0.125, 2).rz(theta, 0).ry(2 * theta + 1, 1)
    q0, q1, q2 = _lines(circ)
    assert "[H]" in q0 and "[RZ(theta)]" in q0
    assert "[RX(0.5)]" in q1
    assert "[U3(1, 2.25, -0.125)]" in q2
    expr_label = re.search(r"\[RY\(([^\]]*)\)\]", q1)
    assert expr_label is not None and "theta" in expr_label.group(1)
    for label, line in (("[H]", q0), ("[RX(0.5)]", q1)):
        others = [ln for ln in (q0, q1, q2) if ln is not line]
        assert all(label not in ln for ln in others)


def test_custom_gate_label_is_used() -> None:
    """A Gate ``label`` overrides the upper-cased name."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="rx", num_qubits=1, params=(0.25,), label="Drive"), (0,))
    assert "[Drive(0.25)]" in _lines(circ)[0]


def test_wide_labels_widen_the_column_symmetrically() -> None:
    """The cell width grows with the label and stays odd (centred symbols)."""
    circ = quspect.QuantumCircuit()
    circ.u3(0.123456, 1.5, 2.75, 0)
    circ.cx(0, 1)
    q0, q1 = _lines(circ)
    assert q0.startswith("q0: -[U3(0.123456, 1.5, 2.75)]-")
    assert len(q0) == len(q1)


@pytest.mark.parametrize("name", ["crx", "cry", "crz", "cp", "cu3"])
def test_controlled_rotation_shows_angle(name: str) -> None:
    """Controlled rotations display their parameters like qiskit does."""
    circ = quspect.QuantumCircuit()
    params = [0.3, 0.4, 0.5][: 3 if name == "cu3" else 1]
    getattr(circ, name)(*params, 0, 1)
    assert "0.3" in _lines(circ)[1]


# =============================================================================
# Multi-qubit symbols
# =============================================================================


@pytest.mark.parametrize(
    ("name", "target_label"),
    [("cx", "X"), ("cz", "Z"), ("crx", "RX"), ("cry", "RY"), ("crz", "RZ")]
    + [("cp", "P"), ("cu3", "U3")],
)
def test_controlled_gate_layout(name: str, target_label: str) -> None:
    """Control ``o`` and target box share a column; wire ``|`` on lanes between."""
    circ = quspect.QuantumCircuit()
    params = {"cu3": (0.1, 0.2, 0.3)}.get(name, (0.7,) if name[1] in "rp" else ())
    getattr(circ, name)(*params, 2, 0)
    q0, q1, q2 = _lines(circ)
    if params:
        target_label += "(" + ", ".join(f"{p:g}" for p in params) + ")"
    assert f"[{target_label}]" in q0
    ctrl = _columns_of(q2, "o")
    assert len(ctrl) == 1
    target_center = q0.index(f"[{target_label}]") + (len(target_label) + 2) // 2
    assert target_center == ctrl[0]
    assert _columns_of(q1, "|") == ctrl


def test_toffoli_layout() -> None:
    """CCX: two controls and an X target in one column."""
    circ = quspect.QuantumCircuit()
    circ.ccx(0, 3, 1)
    q0, q1, q2, q3 = _lines(circ)
    col = _columns_of(q0, "o")
    assert col == _columns_of(q3, "o")
    assert q1[col[0] - 1 : col[0] + 2] == "[X]"
    assert _columns_of(q2, "|") == col


def test_swap_layout() -> None:
    """SWAP: ``x`` on both qubits and a vertical wire between them."""
    circ = quspect.QuantumCircuit()
    circ.swap(2, 0)
    q0, q1, q2 = _lines(circ)
    assert _columns_of(q0, "x") == _columns_of(q2, "x") == _columns_of(q1, "|")


def test_generic_multi_qubit_gate() -> None:
    """A non-controlled multi-qubit gate is boxed on every operand."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="rzz", num_qubits=2, params=(0.5,)), (0, 2))
    q0, q1, q2 = _lines(circ)
    assert "[RZZ(0.5)]" in q0 and "[RZZ(0.5)]" in q2
    assert len(_columns_of(q1, "|")) == 1


# =============================================================================
# Measurement, barrier, channels
# =============================================================================


def test_measurement_connects_qubit_and_clbit() -> None:
    """``[M]`` on the qubit, clbit index on its classical lane, wire between."""
    circ = quspect.QuantumCircuit()
    circ.h(0).measure([0], [1])
    q0, c0, c1 = _lines(circ)[0], _lines(circ)[1], _lines(circ)[2]
    m_col = q0.index("[M]") + 1
    assert c1[m_col] == "1"
    assert c0[m_col] == "|"
    assert set(c0.split(": ")[1]) == {"=", "|"}


def test_multi_measure_single_column() -> None:
    """A multi-qubit measure draws every pair in one column."""
    circ = quspect.QuantumCircuit()
    circ.x(1).measure([0, 2], [1, 0])
    q0, q1, q2, c0, c1 = _lines(circ)
    col = q0.index("[M]") + 1
    assert q2[col - 1 : col + 2] == "[M]"
    assert c0[col] == "0" and c1[col] == "1"
    assert q1[col] == "|"


def test_barrier_partial_and_full() -> None:
    """A barrier marks only its qubits; an empty-qubit barrier marks all."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1).h(2).barrier(0, 2)
    circ._instructions.append(  # qubit-less barrier (means "all qubits")
        CircuitInstruction(type=InstructionType.BARRIER, qubits=())
    )
    q0, q1, q2 = _lines(circ)
    assert len(_columns_of(q0, "|")) == 2
    assert len(_columns_of(q1, "|")) == 1
    assert len(_columns_of(q2, "|")) == 2
    assert _columns_of(q1, "|")[0] == _columns_of(q0, "|")[1]


_EXPECTED_CHANNEL_LABELS = {
    "bit_flip": "╡BF(0.11)╞",
    "phase_flip": "╡PF(0.12)╞",
    "bit_phase_flip": "╡BPF(0.13)╞",
    "depolarizing": "╡DP(0.14)╞",
    "pauli": "╡PC(0.1,0.05,0.02)╞",
    "amplitude_damping": "╡AD(0.2)╞",
    "phase_damping": "╡PD(0.3)╞",
    "generalized_amplitude_damping": "╡GAD(0.25,0.4)╞",
    "readout": "╡REA╞",
}


@pytest.mark.parametrize(
    "index", range(len(all_channel_examples())), ids=lambda i: str(i)
)
def test_channel_labels(index: int) -> None:
    """Each channel type has its double-bordered short label."""
    ch, nq = all_channel_examples()[index]
    circ = quspect.QuantumCircuit()
    circ.append_channel(ch, list(range(nq)))
    lines = _lines(circ)
    if nq == 1:
        assert _EXPECTED_CHANNEL_LABELS[ch.name] in lines[0]
    else:
        assert "╡DP(0.05)╞" in lines[0] and "╡DP(0.05)╞" in lines[1]


def test_multi_qubit_channel_spans_with_wire() -> None:
    """A 2-qubit channel on non-adjacent qubits draws a wire through the gap."""
    circ = quspect.QuantumCircuit()
    circ.append_channel(
        DepolarizingChannel(name="depolarizing", num_qubits=2, p=0.2), [0, 2]
    )
    q0, q1, q2 = _lines(circ)
    assert "╡DP(0.2)╞" in q0 and "╡DP(0.2)╞" in q2
    assert len(_columns_of(q1, "|")) == 1


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
    line = _lines(circ)[0]
    assert "╡KC(kraus)╞" in line
    assert "╡leakage-╞" in line  # label truncated to 8 characters
    assert "╡LEA╞" in line
