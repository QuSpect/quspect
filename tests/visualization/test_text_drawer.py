"""Tests for the Unicode text circuit drawer."""

from __future__ import annotations

import math
import os
import random

import pytest

from quspect import QuantumCircuit
from quspect.circuit.parameter import Parameter
from quspect.gates.gate import Gate
from quspect.noise.channel import DepolarizingChannel
from quspect.visualization import text_drawer
from quspect.visualization.text_drawer import (
    _balanced_chunks,
    _chunk_layers,
    _format_number,
    _layer_starts,
    _run_width,
    draw_text_circuit,
    terminal_width,
)


def _art(circuit: QuantumCircuit, **options: object) -> str:
    """The drawing on one line per lane, framed by newlines like the goldens."""
    options.setdefault("fold", False)
    return "\n" + draw_text_circuit(circuit, **options) + "\n"  # type: ignore[arg-type]


def _wires(text: str) -> list[str]:
    """The wire lines of a drawing (those that start with a qubit label)."""
    return [line for line in text.splitlines() if line.lstrip().startswith("q")]


# ----------------------------------------------------------------- goldens
def test_bell_circuit_with_measurements() -> None:
    """A measurement names the classical bit it writes."""
    circuit = QuantumCircuit().h(0).cx(0, 1).measure([0, 1], [0, 1])
    expected = """
     ╭───╮       ╭──────╮
q0 ──┤ H ├───●───┤ M→c0 ├──
     ╰───╯   │   ╰──────╯
             │   ╭──────╮
q1 ──────────⊕───┤ M→c1 ├──
                 ╰──────╯
"""
    assert _art(circuit) == expected


def test_controlled_rotation_joins_control_to_the_target_card() -> None:
    """The control line ends on the edge of the card, from above and from below."""
    circuit = QuantumCircuit().crz(math.pi / 4, 0, 2).crz(0.3, 2, 0)
    expected = """
                  ╭─────────╮
q0 ───────●───────┤ RZ(0.3) ├──
          │       ╰────┬────╯
q1 ───────┼────────────┼───────
     ╭────┴────╮       │
q2 ──┤ RZ(π/4) ├───────●───────
     ╰─────────╯
"""
    assert _art(circuit) == expected


@pytest.mark.parametrize(
    ("name", "params", "label"),
    [
        ("crx", (0.3,), "RX(0.3)"),
        ("cry", (0.3,), "RY(0.3)"),
        ("crz", (0.3,), "RZ(0.3)"),
        ("cp", (0.3,), "P(0.3)"),
        ("cu3", (0.3, 0.4, 0.5), "U3(0.3, 0.4, 0.5)"),
    ],
)
def test_controlled_rotation_target_shows_parameters(
    name: str, params: tuple[float, ...], label: str
) -> None:
    """Controlled-rotation targets display their angles (B83)."""
    circuit = QuantumCircuit()
    getattr(circuit, name)(*params, 0, 1)
    control_line, target_line = _wires(draw_text_circuit(circuit))
    card = f"┤ {label} ├"
    assert card in target_line
    # The control dot sits over the target card.
    start = target_line.index(card)
    assert start < control_line.index("●") < start + len(card)


def test_controlled_rotations_with_different_angles_differ() -> None:
    """``crz(0.3)`` and ``crz(1.2)`` do not render identically."""
    a = QuantumCircuit().crz(0.3, 0, 1)
    b = QuantumCircuit().crz(1.2, 0, 1)
    assert draw_text_circuit(a) != draw_text_circuit(b)


def test_toffoli_with_the_target_between_the_controls() -> None:
    expected = """
q0 ───●───
      │
q1 ───⊕───
      │
q2 ───┼───
      │
q3 ───●───
"""
    assert _art(QuantumCircuit().ccx(0, 3, 1)) == expected


def test_swap_and_cz_use_symbols() -> None:
    expected = """
q0 ───╳───●───
      │   │
q1 ───┼───●───
      │
q2 ───╳───────
"""
    assert _art(QuantumCircuit().swap(2, 0).cz(0, 1)) == expected


def test_two_qubit_rotation_is_one_card() -> None:
    """A wire the gate does not act on runs through the card, dashed."""
    circuit = QuantumCircuit().rzz(math.pi / 4, 0, 1).rxx(0.5, 0, 2)
    expected = """
     ╭──────────╮  ╭──────────╮
q0 ──┤          ├──┤          ├──
     │ RZZ(π/4) │  │ RXX(0.5) │
q1 ──┤          ├──┼┄┄┄┄┄┄┄┄┄┄┼──
     ╰──────────╯  │          │
q2 ────────────────┤          ├──
                   ╰──────────╯
"""
    assert _art(circuit) == expected


def test_generic_multi_qubit_gate_numbers_its_operands() -> None:
    circuit = QuantumCircuit()
    circuit.append(Gate(name="foo", num_qubits=3, params=(0.5,)), (3, 0, 1))
    expected = """
     ╭────────────╮
q0 ──┤1           ├──
     │            │
q1 ──┤2           ├──
     │  FOO(0.5)  │
q2 ──┼┄┄┄┄┄┄┄┄┄┄┄┄┼──
     │            │
q3 ──┤0           ├──
     ╰────────────╯
"""
    assert _art(circuit) == expected


def test_label_next_to_the_number_of_a_middle_operand() -> None:
    """With an odd number of lanes the label is on the middle wire."""
    circuit = QuantumCircuit()
    circuit.append(Gate(name="u", num_qubits=3), (2, 1, 0))
    expected = """
     ╭─────╮
q0 ──┤2    ├──
     │     │
q1 ──┤1 U  ├──
     │     │
q2 ──┤0    ├──
     ╰─────╯
"""
    assert _art(circuit) == expected


def test_classical_control_shows_the_condition_on_top() -> None:
    """In the top edge of a card, or over the first symbol of a gate without one."""
    circuit = QuantumCircuit().h(0).measure([0, 1], [0, 1])
    with circuit.if_test([0, 1], 1):
        circuit.rz(0.5, 0)
    with circuit.if_test(1):
        circuit.x(2)
    with circuit.if_test(0, 0):
        circuit.cx(1, 2)
    expected = """
     ╭───╮  ╭──────╮  ╭─ c0=1 c1=0 ─╮
q0 ──┤ H ├──┤ M→c0 ├──┤   RZ(0.5)   ├────────────
     ╰───╯  ╰──────╯  ╰─────────────╯
            ╭──────╮                    ╭─ c0=0
q1 ─────────┤ M→c1 ├────────────────────●────────
            ╰──────╯                    │
                        ╭─ c1=1 ──╮     │
q2 ─────────────────────┤    X    ├─────⊕────────
                        ╰─────────╯
"""
    assert _art(circuit) == expected


def test_condition_of_a_controlled_card() -> None:
    """Over the control when it is on top, else in the top edge of the card."""
    circuit = QuantumCircuit().measure([0], [0])
    with circuit.if_test(0):
        circuit.crz(0.5, 0, 1)
    with circuit.if_test(0):
        circuit.crz(0.5, 1, 0)
    expected = """
     ╭──────╮       ╭─ c0=1  ╭─ c0=1 ──╮
q0 ──┤ M→c0 ├───────●────────┤ RZ(0.5) ├──
     ╰──────╯       │        ╰────┬────╯
               ╭────┴────╮        │
q1 ────────────┤ RZ(0.5) ├────────●───────
               ╰─────────╯
"""
    assert _art(circuit) == expected


def test_conditional_gate_stays_after_the_measurement_it_reads() -> None:
    """Classical bits are not drawn, but they still order the columns."""
    circuit = QuantumCircuit().measure([0], [0])
    with circuit.if_test(0):
        circuit.x(1)
    q0, q1 = _wires(draw_text_circuit(circuit))
    assert q1.index("┤    X    ├") > q0.index("┤ M→c0 ├")


def test_gates_reading_one_bit_may_share_a_column() -> None:
    """Reading a bit does not order two gates; writing it again does."""
    circuit = QuantumCircuit().measure([0], [0])
    with circuit.if_test(0):
        circuit.x(1)
    with circuit.if_test(0):
        circuit.x(2)
    circuit.measure([3], [0])
    q0, q1, q2, q3 = _wires(draw_text_circuit(circuit))
    first = q0.index("M→c0")
    assert first < q1.index("X") == q2.index("X") < q3.index("M→c0")


def test_barriers() -> None:
    """A dashed line over the barrier's qubits; a bare barrier covers all."""
    circuit = QuantumCircuit().h(0).h(1).h(2).barrier(0, 1).x(2).barrier(2).barrier()
    expected = """
     ╭───╮    ┆       ┆
q0 ──┤ H ├────┆───────┆──
     ╰───╯    ┆       ┆
     ╭───╮    ┆       ┆
q1 ──┤ H ├────┆───────┆──
     ╰───╯    ┆       ┆
     ╭───╮  ╭───╮  ┆  ┆
q2 ──┤ H ├──┤ X ├──┆──┆──
     ╰───╯  ╰───╯  ┆  ┆
"""
    assert _art(circuit) == expected


def test_barrier_alone_does_not_add_rows() -> None:
    expected = """
q0 ──┆──
q1 ──┆──
"""
    assert _art(QuantumCircuit().barrier(0, 1)) == expected


def test_noise_channels_have_a_dashed_outline() -> None:
    circuit = QuantumCircuit().depolarize(0.1, 0)
    circuit.append_channel(
        DepolarizingChannel(name="depolarizing", num_qubits=2, p=0.2), [0, 2]
    )
    expected = """
     ╭╌╌╌╌╌╌╌╌╌╮  ╭╌╌╌╌╌╌╌╌╌╮
q0 ──┤ DP(0.1) ├──┤         ├──
     ╰╌╌╌╌╌╌╌╌╌╯  ╎ DP(0.2) ╎
q1 ───────────────┼┄┄┄┄┄┄┄┄┄┼──
                  ╎         ╎
q2 ───────────────┤         ├──
                  ╰╌╌╌╌╌╌╌╌╌╯
"""
    assert _art(circuit) == expected


def test_empty_circuit() -> None:
    assert draw_text_circuit(QuantumCircuit()) == "(empty circuit)"


# ------------------------------------------------------------------ layout
def test_gates_on_disjoint_qubits_share_a_column() -> None:
    """``x(1)`` waits for the CX whose line crosses its wire; ``x(3)`` does not."""
    circuit = QuantumCircuit().h(0).h(1).cx(0, 2).x(1).x(3)
    expected = """
     ╭───╮
q0 ──┤ H ├───●──────────
     ╰───╯   │
     ╭───╮   │   ╭───╮
q1 ──┤ H ├───┼───┤ X ├──
     ╰───╯   │   ╰───╯
q2 ──────────⊕──────────
     ╭───╮
q3 ──┤ X ├──────────────
     ╰───╯
"""
    assert _art(circuit) == expected


def test_measurements_of_different_qubits_share_a_column() -> None:
    circuit = QuantumCircuit().h(0).h(1).h(2).measure_all()
    columns = {line.index("M→") for line in _wires(draw_text_circuit(circuit))}
    assert len(columns) == 1


def test_cards_of_one_measurement_line_up() -> None:
    """``measure_all`` is one column, although q0 is free two columns earlier."""
    circuit = QuantumCircuit().h(0).cx(0, 1).cx(1, 2).measure_all()
    expected = """
     ╭───╮           ╭──────╮
q0 ──┤ H ├───●───────┤ M→c0 ├──
     ╰───╯   │       ╰──────╯
             │       ╭──────╮
q1 ──────────⊕───●───┤ M→c1 ├──
                 │   ╰──────╯
                 │   ╭──────╮
q2 ──────────────⊕───┤ M→c2 ├──
                     ╰──────╯
"""
    assert _art(circuit) == expected


def test_separate_measurements_go_as_early_as_they_can() -> None:
    circuit = (
        QuantumCircuit().h(0).cx(0, 1).cx(1, 2).measure([0], [0]).measure([2], [2])
    )
    q0, _, q2 = _wires(draw_text_circuit(circuit))
    assert q0.index("M→c0") < q2.index("M→c2")


def test_measuring_one_qubit_twice_in_one_instruction() -> None:
    circuit = QuantumCircuit().measure([0, 0], [0, 1])
    (wire,) = _wires(draw_text_circuit(circuit))
    assert wire == "q0 ──┤ M→c0 ├──┤ M→c1 ├──"


def test_idle_qubits_take_one_line_each() -> None:
    expected = """
     ╭───╮
q0 ──┤ H ├───●───
     ╰───╯   │
q1 ──────────┼───
             │
q2 ──────────┼───
             │
q3 ──────────⊕───
"""
    assert _art(QuantumCircuit().h(0).cx(0, 3)) == expected


def test_gate_order_is_kept_on_every_wire() -> None:
    circuit = QuantumCircuit().x(0).h(1).y(0).cx(0, 1).z(1).s(0)
    q0, q1 = _wires(draw_text_circuit(circuit))
    assert q0.index("X") < q0.index("Y") < q0.index("●") < q0.index("S")
    assert q1.index("H") < q1.index("⊕") < q1.index("Z")


def test_qubit_labels_are_right_aligned() -> None:
    wires = _wires(draw_text_circuit(QuantumCircuit().h(10)))
    assert [line[:4] for line in wires] == [f"{f'q{i}':>3} " for i in range(11)]
    assert len({len(line) for line in wires}) == 1


def test_no_trailing_whitespace_and_no_empty_line_inside_a_part() -> None:
    circuit = QuantumCircuit().h(0).cx(0, 2).rzz(0.1, 1, 2).h(5).measure_all()
    text = draw_text_circuit(circuit, width=30)
    for part in text.split("\n\n"):
        for line in part.splitlines():
            assert line and line == line.rstrip()


def _random_circuit(rng: random.Random, num_qubits: int, size: int) -> QuantumCircuit:
    """Gates of every drawn kind, measurements, conditions, noise, barriers."""
    circuit = QuantumCircuit()
    measured: list[int] = []
    for _ in range(size):
        a, b, c = rng.sample(range(num_qubits), 3)
        kind = rng.randrange(12)
        if kind == 0:
            circuit.h(a)
        elif kind == 1:
            circuit.u3(rng.random(), math.pi / 2, -math.pi, a)
        elif kind == 2:
            circuit.cx(a, b)
        elif kind == 3:
            circuit.ccx(a, b, c)
        elif kind == 4:
            circuit.swap(a, b).cz(b, c)
        elif kind == 5:
            circuit.crz(rng.random(), a, b)
        elif kind == 6:
            circuit.rzz(rng.random(), a, b)
        elif kind == 7:
            circuit.append(Gate(name="blk", num_qubits=3), (a, b, c))
        elif kind == 8:
            circuit.measure([a], [a])
            measured.append(a)
        elif kind == 9:
            circuit.barrier(a, b) if rng.random() < 0.5 else circuit.barrier()
        elif kind == 10:
            circuit.depolarize(0.1, a)
        elif measured:
            bits = rng.sample(measured, min(len(set(measured)), rng.randint(1, 2)))
            with circuit.if_test(sorted(set(bits)), 1):
                rng.choice([circuit.x, circuit.h])(a)
                circuit.cx(b, c)
                circuit.crz(0.5, a, c)
    return circuit


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_draw_consistently(seed: int) -> None:
    """Whatever the circuit: aligned wires, nothing lost, nothing overlapping."""
    rng = random.Random(seed)
    num_qubits = rng.randint(3, 7)
    circuit = _random_circuit(rng, num_qubits, rng.randint(5, 40))
    whole = draw_text_circuit(circuit, fold=False)
    wires = _wires(whole)
    assert len(wires) == circuit.num_qubits
    assert len({len(wire) for wire in wires}) == 1
    assert max(len(line) for line in whole.splitlines()) == len(wires[0])
    # every operand of every instruction leaves its mark on its wire
    marks = sum(wire.count(mark) for wire in wires for mark in "●⊕╳")
    cards = sum(wire.count("┤") for wire in wires)
    expected_marks = expected_cards = 0
    for instruction in circuit.instructions:
        if instruction.is_barrier:
            continue
        gate = instruction.gate or instruction.conditional_gate
        name = gate.name if gate is not None else ""
        if name in ("cx", "ccx", "swap", "cz"):
            expected_marks += len(instruction.qubits)
        elif name == "crz":
            expected_marks += 1
            expected_cards += 1
        else:
            expected_cards += len(instruction.qubits)
    assert (marks, cards) == (expected_marks, expected_cards)
    symbol_gates = ("cx", "ccx", "swap", "cz")
    cards_drawn = 0
    for instruction in circuit.instructions:
        gate = instruction.gate or instruction.conditional_gate
        if not instruction.is_barrier and (
            gate is None or gate.name not in symbol_gates
        ):
            cards_drawn += 1
    # the outline of every card is closed ("╭" also starts a condition flag)
    assert whole.count("╮") == whole.count("╰") == whole.count("╯") == cards_drawn
    assert whole.count("╭") >= cards_drawn
    for width in (40, 72):
        folded = draw_text_circuit(circuit, width=width)
        parts = folded.split("\n\n")
        for part in parts:
            part_wires = _wires(part)
            assert len(part_wires) == circuit.num_qubits
            assert len({len(wire) for wire in part_wires}) == 1
            assert all(line and line == line.rstrip() for line in part.splitlines())
        if len(parts) > 1:
            assert all(wire.endswith("┄") for wire in _wires(parts[0]))
            assert all(wire[3] == "┄" for wire in _wires(parts[-1]))


# ----------------------------------------------------------------- folding
def _layered_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit()
    for _ in range(3):
        circuit.h(0).cx(0, 1).rz(0.5, 1)
    return circuit


def test_wide_circuit_is_continued_below() -> None:
    expected = """
     ╭───╮          ╭───╮
q0 ──┤ H ├───●──────┤ H ├─────┄
     ╰───╯   │      ╰───╯
             │   ╭─────────╮
q1 ──────────⊕───┤ RZ(0.5) ├──┄
                 ╰─────────╯

              ╭───╮
q0 ┄───●──────┤ H ├─────┄
       │      ╰───╯
       │   ╭─────────╮
q1 ┄───⊕───┤ RZ(0.5) ├──┄
           ╰─────────╯

q0 ┄───●────────────────
       │   ╭─────────╮
q1 ┄───⊕───┤ RZ(0.5) ├──
           ╰─────────╯
"""
    assert _art(_layered_circuit(), fold=True, width=34) == expected


@pytest.mark.parametrize("width", [20, 24, 30, 41, 57, 80])
def test_folded_lines_fit_the_width(width: int) -> None:
    text = draw_text_circuit(_layered_circuit(), width=width)
    assert max(len(line) for line in text.splitlines()) <= width


def test_folding_keeps_every_column() -> None:
    """Joining the parts of each wire gives back the unfolded wire."""
    circuit = _layered_circuit().measure_all()

    def symbols(wire: str) -> str:
        return wire[3:].replace("─", "").replace("┄", "")

    whole = [symbols(line) for line in _wires(draw_text_circuit(circuit, fold=False))]
    joined = ["", ""]
    for part in draw_text_circuit(circuit, width=28).split("\n\n"):
        for qubit, line in enumerate(_wires(part)):
            joined[qubit] += symbols(line)
    assert joined == whole


def test_parts_have_about_the_same_width() -> None:
    """The last part is not a leftover column under full-width parts."""
    circuit = QuantumCircuit()
    for _ in range(11):
        circuit.h(0)
    parts = draw_text_circuit(circuit, width=40).split("\n\n")
    widths = [len(_wires(part)[0]) for part in parts]
    assert len(parts) == 3
    assert max(widths) - min(widths) <= 8


def test_balanced_folding_properties() -> None:
    """As few parts as filling each to the brim, all within the width."""
    rng = random.Random(7)
    for _ in range(2000):
        widths = [
            rng.choice([1, 3, 3, 5, 8, 11, 25]) for _ in range(rng.randint(1, 40))
        ]
        available = rng.randint(5, 120)
        slim = [width == 3 and rng.random() < 0.5 for width in widths]
        starts = _layer_starts(widths, slim)
        greedy = _chunk_layers(starts, widths, available)
        balanced = _balanced_chunks(starts, widths, available)
        assert len(balanced) == len(greedy)
        assert balanced[0][0] == 0 and balanced[-1][1] == len(widths)
        assert all(a[1] == b[0] for a, b in zip(balanced, balanced[1:], strict=False))
        for first, stop in balanced:
            assert first < stop
            assert (
                stop - first == 1
                or _run_width(starts, widths, first, stop) <= available
            )


def test_column_wider_than_the_display_gets_its_own_part() -> None:
    text = draw_text_circuit(_layered_circuit(), width=5)
    parts = text.split("\n\n")
    assert len(parts) == 7  # one per column
    assert "┤ RZ(0.5) ├" in parts[2]


def test_fold_false_ignores_the_width() -> None:
    text = draw_text_circuit(_layered_circuit(), width=10, fold=False)
    assert "\n\n" not in text and "┄" not in text


def test_draw_method_passes_width_and_fold() -> None:
    circuit = _layered_circuit()
    assert circuit.draw(width=30) == draw_text_circuit(circuit, width=30)
    assert circuit.draw(fold=False) == draw_text_circuit(circuit, fold=False)


def test_default_width_is_the_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "30")
    circuit = _layered_circuit()
    assert draw_text_circuit(circuit) == draw_text_circuit(circuit, width=30)


# ----------------------------------------------------------- display width
def _terminal_of(columns: int) -> object:
    def size(descriptor: int = 1) -> os.terminal_size:
        return os.terminal_size((columns, 24))

    return size


@pytest.fixture
def no_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(descriptor: int = 1) -> os.terminal_size:
        raise OSError("not a terminal")

    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.delenv("JUPYTER_COLUMNS", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(os, "get_terminal_size", refuse)
    monkeypatch.setattr(text_drawer, "_in_jupyter", lambda: False)


@pytest.mark.usefixtures("no_terminal")
def test_terminal_width_without_a_terminal() -> None:
    assert terminal_width() == 80
    assert terminal_width(default=132) == 132


@pytest.mark.usefixtures("no_terminal")
def test_terminal_width_asks_stdin_stdout_stderr_in_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[int] = []

    def only_stderr(descriptor: int = 1) -> os.terminal_size:
        asked.append(descriptor)
        if descriptor != 2:
            raise OSError("not a terminal")
        return os.terminal_size((97, 24))

    monkeypatch.setattr(os, "get_terminal_size", only_stderr)
    monkeypatch.setattr(text_drawer.sys, "platform", "linux")
    assert terminal_width() == 97
    assert asked == [0, 1, 2]


@pytest.mark.usefixtures("no_terminal")
def test_columns_variable_overrides_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "get_terminal_size", _terminal_of(97))
    monkeypatch.setenv("COLUMNS", "64")
    assert terminal_width() == 64
    for ignored in ("", "0", "wide", "-3"):
        monkeypatch.setenv("COLUMNS", ignored)
        assert terminal_width() == 97


@pytest.mark.usefixtures("no_terminal")
def test_dumb_terminal_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "get_terminal_size", _terminal_of(97))
    monkeypatch.setenv("TERM", "dumb")
    assert terminal_width() == 80


@pytest.mark.usefixtures("no_terminal")
def test_jupyter_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_drawer, "_in_jupyter", lambda: True)
    assert terminal_width() == 115
    monkeypatch.setenv("JUPYTER_COLUMNS", "140")
    assert terminal_width() == 140


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.0, "0"),
        (math.pi, "π"),
        (-math.pi, "-π"),
        (2 * math.pi, "2π"),
        (math.pi / 2, "π/2"),
        (-3 * math.pi / 4, "-3π/4"),
        (math.pi / 3, "π/3"),
        (5 * math.pi / 6, "5π/6"),
        (math.pi / 64, "π/64"),
        (0.3, "0.3"),
        (1.0, "1"),
        (0.123456789, "0.1235"),
        (3.14, "3.14"),
        (-2.5, "-2.5"),
        (1e-7, "1e-07"),
    ],
)
def test_angles_are_shown_as_multiples_of_pi(value: float, text: str) -> None:
    assert _format_number(value) == text


def test_parameter_names_and_expressions_are_shown() -> None:
    theta = Parameter("theta")
    text = draw_text_circuit(QuantumCircuit().rx(theta, 0).ry(2 * theta + 1, 0))
    assert "┤ RX(theta) ├" in text
    assert "┤ RY(2*theta + 1) ├" in text


def test_standard_gate_labels() -> None:
    circuit = QuantumCircuit().sdg(0).tdg(0).sx(0).sxdg(0).id(0)
    (wire,) = _wires(draw_text_circuit(circuit, fold=False))
    assert wire == "q0 ──┤ S† ├──┤ T† ├──┤ √X ├──┤ √X† ├──┤ I ├──"
