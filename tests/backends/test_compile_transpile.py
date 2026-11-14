"""Shared device compilation through ``transpile()``.

:func:`compile_for_device` hands a circuit to ``transpile()`` when it can and
keeps the earlier pipeline for everything else.  Offline; compiled circuits
are checked exactly against the simulator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import pytest

from quspect import QuantumCircuit, Simulator
from quspect.backends import _compile
from quspect.backends._compile import (
    _compile_with_passes,
    _compile_with_transpile,
    _placement_problem,
    _without_diagonals_before_measurement,
    compile_for_device,
)
from quspect.backends.originq._backend import QPU_COMPILE_BASIS
from quspect.backends.tianyan._translator import QCIS_BASIS
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler._passes import DecomposePass
from tests.backends._measurement_cases import (
    measurements_are_last,
    random_measured_circuits,
    used_after_measurement,
)
from tests.backends._originq_fakes import (
    assert_same_distribution,
    quspect_exact_distribution,
)

LINE8 = [(i, i + 1) for i in range(7)]
GRID = [(i, i + 1) for i in range(16) if i % 4 < 3] + [(i, i + 4) for i in range(12)]
CZ_BASIS = {"cz", "h", "rx", "ry", "rz"}
#: The gate set each cloud backend compiles to (LogicalQubit devices report
#: none, or one that is not a basis).
BACKEND_BASES = {
    "quafu": CZ_BASIS,
    "originq": set(QPU_COMPILE_BASIS),
    "tianyan": set(QCIS_BASIS),
    "undeclared": None,
}


def _target(num_qubits: int, edges: list, native: set[str] | None = None):
    return CompilationTarget.from_coupling_map(num_qubits, edges, native_gates=native)


def _gates(circuit: QuantumCircuit) -> list:
    return [ins for ins in circuit.instructions if ins.is_gate]


def _two_qubit_gates(circuit: QuantumCircuit, target: CompilationTarget) -> int:
    """Two-qubit gates, with any ``swap`` lowered to the target's gates."""
    lowered = DecomposePass(target).run(circuit)
    return sum(1 for ins in _gates(lowered) if len(ins.qubits) == 2)


def _assert_runs_on(
    routed: QuantumCircuit, layout: list[int] | None, target: CompilationTarget
) -> None:
    assert layout is not None
    assert len(layout) == len(set(layout)) == routed.num_qubits
    for ins in _gates(routed):
        if target.native_gates:
            assert ins.gate.name in target.native_gates
        assert len(ins.qubits) <= 2
        if len(ins.qubits) == 2:
            a, b = (layout[wire] for wire in ins.qubits)
            assert (min(a, b), max(a, b)) in target.coupling_graph


def _same_as_passes(target: CompilationTarget, circuit: QuantumCircuit, **options):
    """Assert that the seam leaves this circuit to the earlier pipeline."""
    assert _compile_with_transpile(target, circuit, **_options(options)) is None
    routed, layout = compile_for_device(target, circuit, **options)
    expected, expected_layout = _compile_with_passes(target, circuit, **options)
    assert routed.instructions == expected.instructions
    assert layout == expected_layout
    return routed, layout


def _read_from(routed: QuantumCircuit, layout: list[int] | None) -> dict[int, int]:
    """The physical qubit each classical bit is read from."""
    assert layout is not None
    return {
        clbit: layout[wire]
        for ins in routed.instructions
        if ins.is_measure
        for wire, clbit in zip(ins.qubits, ins.clbits, strict=True)
    }


def _options(options: dict) -> dict:
    return {"initial_layout": None, "physical_qubits": None, **options}


def _qft(n: int) -> QuantumCircuit:
    circuit = QuantumCircuit()
    for j in range(n):
        circuit.h(j)
        for k in range(j + 1, n):
            circuit.cp(math.pi / 2 ** (k - j), k, j)
    return circuit.measure(list(range(n)), list(range(n)))


def _star(n: int) -> QuantumCircuit:
    circuit = QuantumCircuit().h(0)
    for k in range(1, n):
        circuit.cx(0, k)
    return circuit.measure(list(range(n)), list(range(n)))


# ---------------------------------------------------------------------------
# What the backends need from the compiled circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["measure_all", "subset per qubit"])
@pytest.mark.parametrize("backend", sorted(BACKEND_BASES))
@pytest.mark.parametrize(
    ("num_qubits", "edges"), [(8, LINE8), (16, GRID)], ids=["line", "grid"]
)
def test_compiled_circuits_run_on_the_device_and_give_the_same_results(
    num_qubits: int, edges: list, backend: str, spelling: str
) -> None:
    target = _target(num_qubits, edges, BACKEND_BASES[backend])
    for circuit in random_measured_circuits(spelling, count=25):
        routed, layout = compile_for_device(target, circuit)

        _assert_runs_on(routed, layout, target)
        # Terminal measurements are one run behind every gate.
        assert measurements_are_last(routed)
        assert used_after_measurement(routed) == []
        assert sorted(c for ins in routed.instructions for c in ins.clbits) == sorted(
            c for ins in circuit.instructions for c in ins.clbits
        )
        assert_same_distribution(
            quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
        )


def test_the_gates_are_the_ones_transpile_returns() -> None:
    """Hardware runs get the compiler that ``transpile()`` is in user code."""
    target = _target(8, LINE8, CZ_BASIS)
    circuit = _qft(5)
    body = QuantumCircuit._from_instruction_snapshot(_gates(circuit))

    routed, layout = compile_for_device(target, circuit)
    result = transpile(
        body,
        target,
        optimization_level=_compile.DEVICE_OPTIMIZATION_LEVEL,
        seed=_compile.DEVICE_SEED,
    )

    measured = {
        wire for ins in routed.instructions for wire in ins.qubits if ins.clbits
    }
    assert _gates(routed) == _without_diagonals_before_measurement(
        result.wire_circuit.instructions, measured
    )
    assert layout == result.wire_layout[: routed.num_qubits]
    # Each classical bit reads the qubit that holds its logical qubit at the end.
    (measurement,) = [ins for ins in routed.instructions if ins.is_measure]
    assert [layout[wire] for wire in measurement.qubits] == result.final_layout
    assert measurement.clbits == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    ("circuit", "num_qubits", "edges"),
    [(_qft(5), 8, LINE8), (_qft(8), 12, None), (_star(6), 8, LINE8)],
    ids=["qft5-line8", "qft8-grid3x4", "star6-line8"],
)
def test_fewer_two_qubit_gates_than_the_earlier_pipeline(
    circuit: QuantumCircuit, num_qubits: int, edges: list | None
) -> None:
    if edges is None:  # 3 x 4 grid
        edges = [(i, i + 1) for i in range(12) if i % 4 < 3]
        edges += [(i, i + 4) for i in range(8)]
    target = _target(num_qubits, edges, {"cz", "rz", "sx", "x", "h", "rx", "ry"})

    routed, layout = compile_for_device(target, circuit)
    earlier, _ = _compile_with_passes(target, circuit)

    _assert_runs_on(routed, layout, target)
    assert _two_qubit_gates(routed, target) < _two_qubit_gates(earlier, target)
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


def test_compilation_is_deterministic() -> None:
    target = _target(16, GRID, CZ_BASIS)
    for circuit in random_measured_circuits("measure_all", count=10):
        first, first_layout = compile_for_device(target, circuit)
        second, second_layout = compile_for_device(target, circuit)
        assert first.instructions == second.instructions
        assert first_layout == second_layout


def test_qubit_pool_and_initial_layout_are_honoured() -> None:
    target = _target(16, GRID, CZ_BASIS)
    pool = [5, 6, 9, 10, 13, 14]  # a 3 x 2 block of the grid
    circuit = _star(4)

    routed, layout = compile_for_device(
        target, circuit, physical_qubits=pool, initial_layout=[9, 5, 10, 13]
    )

    _assert_runs_on(routed, layout, target)
    assert layout is not None
    assert layout[:4] == [9, 5, 10, 13]
    assert set(layout) <= set(pool)
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


def test_three_qubit_gate_listed_as_native_is_still_lowered() -> None:
    target = _target(5, LINE8[:4], {"ccx", "cz", "h", "rx", "rz"})
    circuit = QuantumCircuit().h(0).h(1).ccx(0, 1, 2).cx(2, 3).measure_all()

    routed, layout = compile_for_device(target, circuit)

    _assert_runs_on(routed, layout, target)
    assert "ccx" not in {ins.gate.name for ins in _gates(routed)}
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


# ---------------------------------------------------------------------------
# Logical qubits that are only measured
# ---------------------------------------------------------------------------


def test_measured_qubit_without_gates_reads_its_requested_qubit() -> None:
    target = _target(5, LINE8[:4], CZ_BASIS)
    circuit = QuantumCircuit().x(0).cx(0, 1).measure([0, 1, 2, 3], [0, 1, 2, 3])

    routed, layout = compile_for_device(target, circuit, initial_layout=[4, 3, 1, 0])

    _assert_runs_on(routed, layout, target)
    read_from = _read_from(routed, layout)
    assert read_from[2] == 1 and read_from[3] == 0
    assert {read_from[0], read_from[1]} == {4, 3}
    assert quspect_exact_distribution(circuit) == pytest.approx({"1100": 1.0})
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


def test_measured_qubit_without_gates_between_busy_ones_follows_the_routing() -> None:
    target = _target(3, LINE8[:2], CZ_BASIS)
    circuit = QuantumCircuit().x(0).cx(0, 2).measure([0, 1, 2], [0, 1, 2])

    # Logical 1 has no gate and sits between the two that must meet.
    routed, layout = compile_for_device(target, circuit, initial_layout=[0, 1, 2])

    _assert_runs_on(routed, layout, target)
    assert sorted(_read_from(routed, layout).values()) == [0, 1, 2]
    assert quspect_exact_distribution(circuit) == pytest.approx({"101": 1.0})
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


def test_measured_qubit_without_gates_avoids_a_qubit_that_ends_occupied() -> None:
    """Its requested qubit may hold another logical qubit after routing."""
    target = _target(3, LINE8[:2], CZ_BASIS)
    circuit = QuantumCircuit().x(0).cx(0, 1).measure([0, 1, 2], [0, 1, 2])

    # Logical 0 and 1 start on the two ends of the line; logical 2 (measured
    # only) asks for the middle qubit, which the routing moves a qubit onto.
    routed, layout = compile_for_device(target, circuit, initial_layout=[0, 2, 1])

    _assert_runs_on(routed, layout, target)
    read_from = _read_from(routed, layout)
    assert sorted(read_from.values()) == [0, 1, 2]
    assert 1 in (read_from[0], read_from[1])
    assert quspect_exact_distribution(circuit) == pytest.approx({"110": 1.0})
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


@pytest.mark.parametrize("pool", [None, [2, 3, 4, 5]], ids=["device", "exact pool"])
def test_measured_qubits_without_gates_get_qubits_of_their_own(pool) -> None:
    target = _target(8, LINE8, CZ_BASIS)
    circuit = QuantumCircuit().x(1).cx(1, 0).measure([0, 1, 2, 3], [3, 2, 1, 0])

    routed, layout = compile_for_device(target, circuit, physical_qubits=pool)

    _assert_runs_on(routed, layout, target)
    assert layout is not None
    measured = [
        layout[w] for ins in routed.instructions if ins.clbits for w in ins.qubits
    ]
    assert len(set(measured)) == 4
    if pool is not None:
        assert set(layout) <= set(pool)
    assert quspect_exact_distribution(circuit) == pytest.approx({"0011": 1.0})
    assert_same_distribution(
        quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
    )


# ---------------------------------------------------------------------------
# What transpile() cannot take keeps the earlier pipeline
# ---------------------------------------------------------------------------


def test_mid_circuit_measurement_keeps_the_earlier_pipeline() -> None:
    target = _target(5, LINE8[:4], CZ_BASIS)
    circuit = QuantumCircuit().x(0).measure([0], [0]).x(0).cx(0, 2)
    circuit.measure([0, 2], [1, 2])

    _same_as_passes(target, circuit)


def test_barrier_between_gates_stays_a_fence() -> None:
    """``transpile()`` optimizes across barriers; a submission must not."""
    target = _target(5, LINE8[:4], CZ_BASIS)
    folded = QuantumCircuit().h(0).cz(0, 1).barrier().cz(0, 1).barrier().cz(0, 1)
    folded.measure_all()

    routed, _ = _same_as_passes(target, folded)

    assert sum(1 for ins in _gates(routed) if ins.gate.name == "cz") == 3

    # ... while the same gates without barriers cancel,
    plain = QuantumCircuit().h(0).cz(0, 1).cz(0, 1).cz(0, 1).measure_all()
    compiled, _ = compile_for_device(target, plain)
    assert sum(1 for ins in _gates(compiled) if ins.gate.name == "cz") == 1
    # and a barrier behind the last gate orders nothing.
    trailing = QuantumCircuit().h(0).cz(0, 1).barrier().measure_all()
    assert _compile_with_transpile(target, trailing, **_options({})) is not None


def test_gate_set_that_is_not_a_basis_keeps_the_earlier_pipeline() -> None:
    circuit = QuantumCircuit().h(0).cx(0, 2).cx(1, 2).measure_all()

    _same_as_passes(_target(5, LINE8[:4], {"cz", "h", "t", "tdg"}), circuit)

    # LogicalQubit devices may report just ["CZ"]: the earlier pipeline cannot
    # lower to that either, and its error is the one the caller sees.
    for native in ({"cz"}, {"cz", "rx"}):
        with pytest.raises(ValueError, match="Cannot express gate"):
            compile_for_device(_target(5, LINE8[:4], native), circuit)


def test_noise_channel_keeps_the_earlier_pipeline() -> None:
    target = _target(5, LINE8[:4], CZ_BASIS)
    circuit = QuantumCircuit().h(0).depolarize(0.1, 0).cx(0, 2).measure_all()

    assert _compile_with_transpile(target, circuit, **_options({})) is None


def test_qubits_without_couplings_keep_the_earlier_pipeline() -> None:
    target = CompilationTarget(num_qubits=3, native_gates=set(CZ_BASIS))
    circuit = QuantumCircuit().h(0).cx(0, 1).measure_all()

    routed, layout = _same_as_passes(target, circuit)

    assert layout is None
    assert measurements_are_last(routed)


def test_long_circuit_keeps_the_earlier_pipeline(monkeypatch) -> None:
    target = _target(5, LINE8[:4], CZ_BASIS)
    circuit = QuantumCircuit().h(0).cx(0, 2).cx(1, 2).cx(0, 1).measure_all()
    toffoli = QuantumCircuit().ccx(0, 1, 2).measure_all()

    monkeypatch.setattr(_compile, "DEVICE_TRANSPILE_MAX_TWO_QUBIT_GATES", 3)
    assert _compile_with_transpile(target, circuit, **_options({})) is not None
    assert _compile_with_transpile(target, toffoli, **_options({})) is None

    monkeypatch.setattr(_compile, "DEVICE_TRANSPILE_MAX_TWO_QUBIT_GATES", 2)
    _same_as_passes(target, circuit)


def test_invalid_arguments_raise_the_earlier_errors() -> None:
    target = _target(8, LINE8, CZ_BASIS)
    circuit = QuantumCircuit().h(0).cx(0, 1).measure_all()

    with pytest.raises(ValueError, match="connected coupling subgraph"):
        compile_for_device(target, circuit, physical_qubits=[0, 1, 5])
    with pytest.raises(ValueError, match="outside physical_qubits"):
        compile_for_device(
            target, circuit, physical_qubits=[0, 1, 2], initial_layout=[0, 4]
        )
    with pytest.raises(ValueError, match="initial_mapping length"):
        compile_for_device(target, circuit, initial_layout=[0])
    with pytest.raises(ValueError, match="usable two-qubit coupling edge"):
        compile_for_device(target, circuit, physical_qubits=[3])
    with pytest.raises(ValueError, match="provides only 2 physical qubits"):
        compile_for_device(target, _star(3), physical_qubits=[3, 4])


#: Declared gate sets: the backends' bases, and the set a LogicalQubit chip
#: declaring lqcloud's ``xhalf, rz, cz`` compiles to.
DECLARED_BASES = {
    **{name: basis for name, basis in BACKEND_BASES.items() if basis},
    "logicalqubit": {"sx", "rz", "cz"},
}


def _triangle(fence: str) -> QuantumCircuit:
    """Two-qubit gates on a triangle, which a line holds only with SWAPs.

    The ``fence`` (a barrier between gates, or a mid-circuit measurement)
    keeps the circuit from ``transpile()``.
    """
    circuit = QuantumCircuit().x(0).h(1).ry(0.7, 2)
    circuit.cx(0, 1).cx(1, 2)
    if fence == "barrier":
        circuit.barrier()
    else:
        circuit.measure([0], [3])
    circuit.cx(0, 2).t(2).cx(2, 1).rx(0.4, 0)
    return circuit.measure([0, 1, 2], [0, 1, 2])


@pytest.mark.parametrize("fence", ["barrier", "mid-circuit measurement"])
@pytest.mark.parametrize("backend", sorted(DECLARED_BASES))
def test_the_earlier_pipeline_lowers_its_routing_swaps(
    backend: str, fence: str
) -> None:
    """Known issue: routing SWAPs were returned as ``swap``, outside the set."""
    circuit = _triangle(fence)
    undeclared, _ = _compile_with_passes(_target(5, LINE8[:4]), circuit)
    assert any(ins.gate.name == "swap" for ins in _gates(undeclared))
    target = _target(5, LINE8[:4], DECLARED_BASES[backend])

    routed, layout = _same_as_passes(target, circuit)

    _assert_runs_on(routed, layout, target)
    if fence == "barrier":
        assert_same_distribution(
            quspect_exact_distribution(routed), quspect_exact_distribution(circuit)
        )
    else:
        assert used_after_measurement(routed) != []
        expected = Simulator(seed=11).run(circuit, shots=64).counts
        assert Simulator(seed=11).run(routed, shots=64).counts == expected


def test_routing_swaps_the_declared_gates_cannot_express_are_refused() -> None:
    """A set without the gates a SWAP needs gets an error, not a ``swap``."""
    circuit = QuantumCircuit().cz(0, 1).cz(1, 2).barrier().cz(0, 2).measure_all()

    with pytest.raises(ValueError, match="routing SWAPs cannot be expressed"):
        compile_for_device(_target(5, LINE8[:4], {"cz"}), circuit)


# ---------------------------------------------------------------------------
# The check between transpile() and the device
# ---------------------------------------------------------------------------


def test_placement_problem_names_what_the_device_cannot_run() -> None:
    target = _target(4, LINE8[:3], CZ_BASIS)
    native = set(CZ_BASIS)
    good = QuantumCircuit().h(0).cz(0, 1)

    assert _placement_problem(good, [2, 3], [2, 3], 2, target, native) is None
    assert "uncoupled" in _placement_problem(good, [0, 2], [0, 2], 2, target, native)
    assert "final layout" in _placement_problem(good, [2, 3], [2, 2], 2, target, native)
    assert "final layout" in _placement_problem(good, [2, 3], [2], 2, target, native)
    assert "wire layout" in _placement_problem(good, [2], [2, 3], 2, target, native)
    assert "leaves" in _placement_problem(good, [3, 4], [3, 4], 2, target, native)
    foreign = QuantumCircuit().h(0).cx(0, 1)
    assert "not native" in _placement_problem(
        foreign, [0, 1], [0, 1], 2, target, native
    )
    wide = QuantumCircuit().ccx(0, 1, 2)
    assert "3 qubits" in _placement_problem(
        wide, [0, 1, 2], [0, 1, 2], 3, target, set()
    )


def test_a_circuit_the_device_cannot_run_is_never_returned(monkeypatch, caplog) -> None:
    """A wrong ``transpile()`` result is caught, not submitted."""
    from quspect.transpiler import _transpile as transpile_module

    real = transpile_module.transpile

    def off_the_coupling_graph(circuit, target, **options):
        result = real(circuit, target, **options)
        broken = result.wire_circuit.copy().cz(0, result.wire_circuit.num_qubits + 1)
        layout = [*result.wire_layout]
        layout += [q for q in range(target.num_qubits) if q not in layout]
        return replace(result, wire_circuit=broken, wire_layout=layout)

    monkeypatch.setattr(transpile_module, "transpile", off_the_coupling_graph)
    target = _target(8, LINE8, CZ_BASIS)
    circuit = _star(4)

    with caplog.at_level(logging.WARNING, logger=_compile.__name__):
        routed, layout = compile_for_device(target, circuit)

    assert "cannot run" in caplog.text
    expected, expected_layout = _compile_with_passes(target, circuit)
    assert routed.instructions == expected.instructions
    assert layout == expected_layout


def test_a_failure_inside_transpile_does_not_fail_the_submission(
    monkeypatch, caplog
) -> None:
    from quspect.transpiler import _transpile as transpile_module

    def broken(circuit, target, **options):
        raise IndexError("list index out of range")

    monkeypatch.setattr(transpile_module, "transpile", broken)
    target = _target(8, LINE8, CZ_BASIS)
    circuit = _star(4)

    with caplog.at_level(logging.WARNING, logger=_compile.__name__):
        routed, layout = compile_for_device(target, circuit)

    assert "transpile() failed" in caplog.text
    expected, expected_layout = _compile_with_passes(target, circuit)
    assert routed.instructions == expected.instructions
    assert layout == expected_layout


def test_diagonal_gates_in_front_of_a_measurement_are_dropped() -> None:
    gates = QuantumCircuit().h(0).rz(0.3, 0).cz(0, 1).t(0).rz(0.2, 1).z(1).h(2).s(2)
    gates.rz(0.5, 3).h(3)

    kept = _without_diagonals_before_measurement(gates.instructions, {0, 1, 3})

    assert [(ins.gate.name, ins.qubits) for ins in kept] == [
        ("h", (0,)),
        ("rz", (0,)),  # a gate follows it on the wire
        ("cz", (0, 1)),
        ("h", (2,)),
        ("s", (2,)),  # wire 2 is not measured
        ("rz", (3,)),
        ("h", (3,)),
    ]
