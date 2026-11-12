"""Barriers and noise channels through ``transpile()``.

Neither is a gate: a barrier asks that nothing is optimized across it and a
noise channel is not unitary, so both stay where they are, on the wires that
hold their qubits, and no gate is merged, cancelled or moved across them.
The equivalence checks evolve a density matrix with numpy (gates from
``gate_matrix``, channels from their Kraus operators, a measurement as full
dephasing), independently of the simulators.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction
from quspect.gates._matrix_utils import gate_matrix
from quspect.noise.channel import DepolarizingChannel, KrausChannel
from quspect.transpiler import SABRE, CompilationTarget, TranspileResult, transpile

BASES: dict[str, set[str]] = {
    "cz_sx": {"cz", "rz", "sx", "x"},
    "ibm": {"cx", "rz", "sx", "x"},
    "cz_h": {"cz", "h", "rz", "x"},
}
LEVELS = [0, 1, 2, 3]


def _line(n: int, basis: str = "cz_sx") -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASES[basis])
    )


def _grid_2x3(basis: str) -> CompilationTarget:
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    return CompilationTarget.from_coupling_map(6, edges, native_gates=set(BASES[basis]))


def _kind(ins: CircuitInstruction) -> str:
    if ins.is_barrier:
        return "barrier"
    if ins.is_channel:
        return "channel"
    if ins.is_measure:
        return "measure"
    assert ins.gate is not None
    return ins.gate.name


def _kinds(circuit: QuantumCircuit) -> list[str]:
    return [_kind(ins) for ins in circuit]


# ---------------------------------------------------------------------------
# numpy reference: density matrices
# ---------------------------------------------------------------------------


def _apply(
    rho: np.ndarray, op: np.ndarray, qubits: tuple[int, ...], n: int
) -> np.ndarray:
    """``op · rho · op†`` on ``qubits`` of an ``n``-qubit density tensor."""
    k = len(qubits)
    g = np.asarray(op, dtype=complex).reshape([2] * (2 * k))
    rows = list(qubits)
    cols = [n + q for q in qubits]
    rho = np.tensordot(g, rho, axes=(list(range(k, 2 * k)), rows))
    rho = np.moveaxis(rho, list(range(k)), rows)
    rho = np.tensordot(g.conj(), rho, axes=(list(range(k, 2 * k)), cols))
    return np.moveaxis(rho, list(range(k)), cols)


_P0 = np.diag([1.0, 0.0])
_P1 = np.diag([0.0, 1.0])


def _evolve(circuit: QuantumCircuit, rho: np.ndarray, n: int) -> np.ndarray:
    for ins in circuit:
        if ins.is_barrier:
            continue
        if ins.is_measure:
            for q in ins.qubits:
                rho = _apply(rho, _P0, (q,), n) + _apply(rho, _P1, (q,), n)
        elif ins.is_channel:
            assert ins.channel is not None
            rho = sum(
                _apply(rho, k, ins.qubits, n) for k in ins.channel.kraus_operators()
            )  # type: ignore[assignment]
        else:
            assert ins.gate is not None
            rho = _apply(rho, gate_matrix(ins.gate), ins.qubits, n)
    return rho


def _embed(rho: np.ndarray, layout: list[int], num_physical: int) -> np.ndarray:
    """Logical qubit ``q`` on physical ``layout[q]``, the others in ``|0>``."""
    n = len(layout)
    rest = num_physical - n
    free = [p for p in range(num_physical) if p not in layout]
    zero = np.zeros([2] * (2 * rest), dtype=complex)
    zero[(0,) * (2 * rest)] = 1.0
    full = np.tensordot(rho, zero, axes=0)  # rows, cols | free rows, free cols
    row = [
        layout.index(p) if p in layout else 2 * n + free.index(p)
        for p in range(num_physical)
    ]
    col = [
        n + layout.index(p) if p in layout else 2 * n + rest + free.index(p)
        for p in range(num_physical)
    ]
    return np.transpose(full, row + col)


def _assert_same_channel(
    circuit: QuantumCircuit, result: TranspileResult, target: CompilationTarget
) -> None:
    """The compiled circuit acts on every input state like the original."""
    n, big = circuit.num_qubits, target.num_qubits
    for ins in result.circuit:
        if ins.gate is None:
            continue
        assert ins.gate.name in target.native_gates, ins.gate.name
        if len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in target.coupling_graph
    rng = np.random.default_rng(5)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj()).reshape([2] * (2 * n))
    expected = _embed(_evolve(circuit, rho, n), list(result.final_layout), big)
    got = _evolve(result.circuit, _embed(rho, list(result.initial_layout), big), big)
    assert np.max(np.abs(expected - got)) < 1e-7


def _two_qubit_damping(gamma_a: float, gamma_b: float) -> KrausChannel:
    """Amplitude damping on each qubit with different rates (not symmetric)."""

    def kraus(gamma: float) -> list[np.ndarray]:
        return [
            np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex),
        ]

    return KrausChannel(
        name="damp2",
        num_qubits=2,
        kraus_ops=[np.kron(a, b) for a in kraus(gamma_a) for b in kraus(gamma_b)],
    )


def _tilt(p: float, angle: float) -> KrausChannel:
    """With probability ``p`` an ``RY(angle)``: no gate commutes with it, so a
    rotation moved across it or dropped before a measurement shows."""
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return KrausChannel(
        name="tilt",
        num_qubits=1,
        kraus_ops=[
            math.sqrt(1 - p) * np.eye(2, dtype=complex),
            math.sqrt(p) * np.array([[c, -s], [s, c]], dtype=complex),
        ],
    )


def _noisy_circuit(n: int, depth: int, seed: int, *, measure: bool) -> QuantumCircuit:
    """Random gates with barriers, 1q / 2q noise channels and measurements."""
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    clbit = 0
    for _ in range(depth):
        a, b, c = rng.sample(range(n), 3)
        r = rng.randrange(17)
        if r == 0:
            qc.cx(a, b)
        elif r == 1:
            qc.cz(a, b)
        elif r == 2:
            qc.cp(rng.uniform(-3, 3), a, b)
        elif r == 3:
            qc.swap(a, b)
        elif r == 4:
            qc.ccx(a, b, c)
        elif r == 5:
            qc.t(a).h(b)
        elif r == 6:
            qc.rx(rng.uniform(-3, 3), a).rz(rng.uniform(-3, 3), b)
        elif r == 7:
            qc.x(a)
        elif r == 8:
            qc.depolarize(rng.uniform(0.01, 0.3), a)
        elif r == 9:
            qc.amplitude_damp(rng.uniform(0.05, 0.5), a)
        elif r == 10:
            qc.append_channel(
                DepolarizingChannel(
                    name="depolarizing", num_qubits=2, p=rng.uniform(0.01, 0.3)
                ),
                [a, b],
            )
        elif r == 11:
            qc.append_channel(_two_qubit_damping(0.1, 0.45), [a, b])
        elif r == 12:
            qc.barrier(a)
        elif r == 13:
            qc.barrier(a, b, c)
        elif r == 14:
            qc.barrier()
        elif r == 15:
            qc.append_channel(_tilt(rng.uniform(0.1, 0.6), rng.uniform(0.3, 2.5)), [a])
        elif measure:
            qc.measure([a], [clbit])
            clbit += 1
        else:
            qc.phase_damp(rng.uniform(0.05, 0.5), a)
    return qc


# ---------------------------------------------------------------------------
# Barriers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_barrier_is_kept_and_nothing_merges_across_it(level: int) -> None:
    target = _line(2)
    result = transpile(
        QuantumCircuit().x(0).barrier(0).x(0), target, optimization_level=level
    )
    assert _kinds(result.circuit) == ["x", "barrier", "x"]
    # without the barrier the two gates cancel from level 1 on
    plain = transpile(QuantumCircuit().x(0).x(0), target, optimization_level=level)
    assert len(plain.circuit) == (2 if level == 0 else 0)


@pytest.mark.parametrize("level", LEVELS)
def test_two_qubit_gates_do_not_cancel_across_a_barrier(level: int) -> None:
    target = _line(2)
    circuit = QuantumCircuit().cz(0, 1).barrier(0, 1).cz(0, 1)
    result = transpile(circuit, target, optimization_level=level)
    assert _kinds(result.circuit) == ["cz", "barrier", "cz"]
    assert result.stats.two_qubit_gates == 2
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_barrier_only_fences_its_own_qubits(level: int) -> None:
    result = transpile(
        QuantumCircuit().x(0).barrier(1).x(0), _line(2), optimization_level=level
    )
    assert _kinds(result.circuit) == ["barrier"]


@pytest.mark.parametrize("level", LEVELS)
def test_barrier_follows_its_qubits_onto_the_device(level: int) -> None:
    target = _line(3)
    circuit = QuantumCircuit().h(0).cx(0, 1).barrier(2, 0).cx(0, 1)
    result = transpile(
        circuit, target, optimization_level=level, initial_layout=[1, 0, 2]
    )
    barriers = [ins for ins in result.circuit if ins.is_barrier]
    assert [ins.qubits for ins in barriers] == [(2, 1)]
    assert result.stats.two_qubit_gates == 2
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("level", LEVELS)
def test_barrier_before_final_measurements_is_kept(level: int) -> None:
    target = _line(4)
    circuit = QuantumCircuit().h(0).cx(0, 3).cx(1, 2).cx(0, 2)
    circuit.barrier().measure([0, 1, 2, 3], [0, 1, 2, 3])
    result = transpile(circuit, target, optimization_level=level, seed=1)
    kinds = _kinds(result.circuit)
    assert kinds.count("barrier") == 1 and kinds.count("measure") == 4
    assert kinds.index("barrier") < kinds.index("measure")
    assert all(k == "measure" for k in kinds[kinds.index("barrier") + 1 :])
    barrier = next(ins for ins in result.circuit if ins.is_barrier)
    assert sorted(barrier.qubits) == sorted(result.final_layout)
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("level", LEVELS)
def test_barrier_without_qubits_is_kept(level: int) -> None:
    """``barrier()`` on an empty circuit has no qubits; it stays in front."""
    circuit = QuantumCircuit().barrier().h(0).cx(0, 1)
    assert circuit[0].qubits == ()
    result = transpile(circuit, _line(2), optimization_level=level)
    assert result.circuit[0].is_barrier and result.circuit[0].qubits == ()
    assert _kinds(result.circuit).count("barrier") == 1
    _assert_same_channel(circuit, result, _line(2))


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_noise_channel_is_kept(level: int) -> None:
    """``h; depolarize; cx`` used to come back without the channel."""
    target = _line(2)
    circuit = QuantumCircuit().h(0).depolarize(0.05, 0).cx(0, 1)
    result = transpile(circuit, target, optimization_level=level)
    channels = [ins for ins in result.circuit if ins.is_channel]
    assert len(channels) == 1
    assert channels[0].channel is circuit[1].channel
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("level", LEVELS)
def test_gates_do_not_cancel_across_a_channel(level: int) -> None:
    target = _line(2)
    circuit = QuantumCircuit().x(0).amplitude_damp(0.3, 0).x(0)
    result = transpile(circuit, target, optimization_level=level)
    assert _kinds(result.circuit) == ["x", "channel", "x"]
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("level", LEVELS)
def test_rotation_before_a_channel_and_a_measurement_is_kept(level: int) -> None:
    """A diagonal gate right before a measurement can go; not with a channel
    in between that turns its phase into a population."""
    target = _line(2)
    circuit = QuantumCircuit().h(0).rz(0.7, 0)
    circuit.append_channel(_tilt(0.4, 1.1), [0])
    circuit.measure([0], [0])
    result = transpile(circuit, target, optimization_level=level)
    kinds = _kinds(result.circuit)
    assert kinds[-2:] == ["channel", "measure"]
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("level", LEVELS)
def test_two_qubit_channel_keeps_its_qubit_order(level: int) -> None:
    """A channel on qubits that are not coupled needs no SWAP of its own."""
    target = _line(3)
    circuit = QuantumCircuit().h(0).h(2)
    circuit.append_channel(_two_qubit_damping(0.1, 0.6), [2, 0])
    circuit.cx(0, 1).cx(1, 2)
    result = transpile(circuit, target, optimization_level=level, seed=0)
    assert sum(ins.is_channel for ins in result.circuit) == 1
    _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("basis", sorted(BASES))
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", range(4))
def test_noisy_circuits_compile_to_the_same_channel(
    basis: str, level: int, seed: int
) -> None:
    circuit = _noisy_circuit(4, 24, seed, measure=seed % 2 == 0)
    for target in (_line(4, basis), _grid_2x3(basis)):
        result = transpile(circuit, target, optimization_level=level, seed=seed)
        for kind in ("barrier", "channel", "measure"):
            assert _kinds(result.circuit).count(kind) == _kinds(circuit).count(kind)
        _assert_same_channel(circuit, result, target)


@pytest.mark.parametrize("seed", range(3))
def test_gate_only_output_is_unchanged_by_a_trailing_barrier(seed: int) -> None:
    """Fences order the routing; before the first one nothing changes."""
    circuit = _noisy_circuit(4, 0, seed, measure=False).cx(0, 3).cx(1, 3).t(3)
    target = _line(4)
    plain = transpile(circuit, target, seed=seed)
    fenced = transpile(circuit.copy().barrier(), target, seed=seed)
    assert _kinds(fenced.circuit)[-1] == "barrier"
    assert fenced.circuit.instructions[:-1] == plain.circuit.instructions
    assert fenced.final_layout == plain.final_layout


# ---------------------------------------------------------------------------
# SABRE on its own
# ---------------------------------------------------------------------------


def test_sabre_routes_barriers_and_channels_as_fences() -> None:
    target = _line(4)
    circuit = QuantumCircuit().cx(0, 3).barrier(0, 3).depolarize(0.1, 3).cx(0, 3)
    routed, wires, final = SABRE(target).solve(circuit, seed=0)
    kinds = _kinds(routed)
    assert kinds.count("barrier") == 1 and kinds.count("channel") == 1
    # both fences sit between the two cx gates, on the wires of logical 0 and 3
    first, second = (i for i, k in enumerate(kinds) if k == "cx")
    assert first < kinds.index("barrier") < kinds.index("channel") < second
    barrier = routed[kinds.index("barrier")]
    assert set(barrier.qubits) == set(routed[first].qubits)
