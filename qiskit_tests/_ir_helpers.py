"""Helpers shared by the IR / visualization cross-validation tests.

These complement :mod:`qiskit_tests.helpers` with utilities specific to the
OpenQASM 2.0 import/export, QSAL and drawer tests:

* :func:`qiskit_loads` -- parse QASM 2 with qiskit, accepting the extended
  (legacy) ``qelib1.inc`` gate set that quspect emits.
* :func:`qiskit_measure_map` / :func:`quspect_measure_map` -- ordered
  ``(qubit, clbit)`` pairs of all measurements.
* :func:`random_legacy_qiskit_circuit` -- random qiskit circuit restricted to
  the qelib1 gates that quspect's importer understands.
* :func:`random_rich_circuit` -- random quspect circuit with symbolic
  parameters, every noise-channel type, measurements and barriers (for QSAL).
* :func:`assert_circuits_identical` -- instruction-by-instruction equality,
  including noise channels (which ``CircuitInstruction.__eq__`` ignores).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import qiskit
import qiskit.qasm2
from qiskit.circuit import library as qlib

import quspect
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.noise.channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    BitPhaseFlipChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    NoiseChannel,
    PauliChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
)

QELIB_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
"""Standard QASM 2 header used by the importer tests."""


def qiskit_loads(source: str) -> qiskit.QuantumCircuit:
    """Parse QASM 2 with qiskit using the legacy (extended) qelib1 gate set."""
    return qiskit.qasm2.loads(
        source, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
    )


def qiskit_measure_map(qc: qiskit.QuantumCircuit) -> list[tuple[int, int]]:
    """Return the ordered ``(qubit, clbit)`` pairs of all qiskit measurements."""
    pairs: list[tuple[int, int]] = []
    for ci in qc.data:
        if ci.operation.name == "measure":
            q = qc.find_bit(ci.qubits[0]).index
            c = qc.find_bit(ci.clbits[0]).index
            pairs.append((q, c))
    return pairs


def quspect_measure_map(circ: quspect.QuantumCircuit) -> list[tuple[int, int]]:
    """Return the ordered ``(qubit, clbit)`` pairs of all quspect measurements."""
    pairs: list[tuple[int, int]] = []
    for instr in circ.instructions:
        if instr.is_measure:
            pairs.extend(zip(instr.qubits, instr.clbits, strict=True))
    return pairs


def strip_measurements(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Return a copy of ``qc`` without measurements and barriers."""
    out = qc.copy_empty_like()
    for ci in qc.data:
        if ci.operation.name not in ("measure", "barrier"):
            out.append(ci)
    return out


def unitary_only(circ: quspect.QuantumCircuit) -> quspect.QuantumCircuit:
    """Return a copy of a quspect circuit keeping only gate instructions."""
    out = quspect.QuantumCircuit()
    for instr in circ.instructions:
        if instr.is_gate:
            assert instr.gate is not None
            out.append(instr.gate, instr.qubits)
    return out


# qiskit gate factories whose QASM 2 names quspect's importer understands,
# with (num_qubits, num_params).
LEGACY_IMPORTABLE: dict[str, tuple[Any, int, int]] = {
    "id": (qlib.IGate, 1, 0),
    "h": (qlib.HGate, 1, 0),
    "x": (qlib.XGate, 1, 0),
    "y": (qlib.YGate, 1, 0),
    "z": (qlib.ZGate, 1, 0),
    "s": (qlib.SGate, 1, 0),
    "sdg": (qlib.SdgGate, 1, 0),
    "t": (qlib.TGate, 1, 0),
    "tdg": (qlib.TdgGate, 1, 0),
    "sx": (qlib.SXGate, 1, 0),
    "sxdg": (qlib.SXdgGate, 1, 0),
    "rx": (qlib.RXGate, 1, 1),
    "ry": (qlib.RYGate, 1, 1),
    "rz": (qlib.RZGate, 1, 1),
    "p": (qlib.PhaseGate, 1, 1),
    "u1": (qlib.U1Gate, 1, 1),
    "u2": (qlib.U2Gate, 1, 2),
    "u3": (qlib.U3Gate, 1, 3),
    "cx": (qlib.CXGate, 2, 0),
    "cz": (qlib.CZGate, 2, 0),
    "swap": (qlib.SwapGate, 2, 0),
    "crx": (qlib.CRXGate, 2, 1),
    "cry": (qlib.CRYGate, 2, 1),
    "crz": (qlib.CRZGate, 2, 1),
    "cp": (qlib.CPhaseGate, 2, 1),
    "cu1": (qlib.CU1Gate, 2, 1),
    "cu3": (qlib.CU3Gate, 2, 3),
    "ccx": (qlib.CCXGate, 3, 0),
}


def _plain_angle(rng: np.random.Generator) -> float:
    """Random angle that qiskit will not pretty-print as a multiple of pi."""
    while True:
        value = float(rng.uniform(-3.0, 3.0))
        if all(
            abs(value - k * math.pi / d) > 1e-3
            for d in (1, 2, 3, 4, 6, 8)
            for k in (-8, -1, 0, 1, 8)
        ):
            return value


def random_legacy_qiskit_circuit(
    num_qubits: int,
    depth: int,
    seed: int,
    *,
    gate_names: list[str] | None = None,
    measure: bool = False,
) -> qiskit.QuantumCircuit:
    """Build a random qiskit circuit from :data:`LEGACY_IMPORTABLE` gates.

    Angles avoid multiples of ``pi`` so ``qiskit.qasm2.dumps`` prints plain
    decimal literals.  If ``measure`` is true every qubit is measured into a
    randomly permuted clbit.
    """
    rng = np.random.default_rng(seed)
    names = list(LEGACY_IMPORTABLE) if gate_names is None else gate_names
    names = [n for n in names if LEGACY_IMPORTABLE[n][1] <= num_qubits]
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits if measure else 0)
    for _ in range(depth):
        name = names[int(rng.integers(len(names)))]
        factory, nq, npar = LEGACY_IMPORTABLE[name]
        qubits = [int(q) for q in rng.permutation(num_qubits)[:nq]]
        params = [_plain_angle(rng) for _ in range(npar)]
        qc.append(factory(*params), qubits)
    if measure:
        perm = [int(c) for c in rng.permutation(num_qubits)]
        for q in range(num_qubits):
            qc.measure(q, perm[q])
    return qc


# =============================================================================
# Random rich circuits (QSAL)
# =============================================================================


def all_channel_examples() -> list[tuple[NoiseChannel, int]]:
    """One instance of each noise-channel type with its qubit count."""
    k0 = np.sqrt(0.7) * np.eye(2, dtype=complex)
    k1 = np.sqrt(0.3) * np.array([[0, 1], [1, 0]], dtype=complex) * 1j
    return [
        (BitFlipChannel(name="bit_flip", num_qubits=1, p=0.11), 1),
        (PhaseFlipChannel(name="phase_flip", num_qubits=1, p=0.12), 1),
        (BitPhaseFlipChannel(name="bit_phase_flip", num_qubits=1, p=0.13), 1),
        (DepolarizingChannel(name="depolarizing", num_qubits=1, p=0.14), 1),
        (DepolarizingChannel(name="depolarizing", num_qubits=2, p=0.05), 2),
        (PauliChannel(name="pauli", num_qubits=1, px=0.1, py=0.05, pz=0.02), 1),
        (AmplitudeDampingChannel(name="amplitude_damping", num_qubits=1, gamma=0.2), 1),
        (PhaseDampingChannel(name="phase_damping", num_qubits=1, gamma=0.3), 1),
        (
            GeneralizedAmplitudeDampingChannel(
                name="generalized_amplitude_damping",
                num_qubits=1,
                gamma=0.25,
                p_thermal=0.4,
            ),
            1,
        ),
        (KrausChannel(name="readout", num_qubits=1, kraus_ops=[k0, k1]), 1),
    ]


def random_rich_circuit(
    num_qubits: int, depth: int, seed: int, *, symbolic: bool = True
) -> quspect.QuantumCircuit:
    """Random quspect circuit with gates, symbolic params, channels, measures.

    Uses the fluent API for the 25 standard gates (from
    :data:`qiskit_tests.helpers.GATE_SPECS`), mixes in ``Parameter`` /
    ``ParameterExpression`` arguments when ``symbolic`` is true, every noise
    channel type, barriers (partial and full) and measurements.
    """
    from qiskit_tests.helpers import GATE_SPECS

    rng = np.random.default_rng(seed)
    theta, phi = Parameter("theta"), Parameter("phi")
    specs = [s for s in GATE_SPECS.values() if s.num_qubits <= num_qubits]
    channels = [(ch, nq) for ch, nq in all_channel_examples() if nq <= num_qubits]
    circ = quspect.QuantumCircuit(name=f"rich_{seed}")
    for q in range(num_qubits):
        circ.id(q)
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.7:
            spec = specs[int(rng.integers(len(specs)))]
            qubits = [int(q) for q in rng.permutation(num_qubits)[: spec.num_qubits]]
            params: list[Any] = []
            for _ in range(spec.num_params):
                kind = rng.random()
                if symbolic and kind < 0.2:
                    params.append(theta)
                elif symbolic and kind < 0.35:
                    params.append(
                        ParameterExpression(
                            {theta: float(rng.uniform(-2, 2)), phi: 0.5},
                            constant=float(rng.uniform(-1, 1)),
                        )
                    )
                else:
                    params.append(float(rng.uniform(-2 * math.pi, 2 * math.pi)))
            spec.apply_quspect(circ, params, qubits)
        elif roll < 0.85:
            ch, nq = channels[int(rng.integers(len(channels)))]
            qubits = [int(q) for q in rng.permutation(num_qubits)[:nq]]
            circ.append_channel(ch, qubits)
        elif roll < 0.92:
            if rng.random() < 0.5:
                circ.barrier()
            else:
                circ.barrier(*sorted(int(q) for q in rng.permutation(num_qubits)[:2]))
        else:
            q = int(rng.integers(num_qubits))
            c = int(rng.integers(num_qubits))
            circ.measure([q], [c])
    return circ


def channel_signature(ch: NoiseChannel | None) -> Any:
    """Comparable description of a noise channel (type, fields, Kraus ops)."""
    if ch is None:
        return None
    fields = {
        k: getattr(ch, k)
        for k in ("name", "num_qubits", "p", "px", "py", "pz", "gamma", "p_thermal")
        if hasattr(ch, k)
    }
    kraus = [np.round(np.asarray(k, dtype=complex), 12) for k in ch.kraus_operators()]
    return type(ch).__name__, fields, [k.tolist() for k in kraus]


def assert_circuits_identical(
    a: quspect.QuantumCircuit, b: quspect.QuantumCircuit
) -> None:
    """Assert two quspect circuits are instruction-by-instruction identical.

    Compares type, qubits, clbits, gate (name, num_qubits, params -- symbolic
    parameters by value equality), instruction label and the full noise
    channel (``CircuitInstruction.__eq__`` ignores channels).
    """
    assert len(a) == len(b), f"length mismatch {len(a)} != {len(b)}"
    for i, (ia, ib) in enumerate(zip(a.instructions, b.instructions, strict=True)):
        assert ia == ib, f"instruction {i} differs: {ia} != {ib}"
        assert channel_signature(ia.channel) == channel_signature(ib.channel), (
            f"channel {i} differs: {ia.channel} != {ib.channel}"
        )
        if ia.gate is not None:
            assert ib.gate is not None
            assert [type(p) for p in ia.gate.params] == [
                type(p) for p in ib.gate.params
            ], f"param types differ at {i}"
    assert a.num_qubits == b.num_qubits
    assert a.num_clbits == b.num_clbits
