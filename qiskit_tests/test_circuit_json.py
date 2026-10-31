"""``QuantumCircuit.to_json`` / ``from_json`` round-trip tests.

The round trip must preserve gates, symbolic parameters / expressions,
measurements, barriers, labels, and *noise channels including their
strengths*.  Round-tripped circuits are re-checked against qiskit after
binding, and noisy circuits via the Kraus representation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qiskit_tests.helpers import (
    assert_unitary_equal,
    qiskit_circuit_unitary,
    qiskit_probabilities,
    quspect_circuit_unitary,
    random_circuit_pair,
)
from quspect import QuantumCircuit
from quspect.circuit import CircuitInstruction, Parameter
from quspect.circuit.instruction import InstructionType
from quspect.noise.channel import KrausChannel

A = Parameter("a")
B = Parameter("b")


def _param_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(name="param")
    circ.h(0).rx(A, 0).ry(2 * A - B + 1, 1).u3(A, B, 0.3, 0).cu3(0.1, A, B, 0, 1)
    circ.u2(A, 0.2, 1).u1(0.7, 0).crz(-B, 1, 0).ccx(0, 1, 2).swap(2, 0)
    circ.barrier().measure([0, 1, 2], [2, 0, 1])
    return circ


def test_json_roundtrip_gates_parameters_measure() -> None:
    """Gates, Parameter / expression params, barrier and measure survive."""
    circ = _param_circuit()
    text = circ.to_json()
    json.loads(text)  # valid JSON
    back = QuantumCircuit.from_json(text)
    assert back.name == "param"
    assert back.instructions == circ.instructions
    assert back.parameters == {A, B}
    assert back.num_qubits == 3 and back.num_clbits == 3


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(4))
def test_json_roundtrip_preserves_unitary(seed: int) -> None:
    """A random circuit's unitary is unchanged by the JSON round trip."""
    circ, qc = random_circuit_pair(3, 25, seed)
    back = QuantumCircuit.from_json(circ.to_json())
    assert_unitary_equal(quspect_circuit_unitary(back), qiskit_circuit_unitary(qc))


@pytest.mark.qiskit_ref
def test_json_roundtrip_then_bind_matches_qiskit() -> None:
    """Binding a round-tripped symbolic circuit gives the qiskit unitary."""
    circ = QuantumCircuit().rx(A, 0).cp(2 * A - B, 0, 1).u3(B, A, -0.4, 1)
    binding = {A: 0.4, B: -1.3}
    back = QuantumCircuit.from_json(circ.to_json()).bind_parameters(binding)
    assert_unitary_equal(
        quspect_circuit_unitary(back),
        qiskit_circuit_unitary(circ.bind_parameters(binding)),
    )


def test_json_roundtrip_label() -> None:
    """Instruction labels are serialized when non-empty."""
    circ = QuantumCircuit()
    circ._instructions.append(  # labelled instruction (no public setter)
        CircuitInstruction(type=InstructionType.BARRIER, qubits=(0,), label="tag")
    )
    back = QuantumCircuit.from_json(circ.to_json())
    assert back[0].label == "tag"


def _noisy_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(name="noisy").h(0).rx(A, 1)
    circ.bit_flip(0.11, 0).phase_flip(0.12, 1).bit_phase_flip(0.13, 0)
    circ.depolarize(0.14, 1).pauli_error(0.05, 0.06, 0.07, 0)
    circ.amplitude_damp(0.21, 1).phase_damp(0.22, 0)
    circ.measure([0, 1], [0, 1])
    return circ


def test_json_roundtrip_channel_types_and_pauli() -> None:
    """Channel types, qubits and the Pauli probabilities survive."""
    circ = _noisy_circuit()
    back = QuantumCircuit.from_json(circ.to_json())
    assert len(back) == len(circ)
    for orig, new in zip(circ, back, strict=True):
        assert orig.type == new.type
        assert orig.qubits == new.qubits
        if orig.is_channel:
            assert orig.channel is not None and new.channel is not None
            assert type(new.channel) is type(orig.channel)
            assert new.channel.name == orig.channel.name
    pauli = next(i.channel for i in back if i.is_channel and i.channel.name == "pauli")
    assert (pauli.px, pauli.py, pauli.pz) == (0.05, 0.06, 0.07)  # type: ignore[union-attr]


def test_json_roundtrip_preserves_channel_strength() -> None:
    """Every channel's strength (p / gamma) is preserved by the round trip."""
    circ = _noisy_circuit()
    back = QuantumCircuit.from_json(circ.to_json())
    for orig, new in zip(circ, back, strict=True):
        if orig.is_channel:
            assert new.channel == orig.channel, (orig.channel, new.channel)


@pytest.mark.qiskit_ref
def test_json_roundtrip_noisy_distribution_matches_qiskit() -> None:
    """Exact (Kraus) distribution of the round-tripped circuit is unchanged."""
    circ = _noisy_circuit().bind_parameters({A: 0.8})
    back = QuantumCircuit.from_json(circ.to_json())
    ref = qiskit_probabilities(circ)
    got = qiskit_probabilities(back)
    for key in set(ref) | set(got):
        assert got.get(key, 0.0) == pytest.approx(ref.get(key, 0.0), abs=1e-12)


def test_json_roundtrip_generalized_amplitude_damping() -> None:
    """A generalized amplitude damping channel survives the round trip."""
    circ = QuantumCircuit().generalized_amplitude_damp(0.3, 0.2, 0)
    back = QuantumCircuit.from_json(circ.to_json())
    assert back[0].channel == circ[0].channel


def _channel_json(channel: dict[str, object]) -> str:
    return json.dumps(
        {
            "name": "manual",
            "instructions": [
                {"type": "CHANNEL", "qubits": [0], "clbits": [], "channel": channel}
            ],
        }
    )


@pytest.mark.parametrize(
    ("data", "attrs"),
    [
        ({"channel_type": "bit_flip", "p": 0.3}, {"p": 0.3}),
        ({"channel_type": "phase_flip", "params": [0.2]}, {"p": 0.2}),
        ({"channel_type": "bit_phase_flip", "p": 0.1}, {"p": 0.1}),
        ({"channel_type": "depolarizing", "p": 0.4}, {"p": 0.4}),
        ({"channel_type": "amplitude_damping", "gamma": 0.5}, {"gamma": 0.5}),
        ({"channel_type": "phase_damping", "gamma": 0.6}, {"gamma": 0.6}),
        (
            {"channel_type": "generalized_amplitude_damping", "gamma": 0.3},
            {"gamma": 0.3, "p_thermal": 0.0},
        ),
        (
            {"channel_type": "pauli", "px": 0.1, "py": 0.2, "pz": 0.3},
            {"px": 0.1, "py": 0.2, "pz": 0.3},
        ),
    ],
)
def test_from_json_accepts_explicit_channel_fields(
    data: dict[str, object], attrs: dict[str, float]
) -> None:
    """``from_json`` reads ``p`` / ``gamma`` / ``params`` channel fields."""
    back = QuantumCircuit.from_json(_channel_json({**data, "num_qubits": 1}))
    ch = back[0].channel
    assert ch is not None and ch.name == data["channel_type"]
    for key, value in attrs.items():
        assert getattr(ch, key) == pytest.approx(value)


def test_kraus_channel_json_not_deserializable() -> None:
    """A KrausChannel serializes its op count but cannot be deserialized."""
    kraus = KrausChannel(
        name="kraus", num_qubits=1, kraus_ops=[np.eye(2, dtype=complex)]
    )
    circ = QuantumCircuit().append_channel(kraus, [0])
    text = circ.to_json()
    assert json.loads(text)["instructions"][0]["channel"]["kraus_ops_count"] == 1
    with pytest.raises(ValueError, match="Kraus"):
        QuantumCircuit.from_json(text)


@pytest.mark.parametrize(
    "payload",
    [
        _channel_json({"channel_type": "nonsense", "num_qubits": 1}),
        json.dumps(
            {
                "instructions": [
                    {
                        "type": "GATE",
                        "qubits": [0],
                        "gate": {"name": "nope", "num_qubits": 1},
                    }
                ]
            }
        ),
    ],
)
def test_from_json_unknown_names_raise(payload: str) -> None:
    """Unknown channel types and unknown standard gates raise ValueError."""
    with pytest.raises(ValueError, match="Unknown"):
        QuantumCircuit.from_json(payload)


def test_from_json_minimal_document() -> None:
    """Missing optional keys default to an empty, unnamed circuit."""
    circ = QuantumCircuit.from_json("{}")
    assert circ.name == ""
    assert len(circ) == 0
