"""OpenQASM 2.0 export and import refuse what they cannot represent."""

from __future__ import annotations

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.ir import (
    QASM2ExportConfig,
    QASM2Exporter,
    QASM2ExportMode,
    QASM2Importer,
    QASM2ParseError,
    UnsupportedGateError,
    qasm2_loads,
)
from quspect.ir.qsal import dumps, loads
from quspect.noise.channel import KrausChannel


def _noisy_circuits() -> list[QuantumCircuit]:
    kraus = KrausChannel(
        name="kraus",
        num_qubits=1,
        kraus_ops=[np.sqrt(0.9) * np.eye(2), np.sqrt(0.1) * np.diag([1.0, -1.0])],
    )
    return [
        QuantumCircuit().h(0).depolarize(0.1, 0).measure([0], [0]),
        QuantumCircuit().h(0).cx(0, 1).amplitude_damp(0.2, 1),
        QuantumCircuit().bit_flip(0.0, 0),
        QuantumCircuit().x(0).append_channel(kraus, (0,)),
    ]


@pytest.mark.parametrize("mode", list(QASM2ExportMode))
@pytest.mark.parametrize("index", range(4))
def test_exporter_refuses_noise_channels(mode: QASM2ExportMode, index: int) -> None:
    """A noisy circuit is not written as its ideal circuit."""
    circuit = _noisy_circuits()[index]
    channel = next(i.channel for i in circuit.instructions if i.is_channel)
    assert channel is not None
    exporter = QASM2Exporter(QASM2ExportConfig(mode=mode))
    with pytest.raises(UnsupportedGateError, match="noise channel") as excinfo:
        exporter.export(circuit)
    assert excinfo.value.gate_name == channel.name
    assert "OpenQASM 2.0" in excinfo.value.format_name
    assert "QSAL" in str(excinfo.value)


def test_exporter_refuses_noise_channels_when_writing_a_file(tmp_path) -> None:
    target = tmp_path / "noisy.qasm"
    with pytest.raises(UnsupportedGateError, match="noise channel"):
        QASM2Exporter().export_to_file(_noisy_circuits()[0], target)
    assert not target.exists()


def test_qsal_keeps_the_channel_the_message_points_to() -> None:
    circuit = _noisy_circuits()[0]
    restored = loads(dumps(circuit)).circuit
    assert [i.type for i in restored.instructions] == [
        i.type for i in circuit.instructions
    ]
    assert restored.instructions[1].channel == circuit.instructions[1].channel


def test_noiseless_export_is_unchanged() -> None:
    circuit = QuantumCircuit().h(0).barrier(0).measure([0], [0])
    assert QASM2Exporter().export(circuit).splitlines()[-3:] == [
        "h q[0];",
        "barrier q[0];",
        "measure q[0] -> c[0];",
    ]


@pytest.mark.parametrize("source", [None, 3, 1.5, b"OPENQASM 2.0;", ["OPENQASM 2.0;"]])
def test_importer_refuses_non_text_with_a_clear_message(source: object) -> None:
    for parse in (qasm2_loads, QASM2Importer().imports):
        with pytest.raises(TypeError, match="OpenQASM 2.0 source must be a str") as e:
            parse(source)  # type: ignore[arg-type]
        assert type(source).__name__ in str(e.value)
        assert "incompatible function arguments" not in str(e.value)


def test_importer_still_reports_malformed_text_as_parse_errors() -> None:
    with pytest.raises(QASM2ParseError):
        qasm2_loads("")
    with pytest.raises(QASM2ParseError):
        qasm2_loads("OPENQASM 2.0; qreg q[1]; foo q[0];")
