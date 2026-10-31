"""Offline contract tests for the LogicalQubit circuit translator.

These tests exercise the installed lqcloud SDK's circuit and serialization
APIs only.  They never construct a provider or make a network request.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

pytest.importorskip("lqcloud")

from lqcloud.backend.serialization import serialize_circuit

from quspect import QuantumCircuit
from quspect.backends._exceptions import TranslationError
from quspect.backends.logicalqubit._translator import LQCircuitTranslator
from quspect.gates.parameterized import CU3, U2
from quspect.ir.qsal import dumps


def _translate(circuit: QuantumCircuit) -> Any:
    return LQCircuitTranslator().translate(dumps(circuit))


def _gate_rows(circuit: Any) -> list[tuple[str, list[float], list[int]]]:
    rows: list[tuple[str, list[float], list[int]]] = []
    for gate, qubits, _clbits in circuit.data:
        rows.append(
            (
                gate.name,
                list(gate.params),
                [circuit.qubits.index(qubit) for qubit in qubits],
            )
        )
    return rows


def _qsal_with_instruction(instruction: dict[str, Any]) -> str:
    return json.dumps(
        {
            "qsal_version": "0.1.0",
            "circuit": {
                "num_qubits": 1,
                "num_clbits": 1,
                "instructions": [instruction],
            },
        }
    )


class TestLogicalQubitGateSemantics:
    def test_cp_uses_phase_and_crz_decomposition(self) -> None:
        theta = 0.7
        circuit = QuantumCircuit().cp(theta, 0, 1).barrier()
        circuit.measure([0, 1], [0, 1])

        rows = _gate_rows(_translate(circuit))
        assert rows[0] == ("p", pytest.approx([theta / 2]), [0])
        assert rows[1] == ("crz", pytest.approx([theta]), [0, 1])

    def test_tdg_lowers_to_minus_pi_over_four_not_sdg(self) -> None:
        circuit = QuantumCircuit().tdg(0).barrier().measure([0], [0])

        translated = _translate(circuit)
        assert _gate_rows(translated)[0][0] == "tdg"

        instructions = serialize_circuit(translated)["instructions"]
        assert all(instruction["name"] != "sdg" for instruction in instructions)
        rz = next(
            instruction for instruction in instructions if instruction["name"] == "rz"
        )
        assert rz["params"] == pytest.approx([-math.pi / 4])

    def test_u2_and_u3_use_lqcloud_u_method(self) -> None:
        circuit = QuantumCircuit()
        circuit.append(U2.with_params(0.2, -0.3), (0,))
        circuit.u3(0.4, 0.5, -0.6, 1)
        circuit.barrier()
        circuit.measure([0, 1], [0, 1])

        translated = _translate(circuit)
        rows = _gate_rows(translated)

        assert rows[0] == ("u", pytest.approx([math.pi / 2, 0.2, -0.3]), [0])
        assert rows[1] == ("u", pytest.approx([0.4, 0.5, -0.6]), [1])
        # lqcloud 0.4.1 accepts and recursively serializes both U gates.
        serialize_circuit(translated)

    def test_cu3_uses_standard_two_cnot_decomposition(self) -> None:
        theta, phi, lam = 0.8, -0.4, 0.6
        circuit = QuantumCircuit()
        circuit.append(CU3.with_params(theta, phi, lam), (0, 1))
        circuit.barrier()
        circuit.measure([0, 1], [0, 1])

        translated = _translate(circuit)
        rows = _gate_rows(translated)

        assert [row[0] for row in rows[:6]] == ["p", "p", "cx", "u", "cx", "u"]
        assert rows[0][1] == pytest.approx([(lam + phi) / 2])
        assert rows[0][2] == [0]
        assert rows[1][1] == pytest.approx([(lam - phi) / 2])
        assert rows[1][2] == [1]
        assert rows[2][2] == [0, 1]
        assert rows[3][1] == pytest.approx([-theta / 2, 0.0, -(phi + lam) / 2])
        assert rows[3][2] == [1]
        assert rows[4][2] == [0, 1]
        assert rows[5][1] == pytest.approx([theta / 2, phi, 0.0])
        assert rows[5][2] == [1]
        # This catches accidental calls to the non-existent qc.cu3/qc.u3 APIs.
        serialize_circuit(translated)


class TestLogicalQubitMeasurementBarriers:
    def test_measurement_at_circuit_start_gets_real_barrier(self) -> None:
        translated = _translate(QuantumCircuit().measure([0], [0]))

        assert [row[0] for row in _gate_rows(translated)] == ["barrier", "measure"]
        serialize_circuit(translated)

    def test_partial_barrier_is_completed_per_qubit(self) -> None:
        circuit = QuantumCircuit().x(0).x(1)
        circuit.barrier(0)
        circuit.measure([0, 1], [0, 1])

        translated = _translate(circuit)
        rows = _gate_rows(translated)

        barrier_rows = [row for row in rows if row[0] == "barrier"]
        assert [row[2] for row in barrier_rows] == [[0], [1]]
        serialize_circuit(translated)

    def test_repeated_measurement_gets_fresh_barrier(self) -> None:
        circuit = QuantumCircuit().x(0)
        circuit.measure([0], [0])
        circuit.x(0)
        circuit.measure([0], [0])

        translated = _translate(circuit)

        assert [row[0] for row in _gate_rows(translated)] == [
            "x",
            "barrier",
            "measure",
            "x",
            "barrier",
            "measure",
        ]
        serialize_circuit(translated)

    def test_gate_after_barrier_requires_a_new_measurement_barrier(self) -> None:
        circuit = QuantumCircuit().x(0).barrier(0).h(0).measure([0], [0])

        translated = _translate(circuit)

        assert [row[0] for row in _gate_rows(translated)] == [
            "x",
            "barrier",
            "h",
            "barrier",
            "measure",
        ]
        serialize_circuit(translated)


class TestLogicalQubitTranslatorRejections:
    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(TranslationError, match="parse QSAL"):
            LQCircuitTranslator().translate("{")

    def test_rejects_unbound_parameter(self) -> None:
        qsal = _qsal_with_instruction(
            {
                "type": "GATE",
                "qubits": [0],
                "clbits": [],
                "gate": {"name": "rx", "params": [{"_param": "theta"}]},
            }
        )

        with pytest.raises(TranslationError, match="unbound parameter 'theta'"):
            LQCircuitTranslator().translate(qsal)

    def test_rejects_noise_channel(self) -> None:
        qsal = _qsal_with_instruction(
            {
                "type": "CHANNEL",
                "qubits": [0],
                "clbits": [],
            }
        )

        with pytest.raises(TranslationError, match="noise channels"):
            LQCircuitTranslator().translate(qsal)

    def test_rejects_unknown_gate(self) -> None:
        qsal = _qsal_with_instruction(
            {
                "type": "GATE",
                "qubits": [0],
                "clbits": [],
                "gate": {"name": "not_a_gate", "params": []},
            }
        )

        with pytest.raises(TranslationError, match="not supported"):
            LQCircuitTranslator().translate(qsal)

    def test_rejects_measurement_arity_mismatch(self) -> None:
        qsal = _qsal_with_instruction(
            {
                "type": "MEASURE",
                "qubits": [0],
                "clbits": [],
            }
        )

        with pytest.raises(TranslationError, match="1 qubits but 0 classical bits"):
            LQCircuitTranslator().translate(qsal)
