"""Tests for QSAL serialization/deserialization.

Covers round-trip fidelity, parameter preservation, observable
serialization, file I/O, version checking, and edge cases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.ir.errors import QSALVersionError
from quspect.ir.qsal import (
    QSALDocument,
    dump,
    dumps,
    load,
    loads,
)
from quspect.noise.channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    BitPhaseFlipChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    PauliChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_observable() -> object:
    """Create a simple QubitOperator for testing."""
    from quspect.operators.qubit_operator import QubitOperator

    return QubitOperator(
        {
            ((0, "Z"), (1, "Z")): 0.5,
            ((0, "X"),): 0.3,
            ((1, "X"),): 0.3,
        }
    )


# =============================================================================
# Basic serialization tests
# =============================================================================


class TestBasicSerialization:
    """Tests for basic QSAL to/from JSON."""

    def test_simple_circuit_roundtrip(self) -> None:
        """A simple circuit survives a to_json→from_json round-trip."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1).measure_all()

        doc = QSALDocument(name="bell", circuit=circ)
        json_str = doc.to_json()
        doc2 = QSALDocument.from_json(json_str)

        assert doc2.name == "bell"
        assert doc2.circuit is not None
        assert doc2.circuit.num_qubits == 2
        assert doc2.circuit.num_clbits == 2
        assert len(doc2.circuit) == 3

    def test_circuit_preserves_gate_sequence(self) -> None:
        """Gate instructions are preserved in order."""
        circ = QuantumCircuit(name="test")
        circ.h(0).x(1).cx(0, 1).z(0)

        doc = QSALDocument(name="test", circuit=circ)
        json_str = doc.to_json()
        doc2 = QSALDocument.from_json(json_str)

        instrs = list(doc2.circuit)
        assert len(instrs) == 4
        assert instrs[0].gate.name == "h"
        assert instrs[1].gate.name == "x"
        assert instrs[2].gate.name == "cx"
        assert instrs[3].gate.name == "z"

    def test_parameterized_circuit_roundtrip(self) -> None:
        """Symbolic parameters are preserved across round-trip."""
        theta = Parameter("θ")
        phi = Parameter("φ")

        circ = QuantumCircuit(name="vqe")
        circ.rx(theta, 0)
        circ.ry(phi, 1)
        circ.cx(0, 1)

        doc = QSALDocument(name="vqe", circuit=circ)
        json_str = doc.to_json()
        doc2 = QSALDocument.from_json(json_str)

        assert doc2.is_parameterized is True
        circ2 = doc2.circuit
        params = circ2.parameters
        names = {p.name for p in params}
        assert names == {"θ", "φ"}

    def test_parameter_names_preserved(self) -> None:
        """Parameter names survive serialization exactly."""
        theta = Parameter("θ")
        circ = QuantumCircuit(name="test")
        circ.rx(theta, 0)

        json_str = dumps(circ)
        doc = loads(json_str)
        params = doc.circuit.parameters
        assert len(params) == 1
        p = next(iter(params))
        assert p.name == "θ"

    def test_bound_circuit_roundtrip(self) -> None:
        """A circuit with bound parameters round-trips correctly."""
        theta = Parameter("θ")
        circ = QuantumCircuit(name="test")
        circ.rx(theta, 0)
        bound = circ.bind_parameters({theta: 0.5})

        json_str = dumps(bound)
        doc = loads(json_str)

        assert doc.is_parameterized is False
        instr = next(iter(doc.circuit))
        assert instr.gate.params == (0.5,)

    def test_affine_expression_roundtrip(self) -> None:
        """Affine coefficients and constants survive QSAL serialization."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        expression = 2 * theta - 0.5 * phi + 0.125
        circ = QuantumCircuit(name="affine").rx(expression, 0)

        encoded = json.loads(dumps(circ))
        raw_param = encoded["circuit"]["instructions"][0]["gate"]["params"][0]
        assert raw_param == {
            "_expr": {
                "constant": 0.125,
                "terms": {"phi": -0.5, "theta": 2.0},
            }
        }

        restored = loads(json.dumps(encoded)).circuit
        restored_param = restored[0].gate.params[0]
        assert isinstance(restored_param, ParameterExpression)
        assert restored_param == expression
        assert restored.parameters == {theta, phi}

    def test_affine_expression_can_be_bound_after_roundtrip(self) -> None:
        theta = Parameter("θ")
        phi = Parameter("φ")
        circ = QuantumCircuit().ry(-theta + 3 * phi - 1, 0)
        restored = loads(dumps(circ)).circuit
        bound = restored.bind_parameters({theta: 0.5, phi: 0.25})
        assert bound[0].gate.params[0] == pytest.approx(-0.75)

    def test_circuit_name_roundtrip(self) -> None:
        """The circuit name is stored independently of the document name (B44)."""
        circ = QuantumCircuit(name="ansatz")
        circ.h(0)
        encoded = json.loads(dumps(circ, name="document"))
        assert encoded["name"] == "document"
        assert encoded["circuit"]["name"] == "ansatz"
        back = loads(json.dumps(encoded)).circuit
        assert back is not None
        assert back.name == "ansatz"

    def test_unnamed_circuit_omits_name_key(self) -> None:
        """Unnamed circuits keep the legacy layout and still load."""
        encoded = json.loads(dumps(QuantumCircuit().h(0)))
        assert "name" not in encoded["circuit"]
        back = loads(json.dumps(encoded)).circuit
        assert back is not None
        assert back.name == ""

    def test_gate_label_roundtrip(self) -> None:
        """Gate labels survive for standard and parameterized gates (B45)."""
        from quspect.gates.gate import Gate

        circ = QuantumCircuit()
        circ.append(Gate(name="rx", num_qubits=1, params=(0.3,), label="drive"), (0,))
        circ.append(Gate(name="h", num_qubits=1, label="had"), (0,))
        circ.h(0)
        back = loads(dumps(circ)).circuit
        assert back is not None
        labels = [instr.gate.label for instr in back if instr.gate is not None]
        expected = [instr.gate.label for instr in circ if instr.gate is not None]
        assert labels == expected
        assert labels[:2] == ["drive", "had"]
        assert back[0].gate is not None
        assert back[0].gate.params == (0.3,)

    def test_empty_circuit(self) -> None:
        """An empty circuit can be serialized."""
        circ = QuantumCircuit(name="empty")
        json_str = dumps(circ)
        doc = loads(json_str)
        assert doc.circuit is not None
        assert len(doc.circuit) == 0
        assert doc.circuit.num_qubits == 0

    def test_measure_and_barrier_roundtrip(self) -> None:
        """Measure and barrier instructions survive round-trip."""
        circ = QuantumCircuit(name="test")
        circ.h(0).barrier(0).measure([0], [0])

        json_str = dumps(circ)
        doc = loads(json_str)

        instrs = list(doc.circuit)
        assert instrs[0].is_gate is True
        assert instrs[1].is_barrier is True
        assert instrs[2].is_measure is True


class TestChannelSerialization:
    """Noise channels retain executable Kraus data across QSAL round-trips."""

    def test_all_builtin_channels_roundtrip(self) -> None:
        """Every built-in channel preserves its type, name, and Kraus map."""
        channels = [
            PauliChannel(name="biased_pauli", num_qubits=1, px=0.1, py=0.2, pz=0.3),
            BitFlipChannel(name="custom_bit_flip", num_qubits=1, p=0.25),
            PhaseFlipChannel(name="custom_phase_flip", num_qubits=1, p=0.3),
            BitPhaseFlipChannel(name="custom_bit_phase_flip", num_qubits=1, p=0.35),
            DepolarizingChannel(name="custom_depolarizing", num_qubits=1, p=0.8),
            DepolarizingChannel(name="two_qubit_depolarizing", num_qubits=2, p=0.2),
            AmplitudeDampingChannel(
                name="custom_amplitude_damping", num_qubits=1, gamma=0.4
            ),
            PhaseDampingChannel(name="custom_phase_damping", num_qubits=1, gamma=0.45),
            GeneralizedAmplitudeDampingChannel(
                name="custom_gad",
                num_qubits=1,
                gamma=0.5,
                p_thermal=0.2,
            ),
        ]
        circuit = QuantumCircuit(name="builtin_channels")
        for channel in channels:
            circuit.append_channel(channel, tuple(range(channel.num_qubits)))

        serialized = dumps(circuit)
        raw_channels = [
            instruction["channel"]
            for instruction in json.loads(serialized)["circuit"]["instructions"]
        ]
        assert all(channel["channel_kind"] for channel in raw_channels)
        assert all(channel["kraus_ops"] for channel in raw_channels)

        restored = loads(serialized).circuit
        restored_channels = [
            instruction.channel for instruction in restored if instruction.is_channel
        ]
        assert len(restored_channels) == len(channels)
        for original, roundtripped in zip(channels, restored_channels, strict=True):
            assert type(roundtripped) is type(original)
            assert roundtripped.name == original.name
            assert roundtripped.num_qubits == original.num_qubits
            original_ops = original.kraus_operators()
            restored_ops = roundtripped.kraus_operators()
            assert len(restored_ops) == len(original_ops)
            for expected, actual in zip(original_ops, restored_ops, strict=True):
                assert np.allclose(actual, expected, atol=1e-12)

    def test_custom_complex_kraus_roundtrip(self) -> None:
        """Custom Kraus matrices preserve non-zero imaginary components."""
        identity = np.sqrt(0.7) * np.eye(2, dtype=complex)
        pauli_y = np.sqrt(0.3) * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        channel = KrausChannel(
            name="complex_custom",
            num_qubits=1,
            kraus_ops=[identity, pauli_y],
        )
        circuit = QuantumCircuit().append_channel(channel, (0,))

        serialized = dumps(circuit)
        raw_channel = json.loads(serialized)["circuit"]["instructions"][0]["channel"]
        assert raw_channel["channel_kind"] == "kraus"
        assert raw_channel["kraus_ops"][1][0][1][1] == pytest.approx(-np.sqrt(0.3))

        restored = loads(serialized).circuit[0].channel
        assert isinstance(restored, KrausChannel)
        assert restored.name == "complex_custom"
        assert len(restored.kraus_ops) == 2
        assert np.allclose(restored.kraus_ops[0], identity, atol=1e-12)
        assert np.allclose(restored.kraus_ops[1], pauli_y, atol=1e-12)


# =============================================================================
# Observable serialization tests
# =============================================================================


class TestObservableSerialization:
    """Tests for QubitOperator serialization within QSAL."""

    def test_simple_observable_roundtrip(self) -> None:
        """A QubitOperator survives round-trip serialization."""
        from quspect.operators.qubit_operator import QubitOperator

        h_op = QubitOperator(
            {
                ((0, "Z"), (1, "Z")): 0.5,
                ((0, "X"),): 0.3,
            }
        )

        circ = QuantumCircuit(name="test")
        circ.h(0).cx(0, 1)

        json_str = dumps(circ, observables=[("H", h_op)])
        doc = loads(json_str)

        assert doc.has_observables is True
        assert len(doc.observables) == 1
        obs_name, obs_op = doc.observables[0]
        assert obs_name == "H"
        assert len(obs_op.terms) == 2

    def test_multiple_observables(self) -> None:
        """Multiple observables are all preserved."""
        from quspect.operators.qubit_operator import QubitOperator

        h1 = QubitOperator({((0, "X"),): 1.0})
        h2 = QubitOperator({((1, "Z"),): 2.0})

        circ = QuantumCircuit(name="test")
        circ.h(0)

        json_str = dumps(circ, observables=[("H1", h1), ("H2", h2)])
        doc = loads(json_str)

        assert len(doc.observables) == 2
        assert doc.observables[0][0] == "H1"
        assert doc.observables[1][0] == "H2"

    def test_observable_coefficients_preserved(self) -> None:
        """Observable coefficients are numeric after round-trip."""
        from quspect.operators.qubit_operator import QubitOperator

        h_op = QubitOperator(
            {
                ((0, "Z"), (1, "Z")): 0.5,
                ((0, "Y"),): -1.0,
            }
        )

        json_str = dumps(QuantumCircuit(name="t"), observables=[("H", h_op)])
        doc = loads(json_str)
        _, obs_op = doc.observables[0]

        terms_list = list(obs_op.terms.items())
        coeffs = {coeff for _, coeff in terms_list}
        assert 0.5 in coeffs or any(abs(c - 0.5) < 1e-10 for c in coeffs)
        assert -1.0 in coeffs or any(abs(c + 1.0) < 1e-10 for c in coeffs)

    def test_affine_observable_coefficient_roundtrip(self) -> None:
        """Symbolic Hamiltonian coefficients retain their affine Jacobian."""
        from quspect.operators.qubit_operator import QubitOperator

        theta = Parameter("theta")
        phi = Parameter("phi")
        operator = QubitOperator({((0, "Z"),): 2 * theta - phi + 0.25})

        encoded = json.loads(
            dumps(
                QuantumCircuit().h(0),
                observables=[("H", operator)],
            )
        )
        raw_coefficient = encoded["observables"][0]["operator"]["terms"][0][
            "coefficient"
        ]
        assert raw_coefficient == {
            "_expr": {
                "constant": 0.25,
                "terms": {"phi": -1.0, "theta": 2.0},
            }
        }

        restored = loads(json.dumps(encoded)).observables[0][1]
        assert restored.parameters == {theta, phi}
        assert restored.terms[((0, "Z"),)] == 2 * theta - phi + 0.25
        bound = restored.bind_parameters({theta: 0.5, phi: 0.25})
        assert bound.terms[((0, "Z"),)] == pytest.approx(1.0)

    def test_complex_observable_coefficient_roundtrip(self) -> None:
        """Non-real coefficients are stored as [real, imag] pairs (B43)."""
        from quspect.operators.qubit_operator import QubitOperator

        op = QubitOperator(
            {((0, "Z"),): 0.5 + 0.5j, ((1, "X"),): 1j, ((0, "X"),): -2.0 + 0j}
        )
        encoded = json.loads(dumps(QuantumCircuit().h(0), observables=[("H", op)]))
        raw = {
            tuple(tuple(p) for p in term["paulis"]): term["coefficient"]
            for term in encoded["observables"][0]["operator"]["terms"]
        }
        assert raw[((0, "Z"),)] == [0.5, 0.5]
        assert raw[((1, "X"),)] == [0.0, 1.0]
        # Real-valued complex numbers keep the plain-number format.
        assert raw[((0, "X"),)] == -2.0

        restored = loads(json.dumps(encoded)).observables[0][1]
        assert restored.terms == op.terms

    def test_legacy_real_coefficient_still_loads(self) -> None:
        """Documents with plain numeric coefficients remain readable."""
        legacy = {
            "qsal_version": "0.1.0",
            "name": "",
            "circuit": {"num_qubits": 1, "num_clbits": 0, "instructions": []},
            "observables": [
                {
                    "name": "H",
                    "operator": {"terms": [{"paulis": [[0, "Z"]], "coefficient": 1.5}]},
                }
            ],
        }
        op = loads(json.dumps(legacy)).observables[0][1]
        assert op.terms == {((0, "Z"),): 1.5}

    def test_malformed_complex_coefficient_raises(self) -> None:
        """A list coefficient that is not a [real, imag] pair is rejected."""
        bad = {
            "qsal_version": "0.1.0",
            "circuit": None,
            "observables": [
                {
                    "name": "H",
                    "operator": {
                        "terms": [{"paulis": [[0, "Z"]], "coefficient": [1.0]}]
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match=r"\[real, imag\]"):
            loads(json.dumps(bad))

    def test_no_observables(self) -> None:
        """has_observables returns False when none are defined."""
        doc = QSALDocument(name="empty")
        assert doc.has_observables is False

        json_str = dumps(QuantumCircuit(name="t"))
        doc2 = loads(json_str)
        assert doc2.has_observables is False


# =============================================================================
# Metadata and config tests
# =============================================================================


class TestMetadata:
    """Tests for metadata and configuration fields."""

    def test_parameters_dict(self) -> None:
        """The parameters dict is preserved."""
        doc = QSALDocument(
            name="test",
            parameters={"names": ["θ", "φ"], "bindings": {"θ": 0.5}},
        )
        json_str = doc.to_json()
        doc2 = QSALDocument.from_json(json_str)
        assert doc2.parameters == {"names": ["θ", "φ"], "bindings": {"θ": 0.5}}

    def test_config_dict(self) -> None:
        """The config dict is preserved."""
        doc = QSALDocument(
            name="test",
            config={"shots": 1024, "backend": "statevector"},
        )
        json_str = doc.to_json()
        doc2 = QSALDocument.from_json(json_str)
        assert doc2.config == {"shots": 1024, "backend": "statevector"}

    def test_metadata_dict(self) -> None:
        """The metadata dict is preserved."""
        doc = QSALDocument(
            name="test",
            metadata={
                "description": "VQE ansatz",
                "tags": ["vqe", "chemistry"],
            },
        )
        json_str = doc.to_json()
        doc2 = QSALDocument.from_json(json_str)
        assert doc2.metadata["description"] == "VQE ansatz"
        assert doc2.metadata["tags"] == ["vqe", "chemistry"]


# =============================================================================
# File I/O tests
# =============================================================================


class TestFileIO:
    """Tests for QSAL file read/write."""

    def test_write_and_read_file(self) -> None:
        """A document written to disk can be read back."""
        circ = QuantumCircuit(name="file_test")
        circ.h(0).cx(0, 1)

        doc = QSALDocument(name="file_test", circuit=circ)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.qsal.json"
            doc.to_file(path)

            assert path.exists()
            doc2 = QSALDocument.from_file(path)
            assert doc2.name == "file_test"
            assert doc2.circuit.num_qubits == 2

    def test_dump_and_load_convenience(self) -> None:
        """The dump/load convenience functions work correctly."""
        circ = QuantumCircuit(name="dump_test")
        circ.h(0).x(1).measure_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dump.qsal.json"
            dump(circ, path)
            doc = load(path)
            assert doc.name == "dump_test"
            assert doc.circuit.num_qubits == 2

    def test_dump_with_observables(self) -> None:
        """dump with observables preserves them."""
        from quspect.operators.qubit_operator import QubitOperator

        h_op = QubitOperator({((0, "X"),): 1.0})
        circ = QuantumCircuit(name="obs_test")
        circ.h(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "obs.qsal.json"
            dump(circ, path, observables=[("H", h_op)])
            doc = load(path)
            assert doc.has_observables is True
            assert doc.observables[0][0] == "H"


# =============================================================================
# Version compatibility tests
# =============================================================================


class TestVersionCompatibility:
    """Tests for QSAL version checking."""

    def test_current_version_in_json(self) -> None:
        """Serialized documents include the version."""
        json_str = dumps(QuantumCircuit(name="t"))
        data = json.loads(json_str)
        assert data["qsal_version"] == "0.1.0"

    def test_unknown_version_raises(self) -> None:
        """An unknown version raises QSALVersionError."""
        data = json.dumps({"qsal_version": "99.0.0", "name": "test", "circuit": None})
        with pytest.raises(QSALVersionError) as exc_info:
            QSALDocument.from_json(data)
        assert "99.0.0" in str(exc_info.value)

    def test_missing_version_raises(self) -> None:
        """A missing version field raises QSALVersionError."""
        data = json.dumps({"name": "test", "circuit": None})
        with pytest.raises(QSALVersionError):
            QSALDocument.from_json(data)


# =============================================================================
# Compact JSON test
# =============================================================================


class TestCompactJSON:
    """Tests for compact (non-indented) JSON output."""

    def test_compact_no_indent(self) -> None:
        """indent=None produces compact JSON."""
        circ = QuantumCircuit(name="compact")
        circ.h(0)

        doc = QSALDocument(name="compact", circuit=circ)
        json_str = doc.to_json(indent=None)
        # Compact JSON should have no line breaks in key areas
        assert "\n" not in json_str
        assert json_str.startswith('{"qsal_version"')


# =============================================================================
# Integration tests
# =============================================================================


class TestQSALIntegration:
    """End-to-end QSAL integration tests."""

    def test_vqe_full_workflow(self) -> None:
        """A full VQE-like workflow: circuit + observables + parameters."""
        from quspect.operators.qubit_operator import QubitOperator

        theta = Parameter("θ")
        phi = Parameter("φ")

        circ = QuantumCircuit(name="vqe_ansatz")
        circ.ry(theta, 0).ry(theta, 1)
        circ.cx(0, 1)
        circ.rz(phi, 0).rz(phi, 1)
        circ.cx(0, 1)

        h_op = QubitOperator(
            {
                ((0, "Z"), (1, "Z")): 0.5,
                ((0, "X"),): 0.3,
                ((1, "X"),): 0.3,
            }
        )

        # Serialize
        json_str = dumps(
            circ,
            observables=[("energy", h_op)],
            name="vqe_h2",
            parameters={"names": ["θ", "φ"]},
            config={"shots": 2048},
            metadata={"description": "VQE for H2"},
        )

        # Deserialize
        doc = loads(json_str)
        assert doc.name == "vqe_h2"
        assert doc.is_parameterized is True
        assert doc.has_observables is True
        assert doc.parameters == {"names": ["θ", "φ"]}
        assert doc.config == {"shots": 2048}

        # Verify circuit structure
        circ2 = doc.circuit
        assert circ2.num_qubits == 2
        assert len(circ2) == 6  # ry, ry, cx, rz, rz, cx

        # Verify observable
        obs_name, obs_op = doc.observables[0]
        assert obs_name == "energy"
        assert len(obs_op.terms) == 3

        # Bind parameters post-deserialization
        params_set = circ2.parameters
        binding = {p: 0.5 if p.name == "θ" else 1.0 for p in params_set}
        bound_circ = circ2.bind_parameters(binding)
        assert bound_circ.parameters == set()

        # Export bound circuit to QASM
        from quspect.ir import QASM2Exporter

        qasm = QASM2Exporter().export(bound_circ)
        assert "ry(0.5)" in qasm
        assert "rz(1)" in qasm

    def test_complex_circuit_roundtrip(self) -> None:
        """A complex circuit with mix of gates survives round-trip."""
        circ = QuantumCircuit(name="complex")
        circ.h(0).h(1).h(2)
        circ.cx(0, 1)
        circ.cz(1, 2)
        circ.swap(0, 2)
        circ.measure_all()
        circ.barrier()

        json_str = dumps(circ)
        doc = loads(json_str)

        circ2 = doc.circuit
        assert circ2.num_qubits == 3
        assert circ2.num_clbits == 3
        assert len(circ2) == 8

    def test_h_gate_roundtrip_edge_cases(self) -> None:
        """All standard gates survive round-trip."""
        circ = QuantumCircuit(name="all_gates")
        circ.h(0).x(1).y(2).z(3).s(0).sdg(1).t(2).tdg(3)
        circ.cx(0, 1).cz(2, 3).swap(0, 2).ccx(0, 1, 2)

        json_str = dumps(circ)
        doc = loads(json_str)

        cir2 = doc.circuit
        instrs = list(cir2)
        gate_names = [i.gate.name for i in instrs if i.is_gate]
        assert "h" in gate_names
        assert "ccx" in gate_names
        assert "swap" in gate_names
