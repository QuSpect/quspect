"""Tests for the QSAL serializer (:mod:`quspect.ir.qsal`) and IR errors.

Round trips are checked instruction by instruction (including symbolic
parameters, every noise-channel type, measurements and barriers).  The C++
decoder (``quspect.native.decode_qsal`` / ``NativeExecutor``) is used as an
independent consumer of the same JSON, and its statevectors / expectation
values are compared with qiskit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

import quspect
from qiskit_tests._ir_helpers import (
    all_channel_examples,
    assert_circuits_identical,
    channel_signature,
    random_rich_circuit,
)
from qiskit_tests.helpers import assert_states_equal, random_circuit_pair
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.gates.gate import Gate
from quspect.ir import (
    IRDeserializationError,
    IRSerializationError,
    QSALDocument,
    QSALVersionError,
    dump,
    dumps,
    load,
    loads,
)
from quspect.ir.qsal import CURRENT_QSAL_VERSION
from quspect.native import NativeExecutor, decode_qsal
from quspect.noise.channel import NoiseChannel
from quspect.operators.qubit_operator import QubitOperator


def _doc_json(circuit_data: dict[str, Any] | None = None, **extra: Any) -> str:
    """Minimal hand-written QSAL document."""
    data: dict[str, Any] = {"qsal_version": CURRENT_QSAL_VERSION, "name": "x"}
    data["circuit"] = circuit_data
    data.update(extra)
    return json.dumps(data)


def _one_instruction(instr: dict[str, Any]) -> str:
    """QSAL JSON with a single raw instruction dict."""
    return _doc_json({"num_qubits": 1, "num_clbits": 0, "instructions": [instr]})


# =============================================================================
# Round trips
# =============================================================================


@pytest.mark.parametrize("seed", range(10))
def test_random_circuit_roundtrip_dumps_loads(seed: int) -> None:
    """dumps -> loads reproduces every instruction, parameter and channel."""
    circ = random_rich_circuit(3, 60, seed)
    doc = loads(dumps(circ))
    assert doc.circuit is not None
    assert_circuits_identical(doc.circuit, circ)
    assert doc.circuit.parameters == circ.parameters
    assert doc.is_parameterized == bool(circ.parameters)
    # Serialisation is a fixed point.
    assert dumps(doc.circuit, name=doc.name) == dumps(circ)


def test_every_channel_type_roundtrips() -> None:
    """Each noise channel class keeps its type, fields and Kraus operators."""
    circ = quspect.QuantumCircuit()
    for ch, nq in all_channel_examples():
        circ.append_channel(ch, list(range(nq)))
    back = loads(dumps(circ)).circuit
    assert back is not None
    for a, b in zip(circ.instructions, back.instructions, strict=True):
        assert type(a.channel) is type(b.channel)
        assert channel_signature(a.channel) == channel_signature(b.channel)


def test_parameter_and_expression_values_roundtrip() -> None:
    """Parameter / ParameterExpression / float params keep type and value."""
    theta, phi = Parameter("theta"), Parameter("phi")
    expr = 2.5 * theta - 0.5 * phi + 0.125
    circ = quspect.QuantumCircuit()
    circ.u3(theta, expr, -1.75, 0).crz(phi, 0, 1)
    back = loads(dumps(circ)).circuit
    assert back is not None
    params = back[0].gate.params if back[0].gate else ()
    assert isinstance(params[0], Parameter)
    assert isinstance(params[1], ParameterExpression)
    assert params[1] == expr
    assert params[2] == -1.75
    assert back.bind_parameters({theta: 0.3, phi: -0.2}).instructions == (
        circ.bind_parameters({theta: 0.3, phi: -0.2}).instructions
    )


def test_measure_barrier_and_label_roundtrip() -> None:
    """Measurement clbits, barrier qubits and instruction labels survive."""
    circ = quspect.QuantumCircuit()
    circ.h(0).barrier().barrier(1, 2).measure([2, 0], [0, 3])
    circ._instructions.append(  # instruction labels are not in the fluent API
        CircuitInstruction(
            type=InstructionType.GATE,
            gate=Gate(name="x", num_qubits=1),
            qubits=(1,),
            label="flip",
        )
    )
    back = loads(dumps(circ)).circuit
    assert back is not None
    assert_circuits_identical(back, circ)
    assert back[-1].label == "flip"
    assert back.num_clbits == 4


def test_file_roundtrip_and_document_api(tmp_path: Path) -> None:
    """dump/load, to_file/from_file and to_json(indent=None) all agree."""
    circ = random_rich_circuit(2, 20, 3)
    obs = QubitOperator({((0, "Z"), (1, "X")): 0.5, (): -1.0})
    path = tmp_path / "prog.qsal.json"
    dump(circ, path, observables=[("H", obs)], name="prog", config={"shots": 10})
    text = path.read_text()
    assert text.endswith("\n")
    doc = load(str(path))
    assert doc.name == "prog"
    assert doc.config == {"shots": 10}
    assert doc.has_observables
    assert doc.observables[0][0] == "H"
    assert doc.observables[0][1].terms == obs.terms
    assert doc.circuit is not None
    assert_circuits_identical(doc.circuit, circ)

    compact = doc.to_json(indent=None)
    assert "\n" not in compact
    assert QSALDocument.from_json(compact).to_json() == doc.to_json()

    path2 = tmp_path / "doc.qsal.json"
    doc.to_file(path2)
    again = QSALDocument.from_file(path2)
    assert again.to_json() == doc.to_json()


def test_dumps_name_metadata_and_kwargs() -> None:
    """Name defaults to the circuit name; extra kwargs go to metadata."""
    circ = quspect.QuantumCircuit(name="bell")
    circ.h(0).cx(0, 1)
    doc = loads(
        dumps(
            circ,
            metadata={"author": "me"},
            parameters={"theta": {"init": 0.1}},
            config={"backend": "sv"},
            tag="v1",
        )
    )
    assert doc.name == "bell"
    assert doc.metadata == {"author": "me", "tag": "v1"}
    assert doc.parameters == {"theta": {"init": 0.1}}
    assert doc.config == {"backend": "sv"}
    assert doc.qsal_version == CURRENT_QSAL_VERSION
    assert not doc.has_observables
    assert not doc.is_parameterized
    assert loads(dumps(circ, name="override")).name == "override"
    assert loads(dumps(quspect.QuantumCircuit().h(0))).name == ""


def test_document_without_circuit() -> None:
    """A document with ``circuit=None`` serializes ``null`` and loads back."""
    doc = QSALDocument(name="empty")
    assert json.loads(doc.to_json())["circuit"] is None
    back = QSALDocument.from_json(doc.to_json())
    assert back.circuit is None
    assert back.is_parameterized is False
    assert back.has_observables is False


def test_observable_symbolic_coefficients_roundtrip() -> None:
    """Parameter and ParameterExpression coefficients survive a round trip."""
    t = Parameter("t")
    op = QubitOperator({((0, "X"),): t, ((1, "Z"),): 2 * t + 1, ((0, "Y"),): 3.0})
    circ = quspect.QuantumCircuit()
    circ.h(0)
    doc = loads(dumps(circ, observables=[("H", op)]))
    assert doc.observables[0][1].terms == op.terms


def test_observable_complex_coefficient_roundtrip() -> None:
    """Complex Pauli coefficients must survive serialization."""
    op = QubitOperator({((0, "Z"),): 0.5 + 0.5j, ((1, "X"),): 1j})
    circ = quspect.QuantumCircuit()
    circ.h(0)
    doc = loads(dumps(circ, observables=[("H", op)]))
    assert doc.observables[0][1].terms == op.terms


def test_circuit_name_roundtrip() -> None:
    """``QuantumCircuit.name`` survives a QSAL round trip."""
    circ = quspect.QuantumCircuit(name="ansatz")
    circ.h(0)
    doc = loads(dumps(circ, name="document"))
    assert doc.circuit is not None
    assert doc.circuit.name == "ansatz"


def test_gate_label_roundtrip() -> None:
    """A custom ``Gate.label`` survives a QSAL round trip."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="rx", num_qubits=1, params=(0.3,), label="drive"), (0,))
    back = loads(dumps(circ)).circuit
    assert back is not None
    assert back[0].gate is not None
    assert back[0].gate.label == "drive"


# =============================================================================
# Backward-compatible / hand-written channel payloads
# =============================================================================


@pytest.mark.parametrize(
    ("payload", "attrs"),
    [
        ({"channel_type": "bit_flip", "num_qubits": 1, "params": [0.2]}, {"p": 0.2}),
        ({"channel_type": "phase_flip", "num_qubits": 1, "params": [0.3]}, {"p": 0.3}),
        (
            {"channel_type": "bit_phase_flip", "num_qubits": 1, "params": [0.1]},
            {"p": 0.1},
        ),
        ({"channel_type": "depolarizing", "num_qubits": 1}, {"p": 0.0}),
        (
            {"channel_type": "amplitude_damping", "num_qubits": 1, "params": [0.4]},
            {"gamma": 0.4},
        ),
        (
            {"channel_type": "phase_damping", "num_qubits": 1, "params": [0.5]},
            {"gamma": 0.5},
        ),
        (
            {"channel_type": "gen_amp_damp", "num_qubits": 1, "params": [0.2]},
            {"gamma": 0.2, "p_thermal": 0.0},
        ),
        ({"channel_type": "pauli", "num_qubits": 1, "px": 0.1}, {"px": 0.1, "pz": 0}),
        (
            {"channel_type": "kraus", "num_qubits": 1, "kraus_ops": [[[1, 0], [0, 1]]]},
            {"name": "kraus"},
        ),
    ],
)
def test_legacy_channel_payloads(
    payload: dict[str, Any], attrs: dict[str, Any]
) -> None:
    """Payloads without ``channel_kind`` (or with only ``params``) still load."""
    doc = loads(
        _one_instruction({"type": "CHANNEL", "qubits": [0], "channel": payload})
    )
    assert doc.circuit is not None
    ch = doc.circuit[0].channel
    assert ch is not None
    for key, value in attrs.items():
        assert getattr(ch, key) == value
    total = sum(k.conj().T @ k for k in ch.kraus_operators())
    assert np.allclose(total, np.eye(2))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"channel_type": "warp", "num_qubits": 1}, "Unknown channel type"),
        ({"channel_type": "kraus", "num_qubits": 1}, "missing kraus_ops"),
        (
            {
                "channel_type": "kraus",
                "num_qubits": 1,
                "kraus_ops": [[["a", 0], [0, 1]]],
            },
            "real, imag",
        ),
    ],
)
def test_bad_channel_payloads_raise(payload: dict[str, Any], match: str) -> None:
    """Unknown channel kinds and malformed Kraus data raise ValueError."""
    with pytest.raises(ValueError, match=match):
        loads(_one_instruction({"type": "CHANNEL", "qubits": [0], "channel": payload}))


def test_unsupported_channel_class_raises_on_serialize() -> None:
    """A NoiseChannel subclass unknown to QSAL raises TypeError."""

    class Custom(NoiseChannel):
        def kraus_operators(self) -> list[np.ndarray]:
            return [np.eye(2, dtype=complex)]

    circ = quspect.QuantumCircuit()
    circ.append_channel(Custom(name="custom", num_qubits=1), [0])
    with pytest.raises(TypeError, match="Unsupported noise channel type: Custom"):
        dumps(circ)


# =============================================================================
# Errors
# =============================================================================


@pytest.mark.parametrize("version", ["9.9.9", "0.0.1", None])
def test_version_error(version: str | None) -> None:
    """Unknown or missing versions raise QSALVersionError with details."""
    data: dict[str, Any] = {"name": "x"}
    if version is not None:
        data["qsal_version"] = version
    with pytest.raises(QSALVersionError) as info:
        loads(json.dumps(data))
    assert info.value.found_version == (version or "unknown")
    assert info.value.supported_versions == [CURRENT_QSAL_VERSION]
    assert isinstance(info.value, IRDeserializationError)
    assert repr(version or "unknown") in str(info.value)


def test_unknown_standard_gate_raises() -> None:
    """A ``standard`` gate name unknown to quspect raises ValueError."""
    payload = {
        "type": "GATE",
        "qubits": [0],
        "gate": {"name": "warp", "num_qubits": 1, "gate_type": "standard"},
    }
    with pytest.raises(ValueError, match="Unknown standard gate: 'warp'"):
        loads(_one_instruction(payload))


def test_malformed_json_raises_value_error() -> None:
    """Invalid JSON surfaces as ``ValueError`` (``json.JSONDecodeError``)."""
    with pytest.raises(ValueError):
        loads("{not json")


def test_error_hierarchy() -> None:
    """Serialization and deserialization errors have separate bases."""
    assert not issubclass(IRSerializationError, IRDeserializationError)
    assert issubclass(QSALVersionError, IRDeserializationError)


def test_unbound_parameters_are_preserved_not_rejected() -> None:
    """QSAL (unlike QASM) accepts unbound parameters and reports them."""
    t = Parameter("t")
    circ = quspect.QuantumCircuit()
    circ.rx(t, 0)
    doc = loads(dumps(circ))
    assert doc.is_parameterized
    assert doc.circuit is not None
    assert doc.circuit.parameters == {t}


# =============================================================================
# Native C++ decoder cross-check
# =============================================================================


@pytest.mark.parametrize("seed", range(6))
def test_native_decoder_reads_python_qsal(seed: int) -> None:
    """``decode_qsal`` sees the same widths and parameter names."""
    circ = random_rich_circuit(3, 40, seed)
    circ.measure([0, 1, 2], [2, 1, 0])
    prog = decode_qsal(dumps(circ))
    assert prog.num_qubits == circ.num_qubits
    assert prog.num_clbits == circ.num_clbits
    assert set(prog.parameter_names) == {p.name for p in circ.parameters}
    # Decoding the re-serialized document gives the same program summary.
    back = loads(dumps(circ)).circuit
    assert back is not None
    prog2 = decode_qsal(dumps(back))
    assert list(prog2.parameter_names) == list(prog.parameter_names)


def test_native_decoder_observables() -> None:
    """Observable terms (paulis, coefficients, labels) decode identically."""
    op = QubitOperator({((0, "Z"), (1, "X")): 0.5, ((1, "Y"),): -1.0, (): 0.25})
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1)
    prog = decode_qsal(dumps(circ, observables=[("H", op)]))
    decoded = {
        tuple(term.paulis): complex(term.coefficient) for term in prog.observables
    }
    assert {t.label for t in prog.observables} == {"H"}
    assert decoded == {k: complex(v) for k, v in op.terms.items()}


def test_native_decoder_rejects_bad_version_and_symbolic_coefficient() -> None:
    """The C++ decoder rejects unknown versions and symbolic coefficients."""
    with pytest.raises(RuntimeError, match="version"):
        decode_qsal(json.dumps({"qsal_version": "9.9"}))
    circ = quspect.QuantumCircuit()
    circ.h(0)
    op = QubitOperator({((0, "Z"),): Parameter("t")})
    with pytest.raises(RuntimeError, match="non-numeric coefficient"):
        decode_qsal(dumps(circ, observables=[("P", op)]))


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(8))
def test_native_statevector_matches_qiskit(seed: int) -> None:
    """Native execution of python-emitted QSAL reproduces qiskit's statevector."""
    circ, qc = random_circuit_pair(4, 30, seed)
    result = NativeExecutor(seed=seed).run(dumps(circ), return_statevector=True)
    assert_states_equal(
        np.asarray(result.statevector_f64), Statevector(qc).data, up_to_phase=False
    )
    roundtrip = loads(dumps(circ)).circuit
    assert roundtrip is not None
    result2 = NativeExecutor().run(dumps(roundtrip), return_statevector=True)
    assert np.allclose(result2.statevector_f64, result.statevector_f64, atol=1e-12)


def _native_vs_qiskit_expectation(seed: int, *, max_flips: int) -> tuple[float, float]:
    """Random 4-term observable on a random circuit: (native, qiskit) values.

    ``max_flips`` bounds the number of X/Y factors in each Pauli term.
    """
    circ, qc = random_circuit_pair(3, 20, seed)
    rng = np.random.default_rng(seed)
    terms: dict[tuple[tuple[int, str], ...], float] = {}
    sparse: list[tuple[str, list[int], float]] = []
    for _ in range(4):
        qubits = sorted(int(q) for q in rng.permutation(3)[: int(rng.integers(1, 4))])
        paulis = [str(rng.choice(["X", "Y", "Z"])) for _ in qubits]
        flips = [i for i, ch in enumerate(paulis) if ch != "Z"]
        for i in flips[max_flips:]:
            paulis[i] = "Z"
        coef = float(rng.uniform(-1, 1))
        key = tuple(zip(qubits, paulis, strict=True))
        terms[key] = terms.get(key, 0.0) + coef
        sparse.append(("".join(paulis), qubits, coef))
    op = QubitOperator(terms)
    result = NativeExecutor().run(
        dumps(circ, observables=[("O", op)]), return_expectations=True
    )
    ref = SparsePauliOp.from_sparse_list(sparse, num_qubits=3)
    expected = float(Statevector(qc).expectation_value(ref).real)
    return result.expectations_real["O"], expected


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(6))
def test_native_expectation_single_flip_terms_match_qiskit(seed: int) -> None:
    """Native expectations of Paulis with <= 1 X/Y factor agree with qiskit."""
    native, expected = _native_vs_qiskit_expectation(seed, max_flips=1)
    assert math.isclose(native, expected, abs_tol=1e-9)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", [0, 1])
def test_native_expectation_multi_flip_terms_match_qiskit(seed: int) -> None:
    """Native expectations of general Pauli sums agree with qiskit."""
    native, expected = _native_vs_qiskit_expectation(seed, max_flips=3)
    assert math.isclose(native, expected, abs_tol=1e-9)


@pytest.mark.qiskit_ref
def test_native_expectation_xx_on_plus_plus() -> None:
    """Minimal reproduction: ``<++|XX|++> = 1`` and ``<++|YY|++> = 0``."""
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1)
    xx = QubitOperator({((0, "X"), (1, "X")): 1.0})
    yy = QubitOperator({((0, "Y"), (1, "Y")): 1.0})
    out = NativeExecutor().run(
        dumps(circ, observables=[("XX", xx), ("YY", yy)]), return_expectations=True
    )
    assert math.isclose(out.expectations_real["XX"], 1.0, abs_tol=1e-9)
    assert math.isclose(out.expectations_real["YY"], 0.0, abs_tol=1e-9)
