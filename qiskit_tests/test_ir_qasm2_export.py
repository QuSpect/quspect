"""Cross-validate :class:`quspect.ir.QASM2Exporter` against qiskit's QASM 2 parser.

Every quspect circuit is exported to OpenQASM 2.0, re-imported with
``qiskit.qasm2.loads`` and compared with the reference qiskit circuit built
independently by :func:`qiskit_tests.helpers.random_circuit_pair` /
:func:`qiskit_tests.helpers.to_qiskit`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import qiskit.qasm2

import quspect
from qiskit_tests._ir_helpers import (
    qiskit_loads,
    qiskit_measure_map,
    quspect_measure_map,
    strip_measurements,
    unitary_only,
)
from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_unitary_equal,
    qiskit_circuit_unitary,
    quspect_circuit_unitary,
    random_circuit_pair,
    to_qiskit,
)
from quspect.circuit.parameter import Parameter
from quspect.gates.gate import Gate
from quspect.gates.parameterized import RZ
from quspect.gates.standard import CX, H
from quspect.ir import (
    GateDecompositionRule,
    IRSerializationError,
    ParameterizedGateError,
    QASM2ExportConfig,
    QASM2Exporter,
    QASM2ExportMode,
    UnsupportedGateError,
    clear_extra_rules,
    qasm2_loads,
    register_decomposition,
)
from quspect.ir.qasm2 import _format_float, get_supported_gates

GATE_PARAMS: tuple[float, ...] = (0.37, -1.21, 2.53)


@pytest.fixture(autouse=True)
def _clean_rules() -> Iterator[None]:
    """Reset the global decomposition-rule registry around every test."""
    clear_extra_rules()
    yield
    clear_extra_rules()


def _export(circ: quspect.QuantumCircuit, **kwargs: object) -> str:
    """Export ``circ`` with a :class:`QASM2ExportConfig` built from kwargs."""
    return QASM2Exporter(QASM2ExportConfig(**kwargs)).export(circ)  # type: ignore[arg-type]


# =============================================================================
# Export -> qiskit import: unitary and measurement equivalence
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(GATE_SPECS))
def test_each_gate_exports_to_identical_qiskit_unitary(name: str) -> None:
    """Every supported gate exports to QASM that qiskit reads with no phase error."""
    spec = GATE_SPECS[name]
    circ = quspect.QuantumCircuit()
    qubits = list(range(spec.num_qubits))[::-1]  # reversed qubits exercise order
    spec.apply_quspect(circ, GATE_PARAMS[: spec.num_params], qubits)
    qc = qiskit_loads(QASM2Exporter().export(circ))
    assert qc.num_qubits == spec.num_qubits
    assert_unitary_equal(
        qiskit_circuit_unitary(qc), qiskit_circuit_unitary(circ), atol=1e-8
    )


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(12))
def test_random_circuit_export_matches_qiskit(seed: int) -> None:
    """Random 4-qubit circuits survive export -> qiskit.qasm2.loads exactly."""
    circ, qc_ref = random_circuit_pair(4, 30, seed)
    qc = qiskit_loads(QASM2Exporter().export(circ))
    # Exact equality is expected (all qelib1 definitions match quspect's
    # matrices); the 10-significant-digit float formatting limits atol.
    assert_unitary_equal(
        qiskit_circuit_unitary(qc), qiskit_circuit_unitary(qc_ref), atol=1e-8
    )


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(6))
def test_export_measurement_mapping_matches_qiskit(seed: int) -> None:
    """``measure q[i] -> c[j]`` pairs and register sizes survive the export."""
    circ, _ = random_circuit_pair(3, 12, seed)
    rng = np.random.default_rng(seed)
    clbits = [int(c) for c in rng.permutation(5)[:3]]
    circ.barrier()
    circ.measure([0, 1, 2], clbits)
    qc = qiskit_loads(QASM2Exporter().export(circ))
    assert qc.num_qubits == 3
    assert qc.num_clbits == max(clbits) + 1
    assert qiskit_measure_map(qc) == quspect_measure_map(circ)
    assert qiskit_measure_map(qc) == qiskit_measure_map(to_qiskit(circ))
    assert_unitary_equal(
        qiskit_circuit_unitary(strip_measurements(qc)),
        qiskit_circuit_unitary(strip_measurements(to_qiskit(circ))),
        atol=1e-8,
    )


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(6))
def test_export_then_quspect_import_roundtrip(seed: int) -> None:
    """quspect export -> quspect import preserves the unitary and measurements."""
    circ, _ = random_circuit_pair(3, 25, seed)
    circ.measure([2, 0], [0, 1])
    back = qasm2_loads(QASM2Exporter().export(circ))
    assert quspect_measure_map(back) == quspect_measure_map(circ)
    assert_unitary_equal(
        quspect_circuit_unitary(unitary_only(back), num_qubits=3),
        quspect_circuit_unitary(unitary_only(circ), num_qubits=3),
        atol=1e-8,
    )


@pytest.mark.qiskit_ref
def test_strict_output_needs_extended_qelib1_only_for_swap_crx_cry() -> None:
    """Only ``swap``/``crx``/``cry``/``sx``/``sxdg`` are outside spec-strict qelib1.

    (``sx``/``sxdg`` are in the qelib1.inc shipped by Qiskit and QASMBench, not
    in the one of the OpenQASM 2.0 paper.)

    quspect emits them directly in STRICT mode; qiskit only accepts them with
    ``LEGACY_CUSTOM_INSTRUCTIONS`` (the extended qelib1 shipped by qiskit).
    """
    needs_legacy = set()
    for name, spec in GATE_SPECS.items():
        circ = quspect.QuantumCircuit()
        spec.apply_quspect(circ, GATE_PARAMS[: spec.num_params], range(spec.num_qubits))
        text = QASM2Exporter().export(circ)
        try:
            qiskit.qasm2.loads(text)
        except qiskit.qasm2.QASM2ParseError:
            needs_legacy.add(name)
        qiskit_loads(text)  # always fine with the legacy instruction set
    assert needs_legacy == {"swap", "crx", "cry", "sx", "sxdg"}


# =============================================================================
# Text layout and config branches
# =============================================================================


def test_default_export_text_layout() -> None:
    """Default STRICT export emits header, registers, blank line, statements."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 2).p(0.5, 1).cp(-0.25, 1, 0).barrier().measure([0, 2], [1, 0])
    assert QASM2Exporter().export(circ).splitlines() == [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[3];",
        "creg c[2];",
        "",
        "h q[0];",
        "cx q[0],q[2];",
        "u1(0.5) q[1];",
        "cu1(-0.25) q[1],q[0];",
        "barrier q[0],q[1],q[2];",
        "measure q[0] -> c[1];",
        "measure q[2] -> c[0];",
    ]


def test_config_defaults() -> None:
    """``QASM2ExportConfig()`` defaults match the documented values."""
    cfg = QASM2ExportConfig()
    assert cfg.mode is QASM2ExportMode.STRICT
    assert cfg.include_qelib is True
    assert cfg.header_comment == ""
    assert cfg.indent == ""
    assert cfg.fold_constants is True
    assert cfg.barrier_as_comment is False
    assert {m.name for m in QASM2ExportMode} == {"STRICT", "DECOMPOSE", "EXTENDED"}


def test_config_header_indent_qelib_and_barrier_comment() -> None:
    """header_comment / indent / include_qelib / barrier_as_comment all apply."""
    circ = quspect.QuantumCircuit()
    circ.h(0).barrier(0, 1).measure([0, 1], [0, 1])
    text = _export(
        circ,
        header_comment="line one\nline two",
        indent="  ",
        include_qelib=False,
        barrier_as_comment=True,
        fold_constants=False,
    )
    assert text.splitlines() == [
        "// line one",
        "// line two",
        "OPENQASM 2.0;",
        "qreg q[2];",
        "creg c[2];",
        "",
        "  h q[0];",
        "  // barrier q[0],q[1];",
        "  measure q[0] -> c[0];",
        "  measure q[1] -> c[1];",
    ]


@pytest.mark.qiskit_ref
def test_header_comment_and_barrier_comment_still_parse_in_qiskit() -> None:
    """Comment-only config changes keep the output valid for qiskit."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1).barrier().measure([0, 1], [0, 1])
    text = _export(circ, header_comment="generated", barrier_as_comment=True)
    qc = qiskit_loads(text)
    assert [ci.operation.name for ci in qc.data] == ["h", "cx", "measure", "measure"]


def test_export_to_file(tmp_path: Path) -> None:
    """``export_to_file`` writes the export string plus a trailing newline."""
    circ = quspect.QuantumCircuit()
    circ.h(0)
    out = tmp_path / "c.qasm"
    exporter = QASM2Exporter()
    exporter.export_to_file(circ, str(out))
    assert out.read_text() == exporter.export(circ) + "\n"


def test_empty_circuit_raises() -> None:
    """Exporting an empty circuit raises ``ValueError``."""
    with pytest.raises(ValueError, match="empty"):
        QASM2Exporter().export(quspect.QuantumCircuit())


def test_noise_channel_is_emitted_as_comment() -> None:
    """Noise channels have no QASM form and are emitted as a comment line."""
    circ = quspect.QuantumCircuit()
    circ.h(0).depolarize(0.1, 0).x(0)
    lines = QASM2Exporter().export(circ).splitlines()
    assert lines[-3:] == [
        "h q[0];",
        "// unknown instruction type: InstructionType.CHANNEL",
        "x q[0];",
    ]


@pytest.mark.qiskit_ref
def test_qubitless_barrier_export() -> None:
    """A barrier with no qubits becomes ``barrier ;``; qiskit reads a full barrier."""
    circ = quspect.QuantumCircuit()
    circ.barrier()
    text = QASM2Exporter().export(circ)
    assert text.splitlines()[-1] == "barrier ;"
    assert "qreg q[1];" in text
    qc = qiskit_loads(text)
    assert [ci.operation.name for ci in qc.data] == ["barrier"]


def test_float_parameters_roundtrip_exactly() -> None:
    """Exported angles should be bit-exact after re-import."""
    theta = 0.12345678901234567
    circ = quspect.QuantumCircuit()
    circ.rx(theta, 0)
    back = qasm2_loads(QASM2Exporter().export(circ))
    assert back[0].gate is not None
    assert back[0].gate.params == (theta,)


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.0, "0"), (0.5, "0.5"), (-0.25, "-0.25"), (1e-12, "1e-12"), (3, "3")],
)
def test_format_float(value: float, text: str) -> None:
    """Float formatting is compact and keeps the sign."""
    assert _format_float(value) == text


def test_supported_gates_registry() -> None:
    """``get_supported_gates`` is exactly the 25 quspect gate names."""
    assert get_supported_gates() == set(GATE_SPECS)


# =============================================================================
# Symbolic parameters and unsupported gates
# =============================================================================


def test_symbolic_parameter_raises_parameterized_gate_error() -> None:
    """Unbound Parameters / expressions raise ParameterizedGateError."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circ = quspect.QuantumCircuit()
    circ.rx(theta, 0)
    with pytest.raises(ParameterizedGateError) as info:
        QASM2Exporter().export(circ)
    assert info.value.gate_name == "rx"
    assert info.value.param_names == ["theta"]
    assert "bind_parameters" in str(info.value)
    assert isinstance(info.value, IRSerializationError)

    circ2 = quspect.QuantumCircuit()
    circ2.u3(0.1, 2 * theta + phi, 0.3, 0)
    with pytest.raises(ParameterizedGateError) as info2:
        QASM2Exporter().export(circ2)
    assert sorted(info2.value.param_names) == ["phi", "theta"]


@pytest.mark.qiskit_ref
def test_bound_parameters_export_like_floats() -> None:
    """After ``bind_parameters`` the export equals the numeric circuit's."""
    theta = Parameter("theta")
    circ = quspect.QuantumCircuit()
    circ.rx(theta, 0).crz(2 * theta + 0.1, 0, 1)
    bound = circ.bind_parameters({theta: 0.4})
    ref = quspect.QuantumCircuit()
    ref.rx(0.4, 0).crz(0.9, 0, 1)
    qc = qiskit_loads(QASM2Exporter().export(bound))
    assert_unitary_equal(
        qiskit_circuit_unitary(qc), qiskit_circuit_unitary(ref), atol=1e-9
    )


def test_parameterized_gate_error_custom_hint() -> None:
    """A custom hint overrides the default message."""
    err = ParameterizedGateError("rx", ["a"], hint="custom")
    assert str(err) == "custom"
    assert err.hint == "custom"


def test_unsupported_gate_error_default_suggestion() -> None:
    """The default suggestion mentions DECOMPOSE and EXTENDED."""
    err = UnsupportedGateError("foo", "OpenQASM 2.0")
    assert err.gate_name == "foo"
    assert err.format_name == "OpenQASM 2.0"
    assert "DECOMPOSE" in str(err)
    assert "EXTENDED" in str(err)


@pytest.mark.parametrize("mode", list(QASM2ExportMode))
def test_gate_without_rule_raises_in_every_mode(mode: QASM2ExportMode) -> None:
    """A gate with neither a QASM name nor a rule raises UnsupportedGateError."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="mystery", num_qubits=1), (0,))
    with pytest.raises(UnsupportedGateError) as info:
        _export(circ, mode=mode)
    assert info.value.gate_name == "mystery"
    assert mode.name in info.value.format_name


def _register_zz() -> None:
    """Register ``myzz(t)`` = CX; RZ(t) on target; CX (parameterized)."""

    def rule(gate: Gate, qs: tuple[int, ...]) -> list[tuple[Gate, tuple[int, ...]]]:
        a, b = qs
        return [(CX, (a, b)), (RZ.with_params(gate.params[0]), (b,)), (CX, (a, b))]

    register_decomposition(GateDecompositionRule("myzz", 2, rule))


def _register_bell() -> None:
    """Register the parameter-free ``bell`` = H(a); CX(a, b)."""

    def rule(gate: Gate, qs: tuple[int, ...]) -> list[tuple[Gate, tuple[int, ...]]]:
        return [(H, (qs[0],)), (CX, (qs[0], qs[1]))]

    register_decomposition(GateDecompositionRule("bell", 2, rule))


# =============================================================================
# DECOMPOSE mode
# =============================================================================


@pytest.mark.qiskit_ref
def test_decompose_mode_equals_strict_for_supported_gates() -> None:
    """For qelib1-only circuits DECOMPOSE emits the same text as STRICT."""
    circ, _ = random_circuit_pair(3, 20, 3)
    assert _export(circ, mode=QASM2ExportMode.DECOMPOSE) == _export(circ)


@pytest.mark.qiskit_ref
def test_decompose_mode_expands_custom_gate() -> None:
    """A registered custom gate is expanded into qelib1 gates."""
    _register_zz()
    circ = quspect.QuantumCircuit()
    circ.h(0).h(1)
    circ.append(Gate(name="myzz", num_qubits=2, params=(0.7,)), (1, 0))
    circ.append(Gate(name="myzz", num_qubits=2, params=(-0.3,)), (0, 1))
    circ.measure([0], [0])
    text = _export(circ, mode=QASM2ExportMode.DECOMPOSE)
    assert "myzz" not in text
    qc = qiskit_loads(text)
    ref = qiskit.QuantumCircuit(2)
    ref.h(0)
    ref.h(1)
    ref.rzz(0.7, 1, 0)
    ref.rzz(-0.3, 0, 1)
    assert_unitary_equal(
        qiskit_circuit_unitary(strip_measurements(qc)),
        qiskit_circuit_unitary(ref),
        up_to_phase=True,
        atol=1e-8,
    )
    assert qiskit_measure_map(qc) == [(0, 0)]


def test_decompose_mode_lowers_nested_custom_rules() -> None:
    """DECOMPOSE should lower custom gates recursively into qelib1 gates."""
    _register_bell()

    def rule(gate: Gate, qs: tuple[int, ...]) -> list[tuple[Gate, tuple[int, ...]]]:
        return [(Gate(name="bell", num_qubits=2), qs)]

    register_decomposition(GateDecompositionRule("outer", 2, rule))
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="outer", num_qubits=2), (0, 1))
    text = _export(circ, mode=QASM2ExportMode.DECOMPOSE)
    assert "bell" not in text


# =============================================================================
# EXTENDED mode
# =============================================================================


@pytest.mark.qiskit_ref
def test_extended_mode_declares_parameter_free_custom_gate() -> None:
    """EXTENDED emits one ``gate`` block per custom gate and qiskit accepts it."""
    _register_bell()
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="bell", num_qubits=2), (1, 2))
    circ.append(Gate(name="bell", num_qubits=2), (0, 1))
    circ.x(2).measure([0, 1, 2], [0, 1, 2])
    text = _export(circ, mode=QASM2ExportMode.EXTENDED)
    lines = text.splitlines()
    assert lines[:7] == [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "gate bell a,b {",
        "  h a;",
        "  cx a,b;",
        "}",
        "",
    ]
    assert text.count("gate bell") == 1
    assert "bell q[1],q[2];" in lines
    qc = qiskit_loads(text)
    ref = qiskit.QuantumCircuit(3)
    ref.h(1)
    ref.cx(1, 2)
    ref.h(0)
    ref.cx(0, 1)
    ref.x(2)
    assert_unitary_equal(
        qiskit_circuit_unitary(strip_measurements(qc)), qiskit_circuit_unitary(ref)
    )
    assert qiskit_measure_map(qc) == [(0, 0), (1, 1), (2, 2)]


def test_extended_mode_without_custom_gates_equals_strict() -> None:
    """EXTENDED with only qelib1 gates emits no declarations."""
    circ, _ = random_circuit_pair(2, 10, 1)
    assert _export(circ, mode=QASM2ExportMode.EXTENDED) == _export(circ)


@pytest.mark.qiskit_ref
def test_extended_mode_parameterized_custom_gate() -> None:
    """Each instance of a parameterized custom gate keeps its own angle."""
    _register_zz()
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="myzz", num_qubits=2, params=(0.3,)), (0, 1))
    circ.append(Gate(name="myzz", num_qubits=2, params=(0.7,)), (0, 1))
    qc = qiskit_loads(_export(circ, mode=QASM2ExportMode.EXTENDED))
    ref = qiskit.QuantumCircuit(2)
    ref.rzz(1.0, 0, 1)
    assert_unitary_equal(
        qiskit_circuit_unitary(qc), qiskit_circuit_unitary(ref), up_to_phase=True
    )


def test_extended_mode_does_not_drop_unknown_body_gates() -> None:
    """A rule yielding an un-exportable gate must not be silently truncated."""

    def rule(gate: Gate, qs: tuple[int, ...]) -> list[tuple[Gate, tuple[int, ...]]]:
        return [(Gate(name="mystery", num_qubits=1), (qs[0],)), (H, (qs[0],))]

    register_decomposition(GateDecompositionRule("wrap", 1, rule))
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="wrap", num_qubits=1), (0,))
    with pytest.raises(UnsupportedGateError):
        _export(circ, mode=QASM2ExportMode.EXTENDED)


def test_angle_formatting_in_decompose_mode_keeps_precision_for_pi() -> None:
    """Angles such as pi/2 are written with full round-trip precision."""
    circ = quspect.QuantumCircuit()
    circ.rx(math.pi / 2, 0)
    assert "rx(1.5707963267948966) q[0];" in QASM2Exporter().export(circ)
