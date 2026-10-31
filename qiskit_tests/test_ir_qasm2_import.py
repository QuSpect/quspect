"""Cross-validate :func:`quspect.ir.qasm2_loads` / :class:`QASM2Importer`.

QASM text is produced by ``qiskit.qasm2.dumps`` (or written by hand), parsed
by quspect, and the resulting circuit is compared with the qiskit circuit
(``qiskit.qasm2.loads`` of the same text) via unitaries and measurement maps.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
import qiskit
import qiskit.qasm2
from qiskit.quantum_info import Statevector

import quspect
from qiskit_tests._ir_helpers import (
    LEGACY_IMPORTABLE,
    QELIB_HEADER,
    qiskit_loads,
    qiskit_measure_map,
    quspect_measure_map,
    random_legacy_qiskit_circuit,
    strip_measurements,
    unitary_only,
)
from qiskit_tests.helpers import (
    assert_states_equal,
    assert_unitary_equal,
    qiskit_circuit_unitary,
    quspect_circuit_unitary,
)
from quspect.ir import QASM2Importer, QASM2ParseError, qasm2_loads
from quspect.simulation import Simulator


def _ops(circ: quspect.QuantumCircuit) -> list[tuple[str, str | None, tuple[int, ...]]]:
    """Compact ``(type, gate name, qubits)`` view of a circuit."""
    return [
        (i.type.name, i.gate.name if i.gate is not None else None, i.qubits)
        for i in circ.instructions
    ]


def _assert_same_unitary_as_qiskit(source: str, *, up_to_phase: bool = False) -> None:
    """Parse ``source`` with both libraries and compare the unitaries."""
    qc = qiskit_loads(source)
    circ = qasm2_loads(source)
    assert_unitary_equal(
        quspect_circuit_unitary(unitary_only(circ), num_qubits=qc.num_qubits),
        qiskit_circuit_unitary(strip_measurements(qc)),
        up_to_phase=up_to_phase,
        atol=1e-9,
    )


# =============================================================================
# qiskit.qasm2.dumps -> quspect import
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(LEGACY_IMPORTABLE))
def test_each_importable_gate_matches_qiskit(name: str) -> None:
    """Each qelib1 gate quspect knows imports to the exact qiskit unitary."""
    factory, nq, npar = LEGACY_IMPORTABLE[name]
    qc = qiskit.QuantumCircuit(3)
    qc.append(factory(*[0.41, -1.33, 2.27][:npar]), [2, 0, 1][:nq])
    text = qiskit.qasm2.dumps(qc)
    circ = qasm2_loads(text)
    assert_unitary_equal(
        quspect_circuit_unitary(circ, num_qubits=3), qiskit_circuit_unitary(qc)
    )


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(12))
def test_random_qiskit_dump_imports_exactly(seed: int) -> None:
    """Random qiskit circuits (qelib1 subset) dumped by qiskit import exactly."""
    qc = random_legacy_qiskit_circuit(4, 40, seed, measure=True)
    circ = qasm2_loads(qiskit.qasm2.dumps(qc))
    assert circ.num_clbits == 4
    assert quspect_measure_map(circ) == qiskit_measure_map(qc)
    assert_unitary_equal(
        quspect_circuit_unitary(unitary_only(circ), num_qubits=4),
        qiskit_circuit_unitary(strip_measurements(qc)),
    )


@pytest.mark.qiskit_ref
def test_imported_counts_follow_bit_convention() -> None:
    """Sampling an imported circuit gives qiskit's counts reversed per bitstring."""
    qc = qiskit.QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(2)
    qc.measure(0, 2)
    qc.measure(1, 0)
    qc.measure(2, 1)
    circ = qasm2_loads(qiskit.qasm2.dumps(qc))
    counts = Simulator(seed=1).run(circ, shots=50).counts
    # qiskit: c2=1 (from q0), c0=0 (from q1), c1=1 (from q2) -> "110"
    assert {k: v for k, v in counts.items() if v} == {"110"[::-1]: 50}


UNSUPPORTED_QELIB1 = {
    "u": "u(0.3,0.4,0.5) q[0];",
    "U": "U(0.3,0.4,0.5) q[0];",
    "CX": "CX q[0],q[1];",
    "u0": "u0(1) q[0];",
    "sx": "sx q[0];",
    "sxdg": "sxdg q[0];",
    "ch": "ch q[0],q[1];",
    "cy": "cy q[0],q[1];",
    "csx": "csx q[0],q[1];",
    "cswap": "cswap q[0],q[1],q[2];",
    "rzz": "rzz(0.3) q[0],q[1];",
    "rxx": "rxx(0.3) q[0],q[1];",
    "cu": "cu(0.3,0.4,0.5,0.6) q[0],q[1];",
    "rccx": "rccx q[0],q[1],q[2];",
}


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(UNSUPPORTED_QELIB1))
def test_unsupported_qelib1_gates(name: str) -> None:
    """Standard qelib1 gates should import with qiskit's matrix (up to phase)."""
    _assert_same_unitary_as_qiskit(
        QELIB_HEADER + "qreg q[3];\n" + UNSUPPORTED_QELIB1[name], up_to_phase=True
    )


# =============================================================================
# Parameter expressions
# =============================================================================


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("pi", math.pi),
        ("-pi", -math.pi),
        ("+0.25", 0.25),
        ("1e-3", 1e-3),
        ("2.5E+1", 25.0),
        ("3", 3.0),
    ],
)
def test_simple_parameter_literals(expr: str, value: float) -> None:
    """Numbers, signed numbers and ``pi`` parse to the right float."""
    circ = qasm2_loads(QELIB_HEADER + f"qreg q[1];\nrx({expr}) q[0];")
    assert circ[0].gate is not None
    assert circ[0].gate.params == pytest.approx((value,), abs=1e-15)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize(
    "expr", ["pi/2", "-pi/4", "2*pi", "pi/2+0.1", "(1+2)*0.1", "sin(0.3)", "sqrt(2)"]
)
def test_parameter_expressions(expr: str) -> None:
    """Arithmetic/function parameter expressions evaluate like qiskit."""
    _assert_same_unitary_as_qiskit(QELIB_HEADER + f"qreg q[1];\nrx({expr}) q[0];")


@pytest.mark.qiskit_ref
def test_qiskit_dump_with_pi_fraction_angles() -> None:
    """A qiskit dump containing ``rx(pi/2)`` imports with the right unitary."""
    qc = qiskit.QuantumCircuit(1)
    qc.rx(math.pi / 2, 0)
    qc.rz(-math.pi / 4, 0)
    text = qiskit.qasm2.dumps(qc)
    assert "pi/2" in text
    _assert_same_unitary_as_qiskit(text)


def test_leading_dot_float_literal() -> None:
    """``.5`` is a valid QASM real literal equal to 0.5."""
    circ = qasm2_loads(QELIB_HEADER + "qreg q[1];\nrx(.5) q[0];")
    assert circ[0].gate is not None
    assert circ[0].gate.params == (0.5,)


@pytest.mark.parametrize("stmt", ["rx(0.1,0.2) q[0];", "h(0.1) q[0];"])
def test_wrong_parameter_count_raises(stmt: str) -> None:
    """A wrong number of gate parameters is a parse error."""
    with pytest.raises(QASM2ParseError):
        qasm2_loads(QELIB_HEADER + "qreg q[1];\n" + stmt)


def test_missing_parameters_raise() -> None:
    """A parameterized gate without parameters is a parse error."""
    with pytest.raises(QASM2ParseError, match="requires parameters"):
        qasm2_loads(QELIB_HEADER + "qreg q[1];\nrx q[0];")


# =============================================================================
# Registers
# =============================================================================


def test_single_registers_and_measure() -> None:
    """One qreg / creg: indices are taken verbatim."""
    circ = qasm2_loads(
        QELIB_HEADER
        + "qreg q[3];\ncreg c[3];\nh q[2];\n"
        + "measure q[2] -> c[0];\nmeasure q[0] -> c[2];"
    )
    assert _ops(circ) == [
        ("GATE", "h", (2,)),
        ("MEASURE", None, (2,)),
        ("MEASURE", None, (0,)),
    ]
    assert quspect_measure_map(circ) == [(2, 0), (0, 2)]


MULTI_REG = (
    QELIB_HEADER
    + "qreg a[2];\nqreg b[2];\ncreg c[1];\ncreg d[2];\n"
    + "x a[1];\nh b[0];\ncx a[0],b[1];\n"
    + "measure b[1] -> d[0];\nmeasure a[0] -> c[0];\nmeasure a[1] -> d[1];\n"
)


@pytest.mark.qiskit_ref
def test_multiple_registers_concatenate_like_qiskit() -> None:
    """Multiple qreg/creg are concatenated in declaration order (as in qiskit)."""
    qc = qiskit_loads(MULTI_REG)
    circ = qasm2_loads(MULTI_REG)
    assert circ.num_qubits == qc.num_qubits == 4
    assert quspect_measure_map(circ) == qiskit_measure_map(qc)
    assert_unitary_equal(
        quspect_circuit_unitary(unitary_only(circ), num_qubits=4),
        qiskit_circuit_unitary(strip_measurements(qc)),
    )


@pytest.mark.parametrize("stmt", ["h q;", "measure q -> c;", "barrier q;"])
def test_register_broadcast(stmt: str) -> None:
    """Whole-register operands broadcast over every bit."""
    circ = qasm2_loads(QELIB_HEADER + "qreg q[2];\ncreg c[2];\n" + stmt)
    assert len(circ) >= 1
    assert circ.active_qubits == {0, 1}


# =============================================================================
# Custom gates
# =============================================================================


@pytest.mark.qiskit_ref
def test_custom_gate_definition_expands_inline() -> None:
    """A parameter-free ``gate`` block expands with formal->actual qubit mapping."""
    src = (
        QELIB_HEADER
        + "qreg q[3];\n"
        + "gate mygate a,b,c { h a; cx a,c; barrier a,b; rz(0.3) b; ccx c,b,a; }\n"
        + "mygate q[2],q[0],q[1];\nmygate q[0],q[1],q[2];\n"
    )
    _assert_same_unitary_as_qiskit(src)
    circ = qasm2_loads(src)
    assert _ops(circ)[:4] == [
        ("GATE", "h", (2,)),
        ("GATE", "cx", (2, 1)),
        ("GATE", "rz", (0,)),
        ("GATE", "ccx", (1, 0, 2)),
    ]


def test_custom_gate_wrong_arity_raises() -> None:
    """Calling a custom gate with the wrong number of qubits is an error."""
    src = QELIB_HEADER + "qreg q[2];\ngate g a { h a; }\ng q[0],q[1];"
    with pytest.raises(QASM2ParseError, match="expects 1 qubits"):
        qasm2_loads(src)


def test_custom_gate_unknown_formal_qubit_raises() -> None:
    """A body referring to an undeclared formal qubit is an error."""
    src = QELIB_HEADER + "qreg q[2];\ngate g a { h b; }\n"
    with pytest.raises(QASM2ParseError, match="Unknown qubit name 'b'"):
        qasm2_loads(src)


def test_custom_gate_unknown_body_gate_raises_on_use() -> None:
    """An unknown gate inside a body is reported when the gate is applied."""
    src = QELIB_HEADER + "qreg q[1];\ngate g a { nope a; }\ng q[0];"
    with pytest.raises(QASM2ParseError, match="inside custom gate"):
        qasm2_loads(src)


def test_standard_gate_wrong_arity_raises() -> None:
    """A standard gate with the wrong number of qubits is an error."""
    with pytest.raises(QASM2ParseError, match="expects 2 qubits"):
        qasm2_loads(QELIB_HEADER + "qreg q[2];\ncx q[0];")


@pytest.mark.qiskit_ref
def test_parameterized_custom_gate() -> None:
    """``gate g(theta) a { rx(theta) a; }`` substitutes the actual angle."""
    _assert_same_unitary_as_qiskit(
        QELIB_HEADER
        + "qreg q[1];\ngate g(theta) a { rx(theta/2) a; rz(-theta) a; }\ng(0.8) q[0];"
    )


@pytest.mark.qiskit_ref
def test_nested_custom_gates() -> None:
    """A custom gate may use previously declared custom gates."""
    _assert_same_unitary_as_qiskit(
        QELIB_HEADER
        + "qreg q[2];\ngate inner a { h a; }\n"
        + "gate outer a,b { inner a; cx a,b; }\nouter q[1],q[0];"
    )


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(4))
def test_qiskit_dump_of_composite_gate(seed: int) -> None:
    """qiskit dumps a composite instruction as a ``gate`` block; quspect expands it."""
    sub = random_legacy_qiskit_circuit(
        2, 6, seed, gate_names=["h", "x", "cx", "s", "t", "cz", "swap"]
    )
    gate = sub.to_gate(label=None)
    gate.name = f"blk{seed}"
    qc = qiskit.QuantumCircuit(3)
    qc.append(gate, [2, 0])
    qc.h(1)
    qc.append(gate, [0, 1])
    text = qiskit.qasm2.dumps(qc)
    assert f"gate blk{seed}" in text
    circ = qasm2_loads(text)
    assert_unitary_equal(
        quspect_circuit_unitary(circ, num_qubits=3), qiskit_circuit_unitary(qc)
    )


# =============================================================================
# Comments, include, opaque, reset, if, barrier
# =============================================================================


def test_comments_includes_and_stray_semicolons() -> None:
    """Line comments, several includes and stray ``;`` are ignored."""
    src = (
        "// leading comment\nOPENQASM 2.0; // version\n"
        'include "qelib1.inc";\ninclude "other.inc"; // second include\n'
        "qreg q[2]; ;\n// h q[1];\nh q[0]; // trailing\n;;\nx q[1];\n"
    )
    circ = qasm2_loads(src)
    assert _ops(circ) == [("GATE", "h", (0,)), ("GATE", "x", (1,))]


def test_opaque_declaration_is_skipped() -> None:
    """``opaque`` declarations are consumed without adding instructions."""
    circ = qasm2_loads(QELIB_HEADER + "qreg q[1];\nopaque myop(theta) a;\nh q[0];")
    assert _ops(circ) == [("GATE", "h", (0,))]


def test_barriers() -> None:
    """Explicit-qubit and argument-less barriers are imported."""
    circ = qasm2_loads(
        QELIB_HEADER + "qreg q[3];\nh q[0];\nh q[2];\nbarrier q[0],q[2];\nbarrier;"
    )
    assert _ops(circ)[2:] == [("BARRIER", None, (0, 2)), ("BARRIER", None, (0, 2))]


def test_reset_is_rejected() -> None:
    """``reset`` has no quspect equivalent and is reported as a parse error."""
    with pytest.raises(QASM2ParseError, match="reset"):
        qasm2_loads(QELIB_HEADER + "qreg q[1];\nreset q[0];")


@pytest.mark.parametrize(
    "stmt", ["if(c==1) x q[0];", "if (c==1) measure q[0] -> c[0];"]
)
def test_if_statement_not_silently_dropped(stmt: str) -> None:
    """Classically controlled operations must not vanish silently."""
    src = QELIB_HEADER + "qreg q[1];\ncreg c[1];\n" + stmt
    try:
        circ = qasm2_loads(src)
    except QASM2ParseError:
        return
    assert len(circ) == 1


# =============================================================================
# Errors
# =============================================================================


@pytest.mark.parametrize(
    "src",
    [
        "",  # empty input
        "qreg q[1];",  # missing header
        "OPENQASM 3.0;\nqreg q[1];",  # wrong version
        "OPENQASM 2.0\nqreg q[1];",  # missing ';' after version
        QELIB_HEADER + "qreg q[1];\nh q[0]\nx q[0];",  # missing ';'
        QELIB_HEADER + "qreg q[1];\nfoo q[0];",  # unknown gate
        QELIB_HEADER + "qreg q[1];\nh q[0",  # unexpected EOF
        QELIB_HEADER + "qreg q[1];\n-> q[0];",  # stray token
        QELIB_HEADER + "qreg q[x];",  # non-numeric size
        QELIB_HEADER + "qreg q[1];\nrx(foo) q[0];",  # bad identifier param
        QELIB_HEADER + "qreg q[1];\nrx(,) q[0];",  # missing number
        "OPENQASM 2.0;\ninclude qelib1;\nqreg q[1];",  # include without string
    ],
)
def test_syntax_errors_raise_parse_error(src: str) -> None:
    """Malformed input raises :class:`QASM2ParseError` (a ``ValueError``)."""
    with pytest.raises(QASM2ParseError):
        qasm2_loads(src)
    assert issubclass(QASM2ParseError, ValueError)


@pytest.mark.parametrize(
    ("src", "line"),
    [
        (QELIB_HEADER + "qreg q[1];\nh q[0]\nx q[0];", 5),
        (QELIB_HEADER + "qreg q[1];\n\nfoo q[0];", 5),
        (QELIB_HEADER + "qreg q[1];\n-> q[0];", 4),
    ],
)
def test_parse_error_reports_line_number(src: str, line: int) -> None:
    """The error message points at the offending line."""
    with pytest.raises(QASM2ParseError) as info:
        qasm2_loads(src)
    assert re.search(rf"\bline\s*{line}\b|:{line}\b", str(info.value))


@pytest.mark.parametrize("stmt", ["h q[0] @@;", "h $ q[0];", "h q[0]; #"])
def test_invalid_characters_raise(stmt: str) -> None:
    """Characters that are not valid QASM must raise a parse error."""
    with pytest.raises(QASM2ParseError):
        qasm2_loads(QELIB_HEADER + "qreg q[1];\n" + stmt)


# =============================================================================
# Importer object
# =============================================================================


def test_importer_instance_is_reusable_and_resets_state() -> None:
    """A second ``imports`` call does not see the first call's custom gates."""
    importer = QASM2Importer()
    first = importer.imports(QELIB_HEADER + "qreg q[1];\ngate g a { x a; }\ng q[0];")
    assert _ops(first) == [("GATE", "x", (0,))]
    with pytest.raises(QASM2ParseError, match="Unknown gate 'g'"):
        importer.imports(QELIB_HEADER + "qreg q[1];\ng q[0];")


def test_qasm2_loads_equals_importer() -> None:
    """``qasm2_loads`` is a thin wrapper over ``QASM2Importer().imports``."""
    src = QELIB_HEADER + "qreg q[2];\nh q[0];\ncu3(0.1,0.2,0.3) q[0],q[1];"
    assert qasm2_loads(src).instructions == QASM2Importer().imports(src).instructions


@pytest.mark.qiskit_ref
def test_name_mapping_u1_cu1_p_cp() -> None:
    """``u1``/``p`` import as quspect ``p``; ``cu1``/``cp`` as ``cp``."""
    circ = qasm2_loads(
        QELIB_HEADER
        + "qreg q[2];\nu1(0.2) q[0];\np(0.3) q[1];\n"
        + "cu1(0.4) q[0],q[1];\ncp(0.5) q[1],q[0];"
    )
    assert [i.gate.name for i in circ.instructions if i.gate is not None] == [
        "p",
        "p",
        "cp",
        "cp",
    ]


@pytest.mark.qiskit_ref
def test_imported_statevector_matches_qiskit() -> None:
    """Simulating an imported circuit reproduces qiskit's statevector."""
    qc = random_legacy_qiskit_circuit(3, 25, 99)
    circ = qasm2_loads(qiskit.qasm2.dumps(qc))
    sv = Simulator(seed=0).statevector(circ)
    assert_states_equal(sv, Statevector(qc).data, up_to_phase=False)
    assert np.isclose(np.linalg.norm(sv), 1.0)
