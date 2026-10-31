"""Tests for the OpenQASM 2.0 importer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.ir.qasm2 import QASM2Exporter
from quspect.ir.qasm_importer import QASM2Importer, QASM2ParseError, qasm2_loads


class TestQASM2Importer:
    """Tests for basic QASM 2.0 parsing."""

    def test_basic_header_and_registers(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[0], q[1];
"""
        circ = qasm2_loads(qasm)
        # num_qubits is inferred from the largest qubit index used
        assert circ.num_qubits == 2
        assert len(circ) == 1

    def test_standard_gates(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""
        circ = qasm2_loads(qasm)
        assert len(circ) == 2
        assert circ[0].gate.name == "h"
        assert circ[1].gate.name == "cx"

    def test_measurement(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
measure q[0] -> c[0];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].is_gate
        assert circ[1].is_measure
        assert circ[1].qubits == (0,)
        assert circ[1].clbits == (0,)

    def test_barrier(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
barrier q[0], q[1];
cx q[0], q[1];
"""
        circ = qasm2_loads(qasm)
        assert len(circ) == 3
        assert circ[1].is_barrier
        assert circ[1].qubits == (0, 1)

    def test_parameterized_gates(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(0.5) q[0];
ry(1.0) q[0];
rz(-1.5708) q[0];
"""
        circ = qasm2_loads(qasm)
        assert len(circ) == 3
        assert circ[0].gate.name == "rx"
        assert circ[0].gate.params == (0.5,)
        assert circ[2].gate.name == "rz"
        assert circ[2].gate.params == (-1.5708,)

    def test_u3_gate(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
u3(0.1, 0.2, 0.3) q[0];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].gate.name == "u3"
        assert circ[0].gate.params == (0.1, 0.2, 0.3)

    def test_pi_constant(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rz(pi) q[0];
rx(-pi) q[0];
"""
        circ = qasm2_loads(qasm)
        import math

        assert circ[0].gate.params == (math.pi,)
        assert circ[1].gate.params == (-math.pi,)

    def test_ccx_and_swap(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
swap q[0], q[1];
ccx q[0], q[1], q[2];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].gate.name == "swap"
        assert circ[0].gate.num_qubits == 2
        assert circ[1].gate.name == "ccx"
        assert circ[1].gate.num_qubits == 3

    def test_custom_gate_declaration(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
gate my_h_cx a, b {
  h a;
  cx a, b;
}
my_h_cx q[0], q[1];
"""
        circ = qasm2_loads(qasm)
        assert len(circ) == 2
        assert circ[0].gate.name == "h"
        assert circ[1].gate.name == "cx"

    def test_round_trip(self) -> None:
        from quspect.circuit.circuit import QuantumCircuit

        orig = QuantumCircuit(name="bell")
        orig.h(0).cx(0, 1).measure([0, 1], [0, 1])
        exporter = QASM2Exporter()
        qasm_str = exporter.export(orig)
        reimported = qasm2_loads(qasm_str)
        assert reimported.num_qubits == orig.num_qubits
        assert reimported.num_clbits == orig.num_clbits
        # Gates round-trip correctly; measurements may be split/combined
        assert sum(1 for i in reimported if i.is_gate) == sum(
            1 for i in orig if i.is_gate
        )
        assert sum(1 for i in reimported if i.is_measure) >= sum(
            1 for i in orig if i.is_measure
        )

    def test_missing_version_raises(self) -> None:
        with pytest.raises(QASM2ParseError):
            qasm2_loads("qreg q[1];")

    def test_unknown_gate_raises(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
mythical q[0];
"""
        with pytest.raises(QASM2ParseError):
            qasm2_loads(qasm)


class TestQASM2ImporterPrecision:
    """Tests for numeric precision and edge cases."""

    def test_integer_params(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(1) q[0];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].gate.params == (1.0,)

    def test_scientific_notation(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(1e-3) q[0];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].gate.params == (0.001,)

    def test_u1_maps_to_p(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
u1(0.5) q[0];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].gate.name == "p"

    def test_cu1_maps_to_cp(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cu1(0.5) q[0],q[1];
"""
        circ = qasm2_loads(qasm)
        assert circ[0].gate.name == "cp"
        assert circ[0].gate.params == (0.5,)


# =============================================================================
# Regression tests for the spec-conforming parser (B31-B42)
# =============================================================================

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _unitary(circ: QuantumCircuit, n: int) -> np.ndarray:
    """Big-endian (qubit 0 = most significant) unitary of a gate-only circuit."""
    psi = np.eye(1 << n, dtype=complex).reshape((2,) * n + (1 << n,))
    for instr in circ.instructions:
        if not instr.is_gate:
            continue
        assert instr.gate is not None
        k = len(instr.qubits)
        mat = gate_matrix(instr.gate).reshape((2,) * (2 * k))
        psi = np.tensordot(mat, psi, axes=(list(range(k, 2 * k)), list(instr.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(instr.qubits))
    return psi.reshape(1 << n, 1 << n)


def _controlled(u: np.ndarray, controls: int = 1) -> np.ndarray:
    """``u`` controlled on the leading ``controls`` qubits (big-endian)."""
    dim = u.shape[0] << controls
    out = np.eye(dim, dtype=complex)
    out[dim - u.shape[0] :, dim - u.shape[0] :] = u
    return out


_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0 + 0j, -1.0])
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ]
    )


def _pauli_rotation(pauli: np.ndarray, theta: float) -> np.ndarray:
    return math.cos(theta / 2) * np.eye(4) - 1j * math.sin(theta / 2) * np.kron(
        pauli, pauli
    )


class TestQASM2Literals:
    """B34: every OpenQASM real-literal form is parsed correctly."""

    @pytest.mark.parametrize(
        ("text", "value"),
        [(".5", 0.5), ("5.", 5.0), ("1e-3", 1e-3), ("1.5E+2", 150.0), ("2", 2.0)],
    )
    def test_real_literal_forms(self, text: str, value: float) -> None:
        circ = qasm2_loads(HEADER + f"qreg q[1];\nrx({text}) q[0];")
        assert circ[0].gate is not None
        assert circ[0].gate.params == (value,)


class TestQASM2Expressions:
    """B32/B33: full parameter expressions with OpenQASM precedence."""

    @pytest.mark.parametrize(
        ("expr", "value"),
        [
            ("pi/2", math.pi / 2),
            ("-pi/4", -math.pi / 4),
            ("2*pi", 2 * math.pi),
            ("(1+2)*0.1", (1 + 2) * 0.1),
            ("-2^2", -4.0),
            ("2^3^2", 512.0),
            ("2^-1^2", 0.5),
            ("1/2*3", 1.5),
            ("2-3-4", -5.0),
            ("- -1", 1.0),
            ("2*-3", -6.0),
            ("sin(0.3)", math.sin(0.3)),
            ("cos(0.3)+tan(0.2)", math.cos(0.3) + math.tan(0.2)),
            ("exp(1)-ln(2)", math.exp(1) - math.log(2)),
            ("sqrt(2)", math.sqrt(2)),
            ("sin(1)^2", math.sin(1) ** 2),
        ],
    )
    def test_expression_values(self, expr: str, value: float) -> None:
        circ = qasm2_loads(HEADER + f"qreg q[1];\nrz({expr}) q[0];")
        assert circ[0].gate is not None
        assert circ[0].gate.params == pytest.approx((value,), rel=1e-15)

    def test_pi_fraction_is_exact(self) -> None:
        circ = qasm2_loads(HEADER + "qreg q[1];\nrx(pi/2) q[0];")
        assert circ[0].gate is not None
        assert circ[0].gate.params == (math.pi / 2,)

    @pytest.mark.parametrize(
        ("expr", "match"),
        [
            ("1/0", "Division by zero"),
            ("ln(0)", "ln"),
            ("sqrt(-1)", "sqrt"),
            ("theta", "Unknown parameter 'theta'"),
            ("(1", "Expected"),
        ],
    )
    def test_bad_expressions_raise(self, expr: str, match: str) -> None:
        with pytest.raises(QASM2ParseError, match=match):
            qasm2_loads(HEADER + f"qreg q[1];\nrz({expr}) q[0];")


class TestQASM2Diagnostics:
    """B41/B42: errors carry positions; invalid characters are rejected."""

    @pytest.mark.parametrize("stmt", ["h q[0] @@;", "h $ q[0];", "h q[0]; #"])
    def test_invalid_characters_raise(self, stmt: str) -> None:
        with pytest.raises(QASM2ParseError, match="Invalid character"):
            qasm2_loads(HEADER + "qreg q[1];\n" + stmt)

    def test_line_and_column_attributes(self) -> None:
        src = HEADER + "qreg q[1];\n\n  foo q[0];"
        with pytest.raises(QASM2ParseError) as info:
            qasm2_loads(src)
        assert info.value.line == 5
        assert info.value.column == 3
        assert "line 5, column 3" in str(info.value)

    def test_missing_semicolon_reports_next_line(self) -> None:
        with pytest.raises(QASM2ParseError, match="line 5"):
            qasm2_loads(HEADER + "qreg q[1];\nh q[0]\nx q[0];")

    def test_error_without_position(self) -> None:
        err = QASM2ParseError("plain")
        assert str(err) == "plain"
        assert err.line is None


class TestQASM2Registers:
    """B36/B37: register concatenation, validation and broadcasting."""

    def test_registers_concatenate_in_declaration_order(self) -> None:
        circ = qasm2_loads(
            HEADER
            + "qreg a[2];\nqreg b[2];\ncreg c[1];\ncreg d[2];\n"
            + "x a[1];\nh b[0];\ncx a[0],b[1];\n"
            + "measure b[1] -> d[0];\nmeasure a[0] -> c[0];\n"
        )
        assert [i.qubits for i in circ.instructions] == [(1,), (2,), (0, 3), (3,), (0,)]
        assert [i.clbits for i in circ.instructions if i.is_measure] == [(1,), (0,)]

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ("qreg q[2];\nqreg q[1];", "already defined"),
            ("qreg q[2];\ncreg q[1];", "already defined"),
            ("qreg q[2];\nh q[2];", "out of range"),
            ("qreg q[2];\nh r[0];", "Unknown quantum register 'r'"),
            ("qreg q[1];\ncreg c[1];\nmeasure q[0] -> d[0];", "Unknown classical"),
            ("qreg q[0];", "positive size"),
            ("qreg a[2];\nqreg b[3];\ncx a,b;", "same size"),
            ("qreg a[2];\ncreg c[3];\nmeasure a -> c;", "same size"),
            ("qreg a[2];\ncreg c[2];\nmeasure a -> c[0];", "same size"),
            ("qreg q[2];\ncx q[0],q[0];", "Duplicate qubit"),
            ("qreg q[2];\ncx q,q;", "Duplicate qubit"),
        ],
    )
    def test_register_errors(self, body: str, match: str) -> None:
        with pytest.raises(QASM2ParseError, match=match):
            qasm2_loads(HEADER + body)

    def test_whole_register_broadcast(self) -> None:
        circ = qasm2_loads(
            HEADER
            + "qreg a[2];\nqreg b[2];\ncreg c[2];\n"
            + "h a;\ncx a,b;\ncx a[0],b;\nbarrier a,b[0];\nmeasure b -> c;\n"
        )
        ops = [(i.type.name, i.qubits, i.clbits) for i in circ.instructions]
        assert ops == [
            ("GATE", (0,), ()),
            ("GATE", (1,), ()),
            ("GATE", (0, 2), ()),
            ("GATE", (1, 3), ()),
            ("GATE", (0, 2), ()),
            ("GATE", (0, 3), ()),
            ("BARRIER", (0, 1, 2), ()),
            ("MEASURE", (2,), (0,)),
            ("MEASURE", (3,), (1,)),
        ]


class TestQASM2Statements:
    """B40 and unsupported statements."""

    @pytest.mark.parametrize(
        "stmt", ["if(c==1) x q[0];", "if (c==1) measure q[0] -> c[0];"]
    )
    def test_if_statement_rejected_with_line(self, stmt: str) -> None:
        with pytest.raises(QASM2ParseError, match=r"'if'.*line 5"):
            qasm2_loads(HEADER + "qreg q[1];\ncreg c[1];\n" + stmt)

    def test_opaque_gate_cannot_be_applied(self) -> None:
        with pytest.raises(QASM2ParseError, match="Opaque gate 'op'"):
            qasm2_loads(HEADER + "qreg q[1];\nopaque op a;\nop q[0];")


class TestQASM2GateValidation:
    """B35: parameter and qubit counts are validated for every gate."""

    @pytest.mark.parametrize(
        ("stmt", "match"),
        [
            ("rx(0.1,0.2) q[0];", "expects 1 parameters, got 2"),
            ("h(0.1) q[0];", "expects 0 parameters, got 1"),
            ("u3(0.1) q[0];", "expects 3 parameters, got 1"),
            ("cu(1,2,3) q[0],q[1];", "expects 4 parameters, got 3"),
            ("rzz q[0],q[1];", "requires parameters"),
            ("sx q[0],q[1];", "expects 1 qubits, got 2"),
            ("g(1,2) q[0];", "expects 1 parameters, got 2"),
            ("g q[0];", "requires parameters"),
        ],
    )
    def test_wrong_counts_raise(self, stmt: str, match: str) -> None:
        src = HEADER + "qreg q[2];\ngate g(t) a { rx(t) a; }\n" + stmt
        with pytest.raises(QASM2ParseError, match=match):
            qasm2_loads(src)

    def test_body_call_validated_at_definition(self) -> None:
        with pytest.raises(QASM2ParseError, match="expects 2 qubits"):
            qasm2_loads(HEADER + "qreg q[1];\ngate g a { cx a; }\n")


class TestQASM2Qelib1Gates:
    """B31: qelib1 gates without a native QuSpect gate keep their exact matrix."""

    @staticmethod
    def _check(stmt: str, n: int, expected: np.ndarray) -> None:
        circ = qasm2_loads(HEADER + f"qreg q[{n}];\n{stmt}")
        np.testing.assert_allclose(_unitary(circ, n), expected, atol=1e-12)

    def test_builtin_u_and_cx(self) -> None:
        circ = qasm2_loads(HEADER + "qreg q[2];\nU(0.1,0.2,0.3) q[0];\nCX q[0],q[1];")
        assert [(i.gate.name if i.gate else None) for i in circ] == ["u3", "cx"]
        circ = qasm2_loads(HEADER + "qreg q[1];\nu(0.1,0.2,0.3) q[0];\nu0(1) q[0];")
        assert [(i.gate.name if i.gate else None) for i in circ] == ["u3", "id"]

    @pytest.mark.parametrize(
        ("stmt", "n", "expected"),
        [
            ("sx q[0];", 1, _SX),
            ("sxdg q[0];", 1, _SX.conj().T),
            ("ch q[0],q[1];", 2, _controlled(_H)),
            ("cy q[0],q[1];", 2, _controlled(_Y)),
            ("csx q[0],q[1];", 2, _controlled(_SX)),
            ("cswap q[0],q[1],q[2];", 3, _controlled(_SWAP)),
            ("rzz(0.3) q[0],q[1];", 2, _pauli_rotation(_Z, 0.3)),
            ("rxx(0.3) q[0],q[1];", 2, _pauli_rotation(_X, 0.3)),
            (
                "cu(0.3,0.4,0.5,0.6) q[0],q[1];",
                2,
                _controlled(np.exp(0.6j) * _u3(0.3, 0.4, 0.5)),
            ),
            ("c3x q[0],q[1],q[2],q[3];", 4, _controlled(_X, 3)),
            ("c4x q[0],q[1],q[2],q[3],q[4];", 5, _controlled(_X, 4)),
            ("c3sqrtx q[0],q[1],q[2],q[3];", 4, _controlled(_SX, 3)),
        ],
    )
    def test_exact_matrix(self, stmt: str, n: int, expected: np.ndarray) -> None:
        self._check(stmt, n, expected)

    @pytest.mark.parametrize(("name", "n"), [("rccx", 3), ("rc3x", 4)])
    def test_relative_phase_toffolis(self, name: str, n: int) -> None:
        qubits = ",".join(f"q[{i}]" for i in range(n))
        circ = qasm2_loads(HEADER + f"qreg q[{n}];\n{name} {qubits};")
        u = _unitary(circ, n)
        # Equal to the multi-controlled X up to relative phases.
        np.testing.assert_allclose(
            np.abs(u), np.abs(_controlled(_X, n - 1)), atol=1e-12
        )

    def test_native_gates_are_not_expanded(self) -> None:
        circ = qasm2_loads(
            HEADER + "qreg q[3];\ncrz(0.2) q[0],q[1];\nccx q[0],q[1],q[2];"
        )
        assert [i.gate.name for i in circ if i.gate is not None] == ["crz", "ccx"]

    def test_redefining_included_gate_raises(self) -> None:
        with pytest.raises(QASM2ParseError, match="'h' is already defined"):
            qasm2_loads(HEADER + "qreg q[1];\ngate h a { x a; }")

    def test_shadowing_without_include_is_allowed(self) -> None:
        circ = qasm2_loads("OPENQASM 2.0;\nqreg q[1];\ngate h a { x a; }\nh q[0];")
        assert [i.gate.name for i in circ if i.gate is not None] == ["x"]

    def test_builtin_u_cannot_be_redefined(self) -> None:
        with pytest.raises(QASM2ParseError, match="'U' is already defined"):
            qasm2_loads("OPENQASM 2.0;\nqreg q[1];\ngate U a { x a; }")


class TestQASM2CustomGates:
    """B38/B39: parameterized and nested custom gates."""

    def test_parameterized_custom_gate(self) -> None:
        circ = qasm2_loads(
            HEADER
            + "qreg q[1];\ngate g(theta, phi) a { rx(theta/2) a; rz(-theta+phi) a; }\n"
            + "g(0.8, 0.1) q[0];"
        )
        assert [(i.gate.name, i.gate.params) for i in circ if i.gate is not None] == [
            ("rx", (0.4,)),
            ("rz", (-0.8 + 0.1,)),
        ]

    def test_nested_custom_gates_with_parameters(self) -> None:
        circ = qasm2_loads(
            HEADER
            + "qreg q[2];\ngate inner(t) a { rz(2*t) a; }\n"
            + "gate outer(s) a,b { inner(s+1) b; cx a,b; }\nouter(0.5) q[1],q[0];"
        )
        assert [(i.gate.name, i.gate.params, i.qubits) for i in circ if i.gate] == [
            ("rz", (3.0,), (0,)),
            ("cx", (), (1, 0)),
        ]

    def test_unknown_body_gate_reported_at_definition(self) -> None:
        with pytest.raises(QASM2ParseError, match=r"Unknown gate 'nope'.*line 4"):
            qasm2_loads(HEADER + "qreg q[1];\ngate g a { nope a; }\n")

    def test_recursive_gate_is_unknown(self) -> None:
        with pytest.raises(QASM2ParseError, match="Unknown gate 'g'"):
            qasm2_loads(HEADER + "qreg q[1];\ngate g a { g a; }\n")

    def test_unknown_formal_parameter_raises(self) -> None:
        with pytest.raises(QASM2ParseError, match="Unknown parameter 'phi'"):
            qasm2_loads(HEADER + "qreg q[1];\ngate g(t) a { rx(phi) a; }\n")

    def test_redefinition_raises(self) -> None:
        with pytest.raises(QASM2ParseError, match="'g' is already defined"):
            qasm2_loads(HEADER + "qreg q[1];\ngate g a { h a; }\ngate g a { x a; }")

    def test_body_rejects_measure(self) -> None:
        with pytest.raises(QASM2ParseError, match="not allowed inside gate"):
            qasm2_loads(HEADER + "qreg q[1];\ngate g a { measure a; }\n")

    def test_importer_is_reusable(self) -> None:
        importer = QASM2Importer()
        importer.imports(HEADER + "qreg q[1];\ngate g a { x a; }\ng q[0];")
        with pytest.raises(QASM2ParseError, match="Unknown gate 'g'"):
            importer.imports(HEADER + "qreg q[1];\ng q[0];")
