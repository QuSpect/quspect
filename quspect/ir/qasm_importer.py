"""OpenQASM 2.0 importer for QuSpect quantum circuits.

This module parses OpenQASM 2.0 text and converts it into
:class:`~quspect.circuit.circuit.QuantumCircuit` instances.

The importer is a small, spec-conforming recursive-descent parser whose
semantics follow :mod:`qiskit.qasm2`:

* Real literals in every OpenQASM form (``.5``, ``5.``, ``1e-3``,
  ``1.5E+2``) and full parameter expressions (``+ - * / ^``, unary signs,
  parentheses, ``pi`` and the functions ``sin cos tan exp ln sqrt``).
* Any number of ``qreg`` / ``creg`` declarations, concatenated in
  declaration order (exactly like qiskit), and whole-register broadcasting
  (``h q;``, ``cx a,b;``, ``measure q -> c;``, ``barrier q;``).
* The complete ``qelib1.inc`` gate library plus the builtin ``U`` / ``CX``
  gates.  Gates with a native QuSpect equivalent are imported as that gate;
  the remaining ``qelib1.inc`` gates are expanded through their library
  definitions into native QuSpect gates.
* User ``gate`` declarations with formal parameters, whose bodies may call
  any previously defined gate (expanded recursively).

Every :class:`QASM2ParseError` carries the line and column of the offending
token.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from quspect.circuit.circuit import QuantumCircuit
from quspect.gates.gate import Gate

__all__ = ["QASM2Importer", "QASM2ParseError", "qasm2_loads"]


# =============================================================================
# Errors
# =============================================================================


class QASM2ParseError(ValueError):
    """Raised when OpenQASM 2.0 parsing fails.

    When the location of the problem is known, the message ends with
    ``(line L, column C)`` and the position is also available through the
    :attr:`line` and :attr:`column` attributes (1-based).

    Args:
        message: Human-readable description of the problem.
        line: 1-based line number of the offending token, if known.
        column: 1-based column number of the offending token, if known.

    Example:
        >>> err = QASM2ParseError("Unknown gate 'foo'", line=4, column=1)
        >>> str(err)
        "Unknown gate 'foo' (line 4, column 1)"
        >>> err.line
        4
    """

    def __init__(
        self, message: str, line: int | None = None, column: int | None = None
    ) -> None:
        self.message = message
        self.line = line
        self.column = column
        if line is not None:
            location = (
                f"line {line}" if column is None else f"line {line}, column {column}"
            )
            message = f"{message} ({location})"
        super().__init__(message)


# =============================================================================
# Tokenization
# =============================================================================


@dataclass(frozen=True)
class _Token:
    """A lexical token with its 1-based source position."""

    kind: str  # "real", "int", "id", "string", "op", "eof"
    value: str
    line: int
    column: int


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t\r\f\v]+)
    | (?P<newline>\n)
    | (?P<comment>//[^\n]*)
    | (?P<string>"[^"\n]*")
    | (?P<real>(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)? | \d+[eE][+-]?\d+)
    | (?P<int>\d+)
    | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<op>->|==|[\[\](){},;+\-*/^])
    """,
    re.VERBOSE,
)


def _tokenize(source: str) -> list[_Token]:
    """Split OpenQASM 2.0 source text into positioned tokens.

    Whitespace and ``//`` comments are dropped.  The returned list always
    ends with an ``"eof"`` token.

    Raises:
        QASM2ParseError: On any character that cannot start a valid token.
    """
    tokens: list[_Token] = []
    line = 1
    line_start = 0
    pos = 0
    n = len(source)
    while pos < n:
        m = _TOKEN_RE.match(source, pos)
        if m is None:
            raise QASM2ParseError(
                f"Invalid character {source[pos]!r}", line, pos - line_start + 1
            )
        kind = m.lastgroup
        assert kind is not None
        if kind == "newline":
            line += 1
            line_start = m.end()
        elif kind not in ("ws", "comment"):
            tokens.append(_Token(kind, m.group(0), line, pos - line_start + 1))
        pos = m.end()
    tokens.append(_Token("eof", "", line, pos - line_start + 1))
    return tokens


# =============================================================================
# Parameter expressions
# =============================================================================

# An expression is compiled to a closure over the formal-parameter binding.
_Expr = Callable[[dict[str, float]], float]

_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
}


def _const(value: float) -> _Expr:
    return lambda _env: value


# =============================================================================
# Gate library
# =============================================================================


@dataclass(frozen=True)
class _NativeGate:
    """A QASM gate that maps one-to-one onto a QuSpect gate."""

    qs_name: str
    num_params: int
    num_qubits: int


# QASM name -> QuSpect gate.  ``U``/``CX`` are the OpenQASM builtins; the rest
# come from ``qelib1.inc``.  Matrices are identical (no global-phase change).
_NATIVE_GATES: dict[str, _NativeGate] = {
    "U": _NativeGate("u3", 3, 1),
    "CX": _NativeGate("cx", 0, 2),
    "id": _NativeGate("id", 0, 1),
    "h": _NativeGate("h", 0, 1),
    "x": _NativeGate("x", 0, 1),
    "y": _NativeGate("y", 0, 1),
    "z": _NativeGate("z", 0, 1),
    "s": _NativeGate("s", 0, 1),
    "sdg": _NativeGate("sdg", 0, 1),
    "t": _NativeGate("t", 0, 1),
    "tdg": _NativeGate("tdg", 0, 1),
    "sx": _NativeGate("sx", 0, 1),
    "sxdg": _NativeGate("sxdg", 0, 1),
    "cx": _NativeGate("cx", 0, 2),
    "cz": _NativeGate("cz", 0, 2),
    "swap": _NativeGate("swap", 0, 2),
    "ccx": _NativeGate("ccx", 0, 3),
    "rx": _NativeGate("rx", 1, 1),
    "ry": _NativeGate("ry", 1, 1),
    "rz": _NativeGate("rz", 1, 1),
    "p": _NativeGate("p", 1, 1),
    "u1": _NativeGate("p", 1, 1),
    "u2": _NativeGate("u2", 2, 1),
    "u3": _NativeGate("u3", 3, 1),
    "u": _NativeGate("u3", 3, 1),
    "crx": _NativeGate("crx", 1, 2),
    "cry": _NativeGate("cry", 1, 2),
    "crz": _NativeGate("crz", 1, 2),
    "cp": _NativeGate("cp", 1, 2),
    "cu1": _NativeGate("cp", 1, 2),
    "cu3": _NativeGate("cu3", 3, 2),
}

_BUILTIN_NAMES: frozenset[str] = frozenset({"U", "CX"})
"""Gates that OpenQASM 2.0 defines even without any include."""

# The ``qelib1.inc`` gates without a native QuSpect counterpart, defined in
# terms of native gates.  Every definition reproduces qiskit's matrix for the
# gate *exactly* (no global-phase difference).  Where the ``qelib1.inc`` body
# only matches up to a global phase, an exact equivalent is used instead:
# ``ch`` (qiskit's CHGate definition),
# ``cu`` = p(gamma) on the control + cu3, ``csx`` via cp, and rz-based
# ``rzz`` / ``rxx``.
_QELIB1_DEFINITIONS = """
gate u0(gamma) q { id q; }
gate cy a,b { sdg b; cx a,b; s b; }
gate ch a,b { s b; h b; t b; cx a,b; tdg b; h b; sdg b; }
gate cswap a,b,c { cx c,b; ccx a,b,c; cx c,b; }
gate csx a,b { h b; cp(pi/2) a,b; h b; }
gate cu(theta,phi,lambda,gamma) c,t { p(gamma) c; cu3(theta,phi,lambda) c,t; }
gate rxx(theta) a,b { h a; h b; cx a,b; rz(theta) b; cx a,b; h b; h a; }
gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }
gate rccx a,b,c {
  u2(0,pi) c; u1(pi/4) c; cx b,c; u1(-pi/4) c; cx a,c; u1(pi/4) c;
  cx b,c; u1(-pi/4) c; u2(0,pi) c;
}
gate rc3x a,b,c,d {
  u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d;
  cx a,d; u1(pi/4) d; cx b,d; u1(-pi/4) d; cx a,d; u1(pi/4) d;
  cx b,d; u1(-pi/4) d; u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d;
  u2(0,pi) d;
}
gate c3x a,b,c,d {
  h d; p(pi/8) a; p(pi/8) b; p(pi/8) c; p(pi/8) d;
  cx a,b; p(-pi/8) b; cx a,b; cx b,c; p(-pi/8) c; cx a,c; p(pi/8) c;
  cx b,c; p(-pi/8) c; cx a,c; cx c,d; p(-pi/8) d; cx b,d; p(pi/8) d;
  cx c,d; p(-pi/8) d; cx a,d; p(pi/8) d; cx c,d; p(-pi/8) d; cx b,d;
  p(pi/8) d; cx c,d; p(-pi/8) d; cx a,d; h d;
}
gate c3sqrtx a,b,c,d {
  h d; cu1(pi/8) a,d; h d; cx a,b;
  h d; cu1(-pi/8) b,d; h d; cx a,b;
  h d; cu1(pi/8) b,d; h d; cx b,c;
  h d; cu1(-pi/8) c,d; h d; cx a,c;
  h d; cu1(pi/8) c,d; h d; cx b,c;
  h d; cu1(-pi/8) c,d; h d; cx a,c;
  h d; cu1(pi/8) c,d; h d;
}
gate c4x a,b,c,d,e {
  h e; cu1(pi/2) d,e; h e; c3x a,b,c,d;
  h e; cu1(-pi/2) d,e; h e; c3x a,b,c,d; c3sqrtx a,b,c,e;
}
"""


@dataclass
class _GateDef:
    """A gate definition (from ``qelib1.inc`` or a user ``gate`` block)."""

    name: str
    params: list[str]
    num_qubits: int
    body: list[_BodyOp] = field(default_factory=list)
    opaque: bool = False
    builtin: bool = False


@dataclass(frozen=True)
class _BodyOp:
    """One gate call inside a ``gate`` declaration body.

    The callee is bound when the declaration is parsed, so later
    declarations can never change the meaning of an existing gate.
    """

    gate: _GateDef
    params: tuple[_Expr, ...]
    qubits: tuple[int, ...]  # indices into the formal qubit list


@dataclass(frozen=True)
class _Register:
    """A declared quantum or classical register."""

    name: str
    offset: int
    size: int


# A resolved operand: list of global bit indices, plus whether it was a whole
# register (broadcast) or a single indexed bit.
_Operand = tuple[list[int], bool]


# =============================================================================
# Parser
# =============================================================================


class _Parser:
    """Recursive-descent parser for one OpenQASM 2.0 program."""

    def __init__(self, source: str, library: dict[str, _GateDef]) -> None:
        self._tokens = _tokenize(source)
        self._pos = 0
        self._gates: dict[str, _GateDef] = dict(library)
        self._qregs: dict[str, _Register] = {}
        self._cregs: dict[str, _Register] = {}
        self._num_qubits = 0
        self._num_clbits = 0
        self._included_qelib = False
        self.circuit = QuantumCircuit()

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> _Token:
        idx = min(self._pos + offset, len(self._tokens) - 1)
        return self._tokens[idx]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        if tok.kind != "eof":
            self._pos += 1
        return tok

    def _error(self, message: str, tok: _Token | None = None) -> QASM2ParseError:
        tok = tok if tok is not None else self._peek()
        return QASM2ParseError(message, tok.line, tok.column)

    @staticmethod
    def _describe(tok: _Token) -> str:
        return "end of input" if tok.kind == "eof" else repr(tok.value)

    def _is_op(self, value: str, offset: int = 0) -> bool:
        tok = self._peek(offset)
        return tok.kind == "op" and tok.value == value

    def _is_id(self, value: str) -> bool:
        tok = self._peek()
        return tok.kind == "id" and tok.value == value

    def _expect_op(self, value: str) -> _Token:
        tok = self._peek()
        if tok.kind != "op" or tok.value != value:
            raise self._error(f"Expected {value!r}, got {self._describe(tok)}", tok)
        return self._advance()

    def _expect_kind(self, kind: str, what: str) -> _Token:
        tok = self._peek()
        if tok.kind != kind:
            raise self._error(f"Expected {what}, got {self._describe(tok)}", tok)
        return self._advance()

    def _expect_keyword(self, value: str) -> _Token:
        tok = self._peek()
        if tok.kind != "id" or tok.value != value:
            raise self._error(f"Expected {value!r}, got {self._describe(tok)}", tok)
        return self._advance()

    # ------------------------------------------------------------------
    # Program
    # ------------------------------------------------------------------

    def parse_program(self) -> QuantumCircuit:
        """Parse a full program (header + statements)."""
        self._expect_keyword("OPENQASM")
        version = self._peek()
        if version.kind not in ("real", "int") or float(version.value) != 2.0:
            raise self._error(
                f"Expected version '2.0', got {self._describe(version)}", version
            )
        self._advance()
        self._expect_op(";")
        while self._peek().kind != "eof":
            self._parse_statement()
        return self.circuit

    def parse_library(self) -> dict[str, _GateDef]:
        """Parse a sequence of gate declarations (the builtin library)."""
        while self._peek().kind != "eof":
            self._parse_gate_declaration(builtin=True)
        return self._gates

    def _parse_statement(self) -> None:
        tok = self._peek()
        if tok.kind == "op" and tok.value == ";":
            self._advance()  # stray semicolon
            return
        if tok.kind != "id":
            raise self._error(f"Unexpected token {self._describe(tok)}", tok)
        kw = tok.value
        if kw == "include":
            self._parse_include()
        elif kw in ("qreg", "creg"):
            self._parse_register()
        elif kw == "gate":
            self._parse_gate_declaration(builtin=False)
        elif kw == "opaque":
            self._parse_opaque()
        elif kw == "barrier":
            self._parse_barrier()
        elif kw == "measure":
            self._parse_measure()
        elif kw == "reset":
            raise self._error(
                "'reset' is not supported by the QuSpect importer "
                "(QuantumCircuit has no reset instruction)",
                tok,
            )
        elif kw == "if":
            self._parse_if()
        elif kw == "OPENQASM":
            raise self._error("Duplicate 'OPENQASM' header", tok)
        else:
            self._parse_gate_application()

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _parse_include(self) -> None:
        self._expect_keyword("include")
        filename = self._expect_kind("string", "a quoted file name")
        self._expect_op(";")
        if filename.value[1:-1] == "qelib1.inc":
            self._included_qelib = True

    def _parse_register(self) -> None:
        kw = self._advance()
        name_tok = self._expect_kind("id", "a register name")
        self._expect_op("[")
        size_tok = self._expect_kind("int", "an integer register size")
        self._expect_op("]")
        self._expect_op(";")
        name = name_tok.value
        if name in self._qregs or name in self._cregs:
            raise self._error(f"Register {name!r} is already defined", name_tok)
        size = int(size_tok.value)
        if size <= 0:
            raise self._error(
                f"Register {name!r} must have a positive size, got {size}", size_tok
            )
        if kw.value == "qreg":
            self._qregs[name] = _Register(name, self._num_qubits, size)
            self._num_qubits += size
        else:
            self._cregs[name] = _Register(name, self._num_clbits, size)
            self._num_clbits += size

    def _check_new_gate_name(self, name_tok: _Token, builtin: bool) -> None:
        """Reject redefinitions (like qiskit).

        A user declaration may shadow a ``qelib1.inc`` gate only when
        ``qelib1.inc`` was not included; ``U`` and ``CX`` can never be
        redefined.
        """
        name = name_tok.value
        existing = self._gates.get(name)
        if existing is None:
            return
        if (
            existing.builtin
            and not builtin
            and not self._included_qelib
            and name not in _BUILTIN_NAMES
        ):
            return
        raise self._error(f"Gate {name!r} is already defined", name_tok)

    def _parse_identifier_list(self, what: str) -> list[_Token]:
        items = [self._expect_kind("id", what)]
        while self._is_op(","):
            self._advance()
            items.append(self._expect_kind("id", what))
        return items

    def _parse_gate_signature(self) -> tuple[_Token, list[str], list[str]]:
        """Parse ``name [(p1, ...)] q1, ...`` of a gate/opaque declaration."""
        name_tok = self._expect_kind("id", "a gate name")
        params: list[str] = []
        if self._is_op("("):
            self._advance()
            if not self._is_op(")"):
                params = [t.value for t in self._parse_identifier_list("a parameter")]
            self._expect_op(")")
        qubit_toks = self._parse_identifier_list("a qubit argument name")
        qubits = [t.value for t in qubit_toks]
        for names, kind in ((params, "parameter"), (qubits, "qubit argument")):
            if len(set(names)) != len(names):
                raise self._error(
                    f"Duplicate {kind} name in declaration of {name_tok.value!r}",
                    name_tok,
                )
        return name_tok, params, qubits

    def _parse_opaque(self) -> None:
        self._expect_keyword("opaque")
        name_tok, params, qubits = self._parse_gate_signature()
        self._expect_op(";")
        self._check_new_gate_name(name_tok, builtin=False)
        self._gates[name_tok.value] = _GateDef(
            name_tok.value, params, len(qubits), opaque=True
        )

    def _parse_gate_declaration(self, *, builtin: bool) -> None:
        self._expect_keyword("gate")
        name_tok, params, qubits = self._parse_gate_signature()
        self._check_new_gate_name(name_tok, builtin)
        name = name_tok.value
        self._expect_op("{")
        body: list[_BodyOp] = []
        while not self._is_op("}"):
            tok = self._peek()
            if tok.kind == "eof":
                raise self._error(f"Unterminated body of gate {name!r}", tok)
            if tok.kind == "id" and tok.value == "barrier":
                # Barriers inside a gate body carry no unitary content.
                self._advance()
                for q in self._parse_identifier_list("a qubit argument name"):
                    self._formal_index(q, qubits)
                self._expect_op(";")
                continue
            if tok.kind == "id" and tok.value in ("measure", "reset", "if"):
                raise self._error(
                    f"{tok.value!r} is not allowed inside gate {name!r}", tok
                )
            body.append(self._parse_body_op(name, params, qubits))
        self._expect_op("}")
        self._gates[name] = _GateDef(name, params, len(qubits), body, builtin=builtin)

    @staticmethod
    def _formal_index(tok: _Token, formals: list[str]) -> int:
        try:
            return formals.index(tok.value)
        except ValueError as err:
            raise QASM2ParseError(
                f"Unknown qubit name {tok.value!r} in gate declaration",
                tok.line,
                tok.column,
            ) from err

    def _parse_body_op(
        self, gate_name: str, formal_params: list[str], formal_qubits: list[str]
    ) -> _BodyOp:
        name_tok = self._expect_kind("id", "a gate name")
        params: list[_Expr] = []
        if self._is_op("("):
            self._advance()
            if not self._is_op(")"):
                params = self._parse_expression_list(formal_params)
            self._expect_op(")")
        qubit_toks = self._parse_identifier_list("a qubit argument name")
        self._expect_op(";")
        qubits = tuple(self._formal_index(t, formal_qubits) for t in qubit_toks)
        callee = self._gates.get(name_tok.value)
        if callee is None:
            raise self._error(
                f"Unknown gate {name_tok.value!r} inside custom gate "
                f"declaration {gate_name!r}",
                name_tok,
            )
        self._check_arity(callee, len(params), len(qubits), name_tok)
        if len(set(qubits)) != len(qubits):
            raise self._error(
                f"Duplicate qubit argument in call to {name_tok.value!r}", name_tok
            )
        return _BodyOp(callee, tuple(params), qubits)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def _parse_expression_list(self, formals: list[str]) -> list[_Expr]:
        exprs = [self._parse_expr(formals)]
        while self._is_op(","):
            self._advance()
            exprs.append(self._parse_expr(formals))
        return exprs

    def _parse_expr(self, formals: list[str]) -> _Expr:
        left = self._parse_term(formals)
        while self._is_op("+") or self._is_op("-"):
            op = self._advance().value
            right = self._parse_term(formals)
            left = _binary(op, left, right)
        return left

    def _parse_term(self, formals: list[str]) -> _Expr:
        left = self._parse_unary(formals)
        while self._is_op("*") or self._is_op("/"):
            op_tok = self._advance()
            right = self._parse_unary(formals)
            if op_tok.value == "*":
                left = _binary("*", left, right)
            else:
                left = _division(left, right, op_tok)
        return left

    def _parse_unary(self, formals: list[str]) -> _Expr:
        if self._is_op("-"):
            self._advance()
            operand = self._parse_unary(formals)
            return lambda env: -operand(env)
        if self._is_op("+"):
            self._advance()
            return self._parse_unary(formals)
        return self._parse_power(formals)

    def _parse_power(self, formals: list[str]) -> _Expr:
        base = self._parse_atom(formals)
        if self._is_op("^"):
            op_tok = self._advance()
            exponent = self._parse_unary(formals)  # right-associative
            return _power(base, exponent, op_tok)
        return base

    def _parse_atom(self, formals: list[str]) -> _Expr:
        tok = self._peek()
        if tok.kind in ("real", "int"):
            self._advance()
            return _const(float(tok.value))
        if tok.kind == "op" and tok.value == "(":
            self._advance()
            inner = self._parse_expr(formals)
            self._expect_op(")")
            return inner
        if tok.kind == "id":
            self._advance()
            if tok.value == "pi":
                return _const(math.pi)
            if tok.value in _FUNCTIONS:
                self._expect_op("(")
                arg = self._parse_expr(formals)
                self._expect_op(")")
                return _function(tok.value, arg, tok)
            if tok.value in formals:
                name = tok.value
                return lambda env: env[name]
            raise self._error(f"Unknown parameter {tok.value!r} in expression", tok)
        raise self._error(f"Expected an expression, got {self._describe(tok)}", tok)

    # ------------------------------------------------------------------
    # Operands
    # ------------------------------------------------------------------

    def _parse_operand(self, registers: dict[str, _Register], kind: str) -> _Operand:
        name_tok = self._expect_kind("id", f"a {kind} register name")
        reg = registers.get(name_tok.value)
        if reg is None:
            raise self._error(f"Unknown {kind} register {name_tok.value!r}", name_tok)
        if not self._is_op("["):
            return list(range(reg.offset, reg.offset + reg.size)), True
        self._advance()
        idx_tok = self._expect_kind("int", "an integer index")
        self._expect_op("]")
        idx = int(idx_tok.value)
        if idx >= reg.size:
            raise self._error(
                f"Index {idx} out of range for register {reg.name!r} of size "
                f"{reg.size}",
                idx_tok,
            )
        return [reg.offset + idx], False

    def _parse_qubit_operands(self) -> list[_Operand]:
        ops = [self._parse_operand(self._qregs, "quantum")]
        while self._is_op(","):
            self._advance()
            ops.append(self._parse_operand(self._qregs, "quantum"))
        return ops

    def _broadcast(self, operands: list[_Operand], tok: _Token) -> list[list[int]]:
        """Expand register operands per OpenQASM 2.0 broadcasting rules."""
        sizes = {len(bits) for bits, whole in operands if whole}
        if len(sizes) > 1:
            raise self._error("Register arguments must all have the same size", tok)
        if not sizes:
            return [[bits[0] for bits, _ in operands]]
        size = sizes.pop()
        return [
            [bits[i] if whole else bits[0] for bits, whole in operands]
            for i in range(size)
        ]

    # ------------------------------------------------------------------
    # Quantum operations
    # ------------------------------------------------------------------

    def _parse_barrier(self) -> None:
        self._expect_keyword("barrier")
        if self._is_op(";"):
            self._advance()
            self.circuit.barrier()
            return
        qubits: list[int] = []
        for bits, _ in self._parse_qubit_operands():
            qubits.extend(b for b in bits if b not in qubits)
        self._expect_op(";")
        self.circuit.barrier(*qubits)

    def _parse_measure(self) -> None:
        kw = self._expect_keyword("measure")
        q_bits, q_whole = self._parse_operand(self._qregs, "quantum")
        self._expect_op("->")
        c_bits, c_whole = self._parse_operand(self._cregs, "classical")
        self._expect_op(";")
        if q_whole != c_whole or len(q_bits) != len(c_bits):
            raise self._error(
                "measure requires a single qubit and clbit, or registers of the "
                "same size",
                kw,
            )
        for q, c in zip(q_bits, c_bits, strict=True):
            self.circuit.measure([q], [c])

    def _parse_if(self) -> None:
        kw = self._expect_keyword("if")
        self._expect_op("(")
        reg_tok = self._expect_kind("id", "a classical register name")
        if reg_tok.value not in self._cregs:
            raise self._error(f"Unknown classical register {reg_tok.value!r}", reg_tok)
        self._expect_op("==")
        self._expect_kind("int", "an integer")
        self._expect_op(")")
        raise self._error(
            "Classically controlled 'if' statements are not supported by the "
            "QuSpect importer (QuantumCircuit has no classical control flow)",
            kw,
        )

    def _parse_gate_application(self) -> None:
        name_tok = self._expect_kind("id", "a gate name")
        params: list[float] = []
        if self._is_op("("):
            self._advance()
            if not self._is_op(")"):
                params = [e({}) for e in self._parse_expression_list([])]
            self._expect_op(")")
        operands = self._parse_qubit_operands()
        self._expect_op(";")
        gate = self._gates.get(name_tok.value)
        if gate is None:
            raise self._error(
                f"Unknown gate {name_tok.value!r}. Add a custom gate declaration "
                f"before using it, or check that the gate is supported.",
                name_tok,
            )
        self._check_arity(gate, len(params), len(operands), name_tok)
        for qubits in self._broadcast(operands, name_tok):
            if len(set(qubits)) != len(qubits):
                raise self._error(
                    f"Duplicate qubit argument in call to {name_tok.value!r}",
                    name_tok,
                )
            self._apply(gate, params, qubits, name_tok)

    def _check_arity(
        self, gate: _GateDef, num_params: int, num_qubits: int, tok: _Token
    ) -> None:
        expected = len(gate.params)
        if num_params != expected:
            if num_params == 0:
                raise self._error(
                    f"Gate {gate.name!r} requires parameters but none were given "
                    f"(expects {expected})",
                    tok,
                )
            raise self._error(
                f"Gate {gate.name!r} expects {expected} parameters, got {num_params}",
                tok,
            )
        if num_qubits != gate.num_qubits:
            raise self._error(
                f"Gate {gate.name!r} expects {gate.num_qubits} qubits, "
                f"got {num_qubits}",
                tok,
            )

    def _apply(
        self, gate: _GateDef, params: list[float], qubits: list[int], tok: _Token
    ) -> None:
        """Append ``gate`` (native, or expanded through its definition)."""
        native = _NATIVE_GATES.get(gate.name) if gate.builtin else None
        if native is not None:
            self.circuit.append(_make_native(native, params), tuple(qubits))
            return
        if gate.opaque:
            raise self._error(
                f"Opaque gate {gate.name!r} has no definition and cannot be imported",
                tok,
            )
        env = dict(zip(gate.params, params, strict=True))
        for op in gate.body:
            try:
                values = [expr(env) for expr in op.params]
            except QASM2ParseError as err:
                raise self._error(
                    f"{err.message} (in gate {gate.name!r})", tok
                ) from err
            self._apply(op.gate, values, [qubits[i] for i in op.qubits], tok)


def _binary(op: str, left: _Expr, right: _Expr) -> _Expr:
    if op == "+":
        return lambda env: left(env) + right(env)
    if op == "-":
        return lambda env: left(env) - right(env)
    return lambda env: left(env) * right(env)


def _division(left: _Expr, right: _Expr, tok: _Token) -> _Expr:
    def evaluate(env: dict[str, float]) -> float:
        denominator = right(env)
        if denominator == 0.0:
            raise QASM2ParseError("Division by zero", tok.line, tok.column)
        return left(env) / denominator

    return evaluate


def _power(base: _Expr, exponent: _Expr, tok: _Token) -> _Expr:
    def evaluate(env: dict[str, float]) -> float:
        try:
            result = base(env) ** exponent(env)
        except (OverflowError, ZeroDivisionError) as err:
            raise QASM2ParseError(
                f"Invalid power: {err}", tok.line, tok.column
            ) from err
        if isinstance(result, complex):
            raise QASM2ParseError(
                "Power expression has a complex result", tok.line, tok.column
            )
        return float(result)

    return evaluate


def _function(name: str, arg: _Expr, tok: _Token) -> _Expr:
    func = _FUNCTIONS[name]

    def evaluate(env: dict[str, float]) -> float:
        try:
            return func(arg(env))
        except (ValueError, OverflowError) as err:
            raise QASM2ParseError(
                f"Invalid argument to {name}(): {err}", tok.line, tok.column
            ) from err

    return evaluate


def _make_native(native: _NativeGate, params: list[float]) -> Gate:
    """Build the QuSpect gate for a native QASM gate application."""
    if native.num_params == 0:
        return _lookup_standard_gate(native.qs_name)
    return _lookup_parameterized_gate(native.qs_name).with_params(*params)


def _build_library() -> dict[str, _GateDef]:
    """Build the builtin gate library (``U``, ``CX`` and ``qelib1.inc``)."""
    library: dict[str, _GateDef] = {}
    for name, native in _NATIVE_GATES.items():
        params = [f"p{i}" for i in range(native.num_params)]
        library[name] = _GateDef(name, params, native.num_qubits, builtin=True)
    return _Parser(_QELIB1_DEFINITIONS, library).parse_library()


@functools.cache
def _library() -> dict[str, _GateDef]:
    """Return the (lazily built, shared, read-only) builtin gate library."""
    return _build_library()


# =============================================================================
# Public API
# =============================================================================


class QASM2Importer:
    """Import an OpenQASM 2.0 string into a QuSpect QuantumCircuit.

    The importer implements the full OpenQASM 2.0 grammar except classical
    control (``if``) and ``reset``, which QuSpect circuits cannot represent
    and which are therefore reported as :class:`QASM2ParseError`.  Semantics
    follow :mod:`qiskit.qasm2`:

    * Multiple ``qreg`` / ``creg`` declarations are concatenated in
      declaration order; register names map to offsets in that order.
    * Whole-register operands broadcast (``h q;``, ``cx a,b;``,
      ``cx a[0],b;``, ``measure q -> c;``, ``barrier q;``).
    * Parameters are arbitrary OpenQASM expressions.
    * Every ``qelib1.inc`` gate and the builtins ``U`` / ``CX`` are
      available.  Gates with a native QuSpect counterpart are imported as
      that gate (``u1`` -> ``p``, ``cu1`` -> ``cp``, ``U``/``u`` -> ``u3``,
      ``CX`` -> ``cx``, ``sx``/``sxdg`` -> themselves); all other library
      gates (``ch``, ``cswap``,
      ``rzz``, ``c3x``, ...) are expanded through their definitions into
      native QuSpect gates.  Every library gate is imported with exactly
      qiskit's matrix (no global-phase difference).
    * User ``gate`` declarations may have formal parameters and may call
      previously declared gates; they are expanded inline.
    * Gate parameter and qubit counts are validated.

    Example:
        >>> from quspect.ir.qasm_importer import QASM2Importer
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q -> c;
        ... '''
        >>> importer = QASM2Importer()
        >>> circ = importer.imports(qasm)
        >>> circ.num_qubits
        2
    """

    def imports(self, source: str) -> QuantumCircuit:
        """Parse an OpenQASM 2.0 string and return a QuantumCircuit.

        Each call is independent: custom gates and registers declared in one
        call are not visible in another.

        Args:
            source: OpenQASM 2.0 text.

        Returns:
            A :class:`QuantumCircuit` instance.  Qubits and clbits of all
            registers are numbered in declaration order.

        Raises:
            QASM2ParseError: If the source is malformed, uses an unknown or
                mis-applied gate, or uses an unsupported construct
                (``if``, ``reset``, opaque gates).  The message includes the
                line and column of the problem.

        Example:
            >>> circ = QASM2Importer().imports(
            ...     'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; rx(pi/2) q[0];'
            ... )
            >>> circ[0].gate.params
            (1.5707963267948966,)
        """
        return _Parser(source, _library()).parse_program()


def qasm2_loads(source: str) -> QuantumCircuit:
    """Parse an OpenQASM 2.0 string into a QuantumCircuit.

    This is the main entry point for importing QASM text; it is equivalent
    to ``QASM2Importer().imports(source)``.

    Args:
        source: OpenQASM 2.0 text.

    Returns:
        A :class:`QuantumCircuit` instance.

    Raises:
        QASM2ParseError: If the source is malformed or uses an unknown or
            unsupported gate or construct.  The message includes the line
            and column of the problem.

    Example:
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... '''
        >>> circ = qasm2_loads(qasm)
        >>> circ.num_qubits
        2
    """
    return QASM2Importer().imports(source)


# =============================================================================
# Standard gate lookup
# =============================================================================


def _lookup_standard_gate(name: str) -> Gate:
    """Look up a standard (non-parameterized) gate singleton by name.

    Raises:
        ValueError: If the gate name is unknown.
    """
    from quspect.gates.standard import ALL_STANDARD

    for g in ALL_STANDARD:
        if g.name == name:
            return g
    raise ValueError(f"Unknown standard gate: {name!r}")


def _lookup_parameterized_gate(name: str) -> Gate:
    """Look up a parameterized gate template by name.

    Raises:
        ValueError: If the gate name is unknown.
    """
    from quspect.gates.parameterized import ALL_PARAMETERIZED

    for g in ALL_PARAMETERIZED:
        if g.name == name:
            return g
    raise ValueError(f"Unknown parameterized gate: {name!r}")
