"""Differential tests: the C++ OpenQASM 2.0 fast path against the Python parser.

:func:`~quspect.ir.qasm2_loads` first tries the native parser and falls back
to the Python reference parser when the native one returns ``None``.  The
contract checked here: whenever the reference parser raises, the native
parser declines (so the user sees the reference exception, line and column
included); whenever both accept, the circuits are identical — same gate
objects and operands — and the parameters agree to rounding.  They are not
compared bit for bit: the native parser evaluates ``sin`` / ``cos`` / ``tan``
/ ``exp`` / ``ln`` / ``^`` with correctly rounded functions (identical on
every platform), the reference parser with the platform's libm, which may
differ in the last bits.  Expressions without those functions are
bit-identical (same IEEE operations in the same order).
"""

from __future__ import annotations

import math
import random

import pytest

from quspect.circuit._native import native_ir
from quspect.ir import QASM2ParseError, qasm2_loads
from quspect.ir.qasm_importer import _QELIB1_DEFINITIONS, _library, _Parser

pytestmark = pytest.mark.skipif(native_ir() is None, reason="native core not built")

_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _reference(source: str) -> object:
    try:
        return _Parser(source, _library()).parse_program()
    except (QASM2ParseError, RecursionError) as exc:
        return exc


def _key(circuit: object) -> list[tuple[object, ...]]:
    out = []
    for ins in circuit:  # type: ignore[attr-defined]
        gate = ins.gate
        name = None if gate is None else (gate.name, gate.num_qubits, len(gate.params))
        label = None if gate is None else gate.label
        out.append((ins.type, ins.qubits, ins.clbits, name, label, ins.label))
    return out


def _params(circuit: object) -> list[float]:
    return [
        float(p)
        for ins in circuit  # type: ignore[attr-defined]
        if ins.gate is not None
        for p in ins.gate.params
    ]


def _same_circuit(got: object, ref: object, source: str) -> None:
    """Same gates and operands; parameters equal up to a few ulp."""
    assert _key(got) == _key(ref), source
    pa, pb = _params(got), _params(ref)
    assert len(pa) == len(pb), source
    for a, b in zip(pa, pb, strict=True):
        assert a == pytest.approx(b, rel=1e-14, abs=1e-300), source


def _check(source: str) -> str:
    """Compare both parsers on ``source``.

    Returns:
        ``"rejected"`` (both parsers reject), ``"native"`` (fast path taken)
        or ``"fallback"`` (valid input handled by the reference parser).
    """
    core = native_ir()
    assert core is not None
    ref = _reference(source)
    native = core._qasm2_parse(source, _QELIB1_DEFINITIONS)
    if isinstance(ref, Exception):
        assert native is None, source
        with pytest.raises(type(ref)) as info:
            qasm2_loads(source)
        assert str(info.value) == str(ref)
        return "rejected"
    if native is not None:
        got = native.to_instructions()
        _same_circuit(got, ref, source)
        for a, b in zip(got, ref, strict=True):
            if a.gate is not None and not a.gate.params:
                assert a.gate is b.gate  # fixed gates are the shared singletons
    _same_circuit(qasm2_loads(source).instructions, ref.instructions, source)  # type: ignore[attr-defined]
    return "fallback" if native is None else "native"


_GATES_1Q = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "id"]
_GATES_1P = ["rx", "ry", "rz", "p", "u1"]
_GATES_2Q = ["cx", "cz", "swap", "CX", "cy", "ch", "csx"]
_GATES_2P = ["crx", "cry", "crz", "cp", "cu1", "rzz", "rxx"]
_ATOMS = ["pi", "1", "2.5", ".5", "3.", "1e-3", "2E+2", "0.1"]


def _expr(rng: random.Random, depth: int = 0) -> str:
    r = rng.random()
    if depth > 2 or r < 0.35:
        return rng.choice(_ATOMS)
    if r < 0.5:
        return f"-{_expr(rng, depth + 1)}"
    if r < 0.6:
        f = rng.choice(["sin", "cos", "tan", "exp", "sqrt", "ln"])
        arg = rng.choice(["pi/7", "0.3", "2", "1.5"])
        return f"{f}({arg})"
    if r < 0.7:
        return f"({_expr(rng, depth + 1)})"
    op = rng.choice(["+", "-", "*", "/", "^"])
    if op == "^":
        base = rng.choice(["2", "1.5", "(-2)", "0.5"])
        return f"{base}^{rng.choice(['2', '3', '-1', '0.5'])}"
    return f"{_expr(rng, depth + 1)}{op}{_expr(rng, depth + 1)}"


def _program(rng: random.Random) -> str:
    lines = [_HEADER]
    sizes = [rng.randrange(1, 4) for _ in range(rng.randrange(1, 3))]
    regs = [f"r{i}" for i in range(len(sizes))]
    for name, size in zip(regs, sizes, strict=True):
        lines.append(f"qreg {name}[{size}];")
    lines.append(f"creg c[{sum(sizes)}];")
    if rng.random() < 0.5:
        lines.append(
            "gate g(a, b) u, v { rz(a*b/2) u; cx u, v; rzz(a-b) v, u; barrier u; }"
        )
    bits = [f"{n}[{k}]" for n, s in zip(regs, sizes, strict=True) for k in range(s)]
    for _ in range(rng.randrange(1, 25)):
        r = rng.random()
        if r < 0.25:
            lines.append(f"{rng.choice(_GATES_1Q)} {rng.choice(bits + regs)};")
        elif r < 0.45:
            lines.append(f"{rng.choice(_GATES_1P)}({_expr(rng)}) {rng.choice(bits)};")
        elif r < 0.65 and len(bits) > 1:
            a, b = rng.sample(bits, 2)
            name = rng.choice(_GATES_2Q + _GATES_2P)
            params = f"({_expr(rng)})" if name in _GATES_2P else ""
            lines.append(f"{name}{params} {a},{b};")
        elif r < 0.7:
            lines.append(
                f"u3({_expr(rng)},{_expr(rng)},{_expr(rng)}) {rng.choice(bits)};"
            )
        elif r < 0.75 and len(bits) > 1:
            a, b = rng.sample(bits, 2)
            lines.append(f"g({_expr(rng)}, {_expr(rng)}) {a}, {b};")
        elif r < 0.8:
            lines.append(
                "barrier;" if rng.random() < 0.5 else f"barrier {rng.choice(regs)};"
            )
        elif r < 0.85 and len(bits) >= 3:
            a, b, c = rng.sample(bits, 3)
            lines.append(f"{rng.choice(['ccx', 'cswap', 'rccx'])} {a},{b},{c};")
        else:
            k = rng.randrange(len(bits))
            lines.append(f"measure {bits[k]} -> c[{k}];")
    return "\n".join(lines) + "\n"


def _mutate(rng: random.Random, source: str) -> str:
    """Introduce a random (usually invalid) edit."""
    pos = rng.randrange(len(source))
    choice = rng.random()
    if choice < 0.4:
        return source[:pos] + source[pos + 1 :]
    if choice < 0.8:
        junk = rng.choice(
            [";", ",", "(", ")", "[", "]", "q", "1", "-", "^", "/0", "$", "pi", "e"]
        )
        return source[:pos] + junk + source[pos:]
    return source[:pos] + source[pos + rng.randrange(1, 8) :]


@pytest.mark.parametrize("seed", range(40))
def test_random_programs_match(seed: int) -> None:
    rng = random.Random(seed)
    outcomes = [_check(_program(rng)) for _ in range(10)]
    assert "fallback" not in outcomes  # every valid program takes the fast path


@pytest.mark.parametrize("seed", range(40))
def test_mutated_programs_match(seed: int) -> None:
    rng = random.Random(1000 + seed)
    for _ in range(10):
        _check(_mutate(rng, _program(rng)))


@pytest.mark.parametrize(
    "body",
    [
        "qreg q[1]; rz(1e999) q[0];",
        "qreg q[1]; rz(-0.0) q[0]; rz(0^3) q[0]; rz((-0.0)^3) q[0];",
        "qreg q[1]; rz(2^1024) q[0];",
        "qreg q[1]; rz(exp(1000)) q[0];",
        "qreg q[1]; rz((-1)^0.5) q[0];",
        "qreg q[1]; rz(1/(1-1)) q[0];",
        "qreg q[1]; rz(١) q[0];",
        "qreg q[2]; gate h a { x a; } h q[0];",
        "qreg q[99999999999999999999];",
        "qreg q[3]; creg c[3]; measure q -> c; barrier; reset q[0];",
        "qreg q[1]; opaque o(a) b; rz(0.5) q[0];",
        "qreg q[1]; gate g a { } g q[0];",
    ],
)
def test_edge_cases_match(body: str) -> None:
    _check("OPENQASM 2.0;\n" + body)
    _check(_HEADER + body)


def test_deeply_nested_user_gates_fall_back() -> None:
    lines = [_HEADER, "qreg q[1];", "gate g0 a { x a; }"]
    lines += [f"gate g{i} a {{ g{i - 1} a; }}" for i in range(1, 150)]
    lines.append("g149 q[0];")
    source = "\n".join(lines)
    core = native_ir()
    assert core is not None
    assert core._qasm2_parse(source, _QELIB1_DEFINITIONS) is None
    _check(source)


def test_parameters_are_bit_exact() -> None:
    source = _HEADER + "qreg q[1];\nrz(pi/3 + 0.1*7 - sqrt(2)^3 / 7) q[0];\n"
    (ins,) = qasm2_loads(source)
    assert ins.gate is not None
    expected = math.pi / 3 + 0.1 * 7 - math.sqrt(2) ** 3 / 7
    assert ins.gate.params[0].hex() == expected.hex()
