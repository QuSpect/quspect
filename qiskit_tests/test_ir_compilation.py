"""Cross-validate :mod:`quspect.ir.compilation` decomposition rules with qiskit.

Every built-in :class:`GateDecompositionRule` must reproduce the qiskit
unitary of the original gate (up to a global phase).  The custom-rule
registry is global, so every test runs inside a fixture that clears it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

import quspect
from qiskit_tests._ir_helpers import unitary_only
from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_unitary_equal,
    global_phase,
    qiskit_circuit_unitary,
    quspect_circuit_unitary,
    random_circuit_pair,
)
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.gates.gate import Gate
from quspect.gates.standard import CX, H, X
from quspect.ir import (
    GateDecompositionRule,
    clear_extra_rules,
    decompose_circuit,
    decompose_deep,
    decompose_gate,
    get_rules_for,
    register_decomposition,
)
from quspect.ir.compilation import _EXTRA_RULES, BUILTIN_DECOMPOSITIONS

BUILTIN_NAMES = sorted({r.gate_name for r in BUILTIN_DECOMPOSITIONS})

# Rules that reproduce the unitary exactly (no global phase); the others
# (P/U1 -> RZ, U3 -> RZ RY RZ) are only equal up to a global phase.
EXACT_RULES = {
    "swap", "ccx", "cx", "rx", "ry", "u2", "crx", "cry", "crz", "cp", "cu3",
}  # fmt: skip

BASIS = {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rz", "cx", "cz"}


@pytest.fixture(autouse=True)
def _clean_rules() -> Iterator[None]:
    """Reset the global decomposition-rule registry around every test."""
    clear_extra_rules()
    yield
    clear_extra_rules()


def _random_params(rng: np.random.Generator, n: int) -> list[float]:
    """``n`` angles in ``[-2pi, 2pi]``."""
    return [float(x) for x in rng.uniform(-2 * math.pi, 2 * math.pi, n)]


def _circuit_from_pairs(
    pairs: list[tuple[Gate, tuple[int, ...]]], num_qubits: int
) -> quspect.QuantumCircuit:
    """Build a circuit from ``(gate, qubits)`` pairs pinned to ``num_qubits``."""
    circ = quspect.QuantumCircuit()
    circ.id(num_qubits - 1)
    for gate, qubits in pairs:
        circ.append(gate, qubits)
    return circ


# =============================================================================
# Built-in rules
# =============================================================================


def test_builtin_rule_inventory() -> None:
    """Built-in rules cover the documented gates once each."""
    assert (
        sorted(
            ["swap", "ccx", "cx", "cz", "rx", "ry", "p", "u1", "u2", "u3"]
            + ["crz", "crx", "cry", "cp", "cu3", "sx", "sxdg"]
        )
        == BUILTIN_NAMES
    )
    for rule in BUILTIN_DECOMPOSITIONS:
        assert rule.num_qubits == GATE_SPECS[rule.gate_name].num_qubits
        assert rule.description
        assert rule.condition is None


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", BUILTIN_NAMES)
@pytest.mark.parametrize("seed", range(6))
def test_builtin_rule_matches_qiskit_unitary(name: str, seed: int) -> None:
    """Decomposition == original gate (qiskit unitary), on permuted qubits."""
    spec = GATE_SPECS[name]
    rng = np.random.default_rng(seed)
    params = _random_params(rng, spec.num_params)
    width = spec.num_qubits + 1
    qubits = tuple(int(q) for q in rng.permutation(width)[: spec.num_qubits])
    gate = spec.quspect_gate(*params)

    original = _circuit_from_pairs([(gate, qubits)], width)
    decomposed = _circuit_from_pairs(decompose_gate(gate, qubits), width)
    u_ref = qiskit_circuit_unitary(original)
    u_dec = qiskit_circuit_unitary(decomposed)
    phase = assert_unitary_equal(u_ref, u_dec, up_to_phase=True, atol=1e-9)
    if name in EXACT_RULES:
        assert abs(phase - 1) < 1e-9, f"{name} rule introduced phase {phase}"
    # The quspect simulator agrees with qiskit on the decomposed circuit.
    assert_unitary_equal(quspect_circuit_unitary(decomposed), u_dec, atol=1e-9)
    # All emitted qubits are among the original operands.
    for _, sub_qubits in decompose_gate(gate, qubits):
        assert set(sub_qubits) <= set(qubits)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", ["p", "u1", "u3"])
def test_phase_carrying_rules_differ_by_global_phase(name: str) -> None:
    """P/U1 -> RZ and U3 -> RZ.RY.RZ are correct only up to a global phase."""
    spec = GATE_SPECS[name]
    params = [0.9, -0.4, 1.3][: spec.num_params]
    gate = spec.quspect_gate(*params)
    u_ref = qiskit_circuit_unitary(_circuit_from_pairs([(gate, (0,))], 1))
    u_dec = qiskit_circuit_unitary(_circuit_from_pairs(decompose_gate(gate, (0,)), 1))
    phase = global_phase(u_ref, u_dec)
    assert abs(phase - 1) > 1e-3
    assert_unitary_equal(u_ref, u_dec, up_to_phase=True)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", [n for n in BUILTIN_NAMES if GATE_SPECS[n].num_params])
def test_builtin_rule_keeps_symbolic_parameters(name: str) -> None:
    """Rules propagate Parameters symbolically; binding later gives the same unitary."""
    spec = GATE_SPECS[name]
    syms = [Parameter(f"a{i}") for i in range(spec.num_params)]
    values = [0.7, -1.1, 2.2][: spec.num_params]
    qubits = tuple(range(spec.num_qubits))
    gate = spec.quspect_gate(*syms)
    decomposed = _circuit_from_pairs(decompose_gate(gate, qubits), spec.num_qubits)
    assert {p.name for p in decomposed.parameters} <= {s.name for s in syms}
    assert decomposed.parameters  # symbols survive
    bound = decomposed.bind_parameters(dict(zip(syms, values, strict=True)))
    ref = _circuit_from_pairs([(spec.quspect_gate(*values), qubits)], spec.num_qubits)
    assert_unitary_equal(
        qiskit_circuit_unitary(ref),
        qiskit_circuit_unitary(bound),
        up_to_phase=True,
        atol=1e-9,
    )


def test_crz_rule_produces_affine_expression() -> None:
    """CRZ(theta) emits RZ(theta/2) and RZ(-theta/2) as ParameterExpressions."""
    theta = Parameter("theta")
    pairs = decompose_gate(GATE_SPECS["crz"].quspect_gate(theta), (0, 1))
    halves = [g.params[0] for g, _ in pairs if g.name == "rz"]
    assert all(isinstance(h, ParameterExpression) for h in halves)
    assert [h.evaluate({theta: 1.0}) for h in halves] == [0.5, -0.5]


# =============================================================================
# Rule registry
# =============================================================================


def test_get_rules_for_unknown_returns_empty() -> None:
    """Unknown names / wrong arity yield no rule."""
    assert get_rules_for("nope", 1) == []
    assert get_rules_for("cx", 3) == []


def test_decompose_gate_without_rule_raises() -> None:
    """``decompose_gate`` raises ValueError when nothing matches."""
    with pytest.raises(ValueError, match="No decomposition rule"):
        decompose_gate(H, (0,))


def test_register_and_clear_custom_rule() -> None:
    """A registered rule is found and applied; ``clear_extra_rules`` removes it."""

    def rule(gate: Gate, qs: tuple[int, ...]) -> list[tuple[Gate, tuple[int, ...]]]:
        return [(X, (qs[0],)), (X, (qs[0],)), (H, (qs[0],))]

    custom = GateDecompositionRule("myh", 1, rule, description="trivial")
    register_decomposition(custom)
    assert get_rules_for("myh", 1) == [custom]
    assert decompose_gate(Gate(name="myh", num_qubits=1), (4,)) == [
        (X, (4,)),
        (X, (4,)),
        (H, (4,)),
    ]
    clear_extra_rules()
    assert get_rules_for("myh", 1) == []
    assert _EXTRA_RULES == []


def test_priority_ordering_and_tie_break() -> None:
    """Higher priority first; on ties built-in rules precede custom ones."""
    low = GateDecompositionRule("cx", 2, lambda g, q: [], priority=1)
    tie = GateDecompositionRule("cx", 2, lambda g, q: [], priority=5)
    high = GateDecompositionRule("cx", 2, lambda g, q: [(H, (q[0],))], priority=99)
    for r in (low, tie, high):
        register_decomposition(r)
    rules = get_rules_for("cx", 2)
    assert rules[0] is high
    assert rules[1].description == "CX → H + CZ + H"  # built-in, priority 5
    assert rules[2] is tie
    assert rules[3] is low
    assert decompose_gate(CX, (0, 1)) == [(H, (0,))]


def test_condition_skips_rule() -> None:
    """A rule whose condition is false is skipped in favour of the next one."""
    seen: list[Gate] = []

    def never(gate: Gate) -> bool:
        seen.append(gate)
        return False

    blocked = GateDecompositionRule(
        "swap", 2, lambda g, q: [(H, (q[0],))], condition=never, priority=50
    )
    register_decomposition(blocked)
    result = decompose_gate(GATE_SPECS["swap"].quspect_gate(), (1, 0))
    assert result == [(CX, (1, 0)), (CX, (0, 1)), (CX, (1, 0))]
    assert len(seen) == 1

    only_conditional = GateDecompositionRule(
        "cond", 1, lambda g, q: [(H, (q[0],))], condition=lambda g: g.params == (1.0,)
    )
    register_decomposition(only_conditional)
    assert decompose_gate(Gate(name="cond", num_qubits=1, params=(1.0,)), (0,)) == [
        (H, (0,))
    ]
    with pytest.raises(ValueError):
        decompose_gate(Gate(name="cond", num_qubits=1, params=(2.0,)), (0,))


def test_rule_dataclass_is_frozen() -> None:
    """GateDecompositionRule is immutable with documented defaults."""
    rule = GateDecompositionRule("g", 1, lambda g, q: [])
    assert (rule.condition, rule.priority, rule.description) == (None, 0, "")
    with pytest.raises(AttributeError):
        rule.priority = 3  # type: ignore[misc]


# =============================================================================
# decompose_circuit / decompose_deep
# =============================================================================


def test_decompose_circuit_default_supported_set_is_identity() -> None:
    """With the QASM registry as supported set every built-in gate passes."""
    circ, _ = random_circuit_pair(3, 30, 7)
    circ.measure([0], [0])
    out = decompose_circuit(circ)
    assert out.instructions == circ.instructions
    assert out is not circ


@pytest.mark.qiskit_ref
def test_decompose_circuit_single_pass_and_passthrough() -> None:
    """One pass only; measure/barrier/channel/unknown gates pass through."""
    circ = quspect.QuantumCircuit(name="demo")
    circ.ry(0.4, 0).barrier(0, 1)
    circ.depolarize(0.1, 1)
    circ.append(Gate(name="mystery", num_qubits=1), (1,))
    circ.measure([0, 1], [1, 0])
    labelled = CircuitInstruction(
        type=InstructionType.GATE, gate=CX, qubits=(0, 1), label="entangle"
    )
    circ._instructions.append(labelled)  # instruction label is not in the fluent API

    out = decompose_circuit(circ, supported_gates={"h", "s", "sdg", "rx", "cz"})
    assert out.name == "demo"
    names = [(i.type.name, i.gate.name if i.gate else None) for i in out.instructions]
    assert names == [
        ("GATE", "sdg"),
        ("GATE", "rx"),  # single pass: RX is supported here, not lowered further
        ("GATE", "s"),
        ("BARRIER", None),
        ("CHANNEL", None),
        ("GATE", "mystery"),
        ("MEASURE", None),
        ("GATE", "h"),
        ("GATE", "cz"),
        ("GATE", "h"),
    ]
    assert out[4].channel is circ[2].channel
    assert out[6] == circ[4]
    assert {i.label for i in out.instructions[-3:]} == {"entangle"}

    nonrec = decompose_circuit(circ, supported_gates={"h", "s", "sdg", "cz"})
    assert [i.gate.name for i in nonrec.instructions[:3] if i.gate] == [
        "sdg",
        "rx",
        "s",
    ]


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(8))
def test_decompose_deep_to_basis_matches_qiskit(seed: int) -> None:
    """``decompose_deep`` lowers random circuits to a basis; unitary unchanged."""
    circ, qc = random_circuit_pair(3, 25, seed)
    out = decompose_deep(circ, supported_gates=BASIS)
    assert {i.gate.name for i in out.instructions if i.gate} <= BASIS
    assert_unitary_equal(
        qiskit_circuit_unitary(unitary_only(out), num_qubits=3),
        qiskit_circuit_unitary(qc),
        up_to_phase=True,
        atol=1e-8,
    )


def test_decompose_deep_stops_when_no_progress() -> None:
    """Gates without rules are kept once no further progress is possible."""
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="mystery", num_qubits=1), (0,))
    circ.rx(0.3, 0)
    out = decompose_deep(circ, supported_gates={"h", "rz"})
    assert [i.gate.name for i in out.instructions if i.gate] == [
        "mystery",
        "h",
        "rz",
        "h",
    ]


def test_decompose_deep_already_supported_returns_first_pass() -> None:
    """A circuit already in the basis converges after one pass."""
    circ = quspect.QuantumCircuit()
    circ.h(0).cx(0, 1)
    assert decompose_deep(circ, supported_gates={"h", "cx"}).instructions == (
        circ.instructions
    )


def test_decompose_deep_raises_when_not_converging() -> None:
    """A rule that keeps growing raises RecursionError after ``max_passes``."""

    def grow(gate: Gate, qs: tuple[int, ...]) -> list[tuple[Gate, tuple[int, ...]]]:
        return [(gate, qs), (gate, qs)]

    register_decomposition(GateDecompositionRule("grow", 1, grow))
    circ = quspect.QuantumCircuit()
    circ.append(Gate(name="grow", num_qubits=1), (0,))
    circ.append(Gate(name="other", num_qubits=1), (0,))
    with pytest.raises(RecursionError, match=r"3 passes.*\['grow', 'other'\]"):
        decompose_deep(circ, supported_gates={"h"}, max_passes=3)
