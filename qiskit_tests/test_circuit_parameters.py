"""Cross-validate ``Parameter`` / ``ParameterExpression`` and circuit binding.

Reference: qiskit ``Parameter`` / ``ParameterExpression`` and
``QuantumCircuit.assign_parameters``.  quspect expressions are restricted to
the affine form ``c + sum(a_i * theta_i)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import qiskit

from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_unitary_equal,
    qiskit_circuit_unitary,
    quspect_circuit_unitary,
    to_qiskit,
)
from quspect import QuantumCircuit
from quspect.circuit import Parameter, ParameterExpression
from quspect.simulation import Simulator

A = Parameter("a")
B = Parameter("b")


# =============================================================================
# Parameter value semantics
# =============================================================================


def test_parameter_equality_hash_and_order() -> None:
    """Parameters compare, hash and sort by name."""
    assert Parameter("a") == A
    assert hash(Parameter("a")) == hash(A)
    assert A != B
    assert len({A, Parameter("a"), B}) == 2
    assert sorted([B, A]) == [A, B]
    assert repr(A) == "Parameter('a')"
    assert A.as_expression() == ParameterExpression({A: 1.0})


@pytest.mark.qiskit_ref
@pytest.mark.parametrize(
    ("build", "coeffs", "const"),
    [
        (lambda a, b: a + b, {"a": 1.0, "b": 1.0}, 0.0),
        (lambda a, b: a - b, {"a": 1.0, "b": -1.0}, 0.0),
        (lambda a, b: a + 1.5, {"a": 1.0}, 1.5),
        (lambda a, b: 1.5 + a, {"a": 1.0}, 1.5),
        (lambda a, b: a - 2, {"a": 1.0}, -2.0),
        (lambda a, b: 2 - a, {"a": -1.0}, 2.0),
        (lambda a, b: -a, {"a": -1.0}, 0.0),
        (lambda a, b: 3 * a, {"a": 3.0}, 0.0),
        (lambda a, b: a * 3, {"a": 3.0}, 0.0),
        (lambda a, b: a / 4, {"a": 0.25}, 0.0),
        (lambda a, b: 2 * a - 0.5 * b + 1, {"a": 2.0, "b": -0.5}, 1.0),
        (lambda a, b: (a + b) * 2 - (b - a) / 2, {"a": 2.5, "b": 1.5}, 0.0),
        (lambda a, b: -(a - 3) + b, {"a": -1.0, "b": 1.0}, 3.0),
    ],
)
def test_expression_arithmetic_matches_qiskit(
    build: object, coeffs: dict[str, float], const: float
) -> None:
    """Affine arithmetic gives the same coefficients and values as qiskit."""
    expr = build(A, B)  # type: ignore[operator]
    assert isinstance(expr, ParameterExpression)
    assert dict(expr.terms) == {Parameter(k): v for k, v in coeffs.items()}
    assert expr.constant == pytest.approx(const)
    qa, qb = qiskit.circuit.Parameter("a"), qiskit.circuit.Parameter("b")
    qexpr = build(qa, qb)  # type: ignore[operator]
    for va, vb in [(0.3, -1.1), (2.0, 0.5), (-math.pi, math.e)]:
        bound = {qa: va, qb: vb}
        q_val = float(qexpr.bind({p: bound[p] for p in qexpr.parameters}))
        assert expr.evaluate({A: va, B: vb}) == pytest.approx(q_val, abs=1e-12)


def test_scalar_identity_shortcuts() -> None:
    """``a * 1`` and ``a / 1`` return the parameter itself."""
    assert A * 1 is A
    assert 1 * A is A
    assert A / 1 is A
    zero = A * 0
    assert isinstance(zero, ParameterExpression)
    assert zero.is_constant and float(zero) == 0.0


def test_expression_canonical_equality_and_hash() -> None:
    """Expressions are canonical: term order and zero terms do not matter."""
    e1 = A + B + 1
    e2 = 1 + B + A
    assert e1 == e2
    assert hash(e1) == hash(e2)
    cancelled = A - A
    assert cancelled.is_constant
    assert cancelled.parameters == frozenset()
    assert float(cancelled) == 0.0
    assert ParameterExpression({"a": 1.0, A: 1.0}) == 2 * A
    assert ParameterExpression([(A, 0.0)], constant=0) == ParameterExpression()


def test_expression_accessors() -> None:
    """``terms``, ``parameters``, ``coefficient``, ``bind``, str/repr."""
    expr = 2 * A - B + 0.5
    assert expr.parameters == frozenset({A, B})
    assert expr.coefficient(A) == 2.0
    assert expr.coefficient(Parameter("zzz")) == 0.0
    assert not expr.is_constant
    assert expr.bind({A: 1.0, B: 2.0}) == pytest.approx(0.5)
    with pytest.raises(TypeError):
        expr.terms[A] = 3.0  # type: ignore[index]
    assert str(expr) == "2*a - b + 0.5"
    assert str(A + B) == "a + b"
    assert str(ParameterExpression()) == "0"
    assert "ParameterExpression(" in repr(expr)
    assert "constant=0.5" in repr(expr)


@pytest.mark.parametrize(
    ("op", "exc"),
    [
        (lambda: A * B, TypeError),
        (lambda: (A + 1) * (B + 1), TypeError),
        (lambda: A / B, TypeError),
        (lambda: 2 / A, TypeError),
        (lambda: A / 0, ZeroDivisionError),
        (lambda: (A + 1) / 0.0, ZeroDivisionError),
        (lambda: 1 / (A - A), ZeroDivisionError),
        (lambda: A + 1j, TypeError),
        (lambda: A + "x", TypeError),
        (lambda: "x" - A, TypeError),
        (lambda: A * "x", TypeError),
        (lambda: A / "x", TypeError),
        (lambda: "x" / A, TypeError),
        (lambda: (A + 1) + "x", TypeError),
        (lambda: "x" - (A + 1), TypeError),
        (lambda: (A + 1) * "x", TypeError),
        (lambda: (A + 1) / "x", TypeError),
        (lambda: "x" / (A + 1), TypeError),
        (lambda: 2 / (A + 1), TypeError),
        (lambda: A - "x", TypeError),
        (lambda: (A + 1) - "x", TypeError),
        (lambda: float(A + 1), TypeError),
        (lambda: ParameterExpression({1: 1.0}), TypeError),  # type: ignore[dict-item]
        (lambda: ParameterExpression({A: 1j}), TypeError),
        (lambda: ParameterExpression(constant="1"), TypeError),  # type: ignore[arg-type]
        (lambda: (A + B).evaluate({A: 1.0}), ValueError),
        (lambda: A.as_expression().evaluate({A: 1j}), TypeError),  # type: ignore[dict-item]
    ],
)
def test_expression_invalid_operations(op: object, exc: type[Exception]) -> None:
    """Non-affine or non-real operations raise the documented errors."""
    with pytest.raises(exc):
        op()  # type: ignore[operator]


def test_constant_expression_division() -> None:
    """Division by a constant expression and by a scalar both work."""
    const = ParameterExpression(constant=4.0)
    assert (A / const).coefficient(A) == pytest.approx(0.25)
    assert float(2 / const) == pytest.approx(0.5)
    assert (const * (A + 1)).coefficient(A) == pytest.approx(4.0)


# =============================================================================
# Circuit parameters and binding
# =============================================================================


def test_circuit_parameters_collects_from_expressions() -> None:
    """``parameters`` collects bare parameters and expression parameters."""
    c = Parameter("c")
    circ = QuantumCircuit().rx(A, 0).ry(2 * B + 1, 1).h(0).rz(0.3, 0).cp(c - c, 0, 1)
    assert circ.parameters == {A, B}
    qc = to_qiskit(circ)
    assert {p.name for p in qc.parameters} == {"a", "b"}


@pytest.mark.qiskit_ref
@pytest.mark.parametrize(
    "name", sorted(n for n, s in GATE_SPECS.items() if s.num_params > 0)
)
def test_bound_gate_matrix_matches_qiskit_parameter(
    name: str, rng: np.random.Generator
) -> None:
    """Gates with expression parameters bind to the same unitary as qiskit."""
    spec = GATE_SPECS[name]
    exprs = [2 * A - 0.3, B, -A + B + 1.0][: spec.num_params]
    qubits = [1, 0][: spec.num_qubits]
    circ = QuantumCircuit()
    circ.id(1)
    spec.apply_quspect(circ, exprs, qubits)
    table: dict[str, qiskit.circuit.Parameter] = {}
    qc = to_qiskit(circ, parameter_table=table)
    for _ in range(3):
        va, vb = (float(x) for x in rng.uniform(-math.pi, math.pi, size=2))
        bound = circ.bind_parameters({A: va, B: vb})
        assert bound.parameters == set()
        qbound = qc.assign_parameters(
            {table[p.name]: v for p, v in [(A, va), (B, vb)] if p.name in table}
        )
        assert_unitary_equal(
            quspect_circuit_unitary(bound), qiskit_circuit_unitary(qbound)
        )


def test_bind_parameters_resolves_values_and_keeps_others() -> None:
    """Binding replaces symbolic params, keeps measure/barrier/fixed gates."""
    circ = QuantumCircuit(name="p").h(0).rx(A, 0).ry(0.25, 1).cu3(A, B + 1, 0.5, 0, 1)
    circ.barrier().measure([0], [0])
    bound = circ.bind_parameters({A: 0.5, B: -0.25})
    assert bound.name == "p"
    assert len(bound) == len(circ)
    assert bound[0] is circ[0]
    assert bound[1].gate is not None and bound[1].gate.params == (0.5,)
    assert bound[2].gate is not None and bound[2].gate.params == (0.25,)
    assert bound[3].gate is not None and bound[3].gate.params == (0.5, 0.75, 0.5)
    assert bound[-1] is circ[-1]
    # Extra bindings are ignored; the original circuit is untouched.
    circ.bind_parameters({A: 1.0, B: 2.0, Parameter("unused"): 3.0})
    assert circ.parameters == {A, B}


def test_bind_parameters_missing_raises() -> None:
    """Missing bindings raise ValueError (bare parameter and expression)."""
    with pytest.raises(ValueError, match="not bound"):
        QuantumCircuit().rx(A, 0).bind_parameters({})
    with pytest.raises(ValueError, match="not bound"):
        QuantumCircuit().rx(A + B, 0).bind_parameters({A: 1.0})


def test_bind_parameters_rejects_non_real_value() -> None:
    """Binding a parameter to a string must raise TypeError/ValueError."""
    with pytest.raises((TypeError, ValueError)):
        QuantumCircuit().rx(A, 0).bind_parameters({A: "oops"})  # type: ignore[dict-item]


def test_unbound_circuit_cannot_be_simulated() -> None:
    """Simulator rejects unbound parameters but accepts constant expressions."""
    sim = Simulator(seed=0)
    with pytest.raises(ValueError, match="unbound"):
        sim.statevector(QuantumCircuit().rx(A, 0))
    with pytest.raises(ValueError, match="unbound"):
        sim.run(QuantumCircuit().rx(A + 1, 0).measure_all())
    sv = sim.statevector(QuantumCircuit().rx(A - A + 0.5, 0))
    ref = qiskit.quantum_info.Statevector.from_label("0").evolve(
        qiskit.circuit.library.RXGate(0.5)
    )
    np.testing.assert_allclose(sv, ref.data, atol=1e-12)


@pytest.mark.parametrize("value", [np.int64(1), np.float32(0.5)])
def test_numpy_scalar_angles_are_simulable(value: object) -> None:
    """Numpy scalar angles should behave like Python floats (as in qiskit)."""
    circ = QuantumCircuit().rx(value, 0)  # type: ignore[arg-type]
    sv = Simulator(seed=0).statevector(circ)
    ref = qiskit.quantum_info.Statevector.from_label("0").evolve(
        qiskit.circuit.library.RXGate(float(value))  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(sv, ref.data, atol=1e-6)


def test_reflected_expression_operators() -> None:
    """Scalar-on-the-left operators on expressions (radd / rsub / rmul)."""
    e = A + 1
    assert (1 + e) == A + 2
    assert (1 - e) == -A
    assert (2 * e) == 2 * A + 2
    assert (1 - e).evaluate({A: 0.5}) == pytest.approx(-0.5)
