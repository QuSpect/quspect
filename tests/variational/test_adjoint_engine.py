"""Validation tests for :class:`quspect.variational.AdjointEngine`."""

from __future__ import annotations

import math

import pytest

from quspect import Parameter, ParameterExpression, QuantumCircuit, QubitOperator
from quspect.variational import AdjointEngine, ObservableObjective

_A = Parameter("a")
_B = Parameter("b")


@pytest.mark.parametrize(
    ("angle", "message"),
    [
        (math.inf, "non-finite angle"),
        (math.nan, "non-finite angle"),
        (ParameterExpression({}, constant=math.inf), "non-finite constant"),
        (_A + math.inf, "non-finite offset"),
        (ParameterExpression({_A: math.nan}), "non-finite affine coefficient"),
    ],
)
def test_compile_rejects_non_finite_gate_parameters(
    angle: object, message: str
) -> None:
    circuit = QuantumCircuit().ry(_B, 0).rx(angle, 0)  # type: ignore[arg-type]
    objective = ObservableObjective(circuit, {"Z": QubitOperator({((0, "Z"),): 1.0})})
    with pytest.raises(ValueError, match=message):
        AdjointEngine().compile(objective)


# ---------------------------------------------------------------------------
# Two-qubit Pauli rotations are differentiated natively
# ---------------------------------------------------------------------------


def _pauli_pair_objective() -> ObservableObjective:
    circuit = QuantumCircuit()
    for qubit, angle in enumerate((0.3, -1.1, 0.7)):
        circuit.h(qubit).ry(angle, qubit)
    circuit.rxx(_A, 0, 2).ryy(0.5 * _A - _B, 2, 1).rzz(2 * _B + 0.1, 1, 0)
    circuit.cx(1, 2).ryy(_A, 0, 1).rx(_B, 2)
    observables = {
        "H": QubitOperator({((0, "Z"), (2, "Y")): 0.6, ((1, "X"),): -0.3}),
        "ZZ": QubitOperator({((0, "Z"), (1, "Z")): 1.0}),
    }
    return ObservableObjective(circuit, observables)


def test_pauli_pair_rotations_are_not_decomposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quspect.variational import adjoint

    lowered: list[QuantumCircuit] = []

    def capture(circuit: QuantumCircuit, **kwargs: object) -> QuantumCircuit:
        out = decompose_deep(circuit, **kwargs)  # type: ignore[arg-type]
        lowered.append(out)
        return out

    decompose_deep = adjoint.decompose_deep
    monkeypatch.setattr(adjoint, "decompose_deep", capture)
    AdjointEngine().compile(_pauli_pair_objective())
    names = [i.gate.name for i in lowered[0] if i.gate is not None]
    assert names.count("rxx") == 1 and names.count("ryy") == 2
    assert names.count("rzz") == 1


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_pauli_pair_rotation_gradients_match_finite_differences(dtype: str) -> None:
    plan = AdjointEngine(dtype=dtype).compile(_pauli_pair_objective())
    point = {"a": 0.31, "b": -0.47}
    result = plan.evaluate(point)
    reference = AdjointEngine().compile(_pauli_pair_objective())
    eps = 1e-6
    tol = 1e-8 if dtype == "float64" else 2e-5
    for column, name in enumerate(p.name for p in plan.parameter_order):
        plus = dict(point, **{name: point[name] + eps})
        minus = dict(point, **{name: point[name] - eps})
        fd = (reference.evaluate(plus).values - reference.evaluate(minus).values) / (
            2 * eps
        )
        assert result.jacobian[:, column] == pytest.approx(fd, abs=tol)


def test_pauli_pair_rotations_agree_with_their_decomposition() -> None:
    from quspect.ir.compilation import decompose_deep

    objective = _pauli_pair_objective()
    lowered = ObservableObjective(
        decompose_deep(
            objective.circuit, supported_gates={"h", "s", "sdg", "cx", "rx", "ry", "rz"}
        ),
        objective.observables,
    )
    point = {"a": 0.31, "b": -0.47}
    native = AdjointEngine().compile(objective).evaluate(point)
    reference = AdjointEngine().compile(lowered).evaluate(point)
    assert native.values == pytest.approx(reference.values, abs=1e-12)
    assert native.jacobian == pytest.approx(reference.jacobian, abs=1e-12)
