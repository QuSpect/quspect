"""End-to-end tests for the native adjoint objective-plan bridge."""

from __future__ import annotations

import math

import pytest

from quspect import Parameter, QuantumCircuit, QubitOperator
from quspect._native_loader import load_core
from quspect.ir.qsal import dumps

_core = load_core()

pytestmark = pytest.mark.skipif(_core is None, reason="native extension is not built")


def _variational_qsal() -> str:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(2.0 * theta + 0.1, 0)
    hamiltonian = QubitOperator(
        {
            (): 0.3,
            ((0, "Z"),): 0.7,
            ((0, "X"),): -0.2,
        }
    )
    return dumps(circuit, observables=[("energy", hamiltonian)])


def test_adjoint_plan_pybind_metadata_and_reuse() -> None:
    plan = _core._AdjointPlan(_variational_qsal(), "double")

    assert not hasattr(_core, "CompiledGradient")
    assert not hasattr(_core, "GradientResult")
    assert list(plan.parameter_names) == ["theta"]
    assert list(plan.observable_names) == ["energy"]

    for theta in (0.2, -0.7):
        alpha = 2.0 * theta + 0.1
        result = plan.evaluate({"theta": theta})
        expected_energy = 0.3 + 0.7 * math.cos(alpha) - 0.2 * math.sin(alpha)
        expected_gradient = 2.0 * (-0.7 * math.sin(alpha) - 0.2 * math.cos(alpha))
        assert result.values[0] == pytest.approx(expected_energy, abs=1e-12)
        assert result.jacobian[0][0] == pytest.approx(expected_gradient, abs=1e-12)


def test_adjoint_plan_single_precision() -> None:
    plan = _core._AdjointPlan(_variational_qsal(), "single")
    theta = 0.34
    result = plan.evaluate({"theta": theta})
    alpha = 2.0 * theta + 0.1

    assert result.values[0] == pytest.approx(
        0.3 + 0.7 * math.cos(alpha) - 0.2 * math.sin(alpha), abs=2e-6
    )


def test_adjoint_plan_binding_errors_cross_pybind() -> None:
    from quspect.native import NativeExecutor

    plan = _core._AdjointPlan(_variational_qsal(), "double")

    with pytest.raises(ValueError, match="missing value"):
        plan.evaluate({})
    with pytest.raises(ValueError, match="unexpected value"):
        plan.evaluate({"theta": 0.1, "extra": 1.0})

    with pytest.raises(ValueError, match="precision"):
        _core._AdjointPlan(_variational_qsal(), "half")
    with pytest.raises(ValueError, match="precision"):
        NativeExecutor(precision="half")


def test_native_executor_prepares_private_adjoint_plan() -> None:
    from quspect.native import NativeExecutor

    executor = NativeExecutor()
    assert not hasattr(executor, "compile_gradient")

    plan = executor._prepare_adjoint(_variational_qsal())
    result = plan.evaluate({"theta": 0.2})

    assert plan.parameter_names == ("theta",)
    assert plan.observable_names == ("energy",)
    assert result.observable_names == ("energy",)
    assert len(result.values) == 1
    assert len(result.jacobian) == 1


def test_ordinary_native_execution_rejects_symbolic_program() -> None:
    from quspect.native import NativeExecutor

    executor = NativeExecutor()
    with pytest.raises(ValueError, match="cannot execute symbolic parameter"):
        executor.run(_variational_qsal(), return_expectations=True)


def test_adjoint_plan_rejects_non_real_observable_coefficient() -> None:
    """Non-Hermitian observables must not silently lose their imaginary part."""
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0)
    hamiltonian = QubitOperator({((0, "Z"),): 0.5 + 0.25j})
    qsal = dumps(circuit, observables=[("h", hamiltonian)])

    with pytest.raises(ValueError, match="non-Hermitian"):
        _core._AdjointPlan(qsal, "double")
