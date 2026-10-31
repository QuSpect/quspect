"""Tests for binary quadratic and logical Ising data models."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.optimization import (
    BinaryQuadraticModel,
    IsingProblem,
    OptimizationSense,
    VariableType,
)


def test_binary_quadratic_model_evaluates_single_and_batch() -> None:
    model = BinaryQuadraticModel(
        variables=("x", "y"),
        linear=np.array([1.0, -2.0]),
        quadratic=np.array([[0.0, 3.0], [3.0, 0.0]]),
        offset=0.5,
        variable_type=VariableType.BINARY,
    )

    assert model.evaluate([1, 0]) == pytest.approx(1.5)
    np.testing.assert_allclose(
        model.evaluate_many([[0, 0], [1, 0], [1, 1]]),
        [0.5, 1.5, 2.5],
    )


def test_model_snapshots_read_only_coefficients() -> None:
    linear = np.array([1.0, 2.0])
    quadratic = np.array([[0.0, 3.0], [3.0, 0.0]])
    model = BinaryQuadraticModel(("x", "y"), linear, quadratic)
    linear[0] = 99.0
    quadratic[0, 1] = 99.0

    assert model.linear.tolist() == [1.0, 2.0]
    assert model.quadratic[0, 1] == 3.0
    with pytest.raises(ValueError):
        model.linear[0] = 2.0


def test_model_canonicalizes_symmetry_within_tolerance() -> None:
    model = BinaryQuadraticModel(
        variables=("x", "y"),
        linear=np.zeros(2),
        quadratic=np.array([[0.0, 1.0], [1.0 + 5e-13, 0.0]]),
    )

    assert np.array_equal(model.quadratic, model.quadratic.T)
    assert model.evaluate([1, 1]) == pytest.approx(1.0 + 2.5e-13)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "variables": ("x", "x"),
                "linear": np.zeros(2),
                "quadratic": np.zeros((2, 2)),
            },
            "unique",
        ),
        (
            {
                "variables": ("x", "y"),
                "linear": np.zeros(2),
                "quadratic": np.array([[0.0, 1.0], [0.0, 0.0]]),
            },
            "symmetric",
        ),
        (
            {
                "variables": ("x",),
                "linear": np.zeros(1),
                "quadratic": np.ones((1, 1)),
            },
            "zero diagonal",
        ),
    ],
)
def test_model_rejects_ambiguous_representation(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BinaryQuadraticModel(**kwargs)  # type: ignore[arg-type]


def test_model_validates_sample_domain() -> None:
    binary = BinaryQuadraticModel(
        ("x",),
        np.zeros(1),
        np.zeros((1, 1)),
    )
    spin = IsingProblem(("s",), np.zeros(1), np.zeros((1, 1)))

    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        binary.evaluate([-1])
    with pytest.raises(ValueError, match=r"\{-1, \+1\}"):
        spin.evaluate([0])


def test_ising_preserves_maximization_sense() -> None:
    problem = IsingProblem(
        variables=("a", "b"),
        linear=np.zeros(2),
        quadratic=np.array([[0.0, 1.0], [1.0, 0.0]]),
        sense=OptimizationSense.MAXIMIZE,
    )

    assert problem.sense is OptimizationSense.MAXIMIZE
    assert problem.as_bqm().variable_type is VariableType.SPIN


def test_bqm_large_finite_quadratic_does_not_overflow() -> None:
    quadratic = np.array([[0.0, 1e308], [1e308, 0.0]])
    model = BinaryQuadraticModel(("a", "b"), [0.0, 0.0], quadratic)
    np.testing.assert_array_equal(model.quadratic, quadratic)
