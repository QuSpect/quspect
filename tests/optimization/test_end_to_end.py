"""End-to-end tests from symbolic models to the native matrix solver."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quspect._native_loader import load_core

if load_core() is None:
    pytest.skip("QuSpect native extension not built", allow_module_level=True)

from quspect.optimization import (  # noqa: E402
    CompiledQuadraticProgram,
    NativeIsingSolver,
    QuadraticProgram,
    bqm_to_ising,
    compile_hardware_ising,
    compile_quadratic_program,
    spin_samples_to_binary,
)


def _solve(
    program: QuadraticProgram,
) -> tuple[CompiledQuadraticProgram, NDArray[np.int8]]:
    compiled = compile_quadratic_program(program)
    hardware = compile_hardware_ising(bqm_to_ising(compiled.bqm))
    result = NativeIsingSolver(
        seed=20260730,
        num_reads=24,
        sweeps=80,
        beta_start=0.05,
        beta_end=8.0,
    ).solve(hardware)
    bits = spin_samples_to_binary(result.best_sample)
    return compiled, bits


def test_one_hot_problem_crosses_every_layer_and_decodes() -> None:
    program = QuadraticProgram("one-hot")
    choices = program.binary_array("choice", 3)
    program.minimize(3 * choices[0] + choices[1] + 2 * choices[2])
    program.add_linear_equality(
        choices.sum() - 1,
        name="choose-exactly-one",
        penalty=10,
    )

    compiled, bits = _solve(program)

    assert compiled.decode(bits) == {
        "choice[0]": 0,
        "choice[1]": 1,
        "choice[2]": 0,
    }
    assert compiled.is_feasible(bits)
    assert compiled.evaluate_objective(bits) == pytest.approx(1.0)


def test_max_cut_objective_preserves_maximization_direction() -> None:
    program = QuadraticProgram("cycle-max-cut")
    vertices = program.binary_array("side", 4)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    cut_size = sum(
        vertices[left] + vertices[right] - 2 * vertices[left] * vertices[right]
        for left, right in edges
    )
    program.maximize(cut_size)

    compiled, bits = _solve(program)

    assert compiled.is_feasible(bits)
    assert compiled.evaluate_objective(bits) == pytest.approx(4.0)


def test_max_constraint_scale_and_offset_compose_across_every_layer() -> None:
    program = QuadraticProgram("scaled-constrained-max")
    x = program.binary_variable("x")
    y = program.binary_variable("y")
    program.maximize(5 + 3 * x + y)
    program.add_linear_equality(x + y - 1, name="one-hot", penalty=10)

    compiled = compile_quadratic_program(program)
    hardware = compile_hardware_ising(
        bqm_to_ising(compiled.bqm),
        matrix_scale=3.25,
    )
    result = NativeIsingSolver(
        seed=20260730,
        num_reads=16,
        sweeps=60,
        beta_start=0.05,
        beta_end=8.0,
    ).solve(hardware)
    bits = spin_samples_to_binary(result.best_sample)

    assert compiled.decode(bits) == {"x": 1, "y": 0}
    assert compiled.is_feasible(bits)
    assert compiled.evaluate_objective(bits) == pytest.approx(8.0)
    assert result.best_objective_value == pytest.approx(compiled.bqm.evaluate(bits))
