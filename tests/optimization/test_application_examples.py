"""Regression tests for the business-facing optimization examples."""

from __future__ import annotations

from itertools import product

import pytest

from examples.portfolio_selection import (
    ASSETS,
    decode_portfolio,
)
from examples.portfolio_selection import (
    build_program as build_portfolio_program,
)
from examples.wireless_channel_assignment import (
    ACCESS_POINTS,
    decode_assignment,
)
from examples.wireless_channel_assignment import (
    build_program as build_channel_program,
)
from quspect.optimization import (
    CompiledQuadraticProgram,
    OptimizationSense,
    bqm_to_ising,
    compile_hardware_ising,
    compile_quadratic_program,
)

BinarySample = tuple[int, ...]
ScoredSample = tuple[BinarySample, float]


def _feasible_samples(
    compiled: CompiledQuadraticProgram,
) -> list[ScoredSample]:
    samples: list[ScoredSample] = []
    for bits in product((0, 1), repeat=len(compiled.variables)):
        if compiled.is_feasible(bits):
            samples.append((bits, compiled.evaluate_objective(bits)))
    return samples


def _best_samples(
    compiled: CompiledQuadraticProgram,
    samples: list[ScoredSample],
) -> list[ScoredSample]:
    values = [value for _, value in samples]
    if compiled.bqm.sense is OptimizationSense.MAXIMIZE:
        optimum = max(values)
    else:
        optimum = min(values)
    return [(bits, value) for bits, value in samples if value == pytest.approx(optimum)]


def test_portfolio_example_has_one_expected_business_optimum() -> None:
    compiled = compile_quadratic_program(build_portfolio_program())
    feasible = _feasible_samples(compiled)
    best = _best_samples(compiled, feasible)

    assert len(compiled.variables) == 4
    assert len(compiled.constraints) == 1
    assert len(feasible) == 6
    assert len(best) == 1

    bits, objective = best[0]
    assert bits == (1, 0, 1, 0)
    assert objective == pytest.approx(0.245)
    assert decode_portfolio(compiled, bits) == (ASSETS[0], ASSETS[2])

    hardware = compile_hardware_ising(bqm_to_ising(compiled.bqm))
    assert hardware.logical_variable_count == len(compiled.variables)
    assert hardware.bit_num == len(compiled.variables) + 1
    assert hardware.matrix.shape == (hardware.bit_num, hardware.bit_num)


def test_wireless_example_has_one_expected_business_optimum() -> None:
    compiled = compile_quadratic_program(build_channel_program())
    feasible = _feasible_samples(compiled)
    best = _best_samples(compiled, feasible)

    assert len(compiled.variables) == 12
    assert len(compiled.constraints) == 16
    assert len(feasible) == 12
    assert len(best) == 1

    bits, objective = best[0]
    assert objective == pytest.approx(0.0)
    assert decode_assignment(compiled, bits) == {
        "North": "ch-1",
        "East": "ch-6",
        "South": "ch-11",
        "West": "ch-1",
    }

    hardware = compile_hardware_ising(bqm_to_ising(compiled.bqm))
    assert hardware.logical_variable_count == len(compiled.variables)
    assert hardware.bit_num == len(compiled.variables) + 1
    assert hardware.matrix.shape == (hardware.bit_num, hardware.bit_num)
    assert tuple(decode_assignment(compiled, bits)) == ACCESS_POINTS
