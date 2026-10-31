"""Solve a fixed-cardinality mean-variance portfolio selection problem.

The example chooses exactly two of four assets and maximizes

``expected return - risk_aversion * portfolio variance``.

Run it from the repository root with::

    python examples/portfolio_selection.py
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import cast

from quspect.optimization import (
    BinaryVariable,
    CompiledQuadraticProgram,
    NativeIsingSolver,
    QuadraticExpression,
    QuadraticProgram,
    bqm_to_ising,
    compile_hardware_ising,
    compile_quadratic_program,
    spin_samples_to_binary,
)

ASSETS: tuple[str, ...] = (
    "Global Equity",
    "Investment-Grade Bonds",
    "Technology Equity",
    "Infrastructure",
)
EXPECTED_RETURNS: tuple[float, ...] = (0.15, 0.12, 0.1425, 0.10)
COVARIANCE_MATRIX: tuple[tuple[float, ...], ...] = (
    (0.040, 0.012, 0.010, 0.008),
    (0.012, 0.030, 0.009, 0.007),
    (0.010, 0.009, 0.035, 0.006),
    (0.008, 0.007, 0.006, 0.025),
)
TARGET_HOLDINGS = 2
RISK_AVERSION = 0.5
CARDINALITY_PENALTY = cast(Real, 1.0)


@dataclass(frozen=True)
class PortfolioResult:
    """Business-level result returned by :func:`solve`."""

    selected_assets: tuple[str, ...]
    bits: tuple[int, ...]
    expected_return: float
    variance: float
    objective_value: float


def build_program() -> QuadraticProgram:
    """Build the fixed-cardinality mean-variance quadratic program."""
    program = QuadraticProgram("fixed-cardinality-portfolio")
    raw_variables = program.binary_array("select", len(ASSETS))
    variables = tuple(
        cast(BinaryVariable, variable) for variable in raw_variables.tolist()
    )

    expected_return = sum(
        (
            EXPECTED_RETURNS[index] * variable
            for index, variable in enumerate(variables)
        ),
        start=QuadraticExpression(),
    )
    variance = sum(
        (
            COVARIANCE_MATRIX[left][right] * variables[left] * variables[right]
            for left in range(len(variables))
            for right in range(len(variables))
        ),
        start=QuadraticExpression(),
    )

    program.maximize(expected_return - RISK_AVERSION * variance)
    program.add_linear_equality(
        sum(variables, start=QuadraticExpression()) - TARGET_HOLDINGS,
        name="exactly-two-assets",
        penalty=CARDINALITY_PENALTY,
    )
    return program


def decode_portfolio(
    compiled: CompiledQuadraticProgram,
    bits: Sequence[object],
) -> tuple[str, ...]:
    """Map a compiled binary sample to the selected business assets."""
    assignment = compiled.decode(bits)
    return tuple(
        asset
        for index, asset in enumerate(ASSETS)
        if assignment[f"select[{index}]"] == 1
    )


def solve() -> PortfolioResult:
    """Run every optimization layer and return the decoded portfolio."""
    compiled = compile_quadratic_program(build_program())
    logical = bqm_to_ising(compiled.bqm)
    hardware = compile_hardware_ising(logical)
    native_result = NativeIsingSolver(
        seed=20260731,
        num_reads=64,
        sweeps=200,
        beta_start=0.05,
        beta_end=8.0,
    ).solve(hardware)
    bits = tuple(
        int(value)
        for value in spin_samples_to_binary(native_result.best_sample).tolist()
    )

    if not compiled.is_feasible(bits):
        raise RuntimeError(
            "The best retained sample violates the fixed-cardinality constraint."
        )

    selected_assets = decode_portfolio(compiled, bits)
    expected_return = sum(
        EXPECTED_RETURNS[index] * bit for index, bit in enumerate(bits)
    )
    variance = sum(
        COVARIANCE_MATRIX[left][right] * bits[left] * bits[right]
        for left in range(len(bits))
        for right in range(len(bits))
    )
    return PortfolioResult(
        selected_assets=selected_assets,
        bits=bits,
        expected_return=expected_return,
        variance=variance,
        objective_value=compiled.evaluate_objective(bits),
    )


def main() -> None:
    """Solve and print the portfolio in business terms."""
    result = solve()
    print(f"selected assets: {', '.join(result.selected_assets)}")
    print(f"binary selection: {result.bits}")
    print(f"expected return: {result.expected_return:.4f}")
    print(f"portfolio variance: {result.variance:.4f}")
    print(f"mean-variance objective: {result.objective_value:.4f}")


if __name__ == "__main__":
    main()
