"""Assign wireless access points to channels with quadratic optimization.

The interference graph is a triangle with one tail edge.  Adjacent access
points may not use the same channel, while the objective minimizes per-device
channel preferences.

Run with::

    python examples/wireless_channel_assignment.py
"""

from __future__ import annotations

from collections.abc import Sequence
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

AccessPoint = str
Channel = str
InterferenceEdge = tuple[AccessPoint, AccessPoint]

ACCESS_POINTS: tuple[AccessPoint, ...] = ("North", "East", "South", "West")
CHANNELS: tuple[Channel, ...] = ("ch-1", "ch-6", "ch-11")
INTERFERENCE_EDGES: tuple[InterferenceEdge, ...] = (
    ("North", "East"),
    ("East", "South"),
    ("South", "North"),
    ("South", "West"),
)

PREFERENCE_COSTS: dict[tuple[AccessPoint, Channel], float] = {
    ("North", "ch-1"): 0.0,
    ("North", "ch-6"): 2.0,
    ("North", "ch-11"): 5.0,
    ("East", "ch-1"): 4.0,
    ("East", "ch-6"): 0.0,
    ("East", "ch-11"): 3.0,
    ("South", "ch-1"): 5.0,
    ("South", "ch-6"): 3.0,
    ("South", "ch-11"): 0.0,
    ("West", "ch-1"): 0.0,
    ("West", "ch-6"): 1.0,
    ("West", "ch-11"): 4.0,
}

EXACTLY_ONE_PENALTY = cast(Real, 25.0)
INTERFERENCE_PENALTY = cast(Real, 25.0)


def _variable_name(access_point: AccessPoint, channel: Channel) -> str:
    return f"assigned[{access_point},{channel}]"


def build_program() -> QuadraticProgram:
    """Build the constrained binary quadratic channel-assignment program."""
    program = QuadraticProgram("wireless-channel-assignment")
    variables: dict[tuple[AccessPoint, Channel], BinaryVariable] = {
        (access_point, channel): program.binary_variable(
            _variable_name(access_point, channel)
        )
        for access_point in ACCESS_POINTS
        for channel in CHANNELS
    }

    preference_cost = QuadraticExpression()
    for key, variable in variables.items():
        preference_cost += PREFERENCE_COSTS[key] * variable
    program.minimize(preference_cost)

    for access_point in ACCESS_POINTS:
        exactly_one = QuadraticExpression(-1.0)
        for channel in CHANNELS:
            exactly_one += variables[access_point, channel]
        program.add_linear_equality(
            exactly_one,
            name=f"exactly-one:{access_point}",
            penalty=EXACTLY_ONE_PENALTY,
        )

    for left, right in INTERFERENCE_EDGES:
        for channel in CHANNELS:
            program.add_nonnegative_quadratic_zero(
                variables[left, channel] * variables[right, channel],
                name=f"no-interference:{left}:{right}:{channel}",
                penalty=INTERFERENCE_PENALTY,
            )

    return program


def decode_assignment(
    compiled: CompiledQuadraticProgram,
    bits: Sequence[object],
) -> dict[AccessPoint, Channel]:
    """Decode one feasible binary solution into access-point channel labels."""
    if not compiled.is_feasible(bits):
        raise ValueError("Cannot decode an infeasible channel assignment.")

    decoded_bits = compiled.decode(bits)
    assignment: dict[AccessPoint, Channel] = {}
    for access_point in ACCESS_POINTS:
        selected = [
            channel
            for channel in CHANNELS
            if decoded_bits[_variable_name(access_point, channel)] == 1
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one channel for {access_point}, found {len(selected)}."
            )
        assignment[access_point] = selected[0]
    return assignment


def solve() -> dict[AccessPoint, Channel]:
    """Run the complete symbolic-to-native optimization and decode its result."""
    compiled = compile_quadratic_program(build_program())
    logical_ising = bqm_to_ising(compiled.bqm)
    hardware_program = compile_hardware_ising(logical_ising)
    result = NativeIsingSolver(
        seed=20260731,
        num_reads=64,
        sweeps=300,
        beta_start=0.05,
        beta_end=8.0,
    ).solve(hardware_program)

    best_bits: list[int] | None = None
    best_preference_cost = float("inf")
    for logical_sample in result.logical_samples:
        candidate = [
            int(bit) for bit in spin_samples_to_binary(logical_sample).tolist()
        ]
        if not compiled.is_feasible(candidate):
            continue
        preference_cost = compiled.evaluate_objective(candidate)
        if preference_cost < best_preference_cost:
            best_bits = candidate
            best_preference_cost = preference_cost

    if best_bits is None:
        raise RuntimeError("The native solver did not produce a feasible assignment.")
    return decode_assignment(compiled, best_bits)


def main() -> None:
    """Solve and print the channel selected for every access point."""
    assignment = solve()
    total_cost = sum(
        PREFERENCE_COSTS[access_point, channel]
        for access_point, channel in assignment.items()
    )
    print("Wireless channel assignment:")
    for access_point in ACCESS_POINTS:
        print(f"  {access_point}: {assignment[access_point]}")
    print(f"Preference cost: {total_cost:g}")


if __name__ == "__main__":
    main()
