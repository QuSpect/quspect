---
name: quspect-route-planning
description: Build, solve, decode, and validate route-permutation models with the QuSpect quadratic optimization stack. Use for traveling-salesperson tours, directed or undirected visit costs, fixed-start or open-route variants, one-hot position encodings, and their BQM, Ising, hardware-matrix, or native-solver flow.
---

# QuSpect Route Planning

## Inspect the local contracts

Read the [optimization architecture](../../docs/source/optimization.rst), the
[permutation model family](../../docs/source/optimization_applications.rst),
and the [symbolic program API](../../quspect/optimization/program.py). Inspect
`quspect/optimization/expression.py`, `compiler.py`, and
`tests/optimization/test_end_to_end.py` before relying on signatures or
penalty behavior.

## Define the route semantics

Normalize an ordered item list and a finite square cost matrix. Decide before
modeling:

- Use directed costs when `cost[i][j] != cost[j][i]`.
- Include the last-to-first edge for a closed tour.
- Omit that edge for an open route.
- Add fixed endpoint equalities when a start or end is prescribed.
- Combine neither directions nor rotations unless the business problem makes
  them equivalent.

## Build the position encoding

Create `visit[i, p]`, equal to one when item `i` occupies route position `p`.
For `n` items, declare an `(n, n)` binary array.

Minimize adjacent-pair cost:

```text
sum(position p, item i, next item j)
    cost[i][j] * visit[i,p] * visit[j,next(p)]
```

Add both one-hot constraint families:

```text
for every item i:      sum_p visit[i,p] - 1 == 0
for every position p:  sum_i visit[i,p] - 1 == 0
```

Use `QuadraticExpression()` as the explicit start value when building sums.
Add each residual with a unique name through
`program.add_linear_equality(...)`. For a closed tour, iterate all positions
and use `(p + 1) % n`; for an open route, stop at `n - 1`.

Derive the equality penalty from a bound on the largest objective improvement
available to an infeasible assignment. Scale costs first when their magnitudes
are extreme, then confirm feasibility empirically. Penalties are soft and are
not proof that the returned sample is valid.

Fix a designated start with `visit[start_index, 0] - 1 == 0` when removing
closed-tour rotation symmetry is useful. Position encoding already forms one
sequence, so do not add subset-elimination constraints from an edge-based TSP
formulation.

## Compile and solve through the stable boundary

Use the public chain without bypassing a layer:

```python
from quspect.optimization import (
    NativeIsingSolver,
    bqm_to_ising,
    compile_hardware_ising,
    compile_quadratic_program,
    spin_samples_to_binary,
)

compiled = compile_quadratic_program(program)
logical = bqm_to_ising(compiled.bqm)
hardware = compile_hardware_ising(logical)
result = NativeIsingSolver(seed=7, num_reads=128, sweeps=500).solve(hardware)
```

Convert every promising `result.logical_samples` row with
`spin_samples_to_binary`. Filter with `compiled.is_feasible(bits)`, then choose
the feasible sample with the smallest
`compiled.evaluate_objective(bits)`. Raise a clear error if no retained read
is feasible.

## Decode the route

Use `compiled.decode(bits)` to preserve the compiler's stable variable order.
For each position, require exactly one selected item and append that item to
the decoded route. Then:

1. Require every item to appear exactly once.
2. Recompute route cost from the original matrix.
3. Include or omit the closing edge according to the declared route mode.
4. Return business identifiers rather than raw variable names.

Never decode an infeasible assignment silently.

## Validate the model

- Enumerate all permutations for small fixtures and compare the exact optimum.
- Test asymmetric matrices so directed-index mistakes are observable.
- Test open and closed routes separately.
- Test fixed-start constraints and rotationally equivalent optima.
- Compare recomputed cost with `compiled.evaluate_objective(bits)`.
- Assert that `hardware.logical_variables == compiled.variables` and that the
  matrix is finite, symmetric, and zero-diagonal.
- Use deterministic seeds for regression tests while documenting that the
  native annealer is approximate.

## Keep the applicability boundary explicit

- Use this encoding for Hamiltonian paths or tours that visit every item once.
- Do not call it a general shortest-path solver.
- Do not hide time windows, capacities, precedence, or arbitrary inequalities
  in ad hoc penalties. The current `QuadraticProgram` has equality and
  declared-nonnegative-zero helpers, not general inequality/slack builders.
- Account for `n^2` logical bits and a dense hardware matrix over those bits,
  which implies quartic storage growth in the number of route items.
- Keep geocoding, distance acquisition, route display, and submission
  transports outside the optimization core.
