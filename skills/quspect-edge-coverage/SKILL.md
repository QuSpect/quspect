---
name: quspect-edge-coverage
description: Model, implement, explain, or validate fixed-cardinality edge-site selection and demand-coverage problems with QuSpect quadratic optimization. Use for choosing exactly K service locations, maximizing weighted demand coverage, representing coverage witnesses with supported constraints, decoding selected sites and covered demand, or testing coverage models through the BQM, Ising, and hardware-matrix layers.
---

# QuSpect Edge Coverage

## Represent coverage with supported variables

Define:

- `open[site]`: whether a candidate site is selected;
- `covered[demand]`: whether a demand point is served;
- `witness[demand, site]`: whether an eligible selected site witnesses that
  demand point's coverage.

Create witness variables only for eligible site-demand edges. This avoids a
general inequality while preserving ordinary maximum coverage:

```text
sum(open[site]) - K = 0
covered[demand] - sum(witness[demand, site]) = 0
witness[demand, site] * (1 - open[site]) = 0
```

The second equality selects exactly one witness when a demand point is marked
covered. The final expression is nonnegative on binary assignments, so it is
valid for `add_nonnegative_quadratic_zero`. Never pass a signed or potentially
negative expression to that method.

## Build the objective and constraints

1. Preserve stable site, demand, and eligible-edge order.
2. Create `open`, `covered`, and sparse `witness` binary variables.
3. Maximize `sum(weight[d] * covered[d] for d)` and add documented secondary
   site costs only when the business objective requires them.
4. Add the exactly-`K` site constraint with `add_linear_equality`.
5. Add one covered-to-witness linear equality per demand point.
6. Add one witness-to-open nonnegative quadratic-zero constraint per eligible
   edge.
7. Select penalties from a bound on the maximum coverage reward that a
   violation could gain. Verify the chosen values on small exhaustive cases.

Keep coverage topology, demand weights, geography, and presentation in the
application layer. Keep only reusable expressions, compilation, conversion,
hardware lowering, codecs, and solvers in `quspect.optimization`.

## Compile, solve, and decode

Compile through the standard boundary:

```python
compiled = compile_quadratic_program(program)
logical = bqm_to_ising(compiled.bqm)
hardware = compile_hardware_ising(logical)
result = NativeIsingSolver(seed=7, num_reads=64).solve(hardware)
```

Convert logical spins with `spin_samples_to_binary`, scan retained samples for
feasibility, and compare feasible samples with
`compiled.evaluate_objective(bits)`.

Decode:

- selected sites from `open`;
- covered and uncovered demand points from `covered`;
- the selected witness site for every covered demand point;
- covered weight and any secondary cost as separate business metrics.

## Validate the business invariants

- Require exactly `K` selected sites.
- Require every covered demand point to have exactly one eligible witness.
- Require every witness site to be selected.
- Reject infeasible samples with `compiled.is_feasible(bits)`.
- Exhaustively enumerate small graphs and compare the optimum with a direct
  set-based maximum-coverage calculation.
- Test isolated demand points, duplicate neighborhoods, `K=0`, `K` equal to the
  site count, ties, and zero weights.
- Verify variable order and energy preservation through
  `HardwareIsingProgram`; test the native heuristic separately.

Do not silently extend this model with site capacities. Capacity introduces a
general inequality and belongs in the cloud-edge network workflow after the
upper-layer inequality and slack capabilities exist.

Read `docs/source/optimization.rst`,
`docs/source/optimization_applications.rst`,
`quspect/optimization/program.py`, `quspect/optimization/compiler.py`, and
`tests/optimization/test_end_to_end.py` before relying on signatures or
constraint semantics.
