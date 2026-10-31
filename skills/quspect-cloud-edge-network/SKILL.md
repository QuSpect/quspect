---
name: quspect-cloud-edge-network
description: Assess, design, extend, implement, or validate capacitated cloud-edge resource-placement and network-allocation models for QuSpect. Use for open-site and demand-assignment variables, fixed/compute/transport costs, capacity or overflow rules, integer and slack encodings, or deciding which reusable optimization-layer capabilities must be added before a full cloud-edge model can be represented safely.
---

# QuSpect Cloud-Edge Network

## Audit capability before modeling

Inspect `quspect/optimization/program.py` and its tests before writing the
scenario. The current public model supports:

- binary variables and quadratic objectives;
- linear equalities compiled as squared penalties;
- quadratic expressions explicitly known to be nonnegative and constrained to
  zero.

It does not currently provide general linear inequalities, bounded-integer
variables, automatic slack encoding, or capacity helpers. Do not invent method
names, call a missing API, or disguise a signed capacity residual as a
nonnegative quadratic-zero constraint.

## Separate the supported subset

An uncapacitated placement subset can use:

- `open[edge]` for activated sites;
- `assigned[demand, edge]` for demand routing;
- a fixed opening cost plus compute and transport assignment costs;
- `sum(assigned[demand, :]) - 1 = 0` for exactly one destination;
- `assigned[demand, edge] * (1 - open[edge]) = 0` for assignment implies open.

The implication expression is nonnegative for binary variables and is
supported by `add_nonnegative_quadratic_zero`. Name the result uncapacitated;
do not present it as the complete cloud-edge model.

## Extend the upper layer first

Before implementing capacity, overflow, or resource quantities, design and test
reusable capabilities in `quspect.optimization`:

1. Represent a bounded integer as binary bits with explicit weights, bounds,
   stable names, and decode metadata.
2. Normalize a linear `<=` or `>=` constraint and derive a finite slack range
   from declared variable bounds.
3. Encode the slack with the bounded-integer mechanism and lower the
   inequality to a linear equality whose square remains quadratic.
4. Retain the original inequality in compilation metadata so feasibility is
   checked against the business constraint, not merely the generated equality.
5. Add a capacity convenience helper only after it delegates to the generic
   inequality path.
6. Test coefficient signs, non-unit bounds, non-power-of-two slack ranges,
   maximization penalty direction, decode order, and infeasible assignments.

Choose public API names only while implementing that extension. Until those
methods exist and are exported from `quspect/optimization/__init__.py`, describe
them as required capabilities rather than callable code.

## Build the full model after the extension

Use:

- binary `open[site]` and `assigned[demand, site]` variables;
- bounded integer overflow or allocated-resource variables only when genuinely
  required;
- fixed opening, compute, transport, and overflow costs as separately
  recoverable objective components;
- one assignment equality per demand point;
- assignment-implies-open constraints;
- capacity inequalities such as
  `sum(load[d] * assigned[d, site]) <= capacity[site] * open[site] + overflow[site]`;
- explicit bounds and penalties for overflow and slack.

Document units and coefficient scaling. Bound each penalty against the largest
possible business-objective improvement from violating its constraint.

## Compile, decode, and validate

After the generic extension is complete, use
`compile_quadratic_program -> bqm_to_ising -> compile_hardware_ising`, then
solve locally with `NativeIsingSolver` or pass the unchanged hardware artifact
to a future adapter.

Decode selected sites, every demand assignment, per-site load, capacity,
overflow, slack, and each objective component. Validate original assignment,
implication, bound, and capacity rules independently of compiled penalty
energy. Exhaustively enumerate tiny fixtures and verify:

- every business-feasible solution remains feasible after encoding;
- generated auxiliary bits decode within bounds;
- no invalid bit pattern creates an out-of-range integer;
- BQM, Ising, and hardware objective values preserve their documented scale
  and offset;
- heuristic results are checked for feasibility before business decoding.

Read `docs/source/optimization.rst` for the layer contract,
`docs/source/optimization_applications.rst` for the known capability gap,
`quspect/optimization/program.py` and `compiler.py` for current semantics, and
`tests/optimization/test_program.py` plus
`tests/optimization/test_end_to_end.py` for validation patterns.
