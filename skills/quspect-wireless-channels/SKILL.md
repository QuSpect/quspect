---
name: quspect-wireless-channels
description: Build and validate interference-aware wireless channel assignment models with QuSpect's binary quadratic and Ising pipeline. Use for access-point, radio, frequency, spectrum, graph-coloring, or channel-allocation requests involving exactly-one assignments, forbidden co-channel interference, preference costs, QUBO/Ising solving, or structured assignment decoding.
---

# QuSpect Wireless Channels

## Start from the maintained example

Read `examples/wireless_channel_assignment.py` and
`tests/optimization/test_application_examples.py` before changing the model.
For lower-layer matrix or solver work, also use `$quspect-optimization`.

## Normalize the business input

Collect:

- a stable ordered set of access points or radios;
- a stable ordered set of channels;
- an undirected interference edge set;
- an optional finite cost for every access-point/channel pair.

Reject unknown endpoints, self-edges unless intentionally modeled, duplicate
channel labels, and incomplete costs. Decide whether interference is a hard
prohibition or a soft cost before building the program.

## Build the hard-conflict model

Create `x[a, c]` to mean access point `a` uses channel `c`.

1. Minimize `sum_(a,c) preference_cost[a,c] * x[a,c]`.
2. For every access point, add the linear equality:

   ```text
   sum_c x[a,c] - 1 = 0
   ```

3. For every interference edge `(a, b)` and channel `c`, add:

   ```text
   x[a,c] * x[b,c] = 0
   ```

   Use `add_nonnegative_quadratic_zero` because a binary product is always
   nonnegative.

4. Choose penalties that dominate every achievable preference-cost reduction.
5. Compile and solve through the standard QuSpect optimization pipeline.

Keep topology, radio metadata, channel labels, and reporting outside the core
optimization package.

## Decode and validate

- Search retained heuristic samples for feasible candidates.
- Decode exactly one selected channel per access point in input order.
- Recheck every interference edge against the decoded assignment.
- Report preference cost separately from constraint penalties.
- For small graphs, enumerate all channel assignments and confirm the optimum.

Run the maintained checks:

```bash
python examples/wireless_channel_assignment.py
python -m pytest -o addopts='' -q \
  tests/optimization/test_application_examples.py
```

## Extend without changing semantics

- Model soft co-channel interference as weighted quadratic objective terms
  rather than zero constraints.
- Model adjacent-channel leakage with pair coefficients between specified
  channel pairs.
- Treat per-channel capacity, regional spectrum budgets, or minimum-separation
  inequalities as new reusable upper-layer requirements. Do not encode them
  with a zero constraint unless the expression is provably nonnegative and zero
  exactly characterizes feasibility.
