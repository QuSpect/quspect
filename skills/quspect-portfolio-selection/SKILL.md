---
name: quspect-portfolio-selection
description: Build and validate fixed-cardinality mean-variance portfolio selection models with QuSpect's binary quadratic and Ising pipeline. Use for asset or project selection requests involving expected returns, covariance or pairwise risk, a required number of holdings, binary inclusion decisions, risk-aversion tuning, QUBO/Ising solving, or business-level portfolio decoding.
---

# QuSpect Portfolio Selection

## Start from the maintained example

Read `examples/portfolio_selection.py` and
`tests/optimization/test_application_examples.py` before changing signatures or
expected results. For lower-layer conversions, matrix conventions, solver
adapters, or transport work, also use `$quspect-optimization`.

## Confirm the model scope

Use one binary variable `x_i` per candidate asset and solve:

```text
maximize expected_return - risk_aversion * variance
subject to sum_i x_i = K
```

Model the terms as:

```text
expected_return = sum_i mu_i * x_i
variance = sum_i sum_j covariance[i, j] * x_i * x_j
```

Require a finite expected-return vector, a finite symmetric covariance matrix,
`K` within the candidate count, and a nonnegative risk-aversion value. State
that this is equal-weight binary selection, not continuous portfolio weighting.

## Build through the symbolic layer

1. Create stable variables with `program.binary_array("select", n)`.
2. Maximize the linear return minus the quadratic covariance risk.
3. Add `sum(variables) - K == 0` with `add_linear_equality`.
4. Choose a cardinality penalty large enough to dominate every possible
   objective improvement from violating the selection count.
5. Compile and solve through the standard QuSpect optimization pipeline.

Keep asset labels, return estimates, covariance data, and reporting in the
application layer. Do not add them to `quspect.optimization`.

## Decode and report

- Convert logical spins to binary bits.
- Reject infeasible samples with `compiled.is_feasible(bits)`.
- Map selected bits back through the compiled variable order.
- Report selected assets, expected return, variance, and
  `compiled.evaluate_objective(bits)` separately.
- When using heuristic samples, rank only feasible candidates by the original
  maximization objective.

## Handle variants deliberately

- Represent pairwise diversification rewards by negative quadratic terms only
  when their sign and business meaning are explicit.
- Add sector-exclusion pairs with nonnegative quadratic-zero constraints.
- Do not pretend continuous weights, budget inequalities, minimum lots,
  turnover, or transaction-cost piecewise functions are natively supported.
  Introduce reusable integer/slack encodings first when those requirements are
  necessary.

## Validate

For small portfolios, enumerate all `2**n` bit vectors and independently confirm
the feasible optimum and its decoded asset set. Then run:

```bash
python examples/portfolio_selection.py
python -m pytest -o addopts='' -q \
  tests/optimization/test_application_examples.py
```
