---
name: quspect-max-cut
description: Build, solve, decode, and validate weighted Max-Cut models with the QuSpect quadratic optimization stack. Use when implementing graph bipartitioning, cut-value objectives, unconstrained or balanced Max-Cut examples, or debugging their BQM, Ising, hardware-matrix, and native-solver flow.
---

# QuSpect Max-Cut

## Inspect the local contracts

Read the [optimization architecture](../../docs/source/optimization.rst), the
[application patterns](../../docs/source/optimization_applications.rst), and
the [end-to-end Max-Cut test](../../tests/optimization/test_end_to_end.py)
before changing public calls. Treat `quspect/optimization/__init__.py` and the
matching source as authoritative.

## Normalize the graph

1. Assign every vertex a stable application identifier.
2. Reject self-loops or define explicitly how to ignore them.
3. Combine duplicate undirected edges once; do not accidentally count both
   `(u, v)` and `(v, u)`.
4. Require finite weights. State explicitly if negative weights are allowed,
   because they reward keeping endpoints together.

## Build the quadratic program

Create one binary variable `side[v]` per vertex. Interpret `0` and `1` as the
two partitions. For an edge `(u, v)` with weight `w`, add

```text
w * (side[u] + side[v] - 2 * side[u] * side[v])
```

to the maximized cut value. This expression is `w` exactly when the endpoints
are separated and zero otherwise.

Follow this shape:

```python
from quspect.optimization import QuadraticExpression, QuadraticProgram

program = QuadraticProgram("weighted-max-cut")
side = {
    vertex: program.binary_variable(f"side[{vertex}]")
    for vertex in vertices
}
cut_value = QuadraticExpression()
for left, right, weight in edges:
    cut_value += weight * (
        side[left] + side[right] - 2 * side[left] * side[right]
    )
program.maximize(cut_value)
```

Add no constraint for ordinary Max-Cut. For a balanced variant, add
`sum(side.values()) - target_size == 0` with
`program.add_linear_equality(...)`. Derive a penalty that dominates any
objective gain from violating balance; do not rely on an arbitrary constant.

## Run every layer

Keep the application model above the reusable lowering boundary:

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
result = NativeIsingSolver(seed=7, num_reads=64, sweeps=300).solve(hardware)
bits = spin_samples_to_binary(result.best_sample)
```

Preserve this chain:
`QuadraticProgram -> BQM -> IsingProblem -> HardwareIsingProgram ->
NativeIsingSolver`. Do not construct the dense hardware matrix in graph code.

## Decode in graph terms

Call `compiled.decode(bits)`, then return two vertex collections according to
their `side[...]` values. Recompute the cut weight from the original normalized
edge list. Report both partitions and the business cut value.

Expect complementary assignments to have the same value. Do not claim a
unique solution unless symmetry has been fixed or all optimal partitions have
been enumerated.

## Validate the result

Perform all of these checks:

- Require `compiled.is_feasible(bits)` for balanced or other constrained
  variants.
- Compare the recomputed edge cut with
  `compiled.evaluate_objective(bits)`.
- Exhaustively enumerate all bit vectors for small regression fixtures and
  compare the optimum.
- Assert that hardware logical-variable order matches `compiled.variables`.
- Use a fixed native-solver seed in tests, but treat simulated annealing as
  approximate; increase reads and sweeps or retain the best feasible sample.
- Test empty, disconnected, duplicate-edge, negative-weight, and symmetric
  optimum cases according to the declared input policy.

## Keep the applicability boundary explicit

- Use this model for two-way weighted graph partitioning.
- Add a separate encoding for multiway partitioning; one bit per vertex is not
  enough.
- Recognize that a sparse graph is lowered to a dense matrix. With `n`
  vertices, the hardware artifact uses quadratic matrix storage.
- Keep graph ingestion, labels, visualization, and business reporting outside
  `quspect.optimization`.
- Do not add networking or task-submission behavior. The current execution
  target is the local C++ `NativeIsingSolver`.
