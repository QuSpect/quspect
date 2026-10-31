---
name: quspect-optimization
description: QuSpect binary quadratic and Ising optimization architecture guidance. Use when Codex needs to model constrained binary problems, construct or debug `QuadraticProgram`, `BinaryQuadraticModel`, or `IsingProblem`, convert QUBO and Ising forms, lower a model to `HardwareIsingProgram`, encode the dense CSV transport matrix, run `NativeIsingSolver`, decode samples, check feasibility and business objectives, or design a provider-neutral solver adapter.
---

# QuSpect Optimization

## Inspect the authoritative implementation

Work from the active repository rather than remembered signatures:

1. Read `quspect/optimization/__init__.py` for public exports.
2. Read [references/layer-contracts.md](references/layer-contracts.md) for data
   models, energy conventions, conversion invariants, and test locations.
3. Inspect the relevant implementation and focused tests before editing it.
4. Read `docs/source/optimization.rst` for the public pipeline and
   `docs/source/ising_solver.rst` for native-solver behavior.

Use `$quspect-api` only when the request also involves quantum circuits,
variational algorithms, circuit backends, or qubit routing.

## Route application work

Use the matching scenario skill when business modeling is the main task:

- `$quspect-max-cut` for two-way graph partitioning;
- `$quspect-route-planning` for visit-once paths and tours;
- `$quspect-genome-assembly` for directed read-overlap ordering;
- `$quspect-task-scheduling` for task assignment and load balancing;
- `$quspect-edge-coverage` for fixed-cardinality site coverage;
- `$quspect-cloud-edge-network` for capacitated placement and allocation;
- `$quspect-portfolio-selection` for binary mean-variance selection;
- `$quspect-wireless-channels` for interference-aware channel assignment.

Keep this skill active alongside a scenario skill when changing shared
intermediate representations, lowering, codecs, native execution, or the
future remote-adapter boundary.

## Choose the highest useful input layer

- Start with `QuadraticProgram` for named business variables, objectives, and
  supported constraints.
- Start with `BinaryQuadraticModel` when the caller already has canonical
  binary or spin coefficients.
- Use `bqm_from_qubo_matrix` when the input is a QUBO matrix.
- Start with `IsingProblem` when logical fields and couplings are already known.
- Accept `HardwareIsingProgram` only at a solver or transport boundary.

Do not force application code to construct a provider matrix directly. Preserve
variable order and decode metadata at every lowering step.

## Follow the complete local pipeline

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
result = NativeIsingSolver(seed=7, num_reads=64).solve(hardware)
bits = spin_samples_to_binary(result.best_sample)

assert compiled.is_feasible(bits)
solution = compiled.decode(bits)
business_value = compiled.evaluate_objective(bits)
```

Inspect all retained samples when a heuristic solver's lowest penalized energy
may still be infeasible. Rank feasible candidates with the original business
objective, respecting its optimization sense.

## Preserve modeling semantics

- Use `program.minimize(...)` or `program.maximize(...)` explicitly.
- Use `add_linear_equality(residual, ...)` only for a linear residual equal to
  zero; compilation squares it.
- Use `add_nonnegative_quadratic_zero(expression, ...)` only when the expression
  is provably nonnegative on every binary assignment.
- Choose soft-penalty coefficients from an objective bound, not by habit.
- Always report feasibility separately from the unpenalized business value.
- Keep business names and result presentation above the solver-neutral models.

Do not invent APIs for general inequalities, bounded integers, automatic slack
variables, or capacity constraints. Add reusable upper-layer support and tests
first when a problem truly needs those features.

## Preserve the hardware boundary

Treat `HardwareIsingProgram` as the contract shared by the local C++ simulator
and a future remote adapter. Do not add authentication, networking, task
creation, or polling to lowering or CSV encoding. A remote adapter may consume
the same validated matrix and metadata without changing upstream models.

## Validate proportionally

For small models, enumerate all binary assignments and independently confirm
the feasible optimum. Then run:

```bash
python -m pytest -o addopts='' -q tests/optimization
python -m doctest -v quspect/optimization/model.py \
  quspect/optimization/conversion.py \
  quspect/optimization/hardware.py
```

Run the native C++ tests after solver or binding changes:

```bash
ctest --test-dir cpp/build-ninja --output-on-failure
```
