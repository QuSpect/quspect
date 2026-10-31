---
name: quspect-task-scheduling
description: Model, implement, explain, or validate task-to-machine assignment and load-balancing problems with the QuSpect quadratic-optimization pipeline. Use for distributing independent jobs across workers, minimizing load concentration or assignment cost, decoding schedules, choosing penalties, or deciding whether a request is a supported assignment model versus a true job-shop scheduling problem.
---

# QuSpect Task Scheduling

## Fix the model boundary

Treat the currently supported scheduling pattern as assignment and load
balancing:

- Define one binary variable `assigned[task, machine]` for every allowed pair.
- Require each task to select exactly one machine.
- Minimize assignment cost, a quadratic load proxy, or their weighted sum.
- Decode a task-to-machine map and recompute every machine load.

Do not call this a full job-shop model. A true job-shop problem also represents
operation precedence, machine-time exclusion, start times or an ordering
decision, and usually makespan inequalities. Stop and identify those missing
requirements before modeling such a request.

## Build the quadratic program

1. Preserve explicit task and machine order; it determines variable and result
   order.
2. Create variables with
   `QuadraticProgram.binary_array("assigned", (task_count, machine_count))`.
3. Construct each machine load as
   `load[m] = sum(duration[t, m] * assigned[t, m] for t)`.
4. Minimize a quadratic expression such as `sum(load[m] * load[m] for m)` plus
   any linear assignment costs. Document when this is only a load-concentration
   proxy rather than makespan.
5. Add one `add_linear_equality(sum(assigned[t, :]) - 1, ...)` constraint per
   task.
6. Choose penalties above the largest plausible objective improvement from
   violating a constraint. Treat this as a model parameter and test it; never
   assume an arbitrary constant is sufficient.

Use only allowed task-machine pairs when eligibility is sparse. Either omit
forbidden variables or constrain the binary conflict with a supported
nonnegative-zero expression. Keep business data and policies outside
`quspect.optimization`.

## Compile and solve

Use the existing layer sequence without bypassing an intermediate
representation:

```python
compiled = compile_quadratic_program(program)
logical = bqm_to_ising(compiled.bqm)
hardware = compile_hardware_ising(logical)
result = NativeIsingSolver(seed=7, num_reads=64).solve(hardware)
```

Convert each retained logical spin row with `spin_samples_to_binary`. Prefer the
best feasible retained sample under `compiled.evaluate_objective(bits)` over
blindly accepting one heuristic sample. Use `compiled.decode(bits)` for the
stable name-to-bit mapping.

## Decode and validate

Return business-level data, not raw spins:

- Map every task to exactly one machine.
- Recompute per-machine jobs, total duration, and the stated objective terms.
- Reject a sample unless `compiled.is_feasible(bits)` is true.
- Assert that every task appears once and only on an eligible machine.
- Compare small fixtures with exhaustive enumeration and assert the known
  optimum or optimal value.
- Test deterministic variable order and the full
  `QuadraticProgram -> BQM -> Ising -> HardwareIsingProgram` boundary.
- Run a native-solver test separately and tolerate heuristic variability unless
  the fixture and solver settings make the optimum reliable.

Read `docs/source/optimization.rst` for the layer contract,
`docs/source/optimization_applications.rst` for the model-family boundary,
`quspect/optimization/program.py` and `compiler.py` for supported constraints,
and `tests/optimization/test_end_to_end.py` for an end-to-end test pattern.
