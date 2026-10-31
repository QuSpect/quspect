# QuSpect Optimization Layer Contracts

## Contents

1. [Source map](#source-map)
2. [Layered data flow](#layered-data-flow)
3. [Energy conventions](#energy-conventions)
4. [Symbolic constraints](#symbolic-constraints)
5. [Lowering and gauge metadata](#lowering-and-gauge-metadata)
6. [Solving and decoding](#solving-and-decoding)
7. [Transport boundary](#transport-boundary)
8. [Extension rules](#extension-rules)
9. [Evidence and tests](#evidence-and-tests)

## Source map

- Public exports: `quspect/optimization/__init__.py`
- Symbolic expressions: `quspect/optimization/expression.py`
- User-facing programs: `quspect/optimization/program.py`
- Program compilation: `quspect/optimization/compiler.py`
- BQM and logical Ising models: `quspect/optimization/model.py`
- QUBO/binary/spin conversion: `quspect/optimization/conversion.py`
- Hardware matrix lowering: `quspect/optimization/hardware.py`
- CSV matrix codec: `quspect/optimization/codec.py`
- Native solver adapter: `quspect/optimization/native_solver.py`
- C++ solver: `cpp/include/quspect_native/optimization/ising_solver.hpp` and
  `cpp/src/optimization/ising_solver.cpp`

## Layered data flow

```text
business data
    |
QuadraticProgram
    | compile_quadratic_program
CompilationResult + BinaryQuadraticModel
    | bqm_to_ising
IsingProblem
    | compile_hardware_ising
HardwareIsingProgram
    |                         \
NativeIsingSolver             encode_ising_matrix_csv
    |                         future transport artifact
logical spin samples
    |
binary bits -> feasibility -> business decode
```

Keep stable variable names and vector order through the complete path.
`CompilationResult` retains the original objective and constraints separately
from the penalized BQM.

## Energy conventions

`BinaryQuadraticModel` stores:

```text
E(v) = offset + linear @ v + 0.5 * v.T @ quadratic @ v
```

The quadratic matrix is symmetric with a zero diagonal. `variable_type`
selects binary `{0, 1}` or spin `{-1, +1}`, and `sense` records minimization or
maximization.

A QUBO matrix uses:

```text
E(x) = x.T @ Q @ x + offset
```

The importer accepts upper, lower, symmetric, or mixed matrices and merges
opposing off-diagonal entries. The canonical exporter is upper triangular.

Binary and spin samples use:

```text
s = 2x - 1
x = (s + 1) / 2
```

`IsingProblem` retains the BQM-style logical convention:

```text
E(s) = offset + linear @ s + 0.5 * s.T @ quadratic @ s
```

The hardware matrix uses:

```text
E_hw(t) = -t.T @ matrix @ t
```

Never mix these conventions or drop their constant offsets.

## Symbolic constraints

`QuadraticProgram` currently supports:

- named binary variables and row-major arrays;
- linear or quadratic minimization and maximization objectives;
- linear equalities compiled as squared residual penalties;
- declared-nonnegative quadratic expressions constrained to zero and compiled
  as direct penalties.

Penalties are soft. For maximization, compilation subtracts violation penalties
so infeasible samples remain disfavored. Use `compiled.is_feasible(bits)` for
the original constraints and `compiled.evaluate_objective(bits)` for the
unpenalized business value.

## Lowering and gauge metadata

`compile_hardware_ising` converts logical pair coefficients into the full,
symmetric hardware matrix. Because `t.T @ matrix @ t` counts each
off-diagonal entry twice, the matrix stores half-scaled pair couplings.

Logical linear fields become couplings to one auxiliary gauge spin. The
hardware object records:

- hardware variable order;
- logical variable count;
- optional gauge index;
- objective scale and offset;
- versioned matrix format.

Decode by multiplying logical spins by the sampled gauge and then removing
auxiliary spins. For every valid logical assignment:

```text
E_logical = objective_scale * E_hw + objective_offset
```

Treat this equality as the main backend-compatibility invariant.

## Solving and decoding

`NativeIsingSolver` passes the exact hardware matrix to the C++ simulated
annealer. It returns raw hardware samples, gauge-normalized logical samples,
hardware energies, and recovered logical objective values in energy order.

The solver is heuristic. Search `result.logical_samples` for feasible
candidates when penalties or annealing quality could make the first sample
invalid. Convert logical spins with `spin_samples_to_binary`, validate with the
compilation result, and only then decode business fields.

## Transport boundary

`encode_ising_matrix_csv` serializes only the dense numeric matrix. It does not
submit work or serialize local business/decode metadata. Keep the complete
`HardwareIsingProgram` locally so returned hardware samples can be normalized
and objective values reconstructed.

A future remote backend belongs beside `NativeIsingSolver`, consuming
`HardwareIsingProgram`. Do not put provider credentials or network behavior in
the models, conversions, lowering, or codec.

## Extension rules

- Add a reusable symbolic constraint or integer encoding only when multiple
  applications need the same semantics.
- Compile new constructs into the existing BQM boundary and retain their
  original feasibility representation.
- Do not silently approximate a general inequality with a direct quadratic-zero
  constraint.
- Keep application-specific data preparation and result formatting outside the
  core package.
- Preserve backwards-compatible aliases only when they remain provider-neutral.

## Evidence and tests

- Symbolic program tests: `tests/optimization/test_program.py`
- Compiler and feasibility tests: `tests/optimization/test_compiler.py`
- Conversion invariants: `tests/optimization/test_conversion.py`
- Hardware matrix contract: `tests/optimization/test_end_to_end.py`
- Native solver adapter: `tests/optimization/test_native_solver.py`
- Application optima: `tests/optimization/test_application_examples.py`
- C++ matrix solver: `cpp/tests/test_ising_solver.cpp`
