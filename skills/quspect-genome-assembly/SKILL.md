---
name: quspect-genome-assembly
description: Build, solve, decode, and validate sequence-read ordering models with the QuSpect quadratic optimization stack. Use for genome read assembly based on directed suffix-prefix overlap weights, read-position one-hot variables, linear or circular ordering, and their BQM, Ising, hardware-matrix, or native-solver flow.
---

# QuSpect Genome Assembly

## Inspect the local contracts

Read the [optimization architecture](../../docs/source/optimization.rst), the
[assembly application pattern](../../docs/source/optimization_applications.rst),
and the [symbolic program API](../../quspect/optimization/program.py). Inspect
`quspect/optimization/expression.py`, `compiler.py`, and
`hardware.py` before changing the model or lowering path.

## Prepare directed overlaps

Keep biological preprocessing outside the optimizer:

1. Give every read a stable unique identifier.
2. Decide how to handle quality filtering, contained reads, duplicates, and
   reverse complements before building variables.
3. Compute a finite directed overlap weight `overlap[i][j]` for placing read
   `j` after read `i`.
4. Preserve direction. Never replace the overlap matrix with
   `(overlap + overlap.T) / 2`.
5. Retain overlap lengths or alignment metadata needed to construct a contig;
   an optimization score alone may not be enough for sequence stitching.

Use a suffix-of-`i` to prefix-of-`j` score or another explicitly documented
directed alignment score. Set or reject diagonal values consistently.

## Build the read-position model

Create `placed[i, p]`, equal to one when read `i` occupies position `p`. For
`n` reads, declare an `(n, n)` binary array. Maximize adjacent directed
overlap:

```text
sum(position p, reads i != j)
    overlap[i][j] * placed[i,p] * placed[j,p+1]
```

For linear assembly, use positions `0` through `n - 2` and do not connect the
last position to the first. For circular assembly, use `(p + 1) % n`.

Add both one-hot families with `program.add_linear_equality(...)`:

```text
for every read i:      sum_p placed[i,p] - 1 == 0
for every position p:  sum_i placed[i,p] - 1 == 0
```

Name every constraint uniquely. Build sums from `QuadraticExpression()` so
empty and typed cases remain explicit. Choose a penalty that dominates the
largest possible overlap-score gain from violating a row or column, and
validate the bound on small instances. Do not assume a returned sample is
feasible merely because penalties were added.

This is intentionally a directed route-permutation model. The terms for
`i -> j` and `j -> i` use different read-position variable pairs even though
the compiled quadratic coefficient matrix is symmetric.

## Compile and solve through the common layers

Preserve the provider-neutral chain:

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

Convert retained logical spin samples to bits, discard samples for which
`compiled.is_feasible(bits)` is false, and select the feasible candidate with
the largest `compiled.evaluate_objective(bits)`. Raise a clear error when no
feasible candidate is retained.

## Decode and stitch

Decode with `compiled.decode(bits)`. For each position, require exactly one
read and build an ordered read-ID tuple. Require every input read exactly once.

Recompute the directed adjacency score from the original overlap matrix. For
simple exact suffix-prefix overlaps, start with the first read and append only
the non-overlapping suffix of each following read. For alignment-based scores,
use the retained alignment metadata instead of treating the score as an
overlap length.

Return at least:

- ordered read identifiers;
- total directed overlap score;
- feasibility status;
- the stitched sequence when the preprocessing contract supports it.

## Validate the assembly model

- Enumerate all read permutations for small fixtures and compare the maximum
  directed score.
- Include a fixture where `overlap[i][j] != overlap[j][i]`.
- Test linear and circular modes independently; catch accidental wraparound.
- Require the decoded order to contain every read exactly once.
- Compare the recomputed overlap sum with
  `compiled.evaluate_objective(bits)`.
- Assert stable logical-variable order and the hardware matrix invariants.
- Use deterministic native-solver seeds in tests, but treat annealing as
  approximate and report ambiguity among equal-score orders.

## Keep the applicability boundary explicit

- Treat this as read ordering from a precomputed overlap graph, not a complete
  production genome assembler.
- Do not silently model reverse-complement orientation, optional reads,
  branching haplotypes, repeats, indels, or coverage consensus. Extend the
  variables and biological contract deliberately when those are required.
- Account for `n^2` logical bits and a dense matrix over them, yielding
  quartic storage growth in the number of reads.
- Keep FASTA/FASTQ parsing, alignment, consensus generation, and task
  submission outside `quspect.optimization`.
- Use the local C++ `NativeIsingSolver` as the current execution target.
