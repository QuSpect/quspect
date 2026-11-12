# QuSpect Compilation: transpile, synthesis, Hamiltonian simulation

Use this reference to compile a circuit for a gate set and coupling graph
(`quspect.transpiler.transpile`), to synthesize two-qubit unitaries
(`quspect.synthesis`), or to compile `exp(-i H t)` for a Pauli sum
(`quspect.hamsim`). For submitting to a cloud device, which compiles on its own,
read [hardware-routing.md](hardware-routing.md).

## Contents

1. [Targets](#targets)
2. [transpile()](#transpile)
3. [Reading a TranspileResult](#reading-a-transpileresult)
4. [Checking a compiled circuit](#checking-a-compiled-circuit)
5. [Individual passes](#individual-passes)
6. [Two-qubit synthesis](#two-qubit-synthesis)
7. [Hamiltonian simulation](#hamiltonian-simulation)
8. [Limits](#limits)
9. [Tests](#tests)

## Targets

A `CompilationTarget` is a coupling graph plus an optional native gate set:

```python
from quspect.transpiler import CompilationTarget

line = CompilationTarget.from_coupling_map(
    5,
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    native_gates={"cz", "rz", "sx", "x"},
)
pool, dense_to_physical = line.induced_subgraph([1, 2, 3])
```

- `CompilationTarget.from_coupling_map(num_qubits, coupling_map, native_gates=None)`
  takes undirected edges; `.num_qubits`, `.coupling_graph`, `.adjacency` and
  `.native_gates` describe it.
- `CompilationTarget.from_backend(backend)` reads a cloud backend's topology and
  `native_gate_set`.
- `.induced_subgraph(physical_qubits)` returns a dense target and
  `dense_to_physical[dense] = physical`.
- The native gates need an entangler (`cz`, `cx`, or one registered with
  `register_entangler`) and single-qubit gates that reach every rotation, such
  as `{rz, sx}`, `{rz, h}`, `{rz, rx}` or `{rz, ry}`. A set that cannot express
  the circuit raises `ValueError` naming the gates.
- With `native_gates=None` nothing is lowered: gates keep their names and
  single-qubit runs fuse into `u3`.

## transpile()

`transpile` is the one entry point for layout, routing, basis translation and
optimization:

```python
from quspect import QuantumCircuit
from quspect.transpiler import transpile

ghz = QuantumCircuit(name="ghz").h(0).cx(0, 1).cx(0, 2).cx(0, 3).measure_all()
compiled = transpile(ghz, line, optimization_level=2, seed=7)

print(compiled.stats.two_qubit_gates, compiled.stats.two_qubit_depth)
print(compiled.initial_layout, compiled.final_layout)
```

Signature (keyword-only after `target`): `transpile(circuit, target, *,
optimization_level=2, seed=None, initial_layout=None, iter_num=3,
layout_trials=None, swap_trials=None, max_evaluations=None, work_limit=None,
time_limit=None, exact_layout=True, elide_permutations=True,
approximation_degree=1.0, depth_weight=0.0) -> TranspileResult`.

Optimization levels:

| Level | What it does |
| --- | --- |
| 0 | Decompose, route with SABRE, decompose. No optimization. |
| 1 | Cancel self-inverse pairs, consolidate two-qubit blocks before and after routing. Fast. |
| 2 (default) | Level 1 plus three-qubit block resynthesis, phase folding, block-aware routing, a routing refinement and DAG optimization after routing. |
| 3 | Level 2 plus a search over many routings; the result with the fewest two-qubit gates is kept. Never more two-qubit gates than level 2 for the same seed. |

Options worth knowing:

- `seed`: a fixed seed gives the same output. Pass one in tests and examples.
- `initial_layout[logical] = physical` fixes the starting placement and turns
  the layout search off.
- `exact_layout=True` first looks for a placement that needs no SWAP.
- `elide_permutations=True` absorbs trailing routing SWAPs into `final_layout`
  instead of emitting them.
- `approximation_degree`: `1.0` (default) only drops what is zero to numerical
  precision. Lower values allow each two-qubit block an average gate infidelity
  of `1 - approximation_degree`; `None` allows none at all.
  `approximation_tolerance(degree)` gives the number, and
  `stats.approximation_error` what was used.
- Level 3 only: `swap_trials`, `max_evaluations`, `work_limit` (a deterministic
  work budget), `time_limit` (a wall-clock cap; results are then not
  reproducible) and `depth_weight` (trade two-qubit gates for depth).

Symbolic parameters stay symbolic through `transpile`; bind them before or
after.

## Reading a TranspileResult

`TranspileResult` has two views of the same compiled circuit:

- `circuit`: indexed by physical qubits of the target. Every two-qubit gate is
  on a coupling edge.
- `wire_circuit` with `wire_layout[wire] = physical`: SABRE's wire space, the
  form the cloud backends submit together with a layout.

Layouts are `layout[logical] = physical`: logical qubit `q` starts on
`initial_layout[q]` and ends on `final_layout[q]`. The two differ when routing
moved qubits. Measurements keep their classical bits, so counts of
`compiled.circuit` are keyed exactly like counts of the input circuit:

```python
from quspect import Simulator

before = Simulator(seed=1).run(ghz, shots=2000).counts
after = Simulator(seed=1).run(compiled.circuit, shots=2000).counts
assert set(before) == set(after) == {"0000", "1111"}
```

`compiled.stats` (`TranspileStats`) reports `optimization_level`,
`two_qubit_gates`, `two_qubit_depth`, `total_gates`, `depth`, `num_swaps`,
`selected` (which candidate won), `num_candidates`, `num_evaluated`,
`layout_trials`, `swap_trials`, `seconds`, `stage_seconds` and the
approximation fields.

## Checking a compiled circuit

For a circuit without measurements, compare state vectors through the final
layout: logical qubit `q` of the input is on physical `final_layout[q]` at the
end, and qubit 0 is the lowest bit of a state index.

```python
import numpy as np

source = QuantumCircuit().h(0).cx(0, 1).cx(0, 2).t(2).cx(1, 3)
routed = transpile(source, line, seed=3)

expected = np.asarray(Simulator().statevector(source))
actual = np.asarray(Simulator().statevector(routed.circuit))
moved = np.zeros_like(actual)
for index, amplitude in enumerate(expected):
    physical = 0
    for logical in range(source.num_qubits):
        if (index >> logical) & 1:
            physical |= 1 << routed.final_layout[logical]
    moved[physical] = amplitude
assert abs(np.vdot(moved, actual)) ** 2 > 1 - 1e-9
```

This check assumes every logical qubit starts in `|0>` on
`initial_layout[q]`, which holds for a fresh device.
`check_compilation_consistency(circuit, compiler=...)` runs a similar check
against Qiskit and needs the optional `qiskit` dependency.
`generate_random_circuit(num_qubits, depth, seed=...)` makes test inputs.

## Individual passes

Exported from `quspect.transpiler` for building custom pipelines:

- `SABRE(target, ...)`: `.solve(circuit, iter_num=3, seed=None,
  initial_mapping=None, ...)` returns `(routed_circuit, initial_mapping,
  final_mapping)` in the target's index space; `.route_candidates(...)` returns
  `RoutingCandidate` objects for a search.
- `HighLevelOptimizePass(window=256)`, `ConsolidateBlocksPass(target=None,
  ...)`, `ThreeQubitBlockPass()`, `PhaseFoldingPass()`: each has
  `.run(circuit)`; the stats classes (`HighLevelStats`, `ConsolidationStats`,
  `ThreeQubitBlockStats`, `PhaseFoldingStats`) describe what a pass did.

`quspect.transpiler._passes` holds the internal building blocks
(`DecomposePass(target)`, `LayoutRoutingPass(target, ...)`, `OptimizePass()`,
`OptimizePass.from_target(target)`, `TranspilerPipeline(passes)`). Prefer
`transpile`; reach for these only when changing the compiler itself.

## Two-qubit synthesis

`quspect.synthesis` turns a 4×4 unitary into a circuit with the fewest
entangling gates (0 to 3), through the KAK decomposition:

```python
import numpy as np
from quspect.synthesis import (
    TwoQubitBasis,
    num_cz_required,
    synthesize_two_qubit,
    weyl_coordinates,
)

swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
assert num_cz_required(swap) == 3
a, b, c = weyl_coordinates(swap)

on_cz = synthesize_two_qubit(swap)
ibm = TwoQubitBasis.from_gates({"cx", "rz", "sx", "x"})
on_cx = synthesize_two_qubit(swap, basis=ibm)
assert {ins.gate.name for ins in on_cx} <= {"cx", "rz", "sx", "x"}
```

- `synthesize_two_qubit(unitary, *, atol=1e-9, verify_atol=1e-9, basis=None)`
  returns a `QuantumCircuit` on qubits 0 and 1 that equals `unitary` up to a
  global phase. Qubit 0 is the most significant qubit of the matrix. The
  default basis is `cz` with `{h, rz, x}`.
- `TwoQubitBasis(entangler="cz", euler="zxz")`;
  `TwoQubitBasis.from_gates(names)` picks the basis for a native gate set and
  returns `None` when the set has no usable entangler or single-qubit basis.
- `kak_decompose(unitary)` returns a `TwoQubitKAK` (`a`, `b`, `c`,
  `global_phase`, the four local unitaries, `.unitary()`).
- `register_entangler(name, matrix)` adds a native two-qubit gate that is
  locally equivalent to CZ, so targets may list it; `entangler(name)` looks one
  up. A circuit using a registered gate can only be simulated when
  `quspect.gates.gate_matrix` also knows the gate.

## Hamiltonian simulation

`compile_evolution` compiles `exp(-i H t)` with a product formula. It fixes the
formula first (order, steps, term order), then compiles those rotations
exactly, by Pauli-network synthesis on the coupling graph or by the textbook
ladder circuit through `transpile`, and keeps the cheaper circuit.

```python
from quspect.hamsim import compile_evolution

chain = CompilationTarget.from_coupling_map(
    4, [(0, 1), (1, 2), (2, 3)], native_gates={"cz", "h", "rz", "s", "sdg", "x"}
)
heisenberg = [(p, (i, i + 1), 1.0) for i in range(3) for p in ("XX", "YY", "ZZ")]

fixed = compile_evolution(heisenberg, 0.5, chain, steps=4, order=2)
print(fixed.method, fixed.two_qubit_gates, fixed.candidates)

to_accuracy = compile_evolution(heisenberg, 0.5, chain, eps=1e-2, order=2)
print(to_accuracy.steps, to_accuracy.error_metric, to_accuracy.error_estimate)
```

Inputs:

- `hamiltonian`: a sequence of `(letters, qubits, coeff)` triples such as
  `("XX", (0, 1), 1.0)`, a `QubitOperator`, or a Qiskit `SparsePauliOp`.
  Coefficients must be real. Term order is kept; it is the order of the
  formula. `pauli_terms(hamiltonian)` returns the normalized
  `(num_qubits, terms)`.
- `time`: the evolution time `t`.
- `target`: optional; all-to-all when omitted.
- Give `steps`, or `eps` to get the fewest steps whose estimated error is at
  most `eps`.
- `order`: 1 (Lie-Trotter), 2 (Strang, default) or 4 (Suzuki).
- `term_order`: `"input"` (default), `"support"` (terms on the same qubits
  adjacent), or `"auto"` (needs `eps`; compares the candidates by steps and
  compiled gate count, see `select_term_order`).
- `error_metric`: `"state"` simulates the error on random states (at most 16
  qubits), `"bound"` uses a commutator bound of the operator-norm error (orders
  1 and 2 only), `"auto"` (default) picks `"state"` when the size allows.
- `methods`: `("network", "ladder")` by default; pass one to force it.
- `optimization_level`, `seed`: as in `transpile`.

`EvolutionResult` carries `circuit` (on physical qubits), `initial_layout`,
`final_layout`, `order`, `steps`, `term_order`, `rotations`, `method`,
`two_qubit_gates`, `two_qubit_depth`, `error_estimate`, `error_metric`,
`candidates` and `transpile_result`.

A `QubitOperator` works directly, and the pieces are available on their own:

```python
from quspect import QubitOperator
from quspect.hamsim import (
    commutator_bound,
    order_terms,
    pauli_terms,
    product_formula,
    state_error,
)

ising = QubitOperator({((0, "Z"), (1, "Z")): 1.0, ((0, "X"),): 0.5, ((1, "X"),): 0.5})
fourth_order = compile_evolution(ising, 0.3, steps=1, order=4)

num_qubits, terms = pauli_terms(ising)
rotations = product_formula(terms, 0.3, steps=2, order=2)
sampled = state_error(num_qubits, terms, 0.3, rotations)
bounded = commutator_bound(num_qubits, terms, 0.3, steps=2, order=2)
assert sampled <= bounded
grouped = order_terms(heisenberg, "support")
```

A rotation `(letters, qubits, theta)` stands for `exp(-i theta P)`. Other
exports: `one_step`, `merge_commuting`, `ladder_circuit`, `synthesize_network`
(returns a `NetworkCircuit`), `select_term_order` (returns a
`TermOrderChoice`), and the graph helpers `long_path`, `cycle_of_length` and
`connected_active_set`.

## Limits

- Add noise after compiling (`NoiseModel.apply` on the compiled circuit), so
  that the model sees the gates that will run. Releases up to 0.6.2 drop noise
  channels given to `transpile` without an error.
- `"state"` errors are sampled on a few states, so they can be below the
  operator-norm error. Use `"bound"` when a guarantee is needed, and expect it
  to ask for more steps.
- `error_metric="bound"` raises for `order=4`.
- Level 3 with `time_limit` is not reproducible.

## Tests

- `tests/transpiler/` (start with `test_transpile.py` and `test_sabre.py`)
- `tests/synthesis/`
- `tests/hamsim/test_hamsim.py`
- `qiskit_tests/` cross-checks against Qiskit when it is installed
