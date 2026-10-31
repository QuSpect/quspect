# QuSpect Variational Workflows

Use this reference for VQE, QAOA, Max-Cut, expectation objectives, gradients,
external optimizers, parameter sweeps, finite-shot execution, or recoverable QPU
jobs.

## Contents

1. [Execution model](#execution-model)
2. [Build an objective](#build-an-objective)
3. [Choose a differentiation strategy](#choose-a-differentiation-strategy)
4. [Exact optimization with BFGS](#exact-optimization-with-bfgs)
5. [QAOA Max-Cut](#qaoa-max-cut)
6. [VQE pattern](#vqe-pattern)
7. [Shot-based and hardware execution](#shot-based-and-hardware-execution)
8. [Sweeps, jobs, and recovery](#sweeps-jobs-and-recovery)
9. [Result contract](#result-contract)
10. [Limits and cost model](#limits-and-cost-model)
11. [Validation checklist](#validation-checklist)

## Execution model

Keep the problem separate from the execution strategy:

```text
parameterized circuit + named Pauli observables
                    |
          ObservableObjective
             /             \
    AdjointEngine       SamplingEngine
    exact C++ SV       shots / local / QPU
             \             /
      ObjectiveEvaluation / ObjectiveSweep
                    |
           external optimizer
```

Compile once and retain the prepared plan. Recompiling in every optimizer
iteration discards the cached circuit tape, Pauli masks, and parameter layout.

## Build an objective

Use a circuit without measurements and a mapping of stable names to
`QubitOperator` objects:

```python
from quspect import Parameter, QuantumCircuit, QubitOperator
from quspect.variational import ObservableObjective

theta = Parameter("theta")
ansatz = QuantumCircuit(name="ansatz").ry(theta, 0).cx(0, 1)
hamiltonian = QubitOperator({
    (): -0.2,
    ((0, "Z"),): 0.7,
    ((0, "X"), (1, "X")): 0.3,
})
objective = ObservableObjective(ansatz, {"energy": hamiltonian})
```

Use multiple named observables when diagnostics must share the same state
preparation. Mapping order defines result row order.

Parameters may be reused and may appear in affine expressions such as
`2 * theta - phi + 0.1`. Do not create products of symbolic parameters.

## Choose a differentiation strategy

Use exact adjoint differentiation for noiseless local research and optimizer
development:

```python
from quspect.variational import AdjointEngine

plan = AdjointEngine(dtype="float64").compile(objective)
result = plan.evaluate({theta: 0.2})
value = result.value("energy")
gradient = result.derivative("energy")
```

Use `float32` only when its memory/speed trade-off is intentional. Adjoint
differentiation uses a full state vector and one costate per observable.

Use sampling when finite-shot statistics or a QPU is required:

```python
from quspect import SamplingEngine, Simulator

sampled_plan = SamplingEngine(
    Simulator(seed=7),
    shots=8192,
    grouping="qwc",
).compile(objective)
sampled_result = sampled_plan.evaluate({theta: 0.2})
```

Both strategies return the same value/Jacobian result shape. Expect statistical
differences from sampled evaluation; compare them using shot uncertainty, not
exact equality.

## Exact optimization with BFGS

SciPy minimizes. For VQE energy minimization, return the energy and its gradient
directly. For a maximization problem, negate both.

```python
import numpy as np
from scipy.optimize import minimize

x0 = np.zeros(len(plan.parameter_order))

def energy_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
    evaluation = plan.evaluate(x)
    return evaluation.value("energy"), evaluation.derivative("energy")

optimum = minimize(
    energy_and_gradient,
    x0=x0,
    jac=True,
    method="BFGS",
    options={"gtol": 1e-6, "maxiter": 200},
)

bindings = dict(zip(plan.parameter_order, optimum.x.tolist(), strict=True))
optimized_circuit = objective.circuit.bind_parameters(bindings)
```

Use the prepared plan's `parameter_order`; do not reconstruct vector order from
sets or parameter names. Returning `(value, gradient)` with `jac=True` avoids
evaluating the quantum objective twice for separate SciPy `fun` and `jac` calls.

## QAOA Max-Cut

For an unweighted graph, maximize

```text
C = sum_(u,v) (I - Z_u Z_v) / 2.
```

Construct the cost operator:

```python
def max_cut_operator(edges):
    terms = {(): len(edges) / 2}
    for u, v in edges:
        zz = tuple(sorted(((u, "Z"), (v, "Z"))))
        terms[zz] = terms.get(zz, 0.0) - 0.5
    return QubitOperator(terms)
```

Construct a depth-`p` QAOA ansatz using supported differentiable rotations:

```python
def qaoa_ansatz(num_qubits, edges, depth):
    gammas = tuple(Parameter(f"gamma_{layer}") for layer in range(depth))
    betas = tuple(Parameter(f"beta_{layer}") for layer in range(depth))
    circuit = QuantumCircuit(name=f"max_cut_qaoa_p{depth}")

    for qubit in range(num_qubits):
        circuit.h(qubit)

    for gamma, beta in zip(gammas, betas, strict=True):
        for u, v in edges:
            # exp(-i gamma C), ignoring its global phase.
            circuit.cx(u, v).rz(-gamma, v).cx(u, v)
        for qubit in range(num_qubits):
            circuit.rx(2 * beta, qubit)
    return circuit
```

Compile the cut expectation and maximize it with BFGS:

```python
edges = ((0, 1), (1, 2), (2, 3), (3, 0))
circuit = qaoa_ansatz(4, edges, depth=2)
objective = ObservableObjective(circuit, {"cut": max_cut_operator(edges)})
plan = AdjointEngine().compile(objective)
x0 = np.asarray([
    0.4 if parameter.name.startswith("gamma_") else 0.2
    for parameter in plan.parameter_order
])

def loss_and_gradient(x):
    evaluation = plan.evaluate(x)
    return -evaluation.value("cut"), -evaluation.derivative("cut")

optimum = minimize(loss_and_gradient, x0, jac=True, method="BFGS")
```

After optimization, bind `optimum.x`, obtain a state vector or add measurements,
and decode bit `q` from integer state index with `(state >> q) & 1`. Validate the
reported cut against a classical brute-force solution for small example graphs.
Avoid an all-zero QAOA start because it can be a symmetry-induced stationary
point; use a deterministic nonzero start or several random starts.

## VQE pattern

1. Build or import a `FermionOperator`.
2. Transform it with `jordan_wigner` or `bravyi_kitaev`.
3. Construct a hardware-efficient or chemistry-informed ansatz from symbolic
   `RX`, `RY`, and `RZ` rotations plus fixed entangling gates.
4. Wrap the ansatz and qubit Hamiltonian in `ObservableObjective`.
5. Compile once with `AdjointEngine` for development or `SamplingEngine` for
   finite-shot/QPU execution.
6. Minimize the named energy value with the matching derivative.
7. Bind the optimum and evaluate diagnostics in a separate objective if needed.

Example transform:

```python
from quspect import FermionOperator
from quspect.operators.transforms import jordan_wigner

fermionic_hamiltonian = FermionOperator.number_operator(0, coefficient=-1.0)
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
```

## Shot-based and hardware execution

`SamplingEngine` creates basis-change/measurement circuits, groups
qubit-wise-commuting terms, and computes occurrence-level parameter-shift
gradients followed by the affine chain rule.

```python
import os

from quspect.backends.logicalqubit import LogicalQubitProvider

provider = LogicalQubitProvider()
backend = provider.get_backend(os.environ["LQCLOUD_BACKEND"])
hardware_plan = SamplingEngine(
    backend,
    shots=4096,
    execution_options={
        "transpile": True,
        "physical_qubits": [7, 8, 9, 10, 11],
        # Optional fixed start: logical q0→7, q1→8, ...
        "initial_layout": [7, 8],
        "readout_correction": True,
    },
).compile(objective)

job = hardware_plan.submit({theta: 0.2})
evaluation = job.result(timeout=600)
```

Choose backend options only after checking its capabilities and provider
contract; use `provider.list_backends()` to discover valid device names. Let
`SamplingEngine` control `shots` and `result_format`. A practical workflow is to
optimize locally with exact adjoint gradients, then submit the optimum to a QPU
for finite-shot validation. Cloud `SamplingEngine` gradients are parameter-shift
estimates, not exact gradients.

Use `physical_qubits` to constrain every mapped wire and routing SWAP to a
calibrated connected device subset. Add `initial_layout` only when the starting
logical-to-physical assignment must also be fixed. The layout length must equal
the objective circuit's logical qubit count; the physical pool may be larger so
SABRE can use workspace wires. For an already compiled, topology-valid circuit,
use `transpile=False` with direct `initial_layout` pass-through and omit
`physical_qubits`.

Read [hardware-routing.md](hardware-routing.md) before debugging mapping or
result-bit order. A complete local-optimization/QPU-validation pattern is in
`examples/qaoa_max_cut.py`.

## Sweeps, jobs, and recovery

Evaluate exact parameter batches without recompiling:

```python
sweep = plan.evaluate_many([
    {theta: -0.2},
    {theta: 0.0},
    {theta: 0.2},
])
```

For sampled/QPU batches, use `submit_many` or its blocking `evaluate_many`
shortcut. Several points are flattened into one logical execution request:

```python
sweep_job = sampled_plan.submit_many([{theta: -0.2}, {theta: 0.2}])
sweep = sweep_job.result(timeout=600)
```

Persist only the credential-free job reference in a trusted location:

```python
reference = job.reference
if reference is None:
    raise RuntimeError("The immediate objective has no remote job to recover")
saved = reference.to_dict()

# In a later process with fresh credentials:
from quspect import JobReference

reference = JobReference.from_dict(saved)
execution_job = provider.retrieve_job(reference)
evaluation = hardware_plan.attach(execution_job).result(timeout=600)
```

Use `attach_many` for recovered sweeps. Attachment validates workload
fingerprint, backend, labels, shapes, shots, and result format. A job reference
is a manifest, not an authenticity signature.

## Result contract

For `M` observables and `P` parameters:

- `ObjectiveEvaluation.values`: shape `(M,)`
- `ObjectiveEvaluation.jacobian`: shape `(M, P)`
- `ObjectiveSweep.values`: shape `(B, M)`
- `ObjectiveSweep.jacobian`: shape `(B, M, P)`
- `observable_order`: names the observable axis
- `parameter_order`: names the parameter axis
- `metadata`: records backend and differentiation strategy

Use `value(name_or_index)`, `derivative(name_or_index)`, and `sweep.at(index)`
instead of collapsing dimensions manually.

## Limits and cost model

- Symbolic differentiable gates are currently `RX`, `RY`, and `RZ` after
  decomposition. Their angles may be affine expressions and share parameters.
- Adjoint objectives require noiseless unitary state preparation, no
  measurements, and fixed real Hermitian Pauli observables.
- Sampled objectives also reject explicit noise channels in state preparation;
  physical QPU noise still affects results naturally.
- Sampled gradients reject symbolic controlled rotations and symbolic
  observable coefficients.
- Sampling experiments per point are approximately
  `measurement_groups * (1 + 2 * symbolic_rotation_occurrences)`.
- QWC grouping reduces circuits but is not general commuting grouping.
- Exact state-vector memory grows exponentially. Adjoint working memory is
  roughly one primal state plus one costate per observable.
- Exact `evaluate_many` currently iterates points in Python; it avoids
  recompilation but is not a native vectorized kernel.
- Optimizer convergence is non-convex and initialization-dependent. Use
  multiple starts when solution quality matters.

## Validation checklist

1. Verify the objective circuit has no measurements or explicit channels.
2. Verify observables use valid in-range qubits and fixed real coefficients.
3. Inspect `plan.parameter_order` before constructing optimizer vectors.
4. Compare adjoint gradients with central finite differences on a tiny case.
5. For sampled gradients, compare within statistical uncertainty.
6. Confirm minimization/maximization signs for both value and gradient.
7. Validate QAOA/VQE examples against a classically solvable small instance.
8. Run `tests/native/test_gradient.py`, `tests/simulation/test_gradient.py`, and
   `tests/variational/` after changing variational behavior.
