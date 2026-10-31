---
name: quspect-api
description: QuSpect SDK architecture and API guidance for building, editing, explaining, or debugging code that uses the `quspect` Python package. Use for QuSpect circuits, gates, affine parameters, operators, variational objectives, VQE/QAOA/Max-Cut workflows, exact adjoint or shot-based gradients, external optimizer integration, batch and recoverable jobs, simulation, noise, DAG passes, OpenQASM/QSAL, visualization, transpilation, physical-qubit pools and initial layouts, native execution, or LogicalQubit cloud backends.
---

# QuSpect API

## Inspect the installed source

Require QuSpect in the active Python environment. Locate its source before
making version-sensitive claims:

```bash
python3 -c "import pathlib, quspect; print(pathlib.Path(quspect.__file__).parent)"
```

Treat source and tests as authoritative. README and docs provide workflows but
may omit edge cases.

For symbolic binary programs, QUBO/BQM conversion, logical Ising models, dense
hardware matrices, or the native Ising solver, use `$quspect-optimization`.
Keep this skill for circuit-based QAOA, variational objectives, and the wider
quantum-circuit SDK.

## Route the task

1. Read public exports in `quspect/__init__.py` and the relevant subpackage
   `__init__.py`.
2. Read [references/api-map.md](references/api-map.md) for module boundaries,
   imports, core methods, backend execution, and focused test targets.
3. For variational algorithms, gradients, VQE, QAOA, Max-Cut, optimizer loops,
   parameter sweeps, sampled objectives, or hardware recovery, read
   [references/variational.md](references/variational.md) before coding.
4. For LogicalQubit compilation, calibrated physical-qubit selection, initial
   layouts, SABRE behavior, or mapping/result-order investigations, read
   [references/hardware-routing.md](references/hardware-routing.md).
5. Inspect the matching source and tests before relying on a signature.
6. Run the narrowest matching tests first; broaden when changing shared circuit,
   IR, backend, or result behavior.

## Preserve the architecture

- Use `QuantumCircuit` as the fluent circuit builder.
- Store operations as `CircuitInstruction`; do not bypass circuit validation.
- Use `Parameter` and affine `ParameterExpression` values for symbolic angles.
- Compile `ObservableObjective` once, then retain its prepared plan across
  optimizer iterations.
- Use `AdjointEngine` for exact noiseless state-vector values and gradients.
- Use `SamplingEngine` for finite-shot local simulation or batch-capable QPUs.
- Use `ExecutionRequest`/`ExecutionJob` for provider-neutral batch execution and
  `JobReference` for credential-free recovery.
- Use the public LogicalQubit `physical_qubits` and `initial_layout` options for
  constrained compilation; do not call private backend transpilation methods in
  application code.
- Operate optimization passes on `DAGCircuit`; convert back with the existing
  compiler path.
- Let `Simulator` delegate to the native QSAL executor.
- Implement cloud integrations through `Backend`/`CloudBackend` capabilities.

## Common imports

```python
from quspect import (
    AdjointEngine,
    ObservableObjective,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    QubitOperator,
    SamplingEngine,
    Simulator,
)
from quspect.backends import ExecutionRequest, JobReference
```

Use subpackage imports for APIs not re-exported at top level:

```python
from quspect.noise import DepolarizingChannel, NoiseModel
from quspect.ir import QASM2Exporter, dumps, loads, qasm2_loads
from quspect.visualization import compare_results, draw_svg_circuit
from quspect.backends.logicalqubit import LogicalQubitProvider
from quspect.transpiler import CompilationTarget, SABRE
```

## Key conventions

- Construct circuits with `QuantumCircuit(name="...")`; qubit and classical-bit
  counts are inferred from validated non-negative indices.
- Circuit builder methods mutate and return `self`; `compose(..., inplace=False)`
  returns a copy.
- Bind every circuit parameter before ordinary simulation. Prepared variational
  plans instead accept mappings or vectors in `plan.parameter_order`.
- Build differentiable objectives without measurements or explicit noise
  channels. Put measurements under `SamplingEngine` control.
- Negate both value and gradient when maximizing an objective with a minimizer
  such as SciPy BFGS.
- Treat `ObjectiveEvaluation.values` as `(M,)` and `.jacobian` as `(M, P)`;
  never infer axes independently of `observable_order` and `parameter_order`.
- Standard gates are singleton metadata objects. Parameterized templates require
  `.with_params(...)` unless using `QuantumCircuit` helpers.
- With LogicalQubit `transpile=True`, `physical_qubits` constrains every mapped
  wire and routing SWAP, while `initial_layout[logical] = physical` fixes the
  starting assignment. With `transpile=False`, only direct provider
  `initial_layout` pass-through is supported.
- The native simulator can raise `NativeImportError` when the C++ extension is
  unavailable.

## Minimal execution examples

```python
from quspect import QuantumCircuit, Simulator

bell = QuantumCircuit(name="bell").h(0).cx(0, 1).measure_all()
result = Simulator(seed=42).run(bell, shots=1024)
print(result.counts)
```

```python
from quspect import Parameter, QuantumCircuit

theta = Parameter("theta")
template = QuantumCircuit().rx(2 * theta + 0.1, 0)
bound = template.bind_parameters({theta: 0.5})
```

For a complete objective → gradient → external optimizer → decoded solution
workflow, use the QAOA and VQE patterns in
[references/variational.md](references/variational.md).
