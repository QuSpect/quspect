# QuSpect API Map

This reference summarizes the QuSpect architecture and public API surface
discovered from source, docs, and tests. For end-to-end VQE/QAOA and optimizer
workflows, also read [variational.md](variational.md). For physical-qubit
selection and mapping semantics, read
[hardware-routing.md](hardware-routing.md).

## Contents

1. [Project shape](#project-shape)
2. [Top-level exports](#top-level-exports)
3. [Circuit API](#circuit-api)
4. [Gates](#gates)
5. [Simulation](#simulation)
6. [Density matrix API](#density-matrix-api)
7. [Noise](#noise)
8. [DAG and optimization](#dag-and-optimization)
9. [Operators](#operators)
10. [IR: QASM, QSAL, decomposition](#ir-qasm-qsal-decomposition)
11. [Visualization](#visualization)
12. [Variational objectives](#variational-objectives)
13. [Backends and LogicalQubit](#backends-and-logicalqubit)
14. [Transpiler](#transpiler)
15. [Native bridge](#native-bridge)
16. [Test targets](#test-targets)
17. [Common pitfalls](#common-pitfalls)

## Project Shape

- Package root: `quspect/`
- Core user API: `quspect/__init__.py`
- Tests: `tests/`, organized by subpackage
- Docs: `docs/source/{quickstart,api,architecture,variational}.rst`
- Complete algorithms: `examples/qaoa_max_cut.py`, `examples/grover_long.py`
- Native backend: `cpp/` plus Python bridge `quspect/native/__init__.py`

Main architecture:

- `algorithms/`: reusable circuit builders, currently including QFT/IQFT
- `circuit/`: `QuantumCircuit`, `CircuitInstruction`, `Parameter`
- `gates/`: `Gate` metadata and fixed/parameterized gate templates
- `dag/`: circuit-to-DAG compiler, DAG representation, optimization passes
- `simulation/`: native-backed statevector simulator, density matrix simulator, Monte Carlo simulator, results
- `noise/`: Kraus channels, noise models, circuit noise insertion
- `operators/`: `QubitOperator`, `FermionOperator`, fermion-to-qubit transforms
- `ir/`: OpenQASM 2.0 export/import, QSAL JSON, decomposition
- `transpiler/`: target model, SABRE layout/routing, decomposition/optimization pipeline
- `backends/`: abstract backend interfaces and LogicalQubit provider/backend/job
- `variational/`: observable objectives, exact adjoint plans, sampled
  parameter-shift plans, structured results, and recoverable objective jobs
- `optimization/`: symbolic binary programs, BQM/QUBO and Ising intermediate
  representations, dense hardware-matrix lowering, CSV encoding, and native
  Ising execution
- `visualization/`: text circuit drawings, SVG circuit rendering, sampling charts

## Top-Level Exports

From `quspect`:

- Circuit: `QuantumCircuit`, `CircuitInstruction`, `Parameter`,
  `ParameterExpression`, `ParamValue`
- Gates: `Gate`
- Operators: `QubitOperator`, `FermionOperator`
- DAG/optimization: `PassManager`, `GateCancellation`, `CommutationAnalysis`, `RotationMerging`
- Simulation: `Simulator`, `SimulationResult`
- Variational: `ObservableObjective`, `AdjointEngine`, `PreparedObjective`,
  `SamplingEngine`, `PreparedSamplingObjective`, `ObjectiveEvaluation`,
  `ObjectiveSweep`, `ObjectiveJob`, `ObjectiveSweepJob`
- Backends: `Backend`, `CloudBackend`, `CloudJob`, `CompositeJob`,
  `ExecutionJob`, `JobStatus`, `CircuitExperiment`, `ExecutionRequest`,
  `ExecutionBatch`, `JobReference`, `BackendCapabilities`,
  `PartialSubmissionError`

Top-level exports are intentionally smaller than the full package. Use subpackage imports for noise, visualization, IR, transpiler, density simulation, and LogicalQubit.

Import the QFT circuit builder from its algorithms subpackage:

```python
from quspect.algorithms import build_qft_circuit
```

`build_qft_circuit` is intentionally not exported from the top-level
`quspect` package. QFT follows the little-endian convention (`q0` is the
least-significant bit); with `include_swaps=False`, its output is bit-reversed.

## Circuit API

Import:

```python
from quspect import Parameter, ParameterExpression, QuantumCircuit
```

Constructor and properties:

- `QuantumCircuit(name="")`
- `.name`
- `.instructions`
- `.num_qubits`, `.num_clbits`
- `.active_qubits`, `.active_clbits`
- `len(circuit)`, iteration, indexing

Gate builders:

- Single qubit: `.id(q)`, `.h(q)`, `.x(q)`, `.y(q)`, `.z(q)`, `.s(q)`, `.sdg(q)`, `.t(q)`, `.tdg(q)`
- Parameterized one-qubit: `.rx(theta, q)`, `.ry(theta, q)`, `.rz(theta, q)`,
  `.p(theta, q)`, `.u1(lam, q)`, `.u2(phi, lam, q)`,
  `.u3(theta, phi, lam, q)`
- Two-qubit: `.cx(control, target)`, `.cz(control, target)`, `.swap(a, b)`
- Controlled rotations: `.crx(theta, control, target)`, `.cry(...)`, `.crz(...)`
- Controlled phase/universal: `.cp(theta, control, target)`,
  `.cu3(theta, phi, lam, control, target)`
- Three-qubit: `.ccx(control1, control2, target)`
- Generic: `.append(gate, qubits, clbits=())`

Measurement and structure:

- `.measure(qubits, clbits)`
- `.measure_all()`
- `.barrier(*qubits)`
- `.compose(other, qubits, clbits=None, inplace=False)`
- `.copy(name=None)`
- `.dagger()`

Noise channel helpers on circuits:

- `.append_channel(channel, qubits)`
- `.bit_flip(p, q)`
- `.phase_flip(p, q)`
- `.bit_phase_flip(p, q)`
- `.depolarize(p, q)`
- `.pauli_error(px, py, pz, q)`
- `.amplitude_damp(gamma, q)`
- `.phase_damp(gamma, q)`
- `.generalized_amplitude_damp(gamma, p_thermal, q)`

Parameters and serialization:

- `.parameters`
- `.bind_parameters({Parameter(...): float})`
- `.to_dag()`
- `.to_json()`, `QuantumCircuit.from_json(data)`
- `.draw()`
- `.svg(style="official", width=None, scale=1.0)`
- README mentions `.to_qasm()` and `.to_qsal()`; if using these, verify current source because the canonical implementations are in `quspect.ir`.

Important behavior:

- Qubit/classical bit counts are inferred from max referenced index plus one.
- Gaps are allowed, e.g. using qubit 3 implies `num_qubits == 4`.
- `measure_all()` only measures currently active qubits.
- `compose` requires `len(qubits) == other.num_qubits`.
- Gate qubit and classical-bit indices must be non-negative integers; repeated
  operands are rejected for gates that require distinct qubits.
- `ParameterExpression` is immutable and affine. Products or quotients that
  would produce nonlinear symbolic expressions raise `TypeError`.
- Parameterized gates must be bound before ordinary `Simulator.run`; prepared
  variational plans accept parameter points directly.

## Gates

Import:

```python
from quspect.gates.standard import H, X, CX
from quspect.gates.parameterized import RX, RY, RZ, U3
from quspect.gates import Gate, gate_matrix, ALL_STANDARD, ALL_PARAMETERIZED
```

Standard singleton gates:

- `I`, `H`, `X`, `Y`, `Z`, `S`, `SDG`, `T`, `TDG`
- `CX`, `CZ`, `SWAP`
- `CCX`
- `ALL_STANDARD`

Parameterized templates:

- `RX`, `RY`, `RZ`, `P`, `U1`, `U2`, `U3`
- `CRX`, `CRY`, `CRZ`, `CP`, `CU3`
- `ALL_PARAMETERIZED`

`Gate` is metadata: name, qubit count, label, params. Create concrete parameterized gates with `RX.with_params(0.5)`. Matrix computation lives in `quspect.gates._matrix_utils.gate_matrix`.

## Simulation

Import:

```python
from quspect.simulation import Simulator, SimulationResult
```

`Simulator(seed=None)` implements `Backend`.

- `.run(circuit, shots=1024, save_statevector=False, **options) -> SimulationResult`
- `.statevector(circuit) -> np.ndarray`
- `.name == "simulator"`

Execution behavior:

- Empty circuit returns no counts and zero shots.
- No-measurement circuit evolves once and reports `num_shots == 0`; use `save_statevector=True` or `.statevector`.
- Terminal measurements evolve once, then sample.
- Mid-circuit measurements re-execute per shot with collapse.
- Native executor is used via QSAL; unbound symbolic parameters raise `ValueError`.

`SimulationResult`:

- Attributes: `counts`, `probabilities`, `num_shots`, `num_qubits`, `num_clbits`, `statevector`
- Methods: `.most_frequent()`, `.top_outcomes(n=5)`, `.get_count(bitstring)`, `.get_probability(bitstring)`, `.display(...)`
- Constructors: `SimulationResult.from_native(...)`, `SimulationResult.from_statevector(...)`

Additional simulation exports:

- `DensityMatrix`
- `DensityMatrixSimulator`
- `MonteCarloSimulator`
- `apply_gate`, `apply_gate_out_of_place`

## Density Matrix API

Import:

```python
from quspect.simulation import DensityMatrix, DensityMatrixSimulator
```

`DensityMatrix` constructors and methods:

- `DensityMatrix(num_qubits)`
- `DensityMatrix.from_array(data)`
- `DensityMatrix.from_pure_state(statevector)`
- `DensityMatrix.from_labels(label_probs)`
- `DensityMatrix.maximally_mixed(num_qubits)`
- `.trace()`, `.normalize()`, `.purity()`, `.subsystem_purity(qubits)`
- `.partial_trace(qubits)`, `.fidelity(other)`
- `.apply_gate(matrix, qubits)`, `.apply_channel(kraus_ops, qubits)`
- `.probabilities(qubits)`, `.measure(qubits, rng)`, `.copy()`

`DensityMatrixSimulator(seed=None)`:

- `.run(circuit, shots=1024, save_statevector=False, **options)`
- `.density_matrix(circuit)`

`MonteCarloSimulator(seed=None)`:

- `.run(circuit, shots=1024, save_statevector=False, **options)`

## Noise

Import:

```python
from quspect.noise import (
    NoiseModel, NoiseRule, DepolarizingChannel,
    BitFlipChannel, PhaseFlipChannel, AmplitudeDampingChannel,
)
```

Channel classes:

- `NoiseChannel`
- `PauliChannel`
- `BitFlipChannel`, `PhaseFlipChannel`, `BitPhaseFlipChannel`
- `DepolarizingChannel`
- `AmplitudeDampingChannel`, `PhaseDampingChannel`
- `GeneralizedAmplitudeDampingChannel`
- `KrausChannel`

Channels expose `.kraus_operators()` and `.is_unitary()`.

`NoiseModel(name="...")`:

- `.add_rule(rule)`
- `.add_all_qubit_gate_error(channel, gate_names=None, gate_num_qubits=None, before_gate=False)`
- `.add_qubit_gate_error(channel, qubits, gate_names=None, gate_num_qubits=None, before_gate=False)`
- `.add_readout_error(p0_given_1, p1_given_0, qubits=None)`
- `.add_thermal_relaxation(t1, t2, gate_time=..., excited_state_population=...)`
- `.apply(circuit)`

Convenience constructors:

- `ideal()`
- `depolarizing_model(p_single, p_double)`
- `thermal_relaxation_model(t1, t2, gate_time_1q, gate_time_2q)`

## DAG and Optimization

Import:

```python
from quspect.dag import CircuitCompiler, DAGCircuit, DAGToCircuitCompiler
from quspect.dag.passes import PassManager, GateCancellation
```

`DAGCircuit`:

- `.num_qubits`, `.num_clbits`
- `.nodes`, `.edges`, `.num_nodes()`, `.num_edges()`
- `.add_node(...)`, `.add_edge(...)`, `.remove_node(...)`
- `.predecessors(node_id)`, `.successors(node_id)`
- `.input_nodes()`, `.output_nodes()`, `.operation_nodes()`
- `.topological_order()`
- `.layers()`
- `.critical_path_length()`
- `.nodes_on_wire(kind, index)`, `.qubit_wires()`, `.clbit_wires()`
- `.summary()`

Compilers:

- `CircuitCompiler.compile(circuit)`
- `DAGToCircuitCompiler.compile(dag)`
- `QuantumCircuit.to_dag()` is the user-facing shortcut.

Passes:

- `BasePass`
- `PassManager(passes).run(dag, max_iterations=...)`
- `PassStats`
- `GateCancellation`
- `CommutationAnalysis`
- `RotationMerging`
- `SingleQubitGateFusion`
- `EliminateDiagonalBeforeMeasurement`

Phase polynomial tools:

- `PhaseTerm`
- `ParityMatrix`
- `PhasePolynomial`
- `PhasePolynomialCompiler`

## Operators

Import:

```python
from quspect import QubitOperator, FermionOperator
from quspect.operators.transforms import jordan_wigner
```

`QubitOperator` represents weighted Pauli strings:

- Constructor: `QubitOperator({((0, "X"), (1, "Z")): 0.5})`
- Constructors: `.zero()`, `.identity(coeff=1.0)`, `.from_pauli_string("XZ", coeff=...)`
- Properties: `.terms`, `.is_zero`, `.is_identity`, `.qubit_count`, `.parameters`
- Methods: `.simplify()`, `.compress(abs_tol=...)`, `.bind_parameters(binding)`
- Operators: immutable arithmetic is implemented; inspect tests for exact supported combinations.

`FermionOperator` represents ladder operators `(mode, action)` where `1` is creation and `0` is annihilation.

- Constructors: `.zero()`, `.identity(...)`, `.number_operator(mode, coefficient)`, `.hopping_term(i, j, coefficient)`, `.coulomb_term(i, j, coefficient)`
- Properties: `.terms`, `.is_zero`, `.is_identity`, `.mode_count`, `.parameters`
- Methods: `.simplify()`, `.compress(...)`, `.normal_order()`, `.bind_parameters(binding)`

Transforms:

- `jordan_wigner(fermion_op)`
- `bravyi_kitaev(fermion_op)` exists but verify implementation before relying on it.
- `reverse_jordan_wigner(qubit_op)`

## IR: QASM, QSAL, Decomposition

Import:

```python
from quspect.ir import QASM2Exporter, QASM2ExportConfig, QASM2ExportMode
from quspect.ir import qasm2_loads, dumps, loads, dump, load
```

OpenQASM:

- `QASM2Exporter(config=None).export(circuit)`
- `.export_to_file(circuit, path)`
- `QASM2ExportConfig(...)`
- `QASM2ExportMode`
- `get_supported_gates()`
- `QASM2Importer().imports(source)`
- `qasm2_loads(source)`

QSAL:

- `dumps(circuit, observables=None, name=None, **kwargs)`
- `loads(data)`
- `dump(circuit, path, observables=None, **kwargs)`
- `load(path)`
- `QSALDocument.to_json(indent=...)`
- `QSALDocument.from_json(data)`, `.to_file(path)`, `.from_file(path)`
- `.has_observables()`, `.is_parameterized()`

Decomposition:

- `GateDecompositionRule`
- `register_decomposition(...)`
- `clear_extra_rules()`
- `get_rules_for(gate_name)`
- `decompose_gate(...)`
- `decompose_circuit(circuit, supported_gates=...)`
- `decompose_deep(circuit, supported_gates=...)`

Errors:

- `IRSerializationError`, `IRDeserializationError`
- `ParameterizedGateError`
- `UnsupportedGateError`
- `QSALVersionError`

## Visualization

Import:

```python
from quspect.visualization import draw_text_circuit, draw_svg_circuit, SvgStyle
from quspect.visualization import display_sampling, compare_results
```

Circuit diagrams:

- `QuantumCircuit.draw()`
- `draw_text_circuit(circuit)`
- `QuantumCircuit.svg(style="official", width=None, scale=1.0)`
- `draw_svg_circuit(circuit, style="official" | "light" | "dark" | SvgStyle, width=None, scale=1.0)`
- `CircuitSVG.to_file(filename)`
- `SvgStyle(...)`, `.as_dict()`, `.from_mapping(mapping)`

Sampling charts:

- `SimulationResult.display(style="hbar", sort_mode="count_desc", max_outcomes=16, theme="official", svg_width=680)`
- `display_sampling(result, ...)`
- `compare_results(result_a, result_b, label_a="A", label_b="B", ...)`
- `SamplingDisplay.to_file(filename)`, `.stats()`
- `ComparisonDisplay.to_file(filename)`
- Styles include `hbar`, `vbar`, `dot`, `svg`.
- Sort modes include `count_desc`, `count_asc`, `bitstring`, `probability`.

## Variational Objectives

Import:

```python
from quspect import (
    AdjointEngine, ObservableObjective, SamplingEngine,
    ObjectiveEvaluation, ObjectiveSweep,
)
```

Problem definition:

- `ObservableObjective(circuit, observables)` snapshots one parameterized state
  preparation and a non-empty named mapping of `QubitOperator` observables.
- Mapping order defines `observable_order`.

Exact strategy:

- `AdjointEngine(dtype="float32" | "float64").compile(objective)` returns a
  reusable `PreparedObjective`.
- `PreparedObjective.evaluate(point)` returns values and a full Jacobian.
- `.evaluate_many(points)` returns an `ObjectiveSweep`.
- `.parameter_order`, `.observable_order`, `.dtype`, `.metadata` define axes and
  execution details.

Sampled strategy:

- `SamplingEngine(backend, shots=4096, grouping="qwc" | "none",
  execution_options=None).compile(objective)` returns a
  `PreparedSamplingObjective`.
- `.evaluate(point, timeout=None)` and `.evaluate_many(points, timeout=None)` are
  blocking shortcuts.
- `.submit(point)`/`.submit_many(points)` return `ObjectiveJob` or
  `ObjectiveSweepJob`.
- `.attach(job)`/`.attach_many(job)` recover submitted workloads.
- `.num_measurement_groups`, `.num_experiments_per_point`, and `.workload_id`
  expose the compiled sampling plan.

Results:

- `ObjectiveEvaluation`: `values (M,)`, `jacobian (M,P)`, `value(...)`,
  `derivative(...)`, axis orders, and metadata.
- `ObjectiveSweep`: `values (B,M)`, `jacobian (B,M,P)`, `.at(index)`.

Read [variational.md](variational.md) for complete VQE/QAOA, SciPy BFGS,
finite-shot, hardware, sweep, recovery, limitation, and validation workflows.

## Backends and LogicalQubit

Import:

```python
from quspect.backends import (
    Backend, BackendCapabilities, CircuitExperiment, CloudBackend,
    CompositeJob, ExecutionBatch, ExecutionJob, ExecutionRequest,
    JobReference, JobStatus,
)
from quspect.backends.logicalqubit import LogicalQubitProvider
```

Requests and batches:

- `CircuitExperiment(circuit, bindings={}, label=None, metadata={})` snapshots a
  circuit template and one complete parameter binding.
- `ExecutionRequest.single(...)`, `.batch(...)`, and `.sweep(...)` build ordered,
  labelled workloads with `shots`, `result_format`, options, and metadata.
- `ExecutionRequest.labels`, `.experiment_shapes`, `.bound_circuits()`.
- `ExecutionBatch` is an ordered result sequence with `.single()`,
  `.by_label(label)`, `.success`, `.failed_indices`, and `.get_counts()`.

Backend interfaces:

- `Backend.run(circuit, shots=1024, save_statevector=False, **options)`
- `Backend.execute(request) -> ExecutionBatch`
- `Backend.submit_request(request) -> ExecutionJob`
- `Backend.name`, `.capabilities`
- `CloudBackend.submit(circuit, shots=1024, **options)`
- `CloudBackend.submit_request(request)`
- `CloudBackend.platform_name`
- `CloudBackend.num_qubits`
- `CloudBackend.native_gate_set`
- `ExecutionJob.result(timeout=None)`, `.status()`, `.cancel()`, `.done()`
- `ExecutionJob.job_id`, `.job_ids`, `.reference`
- `CompositeJob` combines partitioned child jobs while preserving label order.
- `JobStatus`: status enum for queued/running/completed/etc.

Recovery and capabilities:

- `JobReference.to_dict()`/`.from_dict(...)` serialize a credential-free
  manifest containing platform, backend, IDs, labels, shapes, shots, result
  format, and optional workload fingerprint.
- `BackendCapabilities` reports sampling, batching, estimator, state-vector,
  recovery, and cancellation support together with circuit/shot limits,
  execution modes, result formats, and measurement types.

LogicalQubit:

- `LogicalQubitProvider(api_key=None, url=None)`
- `.url`
- `.list_backends()`
- `.get_backend(name)`
- `.retrieve_job(reference)`
- `LogicalQubitBackend.run(circuit, shots=1024, transpile=True,
  initial_layout=None, physical_qubits=None, timeout=None, poll_interval=2.0,
  **options)`
- `.submit(circuit, shots=1024, transpile=True, initial_layout=None,
  physical_qubits=None, **options)`
- `.submit_request(request)` with provider batching and automatic partitioning
- `.name`, `.platform_name`, `.num_qubits`, `.native_gate_set`, `.config`

Operational notes:

- Requires optional `lqcloud` dependency and credentials, usually `LQCLOUD_API_KEY`.
- `run(..., transpile=True)` decomposes to native gates, optimizes, then maps
  and routes before submission.
- `physical_qubits` constrains layout and routing to the induced connected
  device subgraph. `initial_layout[logical] = physical` fixes the starting
  mapping. Both options are shared across provider-native batches.
- Use `transpile=False` only when the circuit already matches chip constraints.
  In that mode `initial_layout` passes directly to the provider and
  `physical_qubits` is invalid.
- Native provider batches larger than backend limits are represented as a
  recoverable `CompositeJob`.

Read [hardware-routing.md](hardware-routing.md) for validation rules, workspace
wires, mapping domains, result-order evidence, and debugging tests.

Backend exceptions:

- `BackendError`
- `AuthenticationError`
- `TranslationError`
- `JobError`
- `JobTimeoutError`
- `BackendNotFoundError`
- `OptionalDependencyError`
- `PartialSubmissionError`

## Transpiler

Import:

```python
from quspect.transpiler import CompilationTarget, SABRE
```

Public exports:

- `CompilationTarget.from_backend(backend)`
- `CompilationTarget.from_coupling_map(num_qubits, coupling_map, native_gates=None)`
- `CompilationTarget.induced_subgraph(physical_qubits)` returns
  `(dense_target, dense_to_physical)`
- `SABRE(target).solve(circuit, iter_num=3, seed=None,
  initial_mapping=None)` returns
  `(routed_circuit, initial_mapping, final_mapping)`

Internal-but-useful classes in `quspect.transpiler._passes`:

- `DecomposePass(target).run(circuit)`
- `LayoutRoutingPass(target, iter_num=3, seed=None,
  initial_mapping=None).run(circuit)` returns
  `(routed_circuit, initial_mapping, final_mapping)`
- `OptimizePass().run(circuit)`
- `TranspilerPipeline(passes).run(circuit)`

## Native Bridge

Import:

```python
from quspect.native import NativeExecutor, NativeImportError
```

API:

- `NativeExecutor(precision="double", seed=0)`
- `.run(qsal_json, shots=0, return_statevector=False, return_expectations=False, return_samples=False)`
- `.decode(qsal_json)`
- `.run_program(program, shots=0, return_statevector=False, return_expectations=False, return_samples=False)`
- Internal `_prepare_adjoint(qsal_json)` creates the reusable native plan used by
  public `AdjointEngine`; prefer the public variational API in application code.
- `ExecutionResult`: `statevector_f64`, `statevector_f32`, `packed_samples`, `counts`, `expectations_real`, `metadata`
- helpers: `decode_qsal`, `validate`, `capabilities`

If the native extension (`quspect._core_scalar`, or `quspect._core_avx2` on AVX2 + FMA CPUs, selected by `quspect._native_loader`) cannot be imported, the bridge raises `NativeImportError` and instructs building the C++ extension.

## Test Targets

Use these tests for focused validation:

- Circuit: `tests/circuit/test_circuit.py`, `test_instruction.py`, `test_parameter.py`
- Gates: `tests/gates/`
- DAG: `tests/dag/`
- IR/QASM/QSAL: `tests/ir/`, `tests/test_qasm_importer.py`
- Simulation: `tests/simulation/`, `tests/native/`
- Noise: `tests/noise/`
- Operators: `tests/operators/`
- Transpiler: `tests/transpiler/test_sabre.py`
- Visualization: `tests/visualization/`
- Cloud backend: `tests/backends/test_logicalqubit.py`
- Provider-neutral execution: `tests/backends/test_execution.py`
- LogicalQubit batching/recovery: `tests/backends/test_logicalqubit_backend.py`,
  `test_logicalqubit_job.py`, `test_logicalqubit_provider.py`,
  `test_logicalqubit_translator.py`
- Variational sampling: `tests/variational/`
- Exact/adjoint gradients: `tests/native/test_gradient.py`,
  `tests/simulation/test_gradient.py`
- End-to-end: `tests/test_integration.py`

## Common Pitfalls

- Do not instantiate `QuantumCircuit` with a qubit count; it only accepts keyword `name`.
- Do not assume top-level `quspect` exports every subpackage API.
- Do not pass unbound `Parameter` objects to `Simulator.run`.
- Do not add measurement or explicit noise operations to an
  `ObservableObjective` state-preparation circuit.
- Do not assume optimizer vectors follow name sorting; read
  `plan.parameter_order`.
- Do not negate only the value when maximizing through a minimizing optimizer;
  negate the gradient too.
- Do not demand exact agreement from finite-shot values or parameter-shift
  gradients.
- Do not rely on cloud backend behavior without checking optional dependencies and credentials.
- Do not treat a preferred initial layout as a physical-qubit constraint. Use
  `physical_qubits` to restrict the complete mapping/routing search space.
- Do not pass `physical_qubits` with `transpile=False`, and do not call the
  private backend `_transpile` method from application code.
- Do not confuse SABRE's final mapping, the static provider initial layout,
  routed circuit-wire indices, and classical result-bit order.
- Do not persist credentials with `JobReference`; it intentionally contains only
  recovery metadata and is not an authenticity signature.
- Do not treat `Gate` as a matrix container; call `gate_matrix` when a numeric matrix is needed.
- Check current source before using README-only convenience methods if they are not visible in `circuit/circuit.py`.
