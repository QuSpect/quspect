# QuSpect

*A quantum computing SDK for research and education.*

[![PyPI](https://img.shields.io/pypi/v/quspect)](https://pypi.org/project/quspect/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.15-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/QuSpect/quspect/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/quspect/badge/?version=latest)](https://quspect.readthedocs.io/en/latest/)

> 🇨🇳 中文用户请参阅 [README_CN.md](README_CN.md)

---

QuSpect provides an intuitive, fluent Python API for constructing quantum circuits,
visualizing them as ASCII diagrams or scalable SVG, and simulating state-vector
evolution with shot-based sampling. Simulation and exact adjoint gradients are backed
by the C++ high-performance engine.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Build a Circuit](#build-a-circuit)
  - [Visualize](#visualize)
  - [Simulate](#simulate)
  - [Parameterized Circuits](#parameterized-circuits)
  - [Variational Objectives and Gradients](#variational-objectives-and-gradients)
  - [Circuit Composition](#circuit-composition)
  - [Algorithm Examples](#algorithm-examples)
  - [Pulse Envelopes](#pulse-envelopes)
- [Core Modules](#core-modules)
  - [QuantumCircuit](#quantumcircuit)
  - [Gates](#gates)
  - [DAG IR & Optimization](#dag-ir--optimization)
  - [Operators](#operators)
  - [IR Export](#ir-export)
- [Visualization](#visualization)
  - [Terminal Diagrams](#terminal-diagrams)
  - [SVG Rendering](#svg-rendering)
  - [Sampling Charts](#sampling-charts)
- [Simulation](#simulation)
  - [Backend-neutral Parameter Sweeps](#backend-neutral-parameter-sweeps)
- [Cloud QPU](#cloud-qpu)
  - [Setup](#cloud-setup)
  - [Run on Real Hardware](#run-on-real-hardware)
  - [Compilation and Physical Layout](#compilation-and-physical-layout)
  - [Async Submission](#async-submission)
  - [Batch Sampling and VQE](#batch-sampling-and-vqe)
  - [Simulator vs QPU](#simulator-vs-qpu)
- [C++ Native Backend](#c-native-backend)
- [Documentation](#documentation)
- [Development](#development)

---

## Features

- **Fluent circuit API** — build circuits with method chaining; no pre-declared qubit counts
- **24 built-in gate templates** — standard gates (H, X, CX, CCX, SWAP…) and parameterized rotations (RX, RY, RZ, U3…), plus measurements and barriers
- **Symbolic parameters** — define variational circuits with named `Parameter` placeholders and bind to values later
- **DAG compiler** — compile circuits into a DAG IR with topological ordering, layer partitioning, and critical path analysis
- **Optimization passes** — `GateCancellation`, `CommutationAnalysis`, `RotationMerging`, `SingleQubitGateFusion`, `EliminateDiagonalBeforeMeasurement`
- **Operator algebra** — `QubitOperator` (Pauli strings) and `FermionOperator` (ladder operators) with Jordan-Wigner transforms
- **State-vector simulation** — native C++ evolution, projective measurement, and shot-based sampling
- **Variational objectives** — exact C++ adjoint gradients and shot-based parameter-shift gradients through one structured API
- **Recoverable batch execution** — labelled circuit batches, flattened parameter sweeps, and credential-free job references
- **Visualization** — terminal circuit diagrams, SVG circuit rendering (3 themes), sampling result charts (hbar, vbar, dot, SVG)
- **IR export** — OpenQASM 2.0 and QSAL (native JSON) export
- **Cloud QPU** — unified `Backend` interface for local simulators and real quantum hardware
- **iSing optimization** — QUBO/Ising model conversion, recoverable cloud submission,
  and objective decoding with `quspect[isingq]`; see [iSing cloud optimization](docs/source/isingq.rst).
- **Transpiler** — native-gate decomposition, topology-safe DAG optimization,
  constrained physical-qubit mapping, and SABRE SWAP routing
- **Pulse envelopes** — `quspect.pulse` builds and decodes superconducting-qubit control envelopes
  (DRAG, flat-top, custom waveforms, sideband mixing, schedules); a C++ rebuild of envelopes-qc 0.1.5
  with bit-identical output; see [Pulse envelopes](docs/source/pulse.rst)
- **C++ backend** — high-performance simulation and gradient execution with AVX2 SIMD acceleration and a pybind11 bridge

---

## Installation

```bash
python -m pip install quspect
```

Pre-built binary wheels include the C++ simulation engine (with OpenMP), so
`Simulator`, `NativeExecutor`, and exact adjoint gradients work right after
`pip install` — no compiler needed.

| Platform | Architecture | Python | Wheel |
|----------|--------------|--------|-------|
| Linux (glibc ≥ 2.28, e.g. Ubuntu 20.04+, RHEL/Rocky 8+, Debian 10+) | x86_64 | 3.10–3.15 | `manylinux_2_28_x86_64` |
| Linux (glibc ≥ 2.28, e.g. Ubuntu 20.04+, RHEL/Rocky 8+, Debian 10+) | aarch64 (arm64) | 3.10–3.15 | `manylinux_2_28_aarch64` |
| Windows 10/11 | x86_64 (64-bit Python) | 3.10–3.15 | `win_amd64` |
| macOS 14 (Sonoma) or newer | Apple Silicon (arm64) | 3.10–3.15 | `macosx_14_0_arm64` |

- **Windows:** 64-bit CPython 3.10–3.15 needs nothing extra — the MSVC C++ and
  OpenMP runtimes are bundled in the wheel.
- **Other platforms** (musl/Alpine, 32-bit Windows, Intel Macs, macOS < 14,
  PyPy, …) install from the source distribution, which compiles the
  C++ extension. This needs CMake ≥ 3.22 and a C++17 compiler — Visual Studio
  Build Tools on Windows, Xcode Command Line Tools (+ `brew install libomp` for
  OpenMP) on macOS, GCC or Clang on Linux — and network access to fetch
  nlohmann/json.

Runtime knobs:

- On x86_64 the AVX2 + FMA kernels are selected automatically when the CPU
  supports them; set `QUSPECT_DISABLE_AVX2=1` to force the portable scalar
  kernels (read once, at the first native call).
- Parallel kernels use OpenMP; set `OMP_NUM_THREADS` to limit the number of
  threads.

```bash
python -c "from quspect.native import capabilities; print(capabilities())"
```

For development (includes linters, type checker, test runner, and docs tools):

```bash
git clone https://github.com/QuSpect/quspect.git
cd quspect
python -m pip install -e ".[dev]"
python -m pip install pybind11
```

The editable install is pure Python and does not compile the C++ extension;
build it as described below.

### C++ Native Backend

In a source checkout, build the extension before using `Simulator`,
`DensityMatrixSimulator`, `MonteCarloSimulator`, `NativeExecutor`, or
`AdjointEngine`:

```bash
cmake -S cpp -B cpp/build -G Ninja \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
python -c "from quspect.native import capabilities; print(capabilities())"
```

Requirements: C++17 compiler, CMake ≥ 3.22, Ninja, pybind11, and optionally an
AVX2-capable CPU. The build writes the extension into the local `quspect/`
package, so the editable install above can import it directly. The first
configure requires network access to fetch nlohmann/json and, when tests are
enabled, GoogleTest. Pass `-DQSPECT_BUILD_AVX2=OFF` for a scalar-only build.

Maintainers: see [docs/source/releasing.rst](https://github.com/QuSpect/quspect/blob/main/docs/source/releasing.rst) for how
wheels are built and published to PyPI.

---

## Quick Start

### Build a Circuit

```python
from quspect import QuantumCircuit

# Create a Bell state (entanglement)
circ = QuantumCircuit(name="bell")
circ.h(0)           # Hadamard on qubit 0
circ.cx(0, 1)       # CNOT with control=0, target=1
circ.measure_all()  # measure all qubits

print(circ.num_qubits)   # 2 — dynamically inferred
print(circ.num_clbits)   # 2
print(len(circ))         # 3 instructions
```

### Visualize

```python
# Terminal ASCII diagram
print(circ.draw())
```

```python
# SVG rendering (inline in Jupyter or save to file)
svg = circ.svg(style="dark")
svg.to_file("bell_circuit.svg")
```

The Bell circuit rendered by `circ.svg()` (`"light"` / `"dark"` themes, following your GitHub color scheme):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/QuSpect/quspect/main/docs/source/_static/readme/bell_circuit_dark.svg">
  <img src="https://raw.githubusercontent.com/QuSpect/quspect/main/docs/source/_static/readme/bell_circuit_light.svg" alt="Bell circuit: H on q0, CNOT from q0 to q1, then both qubits measured" width="342">
</picture>

### Simulate

```python
from quspect.simulation import Simulator

sim = Simulator(seed=42)
result = sim.run(circ, shots=1024)

print(result.counts)           # approximately {'00': 512, '11': 512}
print(result.most_frequent())  # one of the two Bell-state outcomes

# Statistical summary and bar chart
result.display(style="hbar")
```

### Parameterized Circuits

```python
from quspect import QuantumCircuit, Parameter

theta, phi = Parameter("θ"), Parameter("φ")

circ = QuantumCircuit(name="variational")
circ.rx(theta, 0)
circ.ry(phi, 1)
circ.cx(0, 1)

# Bind to concrete values before simulation
bound = circ.bind_parameters({theta: 0.5, phi: 1.2})
```

### Variational Objectives and Gradients

Define a measurement-free ansatz and named observables once, then compile the
same objective for exact adjoint differentiation or hardware-shaped sampling:

```python
from quspect import (
    AdjointEngine,
    ObservableObjective,
    Parameter,
    QuantumCircuit,
    QubitOperator,
    SamplingEngine,
    Simulator,
)

theta = Parameter("theta")
ansatz = QuantumCircuit(name="one_qubit_vqe").ry(theta, 0)
hamiltonian = QubitOperator({((0, "Z"),): 1.0})
objective = ObservableObjective(ansatz, {"energy": hamiltonian})

# Exact noiseless value and gradient from the native adjoint engine.
exact_plan = AdjointEngine(dtype="float64").compile(objective)
exact = exact_plan.evaluate({theta: 0.5})
print(exact.value("energy"), exact.derivative("energy"))
print(exact_plan.parameter_order)  # authoritative optimizer-vector order

# The same objective through grouped measurements and parameter shifts.
sampled_plan = SamplingEngine(
    Simulator(seed=7),
    shots=4096,
    grouping="qwc",
).compile(objective)
sampled = sampled_plan.evaluate({theta: 0.5})
print(sampled.value("energy"), sampled.derivative("energy"))
print(sampled_plan.num_measurement_groups)
print(sampled_plan.num_experiments_per_point)

# Batch optimizer points without recompiling; result axes stay explicit.
sweep = exact_plan.evaluate_many(
    [{theta: -0.2}, {theta: 0.0}, {theta: 0.2}]
)
print(sweep.values.shape)    # (3, 1)
print(sweep.jacobian.shape)  # (3, 1, 1)
```

Finite-shot values and gradients fluctuate statistically; compare them against
the adjoint result rather than expecting exact equality. See
[Variational Objectives](docs/source/variational.rst) for parameter sweeps,
optimizer integration, asynchronous submission, and job recovery.

### Circuit Composition

```python
# Reusable sub-circuit
bell_pair = QuantumCircuit(name="bell")
bell_pair.h(0).cx(0, 1)

# Compose into a larger circuit
main = QuantumCircuit(name="composed")
main = main.compose(bell_pair, qubits=[0, 2])  # map sub's 0,1 → main's 0,2
main = main.compose(bell_pair, qubits=[1, 3])  # map sub's 0,1 → main's 1,3
```

### Algorithm Examples

The repository includes complete, classically checkable examples:

- [`examples/qaoa_max_cut.py`](examples/qaoa_max_cut.py) — optimize a four-node
  Max-Cut QAOA ansatz with exact adjoint gradients and SciPy BFGS.
- [`examples/grover_long.py`](examples/grover_long.py) — construct and simulate
  Grover–Long exact search, including matched phases and clean work qubits.
- [`examples/portfolio_selection.py`](examples/portfolio_selection.py) — select
  a fixed-size mean-variance portfolio through the full quadratic-to-Ising
  pipeline.
- [`examples/wireless_channel_assignment.py`](examples/wireless_channel_assignment.py)
  — assign wireless channels with one-hot and pairwise interference
  constraints.

### Pulse Envelopes

`quspect.pulse` composes control envelopes and samples them on a fixed time grid
(requires the native extension, see [C++ Native Backend](#c-native-backend)):

```python
import numpy as np
from quspect.pulse import GaussianDRAG, WaveCache, align, decode_envelope

pi_pulse = GaussianDRAG(t0=15, w=10, amp=1.0, coef=0.5, df=0.1, phase=0.0)
xy = align(pi_pulse, np.array([0.0, 50.0, 100.0]), np.array([1.0, 0.5, 1.0], dtype=np.float32))
t_start, wave = decode_envelope(xy, WaveCache(0.5), -50.0, 250.0)  # 0.5 ns grid
print(t_start, wave.dtype, wave.shape)  # -50.0 complex64 (600,)
```

The API matches envelopes-qc 0.1.5 (`envelopes.evc`) and decoded waveforms are
bit-identical; see [Pulse envelopes](docs/source/pulse.rst) for all envelope types.
Code written against envelopes-qc (such as `lqcloud`'s pulse sequences) can run on
it via `quspect.pulse.use_as_envelopes()` / `install_as_envelopes()`.

---

## Core Modules

### `QuantumCircuit`

The central class for building quantum programs. Qubit and classical bit counts are **dynamically inferred** from the indices you use — no need to pre-declare circuit size.

| Method | Description |
|--------|-------------|
| `.h(q)` `.x(q)` `.y(q)` `.z(q)` | Single-qubit gates |
| `.s(q)` `.sdg(q)` `.t(q)` `.tdg(q)` | Phase / rotation gates |
| `.rx(θ,q)` `.ry(θ,q)` `.rz(θ,q)` `.p(θ,q)` | Parameterized rotations |
| `.cx(c,t)` `.cz(c,t)` `.swap(a,b)` | Two-qubit gates |
| `.ccx(c1,c2,t)` | Toffoli (controlled-controlled-X) |
| `.measure(qubits, clbits)` | Projective measurement |
| `.measure_all()` | Measure all qubits |
| `.barrier(*qubits)` | Insert a barrier |
| `.compose(other, qubits, clbits, inplace=False)` | Compose another circuit, returning a copy by default |
| `.bind_parameters({p: v})` | Bind symbolic parameters |
| `.to_dag()` | Compile to DAG IR |
| `.draw()` | ASCII circuit diagram |
| `.svg(style, width, scale)` | SVG circuit diagram |

### Gates

Gates are lightweight metadata objects — they store name, qubit count, and parameter info but **not** unitary matrices. Import the public helper with `from quspect.gates import gate_matrix` when a matrix is needed.

```python
from quspect.gates.standard import H, CX, CCX, SWAP
from quspect.gates.parameterized import RX, RY, RZ, U3

# Standard gates are module-level singletons
circ = QuantumCircuit()
circ.append(H, [0])
circ.append(CX, [0, 1])
circ.append(CCX, [0, 1, 2])

# Parameterized gates — create with values
gate_ry = RY.with_params(0.5)
circ.append(gate_ry, [1])
```

### DAG IR & Optimization

Circuits compile to a **Directed Acyclic Graph** representation that makes qubit/clbit dependencies explicit:

```python
dag = circ.to_dag()

print(dag.num_nodes(), dag.num_edges())
print(dag.topological_order())
print(dag.layers())          # partition into parallel layers
print(dag.critical_path_length())  # number of operations on the longest path
```

The `PassManager` runs optimization passes on the DAG:

```python
from quspect import PassManager, GateCancellation, RotationMerging

pm = PassManager([
    GateCancellation(),
    RotationMerging(),
])
optimized = pm.run(dag)
```

### Operators

Support for second-quantized operator algebra:

```python
from quspect import FermionOperator, QubitOperator
from quspect.operators.transforms import jordan_wigner

# Pauli string operators
H = QubitOperator({
    ((0, "X"), (1, "X")): 1.0,
    ((0, "Z"),): -0.5,
})

# Fermionic operators
f = FermionOperator({((2, 1), (1, 1), (0, 0), (3, 0)): 2.0})

# Jordan-Wigner transformation: fermions → qubits
qubit_op = jordan_wigner(f)
```

### IR Export

```python
from quspect.ir import QASM2ExportConfig, QASM2Exporter, QASM2ExportMode
from quspect.ir.qsal import dumps as qsal_dumps

# OpenQASM 2.0
print(QASM2Exporter().export(circ))  # strict mode

config = QASM2ExportConfig(mode=QASM2ExportMode.DECOMPOSE)
print(QASM2Exporter(config).export(circ))

config = QASM2ExportConfig(mode=QASM2ExportMode.EXTENDED)
print(QASM2Exporter(config).export(circ))

# QSAL (native JSON IR)
print(qsal_dumps(circ))
```

---

## Visualization

### Terminal Diagrams

Zero-dependency terminal circuit diagrams via `circuit.draw()`:

- Quantum wires as `-`, classical wires as `=`
- Gate labels in `[brackets]`, controls as `o`, targets labelled
- SWAP gates as `x` marks connected vertically
- Barriers as `|`

### SVG Rendering

Scalable vector graphics via `circuit.svg()`, with Jupyter notebook inline support:

```python
svg = circ.svg(style="dark", width=600, scale=1.2)
print(svg)                # SVG string
svg.to_file("out.svg")    # save to file
```

**Three built-in themes:**

| Theme | Background | Best for |
|-------|-----------|----------|
| `"official"` | White, slate text | Publications, papers |
| `"light"` | Cool gray | Light-mode IDEs, docs |
| `"dark"` | Dark navy | Dark-mode notebooks, slides |

**Custom themes:**

```python
from quspect.visualization import SvgStyle, draw_svg_circuit

custom = SvgStyle(
    background="#1a1a2e",
    wire="#e94560",
    classical_wire="#b8adc8",
    gate_fill="#0f3460",
    gate_stroke="#16213e",
    text="#ffffff",
    control="#e94560",
    barrier="#8a744d",
    measure_fill="#211040",
    channel_fill="#2a153c",
    channel_stroke="#c77532",
)
svg = draw_svg_circuit(circ, style=custom)
```

### Sampling Charts

Result visualization with statistical annotations:

```python
result.display(style="hbar", sort_mode="count_desc")
```

Four chart styles: `hbar`, `vbar`, `dot`, `svg`.

Side-by-side comparison:

```python
from quspect.visualization import compare_results

compare_results(
    result1,
    result2,
    label_a="Ideal",
    label_b="Noisy",
)
```

---

## Simulation

The simulator implements the `Backend` ABC, sharing the same `.run()` interface as cloud QPUs. It delegates to the C++ native engine for state-vector evolution:

```python
from quspect import QuantumCircuit
from quspect.simulation import Simulator
from quspect.backends import Backend

sim = Simulator(seed=42)
assert isinstance(sim, Backend)   # same interface as cloud backends

# State-vector evolution uses a measurement-free circuit and opts in to storage.
unitary = QuantumCircuit().h(0).cx(0, 1)
state_result = sim.run(unitary, save_statevector=True)
print(state_result.statevector)

# Shot-based sampling with projective measurement
sampled_circuit = unitary.copy().measure_all()
result = sim.run(sampled_circuit, shots=4096)

# Access results
print(result.counts)              # approximately {'00': 2048, '11': 2048}
print(result.probabilities)       # approximately {'00': 0.5, '11': 0.5}
print(result.most_frequent())     # one of the Bell-state outcomes
print(result.top_outcomes(3))     # top-3 outcomes
```

**Measurement modes:**

- **Terminal measurements** — all `measure` ops at circuit end: evolve once, then sample
- **Mid-circuit measurements** — measurements interleaved with gates: full re-execution per shot with state collapse

### Backend-neutral Parameter Sweeps

`ExecutionRequest.sweep()` snapshots one parameterized template and binds every
point in order. The same labelled request works with a local simulator or a
batch-capable cloud backend:

```python
import math

from quspect import ExecutionRequest, Parameter, QuantumCircuit, Simulator

theta = Parameter("theta")
template = QuantumCircuit().ry(theta, 0).measure_all()
request = ExecutionRequest.sweep(
    template,
    [{theta: 0.0}, {theta: math.pi}],
    labels=["zero", "one"],
    shots=128,
)

batch = Simulator(seed=7).execute(request)
print(batch.by_label("zero").counts)  # {'0': 128}
print(batch.by_label("one").counts)   # {'1': 128}
```

---

## Cloud QPU

QuSpect provides a **unified `Backend` interface** — the same `backend.run(circuit, shots=N)` API works for both local simulators and remote quantum hardware. Currently **LogicalQubit (逻辑比特)** is supported; additional platforms can be added by implementing `CloudBackend`.

### Cloud Setup

```bash
# Set your API key (get one from https://cloud.logicalqubit.com)
export LQCLOUD_API_KEY=qk_your_api_key_here

# Install QuSpect's provider extra
pip install "quspect[logicalqubit]"
```

### Run on Real Hardware

The calls below create real provider jobs and may consume quota or incur cost.
Check `provider.list_backends()` and the selected backend's current metadata
before choosing its name or physical layout.

```python
from quspect import QuantumCircuit
from quspect.backends.logicalqubit import LogicalQubitProvider

# 1. Connect to the cloud platform
provider = LogicalQubitProvider()
backend = provider.get_backend("QZ02")  # select a QPU visible to your account

print(backend.name)
print(backend.num_qubits)
print(backend.native_gate_set)
print(backend.platform_name)

# 2. Build any circuit
qc = QuantumCircuit(name="bell")
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 3. Run — same API as the local Simulator
result = backend.run(qc, shots=2000)
print(result.most_frequent())
print(result.counts)
```

### Compilation and Physical Layout

By default `backend.run()` attempts to compile a bound circuit for the target:

1. **Decompose** — non-native gates are recursively decomposed to the chip's native gate set (e.g. `CX → H + CZ + H`, `RX → H + RZ + H`)
2. **Optimize** — topology-independent cancellation, commutation, rotation
   merging, and single-qubit fusion run before physical routing
3. **Layout + Route** — SABRE finds an initial logical→physical mapping and
   inserts SWAP gates so every two-qubit gate respects the coupling topology

```python
# A nontrivial logical circuit that may require routing.
qc = QuantumCircuit(name="calibrated_subset")
qc.h(0)
qc.cx(0, 1).cx(0, 2).cx(0, 3)
qc.measure_all()

# Restrict layout and every inserted SWAP to a calibrated physical-qubit
# pool. SABRE chooses the starting layout inside the induced subgraph.
job = backend.submit(
    qc,
    shots=1000,
    transpile=True,
    physical_qubits=[7, 8, 9, 10, 11],
)

# Optionally fix the starting logical→physical assignment as well:
# logical q0→physical 7, q1→8, q2→9, q3→10.
job = backend.submit(
    qc,
    shots=1000,
    transpile=True,
    physical_qubits=[7, 8, 9, 10, 11],
    initial_layout=[7, 8, 9, 10],
)
```

`physical_qubits` must be unique, in range, large enough for the circuit, and
must induce a connected coupling subgraph. When both options are present,
every `initial_layout` entry must belong to `physical_qubits`. Routing may use
additional workspace wires, but only from that pool; QuSpect forwards the
expanded logical→physical layout required by the routed circuit.

To submit an already compiled/topology-valid circuit without QuSpect routing,
disable transpilation. In this mode `initial_layout` is passed directly to
the provider and `physical_qubits` is not accepted:

```python
result = backend.run(
    qc,
    shots=1000,
    transpile=False,
    initial_layout=[7, 8, 9, 10],
)
```

The routed circuit preserves original classical measurement targets and uses a
static provider layout even when SABRE inserts workspace SWAPs. DAG
optimization checks adjacency on every operand wire, so structures such as the
QAOA `CX–RZ(target)–CX` implementation of a ZZ phase are not incorrectly
cancelled.

### Async Submission

For long-running jobs, submit asynchronously and poll for results:

```python
# Submit without blocking
job = backend.submit(qc, shots=2000)
print(f"Job ID: {job.job_id}")
print(f"Status: {job.status().value}")  # queued / running / completed

# Block until complete (with optional timeout)
result = job.result(timeout=600)

# Cancel if needed
job.cancel()
```

### Batch Sampling and VQE

`ExecutionRequest` is the common envelope for simulators and cloud devices.
Each experiment has a stable label, and batches larger than the provider's
64-circuit limit are exposed as one recoverable composite job:

```python
from quspect import ExecutionRequest

request = ExecutionRequest.batch(
    [circuit_a, circuit_b],
    labels=["theta-plus", "theta-minus"],
    shots=2000,
    result_format="auto",
)
job = backend.submit_request(request)
batch = job.result(timeout=600)
print(batch.by_label("theta-plus").counts)

# Safe to persist: contains IDs and shapes, never credentials.
saved_reference = job.reference.to_dict()
```

If a later provider chunk fails after earlier chunks were accepted,
`PartialSubmissionError` exposes `submitted_job_ids`, serializable
`submitted_references`, and any `cancellation_failures` instead of losing the
already-created task handles.

On an older LogicalQubit server, lqcloud 0.4.1 may internally fall back from
native batch to individual submissions without exposing partial child IDs if
that fallback itself fails. For experiments requiring complete rollback
visibility, set `options={"provider_batching": "individual"}` (or pass the same
option through `SamplingEngine(execution_options=...)`). QuSpect then owns each
submission and can report/cancel every task it has created; the trade-off is
one provider task per circuit.

With `result_format="auto"`, lqcloud selects `counts` for narrow readout and
`memory` when the measured width requires per-shot output. QuSpect always
provides aggregated `result.counts`; when memory is returned, the original
shot sequence is also available as `result.memory`.

Shot-based observable objectives use the same job layer:

```python
from quspect import ObservableObjective, SamplingEngine

objective = ObservableObjective(ansatz, {"energy": hamiltonian})
plan = SamplingEngine(
    backend,
    shots=4096,
    execution_options={"transpile": False, "initial_layout": [0, 1]},
).compile(objective)

objective_job = plan.submit({theta: 0.2})
evaluation = objective_job.result(timeout=600)
print(evaluation.value("energy"), evaluation.derivative("energy"))

# Several optimizer points are flattened into one asynchronous provider task.
sweep_job = plan.submit_many([{theta: 0.1}, {theta: 0.2}, {theta: 0.3}])
sweep = sweep_job.result(timeout=600)
```

Every sampled plan carries a stable workload fingerprint in its
`JobReference`. On recovery, `plan.attach(job)` validates that fingerprint,
the backend, labels, circuit shapes, shot count, and result format before any
measurement data is decoded. Recovered multi-point jobs use
`plan.attach_many(execution_job)` in the same way.

The reference is a credential-free recovery manifest, not a signed receipt:
keep its JSON in trusted storage. Its fingerprint prevents accidental task
mix-ups but cannot authenticate a manifest deliberately rewritten by an
untrusted party.

### Simulator vs QPU

The unified `Backend` interface lets you switch between local simulation and real hardware by changing **one line**:

```python
from quspect import QuantumCircuit
from quspect.simulation import Simulator
from quspect.backends.logicalqubit import LogicalQubitProvider

qc = QuantumCircuit(name="compare")
qc.h(0).cx(0, 1).measure_all()

# Local simulation
sim = Simulator(seed=42)
sim_result = sim.run(qc, shots=2000)

# Real QPU (latency depends on the provider queue)
provider = LogicalQubitProvider()
backend = provider.get_backend("QZ02")
qpu_result = backend.run(qc, shots=2000)

# Compare
for bs in sorted(sim_result.counts):
    s = sim_result.get_count(bs)
    q = qpu_result.get_count(bs)
    print(f"  |{bs}⟩:  sim={s:4d}  qpu={q:4d}  Δ={s-q:+4d}")
```

---

## C++ Native Backend

The C++ backend (`cpp/`) is the execution engine for local simulators and exact
adjoint gradients:

```
cpp/
├── include/quspect_native/    # Public headers
├── src/
│   ├── kernels/scalar/        # Portable scalar kernels
│   ├── kernels/avx2/          # SIMD-accelerated kernels (AVX2)
│   ├── kernels/dispatch/      # Runtime CPU feature dispatch
│   ├── runtime/               # StateVector, ClassicalRegister, RuntimeEngine
│   ├── ir/                    # QSAL JSON decoder & program builder
│   ├── executor/              # SimulationExecutor, GradientExecutor
│   └── pybind/                # Python → C++ bridge (pybind11)
└── tests/                     # GoogleTest suite (14 test files)
```

**Python bridge:**

```python
from quspect import QuantumCircuit
from quspect.ir.qsal import dumps as qsal_dumps
from quspect.native import NativeExecutor

qc = QuantumCircuit().h(0).measure_all()
executor = NativeExecutor()
result = executor.run(qsal_dumps(qc), shots=1024)
print(result.counts)
```

If the extension is missing, circuit construction and other Python tooling
remain usable, but native-dependent entry points raise `NativeImportError`.

---

## Documentation

Full documentation is hosted on Read the Docs at
[quspect.readthedocs.io](https://quspect.readthedocs.io/en/latest/) (Chinese: [quspect.readthedocs.io/zh-cn/latest/](https://quspect.readthedocs.io/zh-cn/latest/)):

- **[Quickstart](https://quspect.readthedocs.io/en/latest/quickstart.html)** — installation, circuit building, simulation
- **[API Reference](https://quspect.readthedocs.io/en/latest/api.html)** — generated documentation for the currently listed public modules
- **[Architecture](https://quspect.readthedocs.io/en/latest/architecture.html)** — layered design, core concepts, architecture decisions

Build locally (the docs import the source tree directly, so the C++ extension is not needed):

```bash
python -m pip install -r docs/requirements.txt
cd docs
sphinx-build -b html source/ build/html/
sphinx-build -b html -D language=zh_CN source/ build/html/zh_CN/
```

---

## Development

```bash
# Lint & format
ruff check quspect/ tests/ && ruff format quspect/ tests/

# Type check
mypy quspect/

# Run tests
pytest

# Build C++ backend & run gtest (from the repository root)
cmake -S cpp -B cpp/build -G Ninja && cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

### Tooling

| Tool | Purpose |
|------|---------|
| `ruff` | Lint + Format |
| `mypy` | Static type checking (strict mode) |
| `pytest` + `pytest-cov` | Testing + coverage |
| `sphinx` | Documentation generation |
| `cmake` + `ninja` | C++ build system |
| `gtest` | C++ unit testing |
