# LogicalQubit Compilation and Routing

Use this reference for calibrated physical-qubit selection, initial layouts,
SABRE routing, compiled-circuit correctness, mapping investigations, and
LogicalQubit hardware submission.

## Public submission modes

Prefer the public backend API:

```python
job = backend.submit(
    circuit,
    shots=4096,
    transpile=True,
    physical_qubits=[7, 8, 9, 10, 11],
    initial_layout=[7, 8, 9, 10],
)
result = job.result(timeout=600)
```

The same compilation options work with blocking `backend.run(...)` and through
`ExecutionRequest.options` or `SamplingEngine(execution_options=...)`.

With `transpile=True`:

- Compilation order is decompose, topology-independent DAG optimization, then
  layout and routing. Routing must remain the final gate-rewriting stage.
- `physical_qubits` is an ordered physical-device pool. QuSpect builds its
  induced coupling subgraph and restricts both mapping and inserted SWAPs to it.
- `initial_layout[logical] = physical` fixes the starting logical-to-physical
  assignment. It is not the final mapping after routing.
- Supplying only `physical_qubits` lets SABRE choose the starting assignment
  inside that pool.
- Supplying both requires every initial-layout entry to occur in the pool.
- SABRE may use otherwise idle pool members as workspace wires. The layout sent
  to the provider is expanded to cover every wire in the routed circuit.

`physical_qubits` must be a nonempty, unique, in-range integer sequence, contain
at least as many qubits as the input circuit uses, and induce a connected
coupling subgraph when routing a multi-qubit circuit. `initial_layout` must be a
unique, in-range integer sequence whose length equals the input circuit's
logical qubit count.

Use direct submission only for a circuit already compiled for the intended
topology:

```python
job = backend.submit(
    compiled_circuit,
    shots=4096,
    transpile=False,
    initial_layout=[7, 8, 9, 10],
)
```

Here `initial_layout` is passed directly to the provider. Do not combine
`physical_qubits` with `transpile=False`; it raises `ValueError`.

## Lower-level transpiler API

```python
from quspect.transpiler import CompilationTarget, SABRE

target = CompilationTarget.from_backend(backend)
subtarget, dense_to_physical = target.induced_subgraph([7, 8, 9, 10, 11])
routed, initial_mapping, final_mapping = SABRE(subtarget).solve(
    circuit,
    seed=0,
    initial_mapping=[0, 1, 2, 3],
)
```

- `CompilationTarget.induced_subgraph(...)` returns a dense target plus
  `dense_to_physical[dense] = device_physical`.
- `SABRE.solve(..., initial_mapping=...)` uses
  `initial_mapping[logical] = target_physical`.
- Its returned initial and final mappings use the target's index space. Convert
  through `dense_to_physical` after routing within an induced subgraph.
- `LayoutRoutingPass(..., initial_mapping=...)` exposes the same fixed-start
  behavior but is an internal pass; normal cloud code should use the backend.

## Mapping and result correctness

Keep these three domains distinct:

1. Circuit wire index: the index written in each routed instruction.
2. Logical-to-physical provider layout: the static layout submitted with the
   routed circuit.
3. Classical result bit: the clbit targeted by an original measurement.

The routed output rewrites gates into a fixed circuit-wire space and forwards a
static provider layout. Its final SABRE mapping is diagnostic state, not a
second provider remap. Measurements retain their original classical targets and
are emitted as a terminal fence, so result keys remain classical-bit outcomes;
do not reinterpret them as physical-qubit order.

When checking a mapping bug, trace:

- the source measurement `(qubit, clbit)`;
- routed instruction wires and terminal measurement clbits;
- backend-prepared `initial_layout`;
- provider result decoding into `SimulationResult.counts`.

Do not infer order from the displayed `top` summary alone.

## Optimizer safety near routing

DAG cancellation and commutation must test adjacency on every operand wire and
preserve ordered operands for controlled gates. In particular, the QAOA
`CX(u, v) -> RZ(v) -> CX(u, v)` realization of a ZZ phase must not lose its two
CX gates. Optimize before routing so a generic DAG pass cannot move
measurements across mapping-changing SWAPs.

## Focused evidence and tests

Inspect these before changing behavior:

- `quspect/backends/logicalqubit/_backend.py`
- `quspect/transpiler/_mapping.py`
- `quspect/transpiler/_target.py`
- `quspect/transpiler/_passes.py`
- `tests/backends/test_logicalqubit_backend.py`
- `tests/transpiler/test_sabre.py`
- `tests/dag/passes/test_gate_cancellation.py`
- `tests/dag/passes/test_commutation.py`

The live tests in `tests/backends/test_logicalqubit.py` require credentials and
hardware availability; do not substitute them for offline backend regression
tests.
