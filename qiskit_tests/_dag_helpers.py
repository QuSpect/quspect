"""Helpers shared by the DAG / pass / transpiler cross-validation tests.

These complement :mod:`qiskit_tests.helpers` with utilities that are only
needed by ``test_dag_*.py``, ``test_passes_*.py`` and ``test_transpiler_*.py``:

* running a quspect DAG pass on a circuit and getting a circuit back,
* gate counting on quspect and qiskit circuits,
* a qubit-permutation matrix (little-endian) for routed-circuit equivalence,
* an independent numpy evaluation of a phase polynomial.

SABRE layout convention (verified in ``test_transpiler_sabre.py``):
``initial_mapping[l]`` / ``final_mapping[l]`` are **logical -> physical**.
The routed circuit is written on *wires* ``w = 0..m-1`` where wire ``w`` is
statically bound to physical qubit ``initial_mapping[w]`` (``m`` may exceed
the logical width when SABRE borrows idle physical qubits as workspace).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import qiskit
from numpy.typing import NDArray

import quspect
from quspect.dag import DAGCircuit, DAGToCircuitCompiler
from quspect.dag.passes import BasePass, PassManager
from quspect.transpiler import CompilationTarget

__all__ = [
    "apply_pass",
    "apply_passes",
    "line_edges",
    "ring_edges",
    "grid_edges",
    "heavy_hex_edges",
    "permutation_matrix",
    "quspect_gate_counts",
    "qiskit_gate_counts",
    "routed_permutation",
    "target_from_edges",
    "two_qubit_operands",
    "phase_function",
]


def apply_pass(
    pass_factory: Callable[[], BasePass], circ: quspect.QuantumCircuit
) -> quspect.QuantumCircuit:
    """Run one pass on ``circ.to_dag()`` and compile the DAG back to a circuit."""
    dag: DAGCircuit = pass_factory().run(circ.to_dag())
    return DAGToCircuitCompiler.compile(dag)


def apply_passes(
    pass_factories: Sequence[Callable[[], BasePass]],
    circ: quspect.QuantumCircuit,
    *,
    max_iterations: int = 10,
) -> tuple[quspect.QuantumCircuit, PassManager]:
    """Run a :class:`PassManager` over ``circ``; return the circuit and manager."""
    pm = PassManager([f() for f in pass_factories])
    dag = pm.run(circ.to_dag(), max_iterations=max_iterations)
    return DAGToCircuitCompiler.compile(dag), pm


def quspect_gate_counts(circ: quspect.QuantumCircuit) -> Counter[str]:
    """Count gate names of a quspect circuit (measure / barrier excluded)."""
    return Counter(i.gate.name for i in circ.instructions if i.gate is not None)


def qiskit_gate_counts(qc: qiskit.QuantumCircuit) -> Counter[str]:
    """Count gate names of a qiskit circuit (measure / barrier excluded)."""
    return Counter(
        ci.operation.name
        for ci in qc.data
        if ci.operation.name not in ("measure", "barrier")
    )


def two_qubit_operands(circ: quspect.QuantumCircuit) -> list[tuple[int, int]]:
    """Return the ordered operand pairs of every 2-qubit gate in ``circ``."""
    out: list[tuple[int, int]] = []
    for instr in circ.instructions:
        if instr.gate is not None and len(instr.qubits) == 2:
            a, b = instr.qubits
            out.append((a, b))
    return out


def permutation_matrix(sigma: Sequence[int]) -> NDArray[np.complex128]:
    """Little-endian matrix moving the state of qubit ``l`` to qubit ``sigma[l]``.

    ``P |x> = |y>`` with ``y_{sigma[l]} = x_l`` for every bit ``l``.
    """
    m = len(sigma)
    assert sorted(sigma) == list(range(m)), f"not a permutation: {sigma}"
    dim = 1 << m
    mat = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = 0
        for bit in range(m):
            if (x >> bit) & 1:
                y |= 1 << sigma[bit]
        mat[y, x] = 1.0
    return mat


def routed_permutation(
    initial_mapping: Sequence[int], final_mapping: Sequence[int]
) -> list[int]:
    """Wire permutation induced by routing: wire ``l`` -> wire ``sigma[l]``.

    Logical qubit ``l`` starts on wire ``l`` (physical ``initial_mapping[l]``)
    and ends on physical ``final_mapping[l]``, i.e. on the wire statically
    bound to that physical qubit.
    """
    phys_to_wire = {p: w for w, p in enumerate(initial_mapping)}
    return [phys_to_wire[final_mapping[w]] for w in range(len(initial_mapping))]


def line_edges(n: int) -> list[tuple[int, int]]:
    """Edges of a path graph 0-1-...-(n-1)."""
    return [(k, k + 1) for k in range(n - 1)]


def ring_edges(n: int) -> list[tuple[int, int]]:
    """Edges of a cycle graph on ``n`` nodes."""
    return [*line_edges(n), (n - 1, 0)]


def grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Edges of a ``rows x cols`` square grid (row-major numbering)."""
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


def heavy_hex_edges() -> tuple[int, list[tuple[int, int]]]:
    """A 12-qubit heavy-hex fragment (one hexagon of degree-2/3 nodes).

    Ring of 12 qubits where every other qubit is a "bridge" node, plus one
    pendant qubit attached to a degree-2 node -- the IBM heavy-hex motif.
    """
    edges = ring_edges(12)
    return 13, [*edges, (0, 12)]


def target_from_edges(
    num_qubits: int, edges: Iterable[tuple[int, int]]
) -> CompilationTarget:
    """Shorthand for :meth:`CompilationTarget.from_coupling_map`."""
    return CompilationTarget.from_coupling_map(num_qubits, list(edges))


def phase_function(
    terms: Iterable[tuple[float, Sequence[int]]], num_qubits: int
) -> NDArray[np.float64]:
    """Evaluate ``phi(x) = sum_k angle_k * (parity_k . x mod 2)`` for all x.

    ``x`` is the little-endian basis index (bit ``j`` = input variable
    ``x_j``).  Returned as an array of length ``2**num_qubits``.
    """
    out = np.zeros(1 << num_qubits)
    for angle, parity in terms:
        par = np.asarray(parity, dtype=int)
        for x in range(1 << num_qubits):
            bits = np.array([(x >> j) & 1 for j in range(num_qubits)])
            if int(par @ bits) % 2:
                out[x] += angle
    return out
