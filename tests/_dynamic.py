"""Exact reference semantics of circuits with measurements and classical control.

Independent of the simulators under test: plain numpy, an ensemble of
unnormalized pure states per classical record.  A record is the tuple of all
classical bits (clbit 0 first); the squared norms of its states add up to its
probability.  Overwriting a classical bit merges records, so a record can
hold several states (a mixed state); comparisons use density matrices.
"""

from __future__ import annotations

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix

Record = tuple[int, ...]


def _apply(state: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    k = len(qubits)
    tensor = np.asarray(matrix, dtype=complex).reshape([2] * (2 * k))
    state = np.tensordot(tensor, state, axes=(list(range(k, 2 * k)), list(qubits)))
    return np.moveaxis(state, list(range(k)), list(qubits))


def branches(
    circuit: QuantumCircuit, num_qubits: int | None = None, num_clbits: int | None = None
) -> dict[Record, list[np.ndarray]]:
    """Unnormalized final states of every classical record (qubit 0 = first axis)."""
    n = circuit.num_qubits if num_qubits is None else num_qubits
    m = circuit.num_clbits if num_clbits is None else num_clbits
    start = np.zeros([2] * n, dtype=complex)
    start[(0,) * n] = 1.0
    states: dict[Record, list[np.ndarray]] = {(0,) * m: [start]}
    for ins in circuit:
        if ins.is_barrier:
            continue
        if ins.is_gate:
            assert ins.gate is not None
            matrix = gate_matrix(ins.gate)
            states = {
                r: [_apply(s, matrix, ins.qubits) for s in ss] for r, ss in states.items()
            }
        elif ins.is_conditional:
            assert ins.gate is not None and ins.condition is not None
            matrix = gate_matrix(ins.gate)
            for record in states:
                value = sum(record[c] << i for i, c in enumerate(ins.clbits))
                if value == ins.condition:
                    states[record] = [
                        _apply(s, matrix, ins.qubits) for s in states[record]
                    ]
        elif ins.is_measure:
            for qubit, clbit in zip(ins.qubits, ins.clbits, strict=True):
                out: dict[Record, list[np.ndarray]] = {}
                for record, ensemble in states.items():
                    for state in ensemble:
                        for outcome in (0, 1):
                            index: list[slice | int] = [slice(None)] * n
                            index[qubit] = outcome
                            projected = np.zeros_like(state)
                            projected[tuple(index)] = state[tuple(index)]
                            if np.vdot(projected, projected).real < 1e-24:
                                continue
                            key = record[:clbit] + (outcome,) + record[clbit + 1 :]
                            out.setdefault(key, []).append(projected)
                states = out
        else:
            raise AssertionError(f"unsupported instruction {ins.type}")
    return states


def densities(
    circuit: QuantumCircuit, num_qubits: int | None = None, num_clbits: int | None = None
) -> dict[Record, np.ndarray]:
    """Unnormalized density matrix of every reachable classical record."""
    out = {}
    for record, ensemble in branches(circuit, num_qubits, num_clbits).items():
        dim = ensemble[0].size
        rho = np.zeros((dim, dim), dtype=complex)
        for state in ensemble:
            vec = state.reshape(-1)
            rho += np.outer(vec, vec.conj())
        out[record] = rho
    return out


def distribution(
    circuit: QuantumCircuit, num_qubits: int | None = None, num_clbits: int | None = None
) -> dict[Record, float]:
    """Exact probability of every reachable classical record."""
    return {
        r: float(np.trace(rho).real)
        for r, rho in densities(circuit, num_qubits, num_clbits).items()
    }


def assert_same_behaviour(
    a: QuantumCircuit, b: QuantumCircuit, *, atol: float = 1e-8
) -> None:
    """Same records, and per record the same unnormalized density matrix."""
    n = max(a.num_qubits, b.num_qubits)
    m = max(a.num_clbits, b.num_clbits)
    da, db = densities(a, n, m), densities(b, n, m)
    assert set(da) == set(db), (sorted(da), sorted(db))
    for record, rho in da.items():
        error = float(np.max(np.abs(rho - db[record])))
        assert error < atol, (record, error)
