"""Exact reference semantics of circuits with measurements and classical control.

Independent of the simulators under test: plain numpy, an ensemble of
unnormalized pure states per classical record.  A record is the tuple of all
classical bits (clbit 0 first); the squared norms of its states add up to its
probability.  Overwriting a classical bit merges records, so a record can
hold several states (a mixed state); comparisons use density matrices.
"""

from __future__ import annotations

import random

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix

Record = tuple[int, ...]

#: Branches below this probability are numerical noise (a compiled circuit is
#: exact to about 1e-12 in amplitude) and are not followed.
_NEGLIGIBLE = 1e-18


def _apply(
    state: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...]
) -> np.ndarray:
    k = len(qubits)
    tensor = np.asarray(matrix, dtype=complex).reshape([2] * (2 * k))
    state = np.tensordot(tensor, state, axes=(list(range(k, 2 * k)), list(qubits)))
    return np.moveaxis(state, list(range(k)), list(qubits))


def branches(
    circuit: QuantumCircuit,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
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
                r: [_apply(s, matrix, ins.qubits) for s in ss]
                for r, ss in states.items()
            }
        elif ins.is_conditional:
            assert ins.conditional_gate is not None and ins.condition is not None
            matrix = gate_matrix(ins.conditional_gate)
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
                            if np.vdot(projected, projected).real < _NEGLIGIBLE:
                                continue
                            key = record[:clbit] + (outcome,) + record[clbit + 1 :]
                            out.setdefault(key, []).append(projected)
                states = out
        else:
            raise AssertionError(f"unsupported instruction {ins.type}")
    return states


def densities(
    circuit: QuantumCircuit,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
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
    circuit: QuantumCircuit,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
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
    zero = np.zeros((2**n, 2**n), dtype=complex)
    for record in set(da) | set(db):
        # a record missing on one side must be negligible on the other
        error = float(np.max(np.abs(da.get(record, zero) - db.get(record, zero))))
        assert error < atol, (record, error)


def assert_compiled_behaviour(
    logical: QuantumCircuit,
    physical: QuantumCircuit,
    final_layout: list[int],
    num_physical: int,
    *,
    atol: float = 1e-7,
) -> None:
    """``physical`` (all physical qubits start in |0>) behaves as ``logical``.

    Same records with the same probabilities, and per record the same state
    with logical qubit ``q`` on physical qubit ``final_layout[q]`` and every
    other physical qubit in |0>.
    """
    n = len(final_layout)
    m = max(logical.num_clbits, physical.num_clbits)
    want = branches(logical, n, m)
    got = densities(physical, num_physical, m)
    free = [p for p in range(num_physical) if p not in final_layout]
    zeros = np.zeros(2 ** len(free), dtype=complex)
    zeros[0] = 1.0
    order = [
        final_layout.index(p) if p in final_layout else n + free.index(p)
        for p in range(num_physical)
    ]
    empty = np.zeros((2**num_physical, 2**num_physical), dtype=complex)
    for record in set(got) - set(want):
        assert float(np.max(np.abs(got[record]))) < atol, (record, "unexpected record")
    for record, ensemble in want.items():
        rho = np.zeros((2**num_physical, 2**num_physical), dtype=complex)
        for state in ensemble:
            tensor = np.kron(state.reshape(-1), zeros).reshape([2] * num_physical)
            vec = np.transpose(tensor, order).reshape(-1)
            rho += np.outer(vec, vec.conj())
        error = float(np.max(np.abs(rho - got.get(record, empty))))
        assert error < atol, (record, error)


def _random_gate(qc: QuantumCircuit, rng: random.Random, n: int) -> None:
    r = rng.random()
    q = rng.randrange(n)
    if r < 0.45 or n < 2:
        getattr(qc, rng.choice(["h", "x", "z", "s", "t", "sdg", "y", "sx"]))(q)
    elif r < 0.6:
        getattr(qc, rng.choice(["rz", "rx", "ry"]))(rng.choice([0.3, 1.1, -0.7]), q)
    elif r < 0.9 or n < 3:
        a, b = rng.sample(range(n), 2)
        name = rng.choice(["cx", "cz", "swap", "rzz", "cp"])
        if name in ("rzz", "cp"):
            getattr(qc, name)(rng.choice([0.4, -1.3]), a, b)
        else:
            getattr(qc, name)(a, b)
    else:
        qc.ccx(*rng.sample(range(n), 3))


def random_dynamic_circuit(
    seed: int, max_qubits: int = 4, max_size: int = 22
) -> QuantumCircuit:
    """Random circuit with gates, measurements, barriers and conditional gates."""
    rng = random.Random(seed)
    n, m = rng.randint(1, max_qubits), rng.randint(1, 3)
    qc = QuantumCircuit()
    for _ in range(rng.randint(3, max_size)):
        r = rng.random()
        if r < 0.15:
            qc.measure([rng.randrange(n)], [rng.randrange(m)])
        elif r < 0.45:
            k = rng.randint(1, min(2, m))
            with qc.if_test(rng.sample(range(m), k), rng.randrange(1 << k)):
                _random_gate(qc, rng, n)
        elif r < 0.48:
            qc.barrier(*rng.sample(range(n), rng.randint(1, n)))
        else:
            _random_gate(qc, rng, n)
    return qc
