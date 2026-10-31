"""Two-qubit KAK synthesis into other entanglers / Euler bases."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.synthesis.two_qubit import (
    Entangler,
    TwoQubitBasis,
    entangler,
    euler_ops,
    num_cz_required,
    ops_to_instructions,
    synthesize_two_qubit_ops,
)
from tests.synthesis._unitary import circuit_unitary, phase_distance, random_unitary

BASES = [
    TwoQubitBasis("cz", "zxz"),
    TwoQubitBasis("cx", "zxz"),
    TwoQubitBasis("cx", "zsx"),
    TwoQubitBasis("cz", "zsx"),
]


def _unitary(ops: list) -> np.ndarray:  # type: ignore[type-arg]
    circ = QuantumCircuit._from_instruction_snapshot(ops_to_instructions(ops, (0, 1)))
    return circuit_unitary(circ, 2)


def _inputs() -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    us = [random_unitary(4, rng) for _ in range(40)]
    us += [np.diag([1, 1, 1, np.exp(1j * t)]) for t in rng.uniform(0, 6, 5)]
    cx = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
    us += [np.eye(4, dtype=complex), np.eye(4, dtype=complex)[[0, 2, 1, 3]], cx]
    us += [np.kron(random_unitary(2, rng), random_unitary(2, rng))]
    return us


@pytest.mark.parametrize("basis", BASES, ids=str)
def test_synthesis_in_basis_is_exact_and_minimal(basis: TwoQubitBasis) -> None:
    for u in _inputs():
        ops = synthesize_two_qubit_ops(u, basis=basis)
        assert {name for name, _, _ in ops} <= basis.gates
        assert phase_distance(u, _unitary(ops)) <= 1e-9
        n2 = sum(1 for _, q, _ in ops if len(q) == 2)
        assert n2 == num_cz_required(u)


def test_cx_layers_not_longer_than_cz() -> None:
    """The folded Hadamards keep CX results at most as long as CZ ones."""
    tot = {"cz": 0, "cx": 0}
    for u in _inputs():
        for ent in tot:
            tot[ent] += len(synthesize_two_qubit_ops(u, basis=TwoQubitBasis(ent)))
    assert tot["cx"] <= tot["cz"]


def test_cx_orientation_is_chosen_per_gate() -> None:
    """CX with control 1 (target 0) is emitted with a reversed CX."""
    cx10 = np.eye(4, dtype=complex)[[0, 3, 2, 1]]
    ops = synthesize_two_qubit_ops(cx10, basis=TwoQubitBasis("cx"))
    assert [q for name, q, _ in ops if name == "cx"] == [(1, 0)]
    assert len(ops) <= len(synthesize_two_qubit_ops(cx10))


def test_entangler_from_matrix_cy_and_rejects_iswap() -> None:
    cy = np.eye(4, dtype=complex)
    cy[2:, 2:] = [[0, -1j], [1j, 0]]
    ent = Entangler.from_matrix("cy", cy)
    a = np.kron(*ent.after)
    b = np.kron(*ent.before)
    assert phase_distance(np.diag([1, 1, 1, -1]).astype(complex), a @ cy @ b) < 1e-12
    iswap = np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    with pytest.raises(ValueError, match="not locally equivalent"):
        Entangler.from_matrix("iswap", iswap)


def test_basis_from_gates() -> None:
    assert TwoQubitBasis.from_gates({"cx", "rz", "sx", "x"}) == TwoQubitBasis(
        "cx", "zsx"
    )
    assert TwoQubitBasis.from_gates({"cz", "cx", "h", "rz", "x"}) == TwoQubitBasis()
    assert TwoQubitBasis.from_gates({"cx", "h", "rz"}) is None
    with pytest.raises(KeyError):
        entangler("iswap")
    with pytest.raises(ValueError):
        euler_ops(np.eye(2), 0, basis="xyx")


def _clifford_dressed(core: np.ndarray, seed: int) -> np.ndarray:
    """``core`` with random single-qubit Clifford layers on both sides."""
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.standard import H, S

    rng = np.random.default_rng(seed)
    gens = [np.asarray(gate_matrix(g), dtype=complex) for g in (H, S)]

    def cliff() -> np.ndarray:
        m = np.eye(2, dtype=complex)
        for _ in range(6):
            m = gens[rng.integers(2)] @ m
        return m

    return np.kron(cliff(), cliff()) @ core @ np.kron(cliff(), cliff())


@pytest.mark.parametrize(
    ("coords", "ratio"),
    [
        ((0.35, 0.35, 0.35), 0.55),  # Heisenberg: stabilizer V⊗V
        ((0.35, 0.35, 0.2), 0.85),  # XXZ: Rz(φ)⊗Rz(φ)
        ((0.35, 0.0, 0.0), 0.97),  # CP-like: Rx⊗Rx
        ((np.pi / 4, 0.0, 0.0), 0.92),  # CX-like (Clifford dressing)
        ((0.4, 0.3, 0.1), 0.97),  # generic: Pauli pairs only
    ],
)
def test_stabilizer_shortens_outer_layers(
    coords: tuple[float, float, float], ratio: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The KAK stabilizer is used to simplify the outer single-qubit layers.

    Degenerate Weyl points leave the KAK factors undetermined (the
    eigensolver returns arbitrary ones); choosing the stabilizer element
    roughly halves the single-qubit gates of a Clifford-dressed Heisenberg
    gate in ``{rz, sx, x}``.  Never longer than without the search.

    Both counts depend on the KAK factors the platform's LAPACK returns
    (macOS Accelerate vs. OpenBLAS on x86-64 / arm64 Linux differ by a few
    gates in total), so the ratios leave room for that spread.
    """
    from quspect.synthesis import two_qubit as tq

    basis = TwoQubitBasis("cx", "zsx")
    us = [_clifford_dressed(tq.canonical_gate(*coords), seed) for seed in range(12)]

    def count(u: np.ndarray) -> int:
        ops = tq._synthesize_two_qubit_ops(u, 1e-9, 1e-9, basis)
        assert phase_distance(u, _unitary(ops)) <= 1e-9
        return sum(1 for _, q, _ in ops if len(q) == 1)

    new = [count(u) for u in us]
    monkeypatch.setattr(
        tq, "_stabilizer_candidates", lambda *a, **kw: [(tq._I2, tq._I2)]
    )
    old = [count(u) for u in us]
    assert all(n <= o for n, o in zip(new, old, strict=True))
    assert sum(new) <= ratio * sum(old), (sum(new), sum(old))
