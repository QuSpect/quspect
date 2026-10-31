"""Cross-validate two-qubit KAK synthesis and block consolidation with qiskit."""

from __future__ import annotations

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit as QkCircuit  # noqa: E402
from qiskit.circuit.library import CZGate  # noqa: E402
from qiskit.quantum_info import Operator, random_unitary  # noqa: E402
from qiskit.synthesis import TwoQubitBasisDecomposer  # noqa: E402

from quspect.synthesis import synthesize_two_qubit  # noqa: E402


def _to_qiskit(circ: object) -> QkCircuit:
    qc = QkCircuit(2)
    for ins in circ.instructions:  # type: ignore[attr-defined]
        name, qs = ins.gate.name, ins.qubits
        if name == "rz":
            qc.rz(float(ins.gate.params[0]), qs[0])
        elif name == "cz":
            qc.cz(*qs)
        else:
            getattr(qc, name)(qs[0])
    return qc


@pytest.mark.parametrize("seed", range(200))
def test_matches_qiskit_operator_and_cz_count(seed: int) -> None:
    u = random_unitary(4, seed=seed)
    # Special structure every few seeds: products of controlled phases.
    if seed % 4 == 1:
        qc = QkCircuit(2)
        qc.cp(0.1 * seed, 0, 1)
        qc.cp(-0.03 * seed, 1, 0)
        u = Operator(qc)
    elif seed % 4 == 2:
        qc = QkCircuit(2)
        qc.rxx(0.01 * seed, 0, 1)
        qc.ryy(0.02 * seed, 0, 1)
        u = Operator(qc)
    big_endian = u.reverse_qargs().data
    circ = synthesize_two_qubit(big_endian)
    ours = Operator(_to_qiskit(circ))
    assert ours.equiv(u, atol=1e-9)
    n_ours = sum(1 for ins in circ.instructions if ins.gate.name == "cz")
    ref = TwoQubitBasisDecomposer(CZGate())(u.data)
    n_ref = sum(1 for inst in ref.data if inst.operation.name == "cz")
    assert n_ours == n_ref


def _qft(n: int) -> object:
    import math

    import quspect

    c = quspect.QuantumCircuit()
    for j in reversed(range(n)):
        c.h(j)
        for k in reversed(range(j)):
            c.cp(math.pi / 2 ** (j - k), k, j)
    return c


def test_qft16_line_full_pipeline_fidelity() -> None:
    """Regression: the benchmark ``quspect_O1`` flow must keep QFT phases.

    ``decompose → consolidate → SABRE → decompose → consolidate →
    OptimizePass(basis) → decompose → consolidate → decompose`` on a
    16-qubit line. Compared with the ideal (uncompiled) QFT on a random
    product state, after undoing the routing permutation:
    ``F ≥ 1 - 1e-9``. Also checks that the final consolidation only emits
    target-basis gates and never increases the two-qubit gate count.
    """
    import numpy as np
    from qiskit.quantum_info import Statevector

    from qiskit_tests.helpers import to_qiskit
    from quspect.transpiler import SABRE, CompilationTarget, ConsolidateBlocksPass
    from quspect.transpiler._passes import DecomposePass, OptimizePass

    basis = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}
    n = 16
    target = CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=basis
    )
    decompose = DecomposePass(target)
    consolidate = ConsolidateBlocksPass(target)
    src = _qft(n)
    c = consolidate.run(decompose.run(src))  # type: ignore[arg-type]
    routed, pi0, pi_final = SABRE(target).solve(c, seed=0)
    c = consolidate.run(decompose.run(routed))
    c = decompose.run(OptimizePass(basis_gates=basis).run(c))
    before_last = c
    out = decompose.run(consolidate.run(c))

    def n2q(circ: object) -> int:
        return sum(1 for i in circ.instructions if len(i.qubits) == 2)  # type: ignore[attr-defined]

    assert {i.gate.name for i in out.instructions} <= basis
    assert n2q(out) <= n2q(before_last)

    rng = np.random.default_rng(7)
    psi = np.ones(1, dtype=complex)
    for _ in range(n):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = np.kron(v / np.linalg.norm(v), psi)  # qubit 0 least significant
    ideal = Statevector(psi).evolve(to_qiskit(src, num_qubits=n)).data  # type: ignore[arg-type]
    got = Statevector(psi).evolve(to_qiskit(out, num_qubits=n)).data
    # Logical wire w ends on wire wire_of[pi_final[w]] (routed is in wire space).
    wire_of = {p: w for w, p in enumerate(pi0)}
    dest = [wire_of[pi_final[w]] for w in range(n)]
    # Statevector axis n-1-q holds qubit q.
    ideal_t = ideal.reshape([2] * n)
    perm = [0] * n
    for w in range(n):
        perm[n - 1 - dest[w]] = n - 1 - w
    ideal_routed = np.transpose(ideal_t, perm).reshape(-1)
    fidelity = abs(np.vdot(ideal_routed, got)) ** 2
    assert fidelity >= 1 - 1e-9
