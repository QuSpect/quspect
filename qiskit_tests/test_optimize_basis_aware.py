"""Basis-aware OptimizePass: equivalence and gate-count checks.

Random circuits are lowered to the benchmark basis
``{cz, h, rz, s, sdg, t, tdg, x, y, z}`` and optimized with
``OptimizePass(basis_gates=...)``. The result must stay in the basis, be
unitarily equivalent (up to global phase) and never be larger than the
input or than the classic ``u3`` fusion followed by decomposition.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import quspect
from qiskit_tests.helpers import assert_unitary_equal, circuit_unitary, to_qiskit
from quspect.ir.compilation import decompose_deep
from quspect.transpiler import SABRE, CompilationTarget
from quspect.transpiler._passes import DecomposePass, OptimizePass

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _random_basis_circuit(
    seed: int, n: int = 4, size: int = 60
) -> quspect.QuantumCircuit:
    rng = random.Random(seed)
    c = quspect.QuantumCircuit()
    angles = [0.0, math.pi / 2, -math.pi / 2, math.pi, math.pi / 4, 2 * math.pi]
    for _ in range(size):
        k = rng.random()
        q = rng.randrange(n)
        if k < 0.3:
            getattr(c, rng.choice(["h", "x", "y", "z", "s", "sdg", "t", "tdg"]))(q)
        elif k < 0.5:
            c.rz(rng.choice(angles) if rng.random() < 0.5 else rng.uniform(-3, 3), q)
        elif k < 0.6:
            name = rng.choice(["rx", "ry", "u3"])
            params = [rng.uniform(-3, 3) for _ in range(3 if name == "u3" else 1)]
            getattr(c, name)(*params, q)
        else:
            a, b = rng.sample(range(n), 2)
            getattr(c, rng.choice(["cx", "cz"]))(a, b)
    c.h(n - 1)  # pin the width
    return decompose_deep(c, supported_gates=BASIS)


@pytest.mark.parametrize("seed", range(25))
def test_basis_aware_optimization_is_equivalent_and_not_larger(seed: int) -> None:
    circ = _random_basis_circuit(seed)
    n = circ.num_qubits
    out = OptimizePass(basis_gates=BASIS).run(circ)
    assert all(i.gate is not None and i.gate.name in BASIS for i in out.instructions)
    assert len(out.instructions) <= len(circ.instructions)
    assert_unitary_equal(
        circuit_unitary(out, num_qubits=n),
        circuit_unitary(circ, num_qubits=n),
        up_to_phase=True,
        atol=1e-7,
    )
    classic = decompose_deep(OptimizePass().run(circ), supported_gates=BASIS)
    assert len(out.instructions) <= len(classic.instructions)


def test_short_runs_are_not_expanded() -> None:
    """``h·rz`` stays two gates (classic fusion + decompose gave seven)."""
    circ = quspect.QuantumCircuit().h(0).rz(0.3, 0).cz(0, 1).h(1)
    out = OptimizePass(basis_gates=BASIS).run(circ)
    assert [i.gate.name for i in out.instructions if i.gate] == ["h", "rz", "cz", "h"]


def test_identity_rotations_are_removed() -> None:
    circ = (
        quspect.QuantumCircuit()
        .h(0)
        .rz(0.0, 1)
        .cz(0, 1)
        .rz(math.pi, 1)
        .rz(math.pi, 1)
        .h(1)
    )
    for opt in (OptimizePass(), OptimizePass(basis_gates=BASIS)):
        names = [i.gate.name for i in opt.run(circ).instructions if i.gate]
        assert names == ["h", "cz", "h"]


def test_u3_decomposes_to_at_most_five_gates() -> None:
    circ = quspect.QuantumCircuit().u3(0.3, 0.2, 0.1, 0).u3(math.pi / 2, 0.0, 0.0, 0)
    out = decompose_deep(circ, supported_gates=BASIS)
    assert len(out.instructions) <= 5 + 3
    assert all(i.gate is not None and i.gate.name in BASIS for i in out.instructions)
    assert_unitary_equal(
        circuit_unitary(out, num_qubits=1),
        circuit_unitary(circ, num_qubits=1),
        up_to_phase=True,
        atol=1e-9,
    )


def _qft(n: int) -> quspect.QuantumCircuit:
    c = quspect.QuantumCircuit()
    for j in reversed(range(n)):
        c.h(j)
        for k in reversed(range(j)):
            c.cp(math.pi / 2 ** (j - k), k, j)
    return c


def test_qft16_line_keeps_tiny_phases() -> None:
    """Regression: QFT-16 routed on a line lost its ~1e-5 rad phases.

    Mirrors the benchmark's ``quspect_O1`` flow (SABRE → decompose →
    basis-aware OptimizePass → decompose). The old trace-overlap check in
    single-qubit synthesis accepted candidates that dropped tiny rotations,
    giving ``1 - F ≈ 1.7e-6``.
    """
    n = 16
    target = CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=BASIS
    )
    decompose = DecomposePass(target)
    routed, _, _ = SABRE(target).solve(decompose.run(_qft(n)), seed=0)
    routed = decompose.run(routed)
    out = decompose.run(OptimizePass(basis_gates=BASIS).run(routed))
    assert len(out.instructions) < len(routed.instructions)

    rng = np.random.default_rng(1)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    a = Statevector(psi).evolve(to_qiskit(routed, num_qubits=n))
    b = Statevector(psi).evolve(to_qiskit(out, num_qubits=n))
    fidelity = abs(np.vdot(a.data, b.data)) ** 2
    assert fidelity >= 1 - 1e-9
