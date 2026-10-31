"""Single-qubit gate minimization around CX gates (rotation pushing, reversal)."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.standard import SX, H, S, T, X
from quspect.gates.synthesis import synthesize_one_qubit
from quspect.transpiler._gauge import _LengthModel, _rot, minimize_single_qubit_gates

CX_ONE = frozenset({"h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"})
IBM_ONE = frozenset({"rz", "sx", "x"})


def _unitary(circuit: QuantumCircuit, n: int) -> np.ndarray:
    u = np.eye(2**n, dtype=complex).reshape([2] * n + [2**n])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = np.asarray(gate_matrix(ins.gate)).reshape([2] * (2 * k))
        u = np.tensordot(g, u, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        u = np.moveaxis(u, list(range(k)), list(ins.qubits))
    return u.reshape(2**n, 2**n)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    k = int(np.argmax(np.abs(b)))
    phase = a.flat[k] / b.flat[k]
    phase /= abs(phase)
    return float(np.max(np.abs(a - phase * b)))


def _random_circuit(seed: int, ibm: bool) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(30):
        a, b = rng.sample(range(3), 2)
        r = rng.randrange(7)
        if r < 2:
            qc.cx(a, b)
        elif r == 2:
            qc.rz(rng.choice([rng.uniform(-3, 3), math.pi / 2, math.pi]), a)
        elif r == 3:
            if ibm:
                qc.sx(a)
            else:
                qc.h(a)
        elif r == 4:
            qc.x(a)
        elif r == 5:
            qc.cz(a, b)
        else:
            qc.rz(rng.uniform(-3, 3), a)
            if ibm:
                qc.sx(a)
            else:
                qc.h(a)
    return qc


def _n1q(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if i.is_gate and len(i.qubits) == 1)


@pytest.mark.parametrize("ibm", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_equivalent_and_never_longer(ibm: bool, seed: int) -> None:
    basis = IBM_ONE if ibm else CX_ONE
    qc = _random_circuit(seed, ibm)
    out = minimize_single_qubit_gates(qc, basis)
    assert _distance(_unitary(out, 3), _unitary(qc, 3)) < 1e-9
    assert _n1q(out) <= _n1q(qc)
    # the same two-qubit gates on the same pairs, in the same order
    pairs = [
        (i.gate.name, frozenset(i.qubits))
        for i in out
        if i.gate is not None and len(i.qubits) == 2
    ]
    assert pairs == [
        (i.gate.name, frozenset(i.qubits))
        for i in qc
        if i.gate is not None and len(i.qubits) == 2
    ]
    for ins in out:
        if ins.gate is not None and len(ins.qubits) == 1:
            assert ins.gate.name in basis


def test_rz_moves_through_control() -> None:
    qc = QuantumCircuit().rz(0.3, 0).cx(0, 1).rz(0.4, 0).cx(0, 1).rz(0.5, 0)
    out = minimize_single_qubit_gates(qc, CX_ONE)
    assert _n1q(out) == 1
    assert _distance(_unitary(out, 2), _unitary(qc, 2)) < 1e-9


def test_x_rotation_moves_through_target() -> None:
    qc = QuantumCircuit().sx(1).cx(0, 1).sx(1)
    out = minimize_single_qubit_gates(qc, IBM_ONE)
    assert [i.gate.name for i in out if i.gate is not None] == ["cx", "x"]
    assert _distance(_unitary(out, 2), _unitary(qc, 2)) < 1e-9


def test_cx_reversal_absorbs_hadamards() -> None:
    qc = QuantumCircuit().h(0).h(1).cx(0, 1).h(0).h(1)
    out = minimize_single_qubit_gates(qc, CX_ONE)
    assert [(i.gate.name, i.qubits) for i in out if i.gate is not None] == [
        ("cx", (1, 0))
    ]


def test_barrier_blocks_movement() -> None:
    qc = QuantumCircuit().rz(0.3, 0).cx(0, 1)
    qc.barrier()
    qc.cx(0, 1).rz(0.4, 0)
    out = minimize_single_qubit_gates(qc, CX_ONE)
    assert _n1q(out) == 2
    assert [i.is_barrier for i in out].count(True) == 1


def test_untouched_runs_are_kept() -> None:
    qc = QuantumCircuit().h(0).t(0).h(0).cx(0, 1)
    out = minimize_single_qubit_gates(qc, CX_ONE)
    assert [i.gate.name for i in out if i.gate is not None] == ["h", "t", "h", "cx"]


def _matrices() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    ms = [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(100)
    ]
    angles = [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2, 0.3]
    for a, b, c in itertools.product(angles, repeat=3):
        for ax in ("xz", "yz", "zx"):
            ms.append(_rot(ax[0], a) @ _rot(ax[1], b) @ _rot(ax[0], c) * np.exp(1j * a))
    r = random.Random(0)
    gates = [np.asarray(gate_matrix(g), dtype=complex) for g in (H, S, T, X, SX)]
    for _ in range(200):
        m = np.eye(2, dtype=complex)
        for _ in range(r.randrange(1, 6)):
            m = r.choice(gates) @ m
        ms.append(m)
    return ms


@pytest.mark.parametrize(
    "basis",
    [
        CX_ONE,
        IBM_ONE,
        frozenset({"h", "rz"}),
        frozenset({"rz", "sx"}),
        frozenset({"h", "rz", "sx", "sxdg", "x"}),
    ],
)
def test_length_model_matches_synthesis(basis: frozenset[str]) -> None:
    model = _LengthModel(basis)
    assert model.fast
    for m in _matrices():
        seq = synthesize_one_qubit(m, basis)
        assert seq is not None
        assert model(m) == len(seq)


# --- global CX orientation (dual decomposition) ------------------------------


def _xframe_zz_chain(n: int) -> QuantumCircuit:
    """ZZ interactions of a star emitted in the X frame (QAOA-like).

    ``H⊗H · CX(c, t) · RX_c(θ) · CX(c, t) · H⊗H`` per pair: every CX has to be
    reversed together to turn the ``RX`` into target ``RZ`` and cancel the
    Hadamards — a single reversal only adds Hadamards.
    """
    qc = QuantumCircuit()
    for q in range(n):
        qc.h(q)
    for k in range(1, n):
        qc.cx(k, 0).rx(0.3 + 0.1 * k, k).cx(k, 0)
    for k in range(1, n):
        qc.cx(k, 0).rx(0.2 + 0.05 * k, k).cx(k, 0)
    for q in range(n):
        qc.h(q)
    return qc


def test_global_orientation_reverses_whole_region() -> None:
    from quspect.transpiler._gauge import orient_cx_gates

    qc = _xframe_zz_chain(5)
    oriented = orient_cx_gates(qc, IBM_ONE)
    if oriented is qc:
        pytest.skip("native kernels not built")
    assert _distance(_unitary(oriented, 5), _unitary(qc, 5)) < 1e-9
    assert all(i.qubits[0] == 0 for i in oriented if i.gate.name == "cx")
    local = minimize_single_qubit_gates(qc, IBM_ONE)
    glob = minimize_single_qubit_gates(qc, IBM_ONE, global_orientation=True)
    assert _distance(_unitary(glob, 5), _unitary(qc, 5)) < 1e-9
    # every interaction becomes one rz on the hub
    assert _n1q(glob) == 8
    assert _n1q(glob) < _n1q(local)


@pytest.mark.parametrize("ibm", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_global_orientation_equivalent_and_never_longer(ibm: bool, seed: int) -> None:
    basis = IBM_ONE if ibm else CX_ONE
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(40):  # cx-only circuits (cz disables the global step)
        a, b = rng.sample(range(4), 2)
        r = rng.randrange(5)
        if r < 2:
            qc.cx(a, b)
        elif r == 2:
            qc.h(a)
        elif r == 3:
            qc.rx(rng.uniform(-3, 3), a)
        else:
            qc.rz(rng.uniform(-3, 3), a)
    local = minimize_single_qubit_gates(qc, basis)
    out = minimize_single_qubit_gates(qc, basis, global_orientation=True)
    assert _distance(_unitary(out, 4), _unitary(qc, 4)) < 1e-9
    assert _n1q(out) <= _n1q(local)


def test_global_orientation_skips_unsupported_circuits() -> None:
    from quspect.transpiler._gauge import orient_cx_gates

    qc = QuantumCircuit().h(0).cz(0, 1).cx(0, 1).h(1)
    assert orient_cx_gates(qc, IBM_ONE) is qc
    qc = QuantumCircuit().h(0).cx(0, 1)
    assert orient_cx_gates(qc, frozenset({"u3"})) is qc


@pytest.mark.parametrize("seed", range(4))
def test_transpile_orientation_never_deeper(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The globally oriented variant is kept only when it is shallower."""
    import quspect.transpiler._transpile as transpile_module
    from quspect.transpiler import CompilationTarget, transpile

    rng = random.Random(seed)
    qc = QuantumCircuit()
    for q in range(6):
        qc.h(q)
    for _ in range(40):
        a, b = rng.sample(range(6), 2)
        qc.cx(a, b).rz(rng.uniform(-3, 3), b).cx(a, b).rx(rng.uniform(-3, 3), a)
    target = CompilationTarget.from_coupling_map(
        6,
        [(i, j) for i in range(6) for j in range(i + 1, 6)],
        native_gates=set(IBM_ONE | {"cx"}),
    )
    on = transpile(qc, target, optimization_level=2, seed=0)
    monkeypatch.setattr(transpile_module, "_ORIENT_LEVEL", 99)
    off = transpile(qc, target, optimization_level=2, seed=0)
    assert on.stats.two_qubit_gates == off.stats.two_qubit_gates
    assert transpile_module._depth_key(on.wire_circuit) <= transpile_module._depth_key(
        off.wire_circuit
    )
