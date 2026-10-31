"""SWAP-face block templates: synthesis scope and the late rewrite."""

from __future__ import annotations

import numpy as np
import pytest

import quspect.synthesis.two_qubit as tq
from quspect import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler._swap_face import resynthesize_swap_face_blocks

CX_NATIVE = frozenset({"cx", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"})
IBM_NATIVE = frozenset({"cx", "rz", "sx", "x"})
CZ_NATIVE = frozenset({"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"})


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
    return float(np.max(np.abs(a - phase / abs(phase) * b)))


def _swap_zz(theta: float) -> np.ndarray:
    swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
    return swap @ np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))


def test_synthesize_swap_face_block() -> None:
    ops = tq.synthesize_swap_face_block(_swap_zz(0.3), tq.TwoQubitBasis("cx"))
    assert ops is not None
    assert [name for name, _, _ in ops] == ["cx", "rz", "cx", "cx"]
    # not on the SWAP face / fewer entanglers
    cx = np.eye(4)[[0, 1, 3, 2]].astype(complex)
    assert tq.synthesize_swap_face_block(cx, tq.TwoQubitBasis("cx")) is None
    assert (
        tq.synthesize_swap_face_block(_swap_zz(np.pi / 2), tq.TwoQubitBasis()) is None
    )


def test_swap_face_scope_and_cache() -> None:
    u = _swap_zz(0.4)
    basis = tq.TwoQubitBasis("cx")
    generic = tq.synthesize_two_qubit_ops(u, basis=basis)
    with tq.swap_face_templates():
        shaped = tq.synthesize_two_qubit_ops(u, basis=basis)
    assert not tq._SWAP_FACE_TEMPLATE
    assert tq.synthesize_two_qubit_ops(u, basis=basis) == generic  # cache keyed
    assert sum(1 for n, _, _ in shaped if n != "cx") == 1
    assert sum(1 for n, _, _ in generic if n != "cx") > 1


def _haar2(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return np.asarray(np.linalg.qr(z)[0])


@pytest.mark.parametrize("native", [CX_NATIVE, IBM_NATIVE, CZ_NATIVE])
@pytest.mark.parametrize("seed", range(6))
def test_resynthesize_swap_face_blocks(native: frozenset[str], seed: int) -> None:
    rng = np.random.default_rng(seed)
    basis = tq.TwoQubitBasis.from_gates(native)
    assert basis is not None
    qc = QuantumCircuit()
    pairs = [(0, 1), (1, 2), (0, 2), (2, 1), (1, 0)]
    for k in range(6):
        a, b = pairs[(k + seed) % len(pairs)]
        u = np.kron(_haar2(rng), _haar2(rng)) @ _swap_zz(rng.uniform(-3, 3))
        for name, local, params in tq.synthesize_two_qubit_ops(u, basis=basis):
            qs = [(a, b)[q] for q in local]
            if name == "rz":
                qc.rz(params[0], qs[0])
            else:
                getattr(qc, name)(*qs)
    out = resynthesize_swap_face_blocks(qc, native)
    assert out is not qc
    assert _distance(_unitary(out, 3), _unitary(qc, 3)) < 1e-8

    def two(c: QuantumCircuit) -> list[str]:
        return sorted(str(sorted(i.qubits)) for i in c if len(i.qubits) == 2)

    assert two(out) == two(qc)
    assert all(i.gate is not None and i.gate.name in native for i in out)


def test_routed_zz_chain_gets_single_rz_blocks() -> None:
    """A ZZ ring on a line: the SWAP-fused blocks need few single-qubit gates."""
    qc = QuantumCircuit()
    n = 6
    for q in range(n):
        qc.h(q)
    for i in range(n):
        for j in range(i + 1, n):
            qc.cx(i, j).rz(0.1 * (i + j + 1), j).cx(i, j)
    target = CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(CX_NATIVE)
    )
    res = transpile(qc, target, optimization_level=3, seed=0)
    n2 = res.stats.two_qubit_gates
    n1 = sum(1 for i in res.circuit if len(i.qubits) == 1)
    assert n1 <= n2  # generic templates: about two per entangler


@pytest.mark.parametrize("native", [CX_NATIVE, IBM_NATIVE, CZ_NATIVE])
@pytest.mark.parametrize("level", [2, 3])
def test_swap_face_alternatives_never_worse(
    native: frozenset[str], level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quspect.transpiler._transpile as transpile_module

    rng = np.random.default_rng(level)
    qc = QuantumCircuit()
    n = 7
    for q in range(n):
        qc.h(q)
    for _ in range(30):
        a, b = (int(x) for x in rng.choice(n, 2, replace=False))
        qc.cp(float(rng.uniform(0, 3)), a, b).rx(float(rng.uniform(0, 3)), a)
    target = CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(native)
    )
    on = transpile(qc, target, optimization_level=level, seed=1)
    monkeypatch.setattr(transpile_module, "_SWAP_POST", False)
    monkeypatch.setattr(transpile_module, "_SWAP_LATE", False)
    off = transpile(qc, target, optimization_level=level, seed=1)
    key_on = transpile_module._finish_key(on.wire_circuit)
    key_off = transpile_module._finish_key(off.wire_circuit)
    assert key_on <= key_off
    assert _distance(_unitary(on.wire_circuit, n), _unitary(off.wire_circuit, n)) < 1e-7
