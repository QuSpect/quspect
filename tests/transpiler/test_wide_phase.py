"""The level-3 "wide-phase" starting point of :func:`quspect.transpiler.transpile`.

Every gate, Toffolis included, is lowered to the CX level, the rotations are
folded and the CNOT-phase regions are resynthesized on all qubit pairs before
routing (:func:`quspect.transpiler._transpile._wide_phase_regions`).  Level 3
compiles from this starting point as well and keeps the better result.
"""

from __future__ import annotations

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler import _transpile as transpile_module
from quspect.transpiler._transpile import _wide_phase_regions

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}
WORKING = frozenset({"cz", "h", "rz", "x"})


def _line(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=set(BASIS)
    )


def _two_qubit(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if i.is_gate and len(i.qubits) == 2)


def _unitary(circuit: QuantumCircuit, n: int) -> np.ndarray:
    """Big-endian ``2^n x 2^n`` unitary of ``circuit``."""
    u = np.eye(2**n, dtype=complex).reshape([2] * n + [2**n])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
        u = np.tensordot(g, u, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        u = np.moveaxis(u, list(range(k)), list(ins.qubits))
    return u.reshape(2**n, 2**n)


def _assert_same_up_to_phase(a: np.ndarray, b: np.ndarray) -> None:
    k = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[k] / a[k]
    assert abs(abs(phase) - 1.0) < 1e-9
    assert np.max(np.abs(a * phase - b)) < 1e-9


def _embed(state: np.ndarray, layout: list[int], num_physical: int) -> np.ndarray:
    n = len(layout)
    free = [p for p in range(num_physical) if p not in layout]
    zeros = np.zeros(2 ** (num_physical - n), dtype=complex)
    zeros[0] = 1.0
    tensor = np.kron(state, zeros).reshape([2] * num_physical)
    order = [
        layout.index(p) if p in layout else n + free.index(p)
        for p in range(num_physical)
    ]
    return np.transpose(tensor, order).reshape(-1)


def _v_chain() -> QuantumCircuit:
    """A four-controlled X on qubit 6 through two clean ancillas, 4 and 5: the
    Toffolis on the ancillas are computed and uncomputed, so their relative
    phases cancel once they are lowered into one CNOT-phase region."""
    return QuantumCircuit().ccx(0, 1, 4).ccx(2, 4, 5).cx(5, 6).ccx(2, 4, 5).ccx(0, 1, 4)


def test_regions_of_a_toffoli_v_chain_shrink_and_stay_exact() -> None:
    from quspect.ir.compilation import decompose_deep

    circuit = _v_chain()
    lowered = decompose_deep(circuit, supported_gates={"cx", "h", "rz", "x"})
    wide = _wide_phase_regions(circuit, WORKING)
    assert wide is not None
    assert _two_qubit(wide) < _two_qubit(lowered)
    assert all(len(i.qubits) <= 2 for i in wide if i.is_gate)
    _assert_same_up_to_phase(_unitary(circuit, 7), _unitary(wide, 7))


def test_nothing_to_gain_returns_none() -> None:
    # 一串互不相消的 CX：没有相位可折叠，区域也已经最短
    assert (
        _wide_phase_regions(QuantumCircuit().cx(0, 1).cx(1, 2).cx(2, 3), WORKING)
        is None
    )
    assert _wide_phase_regions(QuantumCircuit().h(0).cx(0, 1), WORKING) is None


def test_large_circuits_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transpile_module, "_PHASE_WIDE_MAX_2Q", 10)
    assert _wide_phase_regions(_v_chain(), WORKING) is None


def test_level3_keeps_the_better_start_and_stays_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit, target = _v_chain(), _line(7)
    new = transpile(circuit, target, optimization_level=3, seed=0)
    monkeypatch.setattr(transpile_module, "_PHASE_WIDE", False)
    old = transpile(circuit, target, optimization_level=3, seed=0)
    assert new.stats.two_qubit_gates < old.stats.two_qubit_gates
    for ins in new.circuit:
        assert ins.gate is not None and ins.gate.name in BASIS
        if len(ins.qubits) == 2:
            assert tuple(sorted(ins.qubits)) in target.coupling_graph
    rng = np.random.default_rng(1)
    psi = rng.normal(size=2**7) + 1j * rng.normal(size=2**7)
    psi /= np.linalg.norm(psi)
    expected = _unitary(circuit, 7) @ psi
    got = _unitary(new.circuit, 7) @ _embed(psi, new.initial_layout, 7)
    assert abs(abs(np.vdot(_embed(expected, new.final_layout, 7), got)) - 1.0) < 1e-9


def test_wide_start_is_level3_only_and_skipped_with_a_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = transpile_module._wide_phase_regions

    def spy(circuit: QuantumCircuit, native: frozenset[str]) -> QuantumCircuit | None:
        calls.append(True)
        return original(circuit, native)

    monkeypatch.setattr(transpile_module, "_wide_phase_regions", spy)
    transpile(_v_chain(), _line(7), optimization_level=2, seed=0)
    transpile(_v_chain(), _line(7), optimization_level=3, seed=0, time_limit=60.0)
    assert calls == []
    transpile(_v_chain(), _line(7), optimization_level=3, seed=0)
    assert calls


def test_wide_start_labels_its_result(monkeypatch: pytest.MonkeyPatch) -> None:
    # 再编译（repass）与最后的三比特块抛光会换掉标签；关掉它们，胜出的结果带前缀
    monkeypatch.setattr(transpile_module, "_REPASS", False)
    monkeypatch.setattr(transpile_module, "_NUMERIC_BLOCKS", False)
    res = transpile(_v_chain(), _line(7), optimization_level=3, seed=0)
    assert res.stats.selected.startswith("wide-phase/")
