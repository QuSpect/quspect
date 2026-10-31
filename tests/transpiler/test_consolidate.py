"""Tests for ConsolidateBlocksPass (two-qubit block resynthesis)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.ir.compilation import decompose_deep
from quspect.transpiler import CompilationTarget, ConsolidateBlocksPass
from quspect.transpiler._passes import DecomposePass
from tests.synthesis._unitary import circuit_unitary, phase_distance

BASIS = {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}


def _native(circ: QuantumCircuit) -> QuantumCircuit:
    return decompose_deep(circ, supported_gates=BASIS)


def _n2q(circ: QuantumCircuit) -> int:
    return sum(1 for ins in circ.instructions if ins.is_gate and len(ins.qubits) == 2)


def _assert_equivalent(a: QuantumCircuit, b: QuantumCircuit, n: int) -> None:
    assert phase_distance(circuit_unitary(a, n), circuit_unitary(b, n)) < 1e-9


def _heisenberg_bond(theta: float) -> QuantumCircuit:
    """exp(-iθ(XX+YY+ZZ)) as rxx·ryy·rzz written with 6 CX."""
    c = QuantumCircuit()
    # rzz
    c.cx(0, 1).rz(2 * theta, 1).cx(0, 1)
    # rxx
    c.h(0).h(1).cx(0, 1).rz(2 * theta, 1).cx(0, 1).h(0).h(1)
    # ryy
    c.rx(math.pi / 2, 0).rx(math.pi / 2, 1).cx(0, 1).rz(2 * theta, 1).cx(0, 1)
    c.rx(-math.pi / 2, 0).rx(-math.pi / 2, 1)
    return c


# ---------------------------------------------------------------------------
# Micro-benchmark regressions
# ---------------------------------------------------------------------------


def test_heisenberg_bond_three_cz() -> None:
    src = _native(_heisenberg_bond(0.37))
    assert _n2q(src) == 6
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 3
    _assert_equivalent(src, out, 2)


def test_two_adjacent_cp_two_cz() -> None:
    src = _native(QuantumCircuit().cp(0.3, 0, 1).cp(0.9, 1, 0))
    assert _n2q(src) == 4
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 2
    _assert_equivalent(src, out, 2)


def test_random_su4_block_six_cx_to_three() -> None:
    rng = np.random.default_rng(1)
    c = QuantumCircuit()
    for _ in range(6):
        c.u3(*rng.uniform(0, 2 * math.pi, 3), 0)
        c.u3(*rng.uniform(0, 2 * math.pi, 3), 1)
        c.cx(0, 1) if rng.random() < 0.5 else c.cx(1, 0)
    c.u3(*rng.uniform(0, 2 * math.pi, 3), 0)
    src = _native(c)
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 3
    _assert_equivalent(src, out, 2)


def test_swap_absorbs_cp() -> None:
    """SWAP (3 CX) + CP (2 CX) on the same pair → 3 CZ."""
    src = _native(QuantumCircuit().cp(0.4, 0, 1).swap(0, 1))
    assert _n2q(src) == 5
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 3
    _assert_equivalent(src, out, 2)


def test_raw_swap_gate_is_absorbed() -> None:
    src = QuantumCircuit().swap(0, 1).cz(0, 1)
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 2
    _assert_equivalent(src, out, 2)


def test_cancelling_block_removed() -> None:
    src = QuantumCircuit().cx(0, 1).rz(0.2, 0).cx(0, 1).rz(-0.2, 0)
    out = ConsolidateBlocksPass().run(_native(src))
    assert _n2q(out) == 0
    _assert_equivalent(src, out, 2)


# ---------------------------------------------------------------------------
# Block structure / dependency handling
# ---------------------------------------------------------------------------


def test_interleaved_gate_breaks_block() -> None:
    """A gate on (1, 2) between two cx(0, 1) prevents merging them."""
    src = QuantumCircuit().cx(0, 1).cx(1, 2).cx(0, 1)
    out = ConsolidateBlocksPass().run(_native(src))
    assert _n2q(out) == 3
    _assert_equivalent(src, out, 3)


def test_unrelated_gates_do_not_break_block() -> None:
    src = QuantumCircuit().cx(0, 1).cx(2, 3).h(3).cx(1, 0).rz(0.1, 2).cx(0, 1)
    out = ConsolidateBlocksPass().run(_native(src))
    # SWAP-class block on (0, 1) stays 3, single cx on (2, 3) stays 1.
    assert _n2q(out) == 4
    _assert_equivalent(src, out, 4)


def test_measure_and_barrier_are_boundaries() -> None:
    src = QuantumCircuit().cx(0, 1)
    src.barrier(0, 1)
    src.cx(0, 1)
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 2
    src2 = QuantumCircuit().cx(0, 1).cx(0, 1)
    src2.measure([0, 1], [0, 1])
    out2 = ConsolidateBlocksPass().run(src2)
    assert _n2q(out2) == 0
    assert out2.instructions[-1].is_measure


def test_symbolic_parameter_is_opaque() -> None:
    from quspect.circuit.parameter import Parameter

    theta = Parameter("theta")
    src = QuantumCircuit().cx(0, 1).rz(theta, 1).cx(0, 1)
    out = ConsolidateBlocksPass().run(src)
    assert [i.gate.name for i in out.instructions if i.gate] == ["cx", "rz", "cx"]


def test_target_validation() -> None:
    bad = CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates={"cx", "u3"})
    with pytest.raises(ValueError):
        ConsolidateBlocksPass(bad)
    ok = CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates=BASIS)
    ConsolidateBlocksPass(ok)


def test_stats_are_reported() -> None:
    p = ConsolidateBlocksPass()
    p.run(_native(_heisenberg_bond(0.2)))
    assert p.stats.blocks_replaced == 1
    assert p.stats.two_qubit_before == 6
    assert p.stats.two_qubit_after == 3


# ---------------------------------------------------------------------------
# Randomized equivalence
# ---------------------------------------------------------------------------


def _random_circuit(n: int, depth: int, rng: np.random.Generator) -> QuantumCircuit:
    c = QuantumCircuit()
    one_q = ["h", "x", "s", "t", "rz", "rx", "ry"]
    for _ in range(depth):
        r = rng.random()
        if r < 0.45:
            q = int(rng.integers(n))
            g = one_q[int(rng.integers(len(one_q)))]
            if g in ("rz", "rx", "ry"):
                getattr(c, g)(float(rng.uniform(-math.pi, math.pi)), q)
            else:
                getattr(c, g)(q)
        else:
            # bias towards a few pairs so that blocks form
            pairs = [(0, 1), (1, 0), (1, 2), (0, 1), (2, 3), (0, 1)]
            a, b = pairs[int(rng.integers(len(pairs)))]
            a, b = a % n, b % n
            kind = rng.random()
            if kind < 0.5:
                c.cx(a, b)
            elif kind < 0.7:
                c.cz(a, b)
            elif kind < 0.85:
                c.cp(float(rng.uniform(-math.pi, math.pi)), a, b)
            else:
                c.swap(a, b)
    return c


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_equivalent_and_never_worse(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 3 + seed % 2
    src = _random_circuit(n, 60, rng)
    target = CompilationTarget.from_coupling_map(
        n, [(i, i + 1) for i in range(n - 1)], native_gates=BASIS
    )
    native = DecomposePass(target).run(src)
    out = ConsolidateBlocksPass(target).run(native)
    assert _n2q(out) <= _n2q(native)
    names = {i.gate.name for i in out.instructions if i.gate is not None}
    assert names <= BASIS
    _assert_equivalent(native, out, n)
    # Also directly on the non-native circuit (swap / cp / cx kept or absorbed).
    out_raw = ConsolidateBlocksPass().run(src)
    _assert_equivalent(src, out_raw, n)


def test_linear_scaling_large_circuit() -> None:
    """A 12k-gate circuit consolidates quickly (single linear sweep)."""
    import time

    rng = np.random.default_rng(0)
    c = QuantumCircuit()
    for _ in range(2000):
        a = int(rng.integers(19))
        c.cx(a, a + 1).rz(float(rng.uniform()), a + 1).cx(a, a + 1)
        c.h(a).cx(a + 1, a).h(a)
    native = _native(c)
    t0 = time.perf_counter()
    out = ConsolidateBlocksPass().run(native)
    elapsed = time.perf_counter() - t0
    assert _n2q(out) <= _n2q(native)
    assert elapsed < 20.0


# ---------------------------------------------------------------------------
# Tolerance regressions (linear-scale verification, cf. commit 043ae76)
# ---------------------------------------------------------------------------


def test_small_controlled_phase_block_is_kept() -> None:
    """CZ·CZ (cancels) followed by CP(1e-6): 4 CZ → 2 CZ, not 0."""
    src = _native(QuantumCircuit().cz(0, 1).cz(0, 1).cp(1e-6, 0, 1))
    assert _n2q(src) == 4
    out = ConsolidateBlocksPass().run(src)
    assert _n2q(out) == 2
    assert phase_distance(circuit_unitary(src, 2), circuit_unitary(out, 2)) < 1e-12


def test_single_qubit_fusion_verifies_snapped_angles() -> None:
    """Snapped Euler angles of a fused run must stay within the 1e-9 budget.

    ``rz(a)·h·rz(π/2 + b)·h·rz(c)`` with ``a, b, c ≈ 0.9e-9`` is snapped to
    ``rz·h·rz`` (each deviation is within ``atol = 1e-9``). The fused run is
    verified element-wise (worst case found by random search: ~7.8e-10).
    """
    e = 0.9e-9
    src = QuantumCircuit().rz(e, 0).h(0).rz(math.pi / 2 + e, 0).h(0).rz(e, 0)
    out = ConsolidateBlocksPass().run(src)
    assert phase_distance(circuit_unitary(src, 1), circuit_unitary(out, 1)) < 1e-9


# ---------------------------------------------------------------------------
# Approximation (max_infidelity)
# ---------------------------------------------------------------------------


def _weak_circuit() -> QuantumCircuit:
    """Strong gates mixed with weak interactions that a 1e-12 budget may drop."""
    c = QuantumCircuit()
    c.h(0).h(1).h(2)
    c.cp(2e-6, 0, 1).rz(0.3, 1).cp(1e-6, 1, 2)
    c.cx(0, 2).crz(3e-6, 2, 0).crx(5e-7, 1, 0)
    c.cp(0.7, 0, 1).cp(4e-6, 2, 1).rx(0.2, 2).cp(1e-5, 0, 2)
    return c


def test_approximation_off_by_default() -> None:
    circ = _native(_weak_circuit())
    exact = ConsolidateBlocksPass()
    out = exact.run(circ)
    assert exact.stats.blocks_approximated == 0
    assert exact.stats.approximation_error == 0.0
    _assert_equivalent(out, circ, 3)


@pytest.mark.parametrize("eps", [1e-12, 1e-10, 1e-6])
def test_approximation_error_bounds_block_rewrites(eps: float) -> None:
    from quspect.synthesis.two_qubit import approximation_error

    circ = _native(_weak_circuit())
    exact = ConsolidateBlocksPass().run(circ)
    p = ConsolidateBlocksPass(max_infidelity=eps)
    out = p.run(circ)
    assert p.stats.blocks_approximated > 0
    assert p.stats.max_block_infidelity <= eps
    assert _n2q(out) < _n2q(exact)
    real = approximation_error(circuit_unitary(circ, 3), circuit_unitary(out, 3))
    # 逐块误差之和是整条线路误差的上界（再加精确重综合的数值误差）
    assert real.distance <= p.stats.approximation_error + 1e-8
    assert real.infidelity <= p.stats.approximation_error**2 + 1e-12


def test_approximate_gates_replaces_only_weak_gates() -> None:
    from quspect.synthesis.two_qubit import approximation_error

    circ = _weak_circuit()
    p = ConsolidateBlocksPass(max_infidelity=1e-12)
    out = p.approximate_gates(circ)
    names = [i.gate.name for i in out if i.gate is not None]
    # cp(1e-5) 的 θ²/20 超出 1e-12，保留（cp(0.7) 也是）；其余弱门只剩单比特部分
    assert names.count("cp") == 2
    assert "crz" not in names and "crx" not in names
    assert names.count("cx") == 1
    assert p.stats.blocks_approximated == 5
    real = approximation_error(circuit_unitary(circ, 3), circuit_unitary(out, 3))
    assert real.distance <= p.stats.approximation_error + 1e-12
    # 没有预算时原样返回
    assert ConsolidateBlocksPass().approximate_gates(circ) is circ


def test_negative_max_infidelity_rejected() -> None:
    with pytest.raises(ValueError, match="max_infidelity"):
        ConsolidateBlocksPass(max_infidelity=-1e-3)
