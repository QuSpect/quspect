"""Tests for approximate two-qubit synthesis and its error measures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect.synthesis.two_qubit import (
    TwoQubitBasis,
    _ops_unitary,
    _projected_infidelity,
    approximation_error,
    canonical_gate,
    kak_decompose,
    synthesize_two_qubit_approx,
    synthesize_two_qubit_ops,
)
from tests.synthesis._unitary import random_unitary

BASES = [TwoQubitBasis(), TwoQubitBasis("cx", "zsx"), TwoQubitBasis("cx", "zxz")]


def _cp(theta: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def _n2q(ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]]) -> int:
    return sum(1 for _, q, _ in ops if len(q) == 2)


def _local(rng: np.random.Generator) -> np.ndarray:
    return np.kron(random_unitary(2, rng), random_unitary(2, rng))


def _brute_force_distance(u: np.ndarray, v: np.ndarray) -> float:
    """``min_φ ‖u − e^{iφ}v‖₂`` by bounded scalar minimisation."""
    from scipy.optimize import minimize_scalar

    phases = np.angle(np.linalg.eigvals(u.conj().T @ v))
    centre = -float(np.angle(np.mean(np.exp(1j * phases))))
    res = minimize_scalar(
        lambda phi: float(np.linalg.norm(u - np.exp(1j * phi) * v, 2)),
        bounds=(centre - 0.1, centre + 0.1),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.fun)


# ---------------------------------------------------------------------------
# approximation_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_approximation_error_matches_definitions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    dim = 2 ** (1 + seed % 3)
    u = random_unitary(dim, rng)
    h = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (h + h.conj().T) * 1e-3
    w, vecs = np.linalg.eigh(h)
    v = u @ (vecs @ np.diag(np.exp(1j * w)) @ vecs.conj().T) * np.exp(0.7j)
    err = approximation_error(u, v)
    f_pro = abs(np.trace(u.conj().T @ v)) ** 2 / dim**2
    assert err.infidelity == pytest.approx(dim * (1 - f_pro) / (dim + 1), rel=1e-6)
    # 距离是上界，且（取弧中心相位时）是紧的
    brute = _brute_force_distance(u, v)
    assert err.distance == pytest.approx(brute, rel=1e-5)
    # 线路级保证：过程不保真度 ≤ 距离²
    assert 1 - f_pro <= err.distance**2 + 1e-15


def test_approximation_error_small_angles_accurate() -> None:
    """No cancellation error for infidelities far below double precision's 1e-16·d²."""
    for theta in (1e-4, 1e-6, 1e-8):
        err = approximation_error(_cp(theta), np.eye(4))
        assert err.infidelity == pytest.approx(0.15 * theta**2, rel=1e-6)
        assert err.distance == pytest.approx(theta / 2, rel=1e-6)


def test_approximation_error_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="square"):
        approximation_error(np.eye(2), np.eye(4))


# ---------------------------------------------------------------------------
# synthesize_two_qubit_approx
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", BASES, ids=lambda b: f"{b.entangler}-{b.euler}")
def test_cp_threshold_keeps_local_part(basis: TwoQubitBasis) -> None:
    eps = 1e-12
    limit = math.sqrt(20 * eps)  # CP(θ) → rz(θ/2)⊗rz(θ/2) costs θ²/20
    for theta in (0.5 * limit, 0.99 * limit, 1.01 * limit, 3 * limit):
        u = _cp(theta)
        ops, err = synthesize_two_qubit_approx(u, eps, basis=basis)
        assert {name for name, _, _ in ops} <= basis.gates
        if theta <= limit:
            assert _n2q(ops) == 0
            assert err is not None
            assert err.infidelity == pytest.approx(theta**2 / 20, rel=1e-4)
            assert err.infidelity <= eps
            # 报告的误差就是发出的门的误差
            real = approximation_error(u, _ops_unitary(ops, basis))
            assert real.distance == pytest.approx(err.distance, rel=1e-9)
            # 只丢掉 ZZ 部分：比整体删掉 CP 好约 3 倍
            assert err.infidelity < approximation_error(u, np.eye(4)).infidelity / 2.9
        else:
            assert _n2q(ops) == 2
            assert err is None


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    ("coords", "expected"),
    [
        ((0.5, 0.3, 1e-7), 2),  # c ≈ 0 → 2 CZ
        ((0.5, 0.3, 3e-6), 3),  # c too large for 1e-12
        ((math.pi / 4 - 5e-7, 4e-7, -3e-7), 1),  # CZ-like
        ((math.pi / 4 - 2e-6, 4e-7, -3e-7), 2),  # a too far from π/4
        ((6e-7, 3e-7, 1e-7), 0),  # nearly local
    ],
)
def test_reduced_counts_and_bound(
    seed: int, coords: tuple[float, float, float], expected: int
) -> None:
    rng = np.random.default_rng(seed)
    u = _local(rng) @ canonical_gate(*coords) @ _local(rng)
    eps = 1e-12
    for basis in BASES:
        ops, err = synthesize_two_qubit_approx(u, eps, basis=basis)
        assert _n2q(ops) == expected
        real = approximation_error(u, _ops_unitary(ops, basis))
        assert real.infidelity <= eps
        if err is not None:
            assert real.distance <= err.distance * (1 + 1e-9) + 1e-15
        # 预测（保留局部因子的投影）与实际一致
        if expected < 3:
            v = np.array([kak_decompose(u).a, kak_decompose(u).b, kak_decompose(u).c])
            assert _projected_infidelity(v, expected) == pytest.approx(
                real.infidelity, rel=1e-3, abs=1e-18
            )


def test_zero_budget_is_exact_synthesis() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        u = random_unitary(4, rng)
        ops, err = synthesize_two_qubit_approx(u, 0.0)
        assert err is None
        assert ops == synthesize_two_qubit_ops(u)
    approx_ops, _ = synthesize_two_qubit_approx(_cp(1e-6), 1e-12)
    assert synthesize_two_qubit_ops(_cp(1e-6), max_infidelity=1e-12) == approx_ops


def test_negative_budget_rejected() -> None:
    with pytest.raises(ValueError, match="max_infidelity"):
        synthesize_two_qubit_approx(np.eye(4), -1.0)


def test_max_entanglers_limits_reductions() -> None:
    rng = np.random.default_rng(7)
    u = _local(rng) @ canonical_gate(0.5, 0.3, 1e-7) @ _local(rng)
    ops, err = synthesize_two_qubit_approx(u, 1e-12, max_entanglers=0)
    assert _n2q(ops) == 3 and err is None
    ops, err = synthesize_two_qubit_approx(u, 1e-12, max_entanglers=2)
    assert _n2q(ops) == 2 and err is not None
    ops, err = synthesize_two_qubit_approx(_cp(1e-6), 1e-12, max_entanglers=0)
    assert _n2q(ops) == 0 and err is not None
