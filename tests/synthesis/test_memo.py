"""Tests for the exact-input synthesis caches (:mod:`quspect.synthesis._memo`)."""

from __future__ import annotations

import numpy as np

from quspect.gates import gate_matrix
from quspect.gates.parameterized import U3
from quspect.gates.synthesis import synthesize_one_qubit
from quspect.synthesis._memo import BoundedCache, matrix_key
from quspect.synthesis.two_qubit import num_cz_required, synthesize_two_qubit_ops


def test_bounded_cache_clears_when_full() -> None:
    cache: BoundedCache[int, int] = BoundedCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert len(cache) == 2
    cache.put(3, 3)  # full: cleared first
    assert len(cache) == 1
    assert cache.get(1) is None
    assert cache.get(3) == 3


def test_matrix_key_is_exact() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=complex)
    b = a.copy()
    b[0, 1] = -0.0  # differs only in the sign of a zero
    assert matrix_key(a) == matrix_key(np.eye(2))
    assert matrix_key(a) != matrix_key(b)
    assert matrix_key(a.reshape(4)) != matrix_key(a.reshape(1, 4))  # same bytes


def test_one_qubit_cache_returns_fresh_lists() -> None:
    m = gate_matrix(U3.with_params(0.3, 0.2, 0.1))
    first = synthesize_one_qubit(m, {"h", "rz"})
    assert first is not None
    first.clear()  # caller mutation must not reach the cache
    second = synthesize_one_qubit(m, {"h", "rz"})
    assert second is not None and len(second) > 0
    assert synthesize_one_qubit(np.eye(2), {"h", "rz"}) == []
    assert synthesize_one_qubit(np.eye(2), {"h", "rz"}) == []
    # Not expressible: cached as a miss, still reported as None.
    assert synthesize_one_qubit(m, {"x"}) is None
    assert synthesize_one_qubit(m, {"x"}) is None


def test_two_qubit_cache_matches_fresh_synthesis() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    u, _ = np.linalg.qr(z)
    first = synthesize_two_qubit_ops(u)
    first.append(("cz", (0, 1), ()))
    again = synthesize_two_qubit_ops(u)
    assert again == first[:-1]
    assert num_cz_required(u) == num_cz_required(u.copy()) == 3
