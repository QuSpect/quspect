"""Tests for :mod:`quspect.simulation.gate_applicator` input validation."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.simulation.gate_applicator import apply_gate, apply_gate_out_of_place

X = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def test_apply_gate_updates_complex128_in_place() -> None:
    """A valid complex128 state is updated in place."""
    state = np.array([1, 0], dtype=np.complex128)
    apply_gate(state, X, (0,))
    np.testing.assert_allclose(state, [0, 1])


@pytest.mark.parametrize("dim", [0, 3, 6])
def test_apply_gate_rejects_non_power_of_two_state(dim: int) -> None:
    """B73: a state length that is not a power of two raises ValueError."""
    state = np.zeros(dim, dtype=np.complex128)
    with pytest.raises(ValueError, match="power of two"):
        apply_gate(state, X, (0,))


@pytest.mark.parametrize("dtype", [np.complex64, np.float64])
def test_apply_gate_rejects_wrong_dtype(dtype: type) -> None:
    """B74: non-complex128 arrays would be silently copied, so they raise."""
    state = np.zeros(2, dtype=dtype)
    state[0] = 1
    with pytest.raises(TypeError, match="complex128"):
        apply_gate(state, X, (0,))


def test_apply_gate_rejects_non_contiguous_state() -> None:
    """B74: strided views would be silently copied, so they raise."""
    state = np.zeros(4, dtype=np.complex128)[::2]
    state[0] = 1
    with pytest.raises(ValueError, match="C-contiguous"):
        apply_gate(state, X, (0,))


def test_apply_gate_rejects_read_only_state() -> None:
    """B74: read-only arrays cannot be updated in place."""
    state = np.array([1, 0], dtype=np.complex128)
    state.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        apply_gate(state, X, (0,))


def test_apply_gate_rejects_2d_state() -> None:
    """The state must be one-dimensional."""
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_gate(np.eye(2, dtype=np.complex128), X, (0,))


@pytest.mark.parametrize("kind", ["complex64", "float64", "strided"])
def test_apply_gate_out_of_place_accepts_any_layout(kind: str) -> None:
    """Out-of-place application converts the input and leaves it unchanged."""
    if kind == "strided":
        state = np.zeros(4, dtype=np.complex128)[::2]
    else:
        state = np.zeros(2, dtype=kind)
    state[0] = 1
    result = apply_gate_out_of_place(state, X, (0,))
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result, [0, 1])
    np.testing.assert_allclose(state, [1, 0])
