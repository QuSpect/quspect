"""Behaviour of quspect.pulse beyond bit-exact decoding.

Covers composition rules, pickling, and the inputs that crash (segfault) or hang
the envelopes-qc reference implementation, which raise clean exceptions here.
"""

from __future__ import annotations

import pickle
import struct
from typing import Any

import numpy as np
import pytest


@pytest.fixture()
def pulse() -> Any:
    from quspect import pulse

    return pulse


@pytest.fixture()
def g(pulse: Any) -> Any:
    return pulse.Gaussian(10, 4, 0.7)


@pytest.fixture()
def d(pulse: Any) -> Any:
    return pulse.GaussianDRAG(15, 10, 1.0, 0.5, 0.1, 0.2)


def test_package_reexports_extension(pulse: Any) -> None:
    from quspect import _pulse
    from quspect.pulse import envelopes_compat

    compat = set(envelopes_compat.__all__)
    for name in pulse.__all__:
        source = envelopes_compat if name in compat else _pulse
        assert getattr(pulse, name) is getattr(source, name)
    assert compat <= set(pulse.__all__)


def test_composition_rules(pulse: Any, g: Any, d: Any) -> None:
    assert repr((g * 2) >> 3) == "EnvProd(EnvShift(Gaussian))"
    assert (
        repr(pulse.mix(g + d, 0.1, 0.0, True))
        == "EnvSum(EnvMix(Gaussian), EnvMix(GaussianDRAG))"
    )
    assert type(g * 1j).__name__ == "COMPLEXEnvProd"
    assert (g + 0) is g
    with pytest.raises(ValueError, match="non-zero constant"):
        g + 1.5
    with pytest.raises(TypeError):
        g * "a"


def test_decode_window(pulse: Any, g: Any) -> None:
    t, wave = pulse.decode_envelope(g, pulse.WaveCache(0.5), -5.3, 30.7)
    assert t == -5.3
    assert wave.dtype == np.float32
    assert wave.shape == (72,)
    with pytest.raises(ValueError, match="negative dimensions"):
        pulse.decode_envelope(g, pulse.WaveCache(0.5), 10.0, 5.0)


def test_pickle_roundtrip(pulse: Any, g: Any, d: Any) -> None:
    envs = [
        g,
        d,
        g * 2,
        g * 1j,
        g >> 1.5,
        g + d,
        pulse.mix(g + d, 0.1, 0.0, False),
        pulse.Custom(0.0, 0.3, np.arange(5.0)),
        pulse.Custom(0.0, 0.3, np.arange(5.0) * 1j),
    ]
    for env in envs:
        again = pickle.loads(pickle.dumps(env))
        assert type(again) is type(env)
        assert pulse.serialization(again) == pulse.serialization(env)


# ---- inputs that crash or hang envelopes-qc 0.1.5 ----


def test_empty_sum_raises(pulse: Any, g: Any) -> None:
    empty = pulse.align(g, np.array([]))
    with pytest.raises(RuntimeError, match="EnvSum is empty"):
        empty.start()
    with pytest.raises(RuntimeError, match="EnvSum is empty"):
        pulse.decode_envelope(empty, pulse.WaveCache(0.5))


@pytest.mark.parametrize("cut", [0, 1, 7, 8, 9, 20, 28])
def test_truncated_serialization_raises(pulse: Any, g: Any, d: Any, cut: int) -> None:
    data = pulse.serialization(g + d)
    with pytest.raises(RuntimeError):
        pulse.deserialization(data[:cut])


def test_bad_child_index_raises(pulse: Any) -> None:
    with pytest.raises(RuntimeError):
        pulse.deserialization(struct.pack("<QBdI", 1, 12, 1.0, 5))


@pytest.mark.parametrize("dynamical", [True, False])
def test_mix_of_real_scaled_complex_envelope(
    pulse: Any, d: Any, dynamical: bool
) -> None:
    env = pulse.mix(d * 2, 0.05, 0.3, dynamical)
    assert repr(env) == "EnvProd(EnvMix(GaussianDRAG))"
    expected = pulse.mix(d, 0.05, 0.3, dynamical) * 2
    wc = pulse.WaveCache(0.5)
    t1, w1 = pulse.decode_envelope(env, wc)
    t2, w2 = pulse.decode_envelope(expected, wc)
    assert t1 == t2
    assert np.array_equal(w1, w2)


def test_copy_of_real_scaled_complex_envelope(pulse: Any, g: Any) -> None:
    env = pulse.mix(g, 0.1, 0.0, True) * 2
    assert pulse.serialization(env.copy()) == pulse.serialization(env)


def test_self_inplace_add(g: Any, d: Any) -> None:
    s = g + d
    s += s
    assert repr(s) == "EnvSum(Gaussian, GaussianDRAG, Gaussian, GaussianDRAG)"


def test_none_envelope_raises(pulse: Any, g: Any) -> None:
    with pytest.raises(TypeError):
        g + None
    with pytest.raises(TypeError):
        pulse.mix(None, 0.1, 0.0, True)
