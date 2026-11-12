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
    # ``align`` is the one function wrapped in Python (complex amplitudes of
    # any precision); everything else is the extension's own object.
    wrapped = {"align"}
    for name in pulse.__all__:
        if name in wrapped:
            assert getattr(pulse, name) is not getattr(_pulse, name)
            continue
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


# ---- behaviour shared with envelopes-qc 0.1.5 that is easy to trip over ----


def _sample(pulse: Any, env: Any, t: float) -> complex:
    t0, wave = pulse.decode_envelope(env, pulse.WaveCache(1.0), t - 5.0, t + 5.0)
    return complex(wave[round(t - t0)])


def _align_cases() -> list[tuple[str, Any, Any]]:
    """``(label, amplitudes, the same amplitudes as the extension takes them)``."""
    real = [1.0, 0.25]
    mixed = [1.0, 1j]
    return [
        ("no-amp", None, None),
        ("float32", np.array(real, np.float32), np.array(real, np.float32)),
        ("float64", np.array(real, np.float64), np.array(real, np.float64)),
        ("int", np.array([1, 2]), np.array([1, 2])),
        ("float-list", real, real),
        ("complex64", np.array(mixed, np.complex64), np.array(mixed, np.complex64)),
        ("complex-list", mixed, mixed),
        ("complex128", np.array(mixed, np.complex128), np.array(mixed, np.complex64)),
        (
            "complex128-scalars",
            [np.complex128(1.0), np.complex128(1j)],
            np.array(mixed, np.complex64),
        ),
        (
            "complex128-real-valued",
            np.array(real, np.complex128),
            np.array(real, np.complex64),
        ),
    ]


def test_align_keeps_complex128_amplitudes(pulse: Any) -> None:
    """Known issue: complex128 amplitudes lost their imaginary part.

    envelopes-qc (and ``quspect._pulse.align``) match a complex128 array to
    the float32 overload and only emit a ComplexWarning.
    """
    import warnings

    drag = pulse.GaussianDRAG(0, 10, 0.5, 0.5, 0, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = pulse.align(
            drag, np.array([0.0, 40.0]), np.array([1, 1j], dtype=np.complex128)
        )
    assert caught == []
    assert _sample(pulse, env, 40.0) == pytest.approx(0.5j)
    assert _sample(pulse, env, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("amp", "native_amp"),
    [case[1:] for case in _align_cases()],
    ids=[case[0] for case in _align_cases()],
)
def test_align_is_the_extension_align_for_every_amplitude_type(
    pulse: Any, d: Any, amp: Any, native_amp: Any
) -> None:
    """The wrapper changes nothing but the precision of complex amplitudes."""
    from quspect import _pulse

    dt = np.array([0.0, 40.0])
    if amp is None:
        env, expected = pulse.align(d, dt), _pulse.align(d, dt)
    else:
        env, expected = pulse.align(d, dt, amp), _pulse.align(d, dt, native_amp)
    assert pulse.serialization(env) == pulse.serialization(expected)
    wc = pulse.WaveCache(0.5)
    t1, w1 = pulse.decode_envelope(env, wc)
    t2, w2 = pulse.decode_envelope(expected, wc)
    assert t1 == t2
    assert w1.dtype == w2.dtype
    assert w1.tobytes() == w2.tobytes()


def test_align_keyword_and_error_behaviour(pulse: Any, g: Any) -> None:
    env = pulse.align(env=g, dt=np.array([0.0, 8.0]), amp=np.array([1.0, 2.0]))
    assert repr(env).startswith("EnvSum(")
    with pytest.raises(TypeError):
        pulse.align(None, np.array([0.0]))
    with pytest.raises(TypeError):
        pulse.align(g, np.array([0.0]), "a")


@pytest.mark.parametrize("df", [0.13, -0.02])
def test_static_mix_travels_with_later_shifts_only(pulse: Any, df: float) -> None:
    """``mix(..., dynamical=False)`` and ``>>`` do not commute (as upstream).

    The carrier of a static mix has zero phase at ``t = 0`` when ``mix`` is
    called and then moves with every shift applied afterwards.  A shift
    already inside the envelope does not move it.
    """

    def carrier(t: float) -> complex:
        return complex(np.exp(-2j * np.pi * df * t))

    def g() -> Any:
        return pulse.Gaussian(0, 4, 1)

    mixed_then_shifted = pulse.mix(g(), df, 0.0, False) >> 50
    shifted_then_mixed = pulse.mix(g() >> 50, df, 0.0, False)
    assert _sample(pulse, mixed_then_shifted, 50.0) == pytest.approx(1.0, abs=1e-5)
    assert _sample(pulse, shifted_then_mixed, 50.0) == pytest.approx(
        carrier(50.0), abs=1e-5
    )
    both = pulse.mix(g() >> 30, df, 0.0, False) >> 50
    assert _sample(pulse, both, 80.0) == pytest.approx(carrier(30.0), abs=1e-5)
    # align() shifts after the mix: every copy starts at zero phase.
    copies = pulse.align(pulse.mix(g(), df, 0.0, False), np.array([0.0, 50.0]))
    assert _sample(pulse, copies, 50.0) == pytest.approx(1.0, abs=1e-5)
    # A dynamical mix keeps the absolute carrier in either order.
    for env in (
        pulse.mix(g(), df, 0.0, True) >> 50,
        pulse.mix(g() >> 50, df, 0.0, True),
        pulse.align(pulse.mix(g(), df, 0.0, True), np.array([0.0, 50.0])),
    ):
        assert _sample(pulse, env, 50.0) == pytest.approx(carrier(50.0), abs=1e-5)


def test_decoded_array_is_its_own_buffer(pulse: Any, g: Any) -> None:
    """Each decode returns a new array; the cache is never handed out.

    The docstring kept from envelopes-qc says "zero-copy view of the internal
    buffer": the array wraps the buffer of that one result without a copy.
    """
    wc = pulse.WaveCache(1.0)
    _, first = pulse.decode_envelope(g, wc, 0.0, 20.0)
    _, second = pulse.decode_envelope(g, wc, 0.0, 20.0)
    assert not np.shares_memory(first, second)
    assert first.flags.writeable
    assert not first.flags.owndata  # wraps the C++ result buffer, no copy
    expected = first.copy()
    first[:] = 7.0
    _, third = pulse.decode_envelope(g, wc, 0.0, 20.0)
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(third, expected)
    _, whole_a = pulse.decode_envelope(g, wc)
    _, whole_b = pulse.decode_envelope(g, wc)
    assert not np.shares_memory(whole_a, whole_b)
