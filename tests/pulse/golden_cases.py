"""Envelope cases whose decoded waveforms are pinned by SHA-256 golden values.

The same functions run against envelopes-qc 0.1.5 (to generate ``golden.json``
with ``tools/pulse_compat/make_golden.py``) and against :mod:`quspect._pulse`
(in ``test_golden.py``).  Each case returns raw bytes: serialization bytes,
decoded start times and decoded sample buffers.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

import numpy as np

Case = Callable[[Any], list[bytes]]


def _decode(
    evc: Any, env: Any, res: float = 0.5, window: tuple[float, float] | None = None
) -> list[bytes]:
    wc = evc.WaveCache(res)
    if window is None:
        t, wave = evc.decode_envelope(env, wc)
    else:
        t, wave = evc.decode_envelope(env, wc, *window)
    return [struct.pack("<d", t), str(wave.dtype).encode(), wave.tobytes()]


def _full(evc: Any, env: Any, **kwargs: Any) -> list[bytes]:
    return [evc.serialization(env), *_decode(evc, env, **kwargs)]


def _leaves(evc: Any) -> dict[str, Any]:
    return {
        "Gaussian": evc.Gaussian(10.3, 4.0, 0.7),
        "GaussianDRAG": evc.GaussianDRAG(15.0, 10.0, 1.0, 0.5, 0.1, 0.2),
        "CosineDRAG": evc.CosineDRAG(15.0, 10.0, 0.8, 0.5, -0.07, 1.2),
        "Triangle": evc.Triangle(1.0, 10.0, 0.5, False),
        "Rect": evc.Rect(1.0, 10.0, 0.5),
        "Flattop": evc.Flattop(1.0, 10.0, 2.0, 3.0, 0.5),
        "RippleRect": evc.RippleRect(1.0, 30.0, 0.5, 2.0, 0.05, -0.04, 0.03, 0.02),
        "PowerSineCosine": evc.PowerSineCosine(1.0, 10.0, 0.5, 2, 3, 1.5, 0.25),
        "GaussianXPower": evc.GaussianXPower(1.0, 10.0, 0.5, 2),
        "CustomReal": evc.Custom(0.3, 0.7, np.linspace(-1.0, 1.0, 17)),
        "CustomComplex": evc.Custom(0.3, 0.7, np.linspace(-1.0, 1.0, 17) * (1 - 0.5j)),
    }


def cases() -> dict[str, Case]:
    """Return the golden cases keyed by name."""
    out: dict[str, Case] = {}
    for name in LEAF_NAMES:
        out[f"leaf/{name}"] = lambda evc, n=name: _full(evc, _leaves(evc)[n])
        out[f"leaf/{name}/shift_res"] = lambda evc, n=name: _full(
            evc, _leaves(evc)[n] >> 0.37, res=0.3
        )
        out[f"leaf/{name}/complex"] = lambda evc, n=name: _full(
            evc, _leaves(evc)[n] * (0.6 - 0.8j)
        )
        out[f"leaf/{name}/window"] = lambda evc, n=name: _full(
            evc, _leaves(evc)[n] >> 1.25, window=(-7.3, 52.6)
        )
    out["mix/dynamical"] = lambda evc: _full(
        evc, evc.mix(_leaves(evc)["Flattop"] >> 3.1, 0.07, 0.4, True)
    )
    out["mix/static"] = lambda evc: _full(
        evc, evc.mix(_leaves(evc)["Gaussian"] >> 3.1, -0.11, 1.4, False)
    )
    out["align/real_amp"] = lambda evc: _full(
        evc,
        evc.align(
            _leaves(evc)["GaussianDRAG"],
            np.arange(5) * 37.5,
            np.linspace(0.2, 1.0, 5).astype(np.float32),
        ),
    )
    out["align/complex_amp"] = lambda evc: _full(
        evc,
        evc.align(
            _leaves(evc)["Gaussian"],
            np.arange(4) * 21.3,
            (np.arange(4) * (1 + 1j)).astype(np.complex64),
        ),
    )
    out["split"] = lambda evc: [
        evc.serialization(block)
        for block in evc.split(
            evc.align(_leaves(evc)["GaussianDRAG"], np.array([0.0, 100.0])),
            np.array([-50.0, 60.0]),
            np.array([50.0, 200.0]),
        )
    ]

    def schedule(evc: Any) -> list[bytes]:
        xy = evc.GaussianDRAG(t0=15, w=10, amp=1.0, coef=0.5, df=0.1, phase=0.0)
        seq = evc.align(
            xy, np.arange(200) * 50.0, np.linspace(0.1, 1, 200).astype(np.float32)
        )
        seq += evc.mix(
            evc.align(evc.Flattop(0, 30, 2, 2, 0.3), np.arange(20) * 500.0),
            0.05,
            0.3,
            True,
        )
        return _full(evc, seq, window=(-50.0, 10250.0))

    out["schedule"] = schedule
    # Same envelope as the lqcloud-style consumer in test_envelopes_compat.py.
    out["compat/readout"] = lambda evc: _full(
        evc,
        evc.mix(
            evc.Gaussian(10.0, w=4.0, amp=0.3)
            + evc.Flattop(t0=20.0, tlen=100.0, w_left=2.0, w_right=2.0, amp=0.2),
            0.05,
            0.0,
            True,
        ),
    )
    return out


LEAF_NAMES = [
    "Gaussian",
    "GaussianDRAG",
    "CosineDRAG",
    "Triangle",
    "Rect",
    "Flattop",
    "RippleRect",
    "PowerSineCosine",
    "GaussianXPower",
    "CustomReal",
    "CustomComplex",
]

ERF_DEPENDENT = frozenset(
    {
        f"leaf/{name}{suffix}"
        for name in ("Flattop", "RippleRect")
        for suffix in ("", "/shift_res", "/complex", "/window")
    }
    | {"mix/dynamical", "compat/readout"}
)
"""Cases whose samples go through the C library's real ``erf`` (Flattop, RippleRect).

``erf`` is not correctly rounded and its last bit differs between libm
implementations (Apple libm, glibc, MSVC UCRT).  The goldens were produced on
macOS arm64, where these cases are bit-identical; on other platforms their
samples are compared against the stored reference arrays within one float32 ulp
instead (see ``test_golden.py``).  Every other case uses only correctly rounded
or platform-identical operations and is bit-identical everywhere.

``schedule`` also contains a Flattop, but its samples were observed to be
bit-identical on every CI platform (Linux x86_64/aarch64, Windows, macOS), so it
stays an exact check rather than storing its ~20k-sample array.
"""
