"""Control envelopes for superconducting-qubit pulse sequences.

This package builds and decodes time-domain control envelopes (Gaussian, DRAG,
flat-top, rectangular, custom sampled waveforms, ...) with a C++ backend.  The
API is compatible with ``envelopes.envelopes_cpp`` from envelopes-qc 0.1.5,
and decoded waveforms are bit-identical to that implementation.

Envelopes compose with ``+`` (sum), ``*`` (scale by a real or complex
constant) and ``>>`` (time shift); :func:`align` places many copies of an
envelope at once, :func:`mix` applies sideband mixing, and
:func:`decode_envelope` samples an envelope on the grid of a
:class:`WaveCache`.  Times are in the same unit as the cache resolution
(usually ns).

The classes and functions are re-exported from the native extension
``quspect._pulse``; importing this package raises :class:`ImportError` when
the extension has not been built.

Code written against envelopes-qc (``from envelopes import evc``), such as the
pulse-sequence layer of ``lqcloud``, can run on this implementation through
:func:`use_as_envelopes` / :func:`install_as_envelopes`; see
:mod:`quspect.pulse.envelopes_compat`.

Example:
    >>> import numpy as np
    >>> from quspect.pulse import GaussianDRAG, WaveCache, align, decode_envelope
    >>> pi_pulse = GaussianDRAG(t0=15, w=10, amp=1.0, coef=0.5, df=0.1, phase=0.0)
    >>> xy = align(pi_pulse, np.array([0.0, 50.0, 100.0]), np.array([1.0, 0.3, 0.7]))
    >>> t_start, wave = decode_envelope(xy, WaveCache(0.5), -50.0, 250.0)
    >>> t_start, wave.dtype.name, wave.shape
    (-50.0, 'complex64', (600,))
"""

from __future__ import annotations

from quspect._native_loader import BUILD_HINT

try:
    from quspect._pulse import (
        AbstractEnvelope,
        COMPLEXEnvProd,
        CosineDRAG,
        Custom,
        Envelope,
        EnvMix,
        EnvShift,
        EnvSum,
        Flattop,
        Gaussian,
        GaussianDRAG,
        GaussianXPower,
        MixExp,
        PowerSineCosine,
        RealEnvProd,
        Rect,
        RippleRect,
        Triangle,
        WaveCache,
        align,
        decode_envelope,
        deserialization,
        mix,
        serialization,
        split,
    )
except ModuleNotFoundError as exc:
    if exc.name != "quspect._pulse":
        raise
    raise ImportError(
        f"QuSpect pulse extension (quspect._pulse) not found. {BUILD_HINT}"
    ) from exc

from quspect.pulse.envelopes_compat import (
    ENVELOPES_MODULE_NAMES,
    install_as_envelopes,
    make_envelopes_module,
    use_as_envelopes,
)

__all__ = [
    "ENVELOPES_MODULE_NAMES",
    "AbstractEnvelope",
    "COMPLEXEnvProd",
    "CosineDRAG",
    "Custom",
    "EnvMix",
    "EnvShift",
    "EnvSum",
    "Envelope",
    "Flattop",
    "Gaussian",
    "GaussianDRAG",
    "GaussianXPower",
    "MixExp",
    "PowerSineCosine",
    "RealEnvProd",
    "Rect",
    "RippleRect",
    "Triangle",
    "WaveCache",
    "align",
    "decode_envelope",
    "deserialization",
    "install_as_envelopes",
    "make_envelopes_module",
    "mix",
    "serialization",
    "split",
    "use_as_envelopes",
]
