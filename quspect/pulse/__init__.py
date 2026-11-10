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
the extension has not been built.  :func:`align` is the one exception: it is
a thin Python function around the extension's ``align`` that keeps complex
amplitudes of any precision.

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

import numpy as np
from numpy.typing import ArrayLike

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
        decode_envelope,
        deserialization,
        mix,
        serialization,
        split,
    )
    from quspect._pulse import align as _native_align
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


def align(
    env: AbstractEnvelope, dt: ArrayLike, amp: ArrayLike | None = None
) -> AbstractEnvelope:
    """Place copies of ``env`` at the time shifts ``dt``.

    Without ``amp`` every copy keeps the envelope's amplitude.  Real
    amplitudes are stored as ``float32`` and complex amplitudes as
    ``complex64``; the result is the ``EnvSum`` of the shifted, scaled
    copies and is much faster to build than a Python loop.

    This is ``quspect._pulse.align`` with one difference.  The extension,
    like envelopes-qc, chooses between its real and complex versions by the
    exact array type: a ``complex128`` array (numpy's default complex type)
    matches neither, is converted to ``float32`` with only a
    ``ComplexWarning``, and loses its imaginary part.  Here complex
    amplitudes of any precision are converted to ``complex64`` first, so
    they reach the complex version.  All other inputs are passed through
    unchanged and give bit-identical results.

    Args:
        env: The envelope to copy.
        dt: Time shift of each copy.
        amp: Optional amplitude of each copy, real or complex, as many as
            ``dt``.

    Returns:
        The sum of the copies.

    Raises:
        TypeError: If an argument cannot be converted, as raised by the
            extension.

    Example:
        >>> import numpy as np
        >>> from quspect.pulse import Gaussian, WaveCache, align, decode_envelope
        >>> pair = align(Gaussian(0, 4, 1), np.array([0.0, 40.0]), np.array([1, 1j]))
        >>> t_start, wave = decode_envelope(pair, WaveCache(1.0), 35.0, 45.0)
        >>> complex(wave[5])
        1j
    """
    if amp is None:
        return _native_align(env, dt)
    try:
        is_complex = np.asarray(amp).dtype.kind == "c"
    except Exception:  # not an array-like: the extension reports it
        is_complex = False
    if is_complex:
        amp = np.asarray(amp).astype(np.complex64, copy=False)
    return _native_align(env, dt, amp)


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
