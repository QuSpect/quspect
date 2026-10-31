Pulse envelopes
===============

:mod:`quspect.pulse` builds and decodes time-domain control envelopes for
superconducting-qubit experiments: DRAG-shaped drive pulses, flat-top
readout and flux pulses, arbitrary sampled waveforms, and whole pulse
schedules made from them.

The module is a C++ reimplementation of the extension module of
`envelopes-qc <https://pypi.org/project/envelopes-qc/>`_ 0.1.5.  It keeps the
same classes, function signatures, docstrings and behaviour, and decoded
waveforms are **bit-identical** to the original: every ``float32`` sample,
the serialization bytes, ``repr`` strings and error messages match.  Unlike
the original, it is built from source on every supported platform, including
Linux on aarch64.

The classes and functions are re-exported from the native extension
``quspect._pulse``.  Importing :mod:`quspect.pulse` raises :class:`ImportError`
when the extension has not been built (see :doc:`installation`).

Quick example
-------------

.. code-block:: python

   import numpy as np
   from quspect.pulse import GaussianDRAG, WaveCache, align, decode_envelope

   # A DRAG pi pulse centred at 15 ns with a 10 ns FWHM.
   pi_pulse = GaussianDRAG(t0=15, w=10, amp=1.0, coef=0.5, df=0.1, phase=0.0)

   # Three copies at 0, 50 and 100 ns with individual amplitudes.
   xy = align(pi_pulse, np.array([0.0, 50.0, 100.0]),
              np.array([1.0, 0.5, 1.0], dtype=np.float32))

   # Sample on a 0.5 ns grid inside the window [-50, 250).
   t_start, wave = decode_envelope(xy, WaveCache(0.5), -50.0, 250.0)
   print(t_start, wave.dtype, wave.shape)   # -50.0 complex64 (600,)

Times are in the unit of the cache resolution (usually ns); frequencies are in
the reciprocal unit (GHz for ns).

Envelope types
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Class
     - Constructor
     - Shape
   * - ``Gaussian``
     - ``(t0, w, amp)``
     - Gaussian centred at ``t0`` with full width at half maximum ``w``,
       truncated to ``t0 ± 3w``.
   * - ``GaussianDRAG``
     - ``(t0, w, amp, coef, df, phase)``
     - Gaussian plus ``i·coef`` times its derivative (DRAG), modulated by
       ``exp(i(-2π·df·t + phase))``.  Complex.
   * - ``CosineDRAG``
     - ``(t0, w, amp, coef, df, phase)``
     - Raised cosine of full width ``w`` with DRAG correction and modulation.
       Complex.
   * - ``Triangle``
     - ``(t0, tlen, amp, fall)``
     - Linear ramp over ``[t0, t0 + tlen)``, falling or rising.
   * - ``Rect``
     - ``(t0, tlen, amp)``
     - Rectangle over ``[t0, t0 + tlen)``.
   * - ``Flattop``
     - ``(t0, tlen, w_left, w_right, amp)``
     - Rectangle whose edges are convolved with Gaussian kernels.
   * - ``RippleRect``
     - ``(t0, tlen, amp, w, ripple0, ..., ripple3)``
     - Gaussian-smoothed rectangle with four sinusoidal ripple terms.
   * - ``PowerSineCosine``
     - ``(t0, tlen, amp, m, n, k, phase)``
     - ``sin(kπt/T + phase)^m · cos(kπt/T + phase)^n``.
   * - ``GaussianXPower``
     - ``(t0, w, amp, m)``
     - Gaussian multiplied by ``x^m``.
   * - ``Custom``
     - ``(start, dt, data)``
     - Arbitrary real or complex samples spaced by ``dt``, linearly
       interpolated onto the decoding grid.

``amp`` is stored as ``float32``.  Every envelope exposes ``start()``,
``end()``, ``is_complex()`` and ``copy()``.

Composing envelopes
-------------------

Envelopes form expression trees:

* ``a + b`` builds an ``EnvSum``; adding ``0`` returns the envelope itself,
  so the built-in :func:`sum` works.  ``s += env`` appends to an existing
  ``EnvSum`` in place.
* ``env * c`` scales by a real (``RealEnvProd``) or complex
  (``COMPLEXEnvProd``) constant stored in single precision.
* ``env >> dt`` shifts in time (``EnvShift``).  Shifts and scales are pushed
  through sums, so ``(a + b) >> dt`` is ``(a >> dt) + (b >> dt)``.
* ``mix(env, df, phase, dynamical)`` applies sideband mixing.  With
  ``dynamical=True`` the carrier phase is zero at ``t = 0``; with
  ``dynamical=False`` it is zero at the shifted start of each copy.
* ``align(env, dt, amp=None)`` places copies of ``env`` at the offsets
  ``dt``, optionally scaled by real (``float32``) or complex (``complex64``)
  amplitudes.  It is much faster than summing shifted copies in Python.
* ``split(env, starts, ends)`` distributes the terms of a sum into blocks.
  Each term must lie completely inside a block; empty blocks get a
  zero-length ``Rect``.

.. code-block:: python

   import numpy as np
   from quspect.pulse import Flattop, GaussianDRAG, align, mix, split

   pi_pulse = GaussianDRAG(t0=15, w=10, amp=1.0, coef=0.5, df=0.1, phase=0.0)
   readout = mix(Flattop(0, 200, 2, 2, 0.3), 0.05, 0.0, True)

   seq = align(pi_pulse, np.array([0.0, 200.0]))
   blocks = split(seq, np.array([-20.0, 150.0]), np.array([100.0, 300.0]))
   print(blocks)   # [EnvSum(EnvShift(GaussianDRAG)), EnvSum(EnvShift(GaussianDRAG))]

Decoding
--------

``decode_envelope(env, wc)`` samples an envelope and returns
``(start_time, wave)``, where ``wave`` is a ``float32`` array for real
envelopes and ``complex64`` for complex ones.  Samples lie on the absolute
grid ``n · resolution`` of the :class:`~quspect.pulse.WaveCache`, and the
array is a fresh buffer that may be modified in place.

``decode_envelope(env, wc, start, end)`` crops or zero-pads the result to the
window ``[start, end)``.  The returned start time is ``start`` rounded to
``1e-5``, **not** snapped to the sample grid; the first sample is the grid
point nearest to it.  A ``start`` after the envelope's last sample or an
``end`` before its first sample raises :class:`RuntimeError`.

A :class:`~quspect.pulse.WaveCache` stores the base waveform of each envelope
shape per sub-grid offset, so decoding many copies of the same pulse computes
the shape only once.  Reuse one cache across decodes with the same
resolution.

Serialization
-------------

``serialization(env)`` returns a compact binary encoding of the expression
tree and ``deserialization(data)`` restores it; shared sub-trees are stored
once.  The format is byte-compatible with envelopes-qc 0.1.5.  All envelope
classes support :mod:`pickle` through the same encoding.

Differences from envelopes-qc
-----------------------------

* Import from :mod:`quspect.pulse` instead of ``envelopes.evc``.  The class
  names and call signatures are unchanged.
* The pure-Python reference implementation ``envelopes.evp`` is not
  included.
* Inputs that crash or hang envelopes-qc 0.1.5 raise Python exceptions
  instead, or work as expected:

  - ``start()``, ``end()`` or decoding of an empty ``EnvSum`` (for example
    ``align(env, [])``) raises :class:`RuntimeError`;
  - ``mix()`` and ``copy()`` of a real-scaled complex envelope, such as
    ``mix(GaussianDRAG(...) * 2, ...)``, return the expected result;
  - ``s += s`` on an ``EnvSum`` appends its own terms;
  - passing ``None`` as an envelope raises :class:`TypeError`;
  - truncated or corrupted serialized data raises :class:`RuntimeError`.

Compatibility is checked by ``tools/pulse_compat``, which runs the same cases
against envelopes-qc and :mod:`quspect.pulse` and compares the results byte
for byte, and by golden values in ``tests/pulse``.

Output is bit-identical to envelopes-qc on macOS arm64.  On other platforms
everything is identical as well, except the samples of envelopes built on the
real error function (:class:`~quspect.pulse.Flattop` and
:class:`~quspect.pulse.RippleRect`): ``erf`` comes from the platform C
library, whose last bit differs between implementations, so these samples may
differ from envelopes-qc on macOS by about one ``float32`` ulp.  Their
serialization bytes are identical on every platform.

Using quspect.pulse in place of envelopes-qc
--------------------------------------------

Code written against envelopes-qc imports ``from envelopes import evc``.  The
pulse-sequence layer of the ``lqcloud`` SDK is an example.
:func:`quspect.pulse.use_as_envelopes` temporarily registers a stand-in
``envelopes`` package whose ``evc`` is :mod:`quspect.pulse`'s native module,
so such code runs on this implementation without envelopes-qc installed:

.. code-block:: python

   from quspect.pulse import GaussianDRAG, use_as_envelopes

   with use_as_envelopes(purge=("lqcloud",)):
       from lqcloud.sequence import EnvelopeAligner

       aligner = EnvelopeAligner(
           GaussianDRAG(t0=0, w=10, amp=1.0, coef=0.1, df=0.0, phase=0.0)
       )

For a whole application, call :func:`quspect.pulse.install_as_envelopes`
once at start-up instead.  In tests, wrap the context manager in a fixture;
``tests/pulse/conftest.py`` provides ``envelopes_shim``.

This is safe because decoded waveforms and serialization bytes are
bit-identical to envelopes-qc 0.1.5.  Envelopes built here and serialized by
``lqcloud`` are read correctly by services that deserialize them with
envelopes-qc.

Keep in mind:

* The stand-in must be active **before** the consumer imports its envelope
  modules, because they bind ``evc`` at import time.  Cached modules named in
  ``purge`` (``lqcloud`` by default) are dropped on entry so they are
  re-imported against the stand-in.  When the block exits, everything in
  :data:`sys.modules` is restored.
* Objects from the two implementations are different C++ types and cannot be
  mixed.  Pass only :mod:`quspect.pulse` envelopes to code imported under the
  stand-in.
* ``envelopes.evp`` (the pure-Python implementation) is not provided.
* ``lqcloud`` declares envelopes-qc as a dependency, and envelopes-qc has no
  linux/aarch64 wheel.  On that platform install it with
  ``pip install --no-deps lqcloud`` plus its other dependencies.
