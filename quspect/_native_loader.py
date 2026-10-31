"""Locate and load the QuSpect C++ native extension module.

The C++ backend is built as two Python extension modules that expose the same
API:

* ``quspect._core_scalar`` -- portable scalar kernels, loadable on any CPU.
  It is always built and also exports ``cpu_supports_avx2()``.
* ``quspect._core_avx2`` -- AVX2 + FMA kernels (x86_64 only).  Importing it
  on a CPU without AVX2 could crash the interpreter, so it is only imported
  after the scalar module confirms CPU and OS support.

Setting the environment variable ``QUSPECT_DISABLE_AVX2=1`` forces the scalar
module even on AVX2-capable machines.  The variable is read once, on the first
call to :func:`load_core`, because the result is cached for the lifetime of
the process.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
from typing import Any

__all__ = [
    "AVX2_MODULE",
    "BUILD_HINT",
    "DISABLE_AVX2_ENV",
    "SCALAR_MODULE",
    "kernel_backend",
    "load_core",
]

logger = logging.getLogger(__name__)

SCALAR_MODULE = "quspect._core_scalar"
"""Fully qualified name of the always-available scalar extension module."""

AVX2_MODULE = "quspect._core_avx2"
"""Fully qualified name of the optional AVX2 extension module."""

DISABLE_AVX2_ENV = "QUSPECT_DISABLE_AVX2"
"""Environment variable that forces the scalar module when set to ``"1"``."""

BUILD_HINT = (
    "Build the C++ extension with: "
    "cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release && "
    "cmake --build cpp/build"
)
"""Human-readable instructions appended to "native module not found" errors."""


@functools.cache
def load_core() -> Any | None:
    """Return the best native extension module for this CPU, or ``None``.

    This is the supported, stable public way to detect whether the QuSpect
    C++ native core is available; import it as
    ``from quspect.native import load_core``.  A return value of ``None``
    means the native core has not been built (or installed), so native
    features such as :class:`quspect.native.NativeExecutor` are unavailable.
    Any non-``None`` return value is the loaded extension module; its name and
    attributes are implementation details and may change between releases, so
    treat it only as a truthy "native core is available" signal and use the
    public :mod:`quspect.native` API to interact with it.

    The scalar module is imported first.  If it is missing, ``None`` is
    returned so callers can raise a friendly error or skip native features.
    When the scalar module reports AVX2 + FMA support via
    ``cpu_supports_avx2()`` and :data:`DISABLE_AVX2_ENV` is not ``"1"``, the
    AVX2 module is imported instead; if that import fails for any reason the
    scalar module is used and the failure is logged at DEBUG level.

    The result is cached, so the selection happens only once per process.
    Call ``load_core.cache_clear()`` to force a new selection (tests only).

    Returns:
        The loaded extension module (``quspect._core_avx2`` or
        ``quspect._core_scalar``), or ``None`` when the scalar module has not
        been built.

    Raises:
        ImportError: If the scalar module exists but fails to import, or a
            dependency other than the scalar module itself is missing.  Only
            a ``ModuleNotFoundError`` for :data:`SCALAR_MODULE` is treated as
            "not built".

    Example:
        >>> from quspect.native import load_core
        >>> has_native_core = load_core() is not None
        >>> isinstance(has_native_core, bool)
        True
    """
    try:
        scalar = importlib.import_module(SCALAR_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != SCALAR_MODULE:
            raise
        logger.debug("Native module %s is not available", SCALAR_MODULE)
        return None

    if os.environ.get(DISABLE_AVX2_ENV) == "1":
        logger.debug("%s=1: using %s", DISABLE_AVX2_ENV, SCALAR_MODULE)
        return scalar

    if not bool(scalar.cpu_supports_avx2()):
        logger.debug("CPU lacks AVX2/FMA support: using %s", SCALAR_MODULE)
        return scalar

    try:
        avx2 = importlib.import_module(AVX2_MODULE)
    except ImportError as exc:
        logger.debug(
            "Could not import %s (%s); falling back to %s",
            AVX2_MODULE,
            exc,
            SCALAR_MODULE,
        )
        return scalar

    logger.debug("Using %s", AVX2_MODULE)
    return avx2


def kernel_backend() -> str | None:
    """Return the name of the kernel backend used by the loaded native module.

    This reports the double-precision dispatch table actually selected inside
    the extension (``"avx2"`` or ``"scalar"``), which is the ground truth for
    which kernels execute.

    Returns:
        The backend name, or ``None`` when no native module is available.

    Raises:
        ImportError: Propagated from :func:`load_core`.

    Example:
        >>> from quspect._native_loader import kernel_backend
        >>> kernel_backend() in (None, "scalar", "avx2")
        True
    """
    core = load_core()
    if core is None:
        return None
    return str(core.get_runtime_capabilities()["backend_f64"])
