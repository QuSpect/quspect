"""Serve :mod:`quspect.pulse` under the import name of envelopes-qc.

Third-party code written against envelopes-qc (for example the pulse-sequence
layer of the ``lqcloud`` SDK) imports the C++ module as ``from envelopes import
evc``.  The helpers here register a stand-in ``envelopes`` package in
:data:`sys.modules` whose ``evc`` / ``envelopes_cpp`` attribute is
:mod:`quspect._pulse`, so that code transparently runs on QuSpect's
implementation -- in tests, or on platforms where envelopes-qc has no wheel
(such as linux/aarch64).

This is safe because :mod:`quspect._pulse` is bit-identical to envelopes-qc
0.1.5: decoded waveforms *and* serialization bytes match, so envelopes built
here and serialized by a consumer are understood by services that deserialize
them with envelopes-qc.

Two rules apply:

* The stand-in must be active **before** the consumer imports its envelope
  modules, because those modules bind ``evc`` at import time.  Cached consumer
  modules are therefore purged (``purge=("lqcloud",)`` by default) so that they
  are re-imported against the stand-in.
* Objects from the two implementations are different C++ types and cannot be
  mixed: pass only :mod:`quspect.pulse` envelopes to a consumer imported under
  the stand-in.

The stand-in does not provide envelopes-qc's pure-Python ``evp`` module.

Example:
    A reusable pytest fixture that runs a test against the stand-in:

    .. code-block:: python

        import pytest

        from quspect.pulse import use_as_envelopes


        @pytest.fixture
        def envelopes_shim():
            with use_as_envelopes(purge=("lqcloud",)) as shim:
                yield shim


        def test_sequence(envelopes_shim):
            from lqcloud.sequence import EnvelopeAligner  # imported under the shim

            ...
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import numpy as np

import quspect._pulse as _pulse

__all__ = [
    "ENVELOPES_MODULE_NAMES",
    "install_as_envelopes",
    "make_envelopes_module",
    "use_as_envelopes",
]

ENVELOPES_MODULE_NAMES = ("envelopes", "envelopes.envelopes_cpp", "envelopes.evc")
"""Module names registered in :data:`sys.modules` by the stand-in."""

_DEFAULT_PURGE = ("lqcloud",)
_SHIM_MARKER = "__quspect_envelopes_shim__"
_MISSING = object()

_EVP_MESSAGE = (
    "envelopes.evp (the pure-Python implementation of envelopes-qc) is not "
    "provided by quspect.pulse; use envelopes.evc"
)


class _EnvelopesModule(types.ModuleType):
    """Module type of the stand-in, with a helpful error for ``evp``."""

    def __getattr__(self, name: str) -> object:
        if name == "evp":
            raise AttributeError(_EVP_MESSAGE)
        raise AttributeError(f"module 'envelopes' has no attribute {name!r}")


def make_envelopes_module() -> types.ModuleType:
    """Create a stand-in for the ``envelopes`` package of envelopes-qc.

    The returned module exposes the same names consumers import from
    envelopes-qc: ``evc`` and ``envelopes_cpp`` (both :mod:`quspect._pulse`),
    ``EVC_COMPLEX_DTYPE`` (:class:`numpy.complex64`) and ``EVC_FLOAT_DTYPE``
    (:class:`numpy.float32`).  It is marked as a package (empty ``__path__``)
    so ``import envelopes.envelopes_cpp`` resolves once the submodule names
    are registered in :data:`sys.modules`.  It is not registered anywhere;
    use :func:`use_as_envelopes` or :func:`install_as_envelopes` for that.

    Returns:
        A new module object named ``"envelopes"``.  Accessing ``evp`` on it
        raises :class:`AttributeError`, because the pure-Python
        implementation is not provided.

    Example:
        >>> import quspect._pulse
        >>> from quspect.pulse import make_envelopes_module
        >>> shim = make_envelopes_module()
        >>> shim.evc is quspect._pulse
        True
        >>> shim.EVC_COMPLEX_DTYPE.__name__
        'complex64'
    """
    module = _EnvelopesModule(
        "envelopes", "Stand-in for envelopes-qc backed by quspect.pulse."
    )
    module.__path__ = []
    module.__package__ = "envelopes"
    exports: dict[str, object] = {
        _SHIM_MARKER: True,
        "evc": _pulse,
        "envelopes_cpp": _pulse,
        "EVC_COMPLEX_DTYPE": np.complex64,
        "EVC_FLOAT_DTYPE": np.float32,
    }
    for name, value in exports.items():
        setattr(module, name, value)
    return module


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(f"{p}.") for p in prefixes)


def _register(shim: types.ModuleType) -> None:
    sys.modules["envelopes"] = shim
    sys.modules["envelopes.envelopes_cpp"] = _pulse
    sys.modules["envelopes.evc"] = _pulse


def _sync_parent_attribute(name: str, module: object) -> None:
    """Keep ``parent.child`` consistent with ``sys.modules[name]``."""
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    if parent is None or parent_name == "envelopes":
        return
    if module is _MISSING:
        if child in vars(parent):
            delattr(parent, child)
    else:
        setattr(parent, child, module)


@contextmanager
def use_as_envelopes(
    *, purge: Iterable[str] = _DEFAULT_PURGE
) -> Iterator[types.ModuleType]:
    """Temporarily make ``import envelopes`` resolve to :mod:`quspect.pulse`.

    On entry, every cached ``envelopes`` / ``envelopes.*`` module and every
    cached module named by ``purge`` (a package name matches itself and its
    submodules) is removed from :data:`sys.modules`, and the stand-in from
    :func:`make_envelopes_module` is registered under
    :data:`ENVELOPES_MODULE_NAMES`.  Consumer modules imported inside the
    block therefore bind :mod:`quspect._pulse`.

    On exit -- normally or through an exception -- all of those
    :data:`sys.modules` entries are restored exactly: modules imported inside
    the block are dropped again and the previously cached modules (including a
    real envelopes-qc) come back, together with the matching attributes on
    their parent packages.  Blocks can be nested.  The function edits global
    interpreter state and is not thread-safe.

    Args:
        purge: Top-level or dotted module names whose cached modules are
            re-imported against the stand-in.  Defaults to ``("lqcloud",)``.

    Yields:
        The registered stand-in ``envelopes`` module.

    Example:
        >>> import sys
        >>> import quspect._pulse
        >>> from quspect.pulse import use_as_envelopes
        >>> with use_as_envelopes():
        ...     from envelopes import evc
        ...     evc is quspect._pulse
        True
        >>> "envelopes" in sys.modules
        False
    """
    prefixes = ("envelopes", *purge)
    saved: dict[str, object] = {}
    for name in list(sys.modules):
        if _matches(name, prefixes):
            saved[name] = sys.modules.pop(name)
    for name in ENVELOPES_MODULE_NAMES:
        saved.setdefault(name, _MISSING)
    shim = make_envelopes_module()
    _register(shim)
    try:
        yield shim
    finally:
        for name in list(sys.modules):
            if _matches(name, prefixes):
                del sys.modules[name]
                if name not in saved:
                    _sync_parent_attribute(name, _MISSING)
        for name, module in saved.items():
            if module is not _MISSING:
                sys.modules[name] = module  # type: ignore[assignment]
            _sync_parent_attribute(name, module)


def install_as_envelopes(*, purge: Iterable[str] = _DEFAULT_PURGE) -> types.ModuleType:
    """Permanently make ``import envelopes`` resolve to :mod:`quspect.pulse`.

    The process-wide counterpart of :func:`use_as_envelopes`, for
    applications that should run a consumer such as ``lqcloud`` on QuSpect's
    implementation.  Call it before the consumer is imported; cached
    consumer modules named by ``purge`` and any cached ``envelopes`` modules
    are dropped so they are re-imported against the stand-in.  Calling it
    again while the stand-in is installed does nothing.  There is no
    uninstall; use :func:`use_as_envelopes` when the change must be undone.

    Args:
        purge: Module names whose cached modules are dropped, as in
            :func:`use_as_envelopes`.  Defaults to ``("lqcloud",)``.

    Returns:
        The installed stand-in ``envelopes`` module.

    Example:
        .. code-block:: python

            from quspect.pulse import install_as_envelopes

            install_as_envelopes()
            import lqcloud  # its pulse sequences now use quspect.pulse
    """
    current = sys.modules.get("envelopes")
    if current is not None and getattr(current, _SHIM_MARKER, False):
        return current
    prefixes = ("envelopes", *purge)
    for name in list(sys.modules):
        if _matches(name, prefixes):
            del sys.modules[name]
    shim = make_envelopes_module()
    _register(shim)
    return shim
