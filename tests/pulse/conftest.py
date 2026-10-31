"""Skip the pulse tests when the ``quspect._pulse`` extension is not built."""

from __future__ import annotations

import importlib
import types
from collections.abc import Iterator

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every pulse test as skipped if the native module is missing."""
    try:
        importlib.import_module("quspect._pulse")
    except ModuleNotFoundError as exc:
        if exc.name != "quspect._pulse":
            raise
        skip = pytest.mark.skip(reason="quspect._pulse extension is not built")
        for item in items:
            if "tests/pulse" in str(item.fspath).replace("\\", "/"):
                item.add_marker(skip)


@pytest.fixture
def envelopes_shim() -> Iterator[types.ModuleType]:
    """Run a test with ``import envelopes`` served by :mod:`quspect.pulse`.

    Cached ``lqcloud`` modules are purged for the duration of the test, so code
    under test that imports ``lqcloud.sequence`` binds :mod:`quspect._pulse`.
    """
    from quspect.pulse import use_as_envelopes

    with use_as_envelopes(purge=("lqcloud",)) as shim:
        yield shim
