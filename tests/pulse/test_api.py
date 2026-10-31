"""The public API of quspect._pulse matches envelopes-qc 0.1.5 (envelopes_cpp).

``reference_api.json`` was dumped from the reference module, with its module path
rewritten to ``quspect._pulse``.  Generated signatures depend on the pybind11
version, so docstrings are compared without their signature lines.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

REFERENCE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "reference_api.json").read_text()
)
SKIPPED_ATTRS = {
    "__module__",
    "__dict__",
    "__weakref__",
    "__doc__",
    "_pybind11_conduit_v1_",
}


def doc_bodies(doc: str | None) -> list[str] | None:
    """Return the per-overload docstring bodies, without pybind11 signature lines."""
    if doc is None:
        return None
    if "\nOverloaded function.\n\n" in doc:
        parts = re.split(r"\n\n\d+\. ", doc.split("\nOverloaded function.\n\n", 1)[1])
    else:
        parts = [doc]
    return [part.partition("\n\n")[2].strip() for part in parts]


def dump_api() -> dict[str, Any]:
    """Describe quspect._pulse the same way reference_api.json was produced."""
    from quspect import _pulse

    api: dict[str, Any] = {"__doc__": _pulse.__doc__}
    for name in dir(_pulse):
        if name.startswith("__"):
            continue
        obj = getattr(_pulse, name)
        if inspect.isclass(obj):
            entry: dict[str, Any] = {
                "__mro__": [c.__name__ for c in obj.__mro__],
                "doc": obj.__doc__,
            }
            for key, value in obj.__dict__.items():
                if key not in SKIPPED_ATTRS:
                    entry[key] = getattr(value, "__doc__", None)
            api[name] = entry
        else:
            api[name] = obj.__doc__
    return api


def test_module_names_match() -> None:
    assert sorted(dump_api()) == sorted(REFERENCE)


@pytest.mark.parametrize(
    "name", sorted(k for k, v in REFERENCE.items() if isinstance(v, dict))
)
def test_class_matches(name: str) -> None:
    actual = dump_api()[name]
    expected = REFERENCE[name]
    assert actual["__mro__"] == expected["__mro__"]
    assert actual["doc"] == expected["doc"]
    assert sorted(actual) == sorted(expected)
    for key in expected:
        if key not in ("__mro__", "doc"):
            assert doc_bodies(actual[key]) == doc_bodies(expected[key]), key


@pytest.mark.parametrize(
    "name", sorted(k for k, v in REFERENCE.items() if not isinstance(v, dict))
)
def test_function_docs_match(name: str) -> None:
    actual = dump_api()[name]
    if name == "__doc__":
        assert actual == REFERENCE[name]
    else:
        assert doc_bodies(actual) == doc_bodies(REFERENCE[name])
