"""Tests for the pure helpers in ``tools/windows_repair_wheel.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "windows_repair_wheel.py"

# tools/ 不进 sdist；从解包的 sdist 运行测试时跳过本模块。
if not _SCRIPT.is_file():
    pytest.skip(
        f"{_SCRIPT} not found (tools/ is not shipped in the sdist)",
        allow_module_level=True,
    )


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("windows_repair_wheel", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrw = _load()


def test_parse_version() -> None:
    assert wrw.parse_version(" 14.50.35719.0 ") == (14, 50, 35719, 0)
    assert wrw.parse_version("14") == (14,)
    for bad in ("", "v143", "14.", "onecore"):
        with pytest.raises(ValueError):
            wrw.parse_version(bad)


def test_choose_highest_picks_newest_version() -> None:
    candidates = [
        (Path("14.40.33807/x64/Microsoft.VC143.CRT/msvcp140.dll"), (14, 40, 33816, 0)),
        (Path("v145/x64/Microsoft.VC145.CRT/msvcp140.dll"), (14, 50, 35719, 0)),
        (Path("14.50.35710/x64/Microsoft.VC145.CRT/msvcp140.dll"), (14, 50, 35719, 0)),
        (Path("14.44.35112/x64/Microsoft.VC143.CRT/msvcp140.dll"), (14, 44, 35211, 0)),
    ]
    path, version = wrw.choose_highest(candidates)
    assert version == (14, 50, 35719, 0)
    # Ties are broken deterministically by path.
    assert path == Path("v145/x64/Microsoft.VC145.CRT/msvcp140.dll")
    with pytest.raises(ValueError):
        wrw.choose_highest([])


@pytest.mark.parametrize(
    ("runtime", "toolset", "too_old"),
    [
        ((14, 40, 33816, 0), (14, 51), True),
        ((14, 38, 33135, 0), (14, 40), True),
        ((14, 50, 35719, 0), (14, 51), False),
        ((14, 40, 33807, 0), (14, 41), False),
        ((14, 51, 1, 0), (14, 51), False),
        ((15, 0), (14, 51), False),
    ],
)
def test_runtime_too_old(
    runtime: tuple[int, ...], toolset: tuple[int, ...], too_old: bool
) -> None:
    assert wrw.runtime_too_old(runtime, toolset) is too_old


def test_runtime_too_old_rejects_short_versions() -> None:
    with pytest.raises(ValueError):
        wrw.runtime_too_old((14,), (14, 51))


def test_is_bundled_dll() -> None:
    assert wrw.is_bundled_dll("quspect.libs/msvcp140-0a1b2c3d.dll", "msvcp140.dll")
    assert wrw.is_bundled_dll("quspect.libs/VCOMP140-ABC.dll", "vcomp140.dll")
    assert wrw.is_bundled_dll("msvcp140.dll", "msvcp140.dll")
    assert not wrw.is_bundled_dll("quspect.libs/msvcp140_1-0a1b.dll", "msvcp140.dll")
    assert not wrw.is_bundled_dll("quspect.libs/xmsvcp140.dll", "msvcp140.dll")


def test_pe_versions_prefers_file_version() -> None:
    assert wrw.PeVersions((14, 50, 1, 0), (14, 50)).runtime_version == (14, 50, 1, 0)
    assert wrw.PeVersions(None, (14, 51)).runtime_version == (14, 51)
