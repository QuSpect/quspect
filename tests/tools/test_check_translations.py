"""Tests for the PO parsing logic in ``tools/check_translations.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "check_translations.py"

# tools/ 不进 sdist；从解包的 sdist 运行测试时跳过本模块。
if not _SCRIPT.is_file():
    pytest.skip(
        f"{_SCRIPT} not found (tools/ is not shipped in the sdist)",
        allow_module_level=True,
    )


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_translations", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ct = _load()

_HEADER = """# SOME DESCRIPTIVE TITLE.
#, fuzzy
msgid ""
msgstr ""
"Language: zh_CN\\n"
"Content-Type: text/plain; charset=utf-8\\n"

"""

_SAMPLE = (
    _HEADER
    + """#: ../../source/a.rst:1
msgid "Hello"
msgstr "你好"

#: ../../source/a.rst:3
msgid ""
"A long sentence that "
"wraps over \\"several\\" lines."
msgstr ""
"一句"
"很长的话。"

#: ../../source/a.rst:5
msgid "Untranslated"
msgstr ""

#: ../../source/a.rst:7
#, fuzzy
msgid "Fuzzy one"
msgstr "模糊"

#, python-format
msgctxt "ctx"
msgid "With context"
msgstr "带上下文"

msgid "apple"
msgid_plural "apples"
msgstr[0] "苹果"

#~ msgid "Obsolete"
#~ msgstr ""
"""
)


def _stats(text: str) -> Any:
    return ct.compute_stats("x.po", ct.parse_po(text.splitlines()))


def test_parse_po_entries_and_continuations() -> None:
    entries = ct.parse_po(_SAMPLE.splitlines())
    assert entries[0].is_header
    ids = [e.msgid for e in entries[1:]]
    assert ids == [
        "Hello",
        'A long sentence that wraps over "several" lines.',
        "Untranslated",
        "Fuzzy one",
        "With context",
        "apple",
    ]
    assert entries[2].msgstrs == ["一句很长的话。"]
    # 头部的 fuzzy 标记不计入统计，也不应泄漏到后续条目。
    assert not entries[1].fuzzy
    assert entries[4].fuzzy
    assert not entries[5].fuzzy


def test_compute_stats_counts_categories_and_skips_obsolete() -> None:
    stats = _stats(_SAMPLE)
    assert (stats.translated, stats.untranslated, stats.fuzzy) == (4, 1, 1)
    assert stats.total == 6
    assert not stats.complete


def test_empty_plural_form_is_untranslated() -> None:
    text = _HEADER + 'msgid "a"\nmsgid_plural "as"\nmsgstr[0] ""\n'
    stats = _stats(text)
    assert (stats.translated, stats.untranslated) == (0, 1)


def test_header_only_file_is_complete() -> None:
    assert _stats(_HEADER).complete


def test_malformed_line_raises() -> None:
    with pytest.raises(ValueError):
        ct.parse_po(['msgid "ok"', "garbage"])


def test_check_locale_dir_respects_allowlist(tmp_path: Path) -> None:
    complete = _HEADER + 'msgid "Hi"\nmsgstr "嗨"\n'
    incomplete = _HEADER + 'msgid "Hi"\nmsgstr ""\n'
    (tmp_path / "api.po").write_text(incomplete, encoding="utf-8")
    (tmp_path / "index.po").write_text(complete, encoding="utf-8")
    (tmp_path / "guide.po").write_text(incomplete, encoding="utf-8")

    stats, failures = ct.check_locale_dir(tmp_path)
    assert [s.name for s in stats] == ["api.po", "guide.po", "index.po"]
    assert failures == ["guide.po"]
    assert ct.main(["--locale-dir", str(tmp_path)]) == 1

    (tmp_path / "guide.po").write_text(complete, encoding="utf-8")
    assert ct.main(["--locale-dir", str(tmp_path)]) == 0


def test_repository_locale_dir_passes() -> None:
    if not ct.DEFAULT_LOCALE_DIR.is_dir():
        pytest.skip("docs locale directory not available")
    _, failures = ct.check_locale_dir(ct.DEFAULT_LOCALE_DIR)
    assert failures == []
