"""Fail when narrative zh_CN documentation translations are incomplete.

Parses every ``*.po`` file in the Sphinx locale directory with a small
stdlib-only parser and reports, per file, how many entries are translated,
untranslated and fuzzy.  The script exits with status 1 when any file that
is not on the allowlist contains untranslated or fuzzy entries.

``api.po`` is allowlisted: it holds the autodoc API reference, which is
intentionally kept in English.

Usage::

    python tools/check_translations.py
    python tools/check_translations.py --locale-dir docs/source/locale/zh_CN/LC_MESSAGES
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCALE_DIR = _REPO_ROOT / "docs" / "source" / "locale" / "zh_CN" / "LC_MESSAGES"

#: PO files that may stay (partially) untranslated.  ``api.po`` is the English
#: API reference generated from docstrings.
ALLOWLIST: frozenset[str] = frozenset({"api.po"})

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


@dataclass
class PoEntry:
    """One message of a PO file (the header entry has an empty ``msgid``)."""

    msgid: str = ""
    msgstrs: list[str] = field(default_factory=list)
    fuzzy: bool = False

    @property
    def is_header(self) -> bool:
        """Return True for the metadata header entry."""
        return self.msgid == ""

    @property
    def is_translated(self) -> bool:
        """Return True when every ``msgstr`` (or plural form) is non-empty."""
        return bool(self.msgstrs) and all(self.msgstrs)


@dataclass(frozen=True)
class PoStats:
    """Translation statistics for one PO file."""

    name: str
    translated: int
    untranslated: int
    fuzzy: int

    @property
    def total(self) -> int:
        """Return the number of non-header entries."""
        return self.translated + self.untranslated + self.fuzzy

    @property
    def complete(self) -> bool:
        """Return True when there are no untranslated or fuzzy entries."""
        return self.untranslated == 0 and self.fuzzy == 0


def _unquote(token: str) -> str:
    """Decode one double-quoted PO string literal."""
    token = token.strip()
    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise ValueError(f"malformed PO string literal: {token!r}")
    body = token[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            out.append(_ESCAPES.get(body[i + 1], body[i + 1]))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_po(lines: Iterable[str]) -> list[PoEntry]:
    """Parse PO file lines into entries, skipping obsolete (``#~``) entries.

    Args:
        lines: The lines of a PO file.

    Returns:
        The active entries in file order, including the header entry.

    Raises:
        ValueError: If a string literal or continuation line is malformed.
    """
    entries: list[PoEntry] = []
    current: PoEntry | None = None
    # Which field continuation lines append to: ("msgid", -1) or ("msgstr", i).
    target: tuple[str, int] | None = None
    pending_fuzzy = False

    def start_entry() -> PoEntry:
        nonlocal current, pending_fuzzy
        current = PoEntry(fuzzy=pending_fuzzy)
        pending_fuzzy = False
        entries.append(current)
        return current

    for raw in lines:
        line = raw.strip()
        if not line:
            current, target = None, None
            continue
        if line.startswith("#~"):
            current, target = None, None
            continue
        if line.startswith("#"):
            if current is not None and current.msgstrs:
                current, target = None, None
            if line.startswith("#,") and "fuzzy" in {
                flag.strip() for flag in line[2:].split(",")
            }:
                pending_fuzzy = True
            continue
        keyword, _, rest = line.partition(" ")
        if keyword == "msgctxt":
            start_entry()
            target = ("msgctxt", -1)
        elif keyword == "msgid":
            entry = current
            if entry is None or entry.msgstrs or target != ("msgctxt", -1):
                entry = start_entry()
            entry.msgid = _unquote(rest)
            target = ("msgid", -1)
        elif keyword == "msgid_plural":
            target = ("msgid_plural", -1)
        elif keyword == "msgstr" or keyword.startswith("msgstr["):
            if current is None:
                raise ValueError(f"msgstr without msgid: {line!r}")
            current.msgstrs.append(_unquote(rest))
            target = ("msgstr", len(current.msgstrs) - 1)
        elif line.startswith('"'):
            if current is None or target is None:
                raise ValueError(f"unexpected continuation line: {line!r}")
            text = _unquote(line)
            kind, index = target
            if kind == "msgid":
                current.msgid += text
            elif kind == "msgstr":
                current.msgstrs[index] += text
        else:
            raise ValueError(f"unrecognised PO line: {line!r}")
    return entries


def compute_stats(name: str, entries: Sequence[PoEntry]) -> PoStats:
    """Count translated, untranslated and fuzzy non-header entries.

    Args:
        name: File name used in the report.
        entries: Entries returned by :func:`parse_po`.

    Returns:
        The aggregated statistics.  Fuzzy entries are counted only as fuzzy.
    """
    translated = untranslated = fuzzy = 0
    for entry in entries:
        if entry.is_header:
            continue
        if entry.fuzzy:
            fuzzy += 1
        elif entry.is_translated:
            translated += 1
        else:
            untranslated += 1
    return PoStats(name, translated, untranslated, fuzzy)


def check_locale_dir(
    locale_dir: Path, allowlist: frozenset[str] = ALLOWLIST
) -> tuple[list[PoStats], list[str]]:
    """Compute stats for every PO file and list incomplete, non-allowlisted ones.

    Args:
        locale_dir: Directory containing ``*.po`` files.
        allowlist: File names exempt from the completeness requirement.

    Returns:
        A pair of (stats for every file, names of failing files).
    """
    stats: list[PoStats] = []
    failures: list[str] = []
    for path in sorted(locale_dir.glob("*.po")):
        with path.open(encoding="utf-8") as fh:
            result = compute_stats(path.name, parse_po(fh))
        stats.append(result)
        if not result.complete and path.name not in allowlist:
            failures.append(path.name)
    return stats, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and print a per-file report; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locale-dir", type=Path, default=DEFAULT_LOCALE_DIR)
    args = parser.parse_args(argv)
    locale_dir: Path = args.locale_dir

    if not locale_dir.is_dir():
        print(f"error: locale directory not found: {locale_dir}", file=sys.stderr)
        return 1
    stats, failures = check_locale_dir(locale_dir)
    if not stats:
        print(f"error: no .po files in {locale_dir}", file=sys.stderr)
        return 1

    for item in stats:
        if item.name in ALLOWLIST:
            status = "allowlisted"
        else:
            status = "ok" if item.complete else "INCOMPLETE"
        print(
            f"{item.name:32} {item.translated:4}/{item.total:<4} translated, "
            f"{item.untranslated:4} untranslated, {item.fuzzy:3} fuzzy  [{status}]"
        )
    if failures:
        print(
            "\nerror: untranslated or fuzzy entries in: "
            f"{', '.join(failures)}.\n"
            "Translate them in docs/source/locale/zh_CN/LC_MESSAGES; "
            "only the allowlisted API reference may stay English.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
