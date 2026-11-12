r"""Decide whether a pull request has to build and test the wheels.

``.github/workflows/wheels.yml`` (job ``plan``) pipes the paths changed by a
pull request into this script::

    git diff -z --name-only --no-renames HEAD^1 HEAD \
        | python tools/wheel_build_needed.py

It prints ``true`` or ``false`` on stdout and the reason on stderr.

The wheels are skipped only when *every* changed path is known not to affect
the contents of a wheel or the outcome of the wheel tests
(``[tool.cibuildwheel] test-command`` in ``pyproject.toml``).  Anything that
is not listed below builds the wheels, and so does an empty or unreadable
list: when in doubt, build.

Paths that never reach a wheel job:

* ``docs/``, ``skills/``, ``qiskit_tests/``, the READMEs and ``CITATION.cff``
  (``README.md`` is the long description in the wheel metadata, but no wheel
  test reads it);
* ``.github/workflows/ci.yml`` (the wheel workflows are ``wheels.yml`` and
  ``release.yml``);
* the cloud provider packages ``quspect/backends/<provider>/`` and their
  tests in ``tests/backends/``.  ``import quspect`` and the modules covered by
  the wheel tests import only the shared backend core
  (``quspect/backends/*.py``), which therefore still builds the wheels.

Everything else builds: ``cpp/``, ``pyproject.toml``, ``tools/``,
``examples/``, the wheel workflows, the licence files and the rest of
``quspect/`` and ``tests/``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

#: Directories (with trailing slash) whose files never affect a wheel job.
SKIP_PREFIXES: tuple[str, ...] = (
    "docs/",
    "skills/",
    "qiskit_tests/",
    "tests/backends/",
    # 云平台后端子包：import quspect 与 wheel 测试覆盖的模块都不导入它们。
    # 新增平台时在这里登记；未登记的子包按“需要构建”处理。
    "quspect/backends/logicalqubit/",
    "quspect/backends/originq/",
    "quspect/backends/quafu/",
    "quspect/backends/tianyan/",
)

#: Single files that never affect a wheel job.
SKIP_FILES: frozenset[str] = frozenset(
    {
        "README.md",
        "README_CN.md",
        "CITATION.cff",
        ".github/workflows/ci.yml",
    }
)


def affects_wheels(path: str) -> bool:
    """Whether a change to ``path`` can affect a wheel or the wheel tests.

    Args:
        path: Repository-relative path with forward slashes, as printed by
            ``git diff --name-only``.

    Returns:
        ``False`` only for the paths listed in :data:`SKIP_FILES` and below
        :data:`SKIP_PREFIXES`.

    Examples:
        >>> affects_wheels("docs/source/index.rst")
        False
        >>> affects_wheels("quspect/backends/_base.py")
        True
    """
    if path in SKIP_FILES:
        return False
    return not path.startswith(SKIP_PREFIXES)


def relevant_paths(paths: Iterable[str]) -> list[str]:
    """The changed paths that require the wheels, in input order."""
    return [path for path in paths if affects_wheels(path)]


def wheels_needed(paths: Iterable[str]) -> bool:
    """Whether a pull request changing ``paths`` must build the wheels.

    An empty list means the change set could not be determined, which builds.

    Examples:
        >>> wheels_needed(["docs/index.rst", "README.md"])
        False
        >>> wheels_needed(["docs/index.rst", "cpp/CMakeLists.txt"])
        True
        >>> wheels_needed([])
        True
    """
    changed = list(paths)
    return not changed or bool(relevant_paths(changed))


def parse_paths(data: str) -> list[str]:
    """Split the output of ``git diff --name-only`` (``-z`` or one per line)."""
    separator = "\0" if "\0" in data else "\n"
    return [path for path in data.split(separator) if path]


def main() -> int:
    """Read the changed paths from stdin and print ``true`` or ``false``."""
    paths = parse_paths(sys.stdin.buffer.read().decode("utf-8", "surrogateescape"))
    relevant = relevant_paths(paths)
    if not paths:
        print("no changed paths given: building the wheels", file=sys.stderr)
    elif relevant:
        shown = ", ".join(ascii(path) for path in relevant[:10])
        more = f" (+{len(relevant) - 10} more)" if len(relevant) > 10 else ""
        print(
            f"{len(relevant)} of {len(paths)} changed paths affect the wheels: "
            f"{shown}{more}",
            file=sys.stderr,
        )
    else:
        print(
            f"none of the {len(paths)} changed paths affects the wheels: skipping",
            file=sys.stderr,
        )
    print("true" if wheels_needed(paths) else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
