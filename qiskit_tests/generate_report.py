"""Run the qiskit cross-validation suite and render a Markdown report.

Usage::

    .venv/bin/python -m qiskit_tests.generate_report            # run + report
    .venv/bin/python -m qiskit_tests.generate_report --no-run   # report only

The full run executes ``pytest qiskit_tests`` with branch coverage and writes
``coverage.json``, ``htmlcov/`` and ``junit.xml`` into
``qiskit_tests/reports/``.  The script then parses the JUnit XML and the
coverage JSON and writes ``qiskit_tests/reports/REPORT.md`` (in Chinese).

Expected failures (``xfail(strict=True, reason="BUG: ...")``) appear in the
JUnit XML as ``<skipped type="pytest.xfail" message="<reason>">``; they are
collected, de-duplicated by reason and listed as the confirmed defect list.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("qiskit_tests.generate_report")

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parent
REPORTS_DIR = SUITE_DIR / "reports"
JUNIT_PATH = REPORTS_DIR / "junit.xml"
COVERAGE_JSON_PATH = REPORTS_DIR / "coverage.json"
HTMLCOV_DIR = REPORTS_DIR / "htmlcov"
COVERAGE_DATA_PATH = REPORTS_DIR / ".coverage"
REPORT_PATH = REPORTS_DIR / "REPORT.md"

#: Sub-packages summarised in the coverage table, in display order.
SUBPACKAGES: tuple[str, ...] = (
    "circuit",
    "gates",
    "simulation",
    "noise",
    "dag",
    "dag/passes",
    "transpiler",
    "ir",
    "operators",
    "variational",
    "optimization",
    "backends",
    "native",
    "algorithms",
    "visualization",
)

ROOT_GROUP = "(根模块)"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class TestCaseResult:
    """Outcome of a single JUnit ``<testcase>``.

    Attributes:
        file: Test file path relative to the repo root, e.g.
            ``qiskit_tests/test_ir_qsal.py``.
        test_id: pytest-style node id (``file::Class::name``).
        outcome: One of ``passed``, ``xfailed``, ``failed``, ``error``,
            ``skipped``.
        time: Duration in seconds.
        message: Failure / skip / xfail message (empty for passes).
    """

    __test__ = False

    file: str
    test_id: str
    outcome: str
    time: float
    message: str = ""


@dataclass
class JUnitSummary:
    """Parsed JUnit XML results.

    Attributes:
        cases: All test cases in document order.
        total_time: Suite wall time reported by pytest, in seconds.
        timestamp: Suite start timestamp reported by pytest.
    """

    cases: list[TestCaseResult] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: str = ""

    def count(self, outcome: str) -> int:
        """Return the number of cases with the given outcome.

        Args:
            outcome: Outcome label such as ``"passed"``.

        Returns:
            Number of matching test cases.

        Example:
            >>> JUnitSummary().count("passed")
            0
        """
        return sum(1 for c in self.cases if c.outcome == outcome)


@dataclass
class CoverageEntry:
    """Line / branch coverage numbers for one file or one group of files.

    Attributes:
        statements: Number of executable statements.
        covered_lines: Number of executed statements.
        branches: Number of branch destinations.
        covered_branches: Number of taken branch destinations.
        missing_lines: Line numbers that were never executed.
    """

    statements: int = 0
    covered_lines: int = 0
    branches: int = 0
    covered_branches: int = 0
    missing_lines: list[int] = field(default_factory=list)

    def add(self, other: CoverageEntry) -> None:
        """Accumulate the counters of ``other`` into this entry.

        Args:
            other: Entry whose counts are added (missing lines are ignored).

        Example:
            >>> a = CoverageEntry(statements=2, covered_lines=1)
            >>> a.add(CoverageEntry(statements=2, covered_lines=2))
            >>> (a.statements, a.covered_lines)
            (4, 3)
        """
        self.statements += other.statements
        self.covered_lines += other.covered_lines
        self.branches += other.branches
        self.covered_branches += other.covered_branches

    @property
    def line_pct(self) -> float | None:
        """Line coverage in percent, or ``None`` when there are no statements."""
        if self.statements == 0:
            return None
        return 100.0 * self.covered_lines / self.statements

    @property
    def branch_pct(self) -> float | None:
        """Branch coverage in percent, or ``None`` when there are no branches."""
        if self.branches == 0:
            return None
        return 100.0 * self.covered_branches / self.branches


@dataclass
class CoverageSummary:
    """Parsed ``coverage.json`` (format 3).

    Attributes:
        total: Totals over all measured files.
        files: Per-file entries keyed by path relative to the repo root.
        branch_enabled: Whether branch coverage was recorded.
    """

    total: CoverageEntry
    files: dict[str, CoverageEntry]
    branch_enabled: bool


@dataclass
class DefectGroup:
    """All xfail cases of one test file sharing the same ``BUG:`` reason.

    Attributes:
        file: Test file path.
        reason: The xfail reason text.
        test_ids: Node ids of the affected tests (parametrizations merged).
    """

    file: str
    reason: str
    test_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Running pytest
# --------------------------------------------------------------------------- #


def run_pytest() -> int:
    """Run the qiskit suite with coverage and JUnit output.

    The coverage data file is redirected to ``qiskit_tests/reports/.coverage``
    via ``COVERAGE_FILE`` so that this run does not clash with other runs.

    Returns:
        The pytest exit code (``0`` when everything passed / xfailed).

    Example:
        .. code-block:: python

            code = run_pytest()
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "qiskit_tests",
        "-p",
        "no:cacheprovider",
        "-q",
        "--cov=quspect",
        "--cov-branch",
        "--cov-report=json:qiskit_tests/reports/coverage.json",
        "--cov-report=html:qiskit_tests/reports/htmlcov",
        "--junitxml=qiskit_tests/reports/junit.xml",
        "-o",
        "junit_family=xunit2",
    ]
    env = dict(os.environ)
    env["COVERAGE_FILE"] = str(COVERAGE_DATA_PATH.relative_to(REPO_ROOT))
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    logger.info("pytest exited with code %d", proc.returncode)
    return proc.returncode


def count_marker(marker: str) -> int | None:
    """Count the qiskit-suite tests carrying a marker via ``--collect-only``.

    Args:
        marker: Marker expression passed to ``-m``, e.g. ``"qiskit_ref"``.

    Returns:
        Number of selected tests, or ``None`` if the output could not be
        parsed.

    Example:
        .. code-block:: python

            n = count_marker("statistical")
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "qiskit_tests",
        "-p",
        "no:cacheprovider",
        "--no-cov",
        "--collect-only",
        "-q",
        "-m",
        marker,
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    match = re.search(r"(\d+)(?:/\d+)? tests? collected", proc.stdout)
    if match is None:
        if "no tests collected" in proc.stdout:
            return 0
        logger.warning("could not count marker %r: %s", marker, proc.stdout[-500:])
        return None
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _node_location(classname: str, name: str) -> tuple[str, str]:
    """Map a JUnit ``classname`` / ``name`` pair to ``(file, node_id)``."""
    parts = classname.split(".")
    # qiskit_tests.test_x[.TestClass...]
    idx = next(
        (i for i, p in enumerate(parts) if p.startswith("test_")), len(parts) - 1
    )
    file = "/".join(parts[: idx + 1]) + ".py"
    node_id = "::".join([file, *parts[idx + 1 :], name])
    return file, node_id


def parse_junit(path: Path) -> JUnitSummary:
    """Parse a pytest JUnit XML (xunit2) file.

    Args:
        path: Path of the ``junit.xml`` file.

    Returns:
        A :class:`JUnitSummary` with one entry per test case.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        xml.etree.ElementTree.ParseError: If the XML is malformed.

    Example:
        .. code-block:: python

            summary = parse_junit(Path("qiskit_tests/reports/junit.xml"))
            print(summary.count("xfailed"))
    """
    root = ET.parse(path).getroot()
    summary = JUnitSummary()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        summary.total_time += float(suite.get("time", "0") or 0)
        summary.timestamp = summary.timestamp or suite.get("timestamp", "")
        for tc in suite.iter("testcase"):
            file, node_id = _node_location(tc.get("classname", ""), tc.get("name", ""))
            outcome, message = "passed", ""
            for tag, label in (
                ("failure", "failed"),
                ("error", "error"),
                ("skipped", "skipped"),
            ):
                elem = tc.find(tag)
                if elem is None:
                    continue
                message = elem.get("message", "") or (elem.text or "")
                outcome = label
                if tag == "skipped" and elem.get("type") == "pytest.xfail":
                    outcome = "xfailed"
                break
            summary.cases.append(
                TestCaseResult(
                    file=file,
                    test_id=node_id,
                    outcome=outcome,
                    time=float(tc.get("time", "0") or 0),
                    message=message.strip(),
                )
            )
    return summary


def parse_coverage(path: Path) -> CoverageSummary:
    """Parse a coverage.py JSON report (format 3).

    Args:
        path: Path of ``coverage.json``.

    Returns:
        A :class:`CoverageSummary` with totals and per-file entries.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the JSON lacks the expected ``totals`` / ``files`` keys.

    Example:
        .. code-block:: python

            cov = parse_coverage(Path("qiskit_tests/reports/coverage.json"))
            print(cov.total.line_pct)
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def entry(summary: dict[str, Any], missing: Iterable[int]) -> CoverageEntry:
        return CoverageEntry(
            statements=int(summary.get("num_statements", 0)),
            covered_lines=int(summary.get("covered_lines", 0)),
            branches=int(summary.get("num_branches", 0)),
            covered_branches=int(summary.get("covered_branches", 0)),
            missing_lines=sorted(int(x) for x in missing),
        )

    files = {
        name.replace("\\", "/"): entry(info["summary"], info.get("missing_lines", []))
        for name, info in data["files"].items()
    }
    return CoverageSummary(
        total=entry(data["totals"], []),
        files=files,
        branch_enabled=bool(data.get("meta", {}).get("branch_coverage", False)),
    )


def collect_defects(cases: Sequence[TestCaseResult]) -> list[DefectGroup]:
    """Group xfailed ``BUG:`` cases by (test file, reason).

    Parametrized cases (``test_x[a]``, ``test_x[b]``) that share a reason are
    merged into one test id ``test_x[a|b]``-style listing.

    Args:
        cases: Parsed JUnit test cases.

    Returns:
        Defect groups ordered by test file, then by first appearance.

    Example:
        >>> c = TestCaseResult("f.py", "f.py::t[1]", "xfailed", 0.0, "BUG: x")
        >>> [g.reason for g in collect_defects([c])]
        ['BUG: x']
    """
    groups: dict[tuple[str, str], DefectGroup] = {}
    for case in cases:
        if case.outcome != "xfailed" or not case.message.startswith("BUG"):
            continue
        key = (case.file, case.message)
        group = groups.setdefault(key, DefectGroup(case.file, case.message))
        group.test_ids.append(case.test_id)
    return sorted(groups.values(), key=lambda g: g.file)


def merge_param_ids(test_ids: Sequence[str]) -> list[str]:
    """Collapse parametrized node ids into ``name[p1, p2, ...]`` entries.

    Args:
        test_ids: Full node ids, possibly with ``[param]`` suffixes.

    Returns:
        Short ids (without the file part), one per test function.

    Example:
        >>> merge_param_ids(["f.py::t[a]", "f.py::t[b]", "f.py::u"])
        ['t[a, b]', 'u']
    """
    merged: dict[str, list[str]] = {}
    for tid in test_ids:
        short = tid.split("::", 1)[1] if "::" in tid else tid
        match = re.fullmatch(r"(.*?)\[(.*)\]", short)
        base, param = (match.group(1), match.group(2)) if match else (short, None)
        params = merged.setdefault(base, [])
        if param is not None:
            params.append(param)
    return [f"{b}[{', '.join(p)}]" if p else b for b, p in merged.items()]


def line_ranges(lines: Sequence[int]) -> str:
    """Format sorted line numbers as compact ranges.

    Args:
        lines: Sorted line numbers.

    Returns:
        A string such as ``"3-5, 9"`` (empty string for no lines).

    Example:
        >>> line_ranges([3, 4, 5, 9])
        '3-5, 9'
    """
    out: list[str] = []
    start: int | None = None
    prev = 0
    for n in lines:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(out)


def subpackage_of(path: str) -> str:
    """Return the quspect sub-package a source file belongs to.

    Args:
        path: File path relative to the repo root, e.g.
            ``quspect/dag/passes/base.py``.

    Returns:
        Sub-package key such as ``"dag/passes"``, or the root group label for
        top-level modules.

    Example:
        >>> subpackage_of("quspect/dag/passes/base.py")
        'dag/passes'
        >>> subpackage_of("quspect/backends/logicalqubit/_job.py")
        'backends'
    """
    parts = Path(path).parts
    if parts and parts[0] == "quspect":
        parts = parts[1:]
    if len(parts) <= 1:
        return ROOT_GROUP
    if parts[0] == "dag" and len(parts) > 2 and parts[1] == "passes":
        return "dag/passes"
    return parts[0]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _md(text: str) -> str:
    """Escape text for a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "quspect"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("git lookup failed: %s", exc)
        return "未知"
    suffix = "（quspect/ 有未提交改动）" if dirty else ""
    return f"`{sha}`（分支 `{branch}`）{suffix}"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def render_report(
    junit: JUnitSummary,
    coverage: CoverageSummary,
    markers: dict[str, int | None],
    *,
    pytest_exit_code: int | None,
) -> str:
    """Render the Chinese Markdown report.

    Args:
        junit: Parsed JUnit results.
        coverage: Parsed coverage results.
        markers: Marker name -> number of tests carrying it.
        pytest_exit_code: Exit code of the pytest run, or ``None`` when the
            report was generated with ``--no-run``.

    Returns:
        The full Markdown document.

    Example:
        .. code-block:: python

            text = render_report(junit, cov, {"qiskit_ref": 10}, pytest_exit_code=0)
    """
    out: list[str] = []
    w = out.append
    passed, xfailed = junit.count("passed"), junit.count("xfailed")
    failed, errors = junit.count("failed"), junit.count("error")
    skipped = junit.count("skipped")
    total_cases = len(junit.cases)

    w("# qiskit 交叉验证测试报告")
    w("")
    w("> 本文件由 `python -m qiskit_tests.generate_report` 自动生成，请勿手工编辑。")
    w(
        "> 人工整理的约定、误差尺度与缺陷分级见 "
        "[`qiskit_tests/README.md`](../README.md)。"
    )
    w("")
    w("## 运行环境")
    w("")
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    w(f"- 报告生成时间：{now}")
    w(f"- 测试运行时间戳（junit）：{junit.timestamp or '未知'}")
    if pytest_exit_code is None:
        w("- 运行方式：`--no-run`（复用已有 junit.xml / coverage.json）")
    else:
        w(f"- pytest 退出码：{pytest_exit_code}")
    w(f"- Python：{platform.python_version()}（{platform.platform()}）")
    w(f"- qiskit：{_package_version('qiskit')}")
    w(f"- quspect git commit：{_git_commit()}")
    w("")

    if failed or errors:
        w("## ⚠️ 失败 / 错误（需要立即处理）")
        w("")
        w("| 结果 | 测试 | 信息 |")
        w("|---|---|---|")
        for c in junit.cases:
            if c.outcome in ("failed", "error"):
                label = "失败" if c.outcome == "failed" else "错误"
                msg = _md(c.message[:300])
                w(f"| {label} | `{_md(c.test_id)}` | {msg} |")
        w("")
        w(
            "注：`xfail(strict=True)` 的用例若意外通过（XPASS），也会作为失败列在"
            "这里——说明对应缺陷已修复，应删除 xfail 标记。"
        )
        w("")

    w("## 总览")
    w("")
    w("| 指标 | 数值 |")
    w("|---|---|")
    w(f"| 用例总数 | {total_cases} |")
    w(f"| 通过 | {passed} |")
    w(f"| xfail（已确认缺陷） | {xfailed} |")
    w(f"| 失败 | {failed} |")
    w(f"| 错误 | {errors} |")
    w(f"| 跳过 | {skipped} |")
    w(f"| 耗时 | {junit.total_time:.1f} s |")
    w(
        f"| 行覆盖率 | {_pct(coverage.total.line_pct)} "
        f"({coverage.total.covered_lines}/{coverage.total.statements}) |"
    )
    if coverage.branch_enabled:
        w(
            f"| 分支覆盖率 | {_pct(coverage.total.branch_pct)} "
            f"({coverage.total.covered_branches}/{coverage.total.branches}) |"
        )
    else:
        w("| 分支覆盖率 | 未记录（coverage.json 不含分支数据） |")
    for name, n in markers.items():
        w(f"| 标记 `{name}` 的用例 | {'未知' if n is None else n} |")
    w("")
    defects = collect_defects(junit.cases)
    w(
        f"去重后的已确认缺陷（按测试文件 + reason）：**{len(defects)}** 条，"
        f"对应 {sum(len(d.test_ids) for d in defects)} 个 xfail 用例。"
    )
    w("")

    # Per test file
    w("## 按测试文件")
    w("")
    w("| 测试文件 | 通过 | xfail | 失败 | 错误 | 跳过 | 耗时 (s) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    per_file: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for c in junit.cases:
        per_file[c.file][c.outcome] += 1
        per_file[c.file]["time"] += c.time
    for file in sorted(per_file):
        s = per_file[file]
        w(
            f"| `{Path(file).name}` | {int(s['passed'])} | {int(s['xfailed'])} | "
            f"{int(s['failed'])} | {int(s['error'])} | {int(s['skipped'])} | "
            f"{s['time']:.2f} |"
        )
    w("")

    # Coverage per sub-package
    groups: dict[str, CoverageEntry] = defaultdict(CoverageEntry)
    members: dict[str, list[str]] = defaultdict(list)
    for path, entry in coverage.files.items():
        key = subpackage_of(path)
        groups[key].add(entry)
        members[key].append(path)
    order = [k for k in SUBPACKAGES if k in groups]
    order += sorted(k for k in groups if k not in SUBPACKAGES)

    w("## 覆盖率：按子包")
    w("")
    w("| 子包 | 语句 | 行覆盖率 | 分支 | 分支覆盖率 |")
    w("|---|---:|---:|---:|---:|")
    for key in order:
        g = groups[key]
        w(
            f"| `{key}` | {g.statements} | {_pct(g.line_pct)} | {g.branches} | "
            f"{_pct(g.branch_pct)} |"
        )
    t = coverage.total
    w(
        f"| **合计** | {t.statements} | {_pct(t.line_pct)} | {t.branches} | "
        f"{_pct(t.branch_pct)} |"
    )
    w("")
    w("## 覆盖率：按模块文件")
    w("")
    w("完整的逐行视图见 `qiskit_tests/reports/htmlcov/index.html`。")
    w("")
    for key in order:
        w(f"### `{key}`")
        w("")
        w("| 文件 | 行% | 分支% | 未覆盖行 |")
        w("|---|---:|---:|---|")
        for path in sorted(members[key]):
            e = coverage.files[path]
            w(
                f"| `{path}` | {_pct(e.line_pct)} | {_pct(e.branch_pct)} | "
                f"{line_ranges(e.missing_lines)} |"
            )
        w("")

    # Defects
    w("## 已确认缺陷清单（xfail）")
    w("")
    w(
        '每条均来自 `xfail(strict=True, reason="BUG: ...")`：断言描述的是正确行为，'
        "源码修复后该用例会 XPASS 并因 strict 报错提醒。同一测试文件中 reason "
        "相同的用例（含参数化）合并为一条。"
    )
    w("")
    by_file: dict[str, list[DefectGroup]] = defaultdict(list)
    for d in defects:
        by_file[d.file].append(d)
    idx = 0
    for file in sorted(by_file):
        w(f"### `{Path(file).name}`")
        w("")
        w("| # | reason | 用例数 | 测试 |")
        w("|---:|---|---:|---|")
        for d in by_file[file]:
            idx += 1
            ids = "<br>".join(f"`{_md(i)}`" for i in merge_param_ids(d.test_ids))
            w(f"| {idx} | {_md(d.reason)} | {len(d.test_ids)} | {ids} |")
        w("")
    other_xfail = [
        c
        for c in junit.cases
        if c.outcome == "xfailed" and not c.message.startswith("BUG")
    ]
    if other_xfail:
        w("### 非 `BUG:` 前缀的 xfail")
        w("")
        for c in other_xfail:
            w(f"- `{c.test_id}`：{_md(c.message)}")
        w("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success (even if tests failed -- failures
        are listed in the report), ``1`` if the result files are missing.

    Example:
        .. code-block:: console

            $ python -m qiskit_tests.generate_report --no-run
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="不运行测试，只用已有的 junit.xml / coverage.json 生成报告",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    exit_code: int | None = None
    if not args.no_run:
        exit_code = run_pytest()
    for required in (JUNIT_PATH, COVERAGE_JSON_PATH):
        if not required.exists():
            logger.error("missing %s; run without --no-run first", required)
            return 1

    junit = parse_junit(JUNIT_PATH)
    coverage = parse_coverage(COVERAGE_JSON_PATH)
    markers = {name: count_marker(name) for name in ("qiskit_ref", "statistical")}
    REPORT_PATH.write_text(
        render_report(junit, coverage, markers, pytest_exit_code=exit_code),
        encoding="utf-8",
    )
    sys.stdout.write(f"{REPORT_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
