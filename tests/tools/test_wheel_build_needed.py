"""Tests for the pull-request path filter in ``tools/wheel_build_needed.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "wheel_build_needed.py"

# tools/ 不进 sdist；从解包的 sdist 运行测试时跳过本模块。
if not _SCRIPT.is_file():
    pytest.skip(
        f"{_SCRIPT} not found (tools/ is not shipped in the sdist)",
        allow_module_level=True,
    )


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wheel_build_needed", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wbn = _load()


@pytest.mark.parametrize(
    "paths",
    [
        # 只改文档
        ["docs/source/index.rst", "docs/locale/zh_CN/LC_MESSAGES/index.po"],
        ["README.md"],
        ["README.md", "README_CN.md", "CITATION.cff"],
        ["skills/quspect-optimization/SKILL.md"],
        # 只改云平台后端子包及其测试
        ["quspect/backends/tianyan/_client.py", "tests/backends/test_tianyan.py"],
        [
            "quspect/backends/originq/_translator.py",
            "quspect/backends/quafu/_backend.py",
            "quspect/backends/logicalqubit/_job.py",
            "tests/backends/_originq_fakes.py",
            "docs/source/backends.rst",
        ],
        ["qiskit_tests/test_circuit_api.py"],
        [".github/workflows/ci.yml"],
    ],
)
def test_unrelated_changes_skip_the_wheels(paths: list[str]) -> None:
    assert wbn.relevant_paths(paths) == []
    assert not wbn.wheels_needed(paths)


@pytest.mark.parametrize(
    "path",
    [
        # 构建输入与 wheel 流水线本身
        "cpp/src/common/parallel.cpp",
        "cpp/CMakeLists.txt",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "NOTICE",
        "tools/wheel_smoke.py",
        "tools/wheel_build_needed.py",
        "examples/bell.py",
        ".github/workflows/wheels.yml",
        ".github/workflows/release.yml",
        ".github/dependabot.yml",
        # wheel 测试导入 / 覆盖到的包
        "quspect/__init__.py",
        "quspect/_native_loader.py",
        "quspect/native/__init__.py",
        "quspect/transpiler/_transpile.py",
        "quspect/synthesis/two_qubit.py",
        "quspect/dag/passes.py",
        "quspect/gates/synthesis.py",
        "quspect/hamsim/__init__.py",
        "quspect/optimization/isingq.py",
        "quspect/pulse/__init__.py",
        "quspect/circuit/circuit.py",
        "quspect/ir/qasm2.py",
        "quspect/simulation/simulator.py",
        "quspect/variational/sampling.py",
        # 后端的公共部分被 transpiler / simulation 等导入
        "quspect/backends/__init__.py",
        "quspect/backends/_base.py",
        "quspect/backends/_exceptions.py",
        # 未登记的后端子包：拿不准就构建
        "quspect/backends/newcloud/_client.py",
        # wheel 测试及其余测试目录
        "tests/native/test_native_loader.py",
        "tests/optimization/test_isingq.py",
        "tests/pulse/conftest.py",
        "tests/synthesis/test_native_kak.py",
        "tests/transpiler/test_golden_transpile.py",
        "tests/hamsim/test_hamsim.py",
        "tests/circuit/test_circuit.py",
        "tests/__init__.py",
        # 名字相近但不在清单里的路径
        "docs.py",
        "README.rst",
        "tests/backends.py",
        "quspect/backends",
        "src/docs/index.rst",
    ],
)
def test_relevant_change_builds_the_wheels(path: str) -> None:
    assert wbn.affects_wheels(path)
    assert wbn.wheels_needed([path])
    # 与无关改动混在一起时仍然构建
    mixed = ["docs/source/index.rst", path, "quspect/backends/tianyan/_client.py"]
    assert wbn.relevant_paths(mixed) == [path]
    assert wbn.wheels_needed(mixed)


def test_unknown_change_set_builds_the_wheels() -> None:
    assert wbn.wheels_needed([])


def test_parse_paths() -> None:
    assert wbn.parse_paths("") == []
    assert wbn.parse_paths("a/b.py\nc d.md\n") == ["a/b.py", "c d.md"]
    # git diff -z：路径以 NUL 分隔，路径中的换行不再是分隔符
    assert wbn.parse_paths("a/b.py\0docs/x\ny.rst\0") == ["a/b.py", "docs/x\ny.rst"]


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [
        (b"docs/a.rst\0README.md\0", "false"),
        (b"docs/a.rst\0cpp/src/x.cpp\0", "true"),
        (b"docs/\xe6\x96\x87\xe6\xa1\xa3.rst\0", "false"),
        (b"quspect/transpiler/\xff.py\0", "true"),
        (b"", "true"),
    ],
)
def test_command_line(stdin: bytes, expected: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input=stdin,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert proc.stdout.decode().strip() == expected
    assert proc.stderr
