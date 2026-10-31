#!/usr/bin/env python3
"""Smoke-test an installed QuSpect wheel.

Run this script with the interpreter that has the wheel installed, from a
directory that does **not** contain the ``quspect/`` source package (cibuildwheel
copies ``tools/`` into a temporary test directory for exactly this reason)::

    python tools/wheel_smoke.py --project-dir /path/to/checkout

Checks performed:

1. ``quspect`` is imported from the installed distribution, not from the
   source checkout given by ``--project-dir``.
2. The native core loads.  On x86_64 both ``_core_scalar`` and ``_core_avx2``
   must be present in the installed package, and the AVX2 module is selected
   iff the CPU supports AVX2 + FMA and ``QUSPECT_DISABLE_AVX2`` is not ``"1"``;
   when the scalar module is expected, the AVX2 module must not be imported.
   On other architectures there must be no AVX2 module and the backend must be
   ``"scalar"``.
3. With ``QUSPECT_EXPECT_OPENMP=1`` the core must report OpenMP support,
   ``get_num_threads() >= 1`` and ``set_num_threads(2)`` must take effect.
4. A Bell circuit runs through :class:`quspect.native.NativeExecutor` and the
   sampled counts sum to the requested number of shots.

The script exits with status 0 on success and 1 (with a message) on failure.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

EXPECT_OPENMP_ENV = "QUSPECT_EXPECT_OPENMP"
X86_64_MACHINES = frozenset({"x86_64", "amd64", "x64"})
SHOTS = 256


class SmokeTestError(Exception):
    """Raised when a smoke-test check fails."""


def check(condition: bool, message: str) -> None:
    """Raise :class:`SmokeTestError` with ``message`` unless ``condition`` holds.

    Args:
        condition: The asserted condition.
        message: Failure description.

    Raises:
        SmokeTestError: If ``condition`` is false.
    """
    if not condition:
        raise SmokeTestError(message)


def is_relative_to(path: Path, parent: Path) -> bool:
    """Return True if ``path`` is ``parent`` or lies below it (both resolved).

    Args:
        path: The candidate path.
        parent: The directory to test against.

    Returns:
        Whether ``path`` is inside ``parent``.
    """
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def extension_files(package_dir: Path, stem: str) -> list[Path]:
    """Return the extension-module files named ``stem`` in ``package_dir``.

    Args:
        package_dir: Directory of the installed ``quspect`` package.
        stem: Module name without suffix, e.g. ``"_core_avx2"``.

    Returns:
        Matching files with a suffix from
        :data:`importlib.machinery.EXTENSION_SUFFIXES`.
    """
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    return sorted(
        p
        for p in package_dir.iterdir()
        if p.is_file() and any(p.name == stem + suffix for suffix in suffixes)
    )


def check_installed_location(project_dirs: list[Path]) -> Path:
    """Import ``quspect`` and make sure it is not the source checkout.

    Args:
        project_dirs: Source checkouts the package must not be imported from.

    Returns:
        The directory of the imported ``quspect`` package.

    Raises:
        SmokeTestError: If ``quspect`` was imported from a source checkout.
    """
    import quspect

    package_file = Path(quspect.__file__ or "")
    version = importlib.metadata.version("quspect")
    print(f"quspect {version} imported from {package_file}")
    for project_dir in project_dirs:
        check(
            not is_relative_to(package_file, project_dir),
            f"quspect was imported from the source tree {project_dir}, "
            "not from the installed wheel",
        )
    return package_file.parent


def check_native_core(package_dir: Path, expect_backend: str) -> Any:
    """Verify that the right native extension module is installed and loaded.

    Args:
        package_dir: Directory of the installed ``quspect`` package.
        expect_backend: ``"auto"``, ``"scalar"`` or ``"avx2"``.  Anything but
            ``"auto"`` must also agree with the automatically derived choice.

    Returns:
        The loaded native core module.

    Raises:
        SmokeTestError: If any check fails.
    """
    from quspect._native_loader import (
        AVX2_MODULE,
        DISABLE_AVX2_ENV,
        SCALAR_MODULE,
        kernel_backend,
        load_core,
    )

    machine = platform.machine().lower()
    is_x86_64 = machine in X86_64_MACHINES
    scalar_files = extension_files(package_dir, "_core_scalar")
    avx2_files = extension_files(package_dir, "_core_avx2")
    print(f"machine: {machine}")
    print(f"scalar module files: {[p.name for p in scalar_files]}")
    print(f"avx2 module files: {[p.name for p in avx2_files]}")
    check(len(scalar_files) == 1, f"expected one _core_scalar module in {package_dir}")
    if is_x86_64:
        check(len(avx2_files) == 1, f"expected one _core_avx2 module in {package_dir}")
    else:
        check(not avx2_files, f"unexpected _core_avx2 module on {machine}")

    core = load_core()
    if core is None:
        raise SmokeTestError(f"native core {SCALAR_MODULE} could not be loaded")
    scalar = importlib.import_module(SCALAR_MODULE)
    cpu_avx2 = bool(scalar.cpu_supports_avx2())
    disabled = os.environ.get(DISABLE_AVX2_ENV) == "1"
    expected = "avx2" if is_x86_64 and cpu_avx2 and not disabled else "scalar"
    print(f"cpu_supports_avx2(): {cpu_avx2}, {DISABLE_AVX2_ENV}=1: {disabled}")
    if expect_backend != "auto":
        check(
            expected == expect_backend,
            f"--expect-backend={expect_backend} but this CPU/environment implies "
            f"{expected!r} (cpu_supports_avx2()={cpu_avx2}, disabled={disabled})",
        )

    caps = dict(core.get_runtime_capabilities())
    print(f"loaded module: {core.__name__} ({core.__file__})")
    print(f"capabilities: {caps}")
    check(
        core.__name__ == f"quspect._core_{expected}",
        f"loaded {core.__name__!r}, expected quspect._core_{expected}",
    )
    check(kernel_backend() == expected, f"kernel_backend() is {kernel_backend()!r}")
    for key in ("backend_f64", "backend_f32"):
        actual = caps.get(key)
        check(actual == expected, f"{key} is {actual!r}, not {expected!r}")
    if expected == "scalar":
        check(
            AVX2_MODULE not in sys.modules,
            f"{AVX2_MODULE} was imported although the scalar core is expected",
        )
    return core


def check_openmp(core: Any) -> None:
    """Verify OpenMP support when ``QUSPECT_EXPECT_OPENMP=1`` is set.

    Args:
        core: The loaded native core module.

    Raises:
        SmokeTestError: If OpenMP is expected but missing or not working.
    """
    caps = dict(core.get_runtime_capabilities())
    if os.environ.get(EXPECT_OPENMP_ENV) != "1":
        openmp = caps.get("openmp")
        print(f"OpenMP not required ({EXPECT_OPENMP_ENV}!=1): openmp={openmp}")
        return

    check(caps.get("openmp") is True, f"OpenMP expected, capabilities: {caps}")
    original = int(core.get_num_threads())
    check(original >= 1, f"get_num_threads() returned {original}")
    core.set_num_threads(2)
    try:
        now = int(core.get_num_threads())
        check(now == 2, f"get_num_threads() is {now} after set_num_threads(2)")
    finally:
        core.set_num_threads(original)
    print(f"OpenMP OK: default threads={original}, set_num_threads(2) works")


def check_bell_circuit() -> None:
    """Sample a Bell circuit with :class:`quspect.native.NativeExecutor`.

    Raises:
        SmokeTestError: If the counts are inconsistent with a Bell state.
    """
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps
    from quspect.native import NativeExecutor

    circuit = QuantumCircuit(name="bell")
    circuit.h(0).cx(0, 1).measure_all()
    result = NativeExecutor(seed=7).run(dumps(circuit), shots=SHOTS)
    print(f"Bell counts: {result.counts}")
    total = sum(result.counts.values())
    check(total == SHOTS, f"counts sum to {total}, expected {SHOTS}")
    check(
        set(result.counts) <= {"00", "11"},
        f"unexpected Bell outcomes {sorted(result.counts)}",
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` for :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project-dir",
        type=Path,
        action="append",
        default=[],
        help="source checkout that quspect must NOT be imported from (repeatable)",
    )
    parser.add_argument(
        "--expect-backend",
        choices=("auto", "scalar", "avx2"),
        default="auto",
        help="require this kernel backend (default: derive it from CPU and env)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run all smoke checks.

    Args:
        argv: Argument list, or ``None`` for :data:`sys.argv`.

    Returns:
        Process exit status: 0 on success, 1 on failure.
    """
    args = parse_args(argv)
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    try:
        package_dir = check_installed_location(list(args.project_dir))
        core = check_native_core(package_dir, args.expect_backend)
        check_openmp(core)
        check_bell_circuit()
    except SmokeTestError as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print("wheel smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
