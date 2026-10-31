#!/usr/bin/env python3
"""Repair a Windows wheel with delvewheel, bundling an up-to-date MSVC runtime.

cibuildwheel runs this script as the Windows ``repair-wheel-command``::

    python tools/windows_repair_wheel.py --wheel <wheel> --dest-dir <dest_dir>

Plain ``delvewheel repair --include msvcp140.dll`` picks up whatever
``msvcp140.dll`` comes first on ``PATH`` -- on GitHub runners that is the
(older) copy in ``System32``.  An extension built with a newer MSVC STL must
ship a runtime at least as new as its toolset (e.g. the VS 17.10
``std::mutex`` constexpr-constructor change crashes with older
``msvcp140.dll``).  This script therefore:

1. Locates every Visual Studio installation with the x64 C++ tools via
   ``vswhere.exe`` (plus ``VCToolsRedistDir`` / ``VSINSTALLDIR`` if set; Windows
   environment variable names are case-insensitive) and
   globs ``VC\\Redist\\MSVC\\*\\x64\\Microsoft.VC*.CRT\\msvcp140.dll`` and
   ``...\\Microsoft.VC*.OpenMP\\vcomp140.dll``, choosing the copy with the
   highest file version.
2. Runs ``delvewheel repair`` with those directories passed via
   ``--add-path`` (delvewheel prepends them to ``PATH``, so they win over
   ``System32``) and ``--include msvcp140.dll --include vcomp140.dll``.
3. Opens the repaired wheel and fails unless both runtimes are bundled and
   neither is older than the platform toolset (PE linker version) of the
   bundled ``.pyd`` files.

The Windows-only dependency ``pefile`` (installed together with delvewheel)
is imported lazily, so the pure helpers can be imported and tested anywhere.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Version = tuple[int, ...]

REDIST_ARCH = "x64"
CRT_DLL = "msvcp140.dll"
OMP_DLL = "vcomp140.dll"
VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
_VERSION_RE = re.compile(r"\d+(?:\.\d+)*")


@dataclass(frozen=True)
class PeVersions:
    """Version information read from a PE file.

    Attributes:
        file_version: ``VS_FIXEDFILEINFO`` file version (4 components), or
            ``None`` if the file has no version resource.
        linker_version: ``(MajorLinkerVersion, MinorLinkerVersion)`` from the
            optional header, e.g. ``(14, 51)``.
    """

    file_version: Version | None
    linker_version: Version

    @property
    def runtime_version(self) -> Version:
        """Return the version used to compare a runtime DLL to a toolset."""
        return self.file_version if self.file_version else self.linker_version


def parse_version(text: str) -> Version:
    """Parse a dotted version string into a tuple of integers.

    Args:
        text: A version such as ``"14.50.35719"`` (surrounding whitespace is
            ignored).

    Returns:
        The version components, e.g. ``(14, 50, 35719)``.

    Raises:
        ValueError: If ``text`` is not a dotted sequence of integers.

    Example:
        >>> parse_version("14.50.35719.0")
        (14, 50, 35719, 0)
    """
    stripped = text.strip()
    if not _VERSION_RE.fullmatch(stripped):
        raise ValueError(f"not a dotted version: {text!r}")
    return tuple(int(part) for part in stripped.split("."))


def format_version(version: Version) -> str:
    """Format a version tuple as a dotted string.

    Args:
        version: Version components.

    Returns:
        The dotted representation.

    Example:
        >>> format_version((14, 51))
        '14.51'
    """
    return ".".join(str(part) for part in version)


def choose_highest(candidates: Iterable[tuple[Path, Version]]) -> tuple[Path, Version]:
    """Return the candidate with the highest version.

    Ties are broken by path so the choice is deterministic.

    Args:
        candidates: ``(path, version)`` pairs.

    Returns:
        The ``(path, version)`` pair with the highest version.

    Raises:
        ValueError: If ``candidates`` is empty.

    Example:
        >>> candidates = [(Path("a"), (14, 40)), (Path("b"), (14, 50, 1))]
        >>> path, version = choose_highest(candidates)
        >>> path.name, version
        ('b', (14, 50, 1))
    """
    items = list(candidates)
    if not items:
        raise ValueError("no candidates")
    return max(items, key=lambda item: (item[1], str(item[0])))


def runtime_too_old(runtime: Version, toolset: Version) -> bool:
    """Return True if a runtime DLL is too old for the given toolset.

    Uses the same rule as delvewheel's toolset warning: compare the major
    version and the tens digit of the minor version.  A toolset's minimum
    compatible redistributable normally matches its linker version, but e.g.
    VS 17.11 links as 14.41 while shipping the 14.40 redistributable, so a
    strict ``runtime < toolset`` check would reject the redistributable that
    ships with the very toolset used for the build.  The rule still catches
    the binary-incompatible case (a 14.4x build against a 14.3x runtime, or
    a 14.5x build against the 14.40 runtime from ``System32``).

    Args:
        runtime: Version of the runtime DLL (at least ``major.minor``).
        toolset: PE linker version of the extension (at least
            ``major.minor``).

    Returns:
        Whether the runtime is from an older toolset generation.

    Raises:
        ValueError: If either version has fewer than two components.

    Example:
        >>> runtime_too_old((14, 40, 33816, 0), (14, 51))
        True
        >>> runtime_too_old((14, 50, 35719, 0), (14, 51))
        False
    """
    if len(runtime) < 2 or len(toolset) < 2:
        raise ValueError("versions need at least major.minor")
    return (runtime[0], runtime[1] // 10) < (toolset[0], toolset[1] // 10)


def is_bundled_dll(member: str, dll_name: str) -> bool:
    """Return True if a wheel member is a (possibly name-mangled) copy of a DLL.

    delvewheel renames bundled DLLs to ``<stem>-<hash>.dll``.

    Args:
        member: Archive member name, e.g. ``"quspect.libs/msvcp140-ab12.dll"``.
        dll_name: Original DLL name, e.g. ``"msvcp140.dll"``.

    Returns:
        Whether ``member`` is that DLL.

    Example:
        >>> is_bundled_dll("quspect.libs/msvcp140-1a2b3c.dll", "msvcp140.dll")
        True
        >>> is_bundled_dll("quspect.libs/msvcp140_1-1a2b.dll", "msvcp140.dll")
        False
    """
    stem = re.escape(dll_name[: -len(".dll")])
    pattern = rf"(?:.*/)?{stem}(?:-[0-9a-z]+)?\.dll"
    return re.fullmatch(pattern, member, flags=re.IGNORECASE) is not None


def read_pe_versions(path: Path) -> PeVersions:
    """Read the file version and linker version of a PE file using pefile."""
    pefile: Any = importlib.import_module("pefile")
    pe = pefile.PE(str(path))
    try:
        linker = (
            int(pe.OPTIONAL_HEADER.MajorLinkerVersion),
            int(pe.OPTIONAL_HEADER.MinorLinkerVersion),
        )
        file_version: Version | None = None
        fixed = getattr(pe, "VS_FIXEDFILEINFO", None)
        if fixed:
            ms = int(fixed[0].FileVersionMS)
            ls = int(fixed[0].FileVersionLS)
            file_version = (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
        return PeVersions(file_version=file_version, linker_version=linker)
    finally:
        pe.close()


def find_vs_installations() -> list[Path]:
    """Return Visual Studio installation paths that have the x64 C++ tools."""
    roots: list[Path] = []
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if vswhere.is_file():
        result = subprocess.run(
            [
                str(vswhere),
                "-all",
                "-products",
                "*",
                "-requires",
                VC_TOOLS_COMPONENT,
                "-property",
                "installationPath",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        roots.extend(
            Path(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
    else:
        print(f"vswhere not found at {vswhere}", flush=True)
    vs_install_dir = os.environ.get("VSINSTALLDIR")
    if vs_install_dir:
        roots.append(Path(vs_install_dir))
    return list(dict.fromkeys(roots))


def redist_roots(installations: Sequence[Path]) -> list[Path]:
    """Return the ``VC\\Redist\\MSVC`` directories to search."""
    roots = [install / "VC" / "Redist" / "MSVC" for install in installations]
    tools_redist = os.environ.get("VCTOOLSREDISTDIR")
    if tools_redist:
        # VCToolsRedistDir points at ...\VC\Redist\MSVC\<version>\ itself.
        roots.append(Path(tools_redist).parent)
    return [root for root in dict.fromkeys(roots) if root.is_dir()]


def find_runtime(
    roots: Sequence[Path], dll_name: str, folder_glob: str
) -> tuple[Path, Version]:
    """Find the highest-version copy of a redistributable DLL.

    Args:
        roots: ``VC\\Redist\\MSVC`` directories.
        dll_name: DLL file name, e.g. ``"msvcp140.dll"``.
        folder_glob: Redist sub-folder pattern, e.g. ``"Microsoft.VC*.CRT"``.

    Returns:
        ``(dll_path, runtime_version)`` of the chosen copy.

    Raises:
        FileNotFoundError: If no copy exists under any root.
    """
    candidates: list[tuple[Path, Version]] = []
    for root in roots:
        for dll in sorted(root.glob(f"*/{REDIST_ARCH}/{folder_glob}/{dll_name}")):
            version = read_pe_versions(dll).runtime_version
            print(f"  candidate {dll} (version {format_version(version)})", flush=True)
            candidates.append((dll, version))
    if not candidates:
        searched = ", ".join(str(root) for root in roots) or "<no redist roots>"
        raise FileNotFoundError(f"{dll_name} not found under {searched}")
    return choose_highest(candidates)


def verify_repaired_wheel(wheel: Path) -> list[str]:
    """Check the bundled MSVC runtimes in a repaired wheel.

    Returns:
        A list of error messages (empty if the wheel is fine).
    """
    errors: list[str] = []
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as tmp:
        names = archive.namelist()
        pyds = [name for name in names if name.lower().endswith(".pyd")]
        if not pyds:
            return [f"no .pyd files in {wheel.name}"]
        toolset: Version = (0, 0)
        for name in pyds:
            linker = read_pe_versions(Path(archive.extract(name, tmp))).linker_version
            print(f"  {name}: toolset (linker) {format_version(linker)}", flush=True)
            toolset = max(toolset, linker)
        for dll_name in (CRT_DLL, OMP_DLL):
            bundled = [name for name in names if is_bundled_dll(name, dll_name)]
            if not bundled:
                errors.append(f"{dll_name} is not bundled in {wheel.name}")
                continue
            for name in bundled:
                version = read_pe_versions(
                    Path(archive.extract(name, tmp))
                ).runtime_version
                print(f"  {name}: version {format_version(version)}", flush=True)
                if runtime_too_old(version, toolset):
                    errors.append(
                        f"bundled {name} ({format_version(version)}) is older than "
                        f"the build toolset ({format_version(toolset)})"
                    )
                elif version[:2] < toolset[:2]:
                    print(
                        f"  note: {name} {format_version(version[:2])} < toolset "
                        f"{format_version(toolset)} but same toolset generation",
                        flush=True,
                    )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repair; return the process exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--wheel", required=True, type=Path, help="wheel to repair")
    parser.add_argument("--dest-dir", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)
    wheel: Path = args.wheel
    dest_dir: Path = args.dest_dir

    installations = find_vs_installations()
    print(f"Visual Studio installations: {[str(p) for p in installations]}", flush=True)
    roots = redist_roots(installations)
    print(f"Redist roots: {[str(p) for p in roots]}", flush=True)
    print(f"Searching for {CRT_DLL}:", flush=True)
    crt_dll, crt_version = find_runtime(roots, CRT_DLL, "Microsoft.VC*.CRT")
    print(f"Searching for {OMP_DLL}:", flush=True)
    omp_dll, omp_version = find_runtime(roots, OMP_DLL, "Microsoft.VC*.OpenMP")
    print(f"Using {crt_dll} ({format_version(crt_version)})", flush=True)
    print(f"Using {omp_dll} ({format_version(omp_version)})", flush=True)

    add_path = os.pathsep.join(
        dict.fromkeys([str(crt_dll.parent), str(omp_dll.parent)])
    )
    command = [
        sys.executable,
        "-m",
        "delvewheel",
        "repair",
        "--add-path",
        add_path,
        "--include",
        CRT_DLL,
        "--include",
        OMP_DLL,
        "-w",
        str(dest_dir),
        str(wheel),
    ]
    print(f"Running: {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, check=True)

    repaired = dest_dir / wheel.name
    if not repaired.is_file():
        print(f"error: repaired wheel {repaired} not found", file=sys.stderr)
        return 1
    print(f"Verifying {repaired}:", flush=True)
    errors = verify_repaired_wheel(repaired)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
