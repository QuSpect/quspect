"""Run compat_cases.py against envelopes-qc 0.1.5 and quspect._pulse, then diff.

    python tools/pulse_compat/compare.py --ref-python /path/to/ref-venv/bin/python [-v]

Module paths in messages and docstrings (``quspect._pulse`` vs
``envelopes.envelopes_cpp``) are normalized before comparing.  Exit status is
non-zero when any case differs.
"""

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF_MODULE = "envelopes.envelopes_cpp"
NEW_MODULE = "quspect._pulse"


def run(python, out, extra=()):
    cmd = [python, str(HERE / "compat_cases.py"), str(out), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(
            f"{' '.join(cmd)} failed ({res.returncode}):\n{res.stdout}\n{res.stderr}"
        )
    print(res.stdout.strip())
    with open(out, "rb") as f:
        return pickle.load(f)["results"]


def normalize(value, old=NEW_MODULE, new=REF_MODULE):
    """Replace the module path inside every string of a nested result."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, tuple):
        return tuple(normalize(v, old, new) for v in value)
    if isinstance(value, list):
        return [normalize(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v, old, new) for k, v in value.items()}
    return value


def find_arrays(a, b, path=""):
    """Yield (path, ref array, new array) for ndarray leaves whose bytes differ."""
    if (
        isinstance(a, tuple)
        and len(a) == 4
        and a[0] == "ndarray"
        and isinstance(b, tuple)
        and b[:1] == ("ndarray",)
    ):
        if a != b:
            yield path, a, b
        return
    if (
        isinstance(a, (tuple, list))
        and isinstance(b, (tuple, list))
        and len(a) == len(b)
    ):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            yield from find_arrays(x, y, f"{path}[{i}]")


def describe_array_diff(a, b):
    _, dt_a, shape_a, buf_a = a
    _, dt_b, shape_b, buf_b = b
    if (dt_a, shape_a) != (dt_b, shape_b):
        return f"dtype/shape {dt_a}{shape_a} vs {dt_b}{shape_b}"
    x = np.frombuffer(buf_a, dtype=dt_a)
    y = np.frombuffer(buf_b, dtype=dt_b)
    xs = x.view(np.float32) if x.dtype.kind == "c" else x
    ys = y.view(np.float32) if y.dtype.kind == "c" else y
    differ = xs.view(np.int32) != ys.view(np.int32)
    ulp = np.abs(
        xs.view(np.int32).astype(np.int64) - ys.view(np.int32).astype(np.int64)
    )
    max_diff = float(np.nanmax(np.abs(xs.astype(np.float64) - ys.astype(np.float64))))
    return (
        f"{int(differ.sum())}/{differ.size} float32 words differ, max |diff|="
        f"{max_diff:.3g}, "
        f"max ulp={int(ulp[differ].max()) if differ.any() else 0}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref-python",
        required=True,
        help="interpreter with the reference envelopes-qc installed",
    )
    ap.add_argument("--new-python", default=sys.executable)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ref = run(args.ref_python, Path(tmp) / "ref.pkl", ["--module", REF_MODULE])
        new = run(
            args.new_python,
            Path(tmp) / "new.pkl",
            ["--module", NEW_MODULE, "--path", str(ROOT)],
        )
    new = normalize(new)

    assert ref.keys() == new.keys()
    bad = []
    for name in ref:
        if ref[name] != new[name]:
            bad.append(name)
    groups = {}
    for name in ref:
        groups.setdefault(name.split("/")[0], [0, 0])[0] += 1
    for name in bad:
        groups[name.split("/")[0]][1] += 1
    print(f"\n{len(ref)} cases, {len(bad)} mismatches")
    for g, (n, b) in groups.items():
        print(f"  {g:12s} {n - b:5d}/{n} identical")
    for name in bad[: None if args.verbose else 40]:
        a, b = ref[name], new[name]
        arrays = list(find_arrays(a, b))
        if arrays and a[0] == b[0] == "ok":
            for path, x, y in arrays[:3]:
                print(f"  ✗ {name}{path}: {describe_array_diff(x, y)}")
            if not args.verbose:
                continue
        print(f"  ✗ {name}\n      ref: {str(a)[:400]}\n      new: {str(b)[:400]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
