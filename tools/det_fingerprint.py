"""Cross-platform fingerprint of the deterministic numerics layer and KAK synthesis.

Every number this script feeds to the native module is produced
platform-independently (PCG64 bit streams, exact constants and the
deterministic C++ QR / products / math functions), so the outputs must be
bit-identical on every platform and build (macOS arm64, Linux x86_64 with the
AVX2 or the scalar module, Linux aarch64, Windows AMD64).  Run it on two
machines and compare the digests (or the full dumps):

    python tools/det_fingerprint.py --random 100000 --dump out.txt
    python tools/det_fingerprint.py --compare mac.txt linux.txt

``--blocks FILE`` adds externally supplied 4×4 matrices (one per line, 32
``float.hex`` values: real and imaginary parts in row-major order), e.g. the
blocks harvested from benchmark circuits on one machine with ``--harvest``.

Sections of the dump:

* ``scalar``: the correctly rounded math functions on fixed inputs;
* ``kak``: Weyl coordinates, phase and local factors of ``_kak4``;
* ``ops/<basis>``: the synthesized gate sequence (names, qubits and
  ``float.hex`` parameters) for several target bases and template switches.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

_BASES = (("cz", "zxz", False), ("cz", "zsx", True), ("cx", "zsx", False))


def _core() -> Any:
    from quspect._native_loader import load_core

    core = load_core()
    if core is None or not hasattr(core, "_detmath_qr"):
        raise SystemExit("the quspect extension module with detmath is required")
    return core


def _hex_matrix(m: np.ndarray) -> str:
    return " ".join(f"{z.real.hex()},{z.imag.hex()}" for z in np.asarray(m).ravel())


def _parse_matrix(line: str) -> np.ndarray:
    vals = [float.fromhex(t) for t in line.replace(",", " ").split()]
    if len(vals) != 32:
        raise ValueError("expected 32 float.hex values per matrix")
    arr = np.array(vals[0::2]) + 1j * np.array(vals[1::2])
    return np.ascontiguousarray(arr.reshape(4, 4))


class _Builder:
    """Platform-independent construction of test unitaries."""

    def __init__(self, core: Any, synth: Any, seed: int) -> None:
        self.core = core
        self.synth = synth
        self.rng = np.random.Generator(np.random.PCG64(seed))
        sq2 = 1.0 / math.sqrt(2.0)  # sqrt 与除法是正确舍入的
        self.magic = (
            np.array(
                [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
                dtype=complex,
            )
            * sq2
        )
        self.magic_dag = np.ascontiguousarray(self.magic.conj().T)

    def mul(self, *mats: np.ndarray) -> np.ndarray:
        out = np.ascontiguousarray(mats[0], dtype=complex)
        for m in mats[1:]:
            out = self.core._detmath_matmul(out, np.ascontiguousarray(m, dtype=complex))
        return np.asarray(out)

    def unitary(self, n: int) -> np.ndarray:
        # 均匀分布的 [-1, 1) 由 PCG64 的整数位精确得到；QR 为确定性实现
        raw = (
            self.rng.random((n, n)) * 2.0
            - 1.0
            + 1j * (self.rng.random((n, n)) * 2.0 - 1.0)
        )
        q, _ = self.core._detmath_qr(np.ascontiguousarray(raw))
        return np.asarray(q)

    def local(self) -> np.ndarray:
        return np.asarray(self.synth.kron2(self.unitary(2), self.unitary(2)))

    def canonical(self, a: float, b: float, c: float) -> np.ndarray:
        phases = [
            a - b + c,
            a + b - c,
            -a - b - c,
            -a + b + c,
        ]  # _CAN_PHASES @ (a, b, c)
        cos = self.core._detmath_scalar("cos", np.array(phases))
        sin = self.core._detmath_scalar("sin", np.array(phases))
        d = np.diag(cos + 1j * sin)
        return self.mul(self.magic, d, self.magic_dag)


def corpus(
    core: Any, synth: Any, n_random: int, seed: int = 20261101
) -> Iterator[tuple[str, np.ndarray]]:
    """Special points (with random local dressings) followed by random unitaries."""
    b = _Builder(core, synth, seed)
    pi4 = math.pi / 4
    eye = np.eye(4, dtype=complex)
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    cx = eye[[0, 1, 3, 2]]
    swap = eye[[0, 2, 1, 3]]
    iswap = np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    cores: list[tuple[str, np.ndarray]] = [
        ("identity", eye),
        ("cz", cz),
        ("cx", cx),
        ("swap", swap),
        ("iswap", iswap),
        ("swap_cz", b.mul(swap, cz)),
        ("sqrt_iswap", b.canonical(pi4 / 2, pi4 / 2, 0.0)),
        ("b_gate", b.canonical(pi4, pi4 / 2, 0.0)),
        ("heis_pos", b.canonical(0.3, 0.3, 0.3)),
        ("heis_neg", b.canonical(0.3, 0.3, -0.3)),
        ("c_zero", b.canonical(0.4, 0.2, 0.0)),
        ("cp_class", b.canonical(0.4, 0.0, 0.0)),
        ("face_a", b.canonical(pi4, 0.2, -0.1)),
        ("ab_equal", b.canonical(0.5, 0.25, 0.25)),
        ("bc_mirror", b.canonical(0.5, 0.25, -0.25)),
    ]
    for eps in (1e-6, 1e-9, 1e-11, 1e-13):
        cores += [
            (f"near_identity_{eps:g}", b.canonical(eps, 0.0, 0.0)),
            (f"near_cz_{eps:g}", b.canonical(pi4 - eps, eps, 0.0)),
            (f"near_swap_{eps:g}", b.canonical(pi4, pi4 - eps, pi4 - 2 * eps)),
            (f"near_degenerate_{eps:g}", b.canonical(0.3 + eps, 0.3, 0.3 - eps)),
            (f"near_c_zero_{eps:g}", b.canonical(0.4, 0.2, eps)),
        ]
    for name, m in cores:
        yield name, np.ascontiguousarray(m)
        for k in range(3):
            yield f"{name}_dressed{k}", b.mul(b.local(), m, b.local())
    for k in range(4):
        yield f"local{k}", b.local()
    for k in range(n_random):
        yield f"random{k}", b.unitary(4)


def _ops_text(ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]]) -> str:
    return ";".join(
        f"{name}{''.join(map(str, q))}" + "".join(f":{float(p).hex()}" for p in params)
        for name, q, params in ops
    )


def scalar_records(core: Any) -> Iterator[str]:
    rng = np.random.Generator(np.random.PCG64(7))
    wide = rng.random(4096) * 200.0 - 100.0
    unit = rng.random(4096) * 2.0 - 1.0
    tiny = (rng.random(512) - 0.5) * 1e-8
    huge = (rng.random(512) - 0.5) * 1e12
    for name in ("sin", "cos", "tan", "exp", "log", "atan", "asin", "acos"):
        x = np.concatenate(
            [unit] if name in ("asin", "acos") else [wide, unit, tiny, huge]
        )
        if name == "log":
            x = np.abs(x) + 1e-300
        if name == "exp":
            x = x[np.abs(x) < 700]
        y = core._detmath_scalar(name, x)
        yield f"scalar {name} " + hashlib.sha256(np.asarray(y).tobytes()).hexdigest()
    for name in ("atan2", "hypot"):
        y = core._detmath_scalar(name, wide, unit[: len(wide)] * 30.0)
        yield f"scalar {name} " + hashlib.sha256(np.asarray(y).tobytes()).hexdigest()


def kak_records(
    native: Any, mats: Iterator[tuple[str, np.ndarray]], stats: dict[str, list[float]]
) -> Iterator[str]:
    import quspect.synthesis.two_qubit as tq

    synth = native._synth
    bases = [(tq.TwoQubitBasis(e, x), flag) for e, x, flag in _BASES]
    for label, u in mats:
        phase, k1, v, k2 = synth.kak4(u)
        yield (
            f"kak {label} {float(phase).hex()} "
            + ",".join(float(x).hex() for x in v)
            + " "
            + hashlib.sha256(
                np.asarray(k1).tobytes() + np.asarray(k2).tobytes()
            ).hexdigest()[:16]
        )
        for basis, flag in bases:
            old = tq._SWAP_FACE_TEMPLATE
            tq._SWAP_FACE_TEMPLATE = flag
            try:
                ops, uses = native.synthesize(u, 1e-9, 1e-9, basis)
            finally:
                tq._SWAP_FACE_TEMPLATE = old
            if stats is not None:
                err = tq._distance_up_to_phase(u, tq._ops_unitary(ops, basis))
                stats.setdefault(f"{basis.entangler}/{basis.euler}", []).append(err)
            tag = f"ops/{basis.entangler}-{basis.euler}-{int(flag)}"
            yield f"{tag} {label} {int(uses)} {_ops_text(ops)}"


def harvest(path: str, files: list[str], limit: int) -> None:
    """Write the distinct 4×4 blocks synthesized while transpiling QASM files.

    Each file is transpiled at levels 1–3 onto a line and a full coupling map
    with the native gates ``{cz, rz, sx, x}``.
    """
    import quspect.synthesis.two_qubit as tq
    from quspect.ir import qasm2_loads
    from quspect.transpiler import CompilationTarget, transpile

    seen: dict[bytes, np.ndarray] = {}
    original = tq._cached_exact_ops

    def spy(mkey: object, unitary: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        arr = np.ascontiguousarray(np.asarray(unitary, dtype=complex))
        if len(seen) < limit:
            seen.setdefault(arr.tobytes(), arr)
        return original(mkey, unitary, *args, **kwargs)

    tq._cached_exact_ops = spy  # type: ignore[assignment]
    try:
        for f in files:
            if len(seen) >= limit:
                break
            try:
                with open(f) as fh:
                    circuit = qasm2_loads(fh.read())
                n = circuit.num_qubits
                for edges in ([(i, i + 1) for i in range(n - 1)], None):
                    if edges is None:
                        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
                    target = CompilationTarget.from_coupling_map(
                        n, edges, native_gates={"cz", "rz", "sx", "x"}
                    )
                    for level in (1, 2, 3):
                        transpile(circuit, target, optimization_level=level, seed=0)
            except Exception as exc:  # noqa: BLE001 - 跳过不支持的线路
                print(f"skip {f}: {exc}", file=sys.stderr)
    finally:
        tq._cached_exact_ops = original  # type: ignore[assignment]
    with open(path, "w") as fh:
        for arr in seen.values():
            fh.write(_hex_matrix(arr) + "\n")
    print(f"{len(seen)} blocks from {len(files)} files -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--random", type=int, default=2000, help="number of random unitaries"
    )
    ap.add_argument("--blocks", help="file with extra 4x4 matrices (float.hex)")
    ap.add_argument("--dump", help="write every record to this file")
    ap.add_argument(
        "--stats", action="store_true", help="print reconstruction-error statistics"
    )
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two dumps")
    ap.add_argument(
        "--harvest", metavar="FILE", help="harvest benchmark blocks into FILE"
    )
    ap.add_argument("--harvest-limit", type=int, default=20000)
    ap.add_argument("qasm", nargs="*", help="QASM files for --harvest")
    args = ap.parse_args(argv)

    if args.compare:
        with open(args.compare[0]) as fa, open(args.compare[1]) as fb:
            a = fa.read().splitlines()
            b = fb.read().splitlines()
        diff = [i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y]
        print(f"{len(a)} vs {len(b)} records, {len(diff)} differ")
        for i in diff[:20]:
            print(f"- {a[i]}\n+ {b[i]}")
        return 0 if not diff and len(a) == len(b) else 1
    if args.harvest:
        harvest(args.harvest, args.qasm, args.harvest_limit)
        return 0

    import quspect.synthesis.two_qubit as tq

    core = _core()
    native = tq._native_kak()
    if native is None:
        raise SystemExit("native KAK backend unavailable")
    mats: Iterator[tuple[str, np.ndarray]] = corpus(core, native._synth, args.random)
    if args.blocks:
        with open(args.blocks) as fh:
            extra = [
                (f"block{i}", _parse_matrix(line))
                for i, line in enumerate(fh)
                if line.strip()
            ]

        def chained() -> Iterator[tuple[str, np.ndarray]]:
            yield from corpus(core, native._synth, args.random)
            yield from extra

        mats = chained()
    stats: dict[str, list[float]] = {}
    digest = hashlib.sha256()
    sections: dict[str, Any] = {}
    out = open(args.dump, "w") if args.dump else None  # noqa: SIM115
    count = 0
    t0 = time.perf_counter()
    try:
        for rec in (
            *scalar_records(core),
            *kak_records(native, mats, stats if args.stats else None),
        ):
            line = rec + "\n"
            digest.update(line.encode())
            key = rec.split(" ", 1)[0]
            sections.setdefault(key, hashlib.sha256()).update(line.encode())
            count += 1
            if out is not None:
                out.write(line)
    finally:
        if out is not None:
            out.close()
    elapsed = time.perf_counter() - t0
    for key, h in sorted(sections.items()):
        print(f"{key:24s} {h.hexdigest()}")
    print(f"{'ALL':24s} {digest.hexdigest()}  ({count} records, {elapsed:.1f} s)")
    if args.stats:
        for key, errs in sorted(stats.items()):
            arr = np.array(errs)
            print(
                f"error {key}: n={arr.size} max={arr.max():.3e} "
                f"p99={np.quantile(arr, 0.99):.3e} median={np.median(arr):.3e}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
