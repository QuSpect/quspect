"""Whole-compilation golden digests: ``transpile`` is bit-identical everywhere.

The compiler's numerics run on the deterministic C++ layer (fixed-order
arithmetic, own small-matrix linear algebra, correctly rounded math
functions), the routing heuristics only compare exactly computed scores and
parallel work is merged in a fixed order.  The output of
:func:`quspect.transpile` -- gate names, qubits and every parameter bit
(``float.hex``), plus the layouts -- is therefore the same on every platform
and build: macOS arm64, Linux x86_64 (AVX2 and scalar modules), Linux aarch64
and Windows AMD64, for any number of compile threads and any
``PYTHONHASHSEED``.

The circuits are generated here from exact constants (and ``random.Random``,
whose stream is platform independent) and parsed by the native QASM parser,
so the inputs are platform independent as well.

If the compiler's output is changed on purpose, regenerate the digests with
``python tests/transpiler/test_golden_transpile.py`` and check that every
platform agrees (the CI runs this module on all of them).

Level 3 is by far the most expensive level.  The golden digests are one test
per (level, circuit), so that ``pytest -n`` spreads them over the workers.
The invariance checks (compile threads, ``PYTHONHASHSEED``, platform math)
come in two versions: a quick one that compiles a few level-3 cases
(``_QUICK_LEVEL3``) and runs on every pull request, and the exhaustive one,
marked ``slow``, that runs nightly and in release builds.

``QUSPECT_TEST_GOLDEN_LEVELS`` (comma separated, e.g. ``1,2``) restricts this
module to the given optimization levels; the other levels are skipped.  It is
set where level 3 is already covered by another job (the wheel tests and the
scalar-module re-check on pull requests).  Unset, every level runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from quspect._native_loader import load_core
from quspect.ir import qasm2_loads
from quspect.transpiler import CompilationTarget, transpile

pytestmark = pytest.mark.skipif(
    load_core() is None, reason="extension module not built"
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class _Qasm:
    def __init__(self, n: int) -> None:
        self.n = n
        self.lines: list[str] = []

    def g(self, name: str, *qubits: int, params: tuple[float, ...] = ()) -> None:
        # repr 是最短的可往返十进制表示，解析器（strtod）正确舍入回同一个 double
        p = f"({','.join(repr(float(x)) for x in params)})" if params else ""
        self.lines.append(f"{name}{p} {','.join(f'q[{q}]' for q in qubits)};")

    def text(self) -> str:
        head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.n}];"]
        return "\n".join(head + self.lines) + "\n"


def _qft(n: int) -> str:
    b = _Qasm(n)
    for i in range(n):
        b.g("h", i)
        for j in range(i + 1, n):
            b.g("cp", j, i, params=(math.pi / 2 ** (j - i),))
    for i in range(n // 2):
        b.g("swap", i, n - 1 - i)
    return b.text()


def _qaoa(n: int) -> str:
    edges = [(i, (i + 1) % n) for i in range(n)] + [
        (i, i + n // 2) for i in range(n // 2)
    ]
    b = _Qasm(n)
    for q in range(n):
        b.g("h", q)
    for gamma, beta in ((0.7853, 0.3927), (1.1, 0.25)):
        for u, v in edges:
            b.g("cx", u, v)
            b.g("rz", v, params=(2 * gamma,))
            b.g("cx", u, v)
        for q in range(n):
            b.g("rx", q, params=(2 * beta,))
    return b.text()


def _heisenberg(n: int, steps: int = 2) -> str:
    b = _Qasm(n)
    for _ in range(steps):
        for i in range(n - 1):
            j = i + 1
            for pre, post in (("h", "h"), ("rx", "rx")):
                args = (math.pi / 2,) if pre == "rx" else ()
                inv = (-math.pi / 2,) if post == "rx" else ()
                b.g(pre, i, params=args)
                b.g(pre, j, params=args)
                b.g("cx", i, j)
                b.g("rz", j, params=(0.2,))
                b.g("cx", i, j)
                b.g(post, i, params=inv)
                b.g(post, j, params=inv)
            b.g("cx", i, j)
            b.g("rz", j, params=(0.2,))
            b.g("cx", i, j)
    return b.text()


def _random(n: int, seed: int) -> str:
    rng = random.Random(seed)
    b = _Qasm(n)
    for _ in range(n):
        for q in range(n):
            kind = rng.choice(("h", "t", "s", "rz", "ry", "u3"))
            if kind in ("rz", "ry"):
                b.g(kind, q, params=(rng.uniform(-math.pi, math.pi),))
            elif kind == "u3":
                b.g(
                    kind,
                    q,
                    params=tuple(rng.uniform(-math.pi, math.pi) for _ in range(3)),
                )
            else:
                b.g(kind, q)
        qs = list(range(n))
        rng.shuffle(qs)
        for k in range(0, n - 1, 2):
            if rng.random() < 0.7:
                two = rng.choice(("cx", "cz", "rzz"))
                b.g(two, qs[k], qs[k + 1], params=(0.3,) if two == "rzz" else ())
    return b.text()


def _adder(k: int) -> str:
    """Cuccaro ripple-carry adder on ``2k + 2`` qubits (Toffoli-heavy)."""
    n = 2 * k + 2
    b = _Qasm(n)
    a = [2 * i + 1 for i in range(k)]
    bb = [2 * i + 2 for i in range(k)]
    for q in a[::2] + bb[1::2]:
        b.g("x", q)

    def maj(x: int, y: int, z: int) -> None:
        b.g("cx", z, y)
        b.g("cx", z, x)
        b.g("ccx", x, y, z)

    def uma(x: int, y: int, z: int) -> None:
        b.g("ccx", x, y, z)
        b.g("cx", z, x)
        b.g("cx", x, y)

    maj(0, bb[0], a[0])
    for i in range(1, k):
        maj(a[i - 1], bb[i], a[i])
    b.g("cx", a[k - 1], n - 1)
    for i in range(k - 1, 0, -1):
        uma(a[i - 1], bb[i], a[i])
    uma(0, bb[0], a[0])
    return b.text()


def _three_qubit_mix(n: int, seed: int) -> str:
    """Dense three-qubit blocks (resynthesized by ``ThreeQubitBlockPass``)."""
    rng = random.Random(seed)
    b = _Qasm(n)
    for layer in range(4):
        for start in range(layer % 2, n - 2, 2):
            x, y, z = start, start + 1, start + 2
            b.g("u3", x, params=tuple(rng.uniform(-math.pi, math.pi) for _ in range(3)))
            b.g("ccx", x, y, z)
            b.g("crz", z, x, params=(rng.uniform(-math.pi, math.pi),))
            b.g("cswap", y, x, z)
            b.g("ry", y, params=(rng.uniform(-math.pi, math.pi),))
            b.g("cx", y, z)
    return b.text()


CIRCUITS: dict[str, str] = {
    "qft_6": _qft(6),
    "qaoa_6": _qaoa(6),
    "heisenberg_6": _heisenberg(6),
    "random_6": _random(6, seed=3),
    "adder_6": _adder(2),
    "three_qubit_5": _three_qubit_mix(5, seed=11),
    "qft_9": _qft(9),
    "random_10": _random(10, seed=8),
    "adder_10": _adder(4),
    "three_qubit_8": _three_qubit_mix(8, seed=2),
}

BASES = {
    "cz": ("cz", "rz", "sx", "x"),
    "cx": ("cx", "rz", "sx", "x"),
    "std": ("cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"),
}
TOPOLOGIES = ("line", "ring", "grid", "full")
LEVELS = (1, 2, 3)

LEVELS_ENV = "QUSPECT_TEST_GOLDEN_LEVELS"


def _enabled_levels() -> tuple[int, ...]:
    """The levels this run compiles at: all of them unless ``LEVELS_ENV`` is set."""
    raw = os.environ.get(LEVELS_ENV)
    if raw is None:
        return LEVELS
    try:
        chosen = {int(part) for part in raw.split(",")}
    except ValueError:
        chosen = set()
    if not chosen or not chosen <= set(LEVELS):
        raise ValueError(
            f"{LEVELS_ENV}={raw!r}: expected a comma-separated subset of {LEVELS}"
        )
    return tuple(level for level in LEVELS if level in chosen)


ENABLED_LEVELS = _enabled_levels()


def _edges(kind: str, n: int) -> tuple[int, list[tuple[int, int]]]:
    if kind == "line":
        return n, [(i, i + 1) for i in range(n - 1)]
    if kind == "ring":
        return n, [(i, (i + 1) % n) for i in range(n)]
    if kind == "full":
        return n, [(i, j) for i in range(n) for j in range(i + 1, n)]
    cols = math.isqrt(n - 1) + 1
    rows = -(-n // cols)
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return rows * cols, edges


def _record(name: str, kind: str, basis: str, level: int) -> str:
    circuit = qasm2_loads(CIRCUITS[name])
    n, edges = _edges(kind, circuit.num_qubits)
    target = CompilationTarget.from_coupling_map(
        n, edges, native_gates=set(BASES[basis])
    )
    res = transpile(circuit, target, optimization_level=level, seed=0)
    gates = ";".join(
        (ins.gate.name if ins.gate is not None else ins.type.name.lower())
        + "".join(f",{q}" for q in ins.qubits)
        + "".join(
            f":{float(p).hex()}"
            for p in (ins.gate.params if ins.gate is not None else ())
        )
        for ins in res.circuit
    )
    return f"{kind}@{basis} {res.initial_layout} {res.final_layout} {gates}"


def digests(level: int, names: tuple[str, ...] = tuple(CIRCUITS)) -> dict[str, str]:
    """``{circuit: sha256 prefix}`` of all topologies and gate sets at ``level``."""
    out = {}
    for name in names:
        records = [
            _record(name, kind, basis, level) for kind in TOPOLOGIES for basis in BASES
        ]
        out[name] = hashlib.sha256("\n".join(records).encode()).hexdigest()[:16]
    return out


# 由 ``python tests/transpiler/test_golden_transpile.py`` 生成。
GOLDEN: dict[int, dict[str, str]] = {
    1: {
        "qft_6": "cc2374665e61b47b",
        "qaoa_6": "e821d3c57f0ade51",
        "heisenberg_6": "4d94c657e2c5e689",
        "random_6": "5c3f15a48828b86b",
        "adder_6": "72940f9d2c59a961",
        "three_qubit_5": "86fa57ab57ecb613",
        "qft_9": "cecb114438e6c88c",
        "random_10": "ec93e4b4a6bbfcb3",
        "adder_10": "7b2f1b45078e00ed",
        "three_qubit_8": "c92adad084d14177",
    },
    2: {
        "qft_6": "51fd9ad23b2422f2",
        "qaoa_6": "2c4dbc6bd622c750",
        "heisenberg_6": "f30961cec70169b7",
        "random_6": "7e30a14e849ea97a",
        "adder_6": "ac91b581a90f21b1",
        "three_qubit_5": "26ab34a56acc17d9",
        "qft_9": "992d646acffba0fc",
        "random_10": "073b4655ba0ed17d",
        "adder_10": "6b993715a361fef4",
        "three_qubit_8": "0eb12243364e4e36",
    },
    3: {
        "qft_6": "6cf13558f195fbe6",
        "qaoa_6": "0c27f9d376674566",
        "heisenberg_6": "dc5f5a1003058005",
        "random_6": "cbe0dfd25640b9ed",
        "adder_6": "a1b61b2addf66e1e",
        "three_qubit_5": "a7d0529639f27236",
        "qft_9": "3098ae82ae4001b3",
        "random_10": "433bf20739c2e63a",
        "adder_10": "a7e1e505f2664315",
        "three_qubit_8": "3e9cada8dc9c932f",
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cold_caches() -> Iterator[None]:
    """Start from empty synthesis caches, as a fresh process does.

    Other tests switch numerics backends (Python reference numerics) and
    leave results computed that way in the exact-input caches, which a later
    compilation in the same worker would reuse.
    """
    _clear_caches()
    yield
    _clear_caches()


def _clear_caches() -> None:
    from quspect.gates import synthesis as one_qubit_synthesis
    from quspect.synthesis import two_qubit
    from quspect.transpiler import _consolidate, _toffoli

    for cache in (
        two_qubit._SYNTH_2Q_CACHE,
        two_qubit._APPROX_2Q_CACHE,
        two_qubit._SWAP_FACE_CACHE,
        two_qubit._NUM_CZ_CACHE,
        one_qubit_synthesis._SYNTH_1Q_CACHE,
        _consolidate._EULER_CACHE,
        _toffoli._CONTEXT_COST_CACHE,
        _toffoli._CCX_CHOICE_CACHE,
    ):
        cache.clear()


def _require_level(level: int) -> None:
    if level not in ENABLED_LEVELS:
        pytest.skip(f"level {level} is excluded by {LEVELS_ENV}")


def test_golden_table_matches_circuits() -> None:
    """``GOLDEN`` has exactly one digest per (level, circuit)."""
    assert tuple(GOLDEN) == LEVELS
    for level in LEVELS:
        assert set(GOLDEN[level]) == set(CIRCUITS)
    assert sorted(_GOLDEN_CASES) == sorted((lv, n) for lv in LEVELS for n in CIRCUITS)


# 按（级别, 电路）参数化：每个摘要仍覆盖该电路的全部 12 个目标（拓扑 x 门集），
# 与 GOLDEN 的粒度一致。节点名为 test_transpile_golden_digests[<级别>-<电路>]。
# 顺序：最慢的排在最前（第 3 级，其中最慢的几个电路在先）。并行运行时墙钟时间的下限是
# 最慢的单个测试，它越早开始，末尾等它的时间越短；顺序不影响节点名与覆盖范围。
_SLOWEST_FIRST = ("three_qubit_8", "three_qubit_5", "random_10", "qft_9", "adder_10")
_GOLDEN_CASES = [
    (level, name)
    for level in reversed(LEVELS)
    for name in (*_SLOWEST_FIRST, *(n for n in CIRCUITS if n not in _SLOWEST_FIRST))
]


@pytest.mark.parametrize(
    ("level", "name"), _GOLDEN_CASES, ids=[f"{lv}-{n}" for lv, n in _GOLDEN_CASES]
)
def test_transpile_golden_digests(level: int, name: str) -> None:
    """Every platform and build produces exactly these compilations."""
    _require_level(level)
    assert digests(level, (name,))[name] == GOLDEN[level][name]


# 不变性检查精简版在第 3 级编译的（电路, 拓扑, 门集）：含随机电路、Toffoli 电路与
# 三比特块重综合。精简版在第 1、2 级仍覆盖对应全量版的全部电路与全部 12 个目标。
_QUICK_LEVEL3 = (
    ("random_6", "line", "cx"),
    ("random_6", "grid", "cz"),
    ("adder_6", "line", "cx"),
    ("adder_6", "grid", "cz"),
    ("three_qubit_5", "full", "cz"),
)


def _quick_level3_records() -> list[str]:
    if 3 not in ENABLED_LEVELS:
        return []
    return [_record(name, kind, basis, 3) for name, kind, basis in _QUICK_LEVEL3]


# 并行路径（SABRE 链、LNS 窗口投机、SU(4) 路由配置、调度、gauge 束搜索
# 分组）都按下标写结果、按固定顺序合并；线程数只影响耗时。
_THREAD_CASES = (("random_6", 3), ("three_qubit_5", 3), ("qaoa_6", 2))


def _set_compile_threads(monkeypatch: pytest.MonkeyPatch, threads: str | None) -> None:
    if threads is None:
        monkeypatch.delenv("QUSPECT_COMPILE_THREADS", raising=False)
    else:
        monkeypatch.setenv("QUSPECT_COMPILE_THREADS", threads)


@pytest.mark.slow
@pytest.mark.parametrize("threads", ["1", "4", None])
def test_compile_threads_do_not_change_output(
    monkeypatch: pytest.MonkeyPatch, threads: str | None
) -> None:
    _set_compile_threads(monkeypatch, threads)
    for name, level in _THREAD_CASES:
        if level in ENABLED_LEVELS:
            assert digests(level, (name,))[name] == GOLDEN[level][name]


def test_compile_threads_do_not_change_output_quick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quick version: level 2 against ``GOLDEN``, a few level-3 cases compared."""
    names = tuple(name for name, _ in _THREAD_CASES)
    level3 = []
    for threads in ("1", "4", None):
        _set_compile_threads(monkeypatch, threads)
        if 2 in ENABLED_LEVELS:
            assert digests(2, names) == {n: GOLDEN[2][n] for n in names}
        level3.append(_quick_level3_records())
    assert level3[0] == level3[1] == level3[2]


def test_large_parallel_compilation_is_thread_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A circuit large enough for the parallel scheduling and gauge search.

    About 15k single-qubit runs: the initial gauge beam search is split into
    1, 2 and 3 wire groups, and the depth scheduler runs its parallel pass.
    """
    text = _random(14, seed=5)
    body = "".join(text.splitlines(keepends=True)[3:])
    circuit = qasm2_loads(text + body * 29)
    n, edges = _edges("grid", circuit.num_qubits)
    target = CompilationTarget.from_coupling_map(
        n, edges, native_gates={"cz", "rz", "sx", "x"}
    )
    outs = []
    for threads in ("1", "2", "8"):
        monkeypatch.setenv("QUSPECT_COMPILE_THREADS", threads)
        res = transpile(circuit, target, optimization_level=2, seed=1)
        outs.append(
            (
                res.initial_layout,
                res.final_layout,
                [
                    (
                        ins.gate.name,
                        ins.qubits,
                        [float(p).hex() for p in ins.gate.params],
                    )
                    for ins in res.circuit
                ],
            )
        )
    assert outs[0] == outs[1] == outs[2]


_HASH_CASES = ("random_6", "adder_6", "three_qubit_5")
_HASH_SEEDS = ("0", "4242")


def _run_with_hash_seed(seed: str, expression: str) -> str:
    """JSON of ``expression`` (``t`` is this module) under ``PYTHONHASHSEED=seed``."""
    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_golden_transpile as t\n"
        f"print(json.dumps({expression}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONHASHSEED=seed),
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_hash_seed_does_not_change_output() -> None:
    """Set/dict iteration orders that depend on ``hash()`` never reach the output.

    Each child process uses a different ``PYTHONHASHSEED`` (``str`` hashes,
    and with them the iteration order of sets of gate names, change).
    """
    levels = tuple(level for level in (2, 3) if level in ENABLED_LEVELS)
    expression = f"{{l: t.digests(l, {_HASH_CASES!r}) for l in {levels!r}}}"
    expected = json.dumps(
        {level: {n: GOLDEN[level][n] for n in _HASH_CASES} for level in levels}
    )
    for seed in _HASH_SEEDS:
        assert _run_with_hash_seed(seed, expression) == expected


def test_hash_seed_does_not_change_output_quick() -> None:
    """Quick version: level 2 against ``GOLDEN``, a few level-3 cases compared.

    The level-3 records of each child process must equal the ones computed in
    this process (whose hash seed is whatever pytest was started with).
    """
    levels = tuple(level for level in (2,) if level in ENABLED_LEVELS)
    expression = (
        f"[{{l: t.digests(l, {_HASH_CASES!r}) for l in {levels!r}}}, "
        "t._quick_level3_records()]"
    )
    expected = json.dumps(
        [
            {level: {n: GOLDEN[level][n] for n in _HASH_CASES} for level in levels},
            _quick_level3_records(),
        ]
    )
    for seed in _HASH_SEEDS:
        assert _run_with_hash_seed(seed, expression) == expected


def _platform_math_calls(
    names: tuple[str, ...] = tuple(CIRCUITS),
    levels: tuple[int, ...] = LEVELS,
    extra: Callable[[], object] | None = None,
) -> list[str]:
    """Call sites of platform math (``math``/``cmath``/numpy) while compiling.

    Compiles ``names`` on every target at ``levels`` and then runs ``extra``
    (further compilations to watch).

    Diagnostic for a platform whose digests differ: every such call outside
    module constants is a candidate source of the difference.
    """
    import cmath
    import collections

    import numpy as np

    calls: collections.Counter[str] = collections.Counter()
    saved = []

    def wrap(mod: object, name: str, label: str) -> None:
        f = getattr(mod, name)

        def w(*a: object, **k: object) -> object:
            fr = sys._getframe(1)
            where = fr.f_code.co_filename.replace("\\", "/")
            calls[f"{label} {where}:{fr.f_lineno}"] += 1
            return f(*a, **k)

        saved.append((mod, name, f))
        setattr(mod, name, w)

    for n in ("sin", "cos", "tan", "atan", "atan2", "acos", "asin", "exp", "log"):
        wrap(math, n, "math." + n)
        wrap(np, n if hasattr(np, n) else "arctan2", "np." + n)
    for n in ("hypot", "pow"):
        wrap(math, n, "math." + n)
    for n in ("exp", "phase", "polar", "rect", "log", "sqrt"):
        wrap(cmath, n, "cmath." + n)
    for n in ("angle", "arctan2", "abs", "hypot"):
        wrap(np, n, "np." + n)
    for n in ("svd", "eig", "eigh", "eigvals", "eigvalsh", "qr", "det", "inv"):
        wrap(np.linalg, n, "np.linalg." + n)
    try:
        for level in levels:
            digests(level, names)
        if extra is not None:
            extra()
    finally:
        for mod, name, f in reversed(saved):
            setattr(mod, name, f)
    return [f"{v:8d} {k}" for k, v in calls.most_common()]


_MATH_CASES = ("random_6", "adder_6", "three_qubit_5")


def _unexpected_platform_math(calls: list[str]) -> list[str]:
    allowed = ("_gauge.py:", "_schedule.py:", "gates/synthesis.py:")
    return [c for c in calls if not (" np.abs " in c and any(a in c for a in allowed))]


@pytest.mark.slow
def test_compilation_uses_no_platform_math() -> None:
    """No platform math function or LAPACK routine runs while compiling.

    The only allowed calls are ``np.abs`` threshold checks on exactly
    computed values (commutators with Pauli matrices, constant gate
    matrices), where a last-bit difference of ``hypot`` cannot change the
    decision.
    """
    calls = _platform_math_calls(_MATH_CASES, ENABLED_LEVELS)
    assert not _unexpected_platform_math(calls)


def test_compilation_uses_no_platform_math_quick() -> None:
    """Quick version: levels 1 and 2 in full, a few level-3 cases."""
    levels = tuple(level for level in (1, 2) if level in ENABLED_LEVELS)
    calls = _platform_math_calls(_MATH_CASES, levels, _quick_level3_records)
    assert not _unexpected_platform_math(calls)


if __name__ == "__main__":
    # 无参数：摘要（GOLDEN 的格式）；--dump F：全部记录（跨平台逐行 diff）；
    # --calls：编译中调用平台数学库的位置
    import pprint

    args = sys.argv[1:]
    if args[:1] == ["--dump"]:
        with open(args[1], "w") as fh:
            for level in LEVELS:
                for name in CIRCUITS:
                    for kind in TOPOLOGIES:
                        for basis in BASES:
                            rec = _record(name, kind, basis, level)
                            fh.write(f"O{level} {name} {rec}\n")
    elif args[:1] == ["--calls"]:
        print("\n".join(_platform_math_calls()))
    else:
        pprint.pprint(
            {level: digests(level) for level in LEVELS}, sort_dicts=False, width=100
        )
