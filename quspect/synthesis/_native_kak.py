"""Native backend of the two-qubit KAK synthesis.

:mod:`quspect.synthesis.two_qubit` is a chain of small numpy operations:
BLAS products, LAPACK decompositions (whose singular / eigenvectors in
degenerate subspaces depend on the LAPACK build) and element-wise complex
arithmetic whose rounding depends on how numpy and CPython were compiled
(fused multiply-add or not).  The C++ implementation (``_KakSynthesizer``)
reproduces that chain bit for bit by

* calling the very BLAS / LAPACK routines numpy links against (resolved here
  from numpy's build configuration, after checking that numpy's own binaries
  import exactly those symbols), with numpy's arguments and data layouts;
* evaluating every element-wise operation in the order this platform's numpy
  and CPython use; the order is probed here on random inputs and must match
  exactly one candidate.

The backend is enabled only when all of that succeeds, a self-check against
the Python reference passes and the platform is one where the whole
synthesis has been compared against the Python reference on benchmark
corpora (:data:`_VERIFIED_PLATFORMS`); set :data:`BACKEND_ENV` to ``native``
to enable it on other platforms (still subject to the probes and the
self-check) or to ``python`` to disable it.
"""

from __future__ import annotations

import ctypes
import glob
import importlib
import logging
import math
import mmap
import os
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quspect.synthesis.two_qubit import SynthOp, TwoQubitBasis

__all__ = ["BACKEND_ENV", "NativeKak", "load"]

logger = logging.getLogger(__name__)

#: ``auto`` (default), ``native`` or ``python``.
BACKEND_ENV = "QUSPECT_SYNTHESIS_BACKEND"

#: (system, machine, BLAS name) combinations whose native synthesis has been
#: compared against the Python reference on the benchmark corpora.
_VERIFIED_PLATFORMS = frozenset(
    {
        ("Darwin", "arm64", "accelerate"),
        ("Linux", "x86_64", "scipy-openblas"),
    }
)

_C128 = np.dtype(np.complex128)
_FUNCTIONS = (
    "zgemm",
    "zsyrk",
    "dgemv",
    "zdotc_sub",
    "zgesdd",
    "dsyevd",
    "zgetrf",
    "dgetrf",
)
_CBLAS = frozenset({"zgemm", "zsyrk", "dgemv", "zdotc_sub"})


# ---------------------------------------------------------------------------
# BLAS / LAPACK of numpy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scheme:
    """Symbol naming of one BLAS / LAPACK build numpy may be linked against."""

    lapack: str  # format of a LAPACK name, e.g. "{}$NEWLAPACK$ILP64"
    cblas: str  # format of a CBLAS name
    ilp64: bool


_ACCELERATE = (
    _Scheme("{}$NEWLAPACK$ILP64", "cblas_{}$NEWLAPACK$ILP64", True),
    _Scheme("{}$NEWLAPACK", "cblas_{}$NEWLAPACK", False),
)
_SCIPY_OPENBLAS = (
    _Scheme("scipy_{}_64_", "scipy_cblas_{}64_", True),
    _Scheme("scipy_{}_", "scipy_cblas_{}", False),
)


def _binary_mentions(path: str, names: list[bytes]) -> bool:
    """Whether the file at ``path`` contains every byte string of ``names``."""
    try:
        with (
            open(path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        ):
            return all(mm.find(name) >= 0 for name in names)
    except (OSError, ValueError):
        return False


def _numpy_binaries() -> tuple[str, str] | None:
    try:
        core_mod = importlib.import_module("numpy._core._multiarray_umath")
        linalg_mod = importlib.import_module("numpy.linalg._umath_linalg")
    except ImportError:
        return None
    core = getattr(core_mod, "__file__", None)
    linalg = getattr(linalg_mod, "__file__", None)
    if not core or not linalg:
        return None
    return core, linalg


def _openblas_library() -> str | None:
    """Path of the scipy-openblas library bundled with the numpy wheel."""
    pkg = os.path.dirname(np.__file__)
    patterns = [
        os.path.join(os.path.dirname(pkg), "numpy.libs", "libscipy_openblas*"),
        os.path.join(pkg, ".dylibs", "libscipy_openblas*"),
    ]
    found = sorted(p for pattern in patterns for p in glob.glob(pattern))
    return found[0] if len(found) == 1 else None


def _linalg_functions() -> tuple[dict[str, int], bool, str] | None:
    """Addresses of numpy's BLAS / LAPACK routines, integer width and BLAS name."""
    try:
        deps = np.__config__.CONFIG["Build Dependencies"]
        blas = str(deps["blas"]["name"])
        lapack = str(deps["lapack"]["name"])
    except (AttributeError, KeyError, TypeError):
        return None
    if blas != lapack:
        return None
    if blas == "accelerate":
        library = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        schemes = _ACCELERATE
    elif blas == "scipy-openblas":
        found = _openblas_library()
        if found is None:
            return None
        library = found
        schemes = _SCIPY_OPENBLAS
    else:
        return None
    binaries = _numpy_binaries()
    if binaries is None:
        return None
    core_bin, linalg_bin = binaries
    for scheme in schemes:
        # numpy 自己的扩展模块必须导入这些符号（同一个库、同一种整数宽度）
        cblas = [
            scheme.cblas.format(n).encode()
            for n in ("zgemm", "zsyrk", "dgemv", "zdotc_sub")
        ]
        lap = [
            scheme.lapack.format(n).encode()
            for n in ("zgesdd", "dsyevd", "zgetrf", "dgetrf")
        ]
        if not (
            _binary_mentions(core_bin, cblas) and _binary_mentions(linalg_bin, lap)
        ):
            continue
        try:
            lib = ctypes.CDLL(library)
            addresses = {}
            for name in _FUNCTIONS:
                fmt = scheme.cblas if name in _CBLAS else scheme.lapack
                fn = getattr(lib, fmt.format(name))
                addresses[name] = int(ctypes.cast(fn, ctypes.c_void_p).value or 0)
        except (OSError, AttributeError):
            return None
        if not all(addresses.values()):
            return None
        return addresses, scheme.ilp64, blas
    return None


# ---------------------------------------------------------------------------
# Element-wise semantics
# ---------------------------------------------------------------------------


def _bits(x: Any) -> bytes:
    return np.ascontiguousarray(np.asarray(x, dtype=complex)).tobytes()


def _samples(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random complex samples with zeros, signed zeros and exact values mixed in."""
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    special = np.array(
        [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-300, -3.0, np.pi, 1e-17], dtype=float
    )
    re = z.real.copy()
    im = z.imag.copy()
    k = n // 4
    re[:k] = rng.choice(special, size=k)
    im[k : 2 * k] = rng.choice(special, size=k)
    return re + 1j * im


def _pick(candidates: list[int], ok: Any) -> int | None:
    """The first candidate mode for which ``ok(mode)`` holds (``None``: none does)."""
    for mode in candidates:
        if ok(mode):
            return mode
    return None


def _probe_modes(core: Any) -> dict[str, Any] | None:
    """Evaluation orders of this platform's numpy / CPython complex arithmetic."""
    arith = core._synthesis_arith
    scalar = core._synthesis_scalar
    rng = np.random.default_rng(20240531)
    n = 4096
    a = _samples(rng, n)
    b = _samples(rng, n)
    modes: dict[str, Any] = {}

    # numpy: 数组 * 复数标量（u * cmath.exp(...)、c_mat * root、u[:, 0] * scale）
    def arr_scalar(m: int) -> bool:
        for i in range(0, 64):
            mat = a[16 * i : 16 * i + 16].reshape(4, 4)
            s = complex(b[i])
            if _bits(mat * s) != _bits(arith(0, m, mat.ravel(), np.full(16, s))):
                return False
            if _bits(mat[:2, :2].copy() * s) != _bits(
                arith(0, m, mat[:2, :2].ravel(), np.full(4, s))
            ):
                return False
            f = float(b[i].real)
            col = mat[:, 0] * f
            if _bits(col) != _bits(
                arith(0, m, mat[:, 0].copy(), np.full(4, complex(f, 0.0)))
            ):
                return False
        return True

    # numpy: 复数标量 * 数组（phase * v、1j * sin * pauli、cos * _I2、1j * theta）
    def scalar_arr(m: int) -> bool:
        for i in range(0, 64):
            mat = a[16 * i : 16 * i + 16].reshape(4, 4)
            s = complex(b[i])
            if _bits(s * mat) != _bits(arith(0, m, np.full(16, s), mat.ravel())):
                return False
            f = float(b[i].real)
            if _bits(f * mat[:2, :2]) != _bits(
                arith(0, m, np.full(4, complex(f)), mat[:2, :2].ravel())
            ):
                return False
            theta = mat.real[0].copy()
            if _bits(1j * theta) != _bits(
                arith(0, m, np.full(4, 1j), theta.astype(complex))
            ):
                return False
        return True

    # numpy: _kron2 的广播乘法
    def kron(m: int) -> bool:
        for i in range(0, 128):
            left = a[8 * i : 8 * i + 4].reshape(2, 2)
            right = b[8 * i + 4 : 8 * i + 8].reshape(2, 2)
            ref = left[:, None, :, None] * right[None, :, None, :]
            lb = np.broadcast_to(left[:, None, :, None], (2, 2, 2, 2)).ravel()
            rb = np.broadcast_to(right[None, :, None, :], (2, 2, 2, 2)).ravel()
            if _bits(ref) != _bits(arith(0, m, lb.copy(), rb.copy())):
                return False
        return True

    # numpy 标量乘法（_zxz_angles 的 m00 * m11）
    def np_scalar(m: int) -> bool:
        ref = np.array(
            [
                np.complex128(x) * np.complex128(y)
                for x, y in zip(a[:1024], b[:1024], strict=True)
            ]
        )
        return _bits(ref) == _bits(arith(0, m, a[:1024], b[:1024]))

    # CPython complex * complex
    def py(m: int) -> bool:
        ref = np.array(
            [complex(x) * complex(y) for x, y in zip(a[:1024], b[:1024], strict=True)]
        )
        return _bits(ref) == _bits(arith(0, m, a[:1024], b[:1024]))

    # numpy: 数组 / 复数标量
    def np_div(m: int) -> bool:
        for i in range(0, 64):
            mat = a[16 * i : 16 * i + 16].reshape(4, 4)
            s = complex(b[i])
            if s == 0:
                continue
            if _bits(mat / s) != _bits(arith(1, m, mat.ravel(), np.full(16, s))):
                return False
        return True

    for key, fn, cands in (
        ("arr_scalar", arr_scalar, [1, 0, 2]),
        ("scalar_arr", scalar_arr, [1, 0, 2]),
        ("kron", kron, [1, 0, 2]),
        ("np_scalar", np_scalar, [1, 0, 2]),
        ("py", py, [1, 0, 2]),
        ("np_div", np_div, [1, 0]),
    ):
        with np.errstate(all="ignore"):
            mode = _pick(cands, fn)
        if mode is None:
            logger.debug("native KAK: no evaluation order matches %s", key)
            return None
        modes[key] = mode

    # CPython complex * float：3.13 及以前是完整复数乘法，之后是混合运算
    fs = b.real[:1024]
    ref = np.array([complex(x) * float(f) for x, f in zip(a[:1024], fs, strict=True)])
    full = arith(2, modes["py"], a[:1024], fs.astype(complex))
    mixed = arith(2, 3, a[:1024], fs.astype(complex))
    if _bits(ref) == _bits(full):
        modes["py_mixed"] = False
    elif _bits(ref) == _bits(mixed):
        modes["py_mixed"] = True
    else:
        logger.debug("native KAK: CPython complex * float order not reproduced")
        return None

    # CPython / libm 标量函数与 numpy 的 arctan2、exp(1j·θ)
    import cmath

    zs = a[:2048]
    small = zs * 0.25
    py_exp = np.array([cmath.exp(complex(z)) for z in small])
    if _bits(py_exp) == _bits(scalar(1, small)):
        modes["py_exp_joint"] = False
    elif _bits(py_exp) == _bits(scalar(7, small)):
        modes["py_exp_joint"] = True
    else:
        logger.debug("native KAK: cmath.exp not reproduced")
        return None
    checks = [
        (0, zs, [cmath.sqrt(complex(z)) for z in zs]),
        (2, zs, [complex(cmath.phase(complex(z))) for z in zs]),
        (3, zs, [complex(math.atan2(z.imag, z.real)) for z in zs]),
        (6, zs, [complex(abs(complex(z))) for z in zs]),
    ]
    for fn_code, arg, expected in checks:
        if _bits(np.array(expected)) != _bits(scalar(fn_code, arg)):
            logger.debug("native KAK: scalar function %d not reproduced", fn_code)
            return None
    with np.errstate(all="ignore"):
        angle = np.angle(zs)
        theta = zs.real * 3.0
        expo = np.exp(1j * theta)
    if _bits(angle) != _bits(scalar(4, zs).real):
        logger.debug("native KAK: np.angle differs from libm atan2")
        return None
    if _bits(expo) == _bits(scalar(5, 1j * theta)):
        modes["np_exp_joint"] = False
    elif _bits(expo) == _bits(scalar(8, 1j * theta)):
        modes["np_exp_joint"] = True
    else:
        logger.debug("native KAK: np.exp(1j * theta) differs from (cos, sin)")
        return None
    modes["linalg"] = 1  # 由 _probe_linalg 确定
    return modes


def _probe_linalg(
    core: Any,
    functions: dict[str, int],
    ilp64: bool,
    modes: dict[str, Any],
    constants: dict[str, Any],
) -> Any | None:
    """Build the synthesizer with the matching ``np.linalg.det`` evaluation order."""
    rng = np.random.default_rng(777)
    mats = [
        rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        for n in (2, 4)
        for _ in range(200)
    ]
    mats += [
        np.eye(2, dtype=complex),
        np.eye(4, dtype=complex)[[1, 0, 2, 3]],
        np.diag([1j, -1, 1, -1j]),
    ]
    for mode in (1, 0, 2):
        trial = dict(modes, linalg=mode)
        synth = core._KakSynthesizer(functions, ilp64, trial, constants)
        if all(
            _bits(complex(np.linalg.det(m))) == _bits(complex(synth.det(m)))
            for m in mats
        ):
            modes["linalg"] = mode
            return synth
    logger.debug("native KAK: np.linalg.det not reproduced")
    return None


# ---------------------------------------------------------------------------
# Synthesizer wrapper
# ---------------------------------------------------------------------------


_EULER = {"zxz": 0, "zsx": 1}


class NativeKak:
    """Dispatch of the KAK synthesis entry points to the native backend.

    Every method returns ``None`` when the native backend cannot reproduce
    the Python reference for that input (the caller then runs the Python
    code, which also raises the reference exceptions).
    """

    def __init__(self, synth: Any, tq: Any) -> None:
        self._synth = synth
        self._tq = tq
        self._entanglers: dict[str, tuple[object, int]] = {}

    def _entangler(self, basis: TwoQubitBasis) -> int | None:
        if basis.euler not in _EULER:
            return None
        ent = self._tq._ENTANGLERS.get(basis.entangler)
        if ent is None:
            return None
        hit = self._entanglers.get(basis.entangler)
        if hit is not None and hit[0] is ent:
            return hit[1]
        arrays = [ent.matrix, *ent.before, *ent.after]
        if any(np.asarray(x).dtype != _C128 for x in arrays):
            return None
        index = int(
            self._synth.add_entangler(
                ent.name, ent.matrix, *ent.before, *ent.after, bool(ent.symmetric)
            )
        )
        self._entanglers[basis.entangler] = (ent, index)
        return index

    def _usable(self, u: object) -> bool:
        tq = self._tq
        if tq._try_synthesize is not tq._try_synthesize_reference:
            return False  # 测试替换了 Python 实现
        return (
            isinstance(u, np.ndarray)
            and u.dtype == _C128
            and u.shape == (4, 4)
            and u.flags.c_contiguous
        )

    def synthesize(
        self, u: object, atol: float, verify_atol: float, basis: TwoQubitBasis
    ) -> tuple[list[SynthOp], bool] | None:
        """``_synthesize_two_qubit_ops`` and whether it reads the SWAP-face flag."""
        if not self._usable(u):
            return None
        ent = self._entangler(basis)
        if ent is None:
            return None
        tq = self._tq
        result = self._synth.synthesize(
            u,
            ent,
            _EULER[basis.euler],
            atol,
            verify_atol,
            bool(tq._SWAP_FACE_TEMPLATE),
            bool(tq._MIRRORED_STABILIZERS),
        )
        if result is None:
            return None
        return result[0], bool(result[1])

    def approximate(
        self,
        u: object,
        max_infidelity: float,
        atol: float,
        verify_atol: float,
        basis: TwoQubitBasis,
        max_entanglers: int,
        want_exact: bool,
    ) -> tuple[list[SynthOp] | None, bool] | None:
        """``None`` unless no approximation applies; then the exact ops (if wanted)."""
        if not self._usable(u):
            return None
        ent = self._entangler(basis)
        if ent is None:
            return None
        tq = self._tq
        status, ops, uses = self._synth.approximate(
            u,
            max_infidelity,
            atol,
            verify_atol,
            int(max_entanglers),
            ent,
            _EULER[basis.euler],
            bool(tq._SWAP_FACE_TEMPLATE),
            bool(tq._MIRRORED_STABILIZERS),
            want_exact,
        )
        if status != 0:
            return None
        return ops, bool(uses)

    def swap_face(
        self, u: object, basis: TwoQubitBasis, verify_atol: float
    ) -> list[SynthOp] | bool | None:
        """``_synthesize_swap_face_block``: ops, ``False`` (``None``) or ``None``."""
        if not self._usable(u):
            return None
        ent = self._entangler(basis)
        if ent is None:
            return None
        result = self._synth.swap_face(
            u,
            ent,
            _EULER[basis.euler],
            verify_atol,
            bool(self._tq._MIRRORED_STABILIZERS),
        )
        return result  # type: ignore[no-any-return]

    def num_cz(self, u: object, atol: float) -> int | None:
        """``_count_from_coords(_kak4(u)[2], atol)``."""
        if not self._usable(u):
            return None
        count = int(self._synth.num_cz(u, atol))
        return None if count < 0 else count


def _constants(tq: Any) -> dict[str, Any]:
    rng = np.random.default_rng(1234567)
    normals = [float(rng.normal()) for _ in range(196)]
    return {
        "magic": tq._MAGIC,
        "magic_conj": np.ascontiguousarray(tq._MAGIC_DAG.T),
        "i2": tq._I2,
        "x": tq._X,
        "y": tq._Y,
        "z": tq._Z,
        "h": tq._H,
        "sx": tq._SX,
        "cz4": tq._CZ4,
        "swap4": tq._SWAP4,
        "pauli_pairs": list(tq._PAULI_PAIRS),
        "swap_ab": tq._SWAP_AB,
        "swap_bc": tq._SWAP_BC,
        "flip_ab": tq._FLIP_AB,
        "flip_ac": tq._FLIP_AC,
        "flip_bc": tq._FLIP_BC,
        "phase_map": np.ascontiguousarray(tq._PHASE_MAP, dtype=float),
        "can_phases": np.ascontiguousarray(tq._CAN_PHASES, dtype=float),
        "normals": normals,
    }


def _self_check(native: NativeKak, tq: Any) -> bool:
    """Compare the native synthesis with the Python reference on a fixed sample."""
    from quspect.synthesis.two_qubit import TwoQubitBasis

    rng = np.random.default_rng(97531)
    swap = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
    zz = np.diag(np.exp(-0.15j * np.array([1, -1, -1, 1])))
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    cp = np.diag([1, 1, 1, np.exp(0.4j)])
    heis = tq.canonical_gate(0.3, 0.3, -0.3)

    def local() -> np.ndarray:
        a = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        b = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        return np.asarray(np.kron(a, b))

    mats = [
        np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        for _ in range(3)
    ]
    mats += [local() @ m @ local() for m in (swap @ zz, cz, cp, heis, swap)]
    mats += [np.eye(4, dtype=complex), local()]
    mats = [np.ascontiguousarray(m, dtype=complex) for m in mats]
    flags = (tq._SWAP_FACE_TEMPLATE, tq._MIRRORED_STABILIZERS)
    try:
        tq._SWAP_FACE_TEMPLATE, tq._MIRRORED_STABILIZERS = False, True
        for m in mats[:4]:
            ref = tq._kak4(m)
            got = native._synth.kak4(m)
            if got is None or [_bits(x) for x in (ref[0], ref[1], ref[2], ref[3])] != [
                _bits(got[0]),
                _bits(got[1]),
                _bits(np.array(got[2])),
                _bits(got[3]),
            ]:
                return False
        for basis, flag in (
            (TwoQubitBasis("cz", "zxz"), False),
            (TwoQubitBasis("cz", "zsx"), True),
            (TwoQubitBasis("cx", "zsx"), False),
        ):
            tq._SWAP_FACE_TEMPLATE = flag
            for m in mats if basis.entangler == "cz" else mats[:5]:
                ref_ops = tq._synthesize_two_qubit_ops(m, 1e-9, 1e-9, basis)
                got = native.synthesize(m, 1e-9, 1e-9, basis)
                if got is None or repr(got[0]) != repr(ref_ops):
                    return False
        for m in mats[3:5]:
            basis = TwoQubitBasis("cz", "zsx")
            ref_sf = tq._synthesize_swap_face_block(m, basis, 1e-9)
            got_sf = native.swap_face(m, basis, 1e-9)
            if got_sf is None or repr(None if got_sf is False else got_sf) != repr(
                ref_sf
            ):
                return False
    except Exception:  # noqa: BLE001 - 任何异常都视为不一致
        logger.debug("native KAK self-check raised", exc_info=True)
        return False
    finally:
        tq._SWAP_FACE_TEMPLATE, tq._MIRRORED_STABILIZERS = flags
    return True


def load(tq: Any) -> NativeKak | None:
    """Build the native backend for the module ``tq`` (two_qubit), or ``None``."""
    backend = os.environ.get(BACKEND_ENV, "auto") or "auto"
    if backend not in ("auto", "native", "python"):
        raise ValueError(
            f"{BACKEND_ENV} must be auto, native or python; got {backend!r}"
        )
    if backend == "python":
        return None
    from quspect._native_loader import load_core

    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        return None
    if core is None or not hasattr(core, "_KakSynthesizer"):
        return None
    found = _linalg_functions()
    if found is None:
        logger.debug("native KAK: numpy's BLAS / LAPACK not resolved")
        return None
    functions, ilp64, blas = found
    key = (platform.system(), platform.machine(), blas)
    if backend == "auto" and key not in _VERIFIED_PLATFORMS:
        logger.debug("native KAK: platform %s not verified", key)
        return None
    modes = _probe_modes(core)
    if modes is None:
        return None
    try:
        synth = _probe_linalg(core, functions, ilp64, modes, _constants(tq))
    except (
        RuntimeError,
        ValueError,
    ):  # pragma: no cover - LAPACK workspace query failed
        return None
    if synth is None:
        return None
    native = NativeKak(synth, tq)
    if not _self_check(native, tq):
        logger.debug("native KAK: self-check against the Python reference failed")
        return None
    return native
