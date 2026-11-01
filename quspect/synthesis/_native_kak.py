"""Native backend of the two-qubit KAK synthesis.

The C++ synthesizer (``_KakSynthesizer``) runs the two-qubit KAK synthesis
(:func:`~quspect.synthesis.two_qubit.synthesize_two_qubit_ops`, the SWAP-face
block, the CZ count and the "no approximation" decision) on the deterministic
numerics layer: fixed-order small-matrix arithmetic, Jacobi decompositions
and correctly rounded math functions (CORE-MATH), without BLAS / LAPACK or
the platform libm.  Its results are bit-identical on every platform and are
the reference output of the compiler; the Python implementation in
:mod:`~quspect.synthesis.two_qubit` is only used when the extension module is
missing (same algorithm, numerically equivalent but not bit-identical).

The same object also hosts helpers that still follow numpy's semantics bit
for bit (single-qubit synthesis, the verified Euler runs and block products
of block consolidation, and the numpy linear algebra used by the native gauge
search and basis translation).  They reproduce *this platform's* numpy by

* calling the very BLAS / LAPACK routines numpy links against (resolved here
  from numpy's build configuration, after checking that numpy's own binaries
  import exactly those symbols), with numpy's arguments and data layouts;
* evaluating every element-wise operation in the order this platform's numpy
  and CPython use; the order is probed here on random inputs and must match
  exactly one candidate.

That part is enabled only when all of that succeeds, a self-check against
the Python reference passes and the platform is one where it has been
compared against the Python reference on benchmark corpora
(:data:`_VERIFIED_PLATFORMS`); set :data:`BACKEND_ENV` to ``native`` to
enable it on other platforms (still subject to the probes and the
self-check) or to ``python`` to disable the native backend altogether.
"""

from __future__ import annotations

import ctypes
import glob
import importlib
import logging
import math
import os
import platform
import random
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
        with open(path, "rb") as fh:
            data = fh.read(1 << 24)
    except OSError:
        return False
    return all(data.find(name) >= 0 for name in names)


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
    _, linalg_bin = binaries
    for scheme in schemes:
        # numpy.linalg 必须导入这些符号（同一个库、同一种命名与整数宽度；CBLAS
        # 名称与之同属 numpy 构建时选定的一套前后缀）。只查这个较小的扩展模块。
        names = [
            scheme.lapack.format(n).encode()
            for n in ("zgesdd", "dsyevd", "zgetrf", "zcopy")
        ]
        if not _binary_mentions(linalg_bin, names):
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


def _samples(rng: random.Random, n: int) -> np.ndarray:
    """Random complex samples with zeros, signed zeros and exact values mixed in."""
    special = (0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-300, -3.0, math.pi, 1e-17)
    k = n // 4
    re = [rng.choice(special) if i < k else rng.gauss(0.0, 1.0) for i in range(n)]
    im = [
        rng.choice(special) if k <= i < 2 * k else rng.gauss(0.0, 1.0) for i in range(n)
    ]
    return np.array(re, dtype=float) + 1j * np.array(im, dtype=float)


def _pick(candidates: list[int], ok: Any) -> int | None:
    """The first candidate mode for which ``ok(mode)`` holds (``None``: none does)."""
    for mode in candidates:
        if ok(mode):
            return mode
    return None


def _probe_modes(core: Any) -> dict[str, Any] | None:
    """Evaluation orders of this platform's numpy / CPython complex arithmetic.

    Each probe evaluates the expression shapes the synthesis uses (a C
    contiguous array times a scalar, a scalar times an array, the broadcast
    product of ``_kron2``, numpy and CPython scalars, ...) on random inputs
    with zeros and signed zeros mixed in, and keeps the first candidate order
    that reproduces every result bit for bit.
    """
    import cmath

    arith = core._synthesis_arith
    scalar = core._synthesis_scalar
    rng = random.Random(20240531)
    a = _samples(rng, 1024)
    b = _samples(rng, 1024)
    mats = a.reshape(64, 4, 4)
    svals = b[:64].reshape(64, 1, 1)
    fvals = b.real[64:128].reshape(64, 1, 1)
    modes: dict[str, Any] = {}

    def same(ref: Any, op: int, mode: int, x: Any, y: Any) -> bool:
        xs = np.broadcast_to(x, np.shape(ref)).ravel()
        ys = np.broadcast_to(y, np.shape(ref)).ravel()
        return _bits(ref) == _bits(arith(op, mode, xs.copy(), ys.copy()))

    with np.errstate(all="ignore"):
        # numpy：数组 * 复数标量（u * cmath.exp(...)、c_mat * root、u[:, 0] * scale）
        r1 = mats * svals
        r2 = mats[:, :2, :2] * svals
        r3 = mats[:, :, 0] * fvals[:, :, 0]
        f_c = fvals.astype(complex)
        # numpy：复数标量 * 数组（phase * v、1j * sin * pauli、cos * _I2、1j * theta）
        r4 = svals * mats
        r5 = fvals * mats[:, :2, :2]
        theta = mats.real[:, 0, :].copy()
        r6 = 1j * theta
        # _kron2 的广播乘法
        left = a[:256].reshape(64, 2, 2)
        right = b[256:512].reshape(64, 2, 2)
        r7 = left[:, :, None, :, None] * right[:, None, :, None, :]
        # numpy：数组 / 复数标量
        divisors = np.where(svals == 0, 1.0 + 0j, svals)
        r8 = mats / divisors
    checks: list[tuple[str, list[int], Any]] = [
        (
            "arr_scalar",
            [1, 0, 2],
            lambda m: (
                same(r1, 0, m, mats, svals)
                and same(r2, 0, m, mats[:, :2, :2], svals)
                and same(r3, 0, m, mats[:, :, 0], f_c[:, :, 0])
            ),
        ),
        (
            "scalar_arr",
            [1, 0, 2],
            lambda m: (
                same(r4, 0, m, svals, mats)
                and same(r5, 0, m, f_c, mats[:, :2, :2])
                and same(r6, 0, m, 1j, theta.astype(complex))
            ),
        ),
        (
            "kron",
            [1, 0, 2],
            lambda m: same(
                r7, 0, m, left[:, :, None, :, None], right[:, None, :, None, :]
            ),
        ),
        ("np_div", [1, 0], lambda m: same(r8, 1, m, mats, divisors)),
    ]
    xs, ys = a[:256], b[:256]
    np_prod = np.array(
        [np.complex128(x) * np.complex128(y) for x, y in zip(xs, ys, strict=True)]
    )
    py_prod = np.array([complex(x) * complex(y) for x, y in zip(xs, ys, strict=True)])
    checks.append(
        ("np_scalar", [1, 0, 2], lambda m: _bits(np_prod) == _bits(arith(0, m, xs, ys)))
    )
    checks.append(
        ("py", [1, 0, 2], lambda m: _bits(py_prod) == _bits(arith(0, m, xs, ys)))
    )
    for key, cands, ok in checks:
        mode = _pick(cands, ok)
        if mode is None:
            logger.debug("native KAK: no evaluation order matches %s", key)
            return None
        modes[key] = mode

    # CPython complex * float：3.13 及以前是完整复数乘法，之后是混合运算
    fs = b.real[:256]
    ref = np.array([complex(x) * float(f) for x, f in zip(xs, fs, strict=True)])
    if _bits(ref) == _bits(arith(2, modes["py"], xs, fs.astype(complex))):
        modes["py_mixed"] = False
    elif _bits(ref) == _bits(arith(2, 3, xs, fs.astype(complex))):
        modes["py_mixed"] = True
    else:
        logger.debug("native KAK: CPython complex * float order not reproduced")
        return None

    # cmath / math / abs 与 numpy 的 np.angle、np.exp(1j·θ)
    zs = a[:256]
    small = zs * 0.25
    py_exp = np.array([cmath.exp(complex(z)) for z in small])
    if _bits(py_exp) == _bits(scalar(1, small)):
        modes["py_exp_joint"] = False
    elif _bits(py_exp) == _bits(scalar(7, small)):
        modes["py_exp_joint"] = True
    else:
        logger.debug("native KAK: cmath.exp not reproduced")
        return None
    for fn_code, expected in (
        (0, [cmath.sqrt(complex(z)) for z in zs]),
        (2, [complex(cmath.phase(complex(z))) for z in zs]),
        (3, [complex(math.atan2(z.imag, z.real)) for z in zs]),
        (6, [complex(abs(complex(z))) for z in zs]),
        # numpy 标量的 abs（npy_cabs，单比特合成的角度用到）
        (6, [complex(float(abs(np.complex128(z)))) for z in zs]),
    ):
        if _bits(np.array(expected)) != _bits(scalar(fn_code, zs)):
            logger.debug("native KAK: scalar function %d not reproduced", fn_code)
            return None
    with np.errstate(all="ignore"):
        angle = np.angle(a)
        phases = a.real * 3.0
        expo = np.exp(1j * phases)
    if _bits(angle) != _bits(scalar(4, a).real):
        logger.debug("native KAK: np.angle differs from libm atan2")
        return None
    if _bits(expo) == _bits(scalar(5, 1j * phases)):
        modes["np_exp_joint"] = False
    elif _bits(expo) == _bits(scalar(8, 1j * phases)):
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
    rng = random.Random(777)
    mats = [_samples(rng, n * n).reshape(n, n) for n in (2, 4) for _ in range(64)]
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

    The two-qubit entry points always run natively (deterministic numerics)
    and raise the reference exceptions themselves (``ValueError`` for a
    non-unitary input, ``RuntimeError`` if synthesis fails); they return
    ``None`` only for inputs that are not ``4×4`` matrices (the Python code
    then raises).  The numpy-semantics helpers (:attr:`numpy_compat`) return
    ``None`` when they cannot reproduce the Python reference for an input.
    """

    def __init__(self, synth: Any, tq: Any) -> None:
        self._synth = synth
        self._tq = tq
        #: Whether the numpy-semantics helpers are available on this platform.
        self.numpy_compat = bool(synth.has_numpy)
        self._entanglers: dict[str, tuple[object, int]] = {}
        self._one_qubit: dict[frozenset[str], tuple[object, int, tuple[Any, ...]]] = {}

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

    def _matrix(self, u: object) -> np.ndarray | None:
        """``u`` as a ``4×4`` ``complex128`` array (``None``: not a 4×4 matrix)."""
        tq = self._tq
        if tq._try_synthesize is not tq._try_synthesize_reference:
            return None  # 测试替换了 Python 实现
        try:
            arr = np.asarray(u, dtype=complex)
        except (TypeError, ValueError):
            return None
        if arr.shape != (4, 4):
            return None
        return arr

    def synthesize(
        self, u: object, atol: float, verify_atol: float, basis: TwoQubitBasis
    ) -> tuple[list[SynthOp], bool] | None:
        """``_synthesize_two_qubit_ops`` and whether it reads the SWAP-face flag."""
        arr = self._matrix(u)
        ent = None if arr is None else self._entangler(basis)
        if arr is None or ent is None:
            return None
        tq = self._tq
        ops, uses = self._synth.synthesize(
            arr,
            ent,
            _EULER[basis.euler],
            atol,
            verify_atol,
            bool(tq._SWAP_FACE_TEMPLATE),
            bool(tq._MIRRORED_STABILIZERS),
        )
        return ops, bool(uses)

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
        arr = self._matrix(u)
        ent = None if arr is None else self._entangler(basis)
        if arr is None or ent is None:
            return None
        tq = self._tq
        status, ops, uses = self._synth.approximate(
            arr,
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
        arr = self._matrix(u)
        ent = None if arr is None else self._entangler(basis)
        if arr is None or ent is None:
            return None
        result = self._synth.swap_face(
            arr,
            ent,
            _EULER[basis.euler],
            verify_atol,
            bool(self._tq._MIRRORED_STABILIZERS),
        )
        return result  # type: ignore[no-any-return]

    def num_cz(self, u: object, atol: float) -> int | None:
        """``_count_from_coords(_kak4(u)[2], atol)``."""
        arr = self._matrix(u)
        if arr is None:
            return None
        return int(self._synth.num_cz(arr, atol))

    def kak4(
        self, u: object
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
        """``_kak4(u)`` (``(phase, k1, coords, k2)``) on the deterministic layer."""
        arr = self._matrix(u)
        if arr is None:
            return None
        self._synth.num_cz(arr, 0.0)  # 酉性检查（不是酉矩阵时抛出 ValueError）
        phase, k1, v, k2 = self._synth.kak4(arr)
        return float(phase), k1, np.array(v, dtype=float), k2

    def one_qubit(self, m: np.ndarray, em: Any) -> tuple[bool, list[Any] | None]:
        """``_synthesize_one_qubit(m, em.basis)`` as ``(True, gates)``.

        ``em`` is the :class:`~quspect.gates.synthesis._Emitter` of the basis;
        ``(False, None)`` hands the matrix back to the Python reference.
        """
        if not self.numpy_compat:
            return False, None
        entry = self._one_qubit.get(em.basis)
        if entry is None or entry[0] is not em:
            entry = self._register_one_qubit(em)
            self._one_qubit[em.basis] = entry
        _, index, makers = entry
        if index < 0 or m.dtype != _C128 or m.shape != (2, 2):
            return False, None
        status, gates = self._synth.synthesize_one_qubit(np.ascontiguousarray(m), index)
        if status == 1:
            return True, None
        if status != 0:
            return False, None
        out = []
        for kind, idx, p0, p1, p2 in gates:
            if kind == 0:
                out.append(makers[0][idx])
            elif kind == 4:
                out.append(makers[4](p0, p1, p2))
            else:
                out.append(makers[kind](p0))
        return True, out

    def _register_one_qubit(self, em: Any) -> tuple[object, int, tuple[Any, ...]]:
        from quspect.gates import synthesis as gs
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.standard import SDG, SX, SXDG, TDG, H, S, T, X, Y, Z

        fixed = [H, X, Y, Z, S, SDG, T, TDG, SX, SXDG]
        index_of = {id(g): i for i, g in enumerate(fixed)}
        try:
            words = {k: [index_of[id(g)] for g in w] for k, w in em._clifford_t.items()}
            named = [index_of[id(g)] for g, _, _ in em.named]
            special = [g for g in (em.h, em.x, em.sx, em.sxdg) if g is not None]
            if any(id(g) not in index_of for g in special):
                raise KeyError
        except KeyError:
            return em, -1, ()
        makers = (fixed, em._rz_maker, em.rx, em.ry, em.u3)
        mats = [np.asarray(gate_matrix(g), dtype=complex) for g in fixed]
        index = int(
            self._synth.add_one_qubit_basis(
                mats,
                em._rz_maker is not None,
                em.h is not None,
                em.x is not None,
                em.sx is not None,
                em.sxdg is not None,
                em.rx is not None,
                em.ry is not None,
                em.u3 is not None,
                words,
                named,
            )
        )
        entry = (em, index, makers)
        if not _check_one_qubit(self, em, entry, gs):
            logger.debug("native one-qubit synthesis disabled for %s", sorted(em.basis))
            return em, -1, ()
        return entry

    def verified_euler(
        self, u: np.ndarray, atol: float, euler: str
    ) -> list[SynthOp] | bool | None:
        """``_verified_euler_ops_uncached``: ops, ``False`` (``None``) or ``None``."""
        if (
            not self.numpy_compat
            or euler not in _EULER
            or u.dtype != _C128
            or u.shape != (2, 2)
        ):
            return None
        result = self._synth.verified_euler(
            np.ascontiguousarray(u), atol, _EULER[euler]
        )
        return result  # type: ignore[no-any-return]

    def block_unitary(
        self, mats: list[np.ndarray], slots: list[int], swap4: np.ndarray
    ) -> np.ndarray | None:
        """``u = I; u = full @ u`` over a block's members (see the C++ docs).

        ``None`` unless every matrix is C-contiguous ``complex128`` (numpy's
        BLAS calls depend on the operand layout).
        """
        if not self.numpy_compat:
            return None
        strict = getattr(self._synth, "block_unitary_strict", None)
        if strict is not None and type(mats) is list:
            # 布局判断在原生代码中完成；不满足时照常走下面的判断
            found = strict(mats, slots, swap4)
            if found is not None:
                return np.asarray(found)
        if not all(m.dtype == _C128 and m.flags.c_contiguous for m in mats):
            return None
        return np.asarray(self._synth.block_unitary(mats, slots, swap4))

    def run_unitary(self, mats: list[np.ndarray]) -> np.ndarray | None:
        """``u = I; u = m @ u`` over a run of C-contiguous ``2×2`` matrices."""
        if not self.numpy_compat:
            return None
        strict = getattr(self._synth, "run_unitary_strict", None)
        if strict is not None and type(mats) is list:
            found = strict(mats)
            if found is not None:
                return np.asarray(found)
        if not all(
            m.dtype == _C128 and m.shape == (2, 2) and m.flags.c_contiguous
            for m in mats
        ):
            return None
        return np.asarray(self._synth.run_unitary(mats))


def _check_one_qubit(native: NativeKak, em: Any, entry: Any, gs: Any) -> bool:
    """Compare the native single-qubit synthesis of ``em``'s basis with Python."""
    rng = random.Random(4321)
    mats = [np.asarray(gs._sequence_matrix([g]), dtype=complex) for g in entry[2][0]]
    mats += [np.diag([1, np.exp(0.3j)]), np.array([[0, 1j], [np.exp(0.2j), 0]])]
    for _ in range(6):
        g = [complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(4)]
        mats.append(np.linalg.qr(np.array(g).reshape(2, 2))[0])
    mats.append(np.array([[1, 1j], [1j, 1]]) / math.sqrt(2.0) * np.exp(0.7j))
    native._one_qubit[em.basis] = entry
    try:
        for m in mats:
            m = np.ascontiguousarray(m, dtype=complex)
            ok, got = native.one_qubit(m, em)
            if ok and repr(got) != repr(gs._synthesize_one_qubit_reference(m, em)):
                return False
    except Exception:  # noqa: BLE001 - 任何异常都视为不一致
        logger.debug("native one-qubit self-check raised", exc_info=True)
        return False
    finally:
        native._one_qubit.pop(em.basis, None)
    return True


def _constants(tq: Any) -> dict[str, Any]:
    # numpy 语义部分用到的常量（KAK 合成的常量在 C++ 中定义，数值相同）
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
    }


def _self_check(native: NativeKak, tq: Any) -> bool:
    """Compare the numpy-semantics helpers with the Python reference."""
    from quspect.transpiler import _consolidate

    rng = random.Random(97531)

    def haar(n: int) -> np.ndarray:
        g = [complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(n * n)]
        return np.ascontiguousarray(np.linalg.qr(np.array(g).reshape(n, n))[0])

    try:
        twos = [haar(2) for _ in range(6)] + [np.ascontiguousarray(tq._H)]
        twos.append(np.diag([1.0, np.exp(1e-10j)]).astype(complex))
        for m in twos:
            for atol in (1e-9, 1e-12):
                for euler in ("zxz", "zsx"):
                    got = native.verified_euler(m, atol, euler)
                    ref = _consolidate._verified_euler_ops_uncached(m, atol, euler)
                    if got is not None and repr(None if got is False else got) != repr(
                        ref
                    ):
                        return False
        run = twos[:4]
        ref_run = np.eye(2, dtype=complex)
        for m in run:
            ref_run = m @ ref_run
        got_run = native.run_unitary(run)
        if got_run is None or got_run.tobytes() != ref_run.tobytes():
            return False
        fours = [haar(4), haar(4)]
        mats = [fours[0], twos[0], twos[1], fours[1]]
        slots = [0, 1, 2, 3]
        ref_block = np.eye(4, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        rev = [0, 2, 1, 3]
        for m, slot in zip(mats, slots, strict=True):
            if slot == 0:
                full = m
            elif slot == 1:
                full = tq._kron2(m, eye2)
            elif slot == 2:
                full = tq._kron2(eye2, m)
            else:
                full = m[np.ix_(rev, rev)]
            ref_block = full @ ref_block
        got_block = native.block_unitary(mats, slots, tq._SWAP4)
        if got_block is None or got_block.tobytes() != ref_block.tobytes():
            return False
    except Exception:  # noqa: BLE001 - 任何异常都视为不一致
        logger.debug("native numpy-semantics self-check raised", exc_info=True)
        return False
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
    constants = _constants(tq)
    synth = _numpy_synthesizer(core, constants, backend)
    if synth is not None:
        native = NativeKak(synth, tq)
        if _self_check(native, tq):
            return native
        logger.debug("native KAK: numpy-semantics self-check failed")
    # 两比特 KAK 合成总是可用（确定性数值层，与平台无关）
    return NativeKak(core._KakSynthesizer(None, False, None, constants), tq)


def _numpy_synthesizer(
    core: Any, constants: dict[str, Any], backend: str
) -> Any | None:
    """A synthesizer with the numpy-semantics helpers (``None``: unavailable)."""
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
        return _probe_linalg(core, functions, ilp64, modes, constants)
    except (
        RuntimeError,
        ValueError,
    ):  # pragma: no cover - LAPACK workspace query failed
        return None
