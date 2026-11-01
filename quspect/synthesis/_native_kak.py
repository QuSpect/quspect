"""Native backend of the synthesis entry points.

The C++ synthesizer (``_KakSynthesizer``) runs the two-qubit KAK synthesis
(:func:`~quspect.synthesis.two_qubit.synthesize_two_qubit_ops`, the SWAP-face
block, the CZ count and the "no approximation" decision), the single-qubit
synthesis (:func:`~quspect.gates.synthesis.synthesize_one_qubit`) and the
helpers of block consolidation (verified Euler runs, block and run products)
on the deterministic numerics layer: fixed-order small-matrix arithmetic,
Jacobi decompositions and correctly rounded math functions (CORE-MATH),
without BLAS / LAPACK or the platform libm.  Its results are bit-identical on
every platform and are the reference output of the compiler; the Python
implementations are only used when the extension module is missing (same
algorithms, numerically equivalent but not bit-identical).

Set :data:`BACKEND_ENV` to ``python`` to disable the native backend.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quspect.synthesis.two_qubit import SynthOp, TwoQubitBasis

__all__ = ["BACKEND_ENV", "NativeKak", "load"]

#: ``auto`` (default), ``native`` or ``python``.
BACKEND_ENV = "QUSPECT_SYNTHESIS_BACKEND"

_C128 = np.dtype(np.complex128)
_EULER = {"zxz": 0, "zsx": 1}


class NativeKak:
    """Dispatch of the synthesis entry points to the native backend.

    The two-qubit entry points raise the reference exceptions themselves
    (``ValueError`` for a non-unitary input, ``RuntimeError`` if synthesis
    fails); they return ``None`` only for inputs that are not ``4×4``
    matrices (the Python code then raises).  The single-qubit helpers return
    ``None`` / "not handled" for inputs with non-finite entries.
    """

    def __init__(self, synth: Any, tq: Any) -> None:
        self._synth = synth
        self._tq = tq
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
        entry = self._one_qubit.get(em.basis)
        if entry is None or entry[0] is not em:
            entry = self._register_one_qubit(em)
            self._one_qubit[em.basis] = entry
        _, index, makers = entry
        if index < 0 or m.shape != (2, 2):
            return False, None
        status, gates = self._synth.synthesize_one_qubit(m, index)
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
        index = int(
            self._synth.add_one_qubit_basis(
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
        return em, index, makers

    def verified_euler(
        self, u: np.ndarray, atol: float, euler: str
    ) -> list[SynthOp] | bool | None:
        """``_verified_euler_ops_uncached``: ops, ``False`` (``None``) or ``None``."""
        if euler not in _EULER or u.shape != (2, 2):
            return None
        result = self._synth.verified_euler(u, atol, _EULER[euler])
        return result  # type: ignore[no-any-return]

    def block_unitary(
        self, mats: list[np.ndarray], slots: list[int], swap4: np.ndarray | None = None
    ) -> np.ndarray:
        """``u = I; u = full @ u`` over a block's members (see the C++ docs)."""
        return np.asarray(self._synth.block_unitary(mats, slots))

    def swap_residual(
        self, u: np.ndarray, tol: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Numerical part of ``_mapping._swap_residual``: ``(A, B)`` or ``None``."""
        res = self._synth.swap_residual(u, tol)
        return None if res is None else (np.asarray(res[0]), np.asarray(res[1]))

    def zyz_angles(self, m: np.ndarray) -> tuple[float, float, float]:
        """``_decompose_u3(m)``: ``(theta, phi, lam)``."""
        theta, phi, lam = self._synth.zyz_angles(m)
        return float(theta), float(phi), float(lam)

    def core_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``a @ b`` for small square matrices on the deterministic layer."""
        from quspect._native_loader import load_core

        core = load_core()
        assert core is not None  # NativeKak 只在有扩展模块时存在
        return np.asarray(core._detmath_matmul(a, b))

    def run_unitary(self, mats: list[np.ndarray]) -> np.ndarray:
        """``u = I; u = m @ u`` over a run of ``2×2`` matrices."""
        return np.asarray(self._synth.run_unitary(mats))


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
    return NativeKak(core._KakSynthesizer(), tq)
