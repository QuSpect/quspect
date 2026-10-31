"""Two-qubit unitary synthesis with a minimal number of CZ gates.

Any ``U ∈ U(4)`` admits a KAK (Cartan) decomposition

.. math::

    U = e^{i\\phi}\\,(K_{1l} \\otimes K_{1r})\\,
        \\exp\\!\\big(i(a\\,XX + b\\,YY + c\\,ZZ)\\big)\\,
        (K_{2l} \\otimes K_{2r}),

where the *Weyl coordinates* ``(a, b, c)`` can be brought into the Weyl
chamber ``π/4 ≥ a ≥ b ≥ |c|`` by local Clifford corrections (Kraus & Cirac
2001; Zhang et al. 2003). The chamber point determines how many CZ gates are
needed (Vatan & Williams 2004; Shende, Markov & Bullock 2004):

* 0 CZ — ``(0, 0, 0)`` (the unitary is local);
* 1 CZ — ``(π/4, 0, 0)`` (CZ / CNOT equivalent);
* 2 CZ — ``c = 0``;
* 3 CZ — otherwise.

The decomposition is computed in the magic (Bell) basis, where local gates
become real orthogonal matrices and the canonical gate becomes diagonal. The
simultaneous real-orthogonal diagonalisation of the complex-symmetric matrix
``M = U_B^T U_B`` follows the randomized approach of Qiskit's
``TwoQubitWeylDecomposition`` (diagonalise a random real combination of
``Re M`` and ``Im M`` and verify), which is robust for degenerate spectra.
This is an independent implementation; no Qiskit code is reused.

Single-qubit parts are emitted as a ZXZ Euler sequence
``rz · h · rz · h · rz`` (with identity rotations dropped and the ``θ = π``
case shortened to ``rz · x``), so the output only contains gates from
``{cz, h, rz, x}``.

Other targets are described by a :class:`TwoQubitBasis`: the native
entangler (any registered gate locally equivalent to CZ, see
:class:`Entangler`; ``cz`` and ``cx`` are built in) and the single-qubit
Euler basis (``"zxz"`` = ``{rz, h, x}``, ``"zsx"`` = ``{rz, sx, x}`` for
IBM-style devices).  The KAK templates are always CZ templates; every CZ
is rewritten as ``CZ = (A₀⊗A₁) · E · (B₀⊗B₁)`` for the entangler ``E`` and
the local dressings are folded into the neighbouring single-qubit layers
before their Euler synthesis, so the entangler count is the CZ count and
the single-qubit layers stay at most one Euler sequence each (for CX the
dressing is a Hadamard on the target).  An asymmetric entangler is tried in
both orientations per gate and the shortest result is kept.  Entanglers
outside the CZ class (iSWAP, √iSWAP, …) need their own templates and are
rejected by :meth:`Entangler.from_matrix` for now.

Accuracy: every synthesized circuit ``V`` is verified against the input
``U`` with the phase-aligned element-wise distance ``max|U - e^{iφ}V|``,
which is linear in any dropped angle, and accepted only below
``verify_atol = 1e-9``. Reduced CZ counts proposed by the Weyl coordinates
(within ``atol = 1e-9`` of a chamber face) are thus only used when the
dropped interaction really costs less than ``1e-9``; otherwise more CZ gates
are used. Error budget: an element-wise deviation ``ε`` of a ``4×4`` block
is at most ``4ε`` in operator norm and such errors add at most linearly, so
``10⁴`` blocks keep ``1 - F ≤ (4·10⁴·ε)² ≈ 1.6·10⁻⁹`` in the worst case.

Qubit convention: for a two-qubit matrix acting on ``(q0, q1)``, ``q0`` is the
most significant bit (``|q0 q1⟩``), matching :mod:`quspect.gates`.
"""

from __future__ import annotations

import cmath
import functools
import math
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from quspect.synthesis._memo import BoundedCache, matrix_key

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate

__all__ = [
    "EULER_BASES",
    "ApproximationError",
    "approximation_error",
    "Entangler",
    "TwoQubitBasis",
    "TwoQubitKAK",
    "entangler",
    "kak_decompose",
    "canonical_gate",
    "euler_ops",
    "euler_zsx_ops",
    "euler_zxz_ops",
    "register_entangler",
    "num_cz_required",
    "ops_to_instructions",
    "synthesize_two_qubit",
    "synthesize_two_qubit_approx",
    "swap_face_templates",
    "synthesize_swap_face_block",
    "synthesize_two_qubit_ops",
    "weyl_coordinates",
]

#: A synthesized gate: ``(name, local qubit indices, parameters)``.
SynthOp = tuple[str, tuple[int, ...], tuple[float, ...]]

# 纯函数的精确输入缓存（键为矩阵的原始字节），结果与重新计算逐位相同。
_NUM_CZ_CACHE: BoundedCache[tuple[object, ...], int] = BoundedCache()
_SYNTH_2Q_CACHE: BoundedCache[tuple[object, ...], tuple[SynthOp, ...]] = BoundedCache()
_APPROX_2Q_CACHE: BoundedCache[
    tuple[object, ...], tuple[tuple[SynthOp, ...], ApproximationError | None]
] = BoundedCache()

_PI4 = math.pi / 4
_PI2 = math.pi / 2

# ---------------------------------------------------------------------------
# Constant matrices
# ---------------------------------------------------------------------------

_SQ2 = 1.0 / math.sqrt(2.0)
#: Magic basis: columns are (phased) Bell states. ``B^† (A⊗B) B`` is real
#: orthogonal for ``A, B ∈ SU(2)`` and ``B^† Can(a,b,c) B`` is diagonal.
_MAGIC = (
    np.array(
        [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        dtype=complex,
    )
    * _SQ2
)
_MAGIC_DAG = _MAGIC.conj().T

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) * _SQ2
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_RX90 = np.array([[1, -1j], [-1j, 1]], dtype=complex) * _SQ2  # Rx(π/2)
_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2

_PAULI_PAIRS = (np.kron(_X, _X), np.kron(_Y, _Y), np.kron(_Z, _Z))

#: In the magic basis ``Can(a,b,c) = diag(exp(i * _CAN_PHASES @ (a,b,c)))``.
_CAN_PHASES = np.array(
    [[1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]]
)
#: ``theta = _PHASE_MAP @ (g, a, b, c)``; its columns are orthogonal with
#: squared norm 4, so the inverse is ``_PHASE_MAP.T / 4``.
_PHASE_MAP = np.column_stack([np.ones(4), _CAN_PHASES])

_CZ4 = np.diag([1, 1, 1, -1]).astype(complex)

# Local Clifford conjugations that permute the Weyl coordinates:
# C · Can(v) · C^† = Can(σ v).
_SWAP_AB = np.kron(_S, _S)  # XX ↔ YY
_SWAP_BC = np.kron(_RX90, _RX90)  # YY ↔ ZZ
_SWAP_AC = np.kron(_H, _H)  # XX ↔ ZZ
# Local Paulis that flip the sign of two coordinates (self-inverse).
_FLIP_AB = np.kron(_Z, _I2)
_FLIP_AC = np.kron(_Y, _I2)
_FLIP_BC = np.kron(_X, _I2)

#: Default numerical tolerance on Weyl coordinates (radians).
DEFAULT_ATOL = 1e-9

#: Default acceptance tolerance of a synthesized circuit: the phase-aligned,
#: element-wise distance ``max|U - e^{iφ}V|`` (linear in any dropped angle).
#:
#: Error budget (same reasoning as :mod:`quspect.gates.synthesis`): an
#: element-wise deviation ``ε`` of a ``4×4`` block is at most ``4ε`` in
#: operator norm, and operator-norm errors add at most linearly along a
#: circuit. ``10⁴`` resynthesized blocks at ``ε = 1e-9`` thus keep
#: ``‖Δψ‖ ≤ 4·10⁻⁵`` and ``1 - F ≤ ‖Δψ‖² ≈ 1.6·10⁻⁹`` in the worst case
#: (errors of independent blocks do not align in practice; typical KAK
#: reconstruction errors are ~1e-14). A looser value such as ``1e-7`` would
#: allow ``1 - F ≈ 1.6·10⁻⁵`` and must not be used.
DEFAULT_VERIFY_ATOL = 1e-9

#: Euler-angle snapping tolerance of the exact fallback attempt (see
#: :func:`synthesize_two_qubit_ops`): only rotations that are numerically zero
#: are dropped.
_EXACT_EULER_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _kron2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Fast Kronecker product of two ``2×2`` matrices."""
    return np.asarray((left[:, None, :, None] * right[None, :, None, :]).reshape(4, 4))


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[cmath.exp(-0.5j * theta), 0], [0, cmath.exp(0.5j * theta)]], dtype=complex
    )


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def canonical_gate(a: float, b: float, c: float) -> np.ndarray:
    """Return the canonical two-qubit gate ``exp(i(a XX + b YY + c ZZ))``.

    Args:
        a: Coefficient of ``X⊗X``.
        b: Coefficient of ``Y⊗Y``.
        c: Coefficient of ``Z⊗Z``.

    Returns:
        The ``4×4`` unitary.

    Example:
        >>> import numpy as np
        >>> np.allclose(canonical_gate(0.0, 0.0, 0.0), np.eye(4))
        True
    """
    phases = _CAN_PHASES @ np.array([a, b, c], dtype=float)
    return np.asarray(_MAGIC @ np.diag(np.exp(1j * phases)) @ _MAGIC_DAG)


def _factor_local(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor a ``4×4`` local unitary ``k ≈ A ⊗ C`` into ``(A, C)``.

    Uses the rank-1 SVD of the realigned matrix, which is robust to any
    zero blocks of ``A``.
    """
    realigned = k.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = np.linalg.svd(realigned)
    scale = math.sqrt(float(s[0]))
    a_mat: np.ndarray = (u[:, 0] * scale).reshape(2, 2).astype(complex)
    c_mat: np.ndarray = (vh[0, :] * scale).reshape(2, 2).astype(complex)
    det_a = complex(np.linalg.det(a_mat))
    if abs(det_a) > 1e-300:
        root = cmath.sqrt(det_a)
        a_mat = a_mat / root
        c_mat = c_mat * root
    return a_mat, c_mat


def _distance_up_to_phase(u: np.ndarray, v: np.ndarray) -> float:
    """Return the phase-aligned element-wise distance ``max|u - e^{iφ} v|``.

    The global phase ``φ = arg tr(v^† u)`` is the Frobenius-optimal one; the
    distance itself is the element-wise maximum, which is *linear* in any
    angle error (dropping ``exp(iθ ZZ)`` costs ``≈ θ``). The trace overlap is
    only used to pick the phase, never as the distance: ``1 - |tr(v^† u)|/4``
    is quadratic in the angle and would hide small dropped rotations. The
    result is within a factor 2 of ``min_φ ||u - e^{iφ} v||_max``.
    """
    overlap = complex(np.vdot(v, u))
    phase = overlap / abs(overlap) if abs(overlap) > 1e-300 else 1.0
    return float(np.max(np.abs(u - phase * v)))


@dataclass(frozen=True)
class ApproximationError:
    """How far an approximate implementation ``V`` is from a target ``U``.

    Attributes:
        infidelity: Average gate infidelity ``1 - F̄(U, V)`` with
            ``F̄ = (1 + d·F_pro) / (1 + d)`` and the process fidelity
            ``F_pro = |tr(U^† V)|² / d²`` (``d`` = dimension).  This is the
            quantity Qiskit's ``RemoveIdentityEquivalent`` and approximate
            synthesis compare against ``1 - approximation_degree``.
        distance: Upper bound on the operator-norm distance up to global
            phase, ``min_φ ‖U - e^{iφ} V‖₂``.  Unlike the infidelity it adds
            up along a circuit: replacing operations ``U_i`` by ``V_i``
            changes the whole circuit by at most ``Σ distance_i`` (up to
            global phase), and a circuit-level distance ``δ`` implies a
            process infidelity of at most ``δ²``.

    Example:
        >>> import numpy as np
        >>> err = approximation_error(np.eye(2), np.diag([1, np.exp(1e-6j)]))
        >>> err.distance < 6e-7 and err.infidelity < 2e-13
        True
    """

    infidelity: float
    distance: float


def approximation_error(u: np.ndarray, v: np.ndarray) -> ApproximationError:
    """Return the :class:`ApproximationError` of ``v`` as an implementation of ``u``.

    The distance bound comes from the eigenphases of ``W = U^† V``: if they
    lie on an arc of length ``L ≤ π``, choosing the phase at the arc centre
    gives ``‖U - e^{iφ}V‖₂ = max_k |e^{iλ_k} - e^{iφ}| = 2 sin(L/4)``.

    Args:
        u: Target unitary (``d×d``).
        v: Implemented unitary (same shape).

    Returns:
        The average gate infidelity and the phase-optimal operator-norm
        distance bound (``2.0``, the trivial bound, if the eigenphases are
        spread over more than half the circle).

    Raises:
        ValueError: If the shapes differ or are not square.

    Example:
        >>> import numpy as np
        >>> cp = np.diag([1, 1, 1, np.exp(1e-6j)])
        >>> err = approximation_error(cp, np.eye(4))
        >>> round(err.distance / 1e-6, 6), round(err.infidelity / 1e-12, 6)
        (0.5, 0.15)
    """
    a = np.asarray(u, dtype=complex)
    b = np.asarray(v, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape != b.shape:
        raise ValueError(
            f"Expected two square matrices of equal shape, got {a.shape}, {b.shape}."
        )
    d = a.shape[0]
    phases = np.sort(np.angle(np.linalg.eigvals(a.conj().T @ b)))
    # 1 - F_pro = (d² - |Σ e^{iλ}|²) / d² = Σ_{j,k} 2 sin²((λ_j - λ_k)/2) / d²:
    # 无相消误差（直接用 1 - |tr|²/d² 时 1e-12 量级的不保真度只剩约 4 位有效数字）
    half = 0.5 * (phases[:, None] - phases[None, :])
    deficit = 2.0 * float(np.sum(np.sin(half) ** 2)) / d**2
    infidelity = d * deficit / (d + 1)
    gaps = np.diff(phases, append=phases[0] + 2 * math.pi)
    arc = 2 * math.pi - float(np.max(gaps))
    distance = 2.0 * math.sin(arc / 4) if arc <= math.pi else 2.0
    return ApproximationError(infidelity=float(infidelity), distance=float(distance))


# ---------------------------------------------------------------------------
# KAK decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoQubitKAK:
    """Result of a two-qubit KAK decomposition.

    The represented unitary is
    ``exp(i*global_phase) * K1 · Can(a, b, c) · K2`` with
    ``K1 = k1l ⊗ k1r`` and ``K2 = k2l ⊗ k2r`` (``l`` acts on the most
    significant qubit). When produced by :func:`kak_decompose` the
    coordinates lie in the Weyl chamber ``π/4 ≥ a ≥ b ≥ |c|``.

    Attributes:
        a: First Weyl coordinate (``X⊗X`` coefficient).
        b: Second Weyl coordinate (``Y⊗Y`` coefficient).
        c: Third Weyl coordinate (``Z⊗Z`` coefficient).
        global_phase: Global phase in radians.
        k1l: Left (output-side) single-qubit factor on qubit 0.
        k1r: Left (output-side) single-qubit factor on qubit 1.
        k2l: Right (input-side) single-qubit factor on qubit 0.
        k2r: Right (input-side) single-qubit factor on qubit 1.

    Example:
        >>> import numpy as np
        >>> kak = kak_decompose(np.eye(4))
        >>> np.allclose(kak.unitary(), np.eye(4))
        True
    """

    a: float
    b: float
    c: float
    global_phase: float
    k1l: np.ndarray
    k1r: np.ndarray
    k2l: np.ndarray
    k2r: np.ndarray

    def unitary(self) -> np.ndarray:
        """Rebuild the ``4×4`` unitary represented by this decomposition.

        Returns:
            The reconstructed unitary.

        Example:
            >>> import numpy as np
            >>> kak_decompose(np.eye(4)).unitary().shape
            (4, 4)
        """
        k1 = _kron2(self.k1l, self.k1r)
        k2 = _kron2(self.k2l, self.k2r)
        can = canonical_gate(self.a, self.b, self.c)
        return np.asarray(cmath.exp(1j * self.global_phase) * (k1 @ can @ k2))


def _real_orthogonal_diagonalize(m2: np.ndarray) -> np.ndarray:
    """Return ``P ∈ SO(4)`` such that ``P^T m2 P`` is diagonal.

    ``m2`` is complex symmetric and unitary, so its real and imaginary parts
    are commuting real symmetric matrices. A random real combination of them
    has (generically) the joint eigenvectors; the result is verified and the
    attempt repeated with new coefficients if it fails (degenerate spectra).
    """
    m2 = m2.copy()
    m2.real[np.abs(m2.real) < 1e-14] = 0.0
    m2.imag[np.abs(m2.imag) < 1e-14] = 0.0
    rng = np.random.default_rng(1234567)
    best_p: np.ndarray | None = None
    best_err = math.inf
    for attempt in range(100):
        if attempt == 0:
            w_re, w_im = 1.0, 0.0
        elif attempt == 1:
            w_re, w_im = 0.0, 1.0
        else:
            w_re, w_im = rng.normal(), rng.normal()
        _, p = np.linalg.eigh(w_re * m2.real + w_im * m2.imag)
        d = p.T @ m2 @ p
        err = float(np.max(np.abs(d - np.diag(np.diag(d)))))
        if err < best_err:
            best_err, best_p = err, p
        if err < 1e-13:
            break
    assert best_p is not None
    p = best_p
    if np.linalg.det(p) < 0:
        p = p.copy()
        p[:, 0] = -p[:, 0]
    return np.asarray(p)


def _raw_kak(
    u: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Unconstrained KAK: ``u = e^{iφ} K1 Can(v) K2`` with 4×4 locals.

    Returns:
        ``(phase, k1, coords, k2)`` where ``k1`` and ``k2`` are ``4×4`` local
        unitaries and ``coords`` is the length-3 array ``v``.
    """
    det = complex(np.linalg.det(u))
    phase0 = cmath.phase(det) / 4.0
    u_su = u * cmath.exp(-1j * phase0)
    up = _MAGIC_DAG @ u_su @ _MAGIC
    m2 = up.T @ up
    p = _real_orthogonal_diagonalize(m2)
    eig = np.diag(p.T @ m2 @ p)
    theta = np.angle(eig) / 2.0
    d = np.exp(1j * theta)
    o1c = up @ p @ np.diag(d.conj())
    o1 = o1c.real
    if np.linalg.det(o1) < 0:
        theta[0] += math.pi
        o1 = o1.copy()
        o1[:, 0] = -o1[:, 0]
    gabc = _PHASE_MAP.T @ theta / 4.0
    k1 = _MAGIC @ o1 @ _MAGIC_DAG
    k2 = _MAGIC @ p.T @ _MAGIC_DAG
    return phase0 + float(gabc[0]), k1, np.array(gabc[1:], dtype=float), k2


def _canonicalize(
    phase: float,
    k1: np.ndarray,
    v: np.ndarray,
    k2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Move ``Can(v)`` into the Weyl chamber, tracking local corrections.

    The represented unitary ``e^{iφ} k1 Can(v) k2`` is preserved exactly.
    """
    v = v.copy()

    # 1. Shift each coordinate into (-π/4, π/4] using
    #    Can(v) = Can(v + kπ/2 e_i) · (-i P_i P_i)^k.
    for i in range(3):
        k = -math.floor((v[i] + _PI4) / _PI2)
        reduced = v[i] + k * _PI2
        if reduced <= -_PI4 + 1e-12:  # snap the boundary onto +π/4
            k += 1
            reduced += _PI2
        v[i] = reduced
        if k % 2:
            k2 = _PAULI_PAIRS[i] @ k2
        phase += -_PI2 * k  # (-i)^k

    def conj(c_mat: np.ndarray) -> None:
        # Can(v_old) = C^† Can(v_new) C
        nonlocal k1, k2
        k1 = k1 @ c_mat.conj().T
        k2 = c_mat @ k2

    # 2. Sort by absolute value (descending) with coordinate transpositions.
    swaps = {(0, 1): _SWAP_AB, (1, 2): _SWAP_BC, (0, 2): _SWAP_AC}
    for i, j in ((0, 1), (1, 2), (0, 1)):
        if abs(v[i]) < abs(v[j]):
            conj(swaps[(i, j)])
            v[i], v[j] = v[j], v[i]

    # 3. Make a, b non-negative with pairwise sign flips.
    if v[0] < 0 and v[1] < 0:
        conj(_FLIP_AB)
        v[0], v[1] = -v[0], -v[1]
    elif v[0] < 0:
        conj(_FLIP_AC)
        v[0], v[2] = -v[0], -v[2]
    if v[1] < 0:
        conj(_FLIP_BC)
        v[1], v[2] = -v[1], -v[2]

    # 4. On the a = π/4 face, (π/4, b, c) ~ (π/4, b, -c): pick c ≥ 0 via
    #    (a, c) → (-a, -c) followed by a → a + π/2.
    if v[2] < 0 and abs(v[0] - _PI4) <= 1e-11:
        conj(_FLIP_AC)
        v[0], v[2] = -v[0] + _PI2, -v[2]
        k2 = _PAULI_PAIRS[0] @ k2
        phase += -_PI2
    return phase, k1, v, k2


def _kak4(u: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Chamber-canonical KAK with ``4×4`` local factors."""
    return _canonicalize(*_raw_kak(u))


def _check_unitary_4(u: np.ndarray) -> np.ndarray:
    arr = np.asarray(u, dtype=complex)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {arr.shape}.")
    if not np.allclose(arr.conj().T @ arr, np.eye(4), atol=1e-8):
        raise ValueError("Matrix is not unitary.")
    return arr


def kak_decompose(unitary: np.ndarray) -> TwoQubitKAK:
    """Compute the Weyl-chamber KAK decomposition of a two-qubit unitary.

    Args:
        unitary: A ``4×4`` unitary matrix (``q0`` is the most significant
            qubit).

    Returns:
        A :class:`TwoQubitKAK` with chamber coordinates
        ``π/4 ≥ a ≥ b ≥ |c|`` that reconstructs ``unitary`` exactly
        (to floating-point precision).

    Raises:
        ValueError: If ``unitary`` is not a ``4×4`` unitary matrix.

    Example:
        >>> import numpy as np
        >>> cz = np.diag([1, 1, 1, -1]).astype(complex)
        >>> kak = kak_decompose(cz)
        >>> [round(x, 6) for x in (kak.a, kak.b, kak.c)]
        [0.785398, 0.0, 0.0]
    """
    u = _check_unitary_4(unitary)
    phase, k1, v, k2 = _kak4(u)
    k1l, k1r = _factor_local(k1)
    k2l, k2r = _factor_local(k2)
    return TwoQubitKAK(
        a=float(v[0]),
        b=float(v[1]),
        c=float(v[2]),
        global_phase=phase,
        k1l=k1l,
        k1r=k1r,
        k2l=k2l,
        k2r=k2r,
    )


def weyl_coordinates(unitary: np.ndarray) -> tuple[float, float, float]:
    """Return the Weyl-chamber coordinates ``(a, b, c)`` of a 2-qubit unitary.

    Two unitaries are equal up to single-qubit gates (and global phase) iff
    their chamber coordinates coincide.

    Args:
        unitary: A ``4×4`` unitary matrix.

    Returns:
        ``(a, b, c)`` with ``π/4 ≥ a ≥ b ≥ |c|``.

    Raises:
        ValueError: If ``unitary`` is not a ``4×4`` unitary matrix.

    Example:
        >>> import numpy as np
        >>> swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
        >>> [round(x, 6) for x in weyl_coordinates(swap)]
        [0.785398, 0.785398, 0.785398]
    """
    u = _check_unitary_4(unitary)
    _, _, v, _ = _kak4(u)
    return float(v[0]), float(v[1]), float(v[2])


def _count_from_coords(v: np.ndarray, atol: float) -> int:
    a, b, c = (float(x) for x in v)
    if a <= atol:
        return 0
    if abs(a - _PI4) <= atol and b <= atol:
        return 1
    if abs(c) <= atol:
        return 2
    return 3


def num_cz_required(unitary: np.ndarray, *, atol: float = DEFAULT_ATOL) -> int:
    """Return the minimal number of CZ gates needed to implement ``unitary``.

    The count is read off the Weyl coordinates alone, treating a coordinate
    within ``atol`` of a chamber face as lying on it. It is a lower bound
    for :func:`synthesize_two_qubit`, which may use more CZ gates when the
    dropped interaction would exceed its verification tolerance.

    Args:
        unitary: A ``4×4`` unitary matrix.
        atol: Tolerance on the Weyl coordinates.

    Returns:
        0, 1, 2 or 3.

    Raises:
        ValueError: If ``unitary`` is not a ``4×4`` unitary matrix.

    Example:
        >>> import numpy as np
        >>> num_cz_required(np.eye(4))
        0
        >>> num_cz_required(np.diag([1, 1, 1, np.exp(0.3j)]))
        2
    """
    key = (matrix_key(unitary), atol)
    cached = _NUM_CZ_CACHE.get(key)
    if cached is not None:
        return cached
    u = _check_unitary_4(unitary)
    _, _, v, _ = _kak4(u)
    count = _count_from_coords(v, atol)
    _NUM_CZ_CACHE.put(key, count)
    return count


# ---------------------------------------------------------------------------
# CZ templates
# ---------------------------------------------------------------------------

#: A template is a list of ``k + 1`` local layers (``(left, right)`` 2×2
#: pairs, applied in time order) interleaved with ``k`` CZ gates.
_Layer = tuple[np.ndarray, np.ndarray]


def _template(k: int, v: np.ndarray) -> list[_Layer]:
    """Return a CZ template locally equivalent to ``Can(v)``.

    The template's Weyl-chamber point equals ``v`` (with ``c`` dropped for
    ``k = 2`` and ``(a, b, c) = (π/4, 0, 0)`` for ``k = 1``).
    """
    a, b, c = (float(x) for x in v)
    eye = _I2
    if k == 0:
        return [(eye, eye)]
    if k == 1:
        return [(eye, eye), (eye, eye)]
    if k == 2:
        # CZ · (Rx(2a) ⊗ Rx(-2b)) · CZ has chamber point (a, b, 0); the inner
        # layer costs 2 × (h · rz · h).
        return [(eye, eye), (_rx(2 * a), _rx(-2 * b)), (eye, eye)]
    # k == 3: CZ · (H·Rz(2b + π/2) ⊗ Rx(2c)) · CZ · (H ⊗ Rx(2a)) · CZ has
    # chamber point (a, b, c). This CZ-native variant of the Vatan–Williams
    # circuit was chosen (by exhaustive search over cheap {h, rz} layer
    # shapes) so that the two inner layers need only 9 single-qubit gates.
    return [
        (eye, eye),
        (_H, _rx(2 * a)),
        (_H @ _rz(2 * b + _PI2), _rx(2 * c)),
        (eye, eye),
    ]


def _swap_template(c: float) -> list[_Layer]:
    """3-CZ template for the SWAP face (chamber point ``(π/4, π/4, |c|)``).

    ``SWAP·exp(-iθ/2·ZZ) = CX(0,1)·CX(1,0)·Rz_1(θ)·CX(0,1)``: with ``θ =
    π/2 − 2|c|`` and ``CX = H·CZ·H`` on the target, the inner layers are one
    ``rz`` and Hadamards only (5 gates in ``{cz, h, rz}`` instead of 9, and a
    single ``rz`` once the Hadamards merge into CX gates) — routed circuits
    are full of such blocks (a SWAP fused with a ZZ / CP / CX interaction).
    """
    theta = _PI2 - 2.0 * abs(c)
    return [
        (_I2, _H),
        (_H, _rz(theta) @ _H),
        (_H, _H),
        (_I2, _H),
    ]


#: Synthesize SWAP-face blocks with :func:`_swap_template` in
#: :func:`synthesize_two_qubit_ops` (part of the cache keys).  Off by
#: default: the generic template sometimes lets the final CX cancellation
#: find more pairs, so the transpiler runs its post-routing pipeline both
#: ways (see :func:`swap_face_templates`).
_SWAP_FACE_TEMPLATE = False


@contextmanager
def swap_face_templates(enabled: bool = True) -> Iterator[None]:
    """Synthesize SWAP-face blocks with the SWAP-shaped template in a scope.

    Inside the context :func:`synthesize_two_qubit_ops` (and the exact
    fallback of :func:`synthesize_two_qubit_approx`) writes blocks on the
    SWAP face ``a = b = π/4`` as ``SWAP·exp(-iθ/2·ZZ) = CX·CX'·Rz(θ)·CX``
    dressed with outer layers (see :func:`synthesize_swap_face_block`).
    The setting is module-wide (not thread-local).

    Args:
        enabled: The setting inside the context.

    Example:
        >>> import numpy as np
        >>> swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
        >>> zz = np.diag(np.exp(-0.15j * np.array([1, -1, -1, 1])))
        >>> with swap_face_templates():
        ...     ops = synthesize_two_qubit_ops(swap @ zz, basis=TwoQubitBasis("cx"))
        >>> [name for name, _, _ in ops]
        ['cx', 'rz', 'cx', 'cx']
    """
    global _SWAP_FACE_TEMPLATE
    old = _SWAP_FACE_TEMPLATE
    _SWAP_FACE_TEMPLATE = enabled
    try:
        yield
    finally:
        _SWAP_FACE_TEMPLATE = old


def synthesize_swap_face_block(
    unitary: np.ndarray,
    basis: TwoQubitBasis,
    *,
    verify_atol: float = DEFAULT_VERIFY_ATOL,
) -> list[SynthOp] | None:
    """Synthesize a SWAP-face block with the SWAP-shaped 3-entangler template.

    Blocks whose Weyl point is ``(π/4, π/4, c)`` — a SWAP fused with a ZZ /
    CP / CX interaction, as routing produces them — are written as
    ``SWAP·exp(-iθ/2·ZZ) = CX·CX'·Rz(θ)·CX`` dressed with outer
    single-qubit layers: the inner layers hold one ``rz`` (plus Hadamards
    for a CZ entangler) instead of two general rotations.

    Args:
        unitary: A ``4×4`` unitary (``q0`` most significant).
        basis: Target :class:`TwoQubitBasis`.
        verify_atol: Acceptance tolerance of the emitted circuit.

    Returns:
        The gate tuples (three entanglers), or ``None`` if the block is not
        on the SWAP face, needs fewer than three entanglers, or the
        template does not verify.

    Example:
        >>> import numpy as np
        >>> swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
        >>> zz = np.diag(np.exp(-0.15j * np.array([1, -1, -1, 1])))
        >>> ops = synthesize_swap_face_block(swap @ zz, TwoQubitBasis("cx"))
        >>> [name for name, _, _ in ops]
        ['cx', 'rz', 'cx', 'cx']
    """
    u = _check_unitary_4(unitary)
    kak_u = _kak4(u)
    if not _is_swap_face(kak_u[2]) or _count_from_coords(kak_u[2], DEFAULT_ATOL) < 3:
        return None
    for euler_tol in (DEFAULT_ATOL, _EXACT_EULER_ATOL):
        ops, err = _try_synthesize(
            u, kak_u, 3, euler_tol, None, basis, verify_atol, swap_face=True
        )
        if err <= verify_atol:
            return ops
    return None


def _is_swap_face(v: np.ndarray) -> bool:
    """Whether the chamber point lies on the SWAP face ``a = b = π/4``."""
    return abs(float(v[0]) - _PI4) <= 1e-9 and abs(float(v[1]) - _PI4) <= 1e-9


def _layers_unitary(layers: list[_Layer]) -> np.ndarray:
    u = _kron2(layers[0][0], layers[0][1])
    for left, right in layers[1:]:
        u = _kron2(left, right) @ _CZ4 @ u
    return np.asarray(u)


# ---------------------------------------------------------------------------
# Single-qubit Euler synthesis
# ---------------------------------------------------------------------------


def _wrap(angle: float) -> float:
    """Wrap an angle into ``(-π, π]``."""
    w = math.remainder(angle, 2 * math.pi)
    return math.pi if w <= -math.pi else w


def _is_zero_angle(angle: float, atol: float, period: float = 2 * math.pi) -> bool:
    w = math.remainder(angle, period)
    return abs(w) <= atol


def _zxz_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return ``(φ, θ, λ)`` with ``U ∝ Rz(φ) Rx(θ) Rz(λ)``, ``θ ∈ [0, π]``."""
    m = np.asarray(matrix, dtype=complex)
    # Normalise to SU(2): m00 = cos(θ/2) e^{-i(φ+λ)/2},
    # m10 = -i sin(θ/2) e^{i(φ-λ)/2}. The sign ambiguity of the square root
    # only shifts φ by 2π (a global phase).
    m = m / cmath.sqrt(complex(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))
    m00, m10 = complex(m[0, 0]), complex(m[1, 0])
    theta = 2.0 * math.atan2(abs(m10), abs(m00))
    plus = -2.0 * cmath.phase(m00) if abs(m00) > 1e-12 else 0.0
    minus = 2.0 * cmath.phase(m10) + math.pi if abs(m10) > 1e-12 else 0.0
    phi = (plus + minus) / 2.0
    lam = (plus - minus) / 2.0
    return phi, theta, lam


def euler_zxz_ops(
    matrix: np.ndarray, qubit: int, *, atol: float = DEFAULT_ATOL
) -> list[SynthOp]:
    """Synthesize a single-qubit unitary as ``rz · h · rz · h · rz``.

    Uses ``U = e^{iγ} Rz(φ) Rx(θ) Rz(λ)`` with ``Rx(θ) = H Rz(θ) H``.
    Identity rotations are dropped, ``θ ≈ 0`` collapses to one ``rz``,
    ``θ ≈ π/2`` to ``rz · h · rz`` and ``θ ≈ π`` to ``rz · x``. The result
    is exact up to global phase.

    Args:
        matrix: A ``2×2`` unitary.
        qubit: Qubit index written into the emitted operations.
        atol: Angle tolerance used to drop identity rotations.

    Returns:
        A time-ordered list of ``(name, qubits, params)`` tuples.

    Example:
        >>> import numpy as np
        >>> euler_zxz_ops(np.eye(2), 0)
        []
        >>> [op[0] for op in euler_zxz_ops(np.array([[0, 1], [1, 0]]), 0)]
        ['x']
    """
    phi, theta, lam = _zxz_angles(matrix)
    ops: list[SynthOp] = []
    q = (qubit,)
    if _is_zero_angle(theta, atol):
        total = phi + lam
        if not _is_zero_angle(total, atol):
            ops.append(("rz", q, (_wrap(total),)))
        return ops
    if abs(theta - math.pi) <= atol:
        # Rz(φ) Rx(π) Rz(λ) ∝ X · Rz(λ - φ)
        diff = lam - phi
        if not _is_zero_angle(diff, atol):
            ops.append(("rz", q, (_wrap(diff),)))
        ops.append(("x", q, ()))
        return ops
    if abs(theta - _PI2) <= atol:
        # Rz(φ) Rx(π/2) Rz(λ) ∝ Rz(φ - π/2) · H · Rz(λ - π/2)
        if not _is_zero_angle(lam - _PI2, atol):
            ops.append(("rz", q, (_wrap(lam - _PI2),)))
        ops.append(("h", q, ()))
        if not _is_zero_angle(phi - _PI2, atol):
            ops.append(("rz", q, (_wrap(phi - _PI2),)))
        return ops
    if not _is_zero_angle(lam, atol):
        ops.append(("rz", q, (_wrap(lam),)))
    ops.append(("h", q, ()))
    ops.append(("rz", q, (_wrap(theta),)))
    ops.append(("h", q, ()))
    if not _is_zero_angle(phi, atol):
        ops.append(("rz", q, (_wrap(phi),)))
    return ops


def euler_zsx_ops(
    matrix: np.ndarray, qubit: int, *, atol: float = DEFAULT_ATOL
) -> list[SynthOp]:
    """Synthesize a single-qubit unitary as ``rz · sx · rz · sx · rz``.

    The IBM-style counterpart of :func:`euler_zxz_ops`: with
    ``U = e^{iγ} Rz(φ) Rx(θ) Rz(λ)`` and ``Rx(θ) ∝ Rz(π/2)·SX·Rz(θ+π)·SX·
    Rz(π/2)``, the circuit is ``rz(λ+π/2), sx, rz(θ+π), sx, rz(φ+π/2)``.
    Identity rotations are dropped, ``θ ≈ 0`` collapses to one ``rz``,
    ``θ ≈ π/2`` to ``rz · sx · rz`` and ``θ ≈ π`` to ``rz · x``.  The output
    uses only ``rz``, ``sx`` and ``x`` and is exact up to global phase.

    Args:
        matrix: A ``2×2`` unitary.
        qubit: Qubit index written into the emitted operations.
        atol: Angle tolerance used to drop identity rotations.

    Returns:
        A time-ordered list of ``(name, qubits, params)`` tuples.

    Example:
        >>> import numpy as np
        >>> h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        >>> [op[0] for op in euler_zsx_ops(h, 0)]
        ['rz', 'sx', 'rz']
    """
    phi, theta, lam = _zxz_angles(matrix)
    ops: list[SynthOp] = []
    q = (qubit,)

    def rz(angle: float) -> None:
        if not _is_zero_angle(angle, atol):
            ops.append(("rz", q, (_wrap(angle),)))

    if _is_zero_angle(theta, atol):
        rz(phi + lam)
        return ops
    if abs(theta - math.pi) <= atol:
        rz(lam - phi)
        ops.append(("x", q, ()))
        return ops
    if abs(theta - _PI2) <= atol:
        # Rx(π/2) ∝ SX
        rz(lam)
        ops.append(("sx", q, ()))
        rz(phi)
        return ops
    rz(lam + _PI2)
    ops.append(("sx", q, ()))
    rz(theta + math.pi)
    ops.append(("sx", q, ()))
    rz(phi + _PI2)
    return ops


#: Single-qubit Euler bases of the KAK synthesis: name → gates it emits.
EULER_BASES: dict[str, frozenset[str]] = {
    "zxz": frozenset({"rz", "h", "x"}),
    "zsx": frozenset({"rz", "sx", "x"}),
}


def euler_ops(
    matrix: np.ndarray, qubit: int, *, basis: str = "zxz", atol: float = DEFAULT_ATOL
) -> list[SynthOp]:
    """Synthesize a single-qubit unitary in one of the :data:`EULER_BASES`.

    Args:
        matrix: A ``2×2`` unitary.
        qubit: Qubit index written into the emitted operations.
        basis: ``"zxz"`` (:func:`euler_zxz_ops`, gates ``{rz, h, x}``) or
            ``"zsx"`` (:func:`euler_zsx_ops`, gates ``{rz, sx, x}``).
        atol: Angle tolerance used to drop identity rotations.

    Returns:
        A time-ordered list of ``(name, qubits, params)`` tuples.

    Raises:
        ValueError: If ``basis`` is unknown.

    Example:
        >>> import numpy as np
        >>> [op[0] for op in euler_ops(np.array([[0, 1], [1, 0]]), 0, basis="zsx")]
        ['x']
    """
    if basis == "zxz":
        return euler_zxz_ops(matrix, qubit, atol=atol)
    if basis == "zsx":
        return euler_zsx_ops(matrix, qubit, atol=atol)
    raise ValueError(
        f"Unknown Euler basis {basis!r}; expected one of {sorted(EULER_BASES)}."
    )


# ---------------------------------------------------------------------------
# Entanglers and target bases
# ---------------------------------------------------------------------------

_SWAP4 = np.eye(4, dtype=complex)[[0, 2, 1, 3]]


@dataclass(frozen=True)
class Entangler:
    """A native two-qubit gate that is locally equivalent to CZ.

    The KAK templates are written with CZ; each CZ is replaced by the
    entangler through ``CZ = (after[0] ⊗ after[1]) · E · (before[0] ⊗
    before[1])`` (matrices, ``before`` acts first in time), and the
    dressings are folded into the adjacent single-qubit layers.

    Attributes:
        name: Gate name emitted for the entangler (e.g. ``"cx"``).
        matrix: Its ``4×4`` unitary on ``(q0, q1)`` (``q0`` most
            significant).
        before: ``(B0, B1)`` applied to ``(q0, q1)`` before ``E``.
        after: ``(A0, A1)`` applied to ``(q0, q1)`` after ``E``.
        symmetric: Whether ``E`` is invariant under exchanging its qubits
            (then only one orientation is tried).

    Example:
        >>> cx = entangler("cx")
        >>> cx.name, cx.symmetric
        ('cx', False)
    """

    name: str
    matrix: np.ndarray
    before: tuple[np.ndarray, np.ndarray]
    after: tuple[np.ndarray, np.ndarray]
    symmetric: bool

    @classmethod
    def from_matrix(cls, name: str, matrix: np.ndarray) -> Entangler:
        """Build an entangler from its matrix, deriving the CZ dressing by KAK.

        Args:
            name: Gate name emitted for the entangler.
            matrix: Its ``4×4`` unitary; must be locally equivalent to CZ
                (Weyl point ``(π/4, 0, 0)``).

        Returns:
            The :class:`Entangler`.

        Raises:
            ValueError: If ``matrix`` is not a ``4×4`` unitary or not in the
                CZ class (e.g. iSWAP, which needs its own templates).

        Example:
            >>> import numpy as np
            >>> cy = np.eye(4, dtype=complex)
            >>> cy[2:, 2:] = [[0, -1j], [1j, 0]]
            >>> Entangler.from_matrix("cy", cy).name
            'cy'
        """
        e = _check_unitary_4(matrix)
        _, k1e, ve, k2e = _kak4(e)
        if _count_from_coords(ve, 1e-9) != 1:
            raise ValueError(
                f"Entangler {name!r} is not locally equivalent to CZ (Weyl point "
                f"{tuple(round(float(x), 6) for x in ve)}); only CZ-class "
                "entanglers are supported."
            )
        _, k1z, _, k2z = _kak4(_CZ4)
        # E ∝ k1e Can k2e and CZ ∝ k1z Can k2z ⇒ CZ ∝ (k1z k1e^†) E (k2e^† k2z).
        a0, a1 = _factor_local(k1z @ k1e.conj().T)
        b0, b1 = _factor_local(k2e.conj().T @ k2z)
        rebuilt = _kron2(a0, a1) @ e @ _kron2(b0, b1)
        if _distance_up_to_phase(_CZ4, rebuilt) > 1e-12:  # pragma: no cover
            raise ValueError(f"Could not derive the CZ dressing of {name!r}.")
        sym = _distance_up_to_phase(e, _SWAP4 @ e @ _SWAP4) < 1e-12
        return cls(name, e, (b0, b1), (a0, a1), sym)

    def oriented(self, reverse: bool) -> tuple[np.ndarray, Entangler]:
        """Return ``(matrix, dressing)`` of the gate applied as ``(q1, q0)``.

        Args:
            reverse: If False, return this entangler unchanged.

        Returns:
            The ``4×4`` matrix on ``(q0, q1)`` and an entangler whose
            ``before``/``after`` are expressed on ``(q0, q1)``.

        Example:
            >>> m, _ = entangler("cx").oriented(True)
            >>> int(m[1, 3].real)  # CX with control q1, target q0
            1
        """
        if not reverse:
            return self.matrix, self
        m = _SWAP4 @ self.matrix @ _SWAP4
        flipped = Entangler(
            self.name,
            m,
            (self.before[1], self.before[0]),
            (self.after[1], self.after[0]),
            self.symmetric,
        )
        return m, flipped


_CX4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

_ENTANGLERS: dict[str, Entangler] = {
    "cz": Entangler("cz", _CZ4, (_I2, _I2), (_I2, _I2), True),
    # CZ(a, b) = H(b) · CX(a, b) · H(b)
    "cx": Entangler("cx", _CX4, (_I2, _H), (_I2, _H), False),
}


def entangler(name: str) -> Entangler:
    """Return the registered :class:`Entangler` called ``name``.

    Args:
        name: Gate name (``"cz"``, ``"cx"`` or one added with
            :func:`register_entangler`).

    Returns:
        The entangler.

    Raises:
        KeyError: If no entangler of that name is registered.

    Example:
        >>> entangler("cz").symmetric
        True
    """
    try:
        return _ENTANGLERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown entangler {name!r}; registered: {sorted(_ENTANGLERS)}."
        ) from None


def register_entangler(name: str, matrix: np.ndarray) -> Entangler:
    """Register a CZ-class native two-qubit gate for synthesis.

    The gate's matrix must also be known to :func:`quspect.gates.gate_matrix`
    for circuits using it to be simulated or verified.

    Args:
        name: Gate name.
        matrix: Its ``4×4`` unitary (see :meth:`Entangler.from_matrix`).

    Returns:
        The registered :class:`Entangler`.

    Raises:
        ValueError: If the gate is not locally equivalent to CZ.

    Example:
        >>> import numpy as np
        >>> ecr = np.array(
        ...     [[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]]
        ... ) / np.sqrt(2)
        >>> ent = register_entangler("ecr", ecr)
        >>> TwoQubitBasis.from_gates({"ecr", "rz", "sx", "x"})
        TwoQubitBasis(entangler='ecr', euler='zsx')
    """
    ent = Entangler.from_matrix(name, matrix)
    _ENTANGLERS[name] = ent
    return ent


@dataclass(frozen=True)
class TwoQubitBasis:
    """Target gate set of the two-qubit synthesis.

    Attributes:
        entangler: Name of the native entangler (see :func:`entangler`).
        euler: Single-qubit Euler basis (a key of :data:`EULER_BASES`).

    Example:
        >>> TwoQubitBasis.from_gates({"cx", "rz", "sx", "x"})
        TwoQubitBasis(entangler='cx', euler='zsx')
        >>> sorted(TwoQubitBasis().gates)
        ['cz', 'h', 'rz', 'x']
    """

    entangler: str = "cz"
    euler: str = "zxz"

    @property
    def gates(self) -> frozenset[str]:
        """Names of all gates the synthesis may emit in this basis."""
        return EULER_BASES[self.euler] | {self.entangler}

    @classmethod
    def from_gates(cls, gates: Collection[str]) -> TwoQubitBasis | None:
        """Pick a synthesis basis contained in a native gate set.

        The entangler is ``cz`` if available, else ``cx``, else any other
        registered entangler (alphabetically); the Euler basis is
        ``"zxz"`` if ``{rz, h, x}`` is available, else ``"zsx"``.

        Args:
            gates: Native gate names.

        Returns:
            The basis, or ``None`` if the set lacks an entangler or an
            Euler basis.

        Example:
            >>> TwoQubitBasis.from_gates({"cz", "h", "rz", "x", "t"})
            TwoQubitBasis(entangler='cz', euler='zxz')
            >>> TwoQubitBasis.from_gates({"cx", "u3"}) is None
            True
        """
        names = set(gates)
        order = ["cz", "cx", *sorted(set(_ENTANGLERS) - {"cz", "cx"})]
        ent = next((e for e in order if e in names and e in _ENTANGLERS), None)
        euler = next((b for b, req in EULER_BASES.items() if req <= names), None)
        if ent is None or euler is None:
            return None
        return cls(ent, euler)


_DEFAULT_BASIS = TwoQubitBasis()


# ---------------------------------------------------------------------------
# Two-qubit synthesis
# ---------------------------------------------------------------------------


def _ops_layer_matrix(ops: list[SynthOp]) -> np.ndarray:
    """Return the ``2×2`` matrix of a time-ordered single-qubit op list."""
    m = _I2
    for name, _, params in ops:
        if name == "rz":
            g = _rz(params[0])
        elif name == "h":
            g = _H
        elif name == "sx":
            g = _SX
        else:  # "x"
            g = _X
        m = g @ m
    return m


def _try_synthesize(
    u: np.ndarray,
    kak_u: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    k: int,
    atol: float,
    template_coords: np.ndarray | None = None,
    basis: TwoQubitBasis = _DEFAULT_BASIS,
    verify_atol: float = DEFAULT_VERIFY_ATOL,
    swap_face: bool = False,
) -> tuple[list[SynthOp], float]:
    """Build the ``k``-entangler circuit for ``u``; return ``(ops, error)``.

    ``error`` is :func:`_distance_up_to_phase` between ``u`` and the unitary
    of the *emitted* gate list, so it includes the dropped Weyl coordinates
    of a reduced template as well as every Euler rotation snapped to zero
    (or to ``θ = π/2, π``) by the Euler synthesis. Callers reject results
    whose error exceeds the verification tolerance.

    For a non-CZ entangler each template CZ is dressed into the entangler
    (see :class:`Entangler`); asymmetric entanglers are tried in every
    orientation combination and the one with the fewest gates is returned.
    """
    _, k1u, vu, k2u = kak_u
    coords = vu if template_coords is None else template_coords
    layers = _swap_template(float(coords[2])) if swap_face else _template(k, coords)
    v_mat = _layers_unitary(layers)
    _, k1v, vv, k2v = _kak4(v_mat)
    # u = φ_u k1u Can k2u and V = φ_v k1v Can k2v
    #   ⇒ u ∝ (k1u S k1v^†) V (k2v^† S^† k2u) for every local S that
    #   commutes with Can.  Degenerate chamber points have continuous
    #   stabilizers (e.g. V⊗V for a = b = c); the eigensolver then returns
    #   arbitrary k1u, k2u, so S is chosen to make the outer layers short.
    a = _factor_local(k1u)
    b = _factor_local(k1v)
    d = _factor_local(k2u)
    c = _factor_local(k2v)
    b_dag = (b[0].conj().T, b[1].conj().T)
    c_dag = (c[0].conj().T, c[1].conj().T)

    def fixed(
        stab: tuple[np.ndarray, np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lefts = [a[i] @ stab[i] @ b_dag[i] for i in (0, 1)]
        rights = [c_dag[i] @ stab[i].conj().T @ d[i] for i in (0, 1)]
        out = [(layers[0][0] @ rights[0], layers[0][1] @ rights[1])]
        out += [(lft, rgt) for lft, rgt in layers[1:]]
        out[-1] = (lefts[0] @ out[-1][0], lefts[1] @ out[-1][1])
        return out

    cands = _stabilizer_candidates(k, vv, a, b, c, d, fit_cosets=swap_face)
    # SWAP face: pairs (L, R) with L·Can·R = Can but R ≠ L^† (see
    # _swap_face_gauges); their right factors replace S^†
    gauges = _swap_face_gauges(a, b, c, d) if swap_face else []
    right_stabs = [(s0.conj().T, s1.conj().T) for s0, s1 in cands]
    cands += [left for left, _ in gauges]
    right_stabs += [right for _, right in gauges]
    if len(cands) == 1:  # k = 0 or nothing to choose: no scoring needed
        return _emit_fixed_layers(u, fixed(cands[0]), k, atol, basis, verify_atol)
    # k ≥ 1.  Score the candidates by the Euler lengths of their outer layers
    # (the inner ones do not depend on S) and emit the identity plus the
    # best strictly better-scoring ones.  The products of ``fixed`` run
    # batched over all candidates (each item is the same BLAS call as the
    # single ``@``, operand layouts included) and the lengths come from one
    # native call, so the factored layers and scores equal ``fixed`` +
    # ``len(euler_ops(...))`` exactly.
    firsts = []
    lasts = []
    for i in (0, 1):
        stack = np.array([stab[i] for stab in cands])
        lefts = np.matmul(np.matmul(a[i], stack), b_dag[i])
        if gauges:
            right_stack = np.array([stab[i] for stab in right_stabs])
        else:
            right_stack = stack.conj().transpose(0, 2, 1)
        rights = np.matmul(np.matmul(c_dag[i], right_stack), d[i])
        firsts.append(np.matmul(layers[0][i], rights))
        lasts.append(np.matmul(lefts, layers[-1][i]))
    lengths = _euler_lengths(
        np.concatenate([firsts[0], firsts[1], lasts[0], lasts[1]]), basis.euler, atol
    )
    n = len(cands)
    scores = [
        lengths[j] + lengths[n + j] + lengths[2 * n + j] + lengths[3 * n + j]
        for j in range(n)
    ]

    def factored_of(j: int) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (firsts[0][j].copy(), firsts[1][j].copy()),
            *layers[1:-1],
            (lasts[0][j].copy(), lasts[1][j].copy()),
        ]

    if gauges:
        # SWAP face, non-CZ entanglers: the CZ-frame score ties with the
        # identity for the gauges that cancel through the SWAP (the
        # emitted length is only known after dressing), so emit every
        # candidate that is not worse
        better = sorted(
            (j for j in range(1, n) if scores[j] <= scores[0]),
            key=lambda j: (scores[j], j),
        )
    else:
        better = sorted(
            (j for j in range(1, n) if scores[j] < scores[0]),
            key=lambda j: (scores[j], j),
        )[:_STABILIZER_EMITS]
    if basis.entangler == "cz":
        # With CZ the emitted length is the score plus a constant (the inner
        # layers and the CZ gates), so the result is the first candidate in
        # score order that passes verification, else the identity, else (all
        # failing) the best-scoring one — exactly what emitting all of them
        # and keeping the smallest (failed, length) key returns.
        fallback: tuple[list[SynthOp], float] | None = None
        for j in better:
            result = _emit_fixed_layers(u, factored_of(j), k, atol, basis, verify_atol)
            if result[1] <= verify_atol:
                return result
            if fallback is None:
                fallback = result
        result = _emit_fixed_layers(u, factored_of(0), k, atol, basis, verify_atol)
        if result[1] <= verify_atol or fallback is None:
            return result
        return fallback
    best: tuple[list[SynthOp], float] | None = None
    best_key: tuple[bool, int] | None = None
    for j in [0, *better]:
        result = _emit_fixed_layers(u, factored_of(j), k, atol, basis, verify_atol)
        key = (result[1] > verify_atol, len(result[0]))
        if best_key is None or key < best_key:
            best, best_key = result, key
    assert best is not None
    return best


_EULER_CODE = {"zxz": 0, "zsx": 1}


def _euler_lengths(mats: np.ndarray, basis: str, atol: float) -> list[int]:
    """``len(euler_ops(m, 0, basis=basis, atol=atol))`` for each matrix.

    Uses the native kernel (bit-identical to the Python arithmetic of
    :func:`euler_ops`, checked once against it) when available; any matrix
    it does not reproduce falls back to :func:`euler_ops`.
    """
    core = _native_euler()
    if core is not None and basis in _EULER_CODE:
        out = core._euler_lengths(mats, _EULER_CODE[basis], atol)
        return [
            int(x) if x >= 0 else len(euler_ops(m, 0, basis=basis, atol=atol))
            for x, m in zip(out, mats, strict=True)
        ]
    return [len(euler_ops(m, 0, basis=basis, atol=atol)) for m in mats]


@functools.cache
def _native_euler() -> Any | None:
    """The extension module if its Euler-length kernel matches :func:`euler_ops`."""
    from quspect._native_loader import load_core

    try:
        core = load_core()
    except ImportError:  # pragma: no cover - broken build
        return None
    if core is None or not hasattr(core, "_euler_lengths"):
        return None
    rng = np.random.default_rng(2024)
    raw = rng.normal(size=(48, 2, 2)) + 1j * rng.normal(size=(48, 2, 2))
    mats = [np.linalg.qr(x)[0] for x in raw]
    for theta in (0.0, 1e-10, _PI2, math.pi, -_PI2, 2.5):
        mats.append(_rz(0.3) @ _rx(theta) @ _rz(-1.1))
        mats.append(_H @ _rz(theta) @ _H)
    mats += [_I2, _H, _X, _Y, _Z, _S, _SX]
    arr = np.array(mats)
    for basis, code in _EULER_CODE.items():
        for atol in (DEFAULT_ATOL, _EXACT_EULER_ATOL):
            got = core._euler_lengths(arr, code, atol)
            ref = [len(euler_ops(m, 0, basis=basis, atol=atol)) for m in mats]
            if [int(x) for x in got] != ref:  # pragma: no cover - platform specific
                return None
    return core


#: Stabilizer candidates (besides the identity) fully emitted per template.
_STABILIZER_EMITS = 4

_Pair = tuple[np.ndarray, np.ndarray]
_PAULI_LOCAL: tuple[_Pair, ...] = ((_X, _X), (_Y, _Y), (_Z, _Z))


def _axis_angle(w: np.ndarray, pauli: np.ndarray) -> float:
    """Angle ``φ`` of the rotation ``exp(-iφ/2·P)`` closest to ``w`` (up to phase)."""
    x = complex(np.trace(w))
    y = 1j * complex(np.trace(pauli @ w))
    return float(math.atan2(2.0 * (x * y.conjugate()).real, abs(x) ** 2 - abs(y) ** 2))


def _axis_rot(pauli: np.ndarray, phi: float) -> np.ndarray:
    return np.asarray(math.cos(phi / 2) * _I2 - 1j * math.sin(phi / 2) * pauli)


def _stabilizer_candidates(
    k: int,
    v: np.ndarray,
    k1u: _Pair,
    k1v: _Pair,
    k2v: _Pair,
    k2u: _Pair,
    *,
    fit_cosets: bool = False,
) -> list[_Pair]:
    """Local unitaries ``S = s0⊗s1`` commuting with ``Can(v)`` worth trying.

    The factors of the KAK locals are given as ``(q0, q1)`` pairs.  The
    identity and the Pauli pairs ``P⊗P`` (which commute with every
    ``Can``); for degenerate chamber points (``a = b = c``: ``V⊗V``;
    ``b = c = 0``: ``Rx⊗Rx``; ``a = b``: ``Rz(φ)⊗Rz(φ)``; ``b = c``:
    ``Rx(φ)⊗Rx(φ)``) the elements that turn one outer single-qubit factor
    into the identity (or the closest rotation), times the discrete part
    of the stabilizer outside that family.

    The chamber keeps the sign of ``c``, and ``b = -c`` is degenerate as
    well: ``Can(a, b, c) = (I⊗Z)·Can(-a, -b, -c)·(I⊗Z)`` and ``Can(-w) =
    Can(w)^†`` has the stabilizer of ``Can(w)``, so the stabilizer of the
    mirrored point is that of ``(a, b, |c|)`` conjugated by ``Z`` on q1
    (``V⊗Z·V·Z`` for ``a = b = -c`` — e.g. Heisenberg ``XX + YY - ZZ``
    blocks —, ``Rx(φ)⊗Rx(-φ)`` for ``b = -c``).  Those candidates are added
    to the ones of the unmirrored rules.

    ``fit_cosets`` also fits the members ``r·P`` of the discrete coset of
    the ``Rz(φ)⊗Rz(φ)`` / ``Rx(φ)⊗Rx(φ)`` family to the targets (see
    :func:`_stabilizer_family`); the SWAP-face template uses it.
    """
    eye: _Pair = (_I2, _I2)
    if k == 0:
        return [eye]
    a, b, c = (float(x) for x in v)
    # s makes the outer factor identity: left_i = a_i s b_i^†, right_i = c_i^† s^† d_i
    targets = [
        (0, k1u[0].conj().T @ k1v[0]),
        (1, k1u[1].conj().T @ k1v[1]),
        (0, k2u[0] @ k2v[0].conj().T),
        (1, k2u[1] @ k2v[1].conj().T),
    ]
    out = _stabilizer_family(a, b, c, targets, fit_cosets)
    if _MIRRORED_STABILIZERS and c < -_STABILIZER_TOL and abs(b + c) <= _STABILIZER_TOL:
        # solve for s1' = Z·s1·Z in the family of (a, b, |c|), then map back
        mirrored = [(i, _Z @ w @ _Z if i == 1 else w) for i, w in targets]
        out += [
            (s0, _Z @ s1 @ _Z)
            for s0, s1 in _stabilizer_family(a, b, -c, mirrored, fit_cosets)[1:]
        ]
    return out


_STABILIZER_TOL = 1e-9


def _swap_face_gauges(
    k1u: _Pair, k1v: _Pair, k2v: _Pair, k2u: _Pair
) -> list[tuple[_Pair, _Pair]]:
    """Gauge pairs ``(L, R)`` with ``L·Can·R = Can`` on the SWAP face.

    For ``a = b = π/4``, ``Can·(Rz(φ)⊗Rz(ψ)) = (Rz(ψ)⊗Rz(φ))·Can``, so
    ``L = Rz(-ψ)⊗Rz(-φ)`` and ``R = Rz(φ)⊗Rz(ψ)`` (and the same with an
    ``X⊗X`` on both sides) leave ``Can`` unchanged although ``R ≠ L^†``.
    The eigensolver may return KAK factors anywhere in that torus (OpenBLAS
    on arm64 Linux leaves ``Z`` before q0 and after q1 of SWAP·ZZ(0.3));
    ``φ`` and ``ψ`` are fitted to make an outer factor the identity.
    """
    # left_i = k1u_i L_i k1v_i^† ≈ I ⇒ L_i ≈ k1u_i^† k1v_i
    # right_i = k2v_i^† R_i k2u_i ≈ I ⇒ R_i ≈ k2v_i k2u_i^†
    lw = [k1u[i].conj().T @ k1v[i] for i in (0, 1)]
    rw = [k2v[i] @ k2u[i].conj().T for i in (0, 1)]
    out: list[tuple[_Pair, _Pair]] = []
    for p in (_I2, _X):
        # L = (Rz(-ψ)·P, Rz(-φ)·P), R = (P·Rz(φ), P·Rz(ψ))
        phis = {0.0, -_axis_angle(lw[1] @ p, _Z), _axis_angle(p @ rw[0], _Z)}
        psis = {0.0, -_axis_angle(lw[0] @ p, _Z), _axis_angle(p @ rw[1], _Z)}
        for phi in sorted(phis):
            for psi in sorted(psis):
                if p is _I2 and phi == 0.0 and psi == 0.0:
                    continue
                left = (_axis_rot(_Z, -psi) @ p, _axis_rot(_Z, -phi) @ p)
                right = (p @ _axis_rot(_Z, phi), p @ _axis_rot(_Z, psi))
                out.append((left, right))
    return out


#: Use the stabilizers of the mirrored degenerate points ``b = -c`` (round 17).
_MIRRORED_STABILIZERS = True


def _stabilizer_family(
    a: float,
    b: float,
    c: float,
    targets: list[tuple[int, np.ndarray]],
    fit_cosets: bool = False,
) -> list[_Pair]:
    """Candidates of :func:`_stabilizer_candidates` for the point ``(a, b, c)``.

    ``targets`` are ``(slot, w)``: a factor ``s_slot = w`` makes one outer
    single-qubit factor the identity.  The identity comes first.

    With ``fit_cosets`` the members ``r·P`` of the discrete coset of an
    ``Rz(φ)⊗Rz(φ)`` / ``Rx(φ)⊗Rx(φ)`` family are fitted to the targets
    directly (``r ≈ w·P``) instead of only ``r·P`` with ``r ≈ w``.  Which
    coset the eigensolver's KAK factors land in depends on the LAPACK
    build, so without it SWAP·ZZ blocks keep long outer layers on some
    platforms (arm64 Linux).  It is limited to the SWAP-face template:
    elsewhere the extra, locally shorter candidates merge worse with the
    neighbouring gates (BV on a line: single-qubit gates ×2).
    """
    eye: _Pair = (_I2, _I2)
    tol = _STABILIZER_TOL
    if abs(a - b) > tol and abs(b - c) > tol and abs(b) > tol:
        return [eye, *_PAULI_LOCAL]
    cands: list[_Pair] = [eye]
    # discrete part of the stabilizer not already in the continuous family
    paulis: tuple[_Pair, ...]
    fitted: list[_Pair] = []  # V⊗V (a = b = c) already holds every coset
    if abs(a - b) <= tol and abs(b - c) <= tol:
        cands += [(w, w) for _, w in targets]
        paulis = _PAULI_LOCAL  # V·P ⊗ V·P: other members of the same family
    elif abs(b) <= tol and abs(c) <= tol:
        paulis = (_PAULI_LOCAL[2],)
        phis: list[list[float]] = [[0.0], [0.0]]
        for slot, w in targets:
            phis[slot].append(_axis_angle(w, _X))
        for p0 in phis[0]:
            for p1 in phis[1]:
                if p0 or p1:
                    cands.append((_axis_rot(_X, p0), _axis_rot(_X, p1)))
    else:
        pauli = _Z if abs(a - b) <= tol else _X
        for _, w in targets:
            r = _axis_rot(pauli, _axis_angle(w, pauli))
            cands.append((r, r))
        paulis = (_PAULI_LOCAL[0] if abs(a - b) <= tol else _PAULI_LOCAL[2],)
        if fit_cosets:
            p = paulis[0][0]
            for _, w in targets:
                r = _axis_rot(pauli, _axis_angle(w @ p, pauli)) @ p
                fitted.append((r, r))
    out = cands + [(s0 @ p0, s1 @ p1) for s0, s1 in cands for p0, p1 in paulis]
    return out + fitted


def _emit_fixed_layers(
    u: np.ndarray,
    factored: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    atol: float,
    basis: TwoQubitBasis,
    verify_atol: float,
) -> tuple[list[SynthOp], float]:
    """Emit factored CZ-template layers in ``basis`` (see :func:`_try_synthesize`)."""
    if basis.entangler == "cz":
        return _emit_layers(u, factored, [(_CZ4, None)] * k, atol, basis)
    ent = entangler(basis.entangler)
    choices = (False,) if ent.symmetric else (False, True)
    best: tuple[list[SynthOp], float] | None = None
    best_key: tuple[bool, int] | None = None
    for combo in product(choices, repeat=k):
        dressed = [list(f) for f in factored]
        gates: list[tuple[np.ndarray, tuple[int, int] | None]] = []
        for i, rev in enumerate(combo):
            mat, dress = ent.oriented(rev)
            # CZ = (A0⊗A1) E (B0⊗B1): B joins layer i (after it in time),
            # A joins layer i + 1 (before it in time).
            dressed[i] = [
                dress.before[0] @ dressed[i][0],
                dress.before[1] @ dressed[i][1],
            ]
            dressed[i + 1] = [
                dressed[i + 1][0] @ dress.after[0],
                dressed[i + 1][1] @ dress.after[1],
            ]
            gates.append((mat, (1, 0) if rev else (0, 1)))
        result = _emit_layers(u, [(a, b) for a, b in dressed], gates, atol, basis)
        # 先要求通过校验，再取门数最少的朝向组合
        key = (result[1] > verify_atol, len(result[0]))
        if best_key is None or key < best_key:
            best, best_key = result, key
    assert best is not None
    return best


def _emit_layers(
    u: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray]],
    gates: list[tuple[np.ndarray, tuple[int, int] | None]],
    atol: float,
    basis: TwoQubitBasis,
) -> tuple[list[SynthOp], float]:
    """Euler-synthesize ``layers`` around the entanglers ``gates``.

    ``gates[i]`` is ``(matrix on (q0, q1), operand order)`` of the entangler
    between ``layers[i]`` and ``layers[i + 1]`` (``None`` order: ``(0, 1)``).
    Returns the gate list and its verified distance to ``u``.
    """
    ops: list[SynthOp] = []
    rebuilt: np.ndarray | None = None
    name = basis.entangler
    for idx, (left, right) in enumerate(layers):
        left_ops = euler_ops(left, 0, basis=basis.euler, atol=atol)
        right_ops = euler_ops(right, 1, basis=basis.euler, atol=atol)
        layer = _kron2(_ops_layer_matrix(left_ops), _ops_layer_matrix(right_ops))
        if rebuilt is None:
            rebuilt = layer
        else:
            mat, order = gates[idx - 1]
            ops.append((name, order or (0, 1), ()))
            rebuilt = layer @ mat @ rebuilt
        ops.extend(left_ops)
        ops.extend(right_ops)
    assert rebuilt is not None
    return ops, _distance_up_to_phase(u, rebuilt)


def synthesize_two_qubit_ops(
    unitary: np.ndarray,
    *,
    atol: float = DEFAULT_ATOL,
    verify_atol: float = DEFAULT_VERIFY_ATOL,
    basis: TwoQubitBasis | None = None,
    max_infidelity: float = 0.0,
) -> list[SynthOp]:
    """Synthesize a two-qubit unitary into time-ordered gate tuples.

    This is the allocation-light core used by
    :class:`~quspect.transpiler.ConsolidateBlocksPass`; see
    :func:`synthesize_two_qubit` for the circuit-returning wrapper.

    The Weyl coordinates only propose the smallest CZ count ``k`` (a
    coordinate within ``atol`` of a chamber face is treated as lying on
    it). Every candidate is then verified on the unitary of the emitted
    gates with the phase-aligned element-wise distance
    ``max|U - e^{iφ}V| ≤ verify_atol``, which is linear in any dropped
    angle. A candidate that fails is retried with exact Euler angles
    (no small rotations dropped), and then with ``k + 1`` CZ gates. Hence a
    weak interaction is only removed when doing so costs less than
    ``verify_atol``: e.g. ``CP(θ)`` (Weyl point ``(θ/4, 0, 0)``) is
    written without CZ only for ``θ < 4e-9``, where the dropped
    ``exp(iθ/4·ZZ)`` costs ``≤ 1e-9`` (its local ``rz(θ/2)`` parts are kept
    when dropping them too would exceed the tolerance); ``CP(1e-6)`` keeps
    two CZ gates and is reproduced to ~1e-15.

    Args:
        unitary: A ``4×4`` unitary matrix (``q0`` most significant).
        atol: Tolerance on the Weyl coordinates (and Euler angles) used to
            propose a reduced CZ count and to drop identity rotations.
        verify_atol: Acceptance tolerance of the synthesized circuit (see
            the error budget in the module documentation).
        basis: Target :class:`TwoQubitBasis` (default: ``cz`` with the
            ``{rz, h, x}`` Euler basis).  The entangler count does not
            depend on the basis.
        max_infidelity: Approximation budget (default ``0``: exact).  When
            positive, the circuit with the fewest entanglers whose average
            gate infidelity to ``unitary`` (see :func:`approximation_error`)
            is at most ``max_infidelity`` is returned: the weak part of the
            interaction is dropped (Weyl point projected onto
            ``(0, 0, 0)``, ``(π/4, 0, 0)`` or the ``c = 0`` face) while the
            local parts are kept, e.g. ``CP(θ)`` becomes
            ``rz(θ/2) ⊗ rz(θ/2)`` with infidelity ``θ²/20``.  See
            :func:`synthesize_two_qubit_approx` for the error of the result.

    Returns:
        A list of ``(name, qubits, params)`` tuples on local qubits 0 and 1,
        using only the gates of ``basis`` (by default ``cz``, ``h``, ``rz``
        and ``x``).

    Raises:
        ValueError: If ``unitary`` is not a ``4×4`` unitary matrix.
        RuntimeError: If no template reproduces the unitary (should not
            happen for valid input).

    Example:
        >>> import numpy as np
        >>> ops = synthesize_two_qubit_ops(np.diag([1, 1, 1, -1]))
        >>> sum(1 for name, _, _ in ops if name == "cz")
        1
        >>> ops = synthesize_two_qubit_ops(
        ...     np.diag([1, 1, 1, -1]), basis=TwoQubitBasis("cx", "zsx"))
        >>> sorted({name for name, _, _ in ops})
        ['cx', 'rz', 'sx']
    """
    if max_infidelity > 0:
        return synthesize_two_qubit_approx(
            unitary,
            max_infidelity,
            atol=atol,
            verify_atol=verify_atol,
            basis=basis,
        )[0]
    basis = _DEFAULT_BASIS if basis is None else basis
    key = (
        matrix_key(unitary),
        atol,
        verify_atol,
        basis.entangler,
        basis.euler,
        _SWAP_FACE_TEMPLATE,
    )
    cached = _SYNTH_2Q_CACHE.get(key)
    if cached is not None:
        return list(cached)
    ops = _synthesize_two_qubit_ops(unitary, atol, verify_atol, basis)
    _SYNTH_2Q_CACHE.put(key, tuple(ops))
    return ops


def synthesize_two_qubit_approx(
    unitary: np.ndarray,
    max_infidelity: float,
    *,
    atol: float = DEFAULT_ATOL,
    verify_atol: float = DEFAULT_VERIFY_ATOL,
    basis: TwoQubitBasis | None = None,
    max_entanglers: int = 2,
) -> tuple[list[SynthOp], ApproximationError | None]:
    """Approximate two-qubit synthesis with an explicit error report.

    Tries the entangler counts below the exact one (from :func:`num_cz_required`)
    in increasing order.  For each count ``k`` the Weyl point ``(a, b, c)``
    of ``unitary`` is projected onto the points reachable with ``k`` CZ
    gates (``(0, 0, 0)``, ``(π/4, 0, 0)``, ``(a, b, 0)``) keeping the local
    KAK factors; a projection whose predicted average gate infidelity
    exceeds ``max_infidelity`` is skipped without synthesis.  Otherwise the
    ``k``-entangler circuit is emitted and its *actual* error (of the
    emitted gates, Euler snapping included) is measured with
    :func:`approximation_error`; it is accepted when the infidelity is at
    most ``max_infidelity``.  If no reduced count qualifies, the exact
    synthesis of :func:`synthesize_two_qubit_ops` is returned.

    Args:
        unitary: A ``4×4`` unitary matrix (``q0`` most significant).
        max_infidelity: Largest accepted average gate infidelity
            ``1 - F̄(U, V)`` of the approximation (non-negative).
        atol: See :func:`synthesize_two_qubit_ops`.
        verify_atol: See :func:`synthesize_two_qubit_ops` (exact fallback).
        basis: Target :class:`TwoQubitBasis` (default ``cz`` +
            ``{rz, h, x}``).
        max_entanglers: Only approximations with at most this many
            entanglers are considered (``0``: only drop the interaction
            entirely, i.e. replace the block by single-qubit gates).

    Returns:
        ``(ops, error)``: the gate tuples and, if a reduced entangler count
        was used, its :class:`ApproximationError` (``None`` for the exact
        synthesis, which is verified to ``verify_atol`` element-wise).

    Raises:
        ValueError: If ``unitary`` is not a ``4×4`` unitary matrix or
            ``max_infidelity`` is negative.
        RuntimeError: If synthesis fails numerically.

    Example:
        >>> import numpy as np
        >>> cp = np.diag([1, 1, 1, np.exp(2e-6j)])
        >>> ops, err = synthesize_two_qubit_approx(cp, 1e-12)
        >>> sum(1 for name, _, _ in ops if name == "cz"), err.infidelity < 1e-12
        (0, True)
        >>> ops, err = synthesize_two_qubit_approx(cp, 1e-14)
        >>> sum(1 for name, _, _ in ops if name == "cz"), err is None
        (2, True)
    """
    if not max_infidelity >= 0:
        raise ValueError(
            f"max_infidelity must be non-negative, got {max_infidelity!r}."
        )
    basis = _DEFAULT_BASIS if basis is None else basis
    key = (
        matrix_key(unitary),
        atol,
        verify_atol,
        basis.entangler,
        basis.euler,
        max_infidelity,
        max_entanglers,
        _SWAP_FACE_TEMPLATE,
    )
    hit = _APPROX_2Q_CACHE.get(key)
    if hit is not None:
        return list(hit[0]), hit[1]
    u = _check_unitary_4(unitary)
    kak_u = _kak4(u)
    result = _approximate_two_qubit_ops(
        u, max_infidelity, atol, basis, max_entanglers, kak_u
    )
    if result is None:
        exact_key = (
            matrix_key(unitary),
            atol,
            verify_atol,
            basis.entangler,
            basis.euler,
            _SWAP_FACE_TEMPLATE,
        )
        cached = _SYNTH_2Q_CACHE.get(exact_key)
        if cached is None:
            # 与 synthesize_two_qubit_ops 相同的结果，复用已算好的 KAK
            cached = tuple(
                _synthesize_two_qubit_ops(u, atol, verify_atol, basis, kak_u)
            )
            _SYNTH_2Q_CACHE.put(exact_key, cached)
        _APPROX_2Q_CACHE.put(key, (cached, None))
        return list(cached), None
    _APPROX_2Q_CACHE.put(key, (tuple(result[0]), result[1]))
    return result


def _projected_infidelity(v: np.ndarray, k: int) -> float:
    """Average gate infidelity of projecting Weyl point ``v`` to ``k`` CZ.

    With the local KAK factors kept, ``U^† V`` is locally equivalent to
    ``Can(Δ)`` with ``Δ = v - proj_k(v)``; its trace in the magic basis
    is the mean of the four eigenphases ``_CAN_PHASES @ Δ``.
    """
    a, b, c = (float(x) for x in v)
    if k == 0:
        delta = (a, b, c)
    elif k == 1:
        delta = (a - _PI4, b, c)
    else:
        delta = (0.0, 0.0, c)
    phases = _CAN_PHASES @ np.array(delta, dtype=float)
    half = 0.5 * (phases[:, None] - phases[None, :])
    deficit = 2.0 * float(np.sum(np.sin(half) ** 2)) / 16.0
    return 4.0 * deficit / 5.0


def _ops_unitary(ops: list[SynthOp], basis: TwoQubitBasis) -> np.ndarray:
    """``4×4`` unitary of time-ordered synthesis ops on local qubits (0, 1)."""
    ent = entangler(basis.entangler)
    u = np.eye(4, dtype=complex)
    for name, qubits, params in ops:
        if len(qubits) == 2:
            g = ent.matrix if tuple(qubits) == (0, 1) else ent.oriented(True)[0]
        else:
            m = _ops_layer_matrix([(name, qubits, params)])
            g = _kron2(m, _I2) if qubits[0] == 0 else _kron2(_I2, m)
        u = g @ u
    return u


def _approximate_two_qubit_ops(
    unitary: np.ndarray,
    max_infidelity: float,
    atol: float,
    basis: TwoQubitBasis,
    max_entanglers: int = 2,
    kak_u: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[SynthOp], ApproximationError] | None:
    """Reduced-entangler approximation within ``max_infidelity``, or ``None``."""
    u = _check_unitary_4(unitary)
    if kak_u is None:
        kak_u = _kak4(u)
    k_exact = _count_from_coords(kak_u[2], atol)
    euler_tols = (atol,) if atol <= _EXACT_EULER_ATOL else (atol, _EXACT_EULER_ATOL)
    for k in range(min(k_exact, max_entanglers + 1)):
        # 预测值是保留局部因子时的精确不保真度；留一点余量给数值误差，
        # 最终以实际发出的门的误差为准
        if _projected_infidelity(kak_u[2], k) > max_infidelity * (1 + 1e-6):
            continue
        for euler_tol in euler_tols:
            ops, _ = _try_synthesize(u, kak_u, k, euler_tol, None, basis, math.inf)
            err = approximation_error(u, _ops_unitary(ops, basis))
            if err.infidelity <= max_infidelity:
                return ops, err
    return None


def _synthesize_two_qubit_ops(
    unitary: np.ndarray,
    atol: float,
    verify_atol: float,
    basis: TwoQubitBasis = _DEFAULT_BASIS,
    kak_u: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[SynthOp]:
    """Uncached core of :func:`synthesize_two_qubit_ops` (``kak_u``: its KAK)."""
    u = _check_unitary_4(unitary)
    if kak_u is None:
        kak_u = _kak4(u)
    k_min = _count_from_coords(kak_u[2], atol)
    euler_tols = (atol,) if atol <= _EXACT_EULER_ATOL else (atol, _EXACT_EULER_ATOL)
    best: tuple[list[SynthOp], float] | None = None
    if _SWAP_FACE_TEMPLATE and k_min == 3 and _is_swap_face(kak_u[2]):
        # SWAP-class blocks: the SWAP-shaped template has (almost) no inner
        # single-qubit gates; fall back to the generic one if it fails
        for euler_tol in euler_tols:
            result = _try_synthesize(
                u, kak_u, 3, euler_tol, None, basis, verify_atol, swap_face=True
            )
            if result[1] <= verify_atol:
                return result[0]
            if best is None or result[1] < best[1]:
                best = result
    for k in range(k_min, 4):
        candidates: list[np.ndarray | None] = [None]
        if k == 3:
            # Guard against a mirror mismatch right at the a = π/4 face.
            mirrored = kak_u[2].copy()
            mirrored[2] = -mirrored[2]
            candidates.append(mirrored)
        for coords in candidates:
            for euler_tol in euler_tols:
                result = _try_synthesize(
                    u, kak_u, k, euler_tol, coords, basis, verify_atol
                )
                if result[1] <= verify_atol:
                    return result[0]
                if best is None or result[1] < best[1]:
                    best = result
    raise RuntimeError(
        "Two-qubit synthesis failed to reproduce the unitary "
        f"(best error {best[1] if best else math.inf:.3e})."
    )


def synthesize_two_qubit(
    unitary: np.ndarray,
    *,
    atol: float = DEFAULT_ATOL,
    verify_atol: float = DEFAULT_VERIFY_ATOL,
    basis: TwoQubitBasis | None = None,
) -> QuantumCircuit:
    """Synthesize a two-qubit unitary with the minimal number of CZ gates.

    The returned circuit acts on qubits 0 and 1 (``q0`` is the most
    significant qubit of ``unitary``), uses 0–3 ``cz`` gates and only
    ``h``, ``rz`` and ``x`` single-qubit gates, and equals ``unitary`` up to
    global phase.

    Args:
        unitary: A ``4×4`` unitary matrix.
        atol: Tolerance on the Weyl coordinates (and Euler angles) used to
            propose a reduced CZ count and to drop identity rotations.
        verify_atol: Acceptance tolerance on ``max|U - e^{iφ}V|`` of the
            result; see :func:`synthesize_two_qubit_ops`.
        basis: Target :class:`TwoQubitBasis` (default ``cz`` + ``{rz, h,
            x}``); e.g. ``TwoQubitBasis("cx")`` emits CX gates.

    Returns:
        The synthesized :class:`~quspect.circuit.circuit.QuantumCircuit`.

    Raises:
        ValueError: If ``unitary`` is not a ``4×4`` unitary matrix.
        RuntimeError: If synthesis fails numerically.

    Example:
        >>> import numpy as np
        >>> swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
        >>> circ = synthesize_two_qubit(swap)
        >>> sum(1 for ins in circ if ins.gate.name == "cz")
        3
    """
    from quspect.circuit.circuit import QuantumCircuit

    ops = synthesize_two_qubit_ops(
        unitary, atol=atol, verify_atol=verify_atol, basis=basis
    )
    return QuantumCircuit._from_instruction_snapshot(ops_to_instructions(ops, (0, 1)))


@functools.cache
def _standard_gates() -> dict[str, Gate]:
    """Parameter-free standard gates by name (for :func:`ops_to_instructions`)."""
    from quspect.gates.standard import ALL_STANDARD

    return {g.name: g for g in ALL_STANDARD}


def ops_to_instructions(
    ops: list[SynthOp], qubits: tuple[int, int]
) -> list[CircuitInstruction]:
    """Convert synthesized gate tuples into circuit instructions.

    Args:
        ops: Gate tuples on local qubits 0/1, as returned by
            :func:`synthesize_two_qubit_ops`.
        qubits: The circuit qubits that local qubits 0 and 1 map to.

    Returns:
        The corresponding :class:`CircuitInstruction` list, built from the
        standard gate templates of :mod:`quspect.gates`.

    Raises:
        KeyError: If an op name is neither ``rz`` nor a parameter-free
            standard gate (``cz``, ``cx``, ``h``, ``x``, ``sx``, ...).

    Example:
        >>> ins = ops_to_instructions([("cz", (0, 1), ())], (3, 5))
        >>> ins[0].qubits
        (3, 5)
    """
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates.parameterized import RZ

    fixed = _standard_gates()
    out: list[CircuitInstruction] = []
    for name, local, params in ops:
        gate = RZ.with_params(*params) if name == "rz" else fixed[name]
        out.append(
            CircuitInstruction(
                type=InstructionType.GATE,
                gate=gate,
                qubits=tuple(qubits[q] for q in local),
            )
        )
    return out
