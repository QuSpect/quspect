"""Native KAK synthesis backend: deterministic, equivalent to the Python code.

The native backend (:mod:`quspect.synthesis._native_kak`) runs
``_synthesize_two_qubit_ops``, ``_synthesize_swap_face_block``, the KAK count
of :func:`num_cz_required` and the "no approximation" decision of
:func:`synthesize_two_qubit_approx` on the deterministic numerics layer
(fixed-order arithmetic, Jacobi decompositions, correctly rounded math
functions).  Its output is the compiler's reference and is bit-identical on
every platform (:func:`test_golden_digests_are_platform_independent`).

The Python implementation is only a fallback for builds without the extension
module.  It uses numpy's LAPACK and the platform libm, so the two are *not*
compared bit for bit: these tests check that both produce valid circuits
(verified to ``verify_atol``) with the same entangler counts, the same
Weyl coordinates and the same decisions.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

import quspect.synthesis.two_qubit as tq
from quspect.synthesis import _native_kak
from quspect.synthesis.two_qubit import TwoQubitBasis

NATIVE = tq._native_kak()

pytestmark = pytest.mark.skipif(NATIVE is None, reason="extension module not built")

#: The numpy-semantics helpers (single-qubit synthesis, block products) are
#: only enabled where they reproduce numpy bit for bit.
needs_numpy = pytest.mark.skipif(
    NATIVE is None or not NATIVE.numpy_compat,
    reason="numpy-semantics helpers not enabled on this platform",
)

BASES = [TwoQubitBasis(e, x) for e in ("cz", "cx") for x in ("zxz", "zsx")]

_SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _haar(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return np.asarray(q * (np.diag(r) / np.abs(np.diag(r))))


def _local(rng: np.random.Generator) -> np.ndarray:
    return np.asarray(np.kron(_haar(rng, 2), _haar(rng, 2)))


def _zz(theta: float) -> np.ndarray:
    return np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))


def _unitaries(seed: int) -> list[np.ndarray]:
    """Random and structured (degenerate Weyl points, near faces) unitaries."""
    rng = np.random.default_rng(seed)
    cores = [
        np.eye(4, dtype=complex),
        _CZ,
        _SWAP,
        _SWAP @ _zz(0.3),
        _SWAP @ _CZ,
        tq.canonical_gate(0.3, 0.3, 0.3),
        tq.canonical_gate(0.3, 0.3, -0.3),
        tq.canonical_gate(0.4, 0.2, 0.0),
        tq.canonical_gate(0.4, 0.0, 0.0),
        tq.canonical_gate(np.pi / 4, 0.2, -0.1),
        tq.canonical_gate(0.5, 0.25, 0.25),
        tq.canonical_gate(0.5, 0.25, -0.25),
        tq.canonical_gate(1e-10, 0.0, 0.0),
        np.diag([1, 1, 1, np.exp(1e-6j)]),
        np.diag([1, 1, 1, np.exp(0.7j)]),
    ]
    out = list(cores)
    out += [_local(rng) @ c @ _local(rng) for c in cores]
    out += [_haar(rng, 4) for _ in range(12)]
    return [np.ascontiguousarray(u, dtype=complex) for u in out]


def _same(a: object, b: object) -> bool:
    return repr(a) == repr(b)


def _entanglers(ops: list[tq.SynthOp]) -> int:
    return sum(1 for _, q, _ in ops if len(q) == 2)


def _check_ops(
    u: np.ndarray, ops: list[tq.SynthOp], basis: TwoQubitBasis, tol: float
) -> None:
    allowed = {basis.entangler, "rz", "x", "h" if basis.euler == "zxz" else "sx"}
    assert {name for name, _, _ in ops} <= allowed
    assert tq._distance_up_to_phase(u, tq._ops_unitary(ops, basis)) <= tol


@pytest.fixture
def flags(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def set_flags(swap_face: bool, mirrored: bool = True) -> None:
        monkeypatch.setattr(tq, "_SWAP_FACE_TEMPLATE", swap_face)
        monkeypatch.setattr(tq, "_MIRRORED_STABILIZERS", mirrored)

    return set_flags


@pytest.mark.parametrize("basis", BASES, ids=str)
@pytest.mark.parametrize("swap_face", [False, True])
def test_exact_synthesis_is_equivalent_to_python(
    basis: TwoQubitBasis, swap_face: bool, flags
) -> None:  # noqa: ANN001
    # 不逐位比较：Python 参考实现用 numpy 的 LAPACK 与平台 libm（见模块说明）
    flags(swap_face)
    assert NATIVE is not None
    for u in _unitaries(11):
        ref = tq._synthesize_two_qubit_ops(u, 1e-9, 1e-9, basis)
        ops, uses = NATIVE.synthesize(u, 1e-9, 1e-9, basis)
        _check_ops(u, ops, basis, 1e-9)
        assert _entanglers(ops) == _entanglers(ref)
        assert len(ops) <= len(ref) + 2
        kak = NATIVE.kak4(u)
        assert kak is not None
        assert uses == tq._uses_swap_face_template(kak[2], 1e-9)


@pytest.mark.parametrize("mirrored", [False, True])
def test_stabilizer_switch_and_exact_euler(mirrored: bool, flags) -> None:  # noqa: ANN001
    flags(False, mirrored)
    assert NATIVE is not None
    for u in _unitaries(5)[:20]:
        for atol, verify in ((1e-9, 1e-9), (1e-12, 1e-9), (1e-9, 1e-14)):
            try:
                ops, _ = NATIVE.synthesize(u, atol, verify, BASES[0])
            except RuntimeError as exc:
                # 同 Python：没有模板在 verify 以内重现输入时报错（附最优误差）
                assert "failed to reproduce" in str(exc)
                continue
            _check_ops(u, ops, BASES[0], verify)


@pytest.mark.parametrize("basis", BASES, ids=str)
def test_swap_face_block_is_equivalent_to_python(basis: TwoQubitBasis) -> None:
    assert NATIVE is not None
    rng = np.random.default_rng(3)
    for theta in (0.3, -0.7, 1e-3, np.pi / 2, 2.0):
        for u in (
            _SWAP @ _zz(theta),
            _local(rng) @ _SWAP @ _zz(theta) @ _local(rng),
            _CZ,
        ):
            u = np.ascontiguousarray(u)
            ref = tq._synthesize_swap_face_block(u, basis, 1e-9)
            got = NATIVE.swap_face(u, basis, 1e-9)
            assert got is not None
            if ref is None:
                assert got is False
                continue
            assert isinstance(got, list)
            _check_ops(u, got, basis, 1e-9)
            assert _entanglers(got) == _entanglers(ref)


def test_num_cz_and_approximation_agree_with_python() -> None:
    assert NATIVE is not None
    for u in _unitaries(17):
        kak = tq._kak4(u)
        for atol in (1e-9, 1e-6):
            assert NATIVE.num_cz(u, atol) == tq._count_from_coords(kak[2], atol)
        for budget, max_ent in ((1e-12, 2), (1e-6, 0), (0.0, 2)):
            ref = tq._approximate_two_qubit_ops(u, budget, 1e-9, BASES[0], max_ent)
            got = NATIVE.approximate(u, budget, 1e-9, 1e-9, BASES[0], max_ent, True)
            if ref is not None:
                assert got is None  # 存在近似时由 Python 处理
            elif got is not None:
                assert got[0] is not None
                _check_ops(u, got[0], BASES[0], 1e-9)


def _can(v: np.ndarray) -> np.ndarray:
    return tq.canonical_gate(float(v[0]), float(v[1]), float(v[2]))


def test_kak_factors_and_helpers() -> None:
    assert NATIVE is not None
    synth = NATIVE._synth
    rng = np.random.default_rng(23)
    for u in _unitaries(29):
        ref = tq._kak4(u)
        phase, k1, v, k2 = synth.kak4(u)
        v = np.array(v)
        # Weyl 坐标唯一；局部因子与相位可以不同，但必须重建输入
        assert np.max(np.abs(v - ref[2])) < 1e-12
        rebuilt = np.exp(1j * phase) * (k1 @ _can(v) @ k2)
        assert np.max(np.abs(rebuilt - u)) < 1e-12
        for k in (k1, k2):
            fa, fc = synth.factor_local(k)
            assert np.max(np.abs(np.kron(fa, fc) - k)) < 1e-12
            assert abs(np.linalg.det(fa) - 1) < 1e-12
    paulis = (tq._X, tq._Y, tq._Z)
    for _ in range(50):
        a, b = _haar(rng, 2), _haar(rng, 2)
        assert np.max(np.abs(synth.kron2(a, b) - tq._kron2(a, b))) < 1e-15
        phi = float(rng.normal())
        for p in paulis:
            assert np.max(np.abs(synth.axis_rot(p, phi) - tq._axis_rot(p, phi))) < 1e-15
            assert abs(synth.axis_angle(a, p) - tq._axis_angle(a, p)) < 1e-12
        for code, name in ((0, "zxz"), (1, "zsx")):
            for atol in (1e-9, 1e-12):
                got = synth.euler_ops(a, 1, code, atol)
                ref_ops = tq.euler_ops(a, 1, basis=name, atol=atol)
                assert [g[:2] for g in got] == [r[:2] for r in ref_ops]
                for g, r in zip(got, ref_ops, strict=True):
                    assert np.allclose(g[2], r[2], atol=1e-12)


def test_registered_entangler(monkeypatch: pytest.MonkeyPatch) -> None:
    assert NATIVE is not None
    ecr = np.array(
        [[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]]
    ) / np.sqrt(2)
    monkeypatch.setitem(tq._ENTANGLERS, "ecr", tq.Entangler.from_matrix("ecr", ecr))
    basis = TwoQubitBasis("ecr", "zsx")
    for u in _unitaries(31)[15:25]:
        ref = tq._synthesize_two_qubit_ops(u, 1e-9, 1e-9, basis)
        ops, _ = NATIVE.synthesize(u, 1e-9, 1e-9, basis)
        _check_ops(u, ops, basis, 1e-9)
        assert _entanglers(ops) == _entanglers(ref)


def test_input_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    assert NATIVE is not None
    u = _unitaries(2)[20]
    ops = NATIVE.synthesize(u, 1e-9, 1e-9, BASES[0])
    # 任意布局 / 类型的 4×4 输入都转换成 complex128 后原生合成（结果相同）
    assert _same(NATIVE.synthesize(np.asfortranarray(u), 1e-9, 1e-9, BASES[0]), ops)
    assert _same(NATIVE.synthesize(u.tolist(), 1e-9, 1e-9, BASES[0]), ops)
    assert NATIVE.synthesize(np.eye(3), 1e-9, 1e-9, BASES[0]) is None
    bad = u.copy()
    bad[0, 0] += 0.1
    with pytest.raises(ValueError, match="not unitary"):
        NATIVE.synthesize(bad, 1e-9, 1e-9, BASES[0])
    with pytest.raises(ValueError, match="not unitary"):
        NATIVE.num_cz(bad, 1e-9)
    with pytest.raises(ValueError, match="not unitary"):
        tq.synthesize_two_qubit_ops(bad)
    nan = u.copy()
    nan[1, 1] = np.nan
    with pytest.raises(ValueError, match="not unitary"):
        NATIVE.synthesize(nan, 1e-9, 1e-9, BASES[0])
    # 测试替换了 Python 实现时不走原生路径
    monkeypatch.setattr(tq, "_try_synthesize", lambda *a, **k: ([], 0.0))
    assert NATIVE.synthesize(u, 1e-9, 1e-9, BASES[0]) is None


def test_public_api_equivalent_with_either_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached public entry points agree (counts, validity) with either backend."""

    def run() -> list[list[tq.SynthOp] | int]:
        for cache in (
            tq._SYNTH_2Q_CACHE,
            tq._APPROX_2Q_CACHE,
            tq._SWAP_FACE_CACHE,
            tq._NUM_CZ_CACHE,
        ):
            cache.clear()
        out: list[list[tq.SynthOp] | int] = []
        for u in _unitaries(41)[::3]:
            out.append(tq.synthesize_two_qubit_ops(u, basis=BASES[1]))
            out.append(tq.synthesize_two_qubit_approx(u, 1e-12)[0])
            sf = tq.synthesize_swap_face_block(u, BASES[2])
            out.append([] if sf is None else sf)
            out.append(tq.num_cz_required(u))
            with tq.swap_face_templates():
                out.append(tq.synthesize_two_qubit_ops(u, basis=BASES[3]))
        return out

    native = run()
    monkeypatch.setattr(tq, "_native_kak", lambda: None)
    python = run()
    assert len(native) == len(python)
    for a, b in zip(native, python, strict=True):
        if isinstance(a, int):
            assert a == b
        else:
            assert isinstance(b, list)
            assert _entanglers(a) == _entanglers(b)
            assert (len(a) == 0) == (len(b) == 0)


def test_backend_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "python")
    assert _native_kak.load(tq) is None
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "bogus")
    with pytest.raises(ValueError, match=_native_kak.BACKEND_ENV):
        _native_kak.load(tq)
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "native")
    assert _native_kak.load(tq) is not None
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "auto")
    assert _native_kak.load(tq) is not None


# ---------------------------------------------------------------------------
# Platform independence
# ---------------------------------------------------------------------------


def _golden_inputs() -> list[np.ndarray]:
    """Platform-independent unitaries (PCG64 bits, deterministic QR and products)."""
    from quspect._native_loader import load_core

    core = load_core()
    rng = np.random.Generator(np.random.PCG64(424242))

    def unitary(n: int) -> np.ndarray:
        raw = rng.random((n, n)) * 2.0 - 1.0 + 1j * (rng.random((n, n)) * 2.0 - 1.0)
        return np.asarray(core._detmath_qr(np.ascontiguousarray(raw))[0])

    def dressed(m: np.ndarray) -> np.ndarray:
        assert NATIVE is not None
        left = NATIVE._synth.kron2(unitary(2), unitary(2))
        right = NATIVE._synth.kron2(unitary(2), unitary(2))
        return np.asarray(core._detmath_matmul(core._detmath_matmul(left, m), right))

    eye = np.eye(4, dtype=complex)
    cores = [eye, _CZ, _SWAP, np.ascontiguousarray(eye[[0, 1, 3, 2]])]
    out = [*cores, *(dressed(m) for m in cores)]
    out += [unitary(4) for _ in range(16)]
    return [np.ascontiguousarray(m) for m in out]


def _digest(records: list[str]) -> str:
    return hashlib.sha256("\n".join(records).encode()).hexdigest()[:24]


def test_golden_digests_are_platform_independent(flags) -> None:  # noqa: ANN001
    """Digests of the native outputs (``float.hex`` of every parameter).

    The inputs and the synthesis are platform independent, so these digests
    must be the same on every platform and build.  If the synthesis is changed
    on purpose, regenerate them (and check that every platform agrees, e.g.
    with ``tools/det_fingerprint.py``).
    """
    assert NATIVE is not None
    flags(False)
    mats = _golden_inputs()
    kak = []
    for u in mats:
        phase, k1, v, k2 = NATIVE._synth.kak4(u)
        kak.append(
            float(phase).hex()
            + ",".join(float(x).hex() for x in v)
            + np.asarray(k1).tobytes().hex()
            + np.asarray(k2).tobytes().hex()
        )
    ops = []
    for basis in BASES:
        for u in mats:
            seq, _ = NATIVE.synthesize(u, 1e-9, 1e-9, basis)
            ops.append(
                ";".join(
                    f"{n}{q}" + "".join(float(p).hex() for p in ps) for n, q, ps in seq
                )
            )
    scalars = [
        float(x).hex()
        for name in ("sin", "cos", "atan2", "hypot", "exp", "log")
        for x in _golden_scalars(name)
    ]
    assert {
        "kak": _digest(kak),
        "ops": _digest(ops),
        "scalar": _digest(scalars),
    } == GOLDEN


def _golden_scalars(name: str) -> np.ndarray:
    from quspect._native_loader import load_core

    core = load_core()
    x = np.linspace(-20.0, 20.0, 997)
    if name in ("atan2", "hypot"):
        return np.asarray(core._detmath_scalar(name, x, x[::-1] * 0.37))
    if name == "log":
        x = np.abs(x) + math.ldexp(1.0, -30)
    return np.asarray(core._detmath_scalar(name, x))


GOLDEN = {
    "kak": "a74138ec6fbefc98f81aa415",
    "ops": "b0dd74e555cf09ce9103fbe2",
    "scalar": "84983e7a6e944230519ade02",
}


# ---------------------------------------------------------------------------
# Single-qubit synthesis (quspect.gates.synthesis)
# ---------------------------------------------------------------------------

ONE_QUBIT_BASES = [
    frozenset({"cz", "rz", "sx", "x"}),
    frozenset({"cz", "h", "rz", "x"}),
    frozenset({"h", "rz"}),
    frozenset({"u3", "cx"}),
    frozenset({"rx", "rz"}),
    frozenset({"ry", "rz"}),
    frozenset({"h", "t", "tdg", "s", "sdg", "z", "x"}),
    frozenset({"h", "s", "t"}),
    frozenset({"sx", "sxdg", "rz", "x"}),
    frozenset({"p", "h"}),
    frozenset({"u1", "sx"}),
]


def _one_qubit_matrices(seed: int) -> list[np.ndarray]:
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.parameterized import RX, RZ, U3
    from quspect.gates.standard import ALL_STANDARD

    rng = np.random.default_rng(seed)
    out = [_haar(rng, 2) for _ in range(40)]
    out += [gate_matrix(g) for g in ALL_STANDARD if g.num_qubits == 1]
    h = gate_matrix(next(g for g in ALL_STANDARD if g.name == "h"))
    for k in range(-8, 9):
        angle = k * np.pi / 4
        out.append(gate_matrix(RZ.with_params(angle)))
        out.append(h @ gate_matrix(RZ.with_params(angle)) @ h)
        out.append(gate_matrix(RZ.with_params(angle + 1e-10)))
        out.append(gate_matrix(RX.with_params(angle)))
        out.append(gate_matrix(U3.with_params(np.pi / 2, 0.3 * k, -0.2 * k)))
        out.append(gate_matrix(U3.with_params(np.pi, 0.3 * k, 0.1)))
        out.append(gate_matrix(U3.with_params(1e-11, 0.3 * k, 0.1)))
    return [np.ascontiguousarray(m, dtype=complex) for m in out]


@needs_numpy
@pytest.mark.parametrize("basis", ONE_QUBIT_BASES, ids=lambda b: "-".join(sorted(b)))
def test_one_qubit_synthesis_matches_python(basis: frozenset[str]) -> None:
    from quspect.gates import synthesis as gs

    assert NATIVE is not None
    em = gs._EMITTERS.get(basis) or gs._EMITTERS.setdefault(basis, gs._Emitter(basis))
    for m in _one_qubit_matrices(7):
        done, got = NATIVE.one_qubit(m, em)
        assert done
        assert _same(got, gs._synthesize_one_qubit_reference(m, em))
    # 不能复刻的输入交回 Python
    bad = np.array([[np.nan, 0], [0, 1]], dtype=complex)
    assert NATIVE.one_qubit(bad, em) == (False, None)


@needs_numpy
def test_one_qubit_zyz_angles_are_bitwise() -> None:
    from quspect.dag.passes.single_qubit_fusion import _decompose_u3

    assert NATIVE is not None
    for m in _one_qubit_matrices(13):
        ref = _decompose_u3(m)
        got = NATIVE._synth.zyz_angles(m)
        assert [x.hex() for x in got] == [float(x).hex() for x in ref]


@needs_numpy
def test_one_qubit_public_api_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from quspect.gates import synthesis as gs

    def run() -> list[object]:
        gs._SYNTH_1Q_CACHE.clear()
        return [
            gs.synthesize_one_qubit(m, b)
            for m in _one_qubit_matrices(19)[::4]
            for b in ONE_QUBIT_BASES
        ]

    native = run()
    monkeypatch.setattr(gs, "_native_backend", lambda: None)
    python = run()
    assert repr(native) == repr(python)


# ---------------------------------------------------------------------------
# Block products and verified Euler runs (consolidation, SWAP-face, mapping)
# ---------------------------------------------------------------------------


@needs_numpy
def test_block_and_run_products_are_bitwise() -> None:
    from quspect.transpiler._mapping import _SWAP_4

    assert NATIVE is not None
    rng = np.random.default_rng(37)
    rev = [0, 2, 1, 3]
    eye2 = np.eye(2, dtype=complex)
    specials = [eye2, np.array([[0, 1], [1, 0]], dtype=complex), tq._H, tq._Z]
    for _ in range(200):
        mats: list[np.ndarray] = []
        slots: list[int] = []
        for _ in range(int(rng.integers(1, 10))):
            slot = int(rng.integers(0, 5))
            if slot in (1, 2):
                pick = int(rng.integers(0, 6))
                mats.append(specials[pick].copy() if pick < 4 else _haar(rng, 2))
            else:
                mats.append(_CZ.copy() if rng.random() < 0.2 else _haar(rng, 4))
            slots.append(slot)
        u = np.eye(4, dtype=complex)
        for m, slot in zip(mats, slots, strict=True):
            if slot == 0:
                full = m
            elif slot == 1:
                full = np.kron(m, eye2)
            elif slot == 2:
                full = np.kron(eye2, m)
            elif slot == 3:
                full = m[np.ix_(rev, rev)]
            else:
                full = _SWAP_4 @ m @ _SWAP_4
            u = full @ u
        got = NATIVE.block_unitary(mats, slots, _SWAP_4)
        assert got is not None and got.tobytes() == u.tobytes()
        run = [m for m in mats if m.shape == (2, 2)]
        v = np.eye(2, dtype=complex)
        for m in run:
            v = m @ v
        got_run = NATIVE.run_unitary(run)
        assert got_run is not None and got_run.tobytes() == v.tobytes()
    # 非 C 连续的矩阵交回 Python（numpy 的 BLAS 调用取决于布局）
    assert (
        NATIVE.block_unitary([np.asfortranarray(_haar(rng, 4))], [0], _SWAP_4) is None
    )
    m4 = _haar(rng, 4)
    assert NATIVE.block_unitary([m4.astype(np.complex64)], [0], _SWAP_4) is None
    assert NATIVE.block_unitary([m4[:, ::-1]], [0], _SWAP_4) is None
    assert NATIVE.run_unitary([_haar(rng, 2).astype(np.complex64)]) is None
    assert NATIVE.run_unitary([_haar(rng, 2).T]) is None
    assert NATIVE.run_unitary([np.zeros(4, dtype=complex)]) is None
    # 原生的布局判断与 Python 端的判断结果相同
    synth = NATIVE._synth
    two = _haar(rng, 2)
    assert synth.run_unitary_strict([two]).tobytes() == (
        synth.run_unitary([two]).tobytes()
    )
    assert synth.block_unitary_strict([m4.T], [0], _SWAP_4) is None


@needs_numpy
def test_verified_euler_matches_consolidation() -> None:
    from quspect.transpiler import _consolidate

    assert NATIVE is not None
    rng = np.random.default_rng(41)
    mats = [_haar(rng, 2) for _ in range(30)]
    mats += [np.diag([1, np.exp(1j * t)]) for t in (0.0, 1e-10, 1e-13, 0.3, np.pi / 2)]
    mats += [tq._H @ np.diag([1, np.exp(1e-10j)]), tq._SX, tq._X]
    for m in mats:
        m = np.ascontiguousarray(m, dtype=complex)
        for atol in (1e-9, 1e-12):
            for euler in ("zxz", "zsx"):
                got = NATIVE.verified_euler(m, atol, euler)
                ref = _consolidate._verified_euler_ops_uncached(m, atol, euler)
                assert got is not None
                assert _same(None if got is False else got, ref)
