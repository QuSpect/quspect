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
            assert got is not None
            if ref is None:
                assert not got[0]
                assert got[1] is not None
                _check_ops(u, got[1], BASES[0], 1e-9)
                continue
            # 同一个近似（纠缠门数相同、误差在预算内且与 Python 的数值一致到舍入）
            approx, ops, (infidelity, distance) = got
            assert approx
            assert ops is not None
            count = sum(1 for _, q, _ in ops if len(q) == 2)
            assert count == sum(1 for _, q, _ in ref[0] if len(q) == 2)
            assert infidelity <= budget
            assert infidelity == pytest.approx(ref[1].infidelity, rel=1e-6, abs=1e-18)
            assert distance == pytest.approx(ref[1].distance, rel=1e-6, abs=1e-12)
            check = tq.approximation_error(u, tq._ops_unitary(ops, BASES[0]))
            assert check.infidelity <= budget * (1 + 1e-6)


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


def test_golden_digests_of_single_qubit_numerics() -> None:
    """Digests of gate matrices, single-qubit synthesis, ZYZ angles, block
    products and verified Euler runs on platform-independent inputs.

    Like :func:`test_golden_digests_are_platform_independent`, these must be
    the same on every platform and build.
    """
    from quspect._native_loader import load_core
    from quspect.gates import synthesis as gs

    assert NATIVE is not None
    core = load_core()
    rng = np.random.Generator(np.random.PCG64(77))

    def unitary(n: int) -> np.ndarray:
        raw = rng.random((n, n)) * 2.0 - 1.0 + 1j * (rng.random((n, n)) * 2.0 - 1.0)
        return np.ascontiguousarray(core._detmath_qr(np.ascontiguousarray(raw))[0])

    def hexes(arr: np.ndarray) -> str:
        return np.ascontiguousarray(arr).tobytes().hex()

    angles = (rng.random(40) * 20.0 - 10.0).tolist() + [0.0, -0.0, 1e-10, math.pi]
    gates = []
    for name, n in (("rx", 1), ("ry", 1), ("rz", 1), ("p", 1), ("u2", 2), ("u3", 3)):
        for k in range(len(angles) - 2):
            gates.append(hexes(core._gate_matrix(name, angles[k : k + n])))
    for name, n in (("crx", 1), ("cry", 1), ("crz", 1), ("cp", 1), ("cu3", 3)):
        for k in range(0, len(angles) - 2, 3):
            gates.append(hexes(core._gate_matrix(name, angles[k : k + n])))
    mats = [unitary(2) for _ in range(60)]
    mats += [np.asarray(core._gate_matrix(g, [])) for g in ("h", "x", "sx", "t", "s")]
    one = []
    for basis in ONE_QUBIT_BASES:
        em = gs._EMITTERS.get(basis) or gs._EMITTERS.setdefault(
            basis, gs._Emitter(basis)
        )
        for m in mats:
            done, got = NATIVE.one_qubit(m, em)
            assert done
            one.append(
                "-"
                if got is None
                else ";".join(
                    g.name + "".join(float(p).hex() for p in g.params) for g in got
                )
            )
    zyz = [",".join(x.hex() for x in NATIVE.zyz_angles(m)) for m in mats]
    euler = []
    for m in mats:
        for euler_basis in ("zxz", "zsx"):
            ops = NATIVE.verified_euler(m, 1e-9, euler_basis)
            euler.append(repr(ops))
    blocks = []
    for k in range(30):
        members = [unitary(4), mats[k], mats[k + 1], unitary(4), unitary(4)]
        blocks.append(hexes(NATIVE.block_unitary(members, [0, 1, 2, 3, 4])))
        blocks.append(hexes(NATIVE.run_unitary(mats[k : k + 5])))
    assert {
        "gates": _digest(gates),
        "one_qubit": _digest(one),
        "zyz": _digest(zyz),
        "euler": _digest(euler),
        "blocks": _digest(blocks),
    } == GOLDEN_1Q


GOLDEN_1Q = {
    "gates": "91c229888f5abb35baac6015",
    "one_qubit": "04ba2a951420a7602ec42a4a",
    "zyz": "a1e44e28d40943bc0fe28bdb",
    "euler": "0988fabd9b14ae7ad47d6e72",
    "blocks": "a3350d350925bd4922a072ea",
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


def _gates_matrix(gates: list[object]) -> np.ndarray:
    from quspect.gates._matrix_utils import gate_matrix

    m = np.eye(2, dtype=complex)
    for g in gates:
        m = gate_matrix(g) @ m  # type: ignore[arg-type]
    return m


def _close_gates(a: list[object] | None, b: list[object] | None) -> None:
    """Same gate names; parameters equal to rounding (``mod 2π`` not needed)."""
    assert (a is None) == (b is None)
    if a is None or b is None:
        return
    assert [g.name for g in a] == [g.name for g in b]  # type: ignore[attr-defined]
    for x, y in zip(a, b, strict=True):
        assert x.params == pytest.approx(y.params, abs=1e-12)  # type: ignore[attr-defined]


# The native single-qubit synthesis, ZYZ angles, block products and verified
# Euler runs run on the deterministic numerics layer; the Python references
# use numpy / CPython arithmetic and the platform libm.  They are compared
# for equivalence (same gate sequences, parameters and products equal to
# rounding), not bit for bit; platform independence of the native results is
# checked by the golden digests above.


@pytest.mark.parametrize("basis", ONE_QUBIT_BASES, ids=lambda b: "-".join(sorted(b)))
def test_one_qubit_synthesis_is_equivalent_to_python(basis: frozenset[str]) -> None:
    from quspect.gates import synthesis as gs

    assert NATIVE is not None
    em = gs._EMITTERS.get(basis) or gs._EMITTERS.setdefault(basis, gs._Emitter(basis))
    for m in _one_qubit_matrices(7):
        done, got = NATIVE.one_qubit(m, em)
        assert done
        _close_gates(got, gs._synthesize_one_qubit_reference(m, em))
        if got is not None:
            assert gs._equal_up_to_phase(_gates_matrix(got), m)
    # 含非有限值的输入交回 Python
    bad = np.array([[np.nan, 0], [0, 1]], dtype=complex)
    assert NATIVE.one_qubit(bad, em) == (False, None)


def test_one_qubit_zyz_angles() -> None:
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.parameterized import U3

    assert NATIVE is not None
    for m in _one_qubit_matrices(13):
        theta, phi, lam = NATIVE._synth.zyz_angles(m)
        assert 0.0 <= phi <= 2 * math.pi and 0.0 <= lam <= 2 * math.pi
        rebuilt = gate_matrix(U3.with_params(theta, phi, lam))
        assert (
            tq._distance_up_to_phase(np.kron(m, np.eye(2)), np.kron(rebuilt, np.eye(2)))
            < 1e-9
        )  # θ < 1e-10 is rounded to 0 by design


def test_one_qubit_public_api_equivalent(monkeypatch: pytest.MonkeyPatch) -> None:
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
    gs._SYNTH_1Q_CACHE.clear()
    for a, b in zip(native, python, strict=True):
        _close_gates(a, b)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Block products and verified Euler runs (consolidation, SWAP-face, mapping)
# ---------------------------------------------------------------------------


def test_block_and_run_products() -> None:
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
        got = NATIVE.block_unitary(mats, slots)
        assert np.max(np.abs(got - u)) < 1e-14
        run = [m for m in mats if m.shape == (2, 2)]
        v = np.eye(2, dtype=complex)
        for m in run:
            v = m @ v
        assert np.max(np.abs(NATIVE.run_unitary(run) - v)) < 1e-14
    # 布局与 dtype 不影响结果（精确转换成 C 连续的 complex128）
    m4 = _haar(rng, 4)
    ref = NATIVE.block_unitary([m4], [0])
    assert NATIVE.block_unitary([np.asfortranarray(m4)], [0]).tobytes() == ref.tobytes()
    assert NATIVE.block_unitary([m4[:, ::-1][:, ::-1]], [0]).tobytes() == ref.tobytes()
    two = _haar(rng, 2)
    assert NATIVE.run_unitary([two.T.copy().T]).tobytes() == (
        NATIVE.run_unitary([two]).tobytes()
    )


def test_verified_euler_is_equivalent_to_consolidation() -> None:
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
                got = None if got is False else got
                assert (got is None) == (ref is None)
                if got is None or ref is None:
                    continue
                assert [(n, q) for n, q, _ in got] == [(n, q) for n, q, _ in ref]
                for (_, _, pa), (_, _, pb) in zip(got, ref, strict=True):
                    assert pa == pytest.approx(pb, abs=1e-12)


def test_swap_residual_and_gate_matrices() -> None:
    from quspect.gates import _matrix_utils as mu
    from quspect.transpiler._mapping import _PRODUCT_TOL

    assert NATIVE is not None
    rng = np.random.default_rng(43)
    for _ in range(20):
        a, b = _haar(rng, 2), _haar(rng, 2)
        u = (_SWAP @ np.kron(a, b)) * np.exp(0.3j)
        res = NATIVE.swap_residual(np.ascontiguousarray(u), _PRODUCT_TOL)
        assert res is not None
        ra, rb = res
        assert abs(np.linalg.det(ra) - 1) < 1e-12 and abs(np.linalg.det(rb) - 1) < 1e-12
        assert tq._distance_up_to_phase(u, _SWAP @ np.kron(ra, rb)) < 1e-12
    assert (
        NATIVE.swap_residual(np.ascontiguousarray(_haar(rng, 4)), _PRODUCT_TOL) is None
    )
    # 门矩阵：原生（确定性）与 Python 公式在舍入误差内相同
    from quspect.gates.parameterized import CP, CRX, CU3, RX, RY, RZ, U1, U2, U3, P

    for t in (0.0, 0.3, -2.1, 7.5):
        for g in (RX, RY, RZ, P, U1, CRX, CP):
            m = mu.gate_matrix(g.with_params(t))
            ref = getattr(mu, f"_{'phase' if g is P else g.name}_matrix")(t)
            assert np.max(np.abs(m - ref)) < 1e-15
        for g, n in ((U2, 2), (U3, 3), (CU3, 3)):
            ps = [t, 0.7, -1.3][:n]
            m = mu.gate_matrix(g.with_params(*ps))
            ref = getattr(mu, f"_{g.name}_matrix")(*ps)
            assert np.max(np.abs(m - ref)) < 1e-15


def test_public_kak_api_is_deterministic() -> None:
    """``kak_decompose`` / ``weyl_coordinates`` run on the deterministic layer.

    Their results are the native KAK bit for bit (hence platform independent,
    see :func:`test_golden_digests_are_platform_independent`) and reconstruct
    the input.
    """
    assert NATIVE is not None
    for u in _golden_inputs():
        phase, k1, v, k2 = NATIVE.kak4(u)
        kak = tq.kak_decompose(u)
        assert [kak.a, kak.b, kak.c] == [float(x) for x in v]
        assert kak.global_phase == phase
        assert tq.weyl_coordinates(u) == (kak.a, kak.b, kak.c)
        k1l, k1r = NATIVE._synth.factor_local(k1)
        assert np.asarray(kak.k1l).tobytes() == np.asarray(k1l).tobytes()
        assert np.asarray(kak.k1r).tobytes() == np.asarray(k1r).tobytes()
        rebuilt = (
            np.exp(1j * kak.global_phase)
            * np.kron(kak.k1l, kak.k1r)
            @ tq.canonical_gate(kak.a, kak.b, kak.c)
            @ np.kron(kak.k2l, kak.k2r)
        )
        assert np.max(np.abs(rebuilt - u)) < 1e-12
