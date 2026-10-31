"""Native KAK synthesis backend: bit-identical to the Python reference.

The native backend (:mod:`quspect.synthesis._native_kak`) replaces
``_synthesize_two_qubit_ops``, ``_synthesize_swap_face_block``, the KAK
count of :func:`num_cz_required` and the "no approximation" decision of
:func:`synthesize_two_qubit_approx`.  Every result it returns must equal the
Python reference exactly (gate names, qubits and the ``repr`` of every
parameter); inputs it cannot reproduce are handed back (``None``).
"""

from __future__ import annotations

import numpy as np
import pytest

import quspect.synthesis.two_qubit as tq
from quspect.synthesis import _native_kak
from quspect.synthesis.two_qubit import TwoQubitBasis

NATIVE = tq._native_kak()

pytestmark = pytest.mark.skipif(
    NATIVE is None, reason="native KAK backend not enabled on this platform"
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


@pytest.fixture
def flags(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def set_flags(swap_face: bool, mirrored: bool = True) -> None:
        monkeypatch.setattr(tq, "_SWAP_FACE_TEMPLATE", swap_face)
        monkeypatch.setattr(tq, "_MIRRORED_STABILIZERS", mirrored)

    return set_flags


@pytest.mark.parametrize("basis", BASES, ids=str)
@pytest.mark.parametrize("swap_face", [False, True])
def test_exact_synthesis_matches_python(
    basis: TwoQubitBasis, swap_face: bool, flags
) -> None:  # noqa: ANN001
    flags(swap_face)
    assert NATIVE is not None
    fallbacks = 0
    for u in _unitaries(11):
        ref = tq._synthesize_two_qubit_ops(u, 1e-9, 1e-9, basis)
        got = NATIVE.synthesize(u, 1e-9, 1e-9, basis)
        if got is None:
            fallbacks += 1
            continue
        assert _same(got[0], ref)
        assert got[1] == tq._uses_swap_face_template(tq._kak4(u)[2], 1e-9)
    assert fallbacks <= 2


@pytest.mark.parametrize("mirrored", [False, True])
def test_stabilizer_switch_and_exact_euler(mirrored: bool, flags) -> None:  # noqa: ANN001
    flags(False, mirrored)
    assert NATIVE is not None
    for u in _unitaries(5)[:20]:
        for atol, verify in ((1e-9, 1e-9), (1e-12, 1e-9), (1e-9, 0.0)):
            try:
                ref = tq._synthesize_two_qubit_ops(u, atol, verify, BASES[0])
            except RuntimeError:
                assert NATIVE.synthesize(u, atol, verify, BASES[0]) is None
                continue
            got = NATIVE.synthesize(u, atol, verify, BASES[0])
            if got is not None:
                assert _same(got[0], ref)


@pytest.mark.parametrize("basis", BASES, ids=str)
def test_swap_face_block_matches_python(basis: TwoQubitBasis) -> None:
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
            assert _same(None if got is False else got, ref)


def test_num_cz_and_approximation_match_python() -> None:
    assert NATIVE is not None
    for u in _unitaries(17):
        kak = tq._kak4(u)
        for atol in (1e-9, 1e-6):
            assert NATIVE.num_cz(u, atol) == tq._count_from_coords(kak[2], atol)
        for budget, max_ent in ((1e-12, 2), (1e-6, 0), (0.0, 2)):
            ref = tq._approximate_two_qubit_ops(u, budget, 1e-9, BASES[0], max_ent)
            got = NATIVE.approximate(u, budget, 1e-9, 1e-9, BASES[0], max_ent, True)
            if ref is not None:
                assert got is None  # 存在近似时交给 Python
            elif got is not None:
                assert got[0] is not None
                assert _same(
                    got[0], tq._synthesize_two_qubit_ops(u, 1e-9, 1e-9, BASES[0])
                )


def test_kak_factors_and_helpers_are_bitwise() -> None:
    assert NATIVE is not None
    synth = NATIVE._synth
    rng = np.random.default_rng(23)
    for u in _unitaries(29):
        ref = tq._kak4(u)
        got = synth.kak4(u)
        assert got is not None
        assert got[0].hex() == ref[0].hex()
        assert got[1].tobytes() == ref[1].tobytes()
        assert np.array(got[2]).tobytes() == ref[2].tobytes()
        assert got[3].tobytes() == ref[3].tobytes()
        for k in (ref[1], ref[3]):
            fa, fc = tq._factor_local(k)
            na, nc = synth.factor_local(k)
            assert na.tobytes() == fa.tobytes()
            assert nc.tobytes() == fc.tobytes()
    paulis = (tq._X, tq._Y, tq._Z)
    for _ in range(50):
        a, b = _haar(rng, 2), _haar(rng, 2)
        assert synth.kron2(a, b).tobytes() == tq._kron2(a, b).tobytes()
        phi = float(rng.normal())
        for p in paulis:
            assert synth.axis_rot(p, phi).tobytes() == tq._axis_rot(p, phi).tobytes()
            assert synth.axis_angle(a, p).hex() == tq._axis_angle(a, p).hex()
        for code, name in ((0, "zxz"), (1, "zsx")):
            for atol in (1e-9, 1e-12):
                assert _same(
                    synth.euler_ops(a, 1, code, atol),
                    tq.euler_ops(a, 1, basis=name, atol=atol),
                )


def test_registered_entangler(monkeypatch: pytest.MonkeyPatch) -> None:
    assert NATIVE is not None
    ecr = np.array(
        [[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]]
    ) / np.sqrt(2)
    monkeypatch.setitem(tq._ENTANGLERS, "ecr", tq.Entangler.from_matrix("ecr", ecr))
    basis = TwoQubitBasis("ecr", "zsx")
    for u in _unitaries(31)[15:25]:
        ref = tq._synthesize_two_qubit_ops(u, 1e-9, 1e-9, basis)
        got = NATIVE.synthesize(u, 1e-9, 1e-9, basis)
        assert got is not None
        assert _same(got[0], ref)


def test_inputs_handed_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    assert NATIVE is not None
    u = _unitaries(2)[20]
    assert NATIVE.synthesize(np.asfortranarray(u), 1e-9, 1e-9, BASES[0]) is None
    assert NATIVE.synthesize(u.tolist(), 1e-9, 1e-9, BASES[0]) is None
    bad = u.copy()
    bad[0, 0] += 0.1
    assert NATIVE.synthesize(bad, 1e-9, 1e-9, BASES[0]) is None
    assert NATIVE.num_cz(bad, 1e-9) is None
    with pytest.raises(ValueError, match="not unitary"):
        tq.synthesize_two_qubit_ops(bad)
    # 测试替换了 Python 实现时不走原生路径
    monkeypatch.setattr(tq, "_try_synthesize", lambda *a, **k: ([], 0.0))
    assert NATIVE.synthesize(u, 1e-9, 1e-9, BASES[0]) is None


def test_public_api_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached public entry points give the same results with either backend."""

    def run() -> list[object]:
        for cache in (
            tq._SYNTH_2Q_CACHE,
            tq._APPROX_2Q_CACHE,
            tq._SWAP_FACE_CACHE,
            tq._NUM_CZ_CACHE,
        ):
            cache.clear()
        out: list[object] = []
        for u in _unitaries(41)[::3]:
            out.append(tq.synthesize_two_qubit_ops(u, basis=BASES[1]))
            out.append(tq.synthesize_two_qubit_approx(u, 1e-12))
            out.append(tq.synthesize_swap_face_block(u, BASES[2]))
            out.append(tq.num_cz_required(u))
            with tq.swap_face_templates():
                out.append(tq.synthesize_two_qubit_ops(u, basis=BASES[3]))
        return out

    native = run()
    monkeypatch.setattr(tq, "_native_kak", lambda: None)
    python = run()
    assert repr(native) == repr(python)


def test_backend_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "python")
    assert _native_kak.load(tq) is None
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "bogus")
    with pytest.raises(ValueError, match=_native_kak.BACKEND_ENV):
        _native_kak.load(tq)
    monkeypatch.setenv(_native_kak.BACKEND_ENV, "native")
    assert _native_kak.load(tq) is not None


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


def test_one_qubit_zyz_angles_are_bitwise() -> None:
    from quspect.dag.passes.single_qubit_fusion import _decompose_u3

    assert NATIVE is not None
    for m in _one_qubit_matrices(13):
        ref = _decompose_u3(m)
        got = NATIVE._synth.zyz_angles(m)
        assert [x.hex() for x in got] == [float(x).hex() for x in ref]


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
