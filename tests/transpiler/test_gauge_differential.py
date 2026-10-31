"""Differential tests: the round-13 single-qubit minimization equals round 12.

Round 13 replaced the per-wire Python beam search of
:func:`~quspect.transpiler._gauge.minimize_single_qubit_gates` by a layered
search over all wires (batched numpy products + C++ kernels) and sped up
the KAK stabilizer choice; outputs must be bit-identical to the frozen
round-12 code (:mod:`tests.transpiler._legacy_gauge_r12`).  Covered here:

* the native kernels against their Python references (run lengths,
  leading angles, rotations) and the batched products against single ``@``;
* ``minimize_single_qubit_gates`` (native and Python backends) against the
  legacy function on random circuits in CZ / CX / IBM gates;
* whole :func:`~quspect.transpiler.transpile` runs against the round-12
  pipeline (:mod:`tests.transpiler._r12_reference`) on benchmark circuits;
  the full benchmark sweep is opt-in (``QUSPECT_DIFFERENTIAL_BENCHMARK=1``).
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.standard import SDG, SX, TDG, H, S, T, X, Y, Z
from quspect.transpiler import CompilationTarget, transpile
from quspect.transpiler import _gauge as gauge
from quspect.transpiler._gauge import (
    _TO_Z,
    _leading_angles,
    _LengthModel,
    _native_kernels,
    _rot,
    minimize_single_qubit_gates,
)
from tests.transpiler._legacy_gauge_r12 import legacy_minimize_single_qubit_gates
from tests.transpiler._r12_reference import (
    current_without_round_17,
    legacy_r12,
    transpile_signature,
)

BENCHMARK_ROOT = Path("/Users/quspect/xuxs/quspect_benchmark")

CZ_ONE = frozenset({"h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"})
IBM_ONE = frozenset({"rz", "sx", "x"})
BASES = {
    "cz": {"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"},
    "cx": {"cx", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"},
    "ibm": {"cx", "rz", "sx", "x"},
}
ONE_QUBIT_BASES = [
    CZ_ONE,
    IBM_ONE,
    frozenset({"rz", "sx", "sxdg", "x", "h"}),
    frozenset({"rz", "h"}),
]

native_only = pytest.mark.skipif(
    _native_kernels(_LengthModel(CZ_ONE)) is None,
    reason="native single-qubit minimization kernels not built",
)


def _signature(circuit: QuantumCircuit) -> list[tuple[object, ...]]:
    return [
        (
            str(i.type),
            None if i.gate is None else i.gate.name,
            tuple(i.qubits),
            () if i.gate is None else tuple(repr(p) for p in i.gate.params),
            tuple(i.clbits),
        )
        for i in circuit
    ]


def _test_matrices(seed: int, n: int) -> list[np.ndarray]:
    """Random unitaries, Clifford+T words and near-threshold rotations."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
    mats = [np.linalg.qr(x)[0] for x in raw]
    basic = [
        np.asarray(gate_matrix(g), dtype=complex)
        for g in (H, S, T, X, SX, Z, Y, SDG, TDG)
    ]
    special = [1e-9, 1.0000001e-9, 9.99e-10, 1e-12, math.pi / 2, math.pi, -math.pi]
    for _ in range(n):
        m = np.eye(2, dtype=complex)
        for _ in range(int(rng.integers(1, 6))):
            m = basic[int(rng.integers(len(basic)))] @ m
        if rng.random() < 0.6:
            axis = "zxy"[int(rng.integers(3))]
            phi = float(rng.choice([*special, float(rng.normal())]))
            m = _rot(axis, phi) @ m
        mats.append(m)
    return mats


# --- native kernels against their Python references ------------------------


@native_only
@pytest.mark.parametrize("basis", ONE_QUBIT_BASES, ids=lambda b: "-".join(sorted(b)))
def test_native_lengths_match_python(basis: frozenset[str]) -> None:
    length = _LengthModel(basis)
    native = _native_kernels(length)
    assert native is not None
    mats = _test_matrices(1, 3000)
    assert native.lengths(np.array(mats)) == [length._fast(m) for m in mats]


@native_only
def test_native_leading_angles_and_rotations_match_python() -> None:
    native = _native_kernels(_LengthModel(CZ_ONE))
    assert native is not None
    for m in _test_matrices(2, 1500):
        for axis in ("z", "x", "y"):
            ref = [float(x).hex() for x in _leading_angles(m, axis)]
            assert [x.hex() for x in native.leading(m, axis)] == ref
    rng = np.random.default_rng(3)
    phis = [0.0, -0.0, math.pi, -math.pi, 2 * math.pi, 1e-300, 1e-9]
    phis += [float(x) for x in rng.normal(size=500) * 4]
    for axis in ("z", "x", "y"):
        for phi in phis:
            assert native.rot(axis, phi).tobytes() == _rot(axis, phi).tobytes()
        stack = native.rots(axis, phis)
        for phi, r in zip(phis, stack, strict=True):
            assert r.tobytes() == _rot(axis, phi).tobytes()


def test_batched_products_equal_single_products() -> None:
    """The layouts the batched search relies on give the single ``@`` bits."""
    mats = _test_matrices(4, 400)
    rots = [_rot("zxy"[i % 3], 0.1 * i - 7.0) for i in range(len(mats))]
    m_stack, r_stack = np.array(mats), np.array(rots)
    prod = np.matmul(m_stack, r_stack)
    left = np.matmul(r_stack, m_stack[0])
    axes = np.arange(len(mats)) % 3
    frame = np.matmul(
        np.matmul(gauge._TO_Z_CONJ[axes].transpose(0, 2, 1), m_stack),
        gauge._TO_Z_STACK[axes],
    )
    for i, (m, r) in enumerate(zip(mats, rots, strict=True)):
        assert prod[i].tobytes() == (m @ r).tobytes()
        assert left[i].tobytes() == (r @ mats[0]).tobytes()
        v = _TO_Z["zxy"[axes[i]]]
        assert frame[i].tobytes() == (v.conj().T @ m @ v).tobytes()


# --- minimize_single_qubit_gates against round 12 --------------------------


def _random_circuit(seed: int, two_qubit: str, ibm: bool, n: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    angles = [math.pi / 2, math.pi, -math.pi / 2, math.pi / 4, 1e-10]
    for _ in range(12 * n):
        a, b = rng.sample(range(n), 2)
        r = rng.randrange(9)
        if r < 3:
            if two_qubit == "cx" or (two_qubit == "mixed" and rng.random() < 0.5):
                qc.cx(a, b)
            else:
                qc.cz(a, b)
        elif r < 5:
            qc.rz(rng.choice([rng.uniform(-4, 4), *angles]), a)
        elif r == 5:
            qc.sx(a) if ibm else qc.h(a)
        elif r == 6:
            qc.x(a)
        elif r == 7:
            if not ibm:
                rng.choice([qc.s, qc.t, qc.sdg, qc.h])(a)
        else:
            qc.barrier(a, b) if rng.random() < 0.1 else qc.rz(rng.uniform(-1, 1), b)
    return qc


@pytest.mark.parametrize("backend", ["native", "python"])
@pytest.mark.parametrize("two_qubit", ["cx", "cz", "mixed"])
@pytest.mark.parametrize("ibm", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_minimize_matches_round_12(
    backend: str, two_qubit: str, ibm: bool, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend == "native" and _native_kernels(_LengthModel(CZ_ONE)) is None:
        pytest.skip("native kernels not built")
    monkeypatch.setenv(gauge.GAUGE_BACKEND_ENV, backend)
    basis = IBM_ONE if ibm else CZ_ONE
    qc = _random_circuit(seed, two_qubit, ibm, 3 + seed % 4)
    expected = _signature(legacy_minimize_single_qubit_gates(qc, basis))
    assert _signature(minimize_single_qubit_gates(qc, basis)) == expected


def test_backend_env_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gauge.GAUGE_BACKEND_ENV, "fast")
    with pytest.raises(ValueError, match=gauge.GAUGE_BACKEND_ENV):
        minimize_single_qubit_gates(QuantumCircuit().h(0), CZ_ONE)


@native_only
def test_native_fallback_on_unreproducible_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kernel that cannot reproduce an input sends the call to Python."""
    calls: list[int] = []
    native = _native_kernels(_LengthModel(CZ_ONE))
    assert native is not None

    def refuse(self: object, *args: object) -> list[int]:
        calls.append(1)
        raise gauge._NativeFallbackError

    def no_search(*args: object) -> None:
        raise ValueError("not reproducible")

    monkeypatch.setattr(gauge._Native, "lengths", refuse)
    monkeypatch.setattr(native.core, "_GaugeBeamSearch", no_search)
    qc = _random_circuit(5, "cx", False, 4)
    expected = _signature(legacy_minimize_single_qubit_gates(qc, CZ_ONE))
    assert _signature(minimize_single_qubit_gates(qc, CZ_ONE)) == expected
    assert not calls  # the batched search failed first and Python took over


@native_only
@pytest.mark.parametrize("seed", range(6))
def test_speculative_sweep_fallbacks_match_round_12(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused commits re-evaluate sequentially; the result does not change."""
    native = _native_kernels(_LengthModel(CZ_ONE))
    assert native is not None
    base = native.core._GaugeBeamSearch
    rng = random.Random(seed)

    class Flaky(base):  # type: ignore[misc, valid-type]
        def commit_slots(self, slots: list[int]) -> bool:
            r = rng.random()
            if r < 0.3:
                return False  # dropped: the sequential evaluation recomputes
            ok = bool(super().commit_slots(slots))
            return ok and r < 0.6  # committed, then re-evaluated anyway

    monkeypatch.setattr(native.core, "_GaugeBeamSearch", Flaky)
    monkeypatch.setenv(gauge.GAUGE_BACKEND_ENV, "native")
    for ibm in (False, True):
        basis = IBM_ONE if ibm else CZ_ONE
        qc = _random_circuit(100 + seed, "cx", ibm, 5)
        expected = _signature(legacy_minimize_single_qubit_gates(qc, basis))
        assert _signature(minimize_single_qubit_gates(qc, basis)) == expected


# --- whole transpile runs against the round-12 pipeline --------------------


def _benchmark() -> object:
    if not (BENCHMARK_ROOT / "compilation").is_dir():
        pytest.skip("quspect_benchmark checkout not available")
    if str(BENCHMARK_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_ROOT))
    from compilation import circuits, topologies

    return circuits, topologies


def _check_transpile(qasm: str, kind: str, basis: str, level: int) -> None:
    _, topologies = _benchmark()  # type: ignore[misc]
    from quspect.ir import qasm2_loads

    circuit = qasm2_loads(qasm)
    topo = topologies.build(kind, circuit.num_qubits)
    target = CompilationTarget.from_coupling_map(
        topo.num_qubits, list(topo.edges), native_gates=set(BASES[basis])
    )
    with legacy_r12():
        old = transpile_signature(
            transpile(circuit, target, optimization_level=level, seed=0)
        )
    with current_without_round_17():
        res = transpile(circuit, target, optimization_level=level, seed=0)
    new = transpile_signature(res)
    assert new == old


SMOKE_CIRCUITS = [
    "mqt_grover_7",
    "qiskit_heisenberg_2d_trotter_9",
    "qiskit_qpe_random_u_8",
    "qasmbench_qpe_9",
]


@pytest.mark.parametrize("basis", ["cz", "cx", "ibm"])
@pytest.mark.parametrize("level", [2, 3])
def test_transpile_matches_round_12_on_standard_circuits(
    basis: str, level: int
) -> None:
    circuits, _ = _benchmark()  # type: ignore[misc]
    std = {c.name: c for c in circuits.load_qasm_dir()}
    names = [n for n in SMOKE_CIRCUITS if n in std] or sorted(std)[:4]
    for name in names:
        for kind in ("line", "heavy_hex"):
            _check_transpile(std[name].qasm, kind, basis, level)


@pytest.mark.parametrize("family", ["qft", "heisenberg", "random", "adder"])
def test_transpile_matches_round_12_on_synthetic_circuits(family: str) -> None:
    circuits, _ = _benchmark()  # type: ignore[misc]
    if family not in circuits.GENERATORS:
        pytest.skip(f"no generator {family}")
    for basis in ("cz", "ibm"):
        _check_transpile(circuits.generate(family, 10).qasm, "grid", basis, 2)


def _sweep_cases() -> list[tuple[str, str, str, str, int]]:
    if os.environ.get("QUSPECT_DIFFERENTIAL_BENCHMARK") != "1":
        return []
    circuits, _ = _benchmark()  # type: ignore[misc]
    cases = []
    for c in circuits.load_qasm_dir():
        for kind in ("line", "ring", "grid", "heavy_hex", "full"):
            for basis in BASES:
                for level in (2, 3):
                    cases.append((c.name, c.qasm, kind, basis, level))
    for fam in circuits.GENERATORS:
        for n in (10, 20, 50):
            c = circuits.generate(fam, n)
            for kind in ("grid", "heavy_hex"):
                for basis in BASES:
                    cases.append((c.name, c.qasm, kind, basis, 2))
    large = BENCHMARK_ROOT / "compilation" / "qasm_large"
    for c in circuits.load_qasm_dir(large):
        for kind in ("grid", "heavy_hex"):
            cases.append((c.name, c.qasm, kind, "cz", 2))
    return cases


@pytest.mark.skipif(
    os.environ.get("QUSPECT_DIFFERENTIAL_BENCHMARK") != "1",
    reason="full benchmark sweep: set QUSPECT_DIFFERENTIAL_BENCHMARK=1",
)
@pytest.mark.parametrize(
    ("name", "qasm", "kind", "basis", "level"),
    _sweep_cases(),
    ids=lambda v: v if isinstance(v, (str, int)) and len(str(v)) < 40 else "",
)
def test_transpile_matches_round_12_benchmark_sweep(
    name: str, qasm: str, kind: str, basis: str, level: int
) -> None:
    _check_transpile(qasm, kind, basis, level)
