"""The native driver of the single-qubit minimization equals the Python body.

:func:`~quspect.transpiler._gauge._minimize_driver` runs the whole
minimization (runs and anchors, beam search, CX orientation sweep, emission)
in C++; with ``QUSPECT_GAUGE_DRIVER=python`` the Python body drives the same
native kernels.  Both must return identical circuits (instruction by
instruction, including reused instruction objects and their order), with
any number of threads for the initial search.
"""

from __future__ import annotations

import math
import random

import pytest

from quspect import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter
from quspect.gates.gate import Gate
from quspect.gates.standard import H, T
from quspect.transpiler import _gauge as gauge
from quspect.transpiler._gauge import (
    _LengthModel,
    _native_kernels,
    minimize_single_qubit_gates,
)

CZ_ONE = frozenset({"h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"})
IBM_ONE = frozenset({"rz", "sx", "x"})
H_ONE = frozenset({"rz", "h"})


def _driver_available() -> bool:
    native = _native_kernels(_LengthModel(CZ_ONE))
    if native is None or not hasattr(native.core, "_gauge_minimize"):
        return False
    from quspect.synthesis.two_qubit import _native_kak

    return _native_kak() is not None


pytestmark = pytest.mark.skipif(
    not _driver_available(), reason="native minimization driver not available"
)


def _random_circuit(
    seed: int, n: int, size: int, extras: bool, basis: frozenset[str] = CZ_ONE
) -> QuantumCircuit:
    """Random circuits in the basis's single-qubit gates (plus CX / CZ).

    ``extras`` mixes in measurements, barriers (with and without qubits),
    symbolic and matrix-less gates, Toffolis, labelled instructions and
    gates outside the basis.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit()
    theta = Parameter("theta")
    custom = Gate(name="mystery", num_qubits=1)
    angles = [math.pi / 2, math.pi, -math.pi / 2, math.pi / 4, 1e-10, 0.0, -0.0]
    fixed = [
        name
        for name in ("h", "x", "sx", "s", "sdg", "t", "tdg", "y", "z")
        if name in basis
    ]
    for _ in range(size):
        a, b = rng.sample(range(n), 2)
        r = rng.randrange(16 if extras else 9)
        if r < 3:
            qc.cx(a, b) if rng.random() < 0.6 else qc.cz(a, b)
        elif r < 5:
            qc.rz(rng.choice([rng.uniform(-4, 4), *angles]), a)
        elif r < 8:
            getattr(qc, rng.choice(fixed))(a)
        elif r == 8:
            qc.barrier(a, b)
        elif r == 9:
            qc.measure([a], [rng.randrange(n)])
        elif r == 10:
            qc.rz(theta, a)  # symbolic: blocks movement
        elif r == 11:
            qc.append(custom, (a,))  # no matrix
        elif r == 12 and n >= 3:
            c = rng.choice([q for q in range(n) if q not in (a, b)])
            qc.ccx(a, b, c)
        elif r == 13:
            qc._instructions.append(
                CircuitInstruction(
                    type=InstructionType.GATE,
                    gate=rng.choice([H, T]),
                    qubits=(a,),
                    label="tagged",
                )
            )
        elif r == 14:
            qc.rx(rng.uniform(-3, 3), a)  # outside the basis
        else:
            qc.barrier()
    return qc


def _signature(circuit: QuantumCircuit) -> list[object]:
    return [(repr(i), type(i).__name__) for i in circuit.instructions]


def _both(
    qc: QuantumCircuit,
    basis: frozenset[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    global_orientation: bool = False,
    threads: str | None = None,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    monkeypatch.setenv(gauge.GAUGE_DRIVER_ENV, "python")
    ref = minimize_single_qubit_gates(qc, basis, global_orientation=global_orientation)
    monkeypatch.setenv(gauge.GAUGE_DRIVER_ENV, "auto")
    if threads is not None:
        monkeypatch.setenv("QUSPECT_COMPILE_THREADS", threads)
    got = minimize_single_qubit_gates(qc, basis, global_orientation=global_orientation)
    return ref, got


@pytest.mark.parametrize("basis", [CZ_ONE, IBM_ONE, H_ONE], ids=["cz", "ibm", "h"])
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_driver_matches_python_body(
    basis: frozenset[str], extras: bool, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    qc = _random_circuit(seed, 3 + seed % 5, 60 + 20 * seed, extras, basis)
    ref, got = _both(qc, basis, monkeypatch, global_orientation=seed % 2 == 1)
    assert _signature(got) == _signature(ref)
    # 未改动的指令对象原样复用
    ids_ref = [id(i) for i in ref.instructions]
    ids_got = [id(i) for i in got.instructions]
    reused = {id(i) for i in qc.instructions}
    assert [x in reused for x in ids_got] == [x in reused for x in ids_ref]


def test_unchanged_circuit_is_returned_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = QuantumCircuit().h(0).cx(0, 1).h(1)
    ref, got = _both(qc, CZ_ONE, monkeypatch)
    assert _signature(got) == _signature(ref)


@pytest.mark.parametrize("threads", ["1", "3", "8"])
def test_threaded_initial_search_is_identical(
    threads: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large enough for the parallel initial search (several wire groups)."""
    qc = _random_circuit(77, 24, 30000, False, IBM_ONE)
    ref, got = _both(qc, IBM_ONE, monkeypatch, threads=threads)
    assert _signature(got) == _signature(ref)


def test_invalid_thread_setting_runs_serially(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = _random_circuit(5, 12, 12000, False)
    ref, got = _both(qc, CZ_ONE, monkeypatch, threads="lots")
    assert _signature(got) == _signature(ref)


def test_driver_env_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gauge.GAUGE_DRIVER_ENV, "fast")
    with pytest.raises(ValueError, match=gauge.GAUGE_DRIVER_ENV):
        minimize_single_qubit_gates(QuantumCircuit().h(0).cx(0, 1), CZ_ONE)


def test_driver_declines_instruction_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tagged(CircuitInstruction):
        pass

    qc = _random_circuit(3, 4, 80, False)
    qc._instructions.append(Tagged(type=InstructionType.GATE, gate=H, qubits=(0,)))
    native = _native_kernels(_LengthModel(CZ_ONE))
    assert native is not None
    assert gauge._minimize_driver(qc, CZ_ONE, native) is None
    ref, got = _both(qc, CZ_ONE, monkeypatch)
    assert _signature(got) == _signature(ref)


def test_probed_arithmetic_kernels_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernels switched to the probed arithmetic still equal the Python body.

    On platforms whose CPython / numpy do not contract complex products to
    FMA the default kernels fail their self-check and the probed ones are
    used; orient_cx_gates then stays disabled.  Here (whatever the platform)
    the switch is forced and undone afterwards.
    """
    native = _native_kernels(_LengthModel(IBM_ONE))
    assert native is not None
    core = native.core
    saved = dict(gauge._NATIVE_CACHE)
    try:
        gauge._NATIVE_CACHE.clear()
        probed = gauge._probed_kernels(core, _LengthModel(IBM_ONE))
        if probed is None:
            pytest.skip("probed arithmetic does not pass the self-check here")
        assert not probed.orientation
        assert gauge._ARITHMETIC == ["probed"]
        qc = _random_circuit(9, 6, 400, True, IBM_ONE)
        assert gauge.orient_cx_gates(qc, IBM_ONE) is qc
        for driver in ("auto", "python"):
            monkeypatch.setenv(gauge.GAUGE_DRIVER_ENV, driver)
            got = gauge._minimize(qc, IBM_ONE, probed) if driver == "python" else None
            if got is None:
                got = gauge._minimize_driver(qc, IBM_ONE, probed)
            ref = gauge._minimize(qc, IBM_ONE, None)
            assert got is not None
            assert _signature(got) == _signature(ref)
    finally:
        core._gauge_arithmetic(None)
        gauge._ARITHMETIC[0] = "default"
        gauge._NATIVE_CACHE.clear()
        gauge._NATIVE_CACHE.update(saved)
