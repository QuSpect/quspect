"""Round-12 reference pipeline for the round-13 differential tests.

:func:`legacy_r12` swaps the two implementations round 13 sped up for their
frozen round-12 copies (:mod:`tests.transpiler._legacy_gauge_r12` and
:mod:`tests.synthesis._legacy_stabilizer_r12`) and clears every synthesis
cache on entry and exit, so a :func:`~quspect.transpiler.transpile` call
inside the context runs exactly the round-12 code and one outside it runs
the current code from scratch.  :func:`transpile_signature` reduces a
result to everything that must stay bit-identical.

Both contexts run the Python numerics (:func:`_python_numerics`): the
native synthesis, gate matrices, block products and single-qubit
minimization run on the deterministic numerics layer, which is not bit for
bit equal to numpy / LAPACK / the platform libm; the differential tests
check the algorithmic speed-ups, not the numerical implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

import quspect.gates._matrix_utils as matrix_utils
import quspect.synthesis.two_qubit as two_qubit
import quspect.transpiler._transpile as transpile_module
from quspect.gates import synthesis as one_qubit_synthesis
from quspect.transpiler import _consolidate, _toffoli
from tests.synthesis._legacy_stabilizer_r12 import legacy_try_synthesize
from tests.transpiler._legacy_gauge_r12 import legacy_minimize_single_qubit_gates


def clear_synthesis_caches() -> None:
    """Empty the exact-input synthesis caches (results depend on the code)."""
    two_qubit._SYNTH_2Q_CACHE.clear()
    two_qubit._APPROX_2Q_CACHE.clear()
    two_qubit._SWAP_FACE_CACHE.clear()
    two_qubit._NUM_CZ_CACHE.clear()
    one_qubit_synthesis._SYNTH_1Q_CACHE.clear()
    _consolidate._EULER_CACHE.clear()
    _toffoli._CONTEXT_COST_CACHE.clear()
    _toffoli._CCX_CHOICE_CACHE.clear()


@contextmanager
def python_numerics() -> Iterator[pytest.MonkeyPatch]:
    """:func:`_python_numerics` with the synthesis caches emptied on entry and exit.

    Results computed with the Python numerics must not stay in the caches,
    where later (native) compilations in the same process would reuse them.
    """
    with pytest.MonkeyPatch.context() as mp:
        _python_numerics(mp)
        clear_synthesis_caches()
        try:
            yield mp
        finally:
            clear_synthesis_caches()


def _python_numerics(mp: pytest.MonkeyPatch) -> None:
    """Use the Python (numpy) numerics everywhere (see the module docs)."""
    mp.setattr(two_qubit, "_native_kak", lambda: None)
    mp.setattr(matrix_utils, "_native_gate_matrix", lambda: None)
    mp.setenv("QUSPECT_GAUGE_BACKEND", "python")
    mp.setenv("QUSPECT_TRANSLATE_DRIVER", "python")


def _without_round_17(mp: pytest.MonkeyPatch) -> None:
    """Switch off the round-17 algorithm changes (not speed-ups).

    The mirrored stabilizers and the SWAP-face template of the KAK synthesis,
    the global CX orientation and the SWAP-face rewrites of the final
    circuit change results on purpose; the
    differential tests check the round-13 speed-ups with them off.
    """
    mp.setattr(two_qubit, "_MIRRORED_STABILIZERS", False)
    mp.setattr(two_qubit, "_SWAP_FACE_TEMPLATE", False)
    mp.setattr(transpile_module, "_ORIENT_LEVEL", 99)
    mp.setattr(transpile_module, "_SWAP_POST", False)
    mp.setattr(transpile_module, "_SWAP_LATE", False)


@contextmanager
def current_without_round_17() -> Iterator[None]:
    """Run the current code with the round-17 algorithm changes off."""
    with pytest.MonkeyPatch.context() as mp:
        _without_round_17(mp)
        _python_numerics(mp)
        clear_synthesis_caches()
        try:
            yield
        finally:
            clear_synthesis_caches()


def _legacy_minimize(circuit: Any, one_qubit: frozenset[str], **_: Any) -> Any:
    return legacy_minimize_single_qubit_gates(circuit, one_qubit)


@contextmanager
def legacy_r12() -> Iterator[None]:
    """Run the round-12 single-qubit minimization and KAK stabilizer choice."""
    with pytest.MonkeyPatch.context() as mp:
        _without_round_17(mp)
        _python_numerics(mp)
        mp.setattr(transpile_module, "minimize_single_qubit_gates", _legacy_minimize)
        mp.setattr(two_qubit, "_try_synthesize", legacy_try_synthesize)
        clear_synthesis_caches()
        try:
            yield
        finally:
            clear_synthesis_caches()


def transpile_signature(result: Any) -> tuple[Any, ...]:
    """Gate names, qubits, parameter reprs, clbits and both layouts."""
    gates = tuple(
        (
            str(ins.type),
            None if ins.gate is None else ins.gate.name,
            tuple(ins.qubits),
            () if ins.gate is None else tuple(repr(p) for p in ins.gate.params),
            tuple(ins.clbits),
        )
        for ins in result.circuit
    )
    return gates, tuple(result.initial_layout), tuple(result.final_layout)
