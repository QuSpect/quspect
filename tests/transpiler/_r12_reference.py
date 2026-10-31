"""Round-12 reference pipeline for the round-13 differential tests.

:func:`legacy_r12` swaps the two implementations round 13 sped up for their
frozen round-12 copies (:mod:`tests.transpiler._legacy_gauge_r12` and
:mod:`tests.synthesis._legacy_stabilizer_r12`) and clears every synthesis
cache on entry and exit, so a :func:`~quspect.transpiler.transpile` call
inside the context runs exactly the round-12 code and one outside it runs
the current code from scratch.  :func:`transpile_signature` reduces a
result to everything that must stay bit-identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

import quspect.synthesis.two_qubit as two_qubit
import quspect.transpiler._transpile as transpile_module
from quspect.gates import synthesis as one_qubit_synthesis
from quspect.transpiler import _consolidate, _toffoli
from tests.synthesis._legacy_stabilizer_r12 import legacy_try_synthesize
from tests.transpiler._legacy_gauge_r12 import legacy_minimize_single_qubit_gates


def clear_synthesis_caches() -> None:
    """Empty the exact-input synthesis caches (results depend on the code)."""
    two_qubit._SYNTH_2Q_CACHE.clear()
    two_qubit._NUM_CZ_CACHE.clear()
    one_qubit_synthesis._SYNTH_1Q_CACHE.clear()
    _consolidate._EULER_CACHE.clear()
    _toffoli._CONTEXT_COST_CACHE.clear()


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
