"""Extra helpers for the circuit / simulation cross-validation tests.

Only used by ``test_circuit_*``, ``test_simulation_*``, ``test_algorithms_*``
and ``test_native_*``.  The shared conversions live in
:mod:`qiskit_tests.helpers`; this module adds the *deferred measurement*
reference used for mid-circuit measurements, which qiskit ``Statevector``
cannot simulate directly.

Deferred measurement principle
------------------------------
A projective Z measurement of qubit ``q`` whose outcome is only *recorded*
(no classical feed-forward) is equivalent, for the joint distribution of all
recorded outcomes, to ``CX(q, a)`` onto a fresh ancilla ``a`` prepared in
``|0>`` followed by a terminal measurement of ``a``.  The CX copies the Z
basis value of ``q`` into ``a`` and decoheres ``q`` in exactly the same way as
the measurement, so later gates on ``q`` see the same reduced state.  When a
clbit is written several times the *last* write wins, as in quspect.
"""

from __future__ import annotations

from typing import Any

import qiskit

import quspect
from qiskit_tests.helpers import GATE_SPECS, qiskit_probabilities


def _float_params(params: tuple[Any, ...]) -> list[float]:
    """Convert bound gate parameters to plain floats."""
    return [float(p) for p in params]


def deferred_measurement_qiskit(circ: quspect.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Translate a quspect circuit with mid-circuit measurements to qiskit.

    Every ``measure(q -> c)`` becomes ``cx(q, a)`` on a fresh ancilla ``a``;
    at the very end each clbit ``c`` is measured from the ancilla of its last
    write.  Ancillas are appended after the ``circ.num_qubits`` data qubits.
    Barriers are dropped.  Noise channels are not supported.

    Args:
        circ: A bound quspect circuit (gates, measurements, barriers only).

    Returns:
        A qiskit circuit with only terminal measurements, suitable for
        :func:`qiskit_tests.helpers.qiskit_probabilities`.

    Raises:
        NotImplementedError: If the circuit contains a noise channel.
    """
    n = circ.num_qubits
    num_meas = sum(len(i.qubits) for i in circ.instructions if i.is_measure)
    qc = qiskit.QuantumCircuit(n + num_meas, max(circ.num_clbits, 1))
    next_anc = n
    last_write: dict[int, int] = {}
    for instr in circ.instructions:
        if instr.is_gate:
            assert instr.gate is not None
            spec = GATE_SPECS[instr.gate.name]
            spec.apply_qiskit(qc, _float_params(instr.gate.params), instr.qubits)
        elif instr.is_measure:
            for q, c in zip(instr.qubits, instr.clbits, strict=True):
                qc.cx(q, next_anc)
                last_write[c] = next_anc
                next_anc += 1
        elif instr.is_channel:
            raise NotImplementedError("noise channels are not supported here")
    for c, anc in sorted(last_write.items()):
        qc.measure(anc, c)
    return qc


def deferred_probabilities(circ: quspect.QuantumCircuit) -> dict[str, float]:
    """Exact quspect-keyed outcome distribution of a mid-circuit circuit.

    Args:
        circ: A bound quspect circuit with (possibly mid-circuit)
            measurements and no noise.

    Returns:
        Dict ``bitstring -> probability`` in quspect convention (``s[c]`` is
        clbit ``c``), computed with qiskit ``Statevector``.
    """
    probs = qiskit_probabilities(deferred_measurement_qiskit(circ))
    nc = circ.num_clbits
    out: dict[str, float] = {}
    for key, p in probs.items():
        k = key[:nc]
        out[k] = out.get(k, 0.0) + p
    return out


def bit_reversal_qiskit(num_qubits: int) -> qiskit.QuantumCircuit:
    """Return a qiskit circuit reversing the qubit order with SWAPs."""
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    return qc
