"""QuSpect circuit → QCIS text, and Tianyan result decoding.

Pure Python.  Every gate spelling is checked against cqlib's QCIS parser and
statevector simulator in the tests.

Rules measured on the Tianyan cloud (November 2026):

* A qubit may be measured only once, and no gate may follow its measurement
  (both are rejected at submission).  ``I`` needs a positive duration.
* Results carry the measurement order explicitly: ``resultStatus[0]`` lists
  the measured qubits in ``M`` order and every following row is one shot in
  that order.  ``probability`` keys are the same bits **reversed** (the first
  measured qubit is the rightmost character); on QPUs it is the
  readout-corrected distribution.
* A native QPU batch must measure the same qubits in the same order in every
  program.

QuSpect therefore measures each measured qubit exactly once, at the end, in
ascending qubit order, and decodes shots through a :class:`MeasurementMap`,
validating it against the order the server reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quspect.backends._exceptions import TranslationError

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit

#: QuSpect gate → QCIS instruction (every other gate is decomposed first).
_GATES: dict[str, str] = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "SD",
    "t": "T",
    "tdg": "TD",
    # X2P / X2M are RX(±pi/2), equal to SX / SXdg up to a global phase.
    "sx": "X2P",
    "sxdg": "X2M",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "cz": "CZ",
}
#: Gates QuSpect must lower everything to before translation.
QCIS_BASIS = frozenset(_GATES)
#: Gates dropped from the program (identity; QCIS ``I`` takes a duration).
_DROPPED = frozenset({"id", "i"})


@dataclass(frozen=True)
class MeasurementMap:
    """How measured qubits map back to QuSpect classical bits.

    Attributes:
        measured_qubits: Submitted qubits in measurement order (ascending).
        clbits: ``clbits[k]`` are the QuSpect classical bits that receive the
            outcome of ``measured_qubits[k]``.
        num_clbits: Width of the QuSpect bitstring; classical bits that no
            measurement writes read ``0``.
    """

    measured_qubits: tuple[int, ...]
    clbits: tuple[tuple[int, ...], ...]
    num_clbits: int


@dataclass(frozen=True)
class QCISProgram:
    """One translated experiment."""

    text: str
    measurement: MeasurementMap


def angle(value: float) -> str:
    """Shortest round-trip decimal, always in fixed-point notation."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        text = format(Decimal(text), "f")
    return text


def translate(
    circuit: QuantumCircuit,
    *,
    qubit_map: Sequence[int] | None = None,
) -> QCISProgram:
    """Translate a bound circuit in the QCIS basis into QCIS text.

    Args:
        circuit: Bound circuit using only :data:`QCIS_BASIS` gates (plus
            identities, barriers and terminal measurements).
        qubit_map: Optional relabelling ``qubit_map[wire] = submitted qubit``.

    Raises:
        TranslationError: For gates outside the basis, unbound parameters,
            noise channels, mid-circuit measurement, a classical bit written
            more than once, or a circuit without measurements.
    """
    if qubit_map is not None:
        mapped = [int(value) for value in qubit_map]
        if len(set(mapped)) != len(mapped):
            raise TranslationError("qubit_map must not contain duplicates.")
        if len(mapped) < circuit.num_qubits:
            raise TranslationError(
                f"qubit_map covers {len(mapped)} wires but the circuit uses "
                f"{circuit.num_qubits}."
            )

    def place(qubit: int) -> int:
        return qubit if qubit_map is None else int(qubit_map[qubit])

    lines: list[str] = []
    measured_by_qubit: dict[int, list[int]] = {}
    written_clbits: set[int] = set()

    for index, instruction in enumerate(circuit.instructions):
        qubits = [place(q) for q in instruction.qubits]
        if instruction.is_measure:
            for qubit, clbit in zip(qubits, instruction.clbits, strict=True):
                if clbit in written_clbits:
                    raise TranslationError(
                        f"Instruction {index}: classical bit {clbit} is written "
                        "more than once; Tianyan results cannot express "
                        "overwritten classical bits."
                    )
                written_clbits.add(clbit)
                measured_by_qubit.setdefault(qubit, []).append(clbit)
            continue
        if instruction.is_channel:
            raise TranslationError(
                f"Instruction {index}: noise channels cannot run on Tianyan "
                "backends. Remove channels or use a local simulator."
            )
        if instruction.is_barrier:
            active = [q for q in qubits if q not in measured_by_qubit]
            if active:
                lines.append("B " + " ".join(f"Q{q}" for q in active))
            continue
        touched = [q for q in qubits if q in measured_by_qubit]
        if touched:
            raise TranslationError(
                f"Instruction {index}: qubit(s) {touched} are used after being "
                "measured. Tianyan supports terminal measurements only."
            )
        gate = instruction.gate
        if gate is None:
            raise TranslationError(f"Instruction {index}: unknown instruction.")
        name = gate.name.lower()
        if name in _DROPPED:
            continue
        mnemonic = _GATES.get(name)
        if mnemonic is None:
            raise TranslationError(
                f"Instruction {index}: gate {name!r} is not a QCIS instruction; "
                f"decompose to {sorted(QCIS_BASIS)} first."
            )
        params: list[str] = []
        for value in gate.params:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise TranslationError(
                    f"Instruction {index}: gate {name!r} has an unbound "
                    "parameter. Bind all parameters before submission."
                ) from None
            if not math.isfinite(numeric):
                raise TranslationError(
                    f"Instruction {index}: gate {name!r} has a non-finite parameter."
                )
            params.append(angle(numeric))
        lines.append(" ".join([mnemonic, *(f"Q{q}" for q in qubits), *params]))

    if not measured_by_qubit:
        raise TranslationError(
            "Circuit has no measurement. A sampling request sent to Tianyan "
            "must measure at least one qubit."
        )
    measured_qubits = tuple(sorted(measured_by_qubit))
    lines.extend(f"M Q{qubit}" for qubit in measured_qubits)
    return QCISProgram(
        text="\n".join(lines),
        measurement=MeasurementMap(
            measured_qubits=measured_qubits,
            clbits=tuple(tuple(measured_by_qubit[q]) for q in measured_qubits),
            num_clbits=circuit.num_clbits,
        ),
    )


def _bitstring(
    values: Sequence[int],
    order: Sequence[int],
    columns: Mapping[int, Sequence[int]],
    num_clbits: int,
) -> str:
    chars = ["0"] * num_clbits
    for qubit, value in zip(order, values, strict=True):
        if int(value) not in (0, 1):
            raise TranslationError(f"Tianyan shot value {value!r} is not 0 or 1.")
        if int(value):
            for clbit in columns[qubit]:
                chars[clbit] = "1"
    return "".join(chars)


def decode_result(
    result: Mapping[str, Any],
    measurement: MeasurementMap | Sequence[Sequence[int]],
    *,
    measured_qubits: Sequence[int] | None = None,
    num_clbits: int | None = None,
) -> tuple[dict[str, int], dict[str, float] | None]:
    """Decode one ``experimentResultModelList`` entry.

    Args:
        result: The provider result (``resultStatus``, ``probability``).
        measurement: The submission's :class:`MeasurementMap`, or just its
            ``clbits`` (then pass ``measured_qubits`` and ``num_clbits``).

    Returns:
        ``(counts, probabilities)`` in QuSpect bitstrings (clbit 0 leftmost).
        ``probabilities`` is the provider's distribution (readout-corrected on
        QPUs), or ``None`` when the provider returned none.

    Raises:
        TranslationError: If the reported measurement order does not match
            the submitted measurements, or the payload is malformed.
    """
    if isinstance(measurement, MeasurementMap):
        clbits = measurement.clbits
        submitted = measurement.measured_qubits
        width = measurement.num_clbits
    else:
        clbits = tuple(tuple(c) for c in measurement)
        submitted = tuple(measured_qubits or ())
        width = int(num_clbits or 0)
    rows = result.get("resultStatus")
    if not isinstance(rows, list) or not rows:
        raise TranslationError("Tianyan result has no resultStatus.")
    order = [int(q) for q in rows[0]]
    if submitted and sorted(order) != sorted(submitted):
        raise TranslationError(
            f"Tianyan measured qubits {order}, but {list(submitted)} were submitted."
        )
    if len(order) != len(clbits):
        raise TranslationError(
            f"Tianyan reported {len(order)} measured qubits for a "
            f"{len(clbits)}-qubit measurement map."
        )
    reference = list(submitted) if submitted else sorted(order)
    columns = {qubit: clbits[k] for k, qubit in enumerate(reference)}

    counts: dict[str, int] = {}
    for row in rows[1:]:
        if not isinstance(row, list) or len(row) != len(order):
            raise TranslationError("Tianyan shot row has the wrong width.")
        key = _bitstring(row, order, columns, width)
        counts[key] = counts.get(key, 0) + 1

    probabilities: dict[str, float] | None = None
    raw = result.get("probability")
    if isinstance(raw, str) and raw.strip():
        import json

        raw = json.loads(raw)
    if isinstance(raw, Mapping) and raw:
        probabilities = {}
        for key, value in raw.items():
            bits = str(key)
            if len(bits) != len(order):
                raise TranslationError(
                    f"Tianyan probability key {key!r} has wrong width."
                )
            # The first measured qubit is the rightmost character.
            values = [int(bit) for bit in reversed(bits)]
            text = _bitstring(values, order, columns, width)
            probabilities[text] = probabilities.get(text, 0.0) + float(value)
    return counts, probabilities
