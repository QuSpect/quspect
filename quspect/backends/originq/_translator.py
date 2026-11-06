"""QuSpect circuit → OriginIR text, and provider result decoding.

Pure Python: no pyqpanda3 needed.  The OriginIR spelling of every gate is the
one pyqpanda3 0.5.1 itself emits (``convert_qprog_to_originir``); the tests
check each one against pyqpanda3's parser and QuSpect's simulator.

Bit-order contract (measured on the Origin Quantum cloud, November 2026):

* QuSpect bitstrings put classical bit 0 **leftmost**.  Origin Quantum
  results are integers (hex keys) whose bit ``k`` is the ``k``-th result bit.
* The cloud ignores the classical-bit operand of ``MEASURE``: result bit
  ``k`` is the ``k``-th ``MEASURE`` instruction.  When measurements are not
  in ascending qubit order the server decodes them incorrectly (two
  outcomes collapse onto one key).
* A malformed program is not rejected; the task stays "computing" forever,
  so the emitter must only produce OriginIR the server is known to accept.

The translator therefore never relies on provider classical bits: every
measured qubit is measured exactly once, at the end, in ascending qubit
order, into dense classical bit ``k`` (its rank).  :func:`decode_distribution`
maps result bit ``k`` back to the original QuSpect classical bit(s) through a
:class:`MeasurementMap`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quspect.backends._exceptions import TranslationError

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit


@dataclass(frozen=True)
class MeasurementMap:
    """How provider result bits map back to QuSpect classical bits.

    Attributes:
        measured_qubits: Submitted qubit of result bit ``k`` (ascending).
        clbits: ``clbits[k]`` are the QuSpect classical bits that receive
            result bit ``k`` (several when one qubit was measured into
            several classical bits).
        num_clbits: Width of the QuSpect bitstring; classical bits that no
            measurement writes read ``0``, as in the QuSpect simulators.
    """

    measured_qubits: tuple[int, ...]
    clbits: tuple[tuple[int, ...], ...]
    num_clbits: int


@dataclass(frozen=True)
class OriginIRProgram:
    """One translated experiment."""

    text: str
    num_qubits: int  # QINIT: highest submitted qubit + 1
    num_clbits: int  # CREG: number of measured qubits
    measurement: MeasurementMap


def _angle(value: float) -> str:
    # pyqpanda3 prints 8 decimals (RX(1e-9) becomes 0); the cloud parses 15.
    # Fixed-point only: never emit exponent notation.
    return f"{value:.15f}"


def _q(qubit: int) -> str:
    return f"q[{qubit}]"


def _args(params: Sequence[float]) -> str:
    return "(" + ",".join(_angle(p) for p in params) + ")"


_Emitter = Callable[[Sequence[int], Sequence[float]], list[str]]


def _simple(name: str) -> _Emitter:
    return lambda q, p: [f"{name} " + ",".join(_q(x) for x in q)]


def _dagger(name: str) -> _Emitter:
    return lambda q, p: ["DAGGER", f"{name} {_q(q[0])}", "ENDDAGGER"]


def _rotation(name: str) -> _Emitter:
    return lambda q, p: [f"{name} {_q(q[0])},{_args(p)}"]


def _controlled_rotation(name: str) -> _Emitter:
    return lambda q, p: [
        f"CONTROL {_q(q[0])}",
        f"{name} {_q(q[1])},{_args(p)}",
        "ENDCONTROL",
    ]


def _cu3(q: Sequence[int], p: Sequence[float]) -> list[str]:
    """Controlled U3 via the standard two-CX decomposition (exact phase)."""
    theta, phi, lam = (float(value) for value in p)
    control, target = q
    return [
        f"P {_q(control)},{_args([(lam + phi) / 2])}",
        f"P {_q(target)},{_args([(lam - phi) / 2])}",
        f"CNOT {_q(control)},{_q(target)}",
        f"U3 {_q(target)},{_args([-theta / 2, 0.0, -(phi + lam) / 2])}",
        f"CNOT {_q(control)},{_q(target)}",
        f"U3 {_q(target)},{_args([theta / 2, phi, 0.0])}",
    ]


def _ccx(q: Sequence[int], p: Sequence[float]) -> list[str]:
    return [f"CONTROL {_q(q[0])},{_q(q[1])}", f"X {_q(q[2])}", "ENDCONTROL"]


_GATE_MAP: dict[str, _Emitter] = {
    "id": _simple("I"),
    "i": _simple("I"),
    "h": _simple("H"),
    "x": _simple("X"),
    "y": _simple("Y"),
    "z": _simple("Z"),
    "s": _simple("S"),
    "sdg": _dagger("S"),
    "t": _simple("T"),
    "tdg": _dagger("T"),
    # X1 = RX(pi/2) differs from SX only by a global phase.
    "sx": _simple("X1"),
    "sxdg": _dagger("X1"),
    "rx": _rotation("RX"),
    "ry": _rotation("RY"),
    "rz": _rotation("RZ"),
    "p": _rotation("P"),
    "u1": _rotation("U1"),
    "u2": _rotation("U2"),
    "u3": _rotation("U3"),
    "cx": _simple("CNOT"),
    "cz": _simple("CZ"),
    "swap": _simple("SWAP"),
    "crx": _controlled_rotation("RX"),
    "cry": _controlled_rotation("RY"),
    "crz": _controlled_rotation("RZ"),
    # OriginIR "CR" is the controlled phase gate.
    "cp": lambda q, p: [f"CR {_q(q[0])},{_q(q[1])},{_args(p)}"],
    "cu3": _cu3,
    "ccx": _ccx,
}
SUPPORTED_GATES = frozenset(_GATE_MAP)


def translate(
    circuit: QuantumCircuit,
    *,
    qubit_map: Sequence[int] | None = None,
) -> OriginIRProgram:
    """Translate a bound QuSpect circuit into canonical OriginIR.

    Args:
        circuit: Bound circuit.  Measurements must be terminal: once a qubit
            is measured no later gate may act on it.
        qubit_map: Optional relabelling ``qubit_map[wire] = submitted qubit``
            (used to place a routed wire-space circuit on physical qubits).

    Raises:
        TranslationError: For unsupported gates, unbound parameters, noise
            channels, mid-circuit measurement, a classical bit written more
            than once, or a circuit without measurements.
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

    body: list[str] = []
    used: set[int] = set()
    measured_by_qubit: dict[int, list[int]] = {}
    written_clbits: set[int] = set()

    for index, instruction in enumerate(circuit.instructions):
        qubits = [place(q) for q in instruction.qubits]
        if instruction.is_measure:
            for qubit, clbit in zip(qubits, instruction.clbits, strict=True):
                if clbit in written_clbits:
                    raise TranslationError(
                        f"Instruction {index}: classical bit {clbit} is written "
                        "more than once; Origin Quantum results cannot express "
                        "overwritten classical bits."
                    )
                written_clbits.add(clbit)
                measured_by_qubit.setdefault(qubit, []).append(clbit)
                used.add(qubit)
            continue
        if instruction.is_channel:
            raise TranslationError(
                f"Instruction {index}: noise channels cannot run on Origin "
                "Quantum backends. Remove channels or use a local simulator."
            )
        if instruction.is_barrier:
            # A barrier after a terminal measurement orders nothing; keep
            # only the still-active qubits.
            active = [q for q in qubits if q not in measured_by_qubit]
            if active:
                body.append("BARRIER " + ",".join(_q(q) for q in active))
                used.update(active)
            continue
        touched_measured = [q for q in qubits if q in measured_by_qubit]
        if touched_measured:
            raise TranslationError(
                f"Instruction {index}: qubit(s) {touched_measured} are used "
                "after being measured. Origin Quantum backends support "
                "terminal measurements only."
            )
        gate = instruction.gate
        if gate is None:
            raise TranslationError(f"Instruction {index}: unknown instruction.")
        name = gate.name.lower()
        params: list[float] = []
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
            params.append(numeric)
        emit = _GATE_MAP.get(name)
        if emit is None:
            raise TranslationError(
                f"Instruction {index}: gate {name!r} is not supported by the "
                f"Origin Quantum translator. Supported: {sorted(_GATE_MAP)}"
            )
        body.extend(emit(qubits, params))
        used.update(qubits)

    if not measured_by_qubit:
        raise TranslationError(
            "Circuit has no measurement. A sampling request sent to Origin "
            "Quantum must measure at least one qubit."
        )

    measured_qubits = tuple(sorted(measured_by_qubit))
    body.extend(
        f"MEASURE {_q(qubit)},c[{rank}]" for rank, qubit in enumerate(measured_qubits)
    )
    num_qubits = max(used) + 1
    num_clbits = len(measured_qubits)
    text = "\n".join([f"QINIT {num_qubits}", f"CREG {num_clbits}", *body]) + "\n"
    return OriginIRProgram(
        text=text,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        measurement=MeasurementMap(
            measured_qubits=measured_qubits,
            clbits=tuple(tuple(measured_by_qubit[q]) for q in measured_qubits),
            num_clbits=circuit.num_clbits,
        ),
    )


def _outcome_value(key: str, width: int) -> int:
    """Parse a hex (``0x..``) or little-endian binary outcome key."""
    text = str(key).strip()
    try:
        value = int(text, 16) if text.lower().startswith("0x") else int(text, 2)
    except ValueError:
        raise TranslationError(
            f"Unrecognized Origin Quantum outcome {key!r}."
        ) from None
    if value < 0 or value >= (1 << width):
        raise TranslationError(
            f"Origin Quantum outcome {key!r} does not fit the {width} measured "
            "qubit(s) of this experiment."
        )
    return value


def decode_distribution(
    raw: Mapping[str, float],
    clbit_map: Sequence[Sequence[int]],
    num_clbits: int,
) -> dict[str, float]:
    """Re-key a provider distribution into QuSpect bitstrings.

    Args:
        raw: Provider outcome → probability or count (hex or binary keys).
        clbit_map: ``clbit_map[k]`` = QuSpect clbits receiving result bit k.
        num_clbits: QuSpect bitstring width.

    Returns:
        QuSpect bitstring (clbit 0 leftmost) → value, with unwritten classical
        bits reading ``0``.
    """
    width = len(clbit_map)
    decoded: dict[str, float] = {}
    for key, value in raw.items():
        outcome = _outcome_value(key, width)
        chars = ["0"] * num_clbits
        for bit, clbits in enumerate(clbit_map):
            if (outcome >> bit) & 1:
                for clbit in clbits:
                    chars[clbit] = "1"
        text = "".join(chars)
        decoded[text] = decoded.get(text, 0.0) + float(value)
    return decoded
