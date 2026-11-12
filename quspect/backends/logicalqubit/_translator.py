"""QSAL JSON → lqcloud QuantumCircuit translator.

Converts a QSAL JSON string into an lqcloud QuantumCircuit that can be
submitted to the LogicalQubit cloud platform.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from quspect.backends._exceptions import TranslationError

# ---------------------------------------------------------------------------
# Gate mapping
# ---------------------------------------------------------------------------

#: lqcloud's names for native gates that QuSpect knows under another name,
#: mapped to the QuSpect gate this translator sends as exactly that native
#: gate: ``sx``/``sxdg`` go out as lqcloud's ``sx``/``sxdg``, which lqcloud
#: serializes as ``xhalf``/``mxhalf``, and ``u3`` goes out as ``u``, one
#: ``su2``.  A chip that declares its native gates in lqcloud's names is read
#: through this table (``LogicalQubitBackend.native_gate_set``).  Native
#: gates without a QuSpect counterpart (``yhalf``, ``xyhalf``, ...) are not
#: listed; the compiler simply does not use them.
LQCLOUD_NATIVE_GATE_NAMES: dict[str, str] = {
    "xhalf": "sx",
    "mxhalf": "sxdg",
    "su2": "u3",
    "u": "u3",
}


def _build_gate_map() -> dict[str, Callable[..., None]]:
    """Return ``{gate_name: handler(qc, qubits, params)}`` for lqcloud circuit."""

    m: dict[str, Callable[..., None]] = {}

    def _reg(name: str, fn: Callable[..., None]) -> None:
        m[name] = fn

    # single-qubit, no params
    _reg("i", lambda qc, q, p: qc.id(q[0]))
    _reg("id", lambda qc, q, p: qc.id(q[0]))
    _reg("h", lambda qc, q, p: qc.h(q[0]))
    _reg("x", lambda qc, q, p: qc.x(q[0]))
    _reg("y", lambda qc, q, p: qc.y(q[0]))
    _reg("z", lambda qc, q, p: qc.z(q[0]))
    _reg("s", lambda qc, q, p: qc.s(q[0]))
    _reg("sdg", lambda qc, q, p: qc.sdg(q[0]))
    _reg("t", lambda qc, q, p: qc.t(q[0]))
    _reg("tdg", lambda qc, q, p: qc.tdg(q[0]))
    # lqcloud 0.6 has √X and its inverse: ``sx``/``sxdg`` are aliases of the
    # native ``xhalf``/``mxhalf`` pulses, one pulse instead of ``H · S · H``.
    _reg("sx", lambda qc, q, p: qc.sx(q[0]))
    _reg("sxdg", lambda qc, q, p: qc.sxdg(q[0]))

    # single-qubit, parameterized
    _reg("rx", lambda qc, q, p: qc.rx(float(p[0]), q[0]))
    _reg("ry", lambda qc, q, p: qc.ry(float(p[0]), q[0]))
    _reg("rz", lambda qc, q, p: qc.rz(float(p[0]), q[0]))
    _reg("p", lambda qc, q, p: qc.rz(float(p[0]), q[0]))
    _reg("u1", lambda qc, q, p: qc.rz(float(p[0]), q[0]))
    _reg("u2", lambda qc, q, p: _u2(qc, q, p))
    _reg("u3", lambda qc, q, p: _u3(qc, q, p))

    # two-qubit
    _reg("cx", lambda qc, q, p: qc.cx(q[0], q[1]))
    _reg("cz", lambda qc, q, p: qc.cz(q[0], q[1]))
    _reg("swap", lambda qc, q, p: qc.swap(q[0], q[1]))

    # two-qubit parameterized
    _reg("crx", lambda qc, q, p: qc.crx(float(p[0]), q[0], q[1]))
    _reg("cry", lambda qc, q, p: qc.cry(float(p[0]), q[0], q[1]))
    _reg("crz", lambda qc, q, p: qc.crz(float(p[0]), q[0], q[1]))
    _reg("cp", lambda qc, q, p: _cp(qc, q, p))
    _reg("cu3", lambda qc, q, p: _cu3(qc, q, p))

    # three-qubit
    _reg("ccx", lambda qc, q, p: qc.ccx(q[0], q[1], q[2]))

    return m


def _u2(qc: Any, qubits: list[int], params: list[float]) -> None:
    """U2(φ, λ) = U3(π/2, φ, λ)."""
    qc.u(math.pi / 2, float(params[0]), float(params[1]), qubits[0])


def _u3(qc: Any, qubits: list[int], params: list[float]) -> None:
    """U3(θ, φ, λ)."""
    qc.u(float(params[0]), float(params[1]), float(params[2]), qubits[0])


def _cu3(qc: Any, qubits: list[int], params: list[float]) -> None:
    """Decompose CU3(θ, φ, λ) using methods available in lqcloud 0.4.1.

    lqcloud exposes ``u`` and ``p`` but no ``cu3`` method.  This is the
    standard OpenQASM/Qiskit decomposition of a controlled U3 into two CNOTs
    and single-qubit gates.  lqcloud may subsequently lower ``p`` to ``rz``;
    that changes only an overall global phase.
    """
    theta, phi, lam = (float(value) for value in params)
    control, target = qubits

    qc.p((lam + phi) / 2, control)
    qc.p((lam - phi) / 2, target)
    qc.cx(control, target)
    qc.u(-theta / 2, 0.0, -(phi + lam) / 2, target)
    qc.cx(control, target)
    qc.u(theta / 2, phi, 0.0, target)


def _cp(qc: Any, qubits: list[int], params: list[float]) -> None:
    """CP(θ) = P(θ/2) on control followed by CRZ(θ)."""
    theta = float(params[0])
    control, target = qubits
    qc.p(theta / 2, control)
    qc.crz(theta, control, target)


_GATE_MAP = _build_gate_map()


def _measurement_run(
    instructions: list[dict[str, Any]], start: int
) -> tuple[list[int], int]:
    """Return the qubits of the consecutive measurements starting at ``start``.

    lqcloud wants a barrier between a qubit's last operation and its
    measurement, and LQCloud reads the measured qubits out together.  A run
    of consecutive ``MEASURE`` instructions therefore shares one barrier over
    all its qubits; a ``barrier(q); measure(q)`` pair per instruction would
    stagger the readout.  The run stops before an instruction that measures
    one of its qubits again, because a repeated measurement needs a barrier
    of its own.

    Returns:
        ``(qubits, end)``: the distinct measured qubits in order and the
        index one past the last instruction of the run.
    """
    qubits: dict[int, None] = {}
    end = start
    while end < len(instructions):
        instruction = instructions[end]
        if instruction.get("type", "") != "MEASURE":
            break
        measured = [int(q) for q in instruction.get("qubits", [])]
        if end > start and not qubits.keys().isdisjoint(measured):
            break
        qubits.update(dict.fromkeys(measured))
        end += 1
    return list(qubits), end


class LQCircuitTranslator:
    """Translate QSAL JSON to an lqcloud QuantumCircuit."""

    def translate(self, qsal_json: str) -> Any:
        """Convert a QSAL JSON string to an lqcloud QuantumCircuit.

        Args:
            qsal_json: A QSAL JSON string from :func:`quspect.ir.qsal.dumps`.

        Returns:
            An lqcloud :class:`~lqcloud.circuit.QuantumCircuit`.

        Raises:
            TranslationError: If QSAL is malformed or contains unsupported
                instructions (noise channels, unbound parameters, unknown gates).
        """
        from lqcloud.circuit.circuit import QuantumCircuit  # noqa: N814

        try:
            doc: dict[str, Any] = json.loads(qsal_json)
        except json.JSONDecodeError as exc:
            raise TranslationError(f"Failed to parse QSAL JSON: {exc}") from exc

        circuit_data = doc.get("circuit")
        if circuit_data is None:
            raise TranslationError("QSAL document has no circuit section")

        n_qubits = int(circuit_data.get("num_qubits", 0))
        n_clbits = int(circuit_data.get("num_clbits", 0))
        if n_qubits == 0 and n_clbits == 0:
            raise TranslationError("QSAL circuit has zero qubits and zero clbits")

        if n_clbits > 0:
            qc = QuantumCircuit(n_qubits, n_clbits)
        else:
            qc = QuantumCircuit(n_qubits)

        instructions = circuit_data.get("instructions", [])
        # lqcloud requires every measured qubit to have been covered by a
        # barrier since its previous measurement (or since circuit start for
        # its first measurement).  Track coverage per qubit: a single global
        # boolean is incorrect for partial barriers and repeated measurements.
        barriered_qubits: set[int] = set()
        # Consecutive measurements share one barrier (see _measurement_run):
        # instructions before this index already had theirs emitted.
        run_end = 0

        for i, instr in enumerate(instructions):
            itype = instr.get("type", "")
            qubits: list[int] = [int(q) for q in instr.get("qubits", [])]
            clbits: list[int] = [int(c) for c in instr.get("clbits", [])]

            if itype == "GATE":
                gate_data = instr.get("gate")
                if gate_data is None:
                    raise TranslationError(
                        f"Instruction {i}: GATE type missing gate data"
                    )
                gate_name = gate_data["name"].lower()
                gate_params_raw = gate_data.get("params", [])

                params: list[float] = []
                for p in gate_params_raw:
                    if isinstance(p, dict) and "_param" in p:
                        raise TranslationError(
                            f"Instruction {i}: gate '{gate_name}' has unbound "
                            f"parameter '{p['_param']}'. "
                            f"Bind all parameters before submission."
                        )
                    params.append(float(p))

                handler = _GATE_MAP.get(gate_name)
                if handler is None:
                    raise TranslationError(
                        f"Instruction {i}: gate '{gate_name}' not supported "
                        f"by LogicalQubit translator. "
                        f"Supported: {sorted(_GATE_MAP.keys())}"
                    )
                handler(qc, qubits, params)
                # A later gate makes any earlier barrier stale for the
                # qubits it touches; measurement needs a new boundary after
                # the last operation on each measured qubit.
                barriered_qubits.difference_update(qubits)

            elif itype == "BARRIER":
                if qubits:
                    qc.barrier(*qubits)
                    barriered_qubits.update(qubits)
                else:
                    qc.barrier()
                    barriered_qubits.update(range(n_qubits))

            elif itype == "MEASURE":
                if len(qubits) != len(clbits):
                    raise TranslationError(
                        f"Instruction {i}: MEASURE has {len(qubits)} qubits "
                        f"but {len(clbits)} classical bits"
                    )

                if i >= run_end:
                    run_qubits, run_end = _measurement_run(instructions, i)
                    missing = [q for q in run_qubits if q not in barriered_qubits]
                    if missing:
                        qc.barrier(*missing)
                        barriered_qubits.update(missing)

                for q_idx, c_idx in zip(qubits, clbits, strict=True):
                    qc.measure(q_idx, c_idx)
                    # A measurement consumes this qubit's barrier coverage;
                    # measuring it again requires another barrier.
                    barriered_qubits.discard(q_idx)

            elif itype == "CHANNEL":
                raise TranslationError(
                    f"Instruction {i}: noise channels cannot run on "
                    f"real hardware. Remove channels or use a simulator."
                )

            else:
                raise TranslationError(
                    f"Instruction {i}: unknown instruction type {itype!r}"
                )

        return qc
