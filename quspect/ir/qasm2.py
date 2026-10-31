"""OpenQASM 2.0 exporter for QuSpect quantum circuits.

This module converts :class:`~quspect.circuit.circuit.QuantumCircuit`
instances into OpenQASM 2.0 text, suitable for submission to IBM
quantum hardware, AWS Braket, or other QASM-compatible backends.

The exporter supports three modes:

- **STRICT**: Only gates in the ``qelib1.inc`` standard library.
  Parameterized gates **must** have concrete (float) parameter values.
- **DECOMPOSE**: Automatically decompose unsupported gates into
  supported primitives, applying the registered decomposition rules
  repeatedly until only ``qelib1.inc`` gates remain.
- **EXTENDED**: Emit custom ``gate`` declarations for unsupported
  gates by decomposing them into primitives and wrapping them in
  named ``gate`` blocks (with formal parameters when the gate has
  parameters).

Angles are written with Python's shortest round-trip float representation,
so exporting and re-importing a circuit reproduces every angle bit-exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quspect.gates.gate import Gate
from quspect.ir.errors import ParameterizedGateError, UnsupportedGateError

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction


# =============================================================================
# Gate registry — QuSpect gate name → OpenQASM 2.0 gate name
# =============================================================================

# Direct mapping for all qelib1.inc-supported gates.
_QASM2_GATE_MAP: dict[str, str] = {
    # Standard single-qubit
    "id": "id",
    "h": "h",
    "x": "x",
    "y": "y",
    "z": "z",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "sx": "sx",
    "sxdg": "sxdg",
    # Standard two-qubit
    "cx": "cx",
    "cz": "cz",
    "swap": "swap",
    # Standard three-qubit
    "ccx": "ccx",
    # Parameterized single-qubit
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "p": "u1",  # P gate → U1 (same matrix)
    "u1": "u1",
    "u2": "u2",
    "u3": "u3",
    # Parameterized two-qubit
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
    "cp": "cu1",
    "cu3": "cu3",
}


def _get_qasm_gate_name(gate_name: str) -> str | None:
    """Return the QASM gate name for a given QuSpect gate name.

    Args:
        gate_name: The QuSpect gate name (e.g. ``"h"``, ``"cx"``, ``"rx"``).

    Returns:
        The corresponding QASM gate name, or ``None`` if there is no
        known direct mapping.
    """
    return _QASM2_GATE_MAP.get(gate_name)


# =============================================================================
# Export mode enum
# =============================================================================


class QASM2ExportMode(Enum):
    """Export strictness mode for OpenQASM 2.0.

    Attributes:
        STRICT: Only ``qelib1.inc`` gates; parameterized gates must be
            fully numeric (no symbolic :class:`~quspect.circuit.parameter.Parameter`).
        DECOMPOSE: Automatically decompose unsupported gates into
            supported primitives before exporting.  Decomposition rules are
            applied repeatedly (up to :data:`MAX_DECOMPOSE_PASSES` passes)
            until a fixed point is reached, so rules that produce other
            custom gates are fully lowered.  The output contains only
            ``qelib1.inc`` gates.
        EXTENDED: Automatically declare unsupported gates as named QASM
            ``gate`` blocks using their decompositions.  The circuit body
            uses the declared gate names, keeping the output structured.
            Parameterized gates are declared with formal parameters
            (``gate g(p0) a,b { rz(0.5*p0) b; }``) when the decomposition
            rule is affine in the parameters; otherwise one parameter-free
            declaration (``g_0``, ``g_1``, ...) is emitted per distinct
            parameter tuple.  Custom gates produced by a rule are declared
            recursively.  Any sub-gate that can be neither exported nor
            decomposed raises :exc:`UnsupportedGateError`.
    """

    STRICT = auto()
    DECOMPOSE = auto()
    EXTENDED = auto()


# =============================================================================
# Export configuration
# =============================================================================


@dataclass
class QASM2ExportConfig:
    """Configuration for OpenQASM 2.0 export.

    Attributes:
        mode: Export strictness mode.  Defaults to STRICT.
        include_qelib: If ``True``, emit ``include "qelib1.inc";``
            in the header.
        header_comment: Optional comment prepended to the output as
            QASM comments (``// ...``).
        indent: Indentation string used before each gate statement.
        fold_constants: If ``True``, evaluate constant numeric
            expressions before emitting (e.g. ``π/2`` → ``1.5708...``).
        barrier_as_comment: If ``True``, emit barriers as QASM
            comments rather than ``barrier`` statements.  Some backends
            do not support ``barrier``.

    Example:
        >>> config = QASM2ExportConfig(
        ...     mode=QASM2ExportMode.STRICT,
        ...     header_comment="Generated by QuSpect",
        ... )
        >>> exporter = QASM2Exporter(config)
    """

    mode: QASM2ExportMode = QASM2ExportMode.STRICT
    include_qelib: bool = True
    header_comment: str = ""
    indent: str = ""
    fold_constants: bool = True
    barrier_as_comment: bool = False


# =============================================================================
# Exporter
# =============================================================================


class QASM2Exporter:
    """Export a QuantumCircuit to OpenQASM 2.0 text.

    This is the primary entry point for generating QASM 2.0 output
    from a QuSpect circuit.

    The exporter walks the circuit instruction list and emits the
    corresponding QASM statements.  Standard gates are emitted directly;
    parameterized gates require all parameters to be concrete numeric
    values — symbolic :class:`~quspect.circuit.parameter.Parameter`
    instances cause a :exc:`ParameterizedGateError`.

    Gates not recognized in the QASM registry cause an
    :exc:`UnsupportedGateError` in STRICT mode.  EXTENDED and DECOMPOSE
    modes provide alternate handling (Phase 2).

    Example:
        >>> from quspect.ir import QASM2Exporter
        >>> circ = QuantumCircuit()
        >>> circ.h(0).cx(0, 1)
        >>> exporter = QASM2Exporter()
        >>> print(exporter.export(circ))
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0],q[1];
    """

    def __init__(self, config: QASM2ExportConfig | None = None) -> None:
        """Initialize the exporter with optional configuration.

        Args:
            config: Export configuration.  Defaults to STRICT mode with
                ``qelib1.inc`` included and no extra comment.
        """
        self._config = config if config is not None else QASM2ExportConfig()
        # Custom gates declared in EXTENDED mode, keyed by
        # (gate name, num_qubits, num_params).
        self._declared_gates: dict[tuple[str, int, int], _Declaration] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, circuit: QuantumCircuit) -> str:
        """Export a circuit to an OpenQASM 2.0 string.

        Args:
            circuit: The quantum circuit to export.

        Returns:
            An OpenQASM 2.0 string.

        Raises:
            ValueError: If the circuit is empty (has no instructions).
            ParameterizedGateError: If the circuit contains symbolic
                parameters that cannot be resolved.
            UnsupportedGateError: If the circuit contains gates not
                supported by OpenQASM 2.0 and (in DECOMPOSE / EXTENDED
                mode) not reducible to supported gates through the
                registered decomposition rules.
            ValueError: If a gate angle is not finite.

        Example:
            >>> circ = QuantumCircuit()
            >>> circ.rx(0.1, 0)
            >>> print(QASM2Exporter().export(circ).splitlines()[-1])
            rx(0.1) q[0];
        """
        if len(circuit) == 0:
            raise ValueError("Cannot export an empty circuit to OpenQASM 2.0.")

        # Pre-process circuit for DECOMPOSE / EXTENDED modes
        processed = circuit
        gate_declarations: list[str] = []

        self._declared_gates = {}
        if self._config.mode == QASM2ExportMode.DECOMPOSE:
            processed = self._decompose_circuit(circuit)
        elif self._config.mode == QASM2ExportMode.EXTENDED:
            builder = _DeclarationBuilder(self._config.mode)
            builder.declare_circuit(circuit)
            gate_declarations = builder.declarations
            self._declared_gates = builder.declared

        lines: list[str] = []

        # Header
        self._emit_header(processed, lines)

        # Gate declarations (EXTENDED mode)
        for decl in gate_declarations:
            lines.append(decl)

        if gate_declarations:
            lines.append("")

        # Registers
        self._emit_registers(processed, lines)

        # Blank line
        lines.append("")

        # Instructions
        self._emit_instructions(processed, lines)

        return "\n".join(lines)

    def export_to_file(self, circuit: QuantumCircuit, path: str | Path) -> None:
        """Export a circuit to an OpenQASM 2.0 file.

        Args:
            circuit: The quantum circuit to export.
            path: Output file path (typically with ``.qasm`` extension).

        Raises:
            ValueError: If the circuit is empty.
            ParameterizedGateError: If the circuit contains symbolic parameters.
            UnsupportedGateError: If the circuit contains unsupported gates.
            OSError: If the file cannot be written.
        """
        qasm_str = self.export(circuit)
        path = Path(path)
        path.write_text(qasm_str + "\n")

    # ------------------------------------------------------------------
    # Internal — header and registers
    # ------------------------------------------------------------------

    def _emit_header(self, circuit: QuantumCircuit, lines: list[str]) -> None:
        """Emit the OPENQASM declaration and optional header comment.

        Args:
            circuit: The circuit (used for comment content only).
            lines: Output line buffer.
        """
        if self._config.header_comment:
            for comment_line in self._config.header_comment.split("\n"):
                lines.append(f"// {comment_line}")
        lines.append("OPENQASM 2.0;")
        if self._config.include_qelib:
            lines.append('include "qelib1.inc";')

    def _emit_registers(self, circuit: QuantumCircuit, lines: list[str]) -> None:
        """Emit ``qreg`` and ``creg`` declarations.

        Args:
            circuit: The circuit whose qubit/clbit counts determine sizes.
            lines: Output line buffer.
        """
        n_q = circuit.num_qubits
        n_c = circuit.num_clbits

        if n_q == 0:
            # Fallback for edge case: no qubit used but we need at least one
            # (shouldn't happen because empty circuit is caught earlier)
            n_q = 1
        lines.append(f"qreg q[{n_q}];")
        if n_c > 0:
            lines.append(f"creg c[{n_c}];")

    # ------------------------------------------------------------------
    # Internal — instructions
    # ------------------------------------------------------------------

    def _emit_instructions(self, circuit: QuantumCircuit, lines: list[str]) -> None:
        """Emit all circuit instructions as QASM statements.

        Args:
            circuit: The quantum circuit.
            lines: Output line buffer.
        """
        from quspect.circuit.instruction import InstructionType

        for instr in circuit:
            indent = self._config.indent
            if instr.type == InstructionType.GATE:
                line = self._emit_gate_instruction(instr)
            elif instr.type == InstructionType.MEASURE:
                line = self._emit_measure_instruction(instr)
            elif instr.type == InstructionType.BARRIER:
                line = self._emit_barrier_instruction(instr)
            else:
                # Unknown instruction type — skip with a comment
                line = f"// unknown instruction type: {instr.type}"
            lines.append(f"{indent}{line}")

    # ------------------------------------------------------------------
    # Internal — individual instruction emitters
    # ------------------------------------------------------------------

    def _emit_gate_instruction(self, instr: CircuitInstruction) -> str:
        """Emit a single gate instruction.

        Args:
            instr: A GATE-type instruction.

        Returns:
            A QASM statement string (e.g. ``"h q[0];"``).

        Raises:
            ParameterizedGateError: If the gate has symbolic parameters.
            UnsupportedGateError: If the gate is not in the QASM registry.
        """
        from quspect.circuit.instruction import InstructionType

        assert instr.type == InstructionType.GATE
        assert instr.gate is not None

        gate: Gate = instr.gate
        qubits: tuple[int, ...] = instr.qubits

        # Look up the QASM gate name
        qasm_name = _get_qasm_gate_name(gate.name)
        if qasm_name is None:
            # In EXTENDED mode, the gate may have been declared
            # in a gate declaration block
            decl = self._declared_gates.get(_gate_key(gate))
            if decl is not None:
                self._check_parameters(gate)
                qubits_str = ",".join(f"q[{q}]" for q in qubits)
                return f"{decl.call_head(gate.params)} {qubits_str};"
            else:
                raise UnsupportedGateError(
                    gate.name,
                    f"OpenQASM 2.0 ({self._config.mode.name} mode)",
                    suggestion=(
                        "No decomposition rule is available for this gate. "
                        "Register a custom GateDecompositionRule in "
                        "quspect.ir.compilation."
                    ),
                )

        # Build the gate call
        if gate.params:
            # Parameterized gate — check for symbolic parameters
            self._check_parameters(gate)
            params_str = ",".join(_format_float(p) for p in gate.params)
            qubits_str = ",".join(f"q[{q}]" for q in qubits)
            return f"{qasm_name}({params_str}) {qubits_str};"
        else:
            # Standard gate
            qubits_str = ",".join(f"q[{q}]" for q in qubits)
            return f"{qasm_name} {qubits_str};"

    def _emit_measure_instruction(self, instr: CircuitInstruction) -> str:
        """Emit a measurement instruction.

        Args:
            instr: A MEASURE-type instruction.

        Returns:
            A QASM ``measure`` statement.
        """
        parts: list[str] = []
        for q, c in zip(instr.qubits, instr.clbits, strict=True):
            parts.append(f"measure q[{q}] -> c[{c}];")
        # Each measure on its own line if there are multiple
        return "\n".join(
            f"{self._config.indent}{p}" if i > 0 else p for i, p in enumerate(parts)
        )

    def _emit_barrier_instruction(self, instr: CircuitInstruction) -> str:
        """Emit a barrier instruction.

        Args:
            instr: A BARRIER-type instruction.

        Returns:
            A QASM ``barrier`` statement, or a comment if
            ``barrier_as_comment`` is configured.
        """
        qubits_str = ",".join(f"q[{q}]" for q in instr.qubits)
        if self._config.barrier_as_comment:
            return f"// barrier {qubits_str};"
        return f"barrier {qubits_str};"

    # ------------------------------------------------------------------
    # Internal — decomposition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decompose_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
        """Decompose unsupported gates repeatedly until a fixed point.

        Args:
            circuit: The original circuit.

        Returns:
            A new circuit in which every gate that can be lowered through
            the registered rules has been replaced by supported primitives.
            Gates without any applicable rule are kept (the emitter then
            raises :exc:`UnsupportedGateError` for them).

        Raises:
            UnsupportedGateError: If decomposition does not reach a fixed
                point within :data:`MAX_DECOMPOSE_PASSES` passes (e.g. the
                rules are cyclic).
        """
        from quspect.ir.compilation import _unsupported_gates, decompose_deep

        supported = get_supported_gates()
        try:
            return decompose_deep(
                circuit, supported_gates=supported, max_passes=MAX_DECOMPOSE_PASSES
            )
        except RecursionError as err:
            remaining = _unsupported_gates(circuit, supported)
            raise UnsupportedGateError(
                remaining[0] if remaining else "<unknown>",
                f"OpenQASM 2.0 ({QASM2ExportMode.DECOMPOSE.name} mode)",
                suggestion=(
                    f"Decomposition did not converge within "
                    f"{MAX_DECOMPOSE_PASSES} passes (cyclic rules?). {err}"
                ),
            ) from err

    # ------------------------------------------------------------------
    # Internal — parameter checking
    # ------------------------------------------------------------------

    def _check_parameters(self, gate: Gate) -> None:
        """Raise :exc:`ParameterizedGateError` if the gate has symbolic parameters.

        Args:
            gate: The gate to inspect.

        Raises:
            ParameterizedGateError: If any parameter is a :class:`Parameter`
                instead of a concrete float.
        """
        _check_symbolic_parameters(gate)


# =============================================================================
# Internal helpers
# =============================================================================


def _format_float(value: float | object) -> str:
    """Format a float value for QASM output.

    Uses Python's shortest round-trip representation (``repr``), so the
    text parses back to exactly the same float: ``0.5``, ``-0.25``,
    ``1e-12``, ``1.5707963267948966``.  Integral values drop the trailing
    ``.0`` (``3`` instead of ``3.0``) and zero is written as ``0``.

    Args:
        value: A float (or int / numpy scalar, which is cast).

    Returns:
        A string suitable for embedding in a QASM gate call.

    Raises:
        ValueError: If ``value`` is NaN or infinite (OpenQASM 2.0 has no
            literal for these).
    """
    f = float(value)  # type: ignore[arg-type]
    if not math.isfinite(f):
        raise ValueError(f"Cannot export non-finite angle {f!r} to OpenQASM 2.0.")
    if f == 0.0:
        return "0"
    text = repr(f)
    if text.endswith(".0"):
        text = text[:-2]
    return text


# =============================================================================
# EXTENDED mode — custom gate declarations
# =============================================================================

MAX_DECOMPOSE_PASSES: int = 32
"""Upper bound on decomposition passes (DECOMPOSE) / nesting depth (EXTENDED)."""

_QUBIT_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Names a declared gate must not take: OpenQASM keywords, builtin gates,
# expression functions and every gate defined by ``qelib1.inc``.
_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "OPENQASM", "include", "qreg", "creg", "gate", "opaque", "barrier",
        "measure", "reset", "if", "pi", "sin", "cos", "tan", "exp", "ln", "sqrt",
        "U", "CX", "u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z",
        "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "sx", "sxdg", "cz", "cy",
        "swap", "ch", "ccx", "cswap", "crx", "cry", "crz", "cu1", "cp", "cu3",
        "csx", "cu", "rxx", "rzz", "rccx", "rc3x", "c3x", "c3sqrtx", "c4x",
    }
)  # fmt: skip


def _gate_key(gate: Gate) -> tuple[str, int, int]:
    """Identify a custom gate by ``(name, num_qubits, num_params)``."""
    return (gate.name, gate.num_qubits, len(gate.params))


def _check_symbolic_parameters(gate: Gate) -> None:
    """Raise :exc:`ParameterizedGateError` if ``gate`` has unbound parameters."""
    from quspect.circuit.parameter import Parameter, ParameterExpression

    symbolic: list[str] = []
    for p in gate.params:
        if isinstance(p, Parameter):
            symbolic.append(p.name)
        elif isinstance(p, ParameterExpression):
            symbolic.extend(parameter.name for parameter in p.parameters)
    if symbolic:
        raise ParameterizedGateError(gate_name=gate.name, param_names=symbolic)


@dataclass
class _Declaration:
    """How a custom gate is declared in EXTENDED mode.

    Attributes:
        name: QASM name of the (symbolic) declaration.
        symbolic: ``True`` when a single declaration with formal parameters
            covers every instance; ``False`` when one parameter-free
            declaration is emitted per distinct parameter tuple.
        instances: For non-symbolic declarations, parameter tuple -> QASM
            name of the matching declaration.
    """

    name: str
    symbolic: bool
    instances: dict[tuple[float, ...], str] = field(default_factory=dict)

    def call_head(self, params: tuple[Any, ...]) -> str:
        """Return ``name`` or ``name(args)`` for a call with ``params``."""
        if not self.symbolic:
            return self.instances[tuple(float(p) for p in params)]
        if not params:
            return self.name
        return f"{self.name}({','.join(_format_float(p) for p in params)})"


class _DeclarationBuilder:
    """Build the QASM ``gate`` blocks needed by EXTENDED mode."""

    def __init__(self, mode: QASM2ExportMode) -> None:
        self._format_name = f"OpenQASM 2.0 ({mode.name} mode)"
        self._supported = get_supported_gates()
        self._used_names: set[str] = set(_RESERVED_NAMES)
        self._in_progress: set[tuple[str, int, int]] = set()
        self.declarations: list[str] = []
        self.declared: dict[tuple[str, int, int], _Declaration] = {}

    # -- entry point ---------------------------------------------------

    def declare_circuit(self, circuit: QuantumCircuit) -> None:
        """Declare every unsupported gate used by ``circuit``."""
        from quspect.circuit.instruction import InstructionType

        groups: dict[tuple[str, int, int], tuple[Gate, list[tuple[float, ...]]]] = {}
        for instr in circuit:
            gate = instr.gate
            if instr.type != InstructionType.GATE or gate is None:
                continue
            if gate.name in self._supported:
                continue
            _check_symbolic_parameters(gate)
            key = _gate_key(gate)
            if key not in groups:
                groups[key] = (gate, [])
            params = tuple(float(p) for p in gate.params)
            if params not in groups[key][1]:
                groups[key][1].append(params)
        for gate, instances in groups.values():
            self._declare(gate, instances, depth=0)

    # -- helpers -------------------------------------------------------

    def _unsupported(self, name: str, detail: str) -> UnsupportedGateError:
        return UnsupportedGateError(name, self._format_name, suggestion=detail)

    def _unique_name(self, base: str) -> str:
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in base)
        if not name or not ("a" <= name[0] <= "z"):
            name = f"g_{name}"
        candidate, k = name, 1
        while candidate in self._used_names:
            candidate = f"{name}_{k}"
            k += 1
        self._used_names.add(candidate)
        return candidate

    @staticmethod
    def _qubit_names(num_qubits: int) -> list[str]:
        if num_qubits <= len(_QUBIT_LETTERS):
            return list(_QUBIT_LETTERS[:num_qubits])
        return [f"q{i}" for i in range(num_qubits)]

    def _decompose(
        self, gate: Gate, parent: str | None
    ) -> list[tuple[Gate, tuple[int, ...]]]:
        from quspect.ir.compilation import decompose_gate

        try:
            return decompose_gate(gate, tuple(range(gate.num_qubits)))
        except ValueError as err:
            where = f" (used inside custom gate {parent!r})" if parent else ""
            raise self._unsupported(
                gate.name,
                f"Gate {gate.name!r}{where} has no QASM 2.0 equivalent and no "
                "applicable decomposition rule. Register a GateDecompositionRule "
                "in quspect.ir.compilation.",
            ) from err

    def _emit(self, header: str, body: list[str]) -> None:
        self.declarations.append("\n".join([header, *body, "}"]))

    # -- declaration ---------------------------------------------------

    def _declare(
        self,
        gate: Gate,
        instances: list[tuple[float, ...]],
        depth: int,
        parent: str | None = None,
    ) -> _Declaration:
        key = _gate_key(gate)
        existing = self.declared.get(key)
        if existing is not None:
            if not existing.symbolic:
                self._emit_instances(existing, gate, instances, depth)
            return existing
        if depth >= MAX_DECOMPOSE_PASSES or key in self._in_progress:
            raise self._unsupported(
                gate.name,
                f"Decomposition of {gate.name!r} is cyclic or nested deeper "
                f"than {MAX_DECOMPOSE_PASSES} levels.",
            )
        # Fail early (with a clear message) if the gate has no rule at all.
        self._decompose(
            Gate(name=gate.name, num_qubits=gate.num_qubits, params=instances[0]),
            parent,
        )
        self._in_progress.add(key)
        try:
            decl = self._try_symbolic(gate, instances, depth)
            if decl is None:
                decl = _Declaration(gate.name, symbolic=False)
                self._emit_instances(decl, gate, instances, depth)
        finally:
            self._in_progress.discard(key)
        self.declared[key] = decl
        return decl

    def _try_symbolic(
        self, gate: Gate, instances: list[tuple[float, ...]], depth: int
    ) -> _Declaration | None:
        """Declare ``gate`` once with formal parameters, if the rule allows it.

        The rule is evaluated on symbolic :class:`Parameter` placeholders;
        the result is accepted only if it is affine in the placeholders and
        reproduces the concrete decomposition of every instance.
        """
        from quspect.circuit.parameter import Parameter

        num_params = len(gate.params)
        formals = [Parameter(f"_qasm2_formal_{i}") for i in range(num_params)]
        formal_names = {f: f"p{i}" for i, f in enumerate(formals)}
        template = Gate(
            name=gate.name, num_qubits=gate.num_qubits, params=tuple(formals)
        )
        try:
            symbolic = self._decompose_symbolic(template)
        except Exception:  # noqa: BLE001 - any failure means "not symbolic"
            return None
        concrete = [
            self._decompose(
                Gate(name=gate.name, num_qubits=gate.num_qubits, params=inst),
                gate.name,
            )
            for inst in instances
        ]
        for inst, conc in zip(instances, concrete, strict=True):
            if not _same_decomposition(
                symbolic, conc, dict(zip(formals, inst, strict=True))
            ):
                return None

        qnames = self._qubit_names(gate.num_qubits)
        body: list[str] = []
        for idx, (sub, sub_qubits) in enumerate(symbolic):
            try:
                args = [_format_symbolic(v, formal_names) for v in sub.params]
            except (KeyError, TypeError, ValueError):
                return None
            qasm_name = _get_qasm_gate_name(sub.name)
            if qasm_name is None:
                nested = self._declare(
                    sub, [tuple(float(v) for v in c[idx][0].params) for c in concrete],
                    depth + 1, gate.name,
                )  # fmt: skip
                if nested.symbolic:
                    qasm_name = nested.name
                elif all(_is_constant(v) for v in sub.params):
                    qasm_name, args = nested.call_head(sub.params), []
                else:
                    return None
            head = f"{qasm_name}({','.join(args)})" if args else qasm_name
            qargs = ",".join(qnames[q] for q in sub_qubits)
            body.append(f"  {head} {qargs};")

        name = self._unique_name(gate.name)
        plist = f"({','.join(formal_names.values())})" if num_params else ""
        self._emit(f"gate {name}{plist} {','.join(qnames)} {{", body)
        return _Declaration(name, symbolic=True)

    @staticmethod
    def _decompose_symbolic(template: Gate) -> list[tuple[Gate, tuple[int, ...]]]:
        from quspect.ir.compilation import decompose_gate

        return decompose_gate(template, tuple(range(template.num_qubits)))

    def _emit_instances(
        self,
        decl: _Declaration,
        gate: Gate,
        instances: list[tuple[float, ...]],
        depth: int,
    ) -> None:
        """Emit one parameter-free declaration per new parameter tuple."""
        qnames = self._qubit_names(gate.num_qubits)
        for inst in instances:
            if inst in decl.instances:
                continue
            concrete = self._decompose(
                Gate(name=gate.name, num_qubits=gate.num_qubits, params=inst),
                gate.name,
            )
            body: list[str] = []
            for sub, sub_qubits in concrete:
                _check_symbolic_parameters(sub)
                qasm_name = _get_qasm_gate_name(sub.name)
                if qasm_name is None:
                    nested = self._declare(
                        sub, [tuple(float(v) for v in sub.params)], depth + 1,
                        gate.name,
                    )  # fmt: skip
                    head = nested.call_head(sub.params)
                elif sub.params:
                    args = ",".join(_format_float(v) for v in sub.params)
                    head = f"{qasm_name}({args})"
                else:
                    head = qasm_name
                body.append(f"  {head} {','.join(qnames[q] for q in sub_qubits)};")
            base = f"{gate.name}_{len(decl.instances)}" if inst else gate.name
            name = self._unique_name(base)
            self._emit(f"gate {name} {','.join(qnames)} {{", body)
            decl.instances[inst] = name


def _is_constant(value: object) -> bool:
    """Whether a (possibly symbolic) parameter value is a plain number."""
    from quspect.circuit.parameter import Parameter, ParameterExpression

    if isinstance(value, Parameter):
        return False
    if isinstance(value, ParameterExpression):
        return value.is_constant
    return True


def _evaluate(value: object, binding: dict[Any, float]) -> float:
    """Evaluate a numeric or symbolic parameter under ``binding``."""
    from quspect.circuit.parameter import Parameter, ParameterExpression

    if isinstance(value, Parameter):
        return binding[value]
    if isinstance(value, ParameterExpression):
        return value.evaluate(binding)
    return float(value)  # type: ignore[arg-type]


def _same_decomposition(
    symbolic: list[tuple[Gate, tuple[int, ...]]],
    concrete: list[tuple[Gate, tuple[int, ...]]],
    binding: dict[Any, float],
) -> bool:
    """Check that a symbolic decomposition, once bound, equals a concrete one."""
    if len(symbolic) != len(concrete):
        return False
    for (sg, sq), (cg, cq) in zip(symbolic, concrete, strict=True):
        if (sg.name, sg.num_qubits, tuple(sq)) != (cg.name, cg.num_qubits, tuple(cq)):
            return False
        if len(sg.params) != len(cg.params):
            return False
        for sv, cv in zip(sg.params, cg.params, strict=True):
            try:
                if not math.isclose(
                    _evaluate(sv, binding), float(cv), rel_tol=1e-12, abs_tol=1e-12
                ):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    return True


def _format_symbolic(value: object, names: dict[Any, str]) -> str:
    """Format a numeric or affine symbolic parameter as a QASM expression."""
    from quspect.circuit.parameter import Parameter, ParameterExpression

    if isinstance(value, Parameter):
        return names[value]
    if not isinstance(value, ParameterExpression):
        return _format_float(value)
    text = ""
    for parameter, coeff in value.terms.items():
        name = names[parameter]
        mag = abs(coeff)
        term = name if mag == 1.0 else f"{_format_float(mag)}*{name}"
        if not text:
            text = term if coeff > 0 else f"-{term}"
        else:
            text += f"+{term}" if coeff > 0 else f"-{term}"
    if value.constant != 0.0 or not text:
        const = _format_float(abs(value.constant))
        if not text:
            text = _format_float(value.constant)
        else:
            text += f"+{const}" if value.constant > 0 else f"-{const}"
    return text


# =============================================================================
# Public utility
# =============================================================================


def get_supported_gates() -> set[str]:
    """Return the set of QuSpect gate names with direct QASM 2.0 mappings.

    Returns:
        A set of gate name strings (e.g. ``{"h", "cx", "rx", ...}``).
    """
    return set(_QASM2_GATE_MAP.keys())
