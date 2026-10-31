"""QSAL — QuSpect Abstract Language.

QSAL is QuSpect's native intermediate representation. Unlike OpenQASM,
QSAL preserves the full semantics of variational quantum circuits,
including symbolic parameters, observable definitions, and execution
configuration.  It is serialized as JSON for readability and portability.

A QSAL document is a self-contained description of a quantum program:

- Circuit (with symbolic parameters preserved)
- Observables (as :class:`QubitOperator` instances)
- Parameter metadata
- Execution configuration
- Document metadata
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from quspect.ir.errors import QSALVersionError

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.operators.qubit_operator import QubitOperator

# Current QSAL version — all serialized documents carry this.
CURRENT_QSAL_VERSION = "0.1.0"

# All versions known to be loadable.
_SUPPORTED_VERSIONS = ["0.1.0"]


# =============================================================================
# QSALDocument
# =============================================================================


@dataclass
class QSALDocument:
    """The top-level QSAL document.

    QSAL (QuSpect Abstract Language) is QuSpect's native intermediate
    representation.  A document bundles a quantum circuit together with
    observables, parameters, execution configuration, and metadata into
    a single JSON-serializable unit.

    Example:
        >>> from quspect.ir import QSALDocument
        >>> doc = QSALDocument(
        ...     name="vqe_h2",
        ...     circuit=circuit,
        ...     observables=[("energy", hamiltonian)],
        ... )
        >>> json_str = doc.to_json()
        >>> doc2 = QSALDocument.from_json(json_str)
        >>> doc2.name
        'vqe_h2'
    """

    qsal_version: str = CURRENT_QSAL_VERSION
    name: str = ""
    circuit: QuantumCircuit | None = None
    observables: list[tuple[str, QubitOperator]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize this QSAL document to a JSON string.

        Args:
            indent: JSON indentation.  ``None`` for compact output.

        Returns:
            A JSON string.
        """
        data: dict[str, Any] = {
            "qsal_version": self.qsal_version,
            "name": self.name,
        }

        # Circuit
        if self.circuit is not None:
            data["circuit"] = _serialize_circuit(self.circuit)
        else:
            data["circuit"] = None

        # Observables
        data["observables"] = [
            {"name": name, "operator": _serialize_operator(op)}
            for name, op in self.observables
        ]

        # Parameters, config, metadata
        data["parameters"] = self.parameters
        data["config"] = self.config
        data["metadata"] = self.metadata

        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> QSALDocument:
        """Deserialize a QSAL document from a JSON string.

        Args:
            data: A JSON string produced by :meth:`to_json`.

        Returns:
            A new :class:`QSALDocument`.

        Raises:
            QSALVersionError: If the version is incompatible.
            ValueError: If the JSON is malformed.
        """
        raw: dict[str, Any] = json.loads(data)

        version = raw.get("qsal_version", "unknown")
        if version not in _SUPPORTED_VERSIONS:
            raise QSALVersionError(version, _SUPPORTED_VERSIONS)

        doc = cls(
            qsal_version=version,
            name=raw.get("name", ""),
        )

        # Circuit
        if raw.get("circuit") is not None:
            doc.circuit = _deserialize_circuit(raw["circuit"])

        # Observables
        for obs_data in raw.get("observables", []):
            op = _deserialize_operator(obs_data["operator"])
            doc.observables.append((obs_data["name"], op))

        doc.parameters = raw.get("parameters", {})
        doc.config = raw.get("config", {})
        doc.metadata = raw.get("metadata", {})

        return doc

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def to_file(self, path: str | Path) -> None:
        """Write this QSAL document to a file (``.qsal.json``).

        Args:
            path: Output file path.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.write_text(self.to_json() + "\n")

    @classmethod
    def from_file(cls, path: str | Path) -> QSALDocument:
        """Read a QSAL document from a file.

        Args:
            path: Path to a ``.qsal.json`` file.

        Returns:
            A new QSALDocument.

        Raises:
            FileNotFoundError: If the file does not exist.
            QSALVersionError: If the version is incompatible.
        """
        path = Path(path)
        return cls.from_json(path.read_text())

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_observables(self) -> bool:
        """Return True if this document defines observables to measure."""
        return len(self.observables) > 0

    @property
    def is_parameterized(self) -> bool:
        """Return True if the circuit contains symbolic parameters."""
        if self.circuit is None:
            return False
        return len(self.circuit.parameters) > 0


# =============================================================================
# Module-level convenience functions
# =============================================================================


def dumps(
    circuit: QuantumCircuit,
    *,
    observables: list[tuple[str, QubitOperator]] | None = None,
    name: str = "",
    **kwargs: Any,
) -> str:
    """Serialize a QuantumCircuit (and optional observables) to QSAL JSON.

    This is the simplest entry point for most use cases:

        >>> json_str = qsal.dumps(my_circuit)
        >>> json_str = qsal.dumps(my_circuit, observables=[("H", h_op)])

    Args:
        circuit: The quantum circuit to serialize.
        observables: Optional list of ``(name, QubitOperator)`` pairs.
        name: Optional program name.
        **kwargs: Additional configuration/metadata fields.

    Returns:
        A QSAL JSON string.
    """
    doc_name = name if name else (circuit.name if circuit.name else "")
    doc = QSALDocument(
        name=doc_name,
        circuit=circuit,
        observables=observables or [],
        metadata=kwargs.pop("metadata", {}),
    )
    doc.parameters = kwargs.pop("parameters", {})
    doc.config = kwargs.pop("config", {})
    doc.metadata.update(kwargs)
    return doc.to_json()


def loads(data: str) -> QSALDocument:
    """Deserialize a QSAL JSON string to a QSALDocument.

    Args:
        data: A JSON string produced by :func:`dumps`.

    Returns:
        A QSALDocument instance.

    Raises:
        QSALVersionError: If the version is incompatible.
    """
    return QSALDocument.from_json(data)


def dump(
    circuit: QuantumCircuit,
    path: str | Path,
    *,
    observables: list[tuple[str, QubitOperator]] | None = None,
    **kwargs: Any,
) -> None:
    """Serialize a circuit to QSAL JSON and write to a file.

    Args:
        circuit: The quantum circuit.
        path: Output file path.
        observables: Optional observable list.
        **kwargs: Passed to :func:`dumps`.
    """
    json_str = dumps(circuit, observables=observables, **kwargs)
    path = Path(path)
    path.write_text(json_str + "\n")


def load(path: str | Path) -> QSALDocument:
    """Read and deserialize a QSAL document from a file.

    Args:
        path: Path to a ``.qsal.json`` file.

    Returns:
        A QSALDocument instance.
    """
    return QSALDocument.from_file(path)


# =============================================================================
# Internal — circuit serialization
# =============================================================================


def _serialize_circuit(circuit: QuantumCircuit) -> dict[str, Any]:
    """Convert a QuantumCircuit to a JSON-serializable dict.

    This is a QSAL-specific serialization that preserves parameter
    names and instruction types exactly as they appear in QuSpect.
    A non-empty ``circuit.name`` is stored under the optional ``"name"``
    key and a non-empty ``Gate.label`` under the optional gate ``"label"``
    key; documents without these keys remain valid.

    Args:
        circuit: The quantum circuit.

    Returns:
        A JSON-safe dict.
    """
    from quspect.circuit.parameter import Parameter, ParameterExpression

    instructions: list[dict[str, Any]] = []
    for instr in circuit:
        instr_data: dict[str, Any] = {
            "type": instr.type.name,
            "qubits": list(instr.qubits),
            "clbits": list(instr.clbits),
        }
        if instr.gate is not None:
            gate_data: dict[str, Any] = {
                "name": instr.gate.name,
                "num_qubits": instr.gate.num_qubits,
            }
            if instr.gate.label:
                gate_data["label"] = instr.gate.label
            if instr.gate.params:
                gate_data["params"] = [
                    {"_param": p.name}
                    if isinstance(p, Parameter)
                    else {
                        "_expr": {
                            "constant": p.constant,
                            "terms": {
                                parameter.name: coefficient
                                for parameter, coefficient in p.terms.items()
                            },
                        }
                    }
                    if isinstance(p, ParameterExpression)
                    else p
                    for p in instr.gate.params
                ]
                gate_data["gate_type"] = "parameterized"
            else:
                gate_data["gate_type"] = "standard"
            instr_data["gate"] = gate_data
        if instr.channel is not None:
            instr_data["channel"] = _serialize_channel(instr.channel)
        if instr.label:
            instr_data["label"] = instr.label
        instructions.append(instr_data)

    circuit_data: dict[str, Any] = {}
    # ``name`` is optional so that documents written by older versions (and
    # unnamed circuits) keep the exact same layout.
    if circuit.name:
        circuit_data["name"] = circuit.name
    circuit_data["num_qubits"] = circuit.num_qubits
    circuit_data["num_clbits"] = circuit.num_clbits
    circuit_data["instructions"] = instructions
    return circuit_data


def _deserialize_circuit(data: dict[str, Any]) -> QuantumCircuit:
    """Reconstruct a QuantumCircuit from a QSAL serialization dict.

    Args:
        data: A dict produced by :func:`_serialize_circuit`.

    Returns:
        A new QuantumCircuit instance.

    Raises:
        ValueError: If the data references unknown gate names.
    """
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.circuit.parameter import Parameter, ParameterExpression
    from quspect.gates.gate import Gate
    from quspect.gates.standard import ALL_STANDARD

    _standard_lookup = {g.name: g for g in ALL_STANDARD}

    name = data.get("name", "")
    circ = QuantumCircuit(name=name)

    for instr_data in data.get("instructions", []):
        itype = InstructionType[instr_data["type"]]
        qubits = tuple(instr_data["qubits"])
        clbits = tuple(instr_data.get("clbits", []))

        gate: Gate | None = None
        channel = None
        if "gate" in instr_data:
            gd = instr_data["gate"]
            if gd.get("gate_type") == "parameterized":
                params_raw = gd.get("params", [])
                params = tuple(
                    Parameter(p["_param"])
                    if isinstance(p, dict) and "_param" in p
                    else ParameterExpression(
                        p["_expr"].get("terms", {}),
                        constant=p["_expr"].get("constant", 0.0),
                    )
                    if isinstance(p, dict) and "_expr" in p
                    else float(p)
                    for p in params_raw
                )
                gate = Gate(
                    name=gd["name"],
                    num_qubits=gd["num_qubits"],
                    params=params,
                    label=gd.get("label", ""),
                )
            else:
                gate_name = gd["name"]
                if gate_name not in _standard_lookup:
                    raise ValueError(f"Unknown standard gate: {gate_name!r}")
                gate = _standard_lookup[gate_name]
                gate_label = gd.get("label", "")
                if gate_label:
                    gate = dataclasses.replace(gate, label=gate_label)

        if "channel" in instr_data:
            channel = _deserialize_channel(instr_data["channel"])

        label = instr_data.get("label", "")

        instr = CircuitInstruction(
            type=itype,
            gate=gate,
            channel=channel,
            qubits=qubits,
            clbits=clbits,
            label=label,
        )
        circ._instructions.append(instr)

    return circ


# =============================================================================
# Internal — channel serialization
# =============================================================================


def _serialize_channel(channel: Any) -> dict[str, Any]:
    """Convert a NoiseChannel to a JSON-serializable dict.

    Args:
        channel: A NoiseChannel instance.

    Returns:
        A JSON-safe dict with channel_type, num_qubits, and params.
    """
    # Store both the user-facing name and a canonical kind.  ``name`` may be
    # customized (for example ``KrausChannel(name="readout")``), while the
    # canonical kind is stable enough for native-runtime fast paths.
    from quspect.noise.channel import (
        AmplitudeDampingChannel,
        BitFlipChannel,
        BitPhaseFlipChannel,
        DepolarizingChannel,
        GeneralizedAmplitudeDampingChannel,
        KrausChannel,
        PauliChannel,
        PhaseDampingChannel,
        PhaseFlipChannel,
    )

    if isinstance(channel, PauliChannel):
        channel_kind = "pauli"
    elif isinstance(channel, BitFlipChannel):
        channel_kind = "bit_flip"
    elif isinstance(channel, PhaseFlipChannel):
        channel_kind = "phase_flip"
    elif isinstance(channel, BitPhaseFlipChannel):
        channel_kind = "bit_phase_flip"
    elif isinstance(channel, DepolarizingChannel):
        channel_kind = "depolarizing"
    elif isinstance(channel, AmplitudeDampingChannel):
        channel_kind = "amplitude_damping"
    elif isinstance(channel, PhaseDampingChannel):
        channel_kind = "phase_damping"
    elif isinstance(channel, GeneralizedAmplitudeDampingChannel):
        channel_kind = "generalized_amplitude_damping"
    elif isinstance(channel, KrausChannel):
        channel_kind = "kraus"
    else:
        raise TypeError(f"Unsupported noise channel type: {type(channel).__name__}")

    ch_data: dict[str, Any] = {
        "channel_type": channel.name,
        "channel_kind": channel_kind,
        "num_qubits": channel.num_qubits,
    }

    # Kraus matrices make QSAL self-contained and allow the C++ decoder to
    # execute built-in and user-defined channels through the same IR.  Complex
    # values are encoded as [real, imag] pairs because JSON has no complex type.
    ch_data["kraus_ops"] = [
        [[[float(value.real), float(value.imag)] for value in row] for row in op]
        for op in channel.kraus_operators()
    ]

    if isinstance(channel, PauliChannel):
        ch_data["px"] = channel.px
        ch_data["py"] = channel.py
        ch_data["pz"] = channel.pz
    elif isinstance(
        channel,
        (
            BitFlipChannel,
            PhaseFlipChannel,
            BitPhaseFlipChannel,
            DepolarizingChannel,
        ),
    ):
        ch_data["p"] = channel.p
    elif isinstance(channel, (AmplitudeDampingChannel, PhaseDampingChannel)):
        ch_data["gamma"] = channel.gamma
    elif isinstance(channel, GeneralizedAmplitudeDampingChannel):
        ch_data["gamma"] = channel.gamma
        ch_data["p_thermal"] = channel.p_thermal
    elif isinstance(channel, KrausChannel):
        ch_data["kraus_ops_count"] = len(channel.kraus_ops)

    if channel.params:
        ch_data["params"] = list(channel.params)
    return ch_data


def _deserialize_channel(data: dict[str, Any]) -> Any:
    """Reconstruct a NoiseChannel from a QSAL serialization dict.

    Args:
        data: A dict produced by :func:`_serialize_channel`.

    Returns:
        A NoiseChannel instance.

    Raises:
        ValueError: If the channel type is unknown or unsupported.
    """
    from quspect.noise.channel import (
        AmplitudeDampingChannel,
        BitFlipChannel,
        BitPhaseFlipChannel,
        DepolarizingChannel,
        GeneralizedAmplitudeDampingChannel,
        PauliChannel,
        PhaseDampingChannel,
        PhaseFlipChannel,
    )

    ch_type = data.get("channel_kind", data["channel_type"])
    channel_name = data.get("channel_type", ch_type)
    num_qubits = data["num_qubits"]

    if ch_type == "bit_flip":
        p = data.get("p", data.get("params", [0.0])[0] if data.get("params") else 0.0)
        return BitFlipChannel(name=channel_name, num_qubits=num_qubits, p=p)
    elif ch_type == "phase_flip":
        p = data.get("p", data.get("params", [0.0])[0] if data.get("params") else 0.0)
        return PhaseFlipChannel(name=channel_name, num_qubits=num_qubits, p=p)
    elif ch_type == "bit_phase_flip":
        p = data.get("p", data.get("params", [0.0])[0] if data.get("params") else 0.0)
        return BitPhaseFlipChannel(name=channel_name, num_qubits=num_qubits, p=p)
    elif ch_type == "pauli":
        px = data.get("px", 0.0)
        py = data.get("py", 0.0)
        pz = data.get("pz", 0.0)
        return PauliChannel(
            name=channel_name, num_qubits=num_qubits, px=px, py=py, pz=pz
        )
    elif ch_type == "depolarizing":
        p = data.get("p", data.get("params", [0.0])[0] if data.get("params") else 0.0)
        return DepolarizingChannel(name=channel_name, num_qubits=num_qubits, p=p)
    elif ch_type == "amplitude_damping":
        gamma = data.get(
            "gamma", data.get("params", [0.0])[0] if data.get("params") else 0.0
        )
        return AmplitudeDampingChannel(
            name=channel_name, num_qubits=num_qubits, gamma=gamma
        )
    elif ch_type == "phase_damping":
        gamma = data.get(
            "gamma", data.get("params", [0.0])[0] if data.get("params") else 0.0
        )
        return PhaseDampingChannel(
            name=channel_name, num_qubits=num_qubits, gamma=gamma
        )
    elif ch_type in ("generalized_amplitude_damping", "gen_amp_damp"):
        gamma = data.get(
            "gamma", data.get("params", [0.0])[0] if data.get("params") else 0.0
        )
        p_thermal = data.get("p_thermal", 0.0)
        return GeneralizedAmplitudeDampingChannel(
            name=channel_name,
            num_qubits=num_qubits,
            gamma=gamma,
            p_thermal=p_thermal,
        )
    elif ch_type in ("kraus",):
        from quspect.noise.channel import KrausChannel

        raw_ops = data.get("kraus_ops")
        if not isinstance(raw_ops, list) or not raw_ops:
            raise ValueError("KrausChannel QSAL data is missing kraus_ops")

        def decode_complex(value: Any) -> complex:
            if isinstance(value, (int, float)):
                return complex(float(value), 0.0)
            if isinstance(value, list) and len(value) == 2:
                return complex(float(value[0]), float(value[1]))
            raise ValueError(
                "Kraus matrix entries must be numbers or [real, imag] pairs"
            )

        kraus_ops = [
            np.array(
                [[decode_complex(value) for value in row] for row in matrix],
                dtype=complex,
            )
            for matrix in raw_ops
        ]
        return KrausChannel(
            name=channel_name, num_qubits=num_qubits, kraus_ops=kraus_ops
        )
    else:
        raise ValueError(f"Unknown channel type: {ch_type!r}")


# =============================================================================
# Internal — operator serialization
# =============================================================================


def _serialize_operator(op: QubitOperator) -> dict[str, Any]:
    """Convert a QubitOperator to a JSON-serializable dict.

    Each term is stored as ``{"paulis": [[qubit, pauli], ...],
    "coefficient": c}``.  ``c`` is a JSON number for real coefficients, a
    ``[real, imag]`` pair for coefficients with a non-zero imaginary part, a
    parameter name for a :class:`Parameter`, or an ``{"_expr": ...}`` object
    for a :class:`ParameterExpression`.

    Args:
        op: The qubit operator.

    Returns:
        A JSON-safe dict with a ``"terms"`` key.
    """
    from quspect.circuit.parameter import Parameter, ParameterExpression

    terms: list[dict[str, Any]] = []
    for term_key, coeff in op.terms.items():
        paulis: list[list[Any]] = [[qubit, pauli] for qubit, pauli in term_key]
        coeff_data: Any
        if isinstance(coeff, Parameter):
            coeff_data = coeff.name
        elif isinstance(coeff, ParameterExpression):
            coeff_data = {
                "_expr": {
                    "constant": coeff.constant,
                    "terms": {
                        parameter.name: factor
                        for parameter, factor in coeff.terms.items()
                    },
                }
            }
        else:
            # Real coefficients stay plain JSON numbers (the historical
            # format); coefficients with a non-zero imaginary part are
            # encoded as ``[real, imag]`` pairs because JSON has no complex
            # type.  ``complex(...)`` also accepts NumPy scalar types.
            value = complex(coeff)
            if value.imag == 0.0:
                coeff_data = float(value.real)
            else:
                coeff_data = [float(value.real), float(value.imag)]
        terms.append({"paulis": paulis, "coefficient": coeff_data})

    return {"terms": terms}


def _deserialize_operator(data: dict[str, Any]) -> QubitOperator:
    """Reconstruct a QubitOperator from a QSAL serialization dict.

    Both plain numeric coefficients (the original format) and
    ``[real, imag]`` pairs are accepted.

    Args:
        data: A dict produced by :func:`_serialize_operator`.

    Returns:
        A new QubitOperator instance.

    Raises:
        ValueError: If a list-valued coefficient is not a ``[real, imag]``
            pair.
    """
    from quspect.circuit.parameter import Parameter, ParameterExpression
    from quspect.operators.qubit_operator import QubitOperator

    terms: dict[tuple[tuple[int, str], ...], Any] = {}
    for term_data in data.get("terms", []):
        paulis_raw = term_data["paulis"]
        term_key = tuple((int(p[0]), str(p[1])) for p in paulis_raw)

        coeff_raw = term_data["coefficient"]
        coeff: Any
        if isinstance(coeff_raw, str):
            coeff = Parameter(coeff_raw)
        elif isinstance(coeff_raw, dict) and "_expr" in coeff_raw:
            expression = coeff_raw["_expr"]
            coeff = ParameterExpression(
                expression.get("terms", {}),
                constant=expression.get("constant", 0.0),
            )
        elif isinstance(coeff_raw, list):
            if len(coeff_raw) != 2:
                raise ValueError(
                    "Complex observable coefficients must be [real, imag] pairs, "
                    f"got {coeff_raw!r}"
                )
            coeff = complex(float(coeff_raw[0]), float(coeff_raw[1]))
        else:
            coeff = float(coeff_raw)
        terms[term_key] = coeff

    return QubitOperator(terms)
