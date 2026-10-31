"""Exception classes for the IR module.

These exceptions provide clear error messages with actionable hints
for common issues encountered during circuit serialization/deserialization.
"""

from __future__ import annotations


class IRSerializationError(Exception):
    """Base class for all IR serialization errors."""


class IRDeserializationError(Exception):
    """Base class for all IR deserialization errors."""


class ParameterizedGateError(IRSerializationError):
    """Raised when exporting a circuit with symbolic parameters to a format
    that requires concrete numeric values.

    OpenQASM does not support symbolic parameters — all gate arguments
    must be numeric literals.  This error tells the user which gates
    are problematic and suggests how to fix them.

    Attributes:
        gate_name: Name of the problematic gate.
        param_names: Names of the symbolic parameters found.
        hint: Suggested workaround.

    Example:
        >>> try:
        ...     exporter.export(circ)
        ... except ParameterizedGateError as e:
        ...     print(e.hint)
        Call circuit.bind_parameters(...) before exporting to OpenQASM,
        or use QSAL format instead.
    """

    def __init__(
        self,
        gate_name: str,
        param_names: list[str],
        *,
        hint: str = "",
    ) -> None:
        self.gate_name = gate_name
        self.param_names = param_names
        self.hint = hint or (
            f"Gate {gate_name!r} has unbound parameters {param_names}. "
            f"Call circuit.bind_parameters(...) before exporting to OpenQASM, "
            f"or use QSAL format instead."
        )
        super().__init__(str(self))

    def __str__(self) -> str:
        return self.hint


class UnsupportedGateError(IRSerializationError):
    """Raised when a gate cannot be represented in the target format.

    Attributes:
        gate_name: Name of the unsupported gate.
        format_name: Target format (e.g. ``"OpenQASM 2.0"``).
        suggestion: Suggested alternative or workaround.

    Example:
        >>> try:
        ...     exporter.export(circ)
        ... except UnsupportedGateError as e:
        ...     print(e.suggestion)
        Try mode=DECOMPOSE to auto-decompose, or mode=EXTENDED to emit
        a custom gate declaration.
    """

    def __init__(
        self,
        gate_name: str,
        format_name: str,
        *,
        suggestion: str = "",
    ) -> None:
        self.gate_name = gate_name
        self.format_name = format_name
        self.suggestion = suggestion or (
            f"Gate {gate_name!r} is not supported by {format_name}. "
            f"Try mode=DECOMPOSE to auto-decompose, or mode=EXTENDED "
            f"to emit a custom gate declaration."
        )
        super().__init__(str(self))

    def __str__(self) -> str:
        return self.suggestion


class QSALVersionError(IRDeserializationError):
    """Raised when loading a QSAL document with an incompatible version.

    Attributes:
        found_version: The version found in the document.
        supported_versions: Versions known to be loadable.
    """

    def __init__(
        self,
        found_version: str,
        supported_versions: list[str],
    ) -> None:
        self.found_version = found_version
        self.supported_versions = supported_versions
        msg = (
            f"QSAL document version {found_version!r} is not supported. "
            f"Supported versions: {supported_versions}"
        )
        super().__init__(msg)
