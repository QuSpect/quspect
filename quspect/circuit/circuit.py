"""Quantum circuit construction and manipulation.

The :class:`QuantumCircuit` is the primary user-facing class for building
quantum circuits. It supports a fluent (method-chaining) API with automatic
qubit/clbit count inference.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterator, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.circuit.parameter import Parameter, ParameterExpression, ParamValue
from quspect.gates.gate import Gate
from quspect.gates.parameterized import (
    CP,
    CRX,
    CRY,
    CRZ,
    CU3,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    P,
)
from quspect.gates.standard import (
    CCX,
    CX,
    CZ,
    SDG,
    SWAP,
    SX,
    SXDG,
    TDG,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
)
from quspect.noise.channel import NoiseChannel

if TYPE_CHECKING:
    from quspect.dag.dag_circuit import DAGCircuit  # noqa: F401


class QuantumCircuit:
    """A quantum circuit composed of gates, measurements, and barriers.

    QuantumCircuit is the primary user-facing class. It supports a fluent
    (method-chaining) API for building circuits gate by gate.

    The number of qubits and classical bits is **dynamically inferred**
    from the indices used in gates and measurements. There is no need to
    pre-declare circuit size — the circuit grows automatically as new
    qubits or classical bits are referenced.

    Gates are added via named methods (e.g., ``.h(0)``) or via the generic
    ``.append(gate, qubits)`` method. Qubit and classical bit indices must
    be non-negative integers; gaps are allowed (e.g., you can use qubit 3
    without first using qubits 0-2).

    Example:
        >>> circ = QuantumCircuit(name="bell")
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> circ.measure([0, 1], [0, 1])
        >>> circ.num_qubits
        2
        >>> circ.num_clbits
        2
        >>> len(circ)
        3
    """

    def __init__(self, *, name: str = "") -> None:
        """Initialize an empty quantum circuit.

        Args:
            name: Optional human-readable identifier for this circuit.
        """
        self._name = name
        self._instructions: list[CircuitInstruction] = []

    # ------------------------------------------------------------------
    # Properties — dynamic inference of qubit/clbit counts
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits inferred from the circuit.

        Computed as ``max(qubit_index) + 1`` across all instructions.
        Returns 0 for an empty circuit.
        """
        max_q = -1
        for instr in self._instructions:
            if instr.qubits:
                instr_max = max(instr.qubits)
                if instr_max > max_q:
                    max_q = instr_max
        return max_q + 1

    @property
    def num_clbits(self) -> int:
        """Return the number of classical bits inferred from the circuit.

        Computed as ``max(clbit_index) + 1`` across all measurement
        instructions. Returns 0 if there are no measurements.
        """
        max_c = -1
        for instr in self._instructions:
            if instr.clbits:
                instr_max = max(instr.clbits)
                if instr_max > max_c:
                    max_c = instr_max
        return max_c + 1

    @property
    def name(self) -> str:
        """Return the circuit name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the circuit name."""
        self._name = value

    @property
    def instructions(self) -> tuple[CircuitInstruction, ...]:
        """Return an immutable view of all instructions in this circuit."""
        return tuple(self._instructions)

    def __len__(self) -> int:
        """Return the number of instructions."""
        return len(self._instructions)

    def __iter__(self) -> Iterator[CircuitInstruction]:
        """Iterate over instructions in order."""
        return iter(self._instructions)

    def __getitem__(self, index: int) -> CircuitInstruction:
        """Get instruction by index.

        Args:
            index: Zero-based index into the instruction list.

        Returns:
            The instruction at the given index.

        Raises:
            IndexError: If index is out of range.
        """
        return self._instructions[index]

    # ------------------------------------------------------------------
    # Active qubits / clbits
    # ------------------------------------------------------------------

    @property
    def active_qubits(self) -> set[int]:
        """Return the set of qubit indices that appear in at least one
        instruction."""
        result: set[int] = set()
        for instr in self._instructions:
            result.update(instr.qubits)
        return result

    @property
    def active_clbits(self) -> set[int]:
        """Return the set of classical bit indices that appear in at least
        one instruction."""
        result: set[int] = set()
        for instr in self._instructions:
            result.update(instr.clbits)
        return result

    # ------------------------------------------------------------------
    # Generic append
    # ------------------------------------------------------------------

    def append(
        self,
        gate: Gate,
        qubits: Sequence[int],
        clbits: Sequence[int] = (),
    ) -> QuantumCircuit:
        """Append a generic gate instruction.

        Args:
            gate: The quantum gate to apply.
            qubits: Qubit indices.
            clbits: Classical bit indices (for measurements).

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If the number of qubits doesn't match the gate.
        """
        qubits_tuple = tuple(qubits)
        clbits_tuple = tuple(clbits)
        instr = CircuitInstruction(
            type=InstructionType.GATE,
            gate=gate,
            qubits=qubits_tuple,
            clbits=clbits_tuple,
        )
        self._instructions.append(instr)
        return self

    # ------------------------------------------------------------------
    # Single-qubit standard gates (fluent API)
    # ------------------------------------------------------------------

    def id(self, qubit: int) -> QuantumCircuit:
        """Apply the Identity gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(I, (qubit,))

    def h(self, qubit: int) -> QuantumCircuit:
        """Apply the Hadamard gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(H, (qubit,))

    def x(self, qubit: int) -> QuantumCircuit:
        """Apply the Pauli-X gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(X, (qubit,))

    def y(self, qubit: int) -> QuantumCircuit:
        """Apply the Pauli-Y gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(Y, (qubit,))

    def z(self, qubit: int) -> QuantumCircuit:
        """Apply the Pauli-Z gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(Z, (qubit,))

    def s(self, qubit: int) -> QuantumCircuit:
        """Apply the S (phase π/2) gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(S, (qubit,))

    def sdg(self, qubit: int) -> QuantumCircuit:
        """Apply the S-dagger gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(SDG, (qubit,))

    def t(self, qubit: int) -> QuantumCircuit:
        """Apply the T (phase π/4) gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(T, (qubit,))

    def tdg(self, qubit: int) -> QuantumCircuit:
        """Apply the T-dagger gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(TDG, (qubit,))

    def sx(self, qubit: int) -> QuantumCircuit:
        """Apply the √X gate (``SX``) to a qubit.

        ``SX = ½[[1+i, 1-i], [1-i, 1+i]]``; ``SX · SX = X``.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.

        Example:
            >>> from quspect import QuantumCircuit
            >>> QuantumCircuit().sx(0).sx(0).instructions[0].gate.name
            'sx'
        """
        return self.append(SX, (qubit,))

    def sxdg(self, qubit: int) -> QuantumCircuit:
        """Apply the inverse √X gate (``SXdg``) to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.

        Example:
            >>> from quspect import QuantumCircuit
            >>> QuantumCircuit().sxdg(0).instructions[0].gate.name
            'sxdg'
        """
        return self.append(SXDG, (qubit,))

    # ------------------------------------------------------------------
    # Single-qubit parameterized gates (fluent API)
    # ------------------------------------------------------------------

    def rx(self, theta: ParamValue, qubit: int) -> QuantumCircuit:
        """Apply an RX rotation gate.

        Args:
            theta: Rotation angle in radians (float or Parameter).
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("rx", 1, _coerce_param(theta))
        return self.append(gate, (qubit,))

    def ry(self, theta: ParamValue, qubit: int) -> QuantumCircuit:
        """Apply an RY rotation gate.

        Args:
            theta: Rotation angle in radians (float or Parameter).
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("ry", 1, _coerce_param(theta))
        return self.append(gate, (qubit,))

    def rz(self, theta: ParamValue, qubit: int) -> QuantumCircuit:
        """Apply an RZ rotation gate.

        Args:
            theta: Rotation angle in radians (float or Parameter).
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("rz", 1, _coerce_param(theta))
        return self.append(gate, (qubit,))

    def p(self, theta: ParamValue, qubit: int) -> QuantumCircuit:
        """Apply a Phase gate.

        Args:
            theta: Phase angle in radians (float or Parameter).
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("p", 1, _coerce_param(theta))
        return self.append(gate, (qubit,))

    def u1(self, lam: ParamValue, qubit: int) -> QuantumCircuit:
        """Apply a U1 phase gate."""
        gate = _make_param_gate("u1", 1, _coerce_param(lam))
        return self.append(gate, (qubit,))

    def u2(self, phi: ParamValue, lam: ParamValue, qubit: int) -> QuantumCircuit:
        """Apply a U2 single-qubit gate."""
        gate = _make_param_gate("u2", 1, _coerce_param(phi), _coerce_param(lam))
        return self.append(gate, (qubit,))

    def u3(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        qubit: int,
    ) -> QuantumCircuit:
        """Apply a U3 universal single-qubit gate.

        Args:
            theta: First rotation angle.
            phi: Second rotation angle.
            lam: Third rotation angle.
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate(
            "u3", 1, _coerce_param(theta), _coerce_param(phi), _coerce_param(lam)
        )
        return self.append(gate, (qubit,))

    # ------------------------------------------------------------------
    # Two-qubit gates (fluent API)
    # ------------------------------------------------------------------

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Apply a CNOT (controlled-X) gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(CX, (control, target))

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Apply a CZ (controlled-Z) gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(CZ, (control, target))

    def swap(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply a SWAP gate.

        Args:
            qubit1: First qubit index.
            qubit2: Second qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(SWAP, (qubit1, qubit2))

    def crx(self, theta: ParamValue, control: int, target: int) -> QuantumCircuit:
        """Apply a CRX (controlled-RX) gate.

        Args:
            theta: Rotation angle in radians (float or Parameter).
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("crx", 2, _coerce_param(theta))
        return self.append(gate, (control, target))

    def cry(self, theta: ParamValue, control: int, target: int) -> QuantumCircuit:
        """Apply a CRY (controlled-RY) gate.

        Args:
            theta: Rotation angle in radians (float or Parameter).
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("cry", 2, _coerce_param(theta))
        return self.append(gate, (control, target))

    def crz(self, theta: ParamValue, control: int, target: int) -> QuantumCircuit:
        """Apply a CRZ (controlled-RZ) gate.

        Args:
            theta: Rotation angle in radians (float or Parameter).
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        gate = _make_param_gate("crz", 2, _coerce_param(theta))
        return self.append(gate, (control, target))

    def cp(self, theta: ParamValue, control: int, target: int) -> QuantumCircuit:
        """Apply a controlled-Phase gate."""
        gate = _make_param_gate("cp", 2, _coerce_param(theta))
        return self.append(gate, (control, target))

    def cu3(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        control: int,
        target: int,
    ) -> QuantumCircuit:
        """Apply a controlled-U3 gate."""
        gate = _make_param_gate(
            "cu3", 2, _coerce_param(theta), _coerce_param(phi), _coerce_param(lam)
        )
        return self.append(gate, (control, target))

    # ------------------------------------------------------------------
    # Three-qubit gates
    # ------------------------------------------------------------------

    def ccx(self, control1: int, control2: int, target: int) -> QuantumCircuit:
        """Apply a Toffoli (CCX) gate.

        Args:
            control1: First control qubit index.
            control2: Second control qubit index.
            target: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        return self.append(CCX, (control1, control2, target))

    # ------------------------------------------------------------------
    # Measurement and barrier
    # ------------------------------------------------------------------

    def measure(self, qubits: Sequence[int], clbits: Sequence[int]) -> QuantumCircuit:
        """Add a measurement instruction.

        Args:
            qubits: Qubit indices to measure.
            clbits: Classical bit indices to store results. Must have
                the same length as ``qubits``.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If ``len(qubits) != len(clbits)``.
        """
        qubits_tuple = tuple(qubits)
        clbits_tuple = tuple(clbits)
        instr = CircuitInstruction(
            type=InstructionType.MEASURE,
            qubits=qubits_tuple,
            clbits=clbits_tuple,
        )
        self._instructions.append(instr)
        return self

    def measure_all(self) -> QuantumCircuit:
        """Measure all currently active qubits to their corresponding
        classical bits.

        Active qubits are those that appear in at least one instruction.
        Classical bit indices match qubit indices (q0→c0, q1→c1, ...).
        For an empty circuit, this does nothing.

        Returns:
            Self, for method chaining.
        """
        qubits_list = sorted(self.active_qubits)
        if not qubits_list:
            return self
        return self.measure(qubits_list, qubits_list)

    def barrier(self, *qubits: int) -> QuantumCircuit:
        """Add a barrier to prevent optimization across the given qubits.

        If no qubits are specified, the barrier applies to all currently
        active qubits.

        Args:
            *qubits: Qubit indices to include in the barrier. If empty,
                all active qubits are used.

        Returns:
            Self, for method chaining.
        """
        qubits_tuple = qubits if qubits else tuple(sorted(self.active_qubits))
        instr = CircuitInstruction(
            type=InstructionType.BARRIER,
            qubits=qubits_tuple,
        )
        self._instructions.append(instr)
        return self

    # ------------------------------------------------------------------
    # Noise channel methods (fluent API)
    # ------------------------------------------------------------------

    def append_channel(
        self, channel: NoiseChannel, qubits: Sequence[int]
    ) -> QuantumCircuit:
        """Append a generic noise channel to the circuit.

        Args:
            channel: The noise channel to apply.
            qubits: Qubit indices this channel acts on.

        Returns:
            Self, for method chaining.
        """

        qubits_tuple = tuple(qubits)
        instr = CircuitInstruction(
            type=InstructionType.CHANNEL,
            channel=channel,
            qubits=qubits_tuple,
        )
        self._instructions.append(instr)
        return self

    def bit_flip(self, p: float, qubit: int) -> QuantumCircuit:
        """Insert a bit-flip (X error) channel.

        Args:
            p: Bit-flip probability in [0, 1].
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import BitFlipChannel

        return self.append_channel(
            BitFlipChannel(name="bit_flip", num_qubits=1, p=p), (qubit,)
        )

    def phase_flip(self, p: float, qubit: int) -> QuantumCircuit:
        """Insert a phase-flip (Z error) channel.

        Args:
            p: Phase-flip probability in [0, 1].
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import PhaseFlipChannel

        return self.append_channel(
            PhaseFlipChannel(name="phase_flip", num_qubits=1, p=p), (qubit,)
        )

    def bit_phase_flip(self, p: float, qubit: int) -> QuantumCircuit:
        """Insert a bit-phase-flip (Y error) channel.

        Args:
            p: Error probability in [0, 1].
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import BitPhaseFlipChannel

        return self.append_channel(
            BitPhaseFlipChannel(name="bit_phase_flip", num_qubits=1, p=p),
            (qubit,),
        )

    def depolarize(self, p: float, qubit: int) -> QuantumCircuit:
        """Insert a single-qubit depolarizing channel.

        Args:
            p: Depolarizing strength in [0, 1], using
                ``E(rho) = (1-p) rho + p I/2``.
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import DepolarizingChannel

        return self.append_channel(
            DepolarizingChannel(name="depolarizing", num_qubits=1, p=p),
            (qubit,),
        )

    def pauli_error(
        self, px: float, py: float, pz: float, qubit: int
    ) -> QuantumCircuit:
        """Insert a Pauli error channel.

        Args:
            px: Probability of X error.
            py: Probability of Y error.
            pz: Probability of Z error.
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import PauliChannel

        return self.append_channel(
            PauliChannel(name="pauli", num_qubits=1, px=px, py=py, pz=pz),
            (qubit,),
        )

    def amplitude_damp(self, gamma: float, qubit: int) -> QuantumCircuit:
        """Insert an amplitude damping channel (T1 process).

        Args:
            gamma: Damping rate in [0, 1].
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import AmplitudeDampingChannel

        return self.append_channel(
            AmplitudeDampingChannel(
                name="amplitude_damping", num_qubits=1, gamma=gamma
            ),
            (qubit,),
        )

    def phase_damp(self, gamma: float, qubit: int) -> QuantumCircuit:
        """Insert a phase damping channel (T2 process).

        Args:
            gamma: Dephasing rate in [0, 1].
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import PhaseDampingChannel

        return self.append_channel(
            PhaseDampingChannel(name="phase_damping", num_qubits=1, gamma=gamma),
            (qubit,),
        )

    def generalized_amplitude_damp(
        self, gamma: float, p_thermal: float, qubit: int
    ) -> QuantumCircuit:
        """Insert a generalized amplitude damping channel.

        The channel is named ``"generalized_amplitude_damping"`` (the same
        name used by JSON / QSAL serialization).

        Args:
            gamma: Damping rate in [0, 1].
            p_thermal: Thermal excitation probability in [0, 0.5].
            qubit: Target qubit index.

        Returns:
            Self, for method chaining.
        """
        from quspect.noise.channel import GeneralizedAmplitudeDampingChannel

        return self.append_channel(
            GeneralizedAmplitudeDampingChannel(
                name="generalized_amplitude_damping",
                num_qubits=1,
                gamma=gamma,
                p_thermal=p_thermal,
            ),
            (qubit,),
        )

    # ------------------------------------------------------------------
    # Parameter binding
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> set[Parameter]:
        """Return the set of all Parameters used in this circuit."""
        params: set[Parameter] = set()
        for instr in self._instructions:
            if instr.gate is not None and instr.gate.params:
                for p in instr.gate.params:
                    if isinstance(p, Parameter):
                        params.add(p)
                    elif isinstance(p, ParameterExpression):
                        params.update(p.parameters)
        return params

    def bind_parameters(self, binding: dict[Parameter, float]) -> QuantumCircuit:
        """Return a new circuit with symbolic parameters replaced by values.

        Bound values must be real numbers (``int``, ``float`` or numpy real
        scalars); they are stored as Python ``float``.

        Args:
            binding: A mapping from Parameter to concrete real values.

        Returns:
            A new QuantumCircuit with all parameters resolved.

        Raises:
            ValueError: If some parameters in the circuit are not bound.
            TypeError: If a bound value is not a real number.

        Example:
            >>> theta = Parameter("theta")
            >>> circ = QuantumCircuit().rx(theta, 0)
            >>> circ.bind_parameters({theta: 0.5})[0].gate.params
            (0.5,)
        """
        new_circuit = QuantumCircuit(name=self._name)
        for instr in self._instructions:
            if instr.gate is not None:
                if instr.gate.params:
                    # Resolve each parameter
                    new_params = []
                    for p in instr.gate.params:
                        if isinstance(p, Parameter):
                            if p not in binding:
                                raise ValueError(f"Parameter {p!r} is not bound.")
                            new_params.append(_require_real_binding(p, binding[p]))
                        elif isinstance(p, ParameterExpression):
                            new_params.append(p.evaluate(binding))
                        else:
                            new_params.append(float(p))
                    # Create new gate with resolved params
                    # Use _make_param_gate helper but with concrete values
                    new_gate = _make_param_gate(
                        instr.gate.name,
                        instr.gate.num_qubits,
                        *new_params,
                    )
                    new_instr = CircuitInstruction(
                        type=instr.type,
                        gate=new_gate,
                        qubits=instr.qubits,
                        clbits=instr.clbits,
                        label=instr.label,
                    )
                    new_circuit._instructions.append(new_instr)
                else:
                    new_circuit._instructions.append(instr)
            else:
                new_circuit._instructions.append(instr)
        return new_circuit

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(
        self,
        other: QuantumCircuit,
        qubits: Sequence[int],
        clbits: Sequence[int] | None = None,
        *,
        inplace: bool = False,
    ) -> QuantumCircuit:
        """Compose another circuit onto this one.

        Since qubit count is dynamically inferred, you must specify how
        ``other``'s qubits map onto this circuit's qubits. Gates,
        measurements, barriers and noise channels are all carried over with
        their qubits (and clbits) remapped.

        Args:
            other: The circuit to append.
            qubits: Remap other's qubits to these indices in this circuit.
                Must have length ``other.num_qubits``.
            clbits: Remap other's clbits to these indices. If None and
                other has measurements, classical bit indices are mapped
                identically from the qubit mapping.
            inplace: If True, modify this circuit. If False, return a copy.

        Returns:
            The composed circuit.

        Raises:
            ValueError: If ``len(qubits) != other.num_qubits``.
        """
        qubits_list = list(qubits)
        if len(qubits_list) != other.num_qubits:
            raise ValueError(
                f"compose() requires len(qubits)=other.num_qubits, "
                f"got {len(qubits_list)} vs {other.num_qubits}"
            )

        target = self if inplace else self.copy()

        # Build qubit mapping: other_qubit_index → target_qubit_index
        qubit_map = dict(enumerate(qubits_list))

        # Build clbit mapping if provided
        clbit_map: dict[int, int] | None = None
        if clbits is not None:
            clbits_list = list(clbits)
            clbit_map = dict(enumerate(clbits_list))

        for instr in other._instructions:
            # Remap qubit indices
            new_qubits = tuple(qubit_map[q] for q in instr.qubits)

            # Remap clbit indices
            if clbit_map is not None:
                new_clbits = tuple(clbit_map[c] for c in instr.clbits)
            else:
                new_clbits = instr.clbits

            target._instructions.append(
                CircuitInstruction(
                    type=instr.type,
                    gate=instr.gate,
                    channel=instr.channel,
                    qubits=new_qubits,
                    clbits=new_clbits,
                    label=instr.label,
                )
            )

        return target

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self, name: str | None = None) -> QuantumCircuit:
        """Return a deep copy of this circuit.

        Args:
            name: Optional new name for the copy. If None, keeps the
                original name.

        Returns:
            A new QuantumCircuit with the same instructions.
        """
        new_circuit = QuantumCircuit(name=name if name is not None else self._name)
        new_circuit._instructions = list(self._instructions)
        return new_circuit

    @classmethod
    def _from_instruction_snapshot(
        cls,
        instructions: Sequence[CircuitInstruction],
        *,
        name: str = "",
    ) -> QuantumCircuit:
        """Build an internal circuit from immutable instruction values.

        This package-private constructor is used by execution planners that
        replace one frozen gate occurrence without mutating the user's source
        circuit. Public code should normally use the fluent circuit methods.
        """
        snapshot = tuple(instructions)
        if not all(isinstance(item, CircuitInstruction) for item in snapshot):
            raise TypeError("Instruction snapshot contains a non-instruction value.")
        circuit = cls(name=name)
        circuit._instructions = list(snapshot)
        return circuit

    def dagger(self) -> QuantumCircuit:
        """Return the Hermitian conjugate (adjoint) of this circuit.

        The adjoint is the inverse of a unitary circuit: gates are reversed
        in order and each gate is replaced by its conjugate transpose.

        Self-adjoint gates (H, X, Y, Z, CX, CZ, SWAP, CCX) are unchanged.
        S ↔ S†, T ↔ T†.  Parameterized rotation gates flip the sign of
        their angle: RX(θ) → RX(-θ), etc.

        Measurements, barriers, and noise channels are preserved in
        reversed order.

        Returns:
            A new QuantumCircuit representing U†.

        Example:
            >>> circ = QuantumCircuit().z(0).ry(math.pi / 2, 0)
            >>> circ.dagger()  # Ry(-π/2) → Z
        """
        dag = QuantumCircuit(name=f"{self._name}†" if self._name else "dagger")
        for instr in reversed(self._instructions):
            if instr.is_barrier:
                if instr.qubits:
                    dag.barrier(*instr.qubits)
                else:
                    dag.barrier()
            elif instr.is_measure:
                dag.measure(instr.qubits, instr.clbits)
            elif instr.is_channel and instr.channel is not None:
                dag.append_channel(instr.channel, instr.qubits)
            elif instr.is_gate and instr.gate is not None:
                dag_gate = _adjoint_gate(instr.gate)
                dag.append(dag_gate, instr.qubits, instr.clbits)
        return dag

    # ------------------------------------------------------------------
    # DAG compilation
    # ------------------------------------------------------------------

    def to_dag(self) -> DAGCircuit:
        """Compile this circuit into a DAG representation.

        Converts the sequential circuit into a directed acyclic graph
        that explicitly represents qubit/clbit dependencies. The DAG
        is the primary intermediate representation for circuit
        compilation, optimization, and analysis.

        Returns:
            A DAGCircuit representing the dependency structure.

        Example:
            >>> circ = QuantumCircuit()
            >>> circ.h(0).cx(0, 1)
            >>> dag = circ.to_dag()
            >>> dag.num_nodes()
            6  # 2 INPUT + 2 gates + 2 OUTPUT
        """
        from quspect.dag.compiler import CircuitCompiler

        return CircuitCompiler.compile(self)

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize this circuit to a JSON string.

        Gates (including symbolic parameters and affine expressions),
        measurements, barriers, labels and noise channels are serialized.
        Noise channels store all of their strength fields (``p``,
        ``gamma``, ``p_thermal``, ``px``/``py``/``pz``) so that
        :meth:`from_json` restores an equal channel. Kraus channels only
        record their operator count and cannot be deserialized.

        Returns:
            A JSON string representing the circuit.

        Example:
            >>> circ = QuantumCircuit(name="test")
            >>> circ.h(0)
            >>> data = circ.to_json()
            >>> circ2 = QuantumCircuit.from_json(data)
            >>> circ2.name
            'test'
            >>> len(circ2)
            1
        """
        data: dict[str, Any] = {
            "name": self._name,
            "instructions": [],
        }
        for instr in self._instructions:
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
                if instr.gate.params:
                    gate_data["params"] = [
                        p.name
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
                        else float(p)
                        for p in instr.gate.params
                    ]
                    gate_data["gate_type"] = "parameterized"
                else:
                    gate_data["gate_type"] = "standard"
                instr_data["gate"] = gate_data
            if instr.channel is not None:
                ch = instr.channel
                ch_data: dict[str, Any] = {
                    "channel_type": ch.name,
                    "num_qubits": ch.num_qubits,
                }
                if ch.params:
                    ch_data["params"] = [float(v) for v in ch.params]
                # Serialize every channel-specific strength field
                # (p, gamma, p_thermal, px/py/pz, ...).
                for ch_field in dataclasses.fields(ch):
                    if ch_field.name in _CHANNEL_BASE_FIELDS:
                        continue
                    value = getattr(ch, ch_field.name)
                    if isinstance(value, Real):
                        ch_data[ch_field.name] = float(value)
                from quspect.noise.channel import KrausChannel

                if isinstance(ch, KrausChannel):
                    # Kraus ops are not serialized to JSON (too large)
                    ch_data["kraus_ops_count"] = len(ch.kraus_ops)
                instr_data["channel"] = ch_data
            if instr.label:
                instr_data["label"] = instr.label
            data["instructions"].append(instr_data)
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> QuantumCircuit:
        """Deserialize a circuit from a JSON string.

        Older documents that omit channel strengths (they then default to
        ``0.0``) or use the legacy channel name ``"gen_amp_damp"`` are still
        accepted; the latter is normalized to
        ``"generalized_amplitude_damping"``.

        Args:
            data: A JSON string produced by :meth:`to_json`.

        Returns:
            A new QuantumCircuit instance.

        Raises:
            ValueError: If the JSON is malformed or references unknown gates.
        """
        from quspect.gates.standard import ALL_STANDARD

        # Build lookup dict from ALL_STANDARD list
        standard_lookup = {g.name: g for g in ALL_STANDARD}

        raw = json.loads(data)
        circ = cls(name=raw.get("name", ""))

        for instr_data in raw.get("instructions", []):
            itype = InstructionType[instr_data["type"]]
            qubits = tuple(instr_data["qubits"])
            clbits = tuple(instr_data.get("clbits", []))

            gate: Gate | None = None
            if "gate" in instr_data:
                gd = instr_data["gate"]
                if gd.get("gate_type") == "parameterized":
                    params_raw = gd.get("params", [])
                    params = tuple(
                        Parameter(p)
                        if isinstance(p, str)
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
                    )
                else:
                    gate_name = gd["name"]
                    if gate_name not in standard_lookup:
                        raise ValueError(f"Unknown standard gate: {gate_name!r}")
                    gate = standard_lookup[gate_name]

            channel = None
            if "channel" in instr_data:
                ch_data = instr_data["channel"]
                channel = _deserialize_channel(ch_data)

            label = instr_data.get("label", "")

            circ._instructions.append(
                CircuitInstruction(
                    type=itype,
                    gate=gate,
                    channel=channel,
                    qubits=qubits,
                    clbits=clbits,
                    label=label,
                )
            )

        return circ

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        name_str = f"name={self._name!r}" if self._name else ""
        return (
            f"QuantumCircuit({name_str}, "
            f"num_qubits={self.num_qubits}, "
            f"num_clbits={self.num_clbits}, "
            f"instructions={len(self)})"
        )

    def draw(self) -> str:
        """Return a text-based circuit diagram.

        Returns:
            A multi-line string showing the circuit layout.
        """
        from quspect.visualization import draw_text_circuit

        return draw_text_circuit(self)

    def svg(
        self,
        style: str | dict[str, str] | Any | None = None,
        width: int | float | None = None,
        scale: float | None = None,
    ) -> Any:
        """Return an SVG representation of the circuit.

        Args:
            style: Optional named theme (``"official"``, ``"light"``, ``"dark"``),
                a custom SVG color dictionary, or a ``SvgStyle`` instance.
            width: Optional minimum SVG width in pixels.
            scale: Optional positive scale factor applied to the final figure.

        Returns:
            A notebook-friendly SVG wrapper object.
        """
        from quspect.visualization import draw_svg_circuit

        return draw_svg_circuit(self, style=style, width=width, scale=scale)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _deserialize_channel(data: dict[str, Any]) -> Any:
    """Deserialize a noise channel from JSON data.

    Args:
        data: Dict with channel_type, num_qubits, and type-specific params.

    Returns:
        A NoiseChannel instance.
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

    ch_type = data["channel_type"]
    num_qubits = data["num_qubits"]
    params = data.get("params", [])

    if ch_type == "bit_flip":
        p = params[0] if params else data.get("p", 0.0)
        return BitFlipChannel(name=ch_type, num_qubits=num_qubits, p=p)
    elif ch_type == "phase_flip":
        p = params[0] if params else data.get("p", 0.0)
        return PhaseFlipChannel(name=ch_type, num_qubits=num_qubits, p=p)
    elif ch_type == "bit_phase_flip":
        p = params[0] if params else data.get("p", 0.0)
        return BitPhaseFlipChannel(name=ch_type, num_qubits=num_qubits, p=p)
    elif ch_type == "pauli":
        px = data.get("px", 0.0)
        py = data.get("py", 0.0)
        pz = data.get("pz", 0.0)
        return PauliChannel(name=ch_type, num_qubits=num_qubits, px=px, py=py, pz=pz)
    elif ch_type == "depolarizing":
        p = params[0] if params else data.get("p", 0.0)
        return DepolarizingChannel(name=ch_type, num_qubits=num_qubits, p=p)
    elif ch_type == "amplitude_damping":
        gamma = params[0] if params else data.get("gamma", 0.0)
        return AmplitudeDampingChannel(name=ch_type, num_qubits=num_qubits, gamma=gamma)
    elif ch_type == "phase_damping":
        gamma = params[0] if params else data.get("gamma", 0.0)
        return PhaseDampingChannel(name=ch_type, num_qubits=num_qubits, gamma=gamma)
    elif ch_type in ("generalized_amplitude_damping", "gen_amp_damp"):
        # "gen_amp_damp" is the legacy name written by older versions.
        gamma = params[0] if params else data.get("gamma", 0.0)
        p_thermal = data.get("p_thermal", 0.0)
        return GeneralizedAmplitudeDampingChannel(
            name="generalized_amplitude_damping",
            num_qubits=num_qubits,
            gamma=gamma,
            p_thermal=p_thermal,
        )
    elif ch_type in ("kraus",):
        # Kraus ops cannot be deserialized without the actual matrices
        raise ValueError(
            "Cannot deserialize KrausChannel from JSON: "
            "Kraus operators are not serializable"
        )
    else:
        raise ValueError(f"Unknown channel type: {ch_type!r}")


#: Dataclass fields shared by every NoiseChannel (not channel strengths).
_CHANNEL_BASE_FIELDS = frozenset({"name", "num_qubits", "params", "label"})

#: Parameterized gate templates by name (used to keep display labels).
_PARAM_TEMPLATES: dict[str, Gate] = {
    gate.name: gate for gate in (RX, RY, RZ, P, U1, U2, U3, CRX, CRY, CRZ, CP, CU3)
}


def _coerce_param(value: object) -> float | Parameter | ParameterExpression:
    """Normalize a fluent-API gate angle.

    Real scalars (including numpy scalars such as ``np.int64`` or
    ``np.float32``) become Python ``float``; symbolic parameters are
    returned unchanged.

    Raises:
        TypeError: If ``value`` is neither a real number nor symbolic.
    """
    if isinstance(value, (Parameter, ParameterExpression)):
        return value
    if isinstance(value, Real):
        return float(value)
    raise TypeError(
        f"Gate parameter must be a real number, Parameter or "
        f"ParameterExpression, got {value!r}."
    )


def _require_real_binding(parameter: Parameter, value: object) -> float:
    """Validate a ``bind_parameters`` value and convert it to ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
    """
    if not isinstance(value, Real):
        raise TypeError(
            f"Binding for {parameter!r} must be a real number, got {value!r}."
        )
    return float(value)


def _make_param_gate(
    name: str,
    num_qubits: int,
    *params: float | Parameter | ParameterExpression,
) -> Gate:
    """Create a parameterized Gate with mixed float/Parameter params.

    Args:
        name: Gate name (e.g., ``"rx"``).
        num_qubits: Number of qubits for this gate.
        *params: Parameter values (float or Parameter).

    Returns:
        A Gate instance with parameters.
    """
    template = _PARAM_TEMPLATES.get(name)
    return Gate(
        name=name,
        num_qubits=num_qubits,
        params=tuple(params),
        label=template.label if template is not None else "",
    )


# ---------------------------------------------------------------------------
# Adjoint gate lookup
# ---------------------------------------------------------------------------

# Self-adjoint gates: name → same name (no change)
_SELF_ADJOINT = frozenset({"id", "h", "x", "y", "z", "cx", "cz", "swap", "ccx"})
# Gates with a built-in dagger sibling
_DAGGER_MAP: dict[str, str] = {
    "s": "sdg",
    "sdg": "s",
    "t": "tdg",
    "tdg": "t",
    "sx": "sxdg",
    "sxdg": "sx",
}


def _adjoint_gate(gate: Gate) -> Gate:
    """Return the Hermitian conjugate of a gate.

    Args:
        gate: The gate to compute the adjoint of.

    Returns:
        A new Gate instance representing the adjoint.
    """
    name = gate.name.lower()

    # Self-adjoint: return unchanged
    if name in _SELF_ADJOINT:
        return gate

    # S ↔ SDG, T ↔ TDG
    if name in _DAGGER_MAP:
        return Gate(name=_DAGGER_MAP[name], num_qubits=gate.num_qubits)

    # Parameterized gates: flip all parameter signs
    params = gate.params
    if params:
        neg_params = tuple(
            -p if isinstance(p, (int, float, Parameter, ParameterExpression)) else p
            for p in params
        )

        # U2(φ, λ) → U3(-π/2, -λ, -φ), since U2 = U3(π/2, φ, λ)
        if name == "u2" and len(neg_params) == 2:
            return Gate(
                name="u3",
                num_qubits=gate.num_qubits,
                params=(-math.pi / 2, neg_params[1], neg_params[0]),
            )

        # U3(θ, φ, λ)† = U3(-θ, -λ, -φ) — swap last two params
        if name in ("u3", "cu3") and len(neg_params) == 3:
            neg_params = (neg_params[0], neg_params[2], neg_params[1])

        return Gate(name=name, num_qubits=gate.num_qubits, params=neg_params)

    return gate
