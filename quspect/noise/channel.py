"""Quantum noise channels — CPTP maps for noisy quantum circuit simulation.

This module defines the :class:`NoiseChannel` base class and nine
concrete noise channel types.  Noise channels represent
completely-positive trace-preserving (CPTP) maps on quantum states.

Unlike gates which are unitary, channels may increase entropy and are
generally irreversible.  Every channel provides a Kraus operator
decomposition:

    ε(ρ) = Σ_i K_i ρ K_i†

where Σ_i K_i† K_i = I (trace-preserving condition).

Channel types:
    - PauliChannel — general Pauli error (px X + py Y + pz Z)
    - BitFlipChannel — X error
    - PhaseFlipChannel — Z error
    - BitPhaseFlipChannel — Y error
    - DepolarizingChannel — isotropic depolarization
    - AmplitudeDampingChannel — T1 energy relaxation
    - PhaseDampingChannel — T2 pure dephasing
    - GeneralizedAmplitudeDampingChannel — T1 at finite temperature
    - KrausChannel — user-defined CPTP channel

Note: Pauli matrix variables (I, X, Y, Z) and Kraus operator
variables (K0, K1, ...) use uppercase naming as per physics convention.
"""

# ruff: noqa: N806, E741
# (physics convention: Pauli matrices I, X, Y, Z; Kraus operators K0, K1, ...)

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# =============================================================================
# Base class
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class NoiseChannel:
    """Base class for quantum noise channels (CPTP maps).

    A noise channel represents a completely-positive trace-preserving
    (CPTP) map on quantum states. Unlike gates which are unitary,
    channels may increase entropy and are generally irreversible.

    Each channel must provide its Kraus operator decomposition:

        ε(ρ) = Σ_i K_i ρ K_i†

    where Σ_i K_i† K_i = I (trace-preserving condition).

    Attributes:
        name: Short canonical name, e.g. ``"depolarizing"``,
            ``"amplitude_damping"``.
        num_qubits: Number of qubits this channel acts on.
        params: Channel parameters (probabilities, rates, etc.).
        label: Optional human-readable label.
    """

    name: str
    num_qubits: int
    params: tuple[float, ...] = field(default_factory=tuple)
    label: str = field(default="", compare=False, hash=False)

    def kraus_operators(self) -> list[np.ndarray]:
        """Return the Kraus operator decomposition of this channel.

        Returns:
            A list of Kraus operators ``K_i`` such that the channel
            action is ε(ρ) = Σ_i K_i ρ K_i†.

        Raises:
            NotImplementedError: Subclasses must implement this.
        """
        raise NotImplementedError

    def is_unitary(self) -> bool:
        """Return True if this channel reduces to a unitary (single Kraus op)."""
        return len(self.kraus_operators()) == 1

    def __repr__(self) -> str:
        label_part = f", label={self.label!r}" if self.label else ""
        params_part = f", params={self.params!r}" if self.params else ""
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"num_qubits={self.num_qubits}{params_part}{label_part})"
        )


def _require_single_qubit(channel: NoiseChannel) -> None:
    """Raise ValueError unless ``channel.num_qubits == 1``.

    Args:
        channel: A channel whose Kraus operators are fixed 2x2 matrices.

    Raises:
        ValueError: If the channel was constructed with ``num_qubits != 1``.
    """
    if channel.num_qubits != 1:
        raise ValueError(
            f"{type(channel).__name__} is a single-qubit channel; "
            f"num_qubits must be 1, got {channel.num_qubits}"
        )


# =============================================================================
# Pauli-type channels
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class PauliChannel(NoiseChannel):
    """General Pauli error channel.

    Applies random Pauli errors with independent probabilities:

        ε(ρ) = (1 - px - py - pz) ρ + px XρX + py YρY + pz ZρZ

    Constraints: ``px, py, pz >= 0``, ``px + py + pz <= 1``.

    Attributes:
        px: Probability of Pauli-X error.
        py: Probability of Pauli-Y error.
        pz: Probability of Pauli-Z error.

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = PauliChannel(name="pauli", num_qubits=1, px=0.01, py=0.02, pz=0.03)
        >>> ops = ch.kraus_operators()
        >>> len(ops)
        4
    """

    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if self.px < 0 or self.py < 0 or self.pz < 0:
            raise ValueError("Pauli channel probabilities must be non-negative")
        if self.px + self.py + self.pz > 1.0:
            raise ValueError(
                f"Sum of Pauli probabilities ({self.px + self.py + self.pz}) "
                f"must not exceed 1.0"
            )

    def kraus_operators(self) -> list[np.ndarray]:
        """Return Pauli channel Kraus operators.

        Returns:
            [sqrt(1-pΣ)·I, sqrt(px)·X, sqrt(py)·Y, sqrt(pz)·Z]
        """
        p_sum = self.px + self.py + self.pz
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        ops = []
        if 1.0 - p_sum > 1e-15:
            ops.append(np.sqrt(1.0 - p_sum) * I)
        if self.px > 1e-15:
            ops.append(np.sqrt(self.px) * X)
        if self.py > 1e-15:
            ops.append(np.sqrt(self.py) * Y)
        if self.pz > 1e-15:
            ops.append(np.sqrt(self.pz) * Z)
        return ops


@dataclass(frozen=True, kw_only=True)
class BitFlipChannel(NoiseChannel):
    """Bit-flip (X error) channel.

    Models random bit-flip errors in the computational basis:

        ε(ρ) = (1 - p) ρ + p XρX

    Attributes:
        p: Probability of bit flip. Must be in [0, 1].

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = BitFlipChannel(name="bit_flip", num_qubits=1, p=0.05)
        >>> len(ch.kraus_operators())
        2
    """

    p: float

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"Bit-flip probability p must be in [0, 1], got {self.p}")

    def kraus_operators(self) -> list[np.ndarray]:
        """Return bit-flip Kraus operators: [sqrt(1-p)·I, sqrt(p)·X]."""
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        ops = []
        if 1.0 - self.p > 1e-15:
            ops.append(np.sqrt(1.0 - self.p) * I)
        if self.p > 1e-15:
            ops.append(np.sqrt(self.p) * X)
        return ops


@dataclass(frozen=True, kw_only=True)
class PhaseFlipChannel(NoiseChannel):
    """Phase-flip (Z error) channel.

    Models random phase-flip errors:

        ε(ρ) = (1 - p) ρ + p ZρZ

    Attributes:
        p: Probability of phase flip. Must be in [0, 1].

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = PhaseFlipChannel(name="phase_flip", num_qubits=1, p=0.05)
        >>> len(ch.kraus_operators())
        2
    """

    p: float

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(
                f"Phase-flip probability p must be in [0, 1], got {self.p}"
            )

    def kraus_operators(self) -> list[np.ndarray]:
        """Return phase-flip Kraus operators: [sqrt(1-p)·I, sqrt(p)·Z]."""
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        ops = []
        if 1.0 - self.p > 1e-15:
            ops.append(np.sqrt(1.0 - self.p) * I)
        if self.p > 1e-15:
            ops.append(np.sqrt(self.p) * Z)
        return ops


@dataclass(frozen=True, kw_only=True)
class BitPhaseFlipChannel(NoiseChannel):
    """Bit-phase-flip (Y error) channel.

    Models simultaneous bit and phase flip errors:

        ε(ρ) = (1 - p) ρ + p YρY

    Attributes:
        p: Probability of Y error. Must be in [0, 1].

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = BitPhaseFlipChannel(name="bit_phase_flip", num_qubits=1, p=0.05)
        >>> len(ch.kraus_operators())
        2
    """

    p: float

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(
                f"Bit-phase-flip probability p must be in [0, 1], got {self.p}"
            )

    def kraus_operators(self) -> list[np.ndarray]:
        """Return bit-phase-flip Kraus operators: [sqrt(1-p)·I, sqrt(p)·Y]."""
        I = np.eye(2, dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        ops = []
        if 1.0 - self.p > 1e-15:
            ops.append(np.sqrt(1.0 - self.p) * I)
        if self.p > 1e-15:
            ops.append(np.sqrt(self.p) * Y)
        return ops


# =============================================================================
# Depolarizing channel
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class DepolarizingChannel(NoiseChannel):
    """Depolarizing channel.

    Uses the depolarizing-strength convention also used by MindQuantum:

        ε(ρ) = (1 - p) ρ + p I / 2**n

    For one qubit this is equivalently:

        ε(ρ) = (1 - 3p/4) ρ + p/4 (XρX + YρY + ZρZ)

    Thus ``p=1`` is the completely depolarizing channel.  The probability of
    selecting a non-identity Pauli trajectory is ``p * (1 - 1 / 4**n)``.

    Attributes:
        p: Depolarizing strength in [0, 1].

    Example:
        >>> ch = DepolarizingChannel(name="depolarizing", num_qubits=1, p=0.01)
        >>> len(ch.kraus_operators())
        4
    """

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(
                f"Depolarizing probability p must be in [0, 1], got {self.p}"
            )

    def kraus_operators(self) -> list[np.ndarray]:
        """Return depolarizing channel Kraus operators.

        For 1-qubit:
            [sqrt(1-3p/4)·I, sqrt(p/4)·X, sqrt(p/4)·Y, sqrt(p/4)·Z]
        For 2-qubit:
            [sqrt(1-15p/16)·I⊗I, sqrt(p/16)·P_i for all non-identity 2q Paulis]
        """
        if self.num_qubits == 1:
            return _depolarizing_1q_kraus(self.p)
        elif self.num_qubits == 2:
            return _depolarizing_2q_kraus(self.p)
        else:
            raise NotImplementedError(
                f"Depolarizing channel not implemented for {self.num_qubits} qubits"
            )


def _depolarizing_1q_kraus(p: float) -> list[np.ndarray]:
    """Compute single-qubit depolarizing Kraus operators."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    ops = []
    if 1.0 - 3.0 * p / 4.0 > 1e-15:
        ops.append(np.sqrt(1.0 - 3.0 * p / 4.0) * I)
    if p / 4.0 > 1e-15:
        sq = np.sqrt(p / 4.0)
        ops.append(sq * X)
        ops.append(sq * Y)
        ops.append(sq * Z)
    return ops


def _depolarizing_2q_kraus(p: float) -> list[np.ndarray]:
    """Compute two-qubit depolarizing Kraus operators.

    Uses all 15 non-identity two-qubit Pauli strings with Kraus weight p/16;
    together with the identity contribution this implements
    ``(1-p) rho + p I/4``.
    """
    paulis_1q = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]

    # Identity term
    II = np.kron(paulis_1q[0], paulis_1q[0])

    ops = []
    if 1.0 - 15.0 * p / 16.0 > 1e-15:
        ops.append(np.sqrt(1.0 - 15.0 * p / 16.0) * II)

    if p / 16.0 > 1e-15:
        sq = np.sqrt(p / 16.0)
        for a in range(4):
            for b in range(4):
                if a == 0 and b == 0:
                    continue  # Skip II — already included as the identity term
                ops.append(sq * np.kron(paulis_1q[a], paulis_1q[b]))

    return ops


# =============================================================================
# Damping channels (T1 / T2 processes)
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class AmplitudeDampingChannel(NoiseChannel):
    """Amplitude damping channel (energy relaxation / T1 process).

    Kraus operators::

        K0 = [[1, 0          ], [0, sqrt(1 - γ)]]
        K1 = [[0, sqrt(γ)    ], [0, 0          ]]

    Models energy loss from |1⟩ → |0⟩ with probability γ.
    This is the zero-temperature limit of generalized amplitude damping.

    Attributes:
        gamma: Damping rate. Must be in [0, 1].

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.05)
        >>> ops = ch.kraus_operators()
        >>> ops[0].shape
        (2, 2)
    """

    gamma: float

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

    def kraus_operators(self) -> list[np.ndarray]:
        """Return amplitude damping Kraus operators."""
        ops = []
        K0 = np.array(
            [[1.0, 0.0], [0.0, np.sqrt(1.0 - self.gamma)]],
            dtype=complex,
        )
        ops.append(K0)
        if self.gamma > 1e-15:
            K1 = np.array(
                [[0.0, np.sqrt(self.gamma)], [0.0, 0.0]],
                dtype=complex,
            )
            ops.append(K1)
        return ops


@dataclass(frozen=True, kw_only=True)
class PhaseDampingChannel(NoiseChannel):
    """Phase damping channel (pure dephasing / T2 process).

    Kraus operators::

        K0 = [[1, 0          ], [0, sqrt(1 - γ)]]
        K1 = [[0, 0          ], [0, sqrt(γ)    ]]

    Models loss of phase coherence without energy exchange.
    This is the pure-dephasing (no T1) limit of the combined
    T1+T2 channel.

    Attributes:
        gamma: Dephasing rate. Must be in [0, 1].

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = PhaseDampingChannel(name="pd", num_qubits=1, gamma=0.05)
        >>> len(ch.kraus_operators())
        2
    """

    gamma: float

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

    def kraus_operators(self) -> list[np.ndarray]:
        """Return phase damping Kraus operators."""
        ops = []
        K0 = np.array(
            [[1.0, 0.0], [0.0, np.sqrt(1.0 - self.gamma)]],
            dtype=complex,
        )
        ops.append(K0)
        if self.gamma > 1e-15:
            K1 = np.array(
                [[0.0, 0.0], [0.0, np.sqrt(self.gamma)]],
                dtype=complex,
            )
            ops.append(K1)
        return ops


@dataclass(frozen=True, kw_only=True)
class GeneralizedAmplitudeDampingChannel(NoiseChannel):
    """Generalized amplitude damping at finite temperature.

    Models T1 relaxation toward a thermal equilibrium state with
    population ``p_thermal`` in |1⟩. At zero temperature
    (``p_thermal = 0``), this reduces to
    :class:`AmplitudeDampingChannel`.

    Attributes:
        gamma: Damping rate. Must be in [0, 1].
        p_thermal: Thermal excitation probability. Must be in [0, 0.5].

    Raises:
        ValueError: If ``num_qubits != 1`` (this is a single-qubit
            channel) or a parameter is out of range.

    Example:
        >>> ch = GeneralizedAmplitudeDampingChannel(
        ...     name="gad", num_qubits=1, gamma=0.05, p_thermal=0.01
        ... )
        >>> len(ch.kraus_operators())
        4
    """

    gamma: float
    p_thermal: float = 0.0

    def __post_init__(self) -> None:
        _require_single_qubit(self)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.p_thermal <= 0.5:
            raise ValueError(f"p_thermal must be in [0, 0.5], got {self.p_thermal}")

    def kraus_operators(self) -> list[np.ndarray]:
        """Return generalized amplitude damping Kraus operators.

        Returns four Kraus operators representing T1 relaxation
        toward a finite-temperature equilibrium.
        """
        p = self.p_thermal
        g = self.gamma

        ops = []
        K0 = np.sqrt(1.0 - p) * np.array(
            [[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]],
            dtype=complex,
        )
        ops.append(K0)
        if g > 1e-15:
            K1 = np.sqrt(1.0 - p) * np.array(
                [[0.0, np.sqrt(g)], [0.0, 0.0]],
                dtype=complex,
            )
            ops.append(K1)
        K2 = np.sqrt(p) * np.array(
            [[np.sqrt(1.0 - g), 0.0], [0.0, 1.0]],
            dtype=complex,
        )
        ops.append(K2)
        if g > 1e-15:
            K3 = np.sqrt(p) * np.array(
                [[0.0, 0.0], [np.sqrt(g), 0.0]],
                dtype=complex,
            )
            ops.append(K3)
        return ops


# =============================================================================
# User-defined Kraus channel
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class KrausChannel(NoiseChannel):
    """Arbitrary CPTP channel defined by user-supplied Kraus operators.

    The trace-preserving condition Σ K_i† K_i = I is validated at
    construction time.

    Two ``KrausChannel`` instances compare equal when they have the same
    class, ``name``, ``num_qubits``, ``params`` and element-wise identical
    Kraus operators (in the same order).  ``label`` is ignored, as for all
    channels.  The hash is consistent with this equality.

    Attributes:
        kraus_ops: List of Kraus operators. Each must be a square matrix
            of dimension ``2^num_qubits``.
        name: User-defined channel name.

    Raises:
        ValueError: If an operator has the wrong shape or the operators
            are not trace preserving.

    Example:
        >>> import numpy as np
        >>> K0 = np.sqrt(0.9) * np.eye(2, dtype=complex)
        >>> K1 = np.sqrt(0.1) * np.array([[0, 1], [1, 0]], dtype=complex)
        >>> # Trace preserving: K0†K0 + K1†K1 = 0.9 I + 0.1 I = I
        >>> ch = KrausChannel(name="custom", num_qubits=1, kraus_ops=[K0, K1])
        >>> len(ch.kraus_operators())
        2
        >>> ch == KrausChannel(name="custom", num_qubits=1, kraus_ops=[K0, K1])
        True
        >>> ch == KrausChannel(name="custom", num_qubits=1, kraus_ops=[K1, K0])
        False
    """

    kraus_ops: list[np.ndarray] = field(repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        dim = 1 << self.num_qubits
        for i, op in enumerate(self.kraus_ops):
            if op.shape != (dim, dim):
                raise ValueError(
                    f"Kraus operator {i} has shape {op.shape}, expected ({dim}, {dim})"
                )
        # Validate trace-preserving: Σ K_i† K_i ≈ I
        tp_sum = np.zeros((dim, dim), dtype=complex)
        for op in self.kraus_ops:
            tp_sum += op.conj().T @ op
        if not np.allclose(tp_sum, np.eye(dim, dtype=complex), atol=1e-10):
            raise ValueError(
                f"Kraus operators do not satisfy Σ K_i† K_i = I\nGot: \n{tp_sum}"
            )

    def __eq__(self, other: object) -> bool:
        """Compare class, metadata and Kraus operators element-wise."""
        if not isinstance(other, KrausChannel) or other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.name == other.name
            and self.num_qubits == other.num_qubits
            and self.params == other.params
            and len(self.kraus_ops) == len(other.kraus_ops)
            and all(
                np.array_equal(a, b)
                for a, b in zip(self.kraus_ops, other.kraus_ops, strict=True)
            )
        )

    def __hash__(self) -> int:
        """Hash consistent with :meth:`__eq__` (operators enter via count)."""
        return hash(
            (
                self.__class__,
                self.name,
                self.num_qubits,
                self.params,
                len(self.kraus_ops),
            )
        )

    def kraus_operators(self) -> list[np.ndarray]:
        """Return the user-supplied Kraus operators.

        Returns:
            A copy of the internal Kraus operator list.
        """
        return [op.copy() for op in self.kraus_ops]
