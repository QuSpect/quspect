"""Noise model — declarative noise specification for quantum circuits.

A :class:`NoiseModel` is a collection of :class:`NoiseRule` instances
that describe when, where, and what noise to insert into a quantum
circuit.  The model can be applied to a circuit to produce a noisy
version, or passed to a simulator for dynamic noise insertion.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.gates.gate import Gate
    from quspect.noise.channel import NoiseChannel


# =============================================================================
# NoiseRule
# =============================================================================


class NoiseRule:
    """A single noise insertion rule.

    Defines when and where to insert noise channels into a quantum circuit.

    The rule fires when ALL of:
    - ``gate_filter(gate)`` returns True (or ``gate_filter`` is None = always)
    - ``qubit_filter(qubit)`` returns True for at least one qubit involved
      in the gate (or ``qubit_filter`` is None = always)
    - ``exclude_filter(gate)`` returns False (or None = never exclude)

    When it fires, it inserts ``channel`` AFTER the matched gate on the
    matched qubits. If ``before_gate=True``, inserts BEFORE.

    Rules with ``on_measure=True`` are measurement rules: they are checked
    against MEASURE instructions (instead of gates) and their channel is
    always inserted immediately before the measurement.

    If the channel acts on exactly as many qubits as the target qubits, it
    is inserted as one instruction on all of them.  A 1-qubit channel that
    matches a multi-qubit gate is instead inserted once per target qubit;
    when a ``qubit_filter`` is set (and ``focus_on`` is not), only the
    target qubits accepted by the filter receive a copy.

    Attributes:
        channel: The noise channel to insert.
        gate_filter: Optional predicate — match gates by name/type.
        qubit_filter: Optional predicate — match specific qubit indices.
        exclude_filter: Optional predicate — explicitly exclude gates.
        before_gate: If True, insert noise BEFORE the gate (default: after).
        focus_on: Specific qubit indices to apply noise to. If None,
            applies to all qubits of the matched gate.
        label: Optional human-readable rule name.
        on_measure: If True, the rule matches MEASURE instructions.
    """

    def __init__(
        self,
        channel: NoiseChannel,
        *,
        gate_filter: Callable[[Gate], bool] | None = None,
        qubit_filter: Callable[[int], bool] | None = None,
        exclude_filter: Callable[[Gate], bool] | None = None,
        before_gate: bool = False,
        focus_on: tuple[int, ...] | None = None,
        label: str = "",
        on_measure: bool = False,
    ) -> None:
        """Initialize a noise insertion rule.

        Args:
            channel: The noise channel to insert when this rule fires.
            gate_filter: If provided, only gates matching this predicate
                are considered for noise insertion.
            qubit_filter: If provided, at least one qubit involved in the
                gate must match this predicate for the rule to fire.
            exclude_filter: If provided, gates matching this predicate
                are explicitly excluded from noise insertion.
            before_gate: If True, insert noise before the matched gate
                rather than after.
            focus_on: If provided, only these qubit indices get noise.
                When None, all qubits of the matched gate get noise.
            label: Optional human-readable identifier for this rule.
            on_measure: If True, this is a measurement rule: the adder
                checks it against MEASURE instructions (via
                :meth:`matches_measure`) and inserts the channel right
                before the measurement.  ``gate_filter`` and
                ``exclude_filter`` are not consulted for measurements.
        """
        self.channel = channel
        self.gate_filter = gate_filter
        self.qubit_filter = qubit_filter
        self.exclude_filter = exclude_filter
        self.before_gate = before_gate
        self.focus_on = focus_on
        self.label = label
        self.on_measure = on_measure

    def matches_measure(self, qubits: tuple[int, ...]) -> bool:
        """Check whether this measurement rule fires for a MEASURE instruction.

        Args:
            qubits: The measured qubit indices.

        Returns:
            True if ``on_measure`` is set and the qubit filter (if any)
            accepts at least one measured qubit.

        Example:
            >>> from quspect.noise.channel import BitFlipChannel
            >>> ch = BitFlipChannel(name="ro", num_qubits=1, p=0.1)
            >>> NoiseRule(ch, on_measure=True).matches_measure((0, 1))
            True
            >>> NoiseRule(ch).matches_measure((0,))
            False
        """
        if not self.on_measure:
            return False
        return self.qubit_filter is None or any(self.qubit_filter(q) for q in qubits)

    def matches_gate(self, gate: Gate, qubits: tuple[int, ...]) -> bool:
        """Check whether this rule should fire for the given gate and qubits.

        Args:
            gate: The gate to check.
            qubits: The qubit indices the gate acts on.

        Returns:
            True if noise should be inserted for this gate.
        """
        # Exclusion first
        if self.exclude_filter is not None and self.exclude_filter(gate):
            return False
        # Gate filter
        if self.gate_filter is not None and not self.gate_filter(gate):
            return False
        # Qubit filter — at least one qubit must match
        return self.qubit_filter is None or any(self.qubit_filter(q) for q in qubits)

    def get_target_qubits(self, qubits: tuple[int, ...]) -> tuple[int, ...]:
        """Return the qubit indices on which noise should be applied.

        Args:
            qubits: All qubits involved in the matched gate.

        Returns:
            The subset of qubits to apply the channel to.
        """
        if self.focus_on is not None:
            return self.focus_on
        return qubits

    def channel_placements(self, qubits: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Return the qubit tuples on which one channel instruction each is placed.

        The target qubits come from :meth:`get_target_qubits`.  If the
        channel width equals the number of targets, a single placement on
        all targets is returned.  A 1-qubit channel on several targets is
        split into one placement per target qubit; without ``focus_on``,
        targets rejected by ``qubit_filter`` are skipped.

        Args:
            qubits: All qubits involved in the matched instruction.

        Returns:
            A list of qubit tuples, each of length ``channel.num_qubits``.

        Raises:
            ValueError: If the channel width matches neither the number of
                target qubits nor 1.

        Example:
            >>> from quspect.noise.channel import BitFlipChannel
            >>> ch = BitFlipChannel(name="bf", num_qubits=1, p=0.1)
            >>> NoiseRule(ch).channel_placements((0, 1))
            [(0,), (1,)]
            >>> NoiseRule(ch, qubit_filter=lambda q: q == 1).channel_placements((0, 1))
            [(1,)]
        """
        targets = self.get_target_qubits(qubits)
        width = self.channel.num_qubits
        if width == len(targets):
            return [targets]
        if width == 1:
            use_filter = self.focus_on is None and self.qubit_filter is not None
            return [
                (q,)
                for q in targets
                if not use_filter
                or (self.qubit_filter is not None and self.qubit_filter(q))
            ]
        raise ValueError(
            f"Channel {self.channel.name!r} acts on {width} qubits but the rule "
            f"targets {len(targets)} qubits {targets}"
        )

    def __repr__(self) -> str:
        label_part = f", label={self.label!r}" if self.label else ""
        if self.on_measure:
            label_part += ", on_measure=True"
        return (
            f"NoiseRule(channel={self.channel!r}, "
            f"before_gate={self.before_gate}{label_part})"
        )


# =============================================================================
# NoiseModel
# =============================================================================


class NoiseModel:
    """A declarative noise model — a collection of noise insertion rules.

    NoiseModel describes a physical noise environment: which qubits
    experience which types of noise under which conditions.

    It provides two primary interfaces:
    1. ``apply(circuit) → QuantumCircuit`` — transform a clean circuit into
       a noisy one by inserting channels at matched locations.
    2. Iteration over ``rules`` — for simulator-side application.

    Rules are applied sequentially in the order they were added.

    Example:
        >>> from quspect.noise import NoiseModel, DepolarizingChannel
        >>> model = NoiseModel(name="ibm_like")
        >>> # Single-qubit gate depolarization at 0.1%
        >>> _ = model.add_all_qubit_gate_error(
        ...     DepolarizingChannel(name="dp", num_qubits=1, p=0.001),
        ...     gate_num_qubits=1,
        ... )
        >>> # Two-qubit gate depolarization at 1%
        >>> _ = model.add_all_qubit_gate_error(
        ...     DepolarizingChannel(name="dp2", num_qubits=2, p=0.01),
        ...     gate_num_qubits=2,
        ... )
    """

    def __init__(self, *, name: str = "") -> None:
        """Initialize an empty noise model.

        Args:
            name: Optional human-readable identifier.
        """
        self._name = name
        self._rules: list[NoiseRule] = []

    @property
    def name(self) -> str:
        """Return the model name."""
        return self._name

    @property
    def rules(self) -> tuple[NoiseRule, ...]:
        """Return an immutable view of all rules."""
        return tuple(self._rules)

    def add_rule(self, rule: NoiseRule) -> NoiseModel:
        """Add a custom noise rule.

        Args:
            rule: The noise insertion rule.

        Returns:
            Self, for method chaining.
        """
        self._rules.append(rule)
        return self

    def add_all_qubit_gate_error(
        self,
        channel: NoiseChannel,
        *,
        gate_names: list[str] | None = None,
        gate_num_qubits: int | None = None,
        before_gate: bool = False,
    ) -> NoiseModel:
        """Add the same noise channel after ALL gates matching the filter.

        Args:
            channel: The noise channel to insert.
            gate_names: If provided, only match gates with these names.
            gate_num_qubits: If provided, only match gates with this many
                qubits.
            before_gate: If True, insert noise BEFORE the gate.

        Returns:
            Self, for method chaining.
        """
        rule = NoiseRule(
            channel=channel,
            gate_filter=_make_gate_filter(gate_names, gate_num_qubits),
            before_gate=before_gate,
        )
        return self.add_rule(rule)

    def add_qubit_gate_error(
        self,
        channel: NoiseChannel,
        qubits: Sequence[int],
        *,
        gate_names: list[str] | None = None,
        gate_num_qubits: int | None = None,
        before_gate: bool = False,
    ) -> NoiseModel:
        """Add noise only for gates touching specific qubits.

        Args:
            channel: The noise channel to insert.
            qubits: Only apply noise to gates that involve these qubits.
            gate_names: Optional gate name filter.
            gate_num_qubits: Optional gate size filter.
            before_gate: Insert before (vs after) the gate.

        Returns:
            Self, for method chaining.
        """
        qubit_set = set(qubits)
        rule = NoiseRule(
            channel=channel,
            gate_filter=_make_gate_filter(gate_names, gate_num_qubits),
            qubit_filter=lambda q: q in qubit_set,
            before_gate=before_gate,
        )
        return self.add_rule(rule)

    def add_readout_error(
        self,
        p0_given_1: float,
        p1_given_0: float,
        *,
        qubits: Sequence[int] | None = None,
    ) -> NoiseModel:
        """Add readout/measurement error.

        Inserts, immediately before every matching single-qubit
        measurement, a 1-qubit Kraus channel that reproduces the readout
        confusion matrix exactly for a Z-basis measurement:

        - ``K0 = sqrt(1 - p1_given_0) |0><0|``
        - ``K1 = sqrt(p1_given_0) |1><0|``
        - ``K2 = sqrt(1 - p0_given_1) |1><1|``
        - ``K3 = sqrt(p0_given_1) |0><1|``

        so that ``P(read 1 | 0) = p1_given_0`` and
        ``P(read 0 | 1) = p0_given_1``.  A measurement of several qubits
        receives one channel per (filtered) measured qubit.

        Args:
            p0_given_1: Probability of measuring 0 when the state was |1⟩.
            p1_given_0: Probability of measuring 1 when the state was |0⟩.
            qubits: Which qubits this applies to (None = all).

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If a probability is outside ``[0, 1]``.

        Example:
            >>> from quspect import QuantumCircuit
            >>> model = NoiseModel().add_readout_error(0.02, 0.01)
            >>> noisy = model.apply(QuantumCircuit().x(0).measure_all())
            >>> [i.channel.name for i in noisy.instructions if i.is_channel]
            ['readout_error']
        """
        from quspect.noise.channel import KrausChannel

        for label, p in (("p0_given_1", p0_given_1), ("p1_given_0", p1_given_0)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {p}")

        candidates = [
            np.sqrt(1.0 - p1_given_0) * np.array([[1, 0], [0, 0]], dtype=complex),
            np.sqrt(p1_given_0) * np.array([[0, 0], [1, 0]], dtype=complex),
            np.sqrt(1.0 - p0_given_1) * np.array([[0, 0], [0, 1]], dtype=complex),
            np.sqrt(p0_given_1) * np.array([[0, 1], [0, 0]], dtype=complex),
        ]
        ch = KrausChannel(
            name="readout_error",
            num_qubits=1,
            kraus_ops=[k for k in candidates if np.any(k)],
        )

        def _is_measure(gate: Gate) -> bool:
            return gate.name == "measure"

        qubit_filter = None
        if qubits is not None:
            qubit_set = set(qubits)

            def _qubit_filter(q: int) -> bool:
                return q in qubit_set

            qubit_filter = _qubit_filter

        rule = NoiseRule(
            channel=ch,
            gate_filter=_is_measure,
            qubit_filter=qubit_filter,
            before_gate=True,  # Before measurement
            on_measure=True,
        )
        return self.add_rule(rule)

    def add_thermal_relaxation(
        self,
        t1: dict[int, float] | float,
        t2: dict[int, float] | float,
        *,
        gate_time: float,
        excited_state_population: float = 0.0,
    ) -> NoiseModel:
        """Add T1/T2 thermal relaxation noise.

        Every gate is followed, on each of its qubits, by a T1 channel and
        (if needed) a pure-dephasing channel such that, over ``gate_time``
        ``t``:

        - the excited population relaxes as ``exp(-t/T1)`` (amplitude
          damping with ``γ = 1 - exp(-t/T1)``; a
          :class:`GeneralizedAmplitudeDampingChannel` with
          ``p_thermal = excited_state_population`` when that is > 0);
        - coherences decay as ``exp(-t/T2)``: the T1 channel contributes
          ``exp(-t/(2 T1))`` and a :class:`PhaseDampingChannel` with
          ``λ = 1 - exp(-2t/T2 + t/T1)`` the rest.  For ``T2 = 2 T1``
          ``λ = 0`` and no phase-damping channel is added.

        A scalar ``t1``/``t2`` applies to **every** qubit.  With dicts,
        only the listed qubits get noise; if one argument is a dict and
        the other a scalar, the scalar is used for all qubits (listed or
        not).  A qubit with a T1 but no T2 uses ``T2 = 2 T1``; a qubit with
        a T2 but no T1 uses ``T1 = inf`` (pure dephasing).

        Args:
            t1: T1 relaxation times (ns) — either dict ``qubit→time`` or
                a single value for all qubits.
            t2: T2 dephasing times (ns) — same format as ``t1``.
            gate_time: Gate duration (ns). Used to compute error rates.
            excited_state_population: Thermal excitation probability
                (equilibrium population of |1⟩), in ``[0, 0.5]``.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If T2 > 2*T1 for any qubit, if a time is not
                positive, or if ``excited_state_population`` is outside
                ``[0, 0.5]``.

        Example:
            >>> from quspect import QuantumCircuit
            >>> model = NoiseModel().add_thermal_relaxation(
            ...     1000.0, 800.0, gate_time=50.0
            ... )
            >>> noisy = model.apply(QuantumCircuit().h(0).h(1))
            >>> sorted({i.qubits for i in noisy.instructions if i.is_channel})
            [(0,), (1,)]
        """
        t1_scalar = None if isinstance(t1, dict) else float(t1)
        t2_scalar = None if isinstance(t2, dict) else float(t2)
        t1_dict = dict(t1) if isinstance(t1, dict) else {}
        t2_dict = dict(t2) if isinstance(t2, dict) else {}
        listed = sorted(set(t1_dict) | set(t2_dict))

        def _times(q: int | None) -> tuple[float, float]:
            t1_q = t1_dict.get(q, t1_scalar) if q is not None else t1_scalar
            t2_q = t2_dict.get(q, t2_scalar) if q is not None else t2_scalar
            t1_val = float("inf") if t1_q is None else float(t1_q)
            t2_val = 2.0 * t1_val if t2_q is None else float(t2_q)
            return t1_val, t2_val

        entries: list[tuple[Callable[[int], bool] | None, str, float, float]] = [
            (_make_qubit_filter(q), f"Qubit {q}", *_times(q)) for q in listed
        ]
        if t1_scalar is not None or t2_scalar is not None:
            # The scalar applies to every qubit not listed explicitly.
            other_filter = _make_exclusion_filter(listed) if listed else None
            entries.append((other_filter, "All qubits", *_times(None)))

        for qubit_filter, who, t1_val, t2_val in entries:
            for chans in self._thermal_channels(
                who, t1_val, t2_val, gate_time, excited_state_population
            ):
                self.add_rule(NoiseRule(channel=chans, qubit_filter=qubit_filter))

        return self

    @staticmethod
    def _thermal_channels(
        who: str,
        t1: float,
        t2: float,
        gate_time: float,
        excited_state_population: float,
    ) -> list[NoiseChannel]:
        """Build the T1 and pure-dephasing channels for one T1/T2 pair.

        Args:
            who: Description of the qubit(s) for error messages.
            t1: T1 time (may be ``inf``).
            t2: T2 time.
            gate_time: Duration over which relaxation acts.
            excited_state_population: Thermal |1⟩ population.

        Returns:
            ``[T1 channel]`` or ``[T1 channel, PhaseDampingChannel]``.

        Raises:
            ValueError: On non-positive times, ``T2 > 2 T1`` or an invalid
                excited-state population.
        """
        from quspect.noise.channel import (
            AmplitudeDampingChannel,
            GeneralizedAmplitudeDampingChannel,
            PhaseDampingChannel,
        )

        if t1 <= 0 or t2 <= 0:
            raise ValueError(f"{who}: T1 ({t1}) and T2 ({t2}) must be positive")
        if t2 > 2 * t1:
            raise ValueError(f"{who}: T2 ({t2}) must be ≤ 2*T1 ({2 * t1})")
        if not 0.0 <= excited_state_population <= 0.5:
            raise ValueError(
                "excited_state_population must be in [0, 0.5], "
                f"got {excited_state_population}"
            )
        gamma_t1 = float(-np.expm1(-gate_time / t1))
        # Pure dephasing so that coherences decay as exp(-t/T2) overall:
        # the T1 channel gives exp(-t/(2 T1)), phase damping sqrt(1 - λ).
        lam = float(-np.expm1(-2.0 * gate_time / t2 + gate_time / t1))
        lam = min(max(lam, 0.0), 1.0)

        channels: list[NoiseChannel] = []
        if excited_state_population > 0:
            channels.append(
                GeneralizedAmplitudeDampingChannel(
                    name="thermal_relax_t1",
                    num_qubits=1,
                    gamma=gamma_t1,
                    p_thermal=excited_state_population,
                )
            )
        else:
            channels.append(
                AmplitudeDampingChannel(
                    name="thermal_relax_t1", num_qubits=1, gamma=gamma_t1
                )
            )
        if lam > 1e-15:
            channels.append(
                PhaseDampingChannel(name="thermal_relax_t2", num_qubits=1, gamma=lam)
            )
        return channels

    def apply(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transform a clean circuit into a noisy one.

        Walks through the circuit instructions and inserts noise
        channels according to the rules.

        Rules are applied sequentially: for each instruction in the
        original circuit, all matching rules are checked and the
        corresponding channels are inserted.

        Args:
            circuit: The original clean quantum circuit.

        Returns:
            A new circuit with noise channels inserted.
        """
        from quspect.noise.adder import NoiseModelAdder

        return NoiseModelAdder(self).apply(circuit)

    def __iter__(self) -> Iterator[NoiseRule]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"NoiseModel(name={self._name!r}, rules={len(self._rules)})"


# =============================================================================
# Built-in noise model constructors
# =============================================================================


def ideal() -> NoiseModel:
    """Return an empty noise model (ideal, noiseless).

    Returns:
        A :class:`NoiseModel` with no rules.
    """
    return NoiseModel(name="ideal")


def depolarizing_model(p_single: float, p_double: float) -> NoiseModel:
    """Construct a simple depolarizing noise model.

    Args:
        p_single: Depolarizing probability for 1-qubit gates.
        p_double: Depolarizing probability for 2-qubit gates.

    Returns:
        A :class:`NoiseModel` with depolarizing errors on all gates.
    """
    from quspect.noise.channel import DepolarizingChannel

    model = NoiseModel(name=f"depolarizing(p1={p_single},p2={p_double})")
    if p_single > 0:
        model.add_all_qubit_gate_error(
            DepolarizingChannel(name="dp_1q", num_qubits=1, p=p_single),
            gate_num_qubits=1,
        )
    if p_double > 0:
        model.add_all_qubit_gate_error(
            DepolarizingChannel(name="dp_2q", num_qubits=2, p=p_double),
            gate_num_qubits=2,
        )
    return model


def thermal_relaxation_model(
    t1: dict[int, float],
    t2: dict[int, float],
    gate_time_1q: float = 50.0,
    gate_time_2q: float = 300.0,
) -> NoiseModel:
    """Construct a thermal relaxation noise model from T1/T2 times.

    Args:
        t1: T1 relaxation times (ns) per qubit.
        t2: T2 dephasing times (ns) per qubit.
        gate_time_1q: Duration of single-qubit gates (ns).
        gate_time_2q: Duration of two-qubit gates (ns).

    Returns:
        A :class:`NoiseModel` with thermal relaxation on all gates.
    """
    model = NoiseModel(name="thermal_relaxation")

    # T1/T2 for 1-qubit gates
    model_1q = NoiseModel()
    model_1q.add_thermal_relaxation(t1, t2, gate_time=gate_time_1q)
    for rule in model_1q:
        # Add gate filter for 1-qubit gates
        model.add_rule(
            NoiseRule(
                channel=rule.channel,
                gate_filter=_make_gate_filter(None, 1),
                qubit_filter=rule.qubit_filter,
                before_gate=rule.before_gate,
            )
        )

    # T1/T2 for 2-qubit gates
    model_2q = NoiseModel()
    model_2q.add_thermal_relaxation(t1, t2, gate_time=gate_time_2q)
    for rule in model_2q:
        model.add_rule(
            NoiseRule(
                channel=rule.channel,
                gate_filter=_make_gate_filter(None, 2),
                qubit_filter=rule.qubit_filter,
                before_gate=rule.before_gate,
            )
        )

    return model


# =============================================================================
# Internal helpers
# =============================================================================


def _make_gate_filter(
    gate_names: list[str] | None,
    num_qubits: int | None,
) -> Callable[[Gate], bool] | None:
    """Build a combined gate filter from name and qubit-count constraints.

    Args:
        gate_names: If provided, match gates with these names.
        num_qubits: If provided, match gates with this many qubits.

    Returns:
        A callable predicate, or None if both are None.
    """
    if gate_names is None and num_qubits is None:
        return None

    name_set = set(gate_names) if gate_names else None

    def _filter(gate: Gate) -> bool:
        return not (
            (name_set is not None and gate.name not in name_set)
            or (num_qubits is not None and gate.num_qubits != num_qubits)
        )

    return _filter


def _make_qubit_filter(target: int) -> Callable[[int], bool]:
    """Create a qubit filter that matches a specific qubit index.

    Args:
        target: The qubit index to match.

    Returns:
        A callable that returns True when called with ``target``.
    """

    def _filter(q: int) -> bool:
        return q == target

    return _filter


def _make_exclusion_filter(excluded: Sequence[int]) -> Callable[[int], bool]:
    """Create a qubit filter that matches every qubit not in ``excluded``.

    Args:
        excluded: Qubit indices to reject.

    Returns:
        A callable returning True for qubits outside ``excluded``.
    """
    excluded_set = frozenset(excluded)

    def _filter(q: int) -> bool:
        return q not in excluded_set

    return _filter
