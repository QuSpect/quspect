"""Tests for ``quspect.noise.noise_model`` and ``quspect.noise.adder``.

Covers NoiseRule, NoiseModel, NoiseModelAdder and the constructors ``ideal``,
``depolarizing_model`` and ``thermal_relaxation_model``.  Placement of the
inserted channels is checked structurally; the resulting noisy circuits are
evolved exactly with qiskit ``DensityMatrix`` (via ``to_qiskit(noise="kraus")``)
and compared with quspect's ``DensityMatrixSimulator`` to 1e-10.  Thermal
relaxation is checked against the analytic T1/T2 formulas.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix

import quspect
from qiskit_tests._noise_helpers import (
    assert_choi_equal,
    choi_of_kraus,
    choi_of_map,
    random_cptp_kraus,
    ref_thermal_relaxation_map,
)
from qiskit_tests.helpers import random_circuit_pair, to_qiskit
from quspect.circuit.instruction import CircuitInstruction
from quspect.gates.standard import CX, CZ, H
from quspect.noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    NoiseChannel,
    NoiseModel,
    NoiseModelAdder,
    NoiseRule,
    PhaseDampingChannel,
    depolarizing_model,
    ideal,
    thermal_relaxation_model,
)
from quspect.simulation import DensityMatrixSimulator

pytestmark = pytest.mark.qiskit_ref

DEP1 = DepolarizingChannel(name="dp_1q", num_qubits=1, p=0.05)
DEP2 = DepolarizingChannel(name="dp_2q", num_qubits=2, p=0.1)


def _summary(circ: quspect.QuantumCircuit) -> list[tuple[str, str, tuple[int, ...]]]:
    """(kind, name, qubits) for every instruction."""
    out: list[tuple[str, str, tuple[int, ...]]] = []
    for i in circ.instructions:
        if i.is_gate and i.gate is not None:
            out.append(("gate", i.gate.name, i.qubits))
        elif i.is_channel and i.channel is not None:
            out.append(("chan", i.channel.name, i.qubits))
        elif i.is_measure:
            out.append(("meas", "measure", i.qubits))
        else:
            out.append(("barrier", "barrier", i.qubits))
    return out


def _assert_rho_matches_qiskit(circ: quspect.QuantumCircuit) -> None:
    """DensityMatrixSimulator rho == qiskit DensityMatrix evolution (1e-10)."""
    rho = np.asarray(DensityMatrixSimulator(seed=0).density_matrix(circ)._data)
    ref = QiskitDensityMatrix(to_qiskit(circ, noise="kraus")).data
    np.testing.assert_allclose(rho, ref, atol=1e-10, rtol=0)


# =============================================================================
# NoiseRule
# =============================================================================


def test_noise_rule_defaults_match_everything() -> None:
    """No filters: fires for any gate; targets are the gate's qubits."""
    rule = NoiseRule(DEP1)
    assert rule.matches_gate(H, (3,))
    assert rule.matches_gate(CX, (2, 0))
    assert rule.get_target_qubits((2, 0)) == (2, 0)
    assert rule.before_gate is False
    assert rule.focus_on is None
    assert rule.label == ""


def test_noise_rule_filters() -> None:
    """gate_filter AND any-qubit qubit_filter AND NOT exclude_filter."""
    rule = NoiseRule(
        DEP1,
        gate_filter=lambda g: g.num_qubits == 2,
        qubit_filter=lambda q: q == 1,
        exclude_filter=lambda g: g.name == "cz",
    )
    cz = CZ
    assert rule.matches_gate(CX, (0, 1))
    assert rule.matches_gate(CX, (1, 2))
    assert not rule.matches_gate(CX, (0, 2))  # no qubit matches
    assert not rule.matches_gate(H, (1,))  # gate filter fails
    assert not rule.matches_gate(cz, (0, 1))  # excluded
    only_exclude = NoiseRule(DEP1, exclude_filter=lambda g: g.name == "h")
    assert not only_exclude.matches_gate(H, (0,))
    assert only_exclude.matches_gate(CX, (0, 1))


def test_noise_rule_focus_on_and_repr() -> None:
    """focus_on overrides the target qubits; repr shows channel and label."""
    rule = NoiseRule(DEP1, focus_on=(4,), before_gate=True, label="r1")
    assert rule.get_target_qubits((0, 1)) == (4,)
    r = repr(rule)
    assert r.startswith("NoiseRule(channel=DepolarizingChannel(")
    assert "before_gate=True" in r and "label='r1'" in r
    assert "label='" not in repr(NoiseRule(DEP1)).split("p=0.05)")[-1]


# =============================================================================
# NoiseModel container API
# =============================================================================


def test_noise_model_container_api() -> None:
    """name, rules (immutable tuple), add_rule chaining, iter, len, repr."""
    model = NoiseModel(name="m")
    assert model.name == "m"
    assert len(model) == 0
    assert model.rules == ()
    r1, r2 = NoiseRule(DEP1), NoiseRule(DEP2)
    assert model.add_rule(r1).add_rule(r2) is model
    assert model.rules == (r1, r2)
    assert list(model) == [r1, r2]
    assert len(model) == 2
    assert repr(model) == "NoiseModel(name='m', rules=2)"
    rules = model.rules
    assert isinstance(rules, tuple)
    assert NoiseModel().name == ""


def test_ideal_model_is_identity() -> None:
    """ideal() has no rules and leaves the circuit unchanged."""
    model = ideal()
    assert model.name == "ideal" and len(model) == 0
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).measure_all()
    assert model.apply(circ).instructions == circ.instructions


def test_depolarizing_model_structure() -> None:
    """depolarizing_model builds 1q/2q rules; zero strengths are skipped."""
    model = depolarizing_model(0.01, 0.02)
    assert model.name == "depolarizing(p1=0.01,p2=0.02)"
    chans = [r.channel for r in model]
    assert chans == [
        DepolarizingChannel(name="dp_1q", num_qubits=1, p=0.01),
        DepolarizingChannel(name="dp_2q", num_qubits=2, p=0.02),
    ]
    assert len(depolarizing_model(0.0, 0.02)) == 1
    assert len(depolarizing_model(0.01, 0.0)) == 1
    assert len(depolarizing_model(0.0, 0.0)) == 0


# =============================================================================
# Adder: placement
# =============================================================================


def test_adder_inserts_after_matching_gates_on_same_qubits() -> None:
    """Each gate is followed by its channel on exactly the gate's qubits."""
    circ = quspect.QuantumCircuit(name="c").h(2).cx(2, 0).barrier().rz(0.3, 1)
    circ.ccx(0, 1, 2).measure_all()
    noisy = depolarizing_model(0.05, 0.1).apply(circ)
    assert noisy.name == "c"
    assert _summary(noisy) == [
        ("gate", "h", (2,)),
        ("chan", "dp_1q", (2,)),
        ("gate", "cx", (2, 0)),
        ("chan", "dp_2q", (2, 0)),
        ("barrier", "barrier", (0, 2)),  # active qubits
        ("gate", "rz", (1,)),
        ("chan", "dp_1q", (1,)),
        ("gate", "ccx", (0, 1, 2)),  # 3q gate: no matching rule
        ("meas", "measure", (0, 1, 2)),
    ]
    # Original circuit untouched.
    assert len(circ.instructions) == 6


def test_adder_before_gate_order_and_existing_channels() -> None:
    """before_gate rules precede the gate; rule order is preserved; existing
    channels and measurements pass through unchanged; labels are copied."""
    ad = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.1)
    bf = BitFlipChannel(name="bf", num_qubits=1, p=0.2)
    model = NoiseModel()
    model.add_all_qubit_gate_error(DEP1, gate_names=["x"], before_gate=True)
    model.add_rule(NoiseRule(ad, gate_filter=lambda g: g.name == "x", label="t1"))
    model.add_rule(NoiseRule(bf, gate_filter=lambda g: g.name == "x"))
    circ = quspect.QuantumCircuit().x(1).phase_flip(0.3, 1).h(0)
    circ.measure([1], [0])
    noisy = NoiseModelAdder(model).apply(circ)
    assert _summary(noisy) == [
        ("chan", "dp_1q", (1,)),
        ("gate", "x", (1,)),
        ("chan", "ad", (1,)),
        ("chan", "bf", (1,)),
        ("chan", "phase_flip", (1,)),
        ("gate", "h", (0,)),
        ("meas", "measure", (1,)),
    ]
    labels = [i.label for i in noisy.instructions if i.is_channel]
    assert labels == ["", "t1", "", ""]
    assert noisy.instructions == model.apply(circ).instructions


def test_add_qubit_gate_error_any_qubit_semantics() -> None:
    """add_qubit_gate_error fires when *any* gate qubit is listed."""
    model = NoiseModel().add_qubit_gate_error(DEP2, [1], gate_num_qubits=2)
    model.add_qubit_gate_error(DEP1, [0, 3], gate_names=["h", "x"])
    circ = quspect.QuantumCircuit().cx(0, 1).cx(3, 2).cx(1, 3).h(0).x(3).y(0)
    assert _summary(model.apply(circ)) == [
        ("gate", "cx", (0, 1)),
        ("chan", "dp_2q", (0, 1)),
        ("gate", "cx", (3, 2)),
        ("gate", "cx", (1, 3)),
        ("chan", "dp_2q", (1, 3)),
        ("gate", "h", (0,)),
        ("chan", "dp_1q", (0,)),
        ("gate", "x", (3,)),
        ("chan", "dp_1q", (3,)),
        ("gate", "y", (0,)),
    ]


def test_focus_on_and_exclude_filter_placement() -> None:
    """focus_on picks absolute qubits; exclude_filter vetoes a gate."""
    model = NoiseModel()
    model.add_rule(NoiseRule(DEP1, gate_filter=lambda g: g.name == "cx", focus_on=(1,)))
    model.add_rule(NoiseRule(DEP1, exclude_filter=lambda g: g.num_qubits != 1))
    circ = quspect.QuantumCircuit().cx(0, 2).h(1)
    assert _summary(model.apply(circ)) == [
        ("gate", "cx", (0, 2)),
        ("chan", "dp_1q", (1,)),
        ("gate", "h", (1,)),
        ("chan", "dp_1q", (1,)),
    ]


@pytest.mark.parametrize(
    "model",
    [
        NoiseModel().add_all_qubit_gate_error(DEP1),
        NoiseModel().add_qubit_gate_error(DEP1, [1]),
    ],
    ids=["all_qubit", "qubit_specific"],
)
def test_adder_channel_arity_matches_qubits(model: NoiseModel) -> None:
    """Every inserted channel acts on exactly channel.num_qubits qubits."""
    noisy = model.apply(quspect.QuantumCircuit().h(0).cx(0, 1))
    for instr in noisy.instructions:
        if instr.is_channel and instr.channel is not None:
            assert len(instr.qubits) == instr.channel.num_qubits
    DensityMatrixSimulator().density_matrix(noisy)


# =============================================================================
# Adder: exact density-matrix agreement with qiskit
# =============================================================================


@pytest.mark.parametrize("seed", range(4))
def test_depolarizing_model_density_matrix_matches_qiskit(seed: int) -> None:
    """Random 3-qubit circuit + depolarizing_model: rho agrees to 1e-10."""
    circ, _ = random_circuit_pair(3, 25, 500 + seed)
    noisy = depolarizing_model(0.03, 0.08).apply(circ)
    assert sum(i.is_channel for i in noisy.instructions) > 10
    _assert_rho_matches_qiskit(noisy)


def test_mixed_model_density_matrix_matches_qiskit() -> None:
    """A model mixing GAD / PD / asymmetric 2q Kraus / before-gate rules."""
    k2 = KrausChannel(name="k2", num_qubits=2, kraus_ops=random_cptp_kraus(2, 2, 7))
    model = NoiseModel(name="mixed")
    model.add_all_qubit_gate_error(
        GeneralizedAmplitudeDampingChannel(
            name="gad", num_qubits=1, gamma=0.1, p_thermal=0.2
        ),
        gate_num_qubits=1,
    )
    model.add_qubit_gate_error(
        PhaseDampingChannel(name="pd", num_qubits=1, gamma=0.2),
        [2],
        gate_num_qubits=1,
        before_gate=True,
    )
    model.add_all_qubit_gate_error(k2, gate_names=["cx", "cz"])
    model.add_all_qubit_gate_error(DEP2, gate_names=["swap"], before_gate=True)
    circ, _ = random_circuit_pair(
        4, 30, 77, gate_set=["h", "rx", "ry", "u3", "cx", "cz", "swap", "s"]
    )
    noisy = model.apply(circ)
    names = {i.channel.name for i in noisy.instructions if i.channel is not None}
    assert names == {"gad", "pd", "k2", "dp_2q"}
    _assert_rho_matches_qiskit(noisy)


def test_thermal_model_on_single_qubit_gates_matches_qiskit() -> None:
    """thermal_relaxation_model on a 1q-gate-only circuit matches qiskit."""
    model = thermal_relaxation_model(
        {0: 4000.0, 1: 2500.0, 2: 6000.0}, {0: 3000.0, 1: 4000.0, 2: 1500.0}
    )
    circ, _ = random_circuit_pair(3, 20, 3, gate_set=["h", "rx", "u3", "t", "y"])
    _assert_rho_matches_qiskit(model.apply(circ))


# =============================================================================
# Thermal relaxation
# =============================================================================


def _channels_on(circ: quspect.QuantumCircuit, qubit: int) -> list[NoiseChannel]:
    """Channels inserted on ``qubit`` (in order)."""
    return [
        i.channel
        for i in circ.instructions
        if i.channel is not None and i.qubits == (qubit,)
    ]


def _compose_kraus(channels: list[NoiseChannel]) -> list[np.ndarray]:
    """Kraus operators of the sequential composition of 1q channels."""
    ops: list[np.ndarray] = [np.eye(2, dtype=complex)]
    for ch in channels:
        ops = [k @ o for k in ch.kraus_operators() for o in ops]
    return ops


@pytest.mark.parametrize(("t1", "time"), [(1000.0, 50.0), (200.0, 300.0)])
def test_thermal_relaxation_t1_population(t1: float, time: float) -> None:
    """|1> population after one gate is exp(-t/T1) (T1 part is correct)."""
    model = NoiseModel().add_thermal_relaxation({0: t1}, {0: 0.8 * t1}, gate_time=time)
    circ = model.apply(quspect.QuantumCircuit().x(0))
    ad = _channels_on(circ, 0)[0]
    assert isinstance(ad, AmplitudeDampingChannel)
    assert ad.gamma == pytest.approx(1 - math.exp(-time / t1), abs=1e-14)
    rho = DensityMatrixSimulator().density_matrix(circ)._data
    assert rho[1, 1].real == pytest.approx(math.exp(-time / t1), abs=1e-12)


@pytest.mark.parametrize(
    ("t1", "t2"), [(1000.0, 800.0), (1000.0, 1500.0), (1000.0, 2000.0)]
)
def test_thermal_relaxation_matches_t1_t2_formula(t1: float, t2: float) -> None:
    """AD + PD per gate == analytic map (coherence decays as exp(-t/T2))."""
    time = 50.0
    model = NoiseModel().add_thermal_relaxation(t1, t2, gate_time=time)
    circ = model.apply(quspect.QuantumCircuit().h(0))
    rho = DensityMatrixSimulator().density_matrix(circ)._data
    assert 2 * abs(rho[0, 1]) == pytest.approx(math.exp(-time / t2), abs=1e-12)
    assert_choi_equal(
        choi_of_kraus(_compose_kraus(_channels_on(circ, 0))),
        choi_of_map(ref_thermal_relaxation_map(t1, t2, time), 2),
    )


def test_thermal_relaxation_rejects_t2_above_2t1() -> None:
    """T2 > 2 T1 is unphysical and raises ValueError."""
    with pytest.raises(ValueError, match="T2"):
        NoiseModel().add_thermal_relaxation(100.0, 201.0, gate_time=10.0)
    with pytest.raises(ValueError):
        thermal_relaxation_model({0: 100.0}, {0: 250.0})


def test_thermal_relaxation_per_qubit_dict() -> None:
    """Dict T1/T2: each qubit gets its own rates; other qubits get none."""
    model = NoiseModel().add_thermal_relaxation(
        {0: 1000.0, 2: 500.0}, {0: 1500.0, 2: 600.0}, gate_time=40.0
    )
    assert len(model) == 4
    circ = model.apply(quspect.QuantumCircuit().h(0).h(1).h(2))
    ad0, pd0 = _channels_on(circ, 0)
    assert isinstance(ad0, AmplitudeDampingChannel)
    assert isinstance(pd0, PhaseDampingChannel)
    assert ad0.gamma == pytest.approx(1 - math.exp(-40.0 / 1000.0))
    assert _channels_on(circ, 1) == []
    ad2, pd2 = _channels_on(circ, 2)
    assert isinstance(ad2, AmplitudeDampingChannel)
    assert isinstance(pd2, PhaseDampingChannel)
    assert ad2.gamma == pytest.approx(1 - math.exp(-40.0 / 500.0))


def test_thermal_relaxation_scalar_applies_to_all_qubits() -> None:
    """A scalar T1/T2 should add relaxation after gates on every qubit."""
    model = NoiseModel().add_thermal_relaxation(1000.0, 800.0, gate_time=50.0)
    circ = model.apply(quspect.QuantumCircuit().h(0).h(1).h(2))
    for q in range(3):
        assert len(_channels_on(circ, q)) == 2


def test_thermal_relaxation_excited_state_population() -> None:
    """With p1 > 0 the |1> population relaxes towards p1."""
    t1, time, p1 = 100.0, 50.0, 0.2
    model = NoiseModel().add_thermal_relaxation(
        {0: t1}, {0: 2 * t1}, gate_time=time, excited_state_population=p1
    )
    rho = (
        DensityMatrixSimulator()
        .density_matrix(model.apply(quspect.QuantumCircuit().id(0)))
        ._data
    )
    assert rho[1, 1].real == pytest.approx(p1 * (1 - math.exp(-time / t1)))


def test_thermal_relaxation_model_single_qubit_gates() -> None:
    """thermal_relaxation_model: gate_time_1q sets the 1q-gate damping."""
    model = thermal_relaxation_model(
        {0: 1000.0, 1: 2000.0}, {0: 900.0, 1: 3000.0}, gate_time_1q=30.0
    )
    assert model.name == "thermal_relaxation"
    # (AD + PD) x 2 qubits x {1q-gate, 2q-gate} rules.
    assert len(model) == 8
    circ = model.apply(quspect.QuantumCircuit().x(0).x(1))
    ad0 = _channels_on(circ, 0)[0]
    assert isinstance(ad0, AmplitudeDampingChannel)
    assert ad0.gamma == pytest.approx(1 - math.exp(-30.0 / 1000.0))
    ad1, _ = _channels_on(circ, 1)
    assert isinstance(ad1, AmplitudeDampingChannel)
    assert ad1.gamma == pytest.approx(1 - math.exp(-30.0 / 2000.0))
    rho = DensityMatrixSimulator().density_matrix(circ)._data
    p11 = math.exp(-30.0 / 1000.0) * math.exp(-30.0 / 2000.0)
    assert rho[3, 3].real == pytest.approx(p11, abs=1e-12)


def test_thermal_relaxation_model_two_qubit_gate() -> None:
    """After cx(0, 1) each qubit should relax with gate_time_2q."""
    t1, t2 = {0: 1000.0, 1: 1000.0}, {0: 2000.0, 1: 2000.0}
    model = thermal_relaxation_model(t1, t2, gate_time_2q=300.0)
    circ = model.apply(quspect.QuantumCircuit().x(0).cx(0, 1))
    rho = DensityMatrixSimulator().density_matrix(circ)._data
    p1 = math.exp(-50.0 / 1000.0)
    assert rho[3, 3].real == pytest.approx(p1 * math.exp(-600.0 / 1000.0))


# =============================================================================
# Readout error
# =============================================================================


def test_readout_error_inserted_before_measurement() -> None:
    """Readout error: a readout channel precedes each measurement."""
    model = NoiseModel().add_readout_error(0.1, 0.1)
    noisy = model.apply(quspect.QuantumCircuit().x(0).measure_all())
    kinds = [k for k, _, _ in _summary(noisy)]
    assert kinds == ["gate", "chan", "meas"]


def test_readout_error_rule_definition() -> None:
    """The readout rule is a before-measure Kraus channel with the exact
    asymmetric confusion matrix P(1|0) = p1_given_0, P(0|1) = p0_given_1."""
    model = NoiseModel().add_readout_error(0.1, 0.3, qubits=[1])
    (rule,) = model.rules
    assert rule.before_gate is True
    assert rule.on_measure is True
    assert isinstance(rule.channel, KrausChannel)
    assert rule.channel.name == "readout_error"
    # Z-basis confusion matrix of the channel: M[out, in] = <out|E(|in><in|)|out>.
    kraus = rule.channel.kraus_operators()
    confusion = np.array(
        [[sum(abs(k[o, i]) ** 2 for k in kraus) for i in range(2)] for o in range(2)]
    )
    np.testing.assert_allclose(confusion, [[0.7, 0.1], [0.3, 0.9]], atol=1e-14)
    # Duck-typed stand-in: the rule only inspects ``gate.name``.
    meas_like: Any = SimpleNamespace(name="measure", num_qubits=1)
    assert rule.matches_gate(meas_like, (1,))
    assert not rule.matches_gate(meas_like, (0,))
    assert not rule.matches_gate(H, (1,))
    assert rule.matches_measure((0, 1))
    assert not rule.matches_measure((0,))
    unrestricted = NoiseModel().add_readout_error(0.0, 0.2).rules[0]
    assert unrestricted.qubit_filter is None


def test_adder_returns_valid_instructions() -> None:
    """Inserted channel instructions are proper CircuitInstruction objects."""
    noisy = depolarizing_model(0.1, 0.1).apply(quspect.QuantumCircuit().h(0))
    chan = noisy.instructions[1]
    assert isinstance(chan, CircuitInstruction)
    assert chan.is_channel and chan.channel == DepolarizingChannel(
        name="dp_1q", num_qubits=1, p=0.1
    )
