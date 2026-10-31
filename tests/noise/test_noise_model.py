"""Regression tests for ``quspect.noise.noise_model`` and the adder."""

from __future__ import annotations

import math

import pytest

from quspect import QuantumCircuit
from quspect.noise import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    NoiseModel,
    PhaseDampingChannel,
    thermal_relaxation_model,
)
from quspect.noise.channel import NoiseChannel
from quspect.simulation.density_simulator import DensityMatrixSimulator

DEP1 = DepolarizingChannel(name="dp_1q", num_qubits=1, p=0.05)
DEP2 = DepolarizingChannel(name="dp_2q", num_qubits=2, p=0.05)


def _channels(circ: QuantumCircuit) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (i.channel.name, i.qubits)
        for i in circ.instructions
        if i.is_channel and i.channel is not None
    ]


def _channels_on(circ: QuantumCircuit, q: int) -> list[NoiseChannel]:
    return [
        i.channel
        for i in circ.instructions
        if i.channel is not None and i.qubits == (q,)
    ]


class TestAdderArity:
    """1-qubit channels on multi-qubit gates are split per qubit."""

    def test_all_qubit_rule_split(self) -> None:
        noisy = (
            NoiseModel()
            .add_all_qubit_gate_error(DEP1)
            .apply(QuantumCircuit().h(0).cx(0, 1))
        )
        assert _channels(noisy) == [("dp_1q", (0,)), ("dp_1q", (0,)), ("dp_1q", (1,))]
        DensityMatrixSimulator().density_matrix(noisy)

    def test_qubit_specific_rule_only_on_matching_qubit(self) -> None:
        noisy = (
            NoiseModel()
            .add_qubit_gate_error(DEP1, [1])
            .apply(QuantumCircuit().h(0).cx(0, 1))
        )
        assert _channels(noisy) == [("dp_1q", (1,))]

    def test_matching_width_kept_whole(self) -> None:
        noisy = (
            NoiseModel()
            .add_all_qubit_gate_error(DEP2, gate_num_qubits=2)
            .apply(QuantumCircuit().cx(1, 0))
        )
        assert _channels(noisy) == [("dp_2q", (1, 0))]

    def test_mismatched_width_raises(self) -> None:
        with pytest.raises(ValueError, match="acts on 2 qubits"):
            NoiseModel().add_all_qubit_gate_error(DEP2).apply(QuantumCircuit().h(0))


class TestThermalRelaxation:
    """T1/T2 formulas, scalar times and thermal population."""

    @pytest.mark.parametrize("t2", [800.0, 1500.0, 2000.0])
    def test_coherence_decays_as_exp_t_over_t2(self, t2: float) -> None:
        t1, time = 1000.0, 50.0
        circ = (
            NoiseModel()
            .add_thermal_relaxation(t1, t2, gate_time=time)
            .apply(QuantumCircuit().h(0))
        )
        rho = DensityMatrixSimulator().density_matrix(circ)._data
        assert 2 * abs(rho[0, 1]) == pytest.approx(math.exp(-time / t2), abs=1e-12)
        kinds = [type(c) for c in _channels_on(circ, 0)]
        if t2 == 2 * t1:
            assert kinds == [AmplitudeDampingChannel]
        else:
            assert kinds == [AmplitudeDampingChannel, PhaseDampingChannel]

    def test_scalar_applies_to_all_qubits(self) -> None:
        model = NoiseModel().add_thermal_relaxation(1000.0, 800.0, gate_time=50.0)
        circ = model.apply(QuantumCircuit().h(0).h(1).h(5))
        for q in (0, 1, 5):
            assert len(_channels_on(circ, q)) == 2

    def test_dict_and_scalar_mix(self) -> None:
        """Dict T1 for qubit 0 + scalar T2 for everyone."""
        model = NoiseModel().add_thermal_relaxation({0: 500.0}, 400.0, gate_time=10.0)
        circ = model.apply(QuantumCircuit().h(0).h(1))
        ad0 = _channels_on(circ, 0)[0]
        assert isinstance(ad0, AmplitudeDampingChannel)
        assert ad0.gamma == pytest.approx(1 - math.exp(-10.0 / 500.0))
        # qubit 1 has no T1 (inf) -> pure dephasing with T2 = 400.
        ad1, pd1 = _channels_on(circ, 1)
        assert isinstance(ad1, AmplitudeDampingChannel) and ad1.gamma == 0.0
        assert isinstance(pd1, PhaseDampingChannel)
        assert pd1.gamma == pytest.approx(1 - math.exp(-20.0 / 400.0))

    def test_excited_state_population(self) -> None:
        t1, time, p1 = 100.0, 50.0, 0.2
        model = NoiseModel().add_thermal_relaxation(
            {0: t1}, {0: 2 * t1}, gate_time=time, excited_state_population=p1
        )
        circ = model.apply(QuantumCircuit().id(0))
        (gad,) = _channels_on(circ, 0)
        assert isinstance(gad, GeneralizedAmplitudeDampingChannel)
        rho = DensityMatrixSimulator().density_matrix(circ)._data
        assert rho[1, 1].real == pytest.approx(p1 * (1 - math.exp(-time / t1)))

    @pytest.mark.parametrize(
        ("t1", "t2", "kwargs"),
        [
            (100.0, 201.0, {}),
            (0.0, 0.0, {}),
            (100.0, 100.0, {"excited_state_population": 0.6}),
        ],
    )
    def test_invalid_parameters(
        self, t1: float, t2: float, kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError):
            NoiseModel().add_thermal_relaxation(t1, t2, gate_time=10.0, **kwargs)

    def test_thermal_model_two_qubit_gate(self) -> None:
        """After cx(0, 1) each qubit relaxes once with gate_time_2q."""
        model = thermal_relaxation_model(
            {0: 1000.0, 1: 1000.0}, {0: 2000.0, 1: 2000.0}, gate_time_2q=300.0
        )
        circ = model.apply(QuantumCircuit().x(0).cx(0, 1))
        rho = DensityMatrixSimulator().density_matrix(circ)._data
        p1 = math.exp(-50.0 / 1000.0)
        assert rho[3, 3].real == pytest.approx(p1 * math.exp(-600.0 / 1000.0))


class TestReadoutError:
    """add_readout_error inserts an exact confusion channel before measure."""

    def test_inserted_before_measurement(self) -> None:
        noisy = (
            NoiseModel()
            .add_readout_error(0.1, 0.2)
            .apply(QuantumCircuit().x(0).x(1).measure_all())
        )
        kinds = [
            "chan" if i.is_channel else "meas" if i.is_measure else "gate"
            for i in noisy.instructions
        ]
        assert kinds == ["gate", "gate", "chan", "chan", "meas"]
        assert all(isinstance(c, KrausChannel) for c in _channels_on(noisy, 0))

    def test_qubit_filter(self) -> None:
        noisy = (
            NoiseModel()
            .add_readout_error(0.1, 0.2, qubits=[1])
            .apply(QuantumCircuit().x(0).id(1).measure_all())
        )
        assert _channels(noisy) == [("readout_error", (1,))]

    @pytest.mark.parametrize("prep", ["0", "1"])
    def test_confusion_matrix_is_exact(self, prep: str) -> None:
        p0_given_1, p1_given_0 = 0.05, 0.25
        circ = QuantumCircuit().id(0)
        if prep == "1":
            circ = QuantumCircuit().x(0)
        noisy = (
            NoiseModel()
            .add_readout_error(p0_given_1, p1_given_0)
            .apply(circ.measure([0], [0]))
        )
        rho = DensityMatrixSimulator().density_matrix(noisy)._data
        expected_p1 = p1_given_0 if prep == "0" else 1 - p0_given_1
        assert rho[1, 1].real == pytest.approx(expected_p1, abs=1e-12)

    def test_invalid_probability(self) -> None:
        with pytest.raises(ValueError):
            NoiseModel().add_readout_error(1.5, 0.0)

    def test_gate_rules_ignore_measurements(self) -> None:
        circ = QuantumCircuit().h(0).measure_all()
        noisy = NoiseModel().add_all_qubit_gate_error(DEP1).apply(circ)
        assert _channels(noisy) == [("dp_1q", (0,))]
        assert len(noisy.instructions) == 3
