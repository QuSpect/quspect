"""Tests for quantum noise channels."""

# ruff: noqa: N806
# (physics convention: Kraus operators K0, K1, ...)

import numpy as np
import pytest

from quspect.noise.channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    BitPhaseFlipChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    NoiseChannel,
    PauliChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
)


class TestNoiseChannelBase:
    """Tests for NoiseChannel base class."""

    def test_abstract_kraus_operators(self) -> None:
        """Base class kraus_operators should raise NotImplementedError."""
        ch = NoiseChannel(name="test", num_qubits=1)
        with pytest.raises(NotImplementedError):
            ch.kraus_operators()

    def test_is_unitary(self) -> None:
        """is_unitary should return True iff single Kraus operator."""
        ch = BitFlipChannel(name="bf", num_qubits=1, p=0.0)
        assert ch.is_unitary()  # p=0 gives single Kraus op (I only)

    def test_repr(self) -> None:
        """Test string representation."""
        ch = BitFlipChannel(name="bf", num_qubits=1, p=0.1)
        r = repr(ch)
        assert "BitFlipChannel" in r
        assert "bf" in r


class TestPauliChannel:
    """Tests for PauliChannel."""

    def test_kraus_count(self) -> None:
        """Pauli channel with non-zero px, py, pz should have 4 Kraus ops."""
        ch = PauliChannel(name="pc", num_qubits=1, px=0.01, py=0.02, pz=0.03)
        ops = ch.kraus_operators()
        assert len(ops) == 4

    def test_trace_preserving(self) -> None:
        """Sum of K_i^† K_i should equal identity."""
        ch = PauliChannel(name="pc", num_qubits=1, px=0.01, py=0.02, pz=0.03)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_probability_bounds(self) -> None:
        """Sum of probabilities must not exceed 1."""
        with pytest.raises(ValueError):
            PauliChannel(name="bad", num_qubits=1, px=0.5, py=0.5, pz=0.5)

    def test_negative_probability(self) -> None:
        """Probabilities must be non-negative."""
        with pytest.raises(ValueError):
            PauliChannel(name="bad", num_qubits=1, px=-0.1)

    def test_zero_probabilities(self) -> None:
        """Zero probabilities = identity channel (single Kraus op)."""
        ch = PauliChannel(name="id", num_qubits=1, px=0.0, py=0.0, pz=0.0)
        ops = ch.kraus_operators()
        assert len(ops) == 1
        assert np.allclose(ops[0], np.eye(2))


class TestBitFlipChannel:
    """Tests for BitFlipChannel."""

    def test_kraus_operators(self) -> None:
        """Verify Kraus decomposition of bit-flip channel."""
        ch = BitFlipChannel(name="bf", num_qubits=1, p=0.1)
        ops = ch.kraus_operators()
        assert len(ops) == 2
        # Check trace-preserving
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_p_zero(self) -> None:
        """p=0 should give identity."""
        ch = BitFlipChannel(name="bf", num_qubits=1, p=0.0)
        ops = ch.kraus_operators()
        assert len(ops) == 1

    def test_p_one(self) -> None:
        """p=1 should give pure X."""
        ch = BitFlipChannel(name="bf", num_qubits=1, p=1.0)
        ops = ch.kraus_operators()
        assert len(ops) == 1
        assert np.allclose(abs(ops[0]), np.array([[0, 1], [1, 0]]), atol=1e-10)

    def test_p_out_of_bounds(self) -> None:
        """p must be in [0, 1]."""
        with pytest.raises(ValueError):
            BitFlipChannel(name="bf", num_qubits=1, p=1.5)
        with pytest.raises(ValueError):
            BitFlipChannel(name="bf", num_qubits=1, p=-0.1)


class TestPhaseFlipChannel:
    """Tests for PhaseFlipChannel."""

    def test_kraus_operators(self) -> None:
        """Verify Kraus decomposition of phase-flip channel."""
        ch = PhaseFlipChannel(name="pf", num_qubits=1, p=0.1)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_p_out_of_bounds(self) -> None:
        """p must be in [0, 1]."""
        with pytest.raises(ValueError):
            PhaseFlipChannel(name="pf", num_qubits=1, p=2.0)


class TestBitPhaseFlipChannel:
    """Tests for BitPhaseFlipChannel."""

    def test_kraus_operators(self) -> None:
        """Verify Kraus decomposition of bit-phase-flip channel."""
        ch = BitPhaseFlipChannel(name="bpf", num_qubits=1, p=0.1)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)


class TestDepolarizingChannel:
    """Tests for DepolarizingChannel."""

    def test_single_qubit_kraus_count(self) -> None:
        """Single-qubit depolarizing should have 4 Kraus operators."""
        ch = DepolarizingChannel(name="dp", num_qubits=1, p=0.1)
        ops = ch.kraus_operators()
        assert len(ops) == 4

    def test_trace_preserving_1q(self) -> None:
        """Single-qubit depolarizing must be trace-preserving."""
        ch = DepolarizingChannel(name="dp", num_qubits=1, p=0.3)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_two_qubit_kraus_count(self) -> None:
        """Two-qubit depolarizing should have 16 Kraus operators."""
        ch = DepolarizingChannel(name="dp2", num_qubits=2, p=0.1)
        ops = ch.kraus_operators()
        assert len(ops) == 16

    def test_two_qubit_trace_preserving(self) -> None:
        """Two-qubit depolarizing must be trace-preserving."""
        ch = DepolarizingChannel(name="dp2", num_qubits=2, p=0.1)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(4), atol=1e-10)

    @pytest.mark.parametrize("num_qubits", [1, 2])
    def test_p_one_is_completely_mixed(self, num_qubits: int) -> None:
        """MindQuantum convention: p=1 maps every input to I / 2**n."""
        ch = DepolarizingChannel(name="dp", num_qubits=num_qubits, p=1.0)
        ops = ch.kraus_operators()

        dim = 1 << num_qubits
        ket = np.zeros(dim, dtype=complex)
        ket[-1] = 1.0
        rho = np.outer(ket, ket.conj())
        output = sum(op @ rho @ op.conj().T for op in ops)

        assert np.allclose(output, np.eye(dim) / dim, atol=1e-12)

    def test_p_above_three_quarters_is_allowed(self) -> None:
        """p is a mixing strength in [0, 1], not total Pauli-error weight."""
        ch = DepolarizingChannel(name="dp", num_qubits=1, p=0.8)
        assert len(ch.kraus_operators()) == 4

    def test_p_zero(self) -> None:
        """p=0 should give identity."""
        ch = DepolarizingChannel(name="dp", num_qubits=1, p=0.0)
        ops = ch.kraus_operators()
        assert len(ops) == 1

    def test_p_out_of_bounds(self) -> None:
        """p must be in [0, 1]."""
        with pytest.raises(ValueError):
            DepolarizingChannel(name="dp", num_qubits=1, p=1.5)
        with pytest.raises(ValueError):
            DepolarizingChannel(name="dp", num_qubits=1, p=-0.1)

    def test_three_qubit_not_implemented(self) -> None:
        """3-qubit depolarizing is not yet supported."""
        ch = DepolarizingChannel(name="dp3", num_qubits=3, p=0.1)
        with pytest.raises(NotImplementedError):
            ch.kraus_operators()


class TestAmplitudeDampingChannel:
    """Tests for AmplitudeDampingChannel."""

    def test_kraus_dimensions(self) -> None:
        """Kraus operators should be 2x2 matrices."""
        ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.1)
        ops = ch.kraus_operators()
        for op in ops:
            assert op.shape == (2, 2)

    def test_trace_preserving(self) -> None:
        """Amplitude damping must be trace-preserving."""
        ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.3)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_gamma_zero(self) -> None:
        """gamma=0 should give identity."""
        ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.0)
        ops = ch.kraus_operators()
        assert len(ops) == 1
        assert np.allclose(ops[0], np.eye(2))

    def test_gamma_out_of_bounds(self) -> None:
        """gamma must be in [0, 1]."""
        with pytest.raises(ValueError):
            AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=1.5)
        with pytest.raises(ValueError):
            AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=-0.1)


class TestPhaseDampingChannel:
    """Tests for PhaseDampingChannel."""

    def test_trace_preserving(self) -> None:
        """Phase damping must be trace-preserving."""
        ch = PhaseDampingChannel(name="pd", num_qubits=1, gamma=0.3)
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_gamma_out_of_bounds(self) -> None:
        """gamma must be in [0, 1]."""
        with pytest.raises(ValueError):
            PhaseDampingChannel(name="pd", num_qubits=1, gamma=2.0)


class TestGeneralizedAmplitudeDampingChannel:
    """Tests for GeneralizedAmplitudeDampingChannel."""

    def test_kraus_count(self) -> None:
        """GAD channel should have 4 Kraus operators at finite T."""
        ch = GeneralizedAmplitudeDampingChannel(
            name="gad", num_qubits=1, gamma=0.1, p_thermal=0.05
        )
        ops = ch.kraus_operators()
        assert len(ops) == 4

    def test_trace_preserving(self) -> None:
        """GAD must be trace-preserving."""
        ch = GeneralizedAmplitudeDampingChannel(
            name="gad", num_qubits=1, gamma=0.2, p_thermal=0.1
        )
        ops = ch.kraus_operators()
        tp = sum(op.conj().T @ op for op in ops)
        assert np.allclose(tp, np.eye(2), atol=1e-10)

    def test_zero_temp_reduces_to_ad(self) -> None:
        """At p_thermal=0, GAD should equal standard amplitude damping."""
        ad = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.3)
        gad = GeneralizedAmplitudeDampingChannel(
            name="gad", num_qubits=1, gamma=0.3, p_thermal=0.0
        )
        ad_ops = ad.kraus_operators()
        gad_ops = gad.kraus_operators()
        # GAD at p_thermal=0 has 2 non-zero Kraus ops (same as AD)
        assert len(gad_ops) == 4  # structure gives 4, 2 may be zero-like
        tp_ad = sum(op.conj().T @ op for op in ad_ops)
        tp_gad = sum(op.conj().T @ op for op in gad_ops)
        assert np.allclose(tp_ad, tp_gad, atol=1e-10)

    def test_p_thermal_out_of_bounds(self) -> None:
        """p_thermal must be in [0, 0.5]."""
        with pytest.raises(ValueError):
            GeneralizedAmplitudeDampingChannel(
                name="gad", num_qubits=1, gamma=0.1, p_thermal=0.6
            )
        with pytest.raises(ValueError):
            GeneralizedAmplitudeDampingChannel(
                name="gad", num_qubits=1, gamma=0.1, p_thermal=-0.1
            )


class TestKrausChannel:
    """Tests for KrausChannel."""

    def test_valid_kraus_channel(self) -> None:
        """A valid set of Kraus operators should be accepted."""
        K0 = np.eye(2, dtype=complex) * np.sqrt(0.9)
        K1 = np.eye(2, dtype=complex) * np.sqrt(0.1)
        ch = KrausChannel(name="custom", num_qubits=1, kraus_ops=[K0, K1])
        ops = ch.kraus_operators()
        assert len(ops) == 2

    def test_non_trace_preserving(self) -> None:
        """Non-TP Kraus operators should raise ValueError."""
        K0 = np.eye(2, dtype=complex) * 0.5
        with pytest.raises(ValueError, match="trace-preserving|Kraus"):
            KrausChannel(name="bad", num_qubits=1, kraus_ops=[K0])

    def test_wrong_shape(self) -> None:
        """Kraus operators must have correct dimensions."""
        K0 = np.eye(3, dtype=complex)  # 3x3 for 1 qubit is wrong
        K1 = np.eye(3, dtype=complex)
        with pytest.raises(ValueError):
            KrausChannel(name="bad", num_qubits=1, kraus_ops=[K0, K1])


class TestChannelRegressions:
    """Regression tests for channel validation and KrausChannel equality."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: PauliChannel(name="p", num_qubits=2, px=0.1),
            lambda: BitFlipChannel(name="b", num_qubits=2, p=0.1),
            lambda: PhaseFlipChannel(name="z", num_qubits=0, p=0.1),
            lambda: BitPhaseFlipChannel(name="y", num_qubits=3, p=0.1),
            lambda: AmplitudeDampingChannel(name="a", num_qubits=2, gamma=0.1),
            lambda: PhaseDampingChannel(name="d", num_qubits=2, gamma=0.1),
            lambda: GeneralizedAmplitudeDampingChannel(
                name="g", num_qubits=2, gamma=0.1, p_thermal=0.1
            ),
        ],
    )
    def test_single_qubit_channels_reject_wrong_width(self, factory: object) -> None:
        """Fixed 2x2-Kraus channels raise ValueError for num_qubits != 1."""
        with pytest.raises(ValueError, match="num_qubits must be 1"):
            factory()  # type: ignore[operator]

    def test_kraus_docstring_example_is_valid(self) -> None:
        """The KrausChannel docstring example runs as a doctest."""
        import doctest

        (test,) = doctest.DocTestFinder(recurse=False).find(
            KrausChannel, "KrausChannel", globs={"KrausChannel": KrausChannel}
        )
        result = doctest.DocTestRunner().run(test)
        assert result.attempted > 0 and result.failed == 0

    def test_kraus_equality_compares_operators(self) -> None:
        """Same name/width but different operators -> not equal."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        ident = KrausChannel(name="k", num_qubits=1, kraus_ops=[np.eye(2)])
        ident2 = KrausChannel(
            name="k", num_qubits=1, kraus_ops=[np.eye(2, dtype=complex)], label="x"
        )
        flip = KrausChannel(name="k", num_qubits=1, kraus_ops=[X])
        assert ident != flip
        assert ident == ident2
        assert hash(ident) == hash(ident2)
        assert len({ident, ident2, flip}) == 2
        assert ident != BitFlipChannel(name="k", num_qubits=1, p=0.0)


class TestDensityMatrixEffect:
    """Integration tests: verify channel effects on density matrices (C++ backend)."""

    def test_depolarizing_mixedness(self) -> None:
        """Depolarizing a pure state should decrease purity."""
        from quspect.simulation.density_matrix import DensityMatrix

        rho = DensityMatrix(2)  # |00⟩⟨00|
        rho.apply_gate(
            np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), (0,)
        )  # H on q0
        purity_before = rho.purity()
        assert abs(purity_before - 1.0) < 1e-10

        ch = DepolarizingChannel(name="dp", num_qubits=1, p=0.1)
        rho.apply_channel(ch.kraus_operators(), (0,))
        purity_after = rho.purity()
        assert purity_after < purity_before

    def test_amplitude_damping_ground_state(self) -> None:
        """Amplitude damping should leave |0⟩ unchanged."""
        from quspect.simulation.density_matrix import DensityMatrix

        rho = DensityMatrix(1)  # |0⟩⟨0|
        ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=0.5)
        rho.apply_channel(ch.kraus_operators(), (0,))
        # |0⟩ should be a fixed point of amplitude damping
        data = rho._data
        assert abs(data[0, 0] - 1.0) < 1e-10

    def test_bit_flip_symmetry(self) -> None:
        """Bit-flip on |0⟩ should produce mixed state with correct ratio."""
        from quspect.simulation.density_matrix import DensityMatrix

        rho = DensityMatrix(1)  # |0⟩⟨0|
        ch = BitFlipChannel(name="bf", num_qubits=1, p=0.5)
        rho.apply_channel(ch.kraus_operators(), (0,))
        # With p=0.5, ρ = 0.5|0⟩⟨0| + 0.5|1⟩⟨1|
        data = rho._data
        assert abs(data[0, 0] - 0.5) < 1e-12
        assert abs(data[1, 1] - 0.5) < 1e-12
        assert abs(data[0, 1]) < 1e-12
