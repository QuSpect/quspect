"""Cross-validate every channel in ``quspect.noise.channel`` against qiskit.

Channels are compared through their Choi matrices (unique, independent of the
Kraus decomposition) against references built with ``qiskit.quantum_info``
(``Operator`` / ``SuperOp`` / ``Kraus`` / ``Choi``) and against independent
textbook formulas (Nielsen & Chuang Sec. 8.3).  See the module docstring of
:mod:`qiskit_tests._noise_helpers` for the parameter-convention table.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest
from qiskit.quantum_info import Choi, Kraus, random_density_matrix
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix

import quspect
from qiskit_tests._noise_helpers import (
    I2,
    X,
    Y,
    Z,
    assert_choi_equal,
    channel_catalogue,
    choi_of_channel,
    choi_of_kraus,
    choi_of_map,
    choi_of_superop,
    kraus_completeness,
    random_cptp_kraus,
    ref_amplitude_damping_kraus,
    ref_amplitude_damping_map,
    ref_depolarizing_map,
    ref_depolarizing_superop,
    ref_generalized_amplitude_damping_kraus,
    ref_pauli_superop,
    ref_phase_damping_map,
)
from quspect.ir import qsal
from quspect.noise import (
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

pytestmark = pytest.mark.qiskit_ref

CATALOGUE = channel_catalogue()
PROBS = [0.0, 0.13, 0.5, 0.87, 1.0]


# =============================================================================
# Choi matrices vs qiskit references
# =============================================================================


@pytest.mark.parametrize(
    ("px", "py", "pz"),
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.1, 0.2, 0.3),
        (0.25, 0.25, 0.5),
        (0.05, 0.0, 0.4),
    ],
)
def test_pauli_channel_choi(px: float, py: float, pz: float) -> None:
    """PauliChannel == sum_P p_P P rho P (aer pauli_error), boundaries included."""
    ch = PauliChannel(name="pauli", num_qubits=1, px=px, py=py, pz=pz)
    ref = ref_pauli_superop({"I": 1 - px - py - pz, "X": px, "Y": py, "Z": pz})
    assert_choi_equal(choi_of_channel(ch), choi_of_superop(ref))


_FLIP_CASES: list[tuple[type[NoiseChannel], str]] = [
    (BitFlipChannel, "X"),
    (PhaseFlipChannel, "Z"),
    (BitPhaseFlipChannel, "Y"),
]


@pytest.mark.parametrize(("cls", "pauli"), _FLIP_CASES)
@pytest.mark.parametrize("p", PROBS)
def test_flip_channels_choi(cls: type[NoiseChannel], pauli: str, p: float) -> None:
    """Bit/phase/bit-phase flip: (1-p) rho + p P rho P (N&C 8.3.3)."""
    ch = cls(name="flip", num_qubits=1, p=p)  # type: ignore[call-arg]
    ref = ref_pauli_superop({"I": 1 - p, pauli: p})
    assert_choi_equal(choi_of_channel(ch), choi_of_superop(ref))


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("p", [0.0, 0.2, 0.75, 1.0])
def test_depolarizing_choi_matches_aer_definition(n: int, p: float) -> None:
    """Depolarizing(p) == (1-p) rho + p I/2^n == aer depolarizing_error(p, n).

    quspect's ``p`` is the *replacement* probability (N&C eq. 8.100), so each
    non-identity Pauli carries probability ``p / 4^n`` -- not ``p/3``.
    """
    ch = DepolarizingChannel(name="dep", num_qubits=n, p=p)
    mine = choi_of_channel(ch)
    assert_choi_equal(mine, choi_of_superop(ref_depolarizing_superop(p, n)))
    assert_choi_equal(mine, choi_of_map(ref_depolarizing_map(p, 2**n), 2**n))


@pytest.mark.parametrize("n", [1, 2])
def test_depolarizing_pauli_weights(n: int) -> None:
    """Depolarizing(p) applies each non-identity Pauli with probability p/4^n."""
    p = 0.48
    labels = [a + b for a in "IXYZ" for b in "IXYZ"] if n == 2 else list("IXYZ")
    w = p / 4**n
    probs = dict.fromkeys(labels, w)
    probs["I" * n] = 1 - (4**n - 1) * w
    ch = DepolarizingChannel(name="dep", num_qubits=n, p=p)
    assert_choi_equal(choi_of_channel(ch), choi_of_superop(ref_pauli_superop(probs)))


def test_depolarizing_vs_nielsen_chuang_p_over_3_convention() -> None:
    """N&C (8.102) 'p/3 per Pauli' with q corresponds to quspect p = 4q/3."""
    q = 0.3
    ch = DepolarizingChannel(name="dep", num_qubits=1, p=4 * q / 3)
    ref = ref_pauli_superop({"I": 1 - q, "X": q / 3, "Y": q / 3, "Z": q / 3})
    assert_choi_equal(choi_of_channel(ch), choi_of_superop(ref))


@pytest.mark.parametrize("n", [1, 2])
def test_depolarizing_full_strength_is_completely_depolarizing(n: int) -> None:
    """p=1 maps any state to I/2^n."""
    ch = DepolarizingChannel(name="dep", num_qubits=n, p=1.0)
    rho = random_density_matrix(2**n, seed=3)
    out = rho.evolve(Kraus(ch.kraus_operators()))
    np.testing.assert_allclose(out.data, np.eye(2**n) / 2**n, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.0, 0.2, 0.64, 1.0])
def test_amplitude_damping_choi(gamma: float) -> None:
    """AmplitudeDamping(gamma) == N&C (8.107) Kraus == N&C (8.108) map."""
    ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=gamma)
    mine = choi_of_channel(ch)
    assert_choi_equal(mine, choi_of_kraus(ref_amplitude_damping_kraus(gamma)))
    assert_choi_equal(mine, choi_of_map(ref_amplitude_damping_map(gamma), 2))


def test_amplitude_damping_relaxes_excited_state() -> None:
    """|1><1| -> (1-gamma)|1><1| + gamma|0><0|; gamma=1 resets to |0>."""
    for gamma in (0.3, 1.0):
        ch = AmplitudeDampingChannel(name="ad", num_qubits=1, gamma=gamma)
        out = QiskitDensityMatrix.from_label("1").evolve(Kraus(ch.kraus_operators()))
        np.testing.assert_allclose(
            np.real(np.diag(out.data)), [gamma, 1 - gamma], atol=1e-12
        )


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.75, 1.0])
def test_phase_damping_choi(gamma: float) -> None:
    """PhaseDamping(gamma): off-diagonals scale by sqrt(1-gamma) (N&C 8.128)."""
    ch = PhaseDampingChannel(name="pd", num_qubits=1, gamma=gamma)
    mine = choi_of_channel(ch)
    assert_choi_equal(mine, choi_of_map(ref_phase_damping_map(gamma), 2))
    # Equivalent phase flip with p = (1 - sqrt(1 - gamma)) / 2.
    p = (1 - math.sqrt(1 - gamma)) / 2
    assert_choi_equal(mine, choi_of_superop(ref_pauli_superop({"I": 1 - p, "Z": p})))


@pytest.mark.parametrize(
    ("gamma", "p_thermal"),
    [(0.0, 0.0), (0.0, 0.3), (0.36, 0.21), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)],
)
def test_generalized_amplitude_damping_choi(gamma: float, p_thermal: float) -> None:
    """GAD(gamma, p_thermal) == N&C (8.108)-(8.111) with N&C p = 1 - p_thermal."""
    ch = GeneralizedAmplitudeDampingChannel(
        name="gad", num_qubits=1, gamma=gamma, p_thermal=p_thermal
    )
    ref = ref_generalized_amplitude_damping_kraus(gamma, p_thermal)
    assert_choi_equal(choi_of_channel(ch), choi_of_kraus(ref))


def test_generalized_amplitude_damping_limits_and_fixed_point() -> None:
    """p_thermal=0 is plain AD; diag(1-p, p) is the stationary state."""
    gad0 = GeneralizedAmplitudeDampingChannel(name="g", num_qubits=1, gamma=0.4)
    ad = AmplitudeDampingChannel(name="a", num_qubits=1, gamma=0.4)
    assert_choi_equal(choi_of_channel(gad0), choi_of_channel(ad))
    pth = 0.27
    ch = GeneralizedAmplitudeDampingChannel(
        name="g", num_qubits=1, gamma=0.4, p_thermal=pth
    )
    fixed = QiskitDensityMatrix(np.diag([1 - pth, pth]).astype(complex))
    out = fixed.evolve(Kraus(ch.kraus_operators()))
    np.testing.assert_allclose(out.data, fixed.data, atol=1e-12)
    # gamma = 1 relaxes any state straight to the thermal state.
    full = GeneralizedAmplitudeDampingChannel(
        name="g", num_qubits=1, gamma=1.0, p_thermal=pth
    )
    out = random_density_matrix(2, seed=4).evolve(Kraus(full.kraus_operators()))
    np.testing.assert_allclose(out.data, fixed.data, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
def test_kraus_channel_returns_user_operators(n: int) -> None:
    """KrausChannel.kraus_operators() returns copies equal to the input."""
    ops = random_cptp_kraus(n, 3, seed=20 + n)
    ch = KrausChannel(name="k", num_qubits=n, kraus_ops=ops)
    out = ch.kraus_operators()
    assert len(out) == len(ops)
    for a, b in zip(out, ops, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a is not b
    out[0][...] = 0.0  # mutating the copy does not touch the channel
    np.testing.assert_array_equal(ch.kraus_operators()[0], ops[0])
    assert_choi_equal(choi_of_channel(ch), choi_of_kraus(ops))


# =============================================================================
# CPTP and Kraus structure
# =============================================================================


def _edge_channels() -> list[NoiseChannel]:
    """Channels at parameter boundaries (some Kraus ops get dropped)."""
    return [
        PauliChannel(name="p", num_qubits=1),
        PauliChannel(name="p", num_qubits=1, px=1.0),
        BitFlipChannel(name="b", num_qubits=1, p=0.0),
        BitFlipChannel(name="b", num_qubits=1, p=1.0),
        PhaseFlipChannel(name="z", num_qubits=1, p=1.0),
        BitPhaseFlipChannel(name="y", num_qubits=1, p=0.0),
        DepolarizingChannel(name="d", num_qubits=1, p=0.0),
        DepolarizingChannel(name="d", num_qubits=1, p=1.0),
        DepolarizingChannel(name="d", num_qubits=2, p=0.0),
        DepolarizingChannel(name="d", num_qubits=2, p=1.0),
        AmplitudeDampingChannel(name="a", num_qubits=1, gamma=0.0),
        AmplitudeDampingChannel(name="a", num_qubits=1, gamma=1.0),
        PhaseDampingChannel(name="pd", num_qubits=1, gamma=0.0),
        PhaseDampingChannel(name="pd", num_qubits=1, gamma=1.0),
        GeneralizedAmplitudeDampingChannel(name="g", num_qubits=1, gamma=0.0),
        GeneralizedAmplitudeDampingChannel(
            name="g", num_qubits=1, gamma=1.0, p_thermal=0.5
        ),
    ]


@pytest.mark.parametrize(
    "channel",
    list(CATALOGUE.values()) + _edge_channels(),
    ids=lambda c: f"{type(c).__name__}-{c.name}",
)
def test_channel_is_cptp(channel: NoiseChannel) -> None:
    """sum K^dagger K = I, Kraus shapes are 2^n, and qiskit agrees it is CPTP."""
    ops = channel.kraus_operators()
    dim = 2**channel.num_qubits
    assert ops
    assert all(np.asarray(k).shape == (dim, dim) for k in ops)
    np.testing.assert_allclose(kraus_completeness(ops), np.eye(dim), atol=1e-12)
    assert Choi(Kraus(ops)).is_cptp()


@pytest.mark.parametrize(
    ("channel", "num_ops", "unitary"),
    [
        (BitFlipChannel(name="b", num_qubits=1, p=0.0), 1, True),
        (BitFlipChannel(name="b", num_qubits=1, p=1.0), 1, True),
        (BitFlipChannel(name="b", num_qubits=1, p=0.5), 2, False),
        (PauliChannel(name="p", num_qubits=1, px=0.1, py=0.2, pz=0.3), 4, False),
        (PauliChannel(name="p", num_qubits=1, py=1.0), 1, True),
        (DepolarizingChannel(name="d", num_qubits=1, p=0.1), 4, False),
        (DepolarizingChannel(name="d", num_qubits=2, p=0.1), 16, False),
        (DepolarizingChannel(name="d", num_qubits=2, p=0.0), 1, True),
        (AmplitudeDampingChannel(name="a", num_qubits=1, gamma=0.0), 1, True),
        (AmplitudeDampingChannel(name="a", num_qubits=1, gamma=0.5), 2, False),
        (PhaseDampingChannel(name="pd", num_qubits=1, gamma=0.0), 1, True),
        (
            GeneralizedAmplitudeDampingChannel(
                name="g", num_qubits=1, gamma=0.2, p_thermal=0.1
            ),
            4,
            False,
        ),
        (
            KrausChannel(name="u", num_qubits=1, kraus_ops=[X.copy()]),
            1,
            True,
        ),
    ],
)
def test_kraus_count_and_is_unitary(
    channel: NoiseChannel, num_ops: int, unitary: bool
) -> None:
    """Zero-weight Kraus operators are dropped; is_unitary() == (len == 1)."""
    assert len(channel.kraus_operators()) == num_ops
    assert channel.is_unitary() is unitary


def test_pauli_matrices_used_are_standard() -> None:
    """The X/Y/Z Kraus operators are the standard Pauli matrices."""
    for cls, pauli in ((BitFlipChannel, X), (PhaseFlipChannel, Z)):
        (op,) = cls(name="c", num_qubits=1, p=1.0).kraus_operators()
        np.testing.assert_allclose(op, pauli, atol=1e-15)
    (op,) = BitPhaseFlipChannel(name="c", num_qubits=1, p=1.0).kraus_operators()
    np.testing.assert_allclose(op, Y, atol=1e-15)
    (op,) = BitFlipChannel(name="c", num_qubits=1, p=0.0).kraus_operators()
    np.testing.assert_allclose(op, I2, atol=1e-15)


# =============================================================================
# Parameter validation
# =============================================================================

_INVALID: list[tuple[str, Callable[[], NoiseChannel]]] = [
    ("pauli_neg", lambda: PauliChannel(name="p", num_qubits=1, px=-0.1)),
    ("pauli_neg_y", lambda: PauliChannel(name="p", num_qubits=1, py=-1e-3)),
    ("pauli_neg_z", lambda: PauliChannel(name="p", num_qubits=1, pz=-1e-3)),
    ("pauli_sum", lambda: PauliChannel(name="p", num_qubits=1, px=0.5, pz=0.6)),
    ("bit_flip_neg", lambda: BitFlipChannel(name="b", num_qubits=1, p=-0.01)),
    ("bit_flip_big", lambda: BitFlipChannel(name="b", num_qubits=1, p=1.01)),
    ("phase_flip_neg", lambda: PhaseFlipChannel(name="z", num_qubits=1, p=-0.01)),
    ("phase_flip_big", lambda: PhaseFlipChannel(name="z", num_qubits=1, p=1.5)),
    ("bpf_neg", lambda: BitPhaseFlipChannel(name="y", num_qubits=1, p=-0.2)),
    ("bpf_big", lambda: BitPhaseFlipChannel(name="y", num_qubits=1, p=2.0)),
    ("dep_neg", lambda: DepolarizingChannel(name="d", num_qubits=1, p=-0.1)),
    # aer allows up to 4/3 (resp. 16/15); quspect restricts to [0, 1].
    ("dep_big", lambda: DepolarizingChannel(name="d", num_qubits=1, p=1.2)),
    ("dep2_big", lambda: DepolarizingChannel(name="d", num_qubits=2, p=1.05)),
    ("ad_neg", lambda: AmplitudeDampingChannel(name="a", num_qubits=1, gamma=-0.1)),
    ("ad_big", lambda: AmplitudeDampingChannel(name="a", num_qubits=1, gamma=1.1)),
    ("pd_neg", lambda: PhaseDampingChannel(name="a", num_qubits=1, gamma=-0.1)),
    ("pd_big", lambda: PhaseDampingChannel(name="a", num_qubits=1, gamma=1.1)),
    (
        "gad_gamma",
        lambda: GeneralizedAmplitudeDampingChannel(name="g", num_qubits=1, gamma=1.1),
    ),
    (
        "gad_p_big",
        lambda: GeneralizedAmplitudeDampingChannel(
            name="g", num_qubits=1, gamma=0.1, p_thermal=0.6
        ),
    ),
    (
        "gad_p_neg",
        lambda: GeneralizedAmplitudeDampingChannel(
            name="g", num_qubits=1, gamma=0.1, p_thermal=-0.1
        ),
    ),
    (
        "kraus_shape",
        lambda: KrausChannel(name="k", num_qubits=2, kraus_ops=[np.eye(2)]),
    ),
    (
        "kraus_not_tp",
        lambda: KrausChannel(name="k", num_qubits=1, kraus_ops=[0.9 * np.eye(2)]),
    ),
    (
        "kraus_trace_increasing",
        lambda: KrausChannel(name="k", num_qubits=1, kraus_ops=[np.eye(2), X]),
    ),
]


@pytest.mark.parametrize(
    "factory", [f for _, f in _INVALID], ids=[i for i, _ in _INVALID]
)
def test_invalid_parameters_raise(factory: Callable[[], NoiseChannel]) -> None:
    """Out-of-range probabilities / non-CPTP Kraus sets raise ValueError."""
    with pytest.raises(ValueError):
        factory()


def test_depolarizing_more_than_two_qubits_not_implemented() -> None:
    """Depolarizing Kraus operators exist only for 1 and 2 qubits."""
    ch = DepolarizingChannel(name="d", num_qubits=3, p=0.1)
    with pytest.raises(NotImplementedError):
        ch.kraus_operators()


def test_base_channel_is_abstract() -> None:
    """NoiseChannel.kraus_operators / is_unitary raise NotImplementedError."""
    base = NoiseChannel(name="base", num_qubits=1)
    with pytest.raises(NotImplementedError):
        base.kraus_operators()
    with pytest.raises(NotImplementedError):
        base.is_unitary()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BitFlipChannel(name="b", num_qubits=2, p=0.1),
        lambda: AmplitudeDampingChannel(name="a", num_qubits=2, gamma=0.1),
        lambda: PauliChannel(name="p", num_qubits=3, px=0.1),
    ],
)
def test_single_qubit_channels_reject_wrong_width(
    factory: Callable[[], NoiseChannel],
) -> None:
    """A 1-qubit channel constructed with num_qubits=2 should raise ValueError."""
    with pytest.raises(ValueError):
        factory()


def test_kraus_channel_docstring_example_constructs() -> None:
    """The documented example should build a valid channel (run as doctest)."""
    import doctest

    finder = doctest.DocTestFinder(recurse=False)
    (test,) = finder.find(
        KrausChannel, "KrausChannel", globs={"KrausChannel": KrausChannel}
    )
    assert "KrausChannel(" in "".join(ex.source for ex in test.examples)
    result = doctest.DocTestRunner(verbose=False).run(test)
    assert result.attempted > 0
    assert result.failed == 0


# =============================================================================
# Equality, hashing, immutability, repr
# =============================================================================


def test_channel_equality_and_hash() -> None:
    """Frozen dataclasses: equal fields -> equal & same hash; label ignored."""
    a = BitFlipChannel(name="bf", num_qubits=1, p=0.1, label="first")
    b = BitFlipChannel(name="bf", num_qubits=1, p=0.1, label="second")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a != BitFlipChannel(name="bf", num_qubits=1, p=0.2)
    assert a != BitFlipChannel(name="other", num_qubits=1, p=0.1)
    # Different classes with identical-looking fields are not equal.
    assert a != PhaseFlipChannel(name="bf", num_qubits=1, p=0.1)
    g1 = GeneralizedAmplitudeDampingChannel(name="g", num_qubits=1, gamma=0.1)
    g2 = GeneralizedAmplitudeDampingChannel(
        name="g", num_qubits=1, gamma=0.1, p_thermal=0.2
    )
    assert g1 != g2
    assert PauliChannel(name="p", num_qubits=1, px=0.1) != PauliChannel(
        name="p", num_qubits=1, py=0.1
    )


@pytest.mark.parametrize("channel", list(CATALOGUE.values()), ids=list(CATALOGUE))
def test_channels_are_frozen(channel: NoiseChannel) -> None:
    """Channels are immutable; params defaults to an empty tuple."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        channel.name = "changed"  # type: ignore[misc]
    assert channel.params == ()


def test_kraus_channel_equality_considers_operators() -> None:
    """KrausChannel(identity) and KrausChannel(X) with one name should differ."""
    ident = KrausChannel(name="k", num_qubits=1, kraus_ops=[np.eye(2, dtype=complex)])
    flip = KrausChannel(name="k", num_qubits=1, kraus_ops=[X.copy()])
    assert ident != flip


def test_repr() -> None:
    """repr shows class, name, width and parameters; base repr is compact."""
    r = repr(BitFlipChannel(name="bf", num_qubits=1, p=0.1, label="lbl"))
    assert r.startswith("BitFlipChannel(")
    assert "name='bf'" in r and "num_qubits=1" in r and "p=0.1" in r
    assert "lbl" in r
    r = repr(GeneralizedAmplitudeDampingChannel(name="g", num_qubits=1, gamma=0.2))
    assert "gamma=0.2" in r and "p_thermal=0.0" in r
    # KrausChannel hides its matrices.
    assert "kraus_ops" not in repr(CATALOGUE["kraus_1q"])
    # The base-class __repr__ (used by NoiseChannel itself) omits empty fields.
    assert repr(NoiseChannel(name="n", num_qubits=2)) == (
        "NoiseChannel(name='n', num_qubits=2)"
    )
    assert repr(NoiseChannel(name="n", num_qubits=1, params=(0.5,), label="x")) == (
        "NoiseChannel(name='n', num_qubits=1, params=(0.5,), label='x')"
    )


# =============================================================================
# Serialization (QSAL round trip) and circuit fluent API
# =============================================================================


@pytest.mark.parametrize("key", list(CATALOGUE))
def test_qsal_round_trip(key: str) -> None:
    """dumps/loads preserve type, parameters and the channel's Choi matrix."""
    ch = CATALOGUE[key]
    qubits = (2, 0) if ch.num_qubits == 2 else (1,)
    circ = quspect.QuantumCircuit().h(0).id(2).append_channel(ch, qubits)
    doc = qsal.loads(qsal.dumps(circ))
    (instr,) = [i for i in doc.circuit.instructions if i.is_channel]
    back = instr.channel
    assert back is not None
    assert type(back) is type(ch)
    assert back.name == ch.name
    assert back.num_qubits == ch.num_qubits
    assert instr.qubits == qubits
    if not isinstance(ch, KrausChannel):
        assert back == ch
    assert_choi_equal(choi_of_channel(back), choi_of_channel(ch), atol=1e-14)


def test_circuit_fluent_channels() -> None:
    """QuantumCircuit noise helpers build the expected channel objects."""
    circ = quspect.QuantumCircuit()
    circ.bit_flip(0.1, 0).phase_flip(0.2, 1).bit_phase_flip(0.3, 2)
    circ.depolarize(0.4, 0).pauli_error(0.1, 0.2, 0.3, 1)
    circ.amplitude_damp(0.5, 2).phase_damp(0.6, 0)
    circ.generalized_amplitude_damp(0.7, 0.2, 1)
    expected = [
        BitFlipChannel(name="bit_flip", num_qubits=1, p=0.1),
        PhaseFlipChannel(name="phase_flip", num_qubits=1, p=0.2),
        BitPhaseFlipChannel(name="bit_phase_flip", num_qubits=1, p=0.3),
        DepolarizingChannel(name="depolarizing", num_qubits=1, p=0.4),
        PauliChannel(name="pauli", num_qubits=1, px=0.1, py=0.2, pz=0.3),
        AmplitudeDampingChannel(name="amplitude_damping", num_qubits=1, gamma=0.5),
        PhaseDampingChannel(name="phase_damping", num_qubits=1, gamma=0.6),
        GeneralizedAmplitudeDampingChannel(
            name="generalized_amplitude_damping", num_qubits=1, gamma=0.7, p_thermal=0.2
        ),
    ]
    got = [i.channel for i in circ.instructions]
    assert got == expected
    assert [i.qubits for i in circ.instructions] == [
        (0,),
        (1,),
        (2,),
        (0,),
        (1,),
        (2,),
        (0,),
        (1,),
    ]
