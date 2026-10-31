"""Quantum noise system for QuSpect.

This package provides quantum noise channels (CPTP maps),
noise models for declarative noise specification, and the
noise model adder for circuit transformation.

Exports:
    Channel types:
        - PauliChannel
        - BitFlipChannel, PhaseFlipChannel, BitPhaseFlipChannel
        - DepolarizingChannel
        - AmplitudeDampingChannel, PhaseDampingChannel
        - GeneralizedAmplitudeDampingChannel
        - KrausChannel

    Noise model:
        - NoiseRule
        - NoiseModel
        - NoiseModelAdder

    Convenience constructors:
        - ideal
        - depolarizing_model
        - thermal_relaxation_model
"""

from quspect.noise.adder import NoiseModelAdder
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
from quspect.noise.noise_model import (
    NoiseModel,
    NoiseRule,
    depolarizing_model,
    ideal,
    thermal_relaxation_model,
)

__all__ = [
    # Channel types
    "NoiseChannel",
    "PauliChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "BitPhaseFlipChannel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
    "GeneralizedAmplitudeDampingChannel",
    "KrausChannel",
    # Noise model
    "NoiseRule",
    "NoiseModel",
    "NoiseModelAdder",
    # Convenience constructors
    "ideal",
    "depolarizing_model",
    "thermal_relaxation_model",
]
