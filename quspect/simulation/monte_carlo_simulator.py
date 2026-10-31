"""Monte Carlo trajectory simulator backed by the native C++ runtime."""

from __future__ import annotations

from quspect.simulation.simulator import Simulator


class MonteCarloSimulator(Simulator):
    """Compatibility entry point for native noisy-trajectory simulation.

    Noise channels are decoded into the native circuit IR and executed as one
    complete quantum trajectory per shot.  A trajectory owns one advancing
    random-number stream, so consecutive channels are sampled independently;
    measurements use the circuit's explicit qubit-to-classical-bit mapping.

    ``Simulator`` now uses this same trajectory path automatically whenever a
    circuit contains noise.  This subclass is kept for users who prefer the
    explicit Monte Carlo name.
    """

    @property
    def name(self) -> str:
        return "monte_carlo_simulator"
