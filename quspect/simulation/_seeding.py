"""Random streams of the successive runs of one simulator object.

A simulator created with ``seed=s`` gives a reproducible *sequence* of runs:
the first run is seeded with ``s`` itself and run ``k`` (``k >= 1``) with a
seed derived from ``(s, k)``.  Two simulators created with the same seed and
used in the same way therefore return the same results, and repeated runs of
one simulator are statistically independent of each other.
"""

from __future__ import annotations

import operator
import secrets

import numpy as np

__all__ = ["RunSeeds", "validate_seed"]


def validate_seed(seed: object, *, limit: int | None = None) -> int | None:
    """Return ``seed`` as a plain ``int`` (or ``None``).

    Args:
        seed: ``None`` or a non-negative integer (numpy integers included).
        limit: Exclusive upper bound, or ``None`` for no bound.

    Raises:
        TypeError: If ``seed`` is not an integer (``bool`` is refused).
        ValueError: If ``seed`` is negative or not below ``limit``.

    Example:
        >>> validate_seed(7), validate_seed(None)
        (7, None)
    """
    if seed is None:
        return None
    if isinstance(seed, bool) or not hasattr(seed, "__index__"):
        raise TypeError(f"seed must be a non-negative integer or None, got {seed!r}.")
    value = operator.index(seed)
    if value < 0:
        raise ValueError(f"seed must be a non-negative integer, got {value}.")
    if limit is not None and value >= limit:
        raise ValueError(
            f"seed must be below 2**{limit.bit_length() - 1}, got {value}."
        )
    return value


class RunSeeds:
    """Seeds of the successive runs of one simulator.

    Args:
        seed: The user's seed; ``None`` draws the root from system entropy.

    Example:
        >>> seeds = RunSeeds(42)
        >>> seeds.next()
        42
        >>> seeds.next() == RunSeeds(42).peek(1)
        True
    """

    def __init__(self, seed: int | None) -> None:
        self._root = secrets.randbits(64) if seed is None else seed
        self._runs = 0

    def peek(self, run: int) -> int:
        """Seed of run number ``run`` (0-based) without advancing."""
        if run == 0:
            return self._root
        sequence = np.random.SeedSequence(self._root, spawn_key=(run,))
        return int(sequence.generate_state(1, np.uint64)[0])

    def next(self) -> int:
        """Return the seed of the next run and advance.

        The first run uses the root seed unchanged, so a single run of a
        freshly created simulator gives the result a per-run seed would give.
        """
        seed = self.peek(self._runs)
        self._runs += 1
        return seed
