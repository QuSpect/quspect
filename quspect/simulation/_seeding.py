"""Seed checks shared by the simulators."""

from __future__ import annotations

import operator

__all__ = ["validate_seed"]


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
