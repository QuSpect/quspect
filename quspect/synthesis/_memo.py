"""Bounded exact-input memoization for the pure synthesis routines.

Compilation resynthesizes the same small unitaries over and over (every
level-3 candidate repeats most blocks of the others, a QFT has one block
class per angle).  The synthesis functions are pure and deterministic, so
caching them on the *exact bytes* of their input matrix (plus every other
argument) returns bit-identical results while skipping the linear algebra.

Keys use the raw ``complex128`` bytes, so two inputs share an entry only if
they are identical to the last bit (``0.0`` and ``-0.0`` stay distinct).
Each cache holds at most ``maxsize`` entries and is simply cleared when it
is full, which keeps memory bounded without per-hit bookkeeping.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

from quspect._ablation import active_components

K = TypeVar("K")
V = TypeVar("V")

DEFAULT_MAXSIZE = 1 << 14


class BoundedCache(Generic[K, V]):
    """A dict-backed cache that is cleared once it reaches ``maxsize``.

    Args:
        maxsize: Maximum number of entries (at least 1).

    Example:
        >>> cache: BoundedCache[str, int] = BoundedCache(2)
        >>> cache.put("a", 1)
        >>> cache.get("a"), cache.get("b")
        (1, None)
    """

    __slots__ = ("_maxsize", "_stores")

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        # 每个消融状态（QUSPECT_ABLATE）一份独立的表：消融配置算出的结果不会被
        # 默认流水线读到，反之亦然。默认状态下与单个 dict 完全相同。
        self._stores: dict[frozenset[str], dict[K, V]] = {}
        self._maxsize = max(1, maxsize)

    @property
    def _data(self) -> dict[K, V]:
        state = active_components()
        store = self._stores.get(state)
        if store is None:
            store = self._stores[state] = {}
        return store

    def get(self, key: K) -> V | None:
        """Return the cached value or ``None``."""
        return self._data.get(key)

    def put(self, key: K, value: V) -> None:
        """Store ``value``; clears the cache first when it is full."""
        data = self._data
        if len(data) >= self._maxsize:
            data.clear()
        data[key] = value

    def clear(self) -> None:
        """Drop every entry."""
        self._stores.clear()

    def __len__(self) -> int:
        return len(self._data)


def matrix_key(matrix: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    """Exact hashable key of a complex matrix (shape and ``complex128`` bytes).

    Args:
        matrix: Any array-like matrix.

    Returns:
        ``(shape, bytes)`` of the matrix converted to C-ordered ``complex128``.

    Example:
        >>> matrix_key(np.eye(2)) == matrix_key(np.eye(2, dtype=complex))
        True
    """
    arr = np.ascontiguousarray(matrix, dtype=complex)
    return arr.shape, arr.tobytes()
