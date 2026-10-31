"""Deterministic codecs for hardware-level Ising matrix artifacts."""

from __future__ import annotations

import csv
import io

from quspect.optimization.hardware import HardwareIsingProgram


def encode_ising_matrix_csv(program: HardwareIsingProgram) -> str:
    """Encode a hardware program as a headerless dense-matrix CSV artifact.

    The transport contains only the dense matrix.  Variable order, the gauge
    index, scaling, and objective offset remain local in
    :class:`~quspect.optimization.HardwareIsingProgram` for result decoding.
    Values use Python's shortest round-trip floating-point representation and
    every row ends with ``"\\n"``.

    Args:
        program: Validated dense hardware Ising program.

    Returns:
        Headerless, index-free CSV text.

    Raises:
        TypeError: If ``program`` is not a ``HardwareIsingProgram``.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import (
        ...     HardwareIsingProgram,
        ...     encode_ising_matrix_csv,
        ... )
        >>> program = HardwareIsingProgram(
        ...     np.array([[0.0, 1.0], [1.0, 0.0]]),
        ...     ("s0", "s1"),
        ...     logical_variable_count=2,
        ... )
        >>> print(encode_ising_matrix_csv(program), end="")
        0.0,1.0
        1.0,0.0
    """
    if not isinstance(program, HardwareIsingProgram):
        raise TypeError(
            "encode_ising_matrix_csv expects HardwareIsingProgram, "
            f"got {type(program).__name__}."
        )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for row in program.matrix:
        writer.writerow(repr(float(value)) for value in row)
    return stream.getvalue()
