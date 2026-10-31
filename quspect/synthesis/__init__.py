"""QuSpect synthesis — exact unitary-to-circuit synthesis routines.

Currently provides two-qubit synthesis via the KAK (Cartan) decomposition,
emitting the minimal number of entangling gates (0–3) plus single-qubit
gates: by default CZ with ``{h, rz, x}``, or any
:class:`~quspect.synthesis.two_qubit.TwoQubitBasis`
(e.g. CX with the IBM ``{rz, sx, x}`` gates).
"""

from quspect.synthesis.two_qubit import (
    Entangler,
    TwoQubitBasis,
    TwoQubitKAK,
    entangler,
    kak_decompose,
    num_cz_required,
    register_entangler,
    synthesize_two_qubit,
    weyl_coordinates,
)

__all__ = [
    "Entangler",
    "TwoQubitBasis",
    "TwoQubitKAK",
    "entangler",
    "kak_decompose",
    "num_cz_required",
    "register_entangler",
    "synthesize_two_qubit",
    "weyl_coordinates",
]
