"""Pauli-sum input normalization for :mod:`quspect.hamsim`."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeAlias

#: One Hamiltonian term ``coeff · P``: Pauli letters, the qubits they act on
#: (letter ``i`` on ``qubits[i]``, as in ``SparsePauliOp.from_sparse_list``)
#: and a real coefficient.
PauliTerm: TypeAlias = tuple[str, tuple[int, ...], float]

_IMAG_TOL = 1e-12


def _real(coeff: Any, label: str) -> float:
    if isinstance(coeff, numbers.Real):
        return float(coeff)
    if isinstance(coeff, numbers.Complex):
        c = complex(coeff)
        if abs(c.imag) > _IMAG_TOL * max(1.0, abs(c)):
            raise ValueError(
                f"term {label} has a complex coefficient {c}: H must be Hermitian"
            )
        return float(c.real)
    raise TypeError(
        f"term {label}: coefficient {coeff!r} is not a number (bind parameters first)"
    )


def _clean(
    letters: str, qubits: Sequence[int], label: str
) -> tuple[str, tuple[int, ...]]:
    out_l: list[str] = []
    out_q: list[int] = []
    seen: set[int] = set()
    for p, q in zip(letters, qubits, strict=True):
        p = p.upper()
        if p not in "IXYZ":
            raise ValueError(f"term {label}: unknown Pauli letter {p!r}")
        q = int(q)
        if q < 0:
            raise ValueError(f"term {label}: negative qubit index {q}")
        if q in seen:
            raise ValueError(f"term {label}: qubit {q} appears twice")
        seen.add(q)
        if p != "I":
            out_l.append(p)
            out_q.append(q)
    return "".join(out_l), tuple(out_q)


def pauli_terms(
    hamiltonian: Any, num_qubits: int | None = None
) -> tuple[int, list[PauliTerm]]:
    """Normalize a Pauli sum to ``(num_qubits, [(letters, qubits, coeff), ...])``.

    Accepted inputs (term order is kept; it is the order of the product
    formula):

    - a sequence of ``(letters, qubits, coeff)`` triples (the
      ``SparsePauliOp.from_sparse_list`` convention);
    - a qiskit ``SparsePauliOp`` (via ``to_sparse_list``);
    - any operator with a ``terms`` mapping ``{((qubit, letter), ...): coeff}``
      such as :class:`quspect.QubitOperator` or OpenFermion's
      ``QubitOperator``.

    Identity terms only contribute a global phase and are dropped, as are
    zero coefficients.

    Args:
        hamiltonian: The Pauli sum.
        num_qubits: Number of qubits; default: one more than the largest
            index used (or the operator's own ``num_qubits``).

    Returns:
        ``(num_qubits, terms)``.

    Raises:
        ValueError: Non-Hermitian (complex) coefficients, repeated qubits in
            a term, unknown letters, or an index ``≥ num_qubits``.
        TypeError: Unsupported input.
    """
    raw: list[tuple[str, Sequence[int], Any]] = []
    declared: int | None = None
    if hasattr(hamiltonian, "to_sparse_list") and hasattr(hamiltonian, "num_qubits"):
        declared = int(hamiltonian.num_qubits)
        raw = [(str(p), list(q), c) for p, q, c in hamiltonian.to_sparse_list()]
    elif hasattr(hamiltonian, "terms") and isinstance(hamiltonian.terms, Mapping):
        for key, coeff in hamiltonian.terms.items():
            raw.append(
                ("".join(str(p) for _, p in key), [int(q) for q, _ in key], coeff)
            )
    elif isinstance(hamiltonian, Iterable) and not isinstance(hamiltonian, str | bytes):
        for item in hamiltonian:
            if not isinstance(item, Sequence) or len(item) != 3:
                raise TypeError(
                    f"expected (letters, qubits, coeff) triples, got {item!r}"
                )
            p, q, c = item
            raw.append((str(p), list(q), c))
    else:
        raise TypeError(f"unsupported Hamiltonian type {type(hamiltonian).__name__}")

    terms: list[PauliTerm] = []
    for i, (p, q, c) in enumerate(raw):
        label = f"#{i} ({p} on {list(q)})"
        letters, qubits = _clean(p, q, label)
        coeff = _real(c, label)
        if not letters or coeff == 0.0:
            continue
        terms.append((letters, qubits, coeff))
    used = 1 + max((q for _, qs, _ in terms for q in qs), default=-1)
    n = (
        num_qubits
        if num_qubits is not None
        else (declared if declared is not None else used)
    )
    if used > n:
        raise ValueError(f"term acts on qubit {used - 1} but num_qubits = {n}")
    return n, terms
