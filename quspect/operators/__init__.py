"""Operator system for QuSpect — qubit and fermionic observables.

This package provides algebraic representations of quantum observables:

- :class:`QubitOperator` — weighted sum of Pauli string products
- :class:`FermionOperator` — weighted sum of fermionic ladder operator products
- :mod:`transforms` — fermion-to-qubit mappings (Jordan-Wigner, etc.)
"""

from quspect.operators.fermion_operator import FermionOperator
from quspect.operators.qubit_operator import QubitOperator

__all__ = [
    "QubitOperator",
    "FermionOperator",
]
