"""Qubit operator — weighted sum of Pauli string products.

This module provides :class:`QubitOperator`, the primary representation
of qubit Hamiltonians and observables in QuSpect.
"""

from __future__ import annotations

import cmath
from collections.abc import Iterator
from typing import TypeAlias

from quspect.circuit.parameter import Parameter, ParameterExpression

#: A concrete number, named parameter, or affine symbolic expression.
CoeffValue: TypeAlias = complex | float | int | Parameter | ParameterExpression

#: A single Pauli term: tuple of (qubit_index, pauli_letter) sorted by qubit.
TermKey = tuple[tuple[int, str], ...]


def _normalize_term(term: TermKey) -> TermKey:
    """Sort a Pauli term by qubit index, then by pauli letter X/Y/Z.

    Args:
        term: A Pauli term as ((qubit, pauli), ...).

    Returns:
        The sorted term tuple.
    """
    return tuple(sorted(term, key=lambda x: (x[0], _pauli_order(x[1]))))


def _pauli_order(letter: str) -> int:
    """Return sort order for Pauli letters: I=0, X=1, Y=2, Z=3."""
    return {"I": 0, "X": 1, "Y": 2, "Z": 3}.get(letter.upper(), 99)


def _is_pauli_term(term: TermKey) -> bool:
    """Check that a term is a valid Pauli term.

    Valid terms are tuples of ``(int, str)`` pairs where the string is
    one of ``"I"``, ``"X"``, ``"Y"``, ``"Z"`` (case-insensitive).
    """
    if not isinstance(term, tuple):
        return False
    valid = {"I", "X", "Y", "Z"}
    for item in term:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        q, p = item
        if not isinstance(q, int) or not isinstance(p, str):
            return False
        if p.upper() not in valid:
            return False
    return True


#: Single-qubit Pauli products ``a * b = phase * result`` for non-identity
#: letters (``"I"`` in the result means the identity).
_SINGLE_PAULI_PRODUCT: dict[tuple[str, str], tuple[str, complex]] = {
    ("X", "X"): ("I", 1),
    ("Y", "Y"): ("I", 1),
    ("Z", "Z"): ("I", 1),
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}


def _reduce_term(term: TermKey) -> tuple[TermKey, complex]:
    """Validate a raw Pauli term and reduce repeated qubits.

    Letters are upper-cased and identities dropped.  When a qubit occurs
    more than once, its letters are multiplied left to right using the
    Pauli algebra (e.g. ``X0 Z0 = -i Y0``), and the accumulated phase is
    returned alongside the sorted term.

    Args:
        term: A raw term ``((qubit, letter), ...)``.

    Returns:
        ``(sorted_term, phase)`` with ``phase`` in ``{1, -1, 1j, -1j}``.

    Raises:
        ValueError: If the term is not a tuple of ``(int, str)`` pairs with
            letters in ``I``/``X``/``Y``/``Z`` (case-insensitive).
    """
    if not _is_pauli_term(term):
        raise ValueError(
            f"Invalid Pauli term {term!r}: expected a tuple of (qubit, letter) "
            f"pairs with integer qubits and letters in 'I', 'X', 'Y', 'Z'."
        )
    phase: complex = 1
    letters: dict[int, str] = {}
    for qubit, raw_letter in term:
        letter = raw_letter.upper()
        if letter == "I":
            continue
        previous = letters.get(qubit)
        if previous is None:
            letters[qubit] = letter
            continue
        product, factor = _SINGLE_PAULI_PRODUCT[(previous, letter)]
        phase *= factor
        if product == "I":
            del letters[qubit]
        else:
            letters[qubit] = product
    return tuple(sorted(letters.items())), phase


def _format_term(term: TermKey) -> str:
    """Format a Pauli term as a human-readable string like ``X0 Z1``.

    Args:
        term: A sorted Pauli term.

    Returns:
        A string like ``"X0 Z1"``, or ``"I"`` for the empty term.
    """
    if not term:
        return "I"
    parts: list[str] = []
    for qubit, pauli in term:
        parts.append(f"{pauli.upper()}{qubit}")
    return " ".join(parts)


def _format_coeff(coeff: CoeffValue) -> str:
    """Format a coefficient for display.

    Args:
        coeff: A numeric or symbolic coefficient value.

    Returns:
        A string representation.
    """
    if isinstance(coeff, Parameter):
        return repr(coeff)
    if isinstance(coeff, ParameterExpression):
        text = str(coeff)
        if coeff.constant != 0.0 or len(coeff.terms) != 1:
            return f"({text})"
        return text
    if isinstance(coeff, complex):
        if coeff.imag == 0:
            coeff_float = float(coeff.real)
            if coeff_float == int(coeff_float):
                s = str(int(coeff_float))
                if s == "-1":
                    return "-"
                if s == "1":
                    return ""
                return s
            return str(coeff_float)
        # Complex with imaginary part
        real_part = float(coeff.real)
        imag_part = float(coeff.imag)
        parts: list[str] = []
        if real_part != 0 or imag_part == 0:
            parts.append(str(real_part))
        if imag_part != 0:
            sign = "+" if imag_part > 0 and parts else ""
            parts.append(f"{sign}{imag_part}j")
        return "".join(parts) if parts else "0"
    if isinstance(coeff, float):
        if coeff == int(coeff):
            s = str(int(coeff))
            if s == "-1":
                return "-"
            if s == "1":
                return ""
            return s
        return str(coeff)
    if isinstance(coeff, int):
        if coeff == -1:
            return "-"
        if coeff == 1:
            return ""
        return str(coeff)
    return str(coeff)


class QubitOperator:
    """A weighted sum of Pauli string products.

    Represents a qubit Hamiltonian or observable of the form::

        H = Σ c_j · P_j

    where each ``P_j`` is a tensor product of single-qubit Pauli operators
    (I, X, Y, Z) and ``c_j`` is a coefficient that may be a real number,
    a complex number, a symbolic :class:`~quspect.circuit.parameter.Parameter`,
    or a real affine :class:`~quspect.circuit.parameter.ParameterExpression`.

    The operator is stored as a mapping from Pauli terms to coefficients.
    Terms are represented as tuples of ``(qubit_index, pauli_letter)``::

        ((0, 'X'), (1, 'Z'))  →  X on qubit 0, Z on qubit 1

    The identity ``I`` is never explicitly stored on a qubit — if a qubit
    does not appear in a term, it implicitly carries ``I``.  The empty tuple
    ``()`` represents the pure identity operator.

    Terms are automatically sorted by qubit index (ascending), then by
    Pauli letter (X < Y < Z).  Zero-coefficient and identity-I entries
    on specific qubits are stripped on construction.

    ``QubitOperator`` is immutable — all arithmetic operations return
    new instances.

    Attributes:
        terms: A dict mapping sorted term tuples to coefficient values.

    Example:
        >>> from quspect import QubitOperator
        >>> H = QubitOperator({
        ...     ((0, 'X'), (1, 'Z')): 0.5,
        ...     ((2, 'Y'),): 1.0,
        ... })
        >>> H
        QubitOperator(0.5 X0 Z1 + 1.0 Y2)

        >>> QubitOperator.from_pauli_string('XZ', coeff=0.5)
        QubitOperator(0.5 X0 Z1)
    """

    __slots__ = ("_terms",)

    _terms: dict[TermKey, CoeffValue]

    def __init__(
        self,
        terms: dict[TermKey, CoeffValue] | None = None,
    ) -> None:
        """Initialize a QubitOperator from a term-to-coefficient mapping.

        Args:
            terms: A dict mapping Pauli term tuples to coefficient values.
                Empty terms ``()`` represent the identity.  Terms will be
                normalized (sorted) and zero-coefficient entries removed.
                Letters are case-insensitive and ``I`` entries are dropped.
                A qubit that appears more than once in a term is reduced
                with the Pauli algebra, multiplying left to right (e.g.
                ``((0, "X"), (0, "Z"))`` becomes ``-1j`` times ``Y0``).
                If ``None``, the operator is initialized to zero.

        Raises:
            ValueError: If a term is not a tuple of ``(qubit, letter)``
                pairs with an integer qubit and a letter in
                ``I``/``X``/``Y``/``Z``, or if reducing a repeated qubit
                yields an imaginary phase on a symbolic coefficient.
        """
        processed: dict[TermKey, CoeffValue] = {}
        if terms:
            for raw_term, raw_coeff in terms.items():
                sorted_term, phase = _reduce_term(raw_term)
                coeff = self._normalize_coeff(raw_coeff)
                if phase == -1:
                    coeff = self._multiply_coeffs(coeff, -1)
                elif phase != 1:
                    coeff = self._multiply_coeffs(coeff, phase)
                if self._is_zero_coeff(coeff):
                    continue
                if sorted_term in processed:
                    # Merge coefficients
                    processed[sorted_term] = self._add_coeffs(
                        processed[sorted_term], coeff
                    )
                    # Re-check after merge
                    if self._is_zero_coeff(processed[sorted_term]):
                        del processed[sorted_term]
                else:
                    processed[sorted_term] = coeff
        object.__setattr__(self, "_terms", processed)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_zero_coeff(coeff: CoeffValue) -> bool:
        """Check whether a coefficient is numerically zero."""
        if isinstance(coeff, Parameter):
            return False  # symbolic — never zero
        if isinstance(coeff, ParameterExpression):
            return bool(coeff.is_constant and coeff.constant == 0.0)
        if isinstance(coeff, complex):
            return coeff == 0j
        return coeff == 0

    @staticmethod
    def _normalize_coeff(coeff: CoeffValue) -> CoeffValue:
        """Canonicalize constant and single-parameter affine coefficients."""
        if not isinstance(coeff, ParameterExpression):
            return coeff
        if coeff.is_constant:
            return coeff.constant
        if coeff.constant == 0.0 and len(coeff.terms) == 1:
            parameter, factor = next(iter(coeff.terms.items()))
            if factor == 1.0:
                return parameter
        return coeff

    @staticmethod
    def _as_expression(coeff: CoeffValue) -> ParameterExpression:
        """Convert a real-compatible coefficient to an affine expression."""
        if isinstance(coeff, ParameterExpression):
            return coeff
        if isinstance(coeff, Parameter):
            return coeff.as_expression()
        if isinstance(coeff, complex):
            if coeff.imag != 0.0:
                raise ValueError(
                    "Affine symbolic coefficients cannot contain an imaginary part."
                )
            return ParameterExpression(constant=float(coeff.real))
        return ParameterExpression(constant=coeff)

    @staticmethod
    def _add_coeffs(a: CoeffValue, b: CoeffValue) -> CoeffValue:
        """Add two coefficients.

        Symbolic coefficients are merged as affine expressions.  A symbolic
        coefficient cannot be combined with a genuinely complex value because
        ``ParameterExpression`` intentionally represents real affine values.
        """
        a = QubitOperator._normalize_coeff(a)
        b = QubitOperator._normalize_coeff(b)
        if isinstance(a, (Parameter, ParameterExpression)) or isinstance(
            b, (Parameter, ParameterExpression)
        ):
            expression = QubitOperator._as_expression(a) + QubitOperator._as_expression(
                b
            )
            return QubitOperator._normalize_coeff(expression)
        return a + b

    @staticmethod
    def _multiply_coeffs(a: CoeffValue, b: CoeffValue) -> CoeffValue:
        """Multiply two coefficients."""
        a = QubitOperator._normalize_coeff(a)
        b = QubitOperator._normalize_coeff(b)
        a_symbolic = isinstance(a, (Parameter, ParameterExpression))
        b_symbolic = isinstance(b, (Parameter, ParameterExpression))
        if a_symbolic and b_symbolic:
            raise ValueError(
                "Cannot multiply two symbolic coefficients; the result would "
                "not be affine."
            )
        if a_symbolic:
            expression = QubitOperator._as_expression(a)
            factor = QubitOperator._as_expression(b)
            return QubitOperator._normalize_coeff(expression * factor)
        if b_symbolic:
            factor = QubitOperator._as_expression(a)
            expression = QubitOperator._as_expression(b)
            return QubitOperator._normalize_coeff(factor * expression)
        return a * b

    @staticmethod
    def _eval_coeff(coeff: CoeffValue, binding: dict[Parameter, float]) -> CoeffValue:
        """Evaluate a coefficient by substituting bound parameters."""
        if isinstance(coeff, Parameter):
            if coeff not in binding:
                raise ValueError(
                    f"Parameter {coeff} is not bound. "
                    f"Available bindings: {list(binding.keys())}"
                )
            return binding[coeff]
        if isinstance(coeff, ParameterExpression):
            return coeff.evaluate(binding)
        if isinstance(coeff, complex):
            return coeff
        return coeff

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> QubitOperator:
        """Return the zero operator (empty sum).

        Returns:
            A ``QubitOperator`` with no terms.

        Example:
            >>> QubitOperator.zero()
            QubitOperator(0)
        """
        return cls({})

    @classmethod
    def identity(cls, coefficient: CoeffValue = 1.0) -> QubitOperator:
        """Return the identity operator with the given coefficient.

        The identity corresponds to an empty Pauli term ``()``.

        Args:
            coefficient: Overall coefficient for the identity term.

        Returns:
            A ``QubitOperator`` with a single identity term.

        Example:
            >>> QubitOperator.identity()
            QubitOperator(1.0 I)
            >>> QubitOperator.identity(0.5)
            QubitOperator(0.5 I)
        """
        return cls({(): coefficient})

    @classmethod
    def from_pauli_string(
        cls,
        pauli: str,
        coefficient: CoeffValue = 1.0,
        *,
        qubits: tuple[int, ...] | None = None,
    ) -> QubitOperator:
        """Create from a Pauli string like ``'XZ'`` or ``'IXY'``.

        Args:
            pauli: A string of Pauli letters (``'I'``, ``'X'``, ``'Y'``, ``'Z'``).
            coefficient: Overall coefficient for the term.
            qubits: Qubit indices.  If ``None``, consecutive indices
                ``0..len(pauli)-1`` are used.

        Returns:
            A ``QubitOperator`` with a single term.

        Raises:
            ValueError: If ``pauli`` contains invalid characters or
                ``qubits`` length doesn't match.

        Example:
            >>> QubitOperator.from_pauli_string('XZ', coefficient=0.5)
            QubitOperator(0.5 X0 Z1)
            >>> QubitOperator.from_pauli_string('XY', qubits=(2, 3))
            QubitOperator(1.0 X2 Y3)
        """
        valid = {"I", "X", "Y", "Z"}
        pauli = pauli.upper()
        for ch in pauli:
            if ch not in valid:
                raise ValueError(
                    f"Invalid Pauli character {ch!r} in string {pauli!r}. "
                    f"Allowed: I, X, Y, Z"
                )

        if qubits is None:
            qubits = tuple(range(len(pauli)))
        elif len(qubits) != len(pauli):
            raise ValueError(
                f"qubits length ({len(qubits)}) must match "
                f"pauli string length ({len(pauli)})"
            )

        term_parts: list[tuple[int, str]] = []
        for q, p in zip(qubits, pauli, strict=True):
            if p != "I":
                term_parts.append((q, p))

        term = tuple(sorted(term_parts, key=lambda x: (x[0], _pauli_order(x[1]))))
        return cls({term: coefficient})

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    @property
    def terms(self) -> dict[TermKey, CoeffValue]:
        """Return a copy of the internal term-to-coefficient mapping."""
        return dict(self._terms)

    @property
    def is_zero(self) -> bool:
        """Return ``True`` if this is the zero operator (no terms)."""
        return len(self._terms) == 0

    @property
    def is_identity(self) -> bool:
        """Return ``True`` if this operator is proportional to identity.

        An operator is proportional to identity if it has exactly one
        term that is the empty tuple ``()``, or is the zero operator.
        """
        if self.is_zero:
            return True
        return len(self._terms) == 1 and () in self._terms

    def qubit_count(self) -> int:
        """Return the maximum qubit index + 1, or 0 if empty.

        Example:
            >>> op = QubitOperator({((0, 'X'), (2, 'Z')): 1.0})
            >>> op.qubit_count()
            3
        """
        max_q = -1
        for term in self._terms:
            for q, _ in term:
                if q > max_q:
                    max_q = q
        return max_q + 1

    @property
    def parameters(self) -> set[Parameter]:
        """Return the set of all ``Parameter`` instances used in coefficients.

        Example:
            >>> from quspect import Parameter
            >>> t = Parameter("t")
            >>> op = t * QubitOperator({((0, 'X'),): 1.0})
            >>> op.parameters
            {Parameter('t')}
        """
        params: set[Parameter] = set()
        for coeff in self._terms.values():
            if isinstance(coeff, Parameter):
                params.add(coeff)
            elif isinstance(coeff, ParameterExpression):
                params.update(coeff.parameters)
        return params

    # ------------------------------------------------------------------
    # iteration / length
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[TermKey, CoeffValue]]:
        """Iterate over ``(term, coefficient)`` pairs."""
        return iter(self._terms.items())

    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self._terms)

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: QubitOperator | CoeffValue) -> QubitOperator:
        """Add two operators (or add a scalar to the identity term)."""
        if isinstance(other, (int, float, complex, Parameter, ParameterExpression)):
            if other == 0:
                return self
            return self + QubitOperator.identity(other)
        if not isinstance(other, QubitOperator):
            return NotImplemented
        merged = dict(self._terms)
        for term, coeff in other._terms.items():
            if term in merged:
                merged[term] = self._add_coeffs(merged[term], coeff)
            else:
                merged[term] = coeff
        return QubitOperator(merged)

    def __radd__(  # type: ignore[misc]
        self, other: QubitOperator | CoeffValue
    ) -> QubitOperator:
        """Right-add: ``other + self``."""
        return self.__add__(other)

    def __sub__(self, other: QubitOperator | CoeffValue) -> QubitOperator:
        """Subtract two operators."""
        if isinstance(other, (int, float, complex, Parameter, ParameterExpression)):
            if other == 0:
                return self
            return self - QubitOperator.identity(other)
        if not isinstance(other, QubitOperator):
            return NotImplemented
        neg_terms: dict[TermKey, CoeffValue] = {}
        for term, coeff in other._terms.items():
            neg_terms[term] = self._multiply_coeffs(coeff, -1)
        return self + QubitOperator(neg_terms)

    def __rsub__(  # type: ignore[misc]
        self, other: QubitOperator | CoeffValue
    ) -> QubitOperator:
        """Right-subtract: ``other - self``."""
        if isinstance(other, (int, float, complex, Parameter, ParameterExpression)):
            return QubitOperator.identity(other) - self
        if not isinstance(other, QubitOperator):
            return NotImplemented
        return other.__sub__(self)

    def __mul__(self, scalar: CoeffValue) -> QubitOperator:
        """Multiply each coefficient by a scalar.

        Args:
            scalar: A number, ``Parameter``, or affine expression.

        Returns:
            A new ``QubitOperator`` with coefficients scaled.

        Example:
            >>> op = QubitOperator({((0, 'X'),): 1.0})
            >>> 2.0 * op
            QubitOperator(2.0 X0)
        """
        if not isinstance(
            scalar, (int, float, complex, Parameter, ParameterExpression)
        ):
            return NotImplemented
        if self._is_zero_coeff(scalar):
            return QubitOperator.zero()
        new_terms: dict[TermKey, CoeffValue] = {}
        for term, coeff in self._terms.items():
            new_terms[term] = self._multiply_coeffs(scalar, coeff)
        return QubitOperator(new_terms)

    def __rmul__(self, scalar: CoeffValue) -> QubitOperator:  # type: ignore[misc]
        """Right-multiply: ``scalar * self``."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: CoeffValue) -> QubitOperator:
        """Divide each coefficient by a scalar."""
        if not isinstance(
            scalar, (int, float, complex, Parameter, ParameterExpression)
        ):
            return NotImplemented
        if isinstance(scalar, Parameter) or (
            isinstance(scalar, ParameterExpression) and not scalar.is_constant
        ):
            raise ValueError("Division by a symbolic coefficient is not supported.")
        if isinstance(scalar, ParameterExpression):
            scalar = scalar.constant
        if isinstance(scalar, complex):
            if scalar == 0j:
                raise ZeroDivisionError("division by zero")
            # Use complex division
            new_terms: dict[TermKey, CoeffValue] = {}
            for term, coeff in self._terms.items():
                if isinstance(coeff, complex):
                    new_terms[term] = coeff / scalar
                elif isinstance(coeff, (int, float)):
                    new_terms[term] = complex(coeff) / scalar
                else:
                    # Parameter coefficient
                    new_terms[term] = self._multiply_coeffs(coeff, 1.0 / scalar)
            return QubitOperator(new_terms)
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return self.__mul__(1.0 / scalar)

    def __neg__(self) -> QubitOperator:
        """Negate the operator (multiply by -1)."""
        return self.__mul__(-1)

    def __pos__(self) -> QubitOperator:
        """Unary plus — returns self."""
        return self

    def __iadd__(self, other: QubitOperator | CoeffValue) -> QubitOperator:
        """In-place addition (returns a new instance).

        Because ``QubitOperator`` is immutable, this returns a new
        operator rather than mutating ``self``.
        """
        return self.__add__(other)

    def __isub__(self, other: QubitOperator | CoeffValue) -> QubitOperator:
        """In-place subtraction (returns a new instance)."""
        return self.__sub__(other)

    def __imul__(self, scalar: CoeffValue) -> QubitOperator:
        """In-place multiplication (returns a new instance)."""
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # simplify / compress
    # ------------------------------------------------------------------

    def simplify(self) -> QubitOperator:
        """Merge terms with identical Pauli strings.

        Normally arithmetic already ensures uniqueness per term, but
        this is useful after manual term manipulation or when terms
        involving ``Parameter`` coefficients become mergeable after
        ``bind_parameters()``.

        Returns:
            A new ``QubitOperator`` with merged terms.

        Example:
            >>> a = QubitOperator({((0, 'X'),): 1.0})
            >>> b = QubitOperator({((0, 'X'),): 2.0})
            >>> a + b
            QubitOperator(3.0 X0)
        """
        # Re-constructing through __init__ does the merge
        return QubitOperator(dict(self._terms))

    def compress(self, *, tol: float = 1e-12) -> QubitOperator:
        """Remove terms whose coefficient magnitude is below *tol*.

        This is essential after ``bind_parameters()`` when symbolic
        cancellations may produce terms that are numerically zero (within
        floating-point tolerance).

        Args:
            tol: Magnitude threshold.  Terms with ``abs(coeff) < tol``
                are removed.  Only applies to numeric coefficients —
                symbolic parameter or expression coefficients are never removed.

        Returns:
            A new ``QubitOperator`` with near-zero terms removed.

        Example:
            >>> op = QubitOperator({((0, 'X'),): 1e-15, ((1, 'Z'),): 1.0})
            >>> op.compress()
            QubitOperator(1.0 Z1)
        """
        new_terms: dict[TermKey, CoeffValue] = {}
        for term, coeff in self._terms.items():
            if isinstance(coeff, (Parameter, ParameterExpression)):
                new_terms[term] = coeff
            elif isinstance(coeff, complex):
                if abs(coeff) >= tol:
                    new_terms[term] = coeff
            elif abs(float(coeff)) >= tol:
                new_terms[term] = coeff
        return QubitOperator(new_terms)

    # ------------------------------------------------------------------
    # parameter binding
    # ------------------------------------------------------------------

    def bind_parameters(self, binding: dict[Parameter, float]) -> QubitOperator:
        """Resolve symbolic parameters and expressions to numeric values.

        Args:
            binding: A mapping from ``Parameter`` to concrete float values.

        Returns:
            A new ``QubitOperator`` with all parameters resolved.

        Raises:
            ValueError: If any ``Parameter`` in this operator is not bound.

        Example:
            >>> from quspect import Parameter
            >>> t = Parameter("t")
            >>> op = t * QubitOperator({((0, 'X'),): 1.0})
            >>> op.bind_parameters({t: 0.5})
            QubitOperator(0.5 X0)
        """
        new_terms: dict[TermKey, CoeffValue] = {}
        for term, coeff in self._terms.items():
            evaluated = self._eval_coeff(coeff, binding)
            if not self._is_zero_coeff(evaluated):
                if term in new_terms:
                    try:
                        merged = self._add_coeffs(new_terms[term], evaluated)
                        new_terms[term] = merged
                    except ValueError:
                        new_terms[term] = evaluated
                else:
                    new_terms[term] = evaluated
        return QubitOperator(new_terms)

    # ------------------------------------------------------------------
    # equality / hash
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Check equality with another ``QubitOperator``.

        Two operators are equal if they have the same term-to-coefficient
        mapping.
        """
        if not isinstance(other, QubitOperator):
            return NotImplemented
        if len(self._terms) != len(other._terms):
            return False
        for term, coeff in self._terms.items():
            if term not in other._terms:
                return False
            other_coeff = other._terms[term]
            if isinstance(coeff, (Parameter, ParameterExpression)) or isinstance(
                other_coeff, (Parameter, ParameterExpression)
            ):
                if coeff != other_coeff:
                    return False
            elif isinstance(coeff, complex) or isinstance(other_coeff, complex):
                if not cmath.isclose(coeff, other_coeff, rel_tol=1e-9, abs_tol=1e-12):
                    return False
            else:
                # Both are float/int
                if abs(float(coeff) - float(other_coeff)) > 1e-12:
                    return False
        return True

    def __hash__(self) -> int:
        """Hash based on sorted term-coefficient pairs.

        Numeric coefficients are rounded to a tolerance for hashing.
        """
        items: list[tuple[TermKey, float | str]] = []
        for term, coeff in sorted(self._terms.items()):
            if isinstance(coeff, (Parameter, ParameterExpression)):
                items.append((term, str(hash(coeff))))
            elif isinstance(coeff, complex):
                # Round for stable hashing
                rkey = round(coeff.real, 9) + round(coeff.imag, 9) * 1j
                items.append((term, rkey.real + rkey.imag))
            else:
                items.append((term, round(float(coeff), 9)))
        return hash(tuple(items))

    # ------------------------------------------------------------------
    # representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable representation.

        Example:
            >>> QubitOperator({((0, 'X'),): 1.0, ((1, 'Z'),): -0.5})
            QubitOperator(1.0 X0 + -0.5 Z1)
        """
        if self.is_zero:
            return "QubitOperator(0)"

        parts: list[str] = []
        # Sort terms for deterministic output: identity first, then by qubit
        sorted_terms = sorted(
            self._terms.items(),
            key=lambda kv: () if not kv[0] else kv[0],
        )

        for term, coeff in sorted_terms:
            coeff_str = _format_coeff(coeff)
            term_str = _format_term(term)
            if coeff_str == "":
                parts.append(term_str)
            elif coeff_str == "-":
                parts.append(f"-{term_str}")
            else:
                parts.append(f"{coeff_str} {term_str}")

        return f"QubitOperator({' + '.join(parts)})"

    def __str__(self) -> str:
        """Compact string representation."""
        return self.__repr__()

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks.

        Returns:
            A LaTeX string suitable for IPython/Jupyter display.
        """
        if self.is_zero:
            return r"$0$"

        parts: list[str] = []
        sorted_terms = sorted(
            self._terms.items(),
            key=lambda kv: () if not kv[0] else kv[0],
        )

        for term, coeff in sorted_terms:
            coeff_str = self._latex_coeff(coeff)
            term_str = self._latex_term(term)
            # ``_latex_coeff`` formats real numbers with ``str(float(...))``,
            # so unit coefficients appear as "1.0" / "-1.0".
            if coeff_str == "1.0":
                parts.append(term_str)
            elif coeff_str == "-1.0":
                parts.append(f"-{term_str}")
            elif coeff_str.startswith("-"):
                parts.append(f"-{coeff_str[1:]}{term_str}")
            else:
                parts.append(f"{coeff_str}{term_str}")

        return f"${' + '.join(parts)}$".replace(" + -", " - ")

    @staticmethod
    def _latex_coeff(coeff: CoeffValue) -> str:
        """Format coefficient for LaTeX."""
        if isinstance(coeff, Parameter):
            return rf"\mathrm{{{coeff.name}}}"
        if isinstance(coeff, ParameterExpression):
            return rf"\mathrm{{{str(coeff)}}}"
        if isinstance(coeff, complex):
            if coeff.imag == 0:
                return str(float(coeff.real))
            return f"({coeff.real}+{coeff.imag}i)"
        return str(float(coeff))

    @staticmethod
    def _latex_term(term: TermKey) -> str:
        """Format a Pauli term for LaTeX."""
        if not term:
            return r"\mathbb{I}"
        parts: list[str] = []
        for qubit, pauli in term:
            parts.append(rf"\sigma_{{{qubit}}}^{{{pauli.upper()}}}")
        return r" \otimes ".join(parts) if len(parts) > 1 else parts[0]
