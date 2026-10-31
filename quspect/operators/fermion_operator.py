"""Fermion operator — weighted sum of fermionic ladder operator products.

This module provides :class:`FermionOperator`, which represents
second-quantized fermionic Hamiltonians and observables.
"""

from __future__ import annotations

from collections.abc import Iterator

from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.operators.qubit_operator import (
    CoeffValue,
    QubitOperator,
    _format_coeff,
)


def _format_fermion_term(term: tuple[tuple[int, int], ...]) -> str:
    """Format a fermionic term as a human-readable string.

    Args:
        term: A tuple of ``(mode, action)`` pairs where ``1`` = creation,
            ``0`` = annihilation.

    Returns:
        A string like ``"a†₀ a₁"``, or ``"I"`` for the empty term.
    """
    if not term:
        return "I"
    parts: list[str] = []
    for mode, action in term:
        if action == 1:
            parts.append(f"a†{mode}")
        elif action == 0:
            parts.append(f"a{mode}")
        else:
            parts.append(f"a^{{{action}}}_{{{mode}}}")
    return " ".join(parts)


class FermionOperator:
    """A weighted sum of fermionic ladder operator products.

    Represents a second-quantized fermionic Hamiltonian or observable::

        H = Σ c_j · a†_{p1} a_{q1} a†_{p2} a_{q2} ...

    where each term is a product of creation (:math:`a^{\\dagger}`) and
    annihilation (:math:`a`) operators on specific fermionic modes.

    **Convention**::

        (i, 1) = creation operator a†_i
        (i, 0) = annihilation operator a_i

    ``FermionOperator`` is immutable — all arithmetic operations return
    new instances.

    Attributes:
        terms: A dict mapping term tuples to coefficient values.

    Example:
        >>> from quspect.operators import FermionOperator
        >>> # Number operator: a†₀ a₀
        >>> op = FermionOperator({((0, 1), (0, 0)): 1.0})
        >>> op
        FermionOperator(1.0 a†0 a0)

        >>> # Hopping term
        >>> t = 1.0
        >>> hop = FermionOperator.hopping_term(0, 1, coefficient=t)
        >>> hop
        FermionOperator(1.0 a†0 a1 + 1.0 a†1 a0)
    """

    __slots__ = ("_terms",)

    _terms: dict[tuple[tuple[int, int], ...], CoeffValue]

    def __init__(
        self,
        terms: dict[tuple[tuple[int, int], ...], CoeffValue] | None = None,
    ) -> None:
        """Initialize a FermionOperator from a term-to-coefficient mapping.

        Args:
            terms: A dict mapping fermionic term tuples to coefficient values.
                Empty terms ``()`` represent the identity.  Zero-coefficient
                entries are removed.  If ``None``, the operator is zero.
        """
        processed: dict[tuple[tuple[int, int], ...], CoeffValue] = {}
        if terms:
            for raw_term, coeff in terms.items():
                coeff = QubitOperator._normalize_coeff(coeff)
                if not isinstance(raw_term, tuple):
                    raise TypeError(
                        f"Term must be a tuple, got {type(raw_term).__name__}"
                    )
                # Validate each element
                for item in raw_term:
                    if not isinstance(item, tuple) or len(item) != 2:
                        raise ValueError(
                            f"Each element of a fermionic term must be "
                            f"(mode, action), got {item!r}"
                        )
                    mode, action = item
                    if not isinstance(mode, int) or mode < 0:
                        raise ValueError(
                            f"Mode index must be non-negative int, got {mode!r}"
                        )
                    if action not in (0, 1):
                        raise ValueError(
                            f"Action must be 0 (annihilate) or 1 (create),"
                            f" got {action!r}"
                        )

                if self._is_zero_coeff(coeff):
                    continue

                if raw_term in processed:
                    processed[raw_term] = self._add_coeffs_numeric(
                        processed[raw_term], coeff
                    )
                    if self._is_zero_coeff(processed[raw_term]):
                        del processed[raw_term]
                else:
                    processed[raw_term] = coeff
        object.__setattr__(self, "_terms", processed)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_zero_coeff(coeff: CoeffValue) -> bool:
        """Check whether a coefficient is numerically zero."""
        return QubitOperator._is_zero_coeff(coeff)

    @staticmethod
    def _add_coeffs_numeric(a: CoeffValue, b: CoeffValue) -> CoeffValue:
        """Add numeric or affine symbolic coefficients."""
        return QubitOperator._add_coeffs(a, b)

    @staticmethod
    def _multiply_coeffs(a: CoeffValue, b: CoeffValue) -> CoeffValue:
        """Multiply two coefficients."""
        return QubitOperator._multiply_coeffs(a, b)

    @staticmethod
    def _eval_coeff(coeff: CoeffValue, binding: dict[Parameter, float]) -> CoeffValue:
        """Evaluate a coefficient by substituting bound parameters."""
        if isinstance(coeff, Parameter):
            if coeff not in binding:
                raise ValueError(
                    f"Parameter {coeff} is not bound. Available: {list(binding.keys())}"
                )
            return binding[coeff]
        if isinstance(coeff, ParameterExpression):
            return coeff.evaluate(binding)
        return coeff

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> FermionOperator:
        """Return the zero operator (empty sum).

        Returns:
            A ``FermionOperator`` with no terms.

        Example:
            >>> FermionOperator.zero()
            FermionOperator(0)
        """
        return cls({})

    @classmethod
    def identity(cls, coefficient: CoeffValue = 1.0) -> FermionOperator:
        """Return the identity operator with the given coefficient.

        The identity corresponds to an empty term ``()``.

        Args:
            coefficient: Coefficient for the identity term.

        Returns:
            A ``FermionOperator`` with a single identity term.

        Example:
            >>> FermionOperator.identity()
            FermionOperator(1.0 I)
        """
        return cls({(): coefficient})

    @classmethod
    def number_operator(
        cls, mode: int, coefficient: CoeffValue = 1.0
    ) -> FermionOperator:
        """Create a number operator :math:`a^{\\dagger}_i a_i` for a given mode.

        Args:
            mode: The fermionic mode index (non-negative int).
            coefficient: Overall coefficient.

        Returns:
            ``coeff * a†_mode a_mode``

        Raises:
            ValueError: If ``mode`` is negative.

        Example:
            >>> FermionOperator.number_operator(0)
            FermionOperator(1.0 a†0 a0)
        """
        if mode < 0:
            raise ValueError(f"Mode index must be non-negative, got {mode}")
        return cls({((mode, 1), (mode, 0)): coefficient})

    @classmethod
    def hopping_term(
        cls,
        i: int,
        j: int,
        coefficient: CoeffValue = 1.0,
    ) -> FermionOperator:
        """Create a Hermitian hopping term a†_i a_j + a†_j a_i.

        Args:
            i: First fermionic mode.
            j: Second fermionic mode.
            coefficient: Overall hopping amplitude.

        For ``i == j`` both summands coincide, so the result is
        ``2 * coeff * a†_i a_i``.

        Returns:
            ``coeff * (a†_i a_j + a†_j a_i)``

        Raises:
            ValueError: If either mode index is negative.

        Example:
            >>> FermionOperator.hopping_term(0, 1, coefficient=0.5)
            FermionOperator(0.5 a†0 a1 + 0.5 a†1 a0)
        """
        if i < 0 or j < 0:
            raise ValueError(f"Mode indices must be non-negative, got i={i}, j={j}")
        forward = cls({((i, 1), (j, 0)): coefficient})
        if i == j:
            # Both summands are the same term a†_i a_i; adding (rather than
            # building a dict literal with a duplicate key) yields 2c n_i.
            return forward + forward
        return forward + cls({((j, 1), (i, 0)): coefficient})

    @classmethod
    def coulomb_term(
        cls,
        i: int,
        j: int,
        k: int,
        ell: int,
        coefficient: CoeffValue = 1.0,
    ) -> FermionOperator:
        """Create a Coulomb interaction term a†_i a†_j a_k a_ell.

        Args:
            i: First creation mode index.
            j: Second creation mode index.
            k: First annihilation mode index.
            ell: Second annihilation mode index.
            coefficient: Overall coupling strength.

        Returns:
            ``coeff * a†_i a†_j a_k a_ell``

        Raises:
            ValueError: If any mode index is negative.

        Example:
            >>> FermionOperator.coulomb_term(0, 1, 1, 0)
            FermionOperator(1.0 a†0 a†1 a1 a0)
        """
        for idx, name in [(i, "i"), (j, "j"), (k, "k"), (ell, "ell")]:
            if idx < 0:
                raise ValueError(f"Mode index {name}={idx} must be non-negative")
        return cls(
            {
                ((i, 1), (j, 1), (k, 0), (ell, 0)): coefficient,
            }
        )

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    @property
    def terms(self) -> dict[tuple[tuple[int, int], ...], CoeffValue]:
        """Return a copy of the internal term-to-coefficient mapping."""
        return dict(self._terms)

    @property
    def is_zero(self) -> bool:
        """Return ``True`` if this is the zero operator."""
        return len(self._terms) == 0

    @property
    def is_identity(self) -> bool:
        """Return ``True`` if this operator is proportional to identity."""
        if self.is_zero:
            return True
        return len(self._terms) == 1 and () in self._terms

    def mode_count(self) -> int:
        """Return the maximum mode index + 1, or 0 if empty.

        Example:
            >>> op = FermionOperator({((0, 1), (2, 0)): 1.0})
            >>> op.mode_count()
            3
        """
        max_m = -1
        for term in self._terms:
            for mode, _ in term:
                if mode > max_m:
                    max_m = mode
        return max_m + 1

    @property
    def parameters(self) -> set[Parameter]:
        """Return the set of all ``Parameter`` instances used in coefficients.

        Example:
            >>> from quspect import Parameter
            >>> t = Parameter("t")
            >>> op = t * FermionOperator.number_operator(0)
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

    def __iter__(self) -> Iterator[tuple[tuple[tuple[int, int], ...], CoeffValue]]:
        """Iterate over ``(term, coefficient)`` pairs."""
        return iter(self._terms.items())

    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self._terms)

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: FermionOperator | CoeffValue) -> FermionOperator:
        """Add two operators (or add a scalar to the identity term)."""
        if isinstance(other, (int, float, complex, Parameter, ParameterExpression)):
            if other == 0:
                return self
            return self + FermionOperator.identity(other)
        if not isinstance(other, FermionOperator):
            return NotImplemented
        merged = dict(self._terms)
        for term, coeff in other._terms.items():
            if term in merged:
                merged[term] = self._add_coeffs_numeric(merged[term], coeff)
            else:
                merged[term] = coeff
        return FermionOperator(merged)

    def __radd__(  # type: ignore[misc]
        self, other: FermionOperator | CoeffValue
    ) -> FermionOperator:
        """Right-add."""
        return self.__add__(other)

    def __sub__(self, other: FermionOperator | CoeffValue) -> FermionOperator:
        """Subtract operators."""
        if isinstance(other, (int, float, complex, Parameter, ParameterExpression)):
            if other == 0:
                return self
            return self - FermionOperator.identity(other)
        if not isinstance(other, FermionOperator):
            return NotImplemented
        neg_terms: dict[tuple[tuple[int, int], ...], CoeffValue] = {}
        for term, coeff in other._terms.items():
            neg_terms[term] = self._multiply_coeffs(coeff, -1)
        return self + FermionOperator(neg_terms)

    def __rsub__(  # type: ignore[misc]
        self, other: FermionOperator | CoeffValue
    ) -> FermionOperator:
        """Right-subtract."""
        if isinstance(other, (int, float, complex, Parameter, ParameterExpression)):
            return FermionOperator.identity(other) - self
        if not isinstance(other, FermionOperator):
            return NotImplemented
        return other.__sub__(self)

    def __mul__(self, scalar: CoeffValue) -> FermionOperator:
        """Multiply each coefficient by a scalar."""
        if not isinstance(
            scalar, (int, float, complex, Parameter, ParameterExpression)
        ):
            return NotImplemented
        if isinstance(scalar, (int, float)) and scalar == 0:
            return FermionOperator.zero()
        if isinstance(scalar, complex) and scalar == 0j:
            return FermionOperator.zero()
        new_terms: dict[tuple[tuple[int, int], ...], CoeffValue] = {}
        for term, coeff in self._terms.items():
            new_terms[term] = self._multiply_coeffs(scalar, coeff)
        return FermionOperator(new_terms)

    def __rmul__(self, scalar: CoeffValue) -> FermionOperator:  # type: ignore[misc]
        """Right-multiply."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: CoeffValue) -> FermionOperator:
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
            new_terms: dict[tuple[tuple[int, int], ...], CoeffValue] = {}
            for term, coeff in self._terms.items():
                if isinstance(coeff, complex):
                    new_terms[term] = coeff / scalar
                elif isinstance(coeff, (int, float)):
                    new_terms[term] = complex(coeff) / scalar
                else:
                    new_terms[term] = self._multiply_coeffs(coeff, 1.0 / scalar)
            return FermionOperator(new_terms)
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return self.__mul__(1.0 / scalar)

    def __neg__(self) -> FermionOperator:
        """Negate the operator."""
        return self.__mul__(-1)

    def __pos__(self) -> FermionOperator:
        """Unary plus — returns self."""
        return self

    def __iadd__(self, other: FermionOperator | CoeffValue) -> FermionOperator:
        """In-place addition (returns new instance)."""
        return self.__add__(other)

    def __isub__(self, other: FermionOperator | CoeffValue) -> FermionOperator:
        """In-place subtraction (returns new instance)."""
        return self.__sub__(other)

    def __imul__(self, scalar: CoeffValue) -> FermionOperator:
        """In-place multiplication (returns new instance)."""
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # simplify / compress / normal_order
    # ------------------------------------------------------------------

    def simplify(self) -> FermionOperator:
        """Merge terms with identical fermionic operator products.

        Normally arithmetic already ensures unique terms, but this is
        useful after ``bind_parameters()`` when symbolic terms become
        numerically mergeable.

        Returns:
            A new ``FermionOperator`` with merged terms.
        """
        return FermionOperator(dict(self._terms))

    def compress(self, *, tol: float = 1e-12) -> FermionOperator:
        """Remove terms whose coefficient magnitude is below *tol*.

        Args:
            tol: Magnitude threshold.

        Returns:
            A new ``FermionOperator`` with near-zero terms removed.
        """
        new_terms: dict[tuple[tuple[int, int], ...], CoeffValue] = {}
        for term, coeff in self._terms.items():
            if isinstance(coeff, (Parameter, ParameterExpression)):
                new_terms[term] = coeff
            elif isinstance(coeff, complex):
                if abs(coeff) >= tol:
                    new_terms[term] = coeff
            elif abs(float(coeff)) >= tol:
                new_terms[term] = coeff
        return FermionOperator(new_terms)

    def normal_order(self) -> FermionOperator:
        """Return the normal-ordered operator.

        Normal ordering moves all creation operators to the left of
        all annihilation operators, respecting fermionic anti-commutation
        relations::

            {a_i, a†_j} = δ_{ij}
            {a_i, a_j} = 0
            {a†_i, a†_j} = 0

        Returns:
            A new ``FermionOperator`` in normal order.

        Example:
            >>> # a₀ a†₁ = -a†₁ a₀ (when 0!=1) + δ_{01} (=0)
            >>> op = FermionOperator({((0, 0), (1, 1)): 1.0})
            >>> op.normal_order()
            FermionOperator(-1.0 a†1 a0)
        """
        result = FermionOperator.zero()
        for term, coeff in self._terms.items():
            n_ordered_terms = self._normal_order_single_term(term, coeff)
            result = result + n_ordered_terms
        return result

    @staticmethod
    def _normal_order_single_term(
        term: tuple[tuple[int, int], ...],
        coeff: CoeffValue,
    ) -> FermionOperator:
        """Normal-order a single fermionic term using bubble-sort-like swaps.

        Walk the operator list; whenever an annihilation operator (action=0)
        appears before a creation operator (action=1), swap them and apply
        the anti-commutation sign flip and delta term.

        Returns:
            A ``FermionOperator`` representing the normal-ordered form.
        """
        ops = list(term)
        current_coeff: CoeffValue = coeff
        delta_terms: list[tuple[list[tuple[int, int]], CoeffValue]] = []

        # Bubble-sort: move all creation ops to the left.
        # Loop until a pass makes no swaps.
        while True:
            swapped = False
            for i in range(len(ops) - 1):
                mode_i, action_i = ops[i]
                mode_j, action_j = ops[i + 1]
                # If annihilation comes before creation, swap them
                if action_i == 0 and action_j == 1:
                    # Swap: {a_i, a†_j} = δ_{ij}
                    # a_i a†_j = δ_{ij} - a†_j a_i
                    # Delta term: if modes are equal, add δ_{ij}I.  It
                    # carries the coefficient *before* this swap's sign
                    # flip (including the signs of all earlier swaps).
                    if mode_i == mode_j:
                        # Remove the two ops that participate in the swap
                        delta_ops = list(ops[:i] + ops[i + 2 :])
                        delta_terms.append((delta_ops, current_coeff))
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    current_coeff = FermionOperator._multiply_coeffs(current_coeff, -1)
                    swapped = True
            if not swapped:
                break

        result = FermionOperator({tuple(ops): current_coeff})
        for delta_ops, d_coeff in delta_terms:
            if delta_ops:
                # Re-normal-order the delta term
                delta_op = FermionOperator._normal_order_single_term(
                    tuple(delta_ops), d_coeff
                )
                result = result + delta_op
            else:
                # Empty ops = identity
                result = result + FermionOperator.identity(d_coeff)
        return result

    # ------------------------------------------------------------------
    # parameter binding
    # ------------------------------------------------------------------

    def bind_parameters(self, binding: dict[Parameter, float]) -> FermionOperator:
        """Resolve all symbolic Parameters in coefficients to numeric values.

        Args:
            binding: Mapping from ``Parameter`` to concrete float values.

        Returns:
            A new ``FermionOperator`` with all parameters resolved.

        Raises:
            ValueError: If any ``Parameter`` in this operator is not bound.

        Example:
            >>> from quspect import Parameter
            >>> t = Parameter("t")
            >>> op = t * FermionOperator.number_operator(0)
            >>> op.bind_parameters({t: 0.5})
            FermionOperator(0.5 a†0 a0)
        """
        new_terms: dict[tuple[tuple[int, int], ...], CoeffValue] = {}
        for term, coeff in self._terms.items():
            evaluated = self._eval_coeff(coeff, binding)
            if not self._is_zero_coeff(evaluated):
                if term in new_terms:
                    try:
                        new_terms[term] = self._add_coeffs_numeric(
                            new_terms[term], evaluated
                        )
                    except ValueError:
                        new_terms[term] = evaluated
                else:
                    new_terms[term] = evaluated
        return FermionOperator(new_terms)

    # ------------------------------------------------------------------
    # equality / hash
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Check equality with another FermionOperator."""
        if not isinstance(other, FermionOperator):
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
                import cmath

                if not cmath.isclose(coeff, other_coeff, rel_tol=1e-9, abs_tol=1e-12):
                    return False
            else:
                if abs(float(coeff) - float(other_coeff)) > 1e-12:
                    return False
        return True

    def __hash__(self) -> int:
        """Hash based on sorted term-coefficient pairs."""
        items: list[tuple[tuple[tuple[int, int], ...], float | str]] = []
        for term, coeff in sorted(self._terms.items()):
            if isinstance(coeff, (Parameter, ParameterExpression)):
                items.append((term, str(hash(coeff))))
            elif isinstance(coeff, complex):
                items.append((term, round(coeff.real, 9) + round(coeff.imag, 9)))
            else:
                items.append((term, round(float(coeff), 9)))
        return hash(tuple(items))

    # ------------------------------------------------------------------
    # representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        if self.is_zero:
            return "FermionOperator(0)"

        parts: list[str] = []
        sorted_terms = sorted(
            self._terms.items(),
            key=lambda kv: () if not kv[0] else kv[0],
        )

        for term, coeff in sorted_terms:
            coeff_str = _format_coeff(coeff)
            term_str = _format_fermion_term(term)
            if coeff_str == "":
                parts.append(term_str)
            elif coeff_str == "-":
                parts.append(f"-{term_str}")
            else:
                parts.append(f"{coeff_str} {term_str}")

        return f"FermionOperator({' + '.join(parts)})"

    def __str__(self) -> str:
        """Compact string representation."""
        return self.__repr__()

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks."""
        if self.is_zero:
            return r"$0$"

        parts: list[str] = []
        sorted_terms = sorted(
            self._terms.items(),
            key=lambda kv: () if not kv[0] else kv[0],
        )

        for term, coeff in sorted_terms:
            coeff_str = self._latex_coeff(coeff)
            term_str = self._latex_fermion_term(term)
            # ``_latex_coeff`` uses ``:.6g`` so unit coefficients render as
            # "1" / "-1"; compare against the formatted string.
            if coeff_str == "1":
                parts.append(term_str)
            elif coeff_str == "-1":
                parts.append(f"-{term_str}")
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
                return f"{coeff.real:.6g}"
            return f"({coeff.real:.6g}+{coeff.imag:.6g}i)"
        return f"{float(coeff):.6g}"

    @staticmethod
    def _latex_fermion_term(term: tuple[tuple[int, int], ...]) -> str:
        """Format a fermionic term for LaTeX."""
        if not term:
            return r"\mathbb{I}"
        parts: list[str] = []
        for mode, action in term:
            if action == 1:
                parts.append(rf"a^{{\dagger}}_{{{mode}}}")
            else:
                parts.append(rf"a_{{{mode}}}")
        return r"\, ".join(parts)
