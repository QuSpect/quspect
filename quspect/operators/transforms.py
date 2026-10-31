"""Fermion-to-qubit transformations.

This module provides transformations that map fermionic operators to
qubit operators: the Jordan-Wigner and Bravyi-Kitaev mappings.
"""

from __future__ import annotations

from quspect.operators.fermion_operator import FermionOperator
from quspect.operators.qubit_operator import CoeffValue, QubitOperator, TermKey

# ---------------------------------------------------------------------------
# Pauli single-qubit multiplication table
# ---------------------------------------------------------------------------
# Maps (pauli_a, pauli_b) → (result_pauli, phase_factor)
# Following the convention: σ_a σ_b = phase * σ_result

_PAULI_MUL: dict[tuple[str, str], tuple[str, complex]] = {
    # I * anything = anything
    ("I", "I"): ("I", 1 + 0j),
    ("I", "X"): ("X", 1 + 0j),
    ("I", "Y"): ("Y", 1 + 0j),
    ("I", "Z"): ("Z", 1 + 0j),
    # anything * I = anything
    ("X", "I"): ("X", 1 + 0j),
    ("Y", "I"): ("Y", 1 + 0j),
    ("Z", "I"): ("Z", 1 + 0j),
    # X * X = I
    ("X", "X"): ("I", 1 + 0j),
    # X * Y = iZ
    ("X", "Y"): ("Z", 1j),
    # X * Z = -iY
    ("X", "Z"): ("Y", -1j),
    # Y * X = -iZ
    ("Y", "X"): ("Z", -1j),
    # Y * Y = I
    ("Y", "Y"): ("I", 1 + 0j),
    # Y * Z = iX
    ("Y", "Z"): ("X", 1j),
    # Z * X = iY
    ("Z", "X"): ("Y", 1j),
    # Z * Y = -iX
    ("Z", "Y"): ("X", -1j),
    # Z * Z = I
    ("Z", "Z"): ("I", 1 + 0j),
}


def _multiply_pauli_terms(
    term_a: TermKey,
    coeff_a: complex,
    term_b: TermKey,
    coeff_b: complex,
) -> list[tuple[TermKey, complex]]:
    """Multiply two Pauli terms and return the result.

    Multiplies two Pauli strings qubit-by-qubit, producing a single
    result term with a coefficient that includes the accumulated
    phase factor.

    Args:
        term_a: First Pauli term as ``((qubit, pauli), ...)``.
        coeff_a: Coefficient of the first term.
        term_b: Second Pauli term.
        coeff_b: Coefficient of the second term.

    Returns:
        A list containing a single ``(result_term, result_coeff)`` pair.
    """
    d_a = dict(term_a)
    d_b = dict(term_b)
    all_qubits = sorted(set(d_a.keys()) | set(d_b.keys()))

    phase: complex = 1 + 0j
    result_parts: list[tuple[int, str]] = []

    for q in all_qubits:
        pa = d_a.get(q, "I")
        pb = d_b.get(q, "I")
        p_res, ph = _PAULI_MUL[(pa, pb)]
        phase *= ph
        if p_res != "I":
            result_parts.append((q, p_res))

    total_coeff = coeff_a * coeff_b * phase
    return [(tuple(result_parts), total_coeff)]


def _operator_mul(op_a: QubitOperator, op_b: QubitOperator) -> QubitOperator:
    """Multiply two QubitOperators using Pauli algebra.

    The result is the sum over all pairwise products of terms from
    *op_a* and *op_b*.

    Args:
        op_a: First operator.
        op_b: Second operator.

    Returns:
        The product operator ``op_a * op_b``.
    """
    import cmath

    from quspect.circuit.parameter import Parameter, ParameterExpression

    accumulated: dict[TermKey, complex] = {}
    for term_a, coeff_a in op_a:
        assert not isinstance(
            coeff_a, (Parameter, ParameterExpression)
        )  # guarded upstream
        c_a = complex(coeff_a)
        for term_b, coeff_b in op_b:
            assert not isinstance(
                coeff_b, (Parameter, ParameterExpression)
            )  # guarded upstream
            c_b = complex(coeff_b)
            products = _multiply_pauli_terms(term_a, c_a, term_b, c_b)
            for r_term, r_coeff in products:
                if r_term in accumulated:
                    accumulated[r_term] = accumulated[r_term] + r_coeff
                    if cmath.isclose(abs(accumulated[r_term]), 0, abs_tol=1e-15):
                        del accumulated[r_term]
                elif cmath.isclose(abs(r_coeff), 0, abs_tol=1e-15):
                    pass  # skip terms with zero coefficient
                else:
                    accumulated[r_term] = r_coeff

    # Convert complex → float when imaginary part is negligible
    final_terms: dict[TermKey, CoeffValue] = {}
    for term, coeff in accumulated.items():
        if abs(coeff.imag) < 1e-15:
            final_terms[term] = float(coeff.real)
        else:
            final_terms[term] = coeff

    return QubitOperator(final_terms)


# ---------------------------------------------------------------------------
# Jordan-Wigner transformation
# ---------------------------------------------------------------------------


def jordan_wigner(operator: FermionOperator) -> QubitOperator:
    """Apply the Jordan-Wigner transformation.

    Transforms a fermionic operator into a qubit operator using the
    standard Jordan-Wigner mapping::

        a†_i → 1/2 (X_i - i Y_i) Z_{i-1} ... Z_0
        a_i  → 1/2 (X_i + i Y_i) Z_{i-1} ... Z_0

    The parity string :math:`Z_{i-1} \\cdots Z_0` ensures proper
    anti-commutation relations between different fermionic modes.

    Args:
        operator: A :class:`FermionOperator` to transform.

    Returns:
        The equivalent :class:`QubitOperator`.

    Raises:
        ValueError: If any coefficient in the fermionic operator is symbolic
            (bind parameters first).

    Example:
        >>> from quspect.operators import FermionOperator
        >>> from quspect.operators.transforms import jordan_wigner
        >>> fop = FermionOperator.number_operator(0, coefficient=1.0)
        >>> qop = jordan_wigner(fop)
        >>> qop  # n_0 = (I - Z_0) / 2
        QubitOperator(0.5 I + -0.5 Z0)
    """
    result = QubitOperator.zero()

    # Determine required qubit count
    max_mode = -1
    for term, _coeff in operator:
        for mode, _action in term:
            if mode > max_mode:
                max_mode = mode
    num_qubits = max_mode + 1

    for term, coeff in operator:
        qubit_op = _jw_transform_single_term(term, coeff, num_qubits)
        result = result + qubit_op

    return result


def _jw_transform_single_term(
    term: tuple[tuple[int, int], ...],
    coeff: CoeffValue,
    num_qubits: int,
) -> QubitOperator:
    """Transform a single fermionic term to a QubitOperator.

    Args:
        term: A fermionic term as ``((mode, action), ...)``.
        coeff: The coefficient of this term.
        num_qubits: Total number of qubits (max mode + 1).

    Returns:
        The transformed QubitOperator.
    """
    from quspect.circuit.parameter import Parameter, ParameterExpression

    if isinstance(coeff, (Parameter, ParameterExpression)):
        raise ValueError(
            f"Coefficient {coeff} is symbolic.  Please call "
            f"bind_parameters() on the FermionOperator before applying "
            f"the Jordan-Wigner transform."
        )

    if not term:
        # Identity — return identity on all qubits
        return QubitOperator({(): coeff})

    # For each (mode, action), create the JW expansion as a QubitOperator
    # on the full qubit space, then multiply sequentially.
    result: QubitOperator | None = None

    for mode, action in term:
        ladder_op = _jw_ladder_operator(mode, action, num_qubits)
        # Multiply operators sequentially — each ladder operator
        # already includes its 1/2 factor.  The coefficient is
        # applied after all multiplications.
        result = ladder_op if result is None else _operator_mul(result, ladder_op)

    if result is None:
        return QubitOperator({(): 1.0})

    # Multiply by the fermionic term coefficient
    return coeff * result


def _jw_ladder_operator(
    mode: int,
    action: int,
    num_qubits: int,
) -> QubitOperator:
    """Create the JW representation of a single ladder operator.

    Args:
        mode: The fermionic mode (qubit index).
        action: 1 for creation (a†), 0 for annihilation (a).
        num_qubits: Total number of qubits in the system.

    Returns:
        The QubitOperator for this ladder operator on *num_qubits* qubits.

    For mode *i*::

        a†_i → (1/2) (X_i - i Y_i) Z_{i-1} ... Z_0
        a_i  → (1/2) (X_i + i Y_i) Z_{i-1} ... Z_0
    """
    # Build Z parity string: Z on all qubits k < mode
    parity_parts: list[tuple[int, str]] = [(k, "Z") for k in range(mode)]

    # X contribution: (1/2) X_mode * parity
    x_parts = [(mode, "X")] + parity_parts
    x_term = tuple(
        sorted(x_parts, key=lambda x: (x[0], {"I": 0, "X": 1, "Y": 2, "Z": 3}[x[1]]))
    )

    op = QubitOperator({x_term: 0.5})

    # Y contribution: (∓i/2) Y_mode * parity
    # a†: (-i/2),  a: (+i/2)
    y_coeff: complex = -0.5j if action == 1 else 0.5j
    y_parts = [(mode, "Y")] + parity_parts
    y_term = tuple(
        sorted(y_parts, key=lambda x: (x[0], {"I": 0, "X": 1, "Y": 2, "Z": 3}[x[1]]))
    )

    y_op = QubitOperator({y_term: y_coeff})
    return op + y_op


# ---------------------------------------------------------------------------
# Bravyi-Kitaev transformation
# ---------------------------------------------------------------------------


def bravyi_kitaev(
    operator: FermionOperator,
    n_qubits: int | None = None,
) -> QubitOperator:
    """Apply the Bravyi-Kitaev transformation.

    Transforms a fermionic operator into a qubit operator using the
    Bravyi-Kitaev mapping of Seeley, Richard and Love (J. Chem. Phys. 137,
    224109, 2012).  Qubit *j* stores the parity of a contiguous block of
    modes given by a Fenwick (binary-indexed) tree, so every ladder
    operator acts on only O(log N) qubits.

    For mode *j* three index sets are used:

    * the **update set** U(j): qubits ``q > j`` whose stored parity
      includes mode *j*;
    * the **parity set** P(j): qubits ``q < j`` whose stored parities
      multiply to the parity of modes ``0 .. j-1``;
    * the **remainder set** R(j) = P(j) \\ F(j), where the **flip set**
      F(j) holds the children of *j* in the Fenwick tree (empty for even
      *j*).

    The ladder operators are then::

        a†_j = ½ X_{U(j)} X_j Z_{P(j)} − i/2 X_{U(j)} Y_j Z_{R(j)}
        a_j  = ½ X_{U(j)} X_j Z_{P(j)} + i/2 X_{U(j)} Y_j Z_{R(j)}

    When the number of qubits is not a power of two the Fenwick tree is
    truncated (update-set members ``>= n_qubits`` are dropped), exactly
    as in OpenFermion's ``bravyi_kitaev``; the result still obeys the
    canonical anticommutation relations.

    Because the encoding depends on the number of qubits, two operators
    must be transformed with the same ``n_qubits`` before their images can
    be added or multiplied.

    Args:
        operator: A :class:`FermionOperator` to transform.
        n_qubits: Number of qubits (= fermionic modes) of the encoding.
            Defaults to ``max_mode + 1`` of *operator* (``0`` for an
            operator without ladder operators).

    Returns:
        The equivalent :class:`QubitOperator`.

    Raises:
        ValueError: If any coefficient in the fermionic operator is symbolic
            (bind parameters first), or if ``n_qubits`` is negative or
            smaller than ``max_mode + 1``.
        TypeError: If ``n_qubits`` is not an integer.

    Example:
        >>> from quspect.operators import FermionOperator
        >>> from quspect.operators.transforms import bravyi_kitaev
        >>> fop = FermionOperator.number_operator(0, coefficient=1.0)
        >>> bravyi_kitaev(fop)  # n_0 = (I - Z_0) / 2
        QubitOperator(0.5 I + -0.5 Z0)
        >>> bravyi_kitaev(fop, n_qubits=4)  # qubit 1 stores n_0 + n_1
        QubitOperator(0.5 I + -0.5 Z0)
    """
    from quspect.circuit.parameter import Parameter, ParameterExpression

    max_mode = -1
    for term, _coeff in operator:
        for mode, _action in term:
            if mode > max_mode:
                max_mode = mode

    if n_qubits is None:
        num_qubits = max_mode + 1
    else:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
            raise TypeError(f"n_qubits must be an int, got {type(n_qubits).__name__}.")
        if n_qubits < max_mode + 1:
            raise ValueError(
                f"n_qubits={n_qubits} is too small for an operator acting on "
                f"mode {max_mode}; need at least {max_mode + 1}."
            )
        num_qubits = n_qubits

    cache: dict[tuple[int, int], QubitOperator] = {}
    result = QubitOperator.zero()

    for term, coeff in operator:
        if isinstance(coeff, (Parameter, ParameterExpression)):
            raise ValueError(
                f"Coefficient {coeff} is symbolic.  Please call "
                f"bind_parameters() on the FermionOperator before applying "
                f"the Bravyi-Kitaev transform."
            )
        if not term:
            result = result + QubitOperator({(): coeff})
            continue
        product: QubitOperator | None = None
        for ladder in term:
            if ladder not in cache:
                cache[ladder] = _bk_ladder_operator(ladder[0], ladder[1], num_qubits)
            ladder_op = cache[ladder]
            product = (
                ladder_op if product is None else _operator_mul(product, ladder_op)
            )
        assert product is not None
        result = result + coeff * product

    return result


def _bk_update_set(index: int, num_qubits: int) -> set[int]:
    """Return the BK update set U(index) (qubits above *index* storing it).

    Args:
        index: Fermionic mode index (0-based).
        num_qubits: Number of qubits; members ``>= num_qubits`` are dropped.

    Returns:
        The qubit indices ``q > index`` whose Fenwick block contains mode
        *index*.
    """
    indices: set[int] = set()
    pos = index + 1  # Fenwick trees are 1-indexed
    pos += pos & -pos
    while pos <= num_qubits:
        indices.add(pos - 1)
        pos += pos & -pos
    return indices


def _bk_flip_set(index: int) -> set[int]:
    """Return the BK flip set F(index) (children of *index* in the tree).

    The parity of the qubits ``{index} ∪ F(index)`` equals the occupation of
    mode *index*.

    Args:
        index: Fermionic mode index (0-based).

    Returns:
        The qubit indices ``q < index`` that are children of *index*.
    """
    indices: set[int] = set()
    pos = index + 1
    parent = pos & (pos - 1)  # start of the Fenwick block of ``pos``
    child = pos - 1
    while child != parent:
        indices.add(child - 1)
        child &= child - 1
    return indices


def _bk_parity_set(index: int) -> set[int]:
    """Return the BK parity set P(index).

    The parity of the qubits in P(index) equals the parity of the modes
    ``0 .. index-1``.

    Args:
        index: Fermionic mode index (0-based).

    Returns:
        The qubit indices ``q < index`` forming the prefix-parity set.
    """
    indices: set[int] = set()
    pos = index  # prefix of length ``index`` (modes 0..index-1)
    while pos > 0:
        indices.add(pos - 1)
        pos &= pos - 1
    return indices


def _build_bk_columns(num_qubits: int) -> list[set[int]]:
    """Build the BK column sets for all modes.

    ``column_sets[p]`` is the set of qubits whose stored value depends on
    mode *p*, i.e. the non-zero rows of column *p* of the BK encoding
    matrix β.  It equals ``{p} ∪ U(p)``.  For a power-of-two size β is
    defined recursively by::

        β₁ = [1]
        β_{2k} = [[β_k, 0], [A, β_k]]

    where ``A`` is zero except for an all-ones last row.  Other sizes use
    the top-left block of the next power of two (truncated Fenwick tree).

    Args:
        num_qubits: Number of fermionic modes (= number of qubits).

    Returns:
        A list where ``column_sets[p]`` is the set of qubit indices *q*
        such that β_{q,p} = 1.
    """
    return [{p} | _bk_update_set(p, num_qubits) for p in range(num_qubits)]


def _pauli_key(parts: dict[int, str]) -> TermKey:
    """Build a canonical (qubit-sorted) Pauli term key."""
    return tuple(sorted(parts.items()))


def _bk_ladder_operator(
    mode: int,
    action: int,
    num_qubits: int,
) -> QubitOperator:
    """Create the BK representation of a single ladder operator.

    For mode *j*::

        a†_j = ½ X_{U(j)} X_j Z_{P(j)} − i/2 X_{U(j)} Y_j Z_{R(j)}
        a_j  = ½ X_{U(j)} X_j Z_{P(j)} + i/2 X_{U(j)} Y_j Z_{R(j)}

    with R(j) = P(j) \\ F(j).

    Args:
        mode: The fermionic mode.
        action: 1 for creation (a†), 0 for annihilation (a).
        num_qubits: Number of qubits in the encoding.

    Returns:
        The QubitOperator for this ladder operator.
    """
    update = _bk_update_set(mode, num_qubits)
    parity = _bk_parity_set(mode)
    remainder = parity - _bk_flip_set(mode)

    x_parts: dict[int, str] = dict.fromkeys(update, "X")
    x_parts.update(dict.fromkeys(parity, "Z"))
    x_parts[mode] = "X"

    y_parts: dict[int, str] = dict.fromkeys(update, "X")
    y_parts.update(dict.fromkeys(remainder, "Z"))
    y_parts[mode] = "Y"

    y_coeff: complex = -0.5j if action == 1 else 0.5j
    return QubitOperator({_pauli_key(x_parts): 0.5, _pauli_key(y_parts): y_coeff})


def reverse_jordan_wigner(operator: QubitOperator) -> FermionOperator:
    """Reverse the Jordan-Wigner transformation. (Not yet implemented.)

    Args:
        operator: A QubitOperator to reverse-transform.

    Returns:
        The equivalent FermionOperator.

    Raises:
        NotImplementedError: Always — this transform is planned for a
            future release.
    """
    raise NotImplementedError(
        "Reverse Jordan-Wigner transformation is not yet implemented."
    )
