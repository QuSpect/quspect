"""Use QAOA and SciPy's BFGS optimizer to solve a small Max-Cut problem.

Run with::

    python -m pip install -e .
    python examples/qaoa_max_cut.py

QuSpect evaluates the exact Max-Cut expectation and its adjoint gradient.  The
external SciPy optimizer only owns the classical parameter-update loop.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from quspect import Parameter, QuantumCircuit, QubitOperator, Simulator
from quspect.variational import AdjointEngine, ObservableObjective

Edge = tuple[int, int]


def max_cut_operator(edges: Sequence[Edge]) -> QubitOperator:
    r"""Return :math:`C = \sum_{(u,v)} (I-Z_u Z_v)/2`."""
    terms: dict[tuple[tuple[int, str], ...], float] = {(): len(edges) / 2}
    for left, right in edges:
        zz = tuple(sorted(((left, "Z"), (right, "Z"))))
        terms[zz] = terms.get(zz, 0.0) - 0.5
    return QubitOperator(terms)


def build_qaoa_ansatz(
    num_qubits: int,
    edges: Sequence[Edge],
    depth: int,
) -> tuple[QuantumCircuit, tuple[Parameter, ...], tuple[Parameter, ...]]:
    """Build a depth-``p`` QAOA circuit for an unweighted Max-Cut graph."""
    gammas = tuple(Parameter(f"gamma_{layer}") for layer in range(depth))
    betas = tuple(Parameter(f"beta_{layer}") for layer in range(depth))

    circuit = QuantumCircuit(name=f"max_cut_qaoa_p{depth}")
    for qubit in range(num_qubits):
        circuit.h(qubit)

    for gamma, beta in zip(gammas, betas, strict=True):
        # exp(-i gamma C) differs only by a global phase from
        # exp(+i gamma Z_u Z_v / 2), implemented as CX RZ(-gamma) CX.
        for left, right in edges:
            circuit.cx(left, right).rz(-gamma, right).cx(left, right)
        for qubit in range(num_qubits):
            # RX(2 beta) = exp(-i beta X).
            circuit.rx(2 * beta, qubit)

    return circuit, gammas, betas


def cut_value(bit_assignment: Sequence[int], edges: Sequence[Edge]) -> int:
    """Count graph edges whose endpoints have different assigned bits."""
    return sum(bit_assignment[left] != bit_assignment[right] for left, right in edges)


def exact_max_cut(num_qubits: int, edges: Sequence[Edge]) -> tuple[int, list[str]]:
    """Brute-force the tiny example graph to provide a classical check."""
    best_value = -1
    best_bitstrings: list[str] = []
    for state in range(1 << num_qubits):
        assignment = tuple((state >> qubit) & 1 for qubit in range(num_qubits))
        value = cut_value(assignment, edges)
        bitstring = format(state, f"0{num_qubits}b")
        if value > best_value:
            best_value = value
            best_bitstrings = [bitstring]
        elif value == best_value:
            best_bitstrings.append(bitstring)
    return best_value, best_bitstrings


def main() -> None:
    """Optimize a four-vertex ring and report the resulting QAOA state."""
    num_qubits = 4
    edges: tuple[Edge, ...] = ((0, 1), (1, 2), (2, 3), (3, 0))
    depth = 2

    circuit, gammas, betas = build_qaoa_ansatz(num_qubits, edges, depth)
    objective = ObservableObjective(circuit, {"cut": max_cut_operator(edges)})
    plan = AdjointEngine(dtype="float64").compile(objective)

    initial_by_parameter = {
        parameter: value
        for layer in range(depth)
        for parameter, value in ((gammas[layer], 0.4), (betas[layer], 0.2))
    }
    x0 = np.asarray(
        [initial_by_parameter[parameter] for parameter in plan.parameter_order]
    )

    evaluations = 0

    def loss_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        # SciPy minimizes, while Max-Cut maximizes, so negate both quantities.
        nonlocal evaluations
        evaluations += 1
        result = plan.evaluate(x)
        return -result.value("cut"), -result.derivative("cut")

    optimum = minimize(
        loss_and_gradient,
        x0=x0,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 200},
    )

    optimized_bindings = dict(
        zip(plan.parameter_order, optimum.x.tolist(), strict=True)
    )
    optimized_circuit = circuit.bind_parameters(optimized_bindings)
    statevector = Simulator().statevector(optimized_circuit)
    probabilities = np.abs(statevector) ** 2
    most_likely_state = int(np.argmax(probabilities))
    most_likely_bits = tuple(
        (most_likely_state >> qubit) & 1 for qubit in range(num_qubits)
    )
    most_likely_bitstring = format(most_likely_state, f"0{num_qubits}b")
    classical_value, classical_solutions = exact_max_cut(num_qubits, edges)

    print(f"graph edges: {edges}")
    print(f"QAOA depth: {depth}")
    print(f"parameter order: {[p.name for p in plan.parameter_order]}")
    print(f"BFGS success: {optimum.success} ({optimum.message})")
    print(f"objective evaluations: {evaluations}")
    print(f"optimized parameters: {optimum.x.round(8).tolist()}")
    print(f"optimized expected cut: {-float(optimum.fun):.8f}")
    print(
        "most likely bitstring: "
        f"{most_likely_bitstring} "
        f"(probability={probabilities[most_likely_state]:.6f}, "
        f"cut={cut_value(most_likely_bits, edges)})"
    )
    print(
        f"classical optimum: {classical_value}, "
        f"optimal bitstrings: {classical_solutions}"
    )


if __name__ == "__main__":
    main()
