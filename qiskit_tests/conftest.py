"""Shared pytest configuration for the qiskit cross-validation suite.

This directory (``qiskit_tests/``) verifies quspect's public interfaces using
qiskit as the reference implementation.  It is not part of the default
``pytest`` run (``testpaths = ["tests"]``); run it explicitly with::

    .venv/bin/pytest qiskit_tests -p no:cacheprovider -q

Reusable conversion / comparison / statistics helpers live in
:mod:`qiskit_tests.helpers` (``from qiskit_tests.helpers import ...``).

Verified conventions (enforced in ``test_conventions.py``)
-----------------------------------------------------------
1. **Statevector**: ``Simulator().statevector(circ)`` is little-endian, qubit
   0 is the least-significant bit of the index -- identical to qiskit
   ``Statevector``.  No permutation is needed.
2. **Circuit API argument order** matches qiskit: ``cx(control, target)``,
   ``ccx(c1, c2, target)``, ``crx/cry/crz/cp(theta, control, target)``,
   ``cu3(theta, phi, lam, control, target)``.  Gate correspondences:
   ``cu3 -> qiskit cu(theta, phi, lam, 0, c, t)``, ``u1 -> p``,
   ``u2(phi, lam) -> u(pi/2, phi, lam)``, ``u3 -> u``, ``id -> id``.  All
   gates match qiskit *exactly*, with no global phase difference.
3. **Gate matrices**: ``quspect.gates._matrix_utils.gate_matrix(gate)`` is
   **big-endian** (the gate's first qubit argument is the most-significant
   bit): ``gate_matrix(g) == Operator(qiskit_gate).reverse_qargs().data``.
   ``NoiseChannel.kraus_operators()`` uses the same big-endian convention.
4. **Count bitstrings**: in quspect ``s[i]`` is clbit ``i`` (clbit 0
   *leftmost*); in qiskit clbit 0 is *rightmost*.  Hence
   ``quspect_bitstring == qiskit_bitstring[::-1]``.  The keys of
   ``SimulationResult.probabilities`` use the quspect convention too (they
   only exist when the circuit has measurements, and for ``Simulator.run``
   they are empirical frequencies ``count / shots``, not exact values).
5. **Known bug**: ``quspect.simulation.density_matrix.DensityMatrix
   .probabilities(qubits)`` with a strict subset of qubits calls
   ``partial_trace(qubits)``, i.e. it traces *out* the qubits that should be
   measured and returns the distribution of the remaining ones.  E.g. for the
   3-qubit state ``|001>`` (qubit 0 = 1) ``probabilities((0,))`` returns
   ``[1, 0, 0, 0]`` instead of ``[0, 1]``.

Bug-handling policy (all agents)
--------------------------------
* **Never modify quspect source code** from this suite.
* Write the assertion for the *correct* behaviour and mark the test with
  ``@pytest.mark.xfail(strict=True, reason="BUG: <one-sentence description>")``.
  Once the bug is fixed the test XPASSes, which ``strict=True`` turns into a
  failure, reminding us to drop the marker.
* Mark tests that compare against qiskit with ``@pytest.mark.qiskit_ref`` and
  tests that rely on random sampling (judged with statistical tolerances)
  with ``@pytest.mark.statistical``.  Statistical tests must use fixed seeds
  and normally 4000-20000 shots; see the tolerance design in
  :mod:`qiskit_tests.helpers`.

Fixtures
--------
* ``rng``: ``np.random.default_rng(12345)``.
* ``simulator``: ``Simulator(seed=1234)``.
* ``density_simulator``: ``DensityMatrixSimulator(seed=1234)``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")

from quspect.simulation import DensityMatrixSimulator, Simulator  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom markers used by the qiskit cross-validation suite."""
    config.addinivalue_line(
        "markers", "qiskit_ref: test cross-validates quspect against qiskit"
    )
    config.addinivalue_line(
        "markers",
        "statistical: test uses random sampling judged with statistical tolerances",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy random generator (seed 12345)."""
    return np.random.default_rng(12345)


@pytest.fixture
def simulator() -> Simulator:
    """Seeded statevector simulator (seed 1234)."""
    return Simulator(seed=1234)


@pytest.fixture
def density_simulator() -> DensityMatrixSimulator:
    """Seeded density-matrix simulator (seed 1234)."""
    return DensityMatrixSimulator(seed=1234)
