Variational Objectives
======================

QuSpect represents a variational calculation as data that is compiled into a
reusable numerical plan.  The four layers have deliberately separate jobs:

.. code-block:: text

   ObservableObjective ─┬─> AdjointEngine  ─> PreparedObjective
          problem       │       exact C++          local plan
                        └─> SamplingEngine ─> PreparedSamplingObjective
                              shots/QPU            submit + decode
                                  |
                                  v
                       ObjectiveEvaluation / ObjectiveSweep

An :class:`~quspect.variational.ObservableObjective` describes *what* to
evaluate: one parameterized circuit and a named mapping of observables.  An
:class:`~quspect.variational.AdjointEngine` selects *how* to evaluate it.  Its
:meth:`~quspect.variational.AdjointEngine.compile` method validates and lowers
the objective once.  The resulting
:class:`~quspect.variational.PreparedObjective` can then be evaluated repeatedly
with small parameter mappings while the circuit tape, Pauli masks, and
parameter layout remain cached in the native backend.

An objective is not tied to a decorated QNode or one provider's primitive.
The native adjoint plan evaluates directly.  A sampled plan can instead submit
a provider-neutral execution job, persist its credential-free reference, and
decode the recovered batch later.  Both strategies return the same structured
value/Jacobian result.

H2-style VQE
------------

The following two-qubit example uses a reduced H2-style Pauli Hamiltonian.  A
single adjoint evaluation returns both its expectation value and the complete
derivative vector.

.. code-block:: python

   from quspect import Parameter, QuantumCircuit, QubitOperator
   from quspect.variational import AdjointEngine, ObservableObjective

   theta = Parameter("theta")

   ansatz = QuantumCircuit(name="h2_ansatz")
   ansatz.ry(theta, 0).cx(0, 1)

   h2 = QubitOperator({
       (): -1.052373245772859,
       ((0, "Z"),): 0.39793742484318045,
       ((1, "Z"),): -0.39793742484318045,
       ((0, "Z"), (1, "Z")): -0.01128010425623538,
       ((0, "X"), (1, "X")): 0.18093119978423156,
   })

   objective = ObservableObjective(ansatz, {"energy": h2})
   plan = AdjointEngine(dtype="float64").compile(objective)

   evaluation = plan.evaluate({theta: 0.2})
   energy = evaluation.value("energy")
   gradient = evaluation.derivative("energy")

``plan`` is the object to retain inside an optimizer loop.  Constructing a
new engine or recompiling the objective for every optimizer step discards the
main benefit of the prepared representation.

For example, SciPy can consume the value and derivative from the same native
evaluation (SciPy is an optional user dependency, not required by QuSpect):

.. code-block:: python

   import numpy as np
   from scipy.optimize import minimize

   def energy_and_gradient(x):
       result = plan.evaluate({theta: float(x[0])})
       return result.value("energy"), result.derivative("energy")

   optimum = minimize(
       energy_and_gradient,
       x0=np.array([0.0]),
       jac=True,
       method="BFGS",
   )

Named observables and fixed shapes
----------------------------------

Observables are always supplied as a mapping.  Their names make selection
explicit and are retained in ``observable_order``:

.. code-block:: python

   diagnostics = ObservableObjective(
       ansatz,
       {
           "energy": h2,
           "parity": QubitOperator({((0, "Z"), (1, "Z")): 1.0}),
       },
   )
   diagnostic_plan = AdjointEngine().compile(diagnostics)
   result = diagnostic_plan.evaluate({theta: 0.2})

   print(result.observable_order)
   print(result.parameter_order)
   print(result.values.shape)
   print(result.jacobian.shape)

The structured result has a stable array contract.  If an objective contains
``M`` observables and ``P`` parameters, an
:class:`~quspect.variational.ObjectiveEvaluation` always contains:

* ``values`` with shape ``(M,)``;
* ``jacobian`` with shape ``(M, P)``;
* ``observable_order`` and ``parameter_order`` describing the corresponding
  axes; and
* ``metadata`` for engine-specific diagnostic information.

The arrays do not collapse when ``M == 1`` or ``P == 1``.  Use
``value(name_or_index)`` and ``derivative(name_or_index)`` when selecting one
observable is more convenient.  This avoids an API whose return type silently
changes between a scalar and an array as observables are added.

Parameter mappings are the recommended interface.  Binding by parameter makes
optimizer boundaries explicit and prevents a circuit edit from silently
changing the meaning of a vector element.  A one-dimensional sequence in
``parameter_order`` is also accepted for vector-based optimizer adapters; code
using that form should read and retain the plan's authoritative order rather
than reconstructing it independently.  Mapping keys may be
:class:`~quspect.circuit.Parameter` objects (recommended) or their string
names.

Batch sweeps
------------

:meth:`~quspect.variational.PreparedObjective.evaluate_many` evaluates several
parameter mappings without recompiling the objective:

.. code-block:: python

   sweep = plan.evaluate_many([
       {theta: -0.5},
       {theta: 0.0},
       {theta: 0.5},
   ])

   first = sweep.at(0)
   print(first.value("energy"))

For a dense optimizer or grid-search workload, the same sweep can be supplied
as a numeric matrix whose columns follow ``plan.parameter_order``:

.. code-block:: python

   import numpy as np

   parameter_matrix = np.array([[-0.5], [0.0], [0.5]])
   dense_sweep = plan.evaluate_many(parameter_matrix)

For ``B`` points, ``M`` observables, and ``P`` parameters, an
:class:`~quspect.variational.ObjectiveSweep` stores ``values`` with shape
``(B, M)`` and ``jacobian`` with shape ``(B, M, P)``.  Calling ``at(index)``
returns the corresponding :class:`~quspect.variational.ObjectiveEvaluation`
without changing the single-point result contract.

Current adjoint engine scope
----------------------------

The adjoint engine performs exact full-state-vector differentiation in C++.
It is intended for noiseless expectation objectives with real Hermitian Pauli
Hamiltonians.  Symbolic ``RX``, ``RY``, and ``RZ`` angles may be affine
expressions and may share parameters.  Measurements and noise operations are
not part of a differentiable objective.  Use the regular simulation APIs for
sampling and measurement workflows.

The engine precision is chosen independently from the problem description:

.. code-block:: python

   fast_plan = AdjointEngine(dtype="float32").compile(objective)
   accurate_plan = AdjointEngine(dtype="float64").compile(objective)

This separation lets exact and sampled engines share one objective and result
contract without changing circuit construction or optimizer integration code.

Shot-based and hardware objectives
----------------------------------

:class:`~quspect.variational.SamplingEngine` lowers each Pauli observable into
measurement circuits.  Qubit-wise commuting terms share a measurement basis;
identity terms are exact constants.  Gradients use local parameter shifts for
each symbolic ``RX``, ``RY``, or ``RZ`` occurrence, followed by the affine
chain rule.  This remains correct when one parameter appears in several gates
or a gate angle is an expression such as ``2 * theta - phi``.

The same H2-style objective can be evaluated through the local sampler:

.. code-block:: python

   from quspect import SamplingEngine, Simulator

   sampled = SamplingEngine(Simulator(seed=7), shots=8192).compile(objective)
   evaluation = sampled.evaluate({theta: 0.2})

Or submitted to a LogicalQubit device without changing the objective:

.. code-block:: python

   from quspect import JobReference, SamplingEngine
   from quspect.backends.logicalqubit import LogicalQubitProvider

   provider = LogicalQubitProvider()  # env or ~/.lqcloud/config.json
   backend = provider.get_backend("QZ02")
   sampled = SamplingEngine(
       backend,
       shots=4096,
       execution_options={
           "transpile": False,
           "initial_layout": [0, 1],
           "readout_correction": True,
       },
   ).compile(objective)

   objective_job = sampled.submit({theta: 0.2})
   saved_reference = objective_job.reference.to_dict()
   evaluation = objective_job.result(timeout=600)

The execution request carries stable labels for every base, shift, and Pauli
group circuit.  LogicalQubit native batches are limited to 64 circuits;
QuSpect partitions a larger objective into child jobs and exposes one
:class:`~quspect.backends.CompositeJob` while preserving label order.
Several points passed to ``sampled.submit_many(...)`` are flattened into one
recoverable execution request; ``sampled.evaluate_many(...)`` is its blocking
shortcut.  An optimizer sweep therefore does not create one cloud task per
point.  The ``auto`` result format lets lqcloud choose between counts and
per-shot memory according to readout width.

For older servers where the SDK's private native-batch fallback must be
avoided, pass ``provider_batching="individual"`` in ``execution_options``.
QuSpect will then submit and track each circuit itself, allowing complete
``PartialSubmissionError`` reporting and best-effort cancellation at the cost
of additional provider tasks.

After a process restart, reconstruct the provider job and attach the compiled
decoder.  A job reference intentionally contains no API key or token:

.. code-block:: python

   reference = JobReference.from_dict(saved_reference)
   execution_job = provider.retrieve_job(reference)
   evaluation = sampled.attach(execution_job).result(timeout=600)

``attach`` validates a stable plan fingerprint together with the backend,
labels, experiment shapes, shots, and result format.  A reference from a
different objective is therefore rejected even if its circuit batch happens
to have the same size.  Multi-point jobs are recovered with
``sampled.attach_many(execution_job)``.

A job reference is a credential-free manifest rather than a signed platform
receipt.  Store it in a trusted location: the fingerprint prevents accidental
misattachment, but it is not intended to authenticate JSON modified by an
untrusted party.

The current sampled gradient rules cover symbolic ``RX``, ``RY``, and ``RZ``.
Symbolic controlled rotations and symbolic observable coefficients are
rejected explicitly rather than evaluated with an invalid two-point rule.
