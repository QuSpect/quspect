Quadratic Optimization
======================

QuSpect separates problem modeling, mathematical intermediate
representations, hardware lowering, and execution:

.. code-block:: text

   constrained problem
          |
          v
   BinaryQuadraticModel
          |
          v
      IsingProblem
          |
          v
   HardwareIsingProgram
          |
          v
   provider codec / local solver

This boundary keeps a provider's matrix convention out of user-facing models.
It also allows the same quadratic objective to be executed by a classical
optimizer, lowered to a QAOA Hamiltonian, or encoded for an Ising-machine
backend.

Symbolic modeling
-----------------

``QuadraticProgram`` is the user-facing layer.  It creates stable binary
variables, accepts objectives of degree at most two, and compiles supported
constraints into penalties:

* a linear equality residual :math:`r(x)=0` becomes
  :math:`\lambda r(x)^2`;
* an expression explicitly declared nonnegative and constrained to zero
  becomes the direct penalty :math:`\lambda q(x)`.

For maximization, the compiler subtracts these penalties so that violations
are still disfavored.  ``CompilationResult`` retains the original objective
and constraints separately from the penalized BQM for business-level
evaluation and feasibility checks.

Penalties are soft constraints.  The caller must choose coefficients large
enough that no objective improvement can make a violating assignment
preferable.  Always check ``compiled.is_feasible(bits)`` after solving and
use ``compiled.evaluate_objective(bits)`` for the unpenalized business value.

.. code-block:: python

   from quspect.optimization import (
       QuadraticProgram,
       compile_quadratic_program,
   )

   problem = QuadraticProgram("one-hot")
   choice = problem.binary_array("choice", 3)
   problem.minimize(3 * choice[0] + choice[1] + 2 * choice[2])
   problem.add_linear_equality(
       choice.sum() - 1,
       name="choose-exactly-one",
       penalty=10,
   )
   compiled = compile_quadratic_program(problem)

Binary quadratic models
-----------------------

``BinaryQuadraticModel`` stores an objective in the canonical form

.. math::

   E(v) = c + a^T v + \frac{1}{2} v^T B v,

where ``B`` is symmetric with a zero diagonal.  Variables can use either the
binary domain :math:`\{0, 1\}` or the spin domain :math:`\{-1, +1\}`.
Variable names and vector order are explicit, and all coefficient arrays are
copied and made read-only.

QUBO matrices use the convention

.. math::

   E(x) = x^T Qx + c.

QuSpect accepts upper-triangular, lower-triangular, symmetric, or mixed QUBO
matrices and normalizes opposing entries into one pair coefficient.  The
canonical exported QUBO matrix is upper triangular.

Ising conversion
----------------

Binary and spin variables are related by

.. math::

   s = 2x - 1,\qquad x = \frac{s + 1}{2}.

``binary_to_ising`` and ``ising_to_binary`` retain the complete constant
offset and preserve the objective value for every assignment.  The logical
``IsingProblem`` convention is

.. math::

   E(s) = c + h^T s + \frac{1}{2} s^T Js.

Hardware matrix lowering
------------------------

``HardwareIsingProgram`` represents the exact dense matrix at the
Ising-machine boundary.  Its hardware energy convention is

.. math::

   E_\mathrm{hw}(t) = -t^T Mt.

The matrix is finite, symmetric, and has a zero diagonal.  A logical linear
field is converted into a coupling to an auxiliary gauge spin.  Local decode
metadata fixes that gauge to ``+1``, removes auxiliary spins, and restores the
original objective using an explicit scale and offset.

Because the provider evaluates the full symmetric matrix, each off-diagonal
entry appears twice in :math:`t^T Mt`.  Lowering therefore writes
:math:`-\frac{1}{2}J_{ij}` for a logical pair coefficient :math:`J_{ij}` and
:math:`-\frac{1}{2}h_i` for a field coupled to the gauge spin.  This factor is
part of the transport contract, not an optimizer-specific convention.

The provider-neutral lowering and CSV codec are local transformations only;
they do not perform authentication, networking, task creation, or result
polling:

.. code-block:: python

   import numpy as np

   from quspect.optimization import (
       IsingProblem,
       compile_hardware_ising,
       encode_ising_matrix_csv,
   )

   logical = IsingProblem(
       variables=("s0", "s1"),
       linear=np.array([0.5, -1.0]),
       quadratic=np.array([[0.0, 2.0], [2.0, 0.0]]),
       offset=3.0,
   )
   artifact = compile_hardware_ising(logical)

   print(artifact.bit_num)
   print(encode_ising_matrix_csv(artifact))

For every valid logical assignment, compilation guarantees

.. math::

   E_\mathrm{logical}
   =
   \mathrm{objective\_scale}\,E_\mathrm{hw}
   + \mathrm{objective\_offset}.

This equality is the primary compatibility invariant for future local and
remote optimization backends.

Local execution
---------------

``NativeIsingSolver`` consumes ``HardwareIsingProgram`` directly, so the
local C++ simulator and a future remote adapter share the same validated
matrix boundary.  See :doc:`ising_solver` for its algorithm, complexity, and
determinism guarantees.

The complete local path is:

.. code-block:: python

   from quspect.optimization import (
       NativeIsingSolver,
       bqm_to_ising,
       compile_hardware_ising,
       spin_samples_to_binary,
   )

   logical = bqm_to_ising(compiled.bqm)
   hardware = compile_hardware_ising(logical)
   result = NativeIsingSolver(seed=7, num_reads=32).solve(hardware)

   bits = spin_samples_to_binary(result.best_sample)
   solution = compiled.decode(bits)
   assert compiled.is_feasible(bits)
   business_value = compiled.evaluate_objective(bits)

Replacing ``NativeIsingSolver`` with a future remote backend will not change
``hardware.matrix``, its variable order, or the local decode metadata.

See :doc:`optimization_applications` for reusable model families and complete
portfolio-selection and wireless-channel-assignment examples.
