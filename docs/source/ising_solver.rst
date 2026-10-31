Native dense-Ising solver
=========================

QuSpect includes a local C++ reference backend for the exact matrix consumed
by :class:`~quspect.optimization.hardware.HardwareIsingProgram`.  It minimizes

.. math::

   E(s) = -s^\mathsf{T} M s,\qquad s_i\in\{-1,+1\},

where ``M`` is finite, symmetric, dense, and has a zero diagonal.  This keeps
the local and future remote-hardware boundaries identical.

Algorithm selection
-------------------

The first implementation uses Metropolis simulated annealing with independent
multi-start reads.  This is the most suitable baseline for an arbitrary dense
floating-point matrix:

* The Metropolis acceptance rule comes from `Metropolis et al. (1953)
  <https://doi.org/10.1063/1.1699114>`_, and its optimization interpretation
  follows `Kirkpatrick, Gelatt, and Vecchi (1983)
  <https://doi.org/10.1126/science.220.4598.671>`_.
* `Isakov et al. <https://arxiv.org/abs/1401.1084>`_ demonstrate much faster
  multi-spin-coded variants for sparse, bounded-degree models with discrete
  couplings.  Those assumptions do not hold for the dense, arbitrary-double
  hardware IR, so QuSpect uses the generic local-field method instead.
* The `D-Wave samplers reference implementation
  <https://github.com/dwavesystems/dwave-samplers>`_ likewise defines a sweep
  as one update per spin over a decreasing-temperature schedule.  It provides
  a useful independent API and algorithm reference.
* Fujitsu's Digital Annealer uses parallel-trial updates and a dynamic escape
  mechanism on specialized fully connected CMOS hardware; see `Aramon et al.
  (2019) <https://doi.org/10.3389/fphy.2019.00048>`_.  Evaluating all trials in
  parallel is attractive for a later SIMD/GPU backend, but it does not map to
  a portable CPU baseline without that hardware parallelism.
* Parallel tempering can cross barriers more effectively by exchanging
  replicas at different temperatures, as introduced by `Hukushima and Nemoto
  <https://arxiv.org/abs/cond-mat/9512035>`_.  It is a future solver option
  because it adds a temperature-ladder and exchange-policy surface beyond the
  minimal hardware-compatible API.

Incremental update
------------------

For a symmetric zero-diagonal matrix, define the cached local field

.. math::

   f_i = \sum_j M_{ij}s_j.

Flipping spin ``i`` changes the energy by

.. math::

   \Delta E_i = 4s_i f_i.

Rejected proposals therefore cost constant time.  After an accepted flip from
the old value ``s_i``, every field is updated as

.. math::

   f_j \leftarrow f_j - 2M_{ji}s_i,

which costs :math:`O(N)`.  Initial energy and fields cost :math:`O(N^2)`.
A sweep attempts ``N`` flips and costs between :math:`O(N)` and
:math:`O(N^2)` depending on the acceptance rate.  Independent reads are
parallelized with OpenMP when available.

Each read derives its own RNG seed solely from ``(base_seed, read_index)`` and
writes to a fixed result slot.  Thread scheduling therefore cannot change the
result.  All reads, including duplicates, are retained and stably sorted by
ascending energy.

Exact oracle
------------

The native test oracle enumerates small models in binary-reflected Gray-code
order.  Consecutive configurations differ by one spin, so it reuses the same
local-field update rather than recomputing every quadratic energy.  Its time
complexity is :math:`O(N2^N)` and memory complexity is :math:`O(N^2)`;
``max_spins`` defaults to 24 to prevent accidental exponential runs.

Python API
----------

.. code-block:: python

   from quspect.optimization.native_solver import NativeIsingSolver

   solver = NativeIsingSolver(
       seed=7,
       num_reads=100,
       sweeps=1000,
       beta_start=0.1,
       beta_end=5.0,
   )
   result = solver.solve(hardware_program)

   print(result.best_sample)
   print(result.best_energy)
   print(result.best_objective_value)

``result.hardware_samples`` preserves the exact matrix-column order.
``result.logical_samples`` applies gauge normalization and removes auxiliary
spins through the hardware program's decode metadata.
``result.objective_values`` restores the logical Ising objective supplied to
hardware lowering.  It therefore includes compiled penalties when the input
came from a constrained ``QuadraticProgram``; evaluate the retained
``CompilationResult`` to obtain the original business objective.
