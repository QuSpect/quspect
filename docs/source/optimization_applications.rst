Optimization application patterns
=================================

Quadratic optimization applications often look unrelated at the business
level while sharing the same small set of binary modeling patterns.  Keeping
those patterns above ``BinaryQuadraticModel`` lets business code change
without changing the solver boundary.

Reusable model families
-----------------------

The representative public examples surveyed for this integration reduce to
the following families:

.. list-table::
   :header-rows: 1
   :widths: 18 24 28 30

   * - Family
     - Typical applications
     - Binary variables
     - Objective and constraints
   * - Graph partitioning
     - Maximum cut
     - One spin or bit per vertex
     - Pairwise edge energy, usually without explicit constraints
   * - Permutation and sequencing
     - Route planning and sequence-read assembly
     - ``x[item, position]``
     - Adjacent-pair cost plus exactly-one constraints on every row and column
   * - Assignment and load balancing
     - Distributing render or compute jobs across machines
     - ``x[task, machine]``
     - Load variance or makespan proxy, with one machine per task
   * - Cardinality selection and coverage
     - Selecting a fixed number of service locations
     - Site-selection and demand-coverage bits
     - Covered demand, a fixed selection count, and coverage implications
   * - Capacitated location and network allocation
     - Cloud-edge resource placement
     - Open-site, assignment, overflow, and slack bits
     - Fixed, compute, and transport costs with assignment and capacity rules

Sequence-read assembly does not require a new optimization primitive: it
reuses a directed route-planning model with overlap scores as edge weights.
Likewise, the surveyed scheduling model is an assignment and load-balancing
model; a full job-shop model would additionally require operation order and
machine-time decisions.

The current ``QuadraticProgram`` directly supports graph-pair objectives,
permutation and assignment one-hot constraints, quadratic load objectives,
fixed-cardinality selection, and nonnegative conflict expressions.  General
inequalities, bounded-integer expansion, automatic slack construction, and
capacity helpers remain future upper-layer features.  Capacitated network
models should wait for those features instead of embedding one-off bit
encodings in business examples.

New example: portfolio selection
--------------------------------

A cardinality-constrained mean-variance portfolio uses a bit ``x_i`` to
indicate whether asset ``i`` is selected:

.. math::

   \max_x\quad \mu^T x - \lambda x^T\Sigma x
   \qquad
   \text{subject to}\quad \sum_i x_i = K.

Expected return supplies the linear reward, while the covariance matrix
creates dense floating-point quadratic couplings.  The exactly-``K``
constraint becomes a squared linear penalty during compilation.  This
example exercises maximization, dense coefficients, business-objective
recovery, and feasibility checking.

Run the complete local pipeline with:

.. code-block:: console

   python examples/portfolio_selection.py

The runnable source is available as
:download:`portfolio_selection.py <../../examples/portfolio_selection.py>`.

New example: wireless channel assignment
----------------------------------------

Let ``x[a, c]`` indicate that access point ``a`` uses channel ``c``.  Every
access point must use exactly one channel:

.. math::

   \sum_c x_{a,c} = 1.

If access points ``a`` and ``b`` interfere, they cannot use the same channel:

.. math::

   x_{a,c}x_{b,c} = 0.

The product is nonnegative over binary assignments, so it maps directly to
``add_nonnegative_quadratic_zero``.  A linear objective represents channel
preferences or operating costs.  This example exercises two-dimensional
variable order, multiple named constraints, conflict modeling, and structured
business decoding.

Run it with:

.. code-block:: console

   python examples/wireless_channel_assignment.py

The runnable source is available as
:download:`wireless_channel_assignment.py <../../examples/wireless_channel_assignment.py>`.

Placement in the architecture
-----------------------------

These models live in ``examples/`` rather than
``quspect.optimization``.  Asset names, covariance estimates, interference
graphs, and result presentation are application policy; the core package
should contain only reusable expressions, intermediate representations,
lowering, codecs, and solver adapters.  Once several applications converge
on stable input and output contracts, a separate applications package can be
introduced without changing the hardware-level matrix boundary.
