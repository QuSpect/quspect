Reproducible compilation
========================

Since 0.6.0, :func:`quspect.transpiler.transpile` gives **bit-identical**
results on every supported platform: for the same input circuit, target,
options and integer ``seed``, the output gates (names, qubits and every bit of
every parameter) and the initial and final layouts are the same on macOS
arm64, Linux x86_64 (both the AVX2 and the scalar extension module),
Linux aarch64 and Windows AMD64, for every supported Python version, any
number of compile threads (``QUSPECT_COMPILE_THREADS``) and any
``PYTHONHASHSEED``.

A circuit compiled on a laptop can therefore be reproduced exactly on a
cluster node or in CI, and compilation results can be compared or cached by
their bytes.

How it works
------------

Everything numerical on the compilation path runs in the C++ extension on a
small deterministic numerics layer instead of numpy, BLAS/LAPACK or the
platform math library:

- **Fixed floating-point rules.**  The compiler sources are built without
  floating-point contraction (``-ffp-contract=off -fno-fast-math``,
  ``/fp:precise``); fused multiply-adds appear only where they are written
  explicitly (``std::fma`` is correctly rounded everywhere).  Every sum and
  product is evaluated in a fixed order, so the AVX2 and the scalar modules
  execute the same operations.
- **Own small-matrix linear algebra.**  The compiler only needs 2×2, 4×4 and
  8×8 complex matrices: products, determinants, Jacobi eigendecompositions
  (real symmetric and Hermitian), one-sided Jacobi SVD, Householder QR and
  the cosine–sine decomposition used by the three-qubit resynthesis are
  implemented with fixed sequences of IEEE operations.
- **Correctly rounded math functions.**  ``sin``, ``cos``, ``tan``,
  ``asin``, ``acos``, ``atan``, ``atan2``, ``exp``, ``log`` and ``pow`` come
  from the `CORE-MATH <https://core-math.gitlabpages.inria.fr/>`_ project
  (correct rounding leaves exactly one admissible result); ``hypot`` and
  ``atan2`` at extreme ratios use short own implementations built only from
  IEEE operations, ``fma`` and ``sqrt``.  This covers gate matrices, the
  single- and two-qubit synthesis, block consolidation, gauge and CX
  orientation, the three-qubit resynthesis, routing heuristics and the
  numeric functions of the OpenQASM 2 parser.
- **Deterministic search.**  Routing scores are computed exactly and ties
  are broken by a C++ reimplementation of CPython's Mersenne Twister seeded
  from ``seed``; parallel work (routing trials, window searches, scheduling,
  the gauge search) is written by index and merged in a fixed order, so the
  thread count only changes the running time; no result depends on the
  iteration order of a hashed container.

:func:`~quspect.synthesis.two_qubit.kak_decompose`,
:func:`~quspect.synthesis.two_qubit.weyl_coordinates` and
:meth:`~quspect.synthesis.two_qubit.Entangler.from_matrix` use the same
deterministic KAK decomposition, and so does the approximate two-qubit
synthesis (``approximation_degree``; its default budget of ``1e-12`` is
always active).

Verification
------------

The test suite stores SHA-256 digests of complete compilations
(``tests/transpiler/test_golden_transpile.py``: generated circuits on line,
ring, grid and fully connected targets with three gate sets at levels 1–3)
and of the synthesis kernels on platform-independent inputs
(``tests/synthesis/test_native_kak.py``).  CI checks them on every platform
and with both x86_64 modules, including on an emulated CPU without AVX2; the
same tests also compile with different thread counts and hash seeds.
``tools/det_fingerprint.py`` produces full dumps (``float.hex`` of every
parameter) for larger cross-machine comparisons.

Scope and exceptions
--------------------

The guarantee applies to the same QuSpect version with its compiled
extension module (always present in the published wheels).  It does **not**
cover:

- the pure-Python fallback, used when the extension module is missing or a
  backend is forced to Python (``QUSPECT_SYNTHESIS_BACKEND``,
  ``QUSPECT_ROUTING_BACKEND``, ``QUSPECT_GAUGE_BACKEND``,
  ``QUSPECT_DAG_PASS_BACKEND`` set to ``python``): it uses numpy and the
  platform math library and gives equivalent, but not bit-identical, results;
- ``seed=None`` (fresh operating-system entropy) and wall-clock limits
  (``time_limit``): the result then depends on the run or on the machine
  speed;
- the inputs themselves: parameters computed in Python with :mod:`math` or
  numpy may differ in the last bit between platforms; the OpenQASM 2 parser
  evaluates parameter expressions with the correctly rounded functions;
- numerical utilities outside compilation (for example
  :func:`~quspect.synthesis.two_qubit.approximation_error`, operators and
  the simulators);
- the pulse envelopes of :mod:`quspect.pulse`: Flattop and RippleRect
  samples go through the platform ``erf`` and agree across platforms within
  one ``float32`` ulp (see :doc:`pulse`).

Compiled outputs changed once, in 0.6.0, on every platform (macOS included)
when the deterministic layer replaced the numpy-based numerics; the
compilation quality is unchanged or better.
