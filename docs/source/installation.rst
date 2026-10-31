Installation
============

Install from PyPI
-----------------

.. code-block:: bash

   python -m pip install quspect

QuSpect is published on PyPI as pre-built binary wheels that already contain
the C++ simulation engine, compiled with OpenMP.  ``Simulator``,
``NativeExecutor`` and exact adjoint gradients therefore work immediately after
``pip install``; no compiler is required.

Supported platforms
-------------------

.. list-table::
   :header-rows: 1

   * - Platform
     - Architecture
     - Python
     - Wheel tag
   * - Linux with glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL/Rocky 8+, ...)
     - x86_64
     - 3.10 -- 3.15
     - ``manylinux_2_28_x86_64``
   * - Linux with glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL/Rocky 8+, ...)
     - aarch64 (arm64)
     - 3.10 -- 3.15
     - ``manylinux_2_28_aarch64``
   * - Windows 10/11
     - x86_64 (64-bit Python)
     - 3.10 -- 3.15
     - ``win_amd64``
   * - macOS 14 (Sonoma) or newer
     - Apple Silicon (arm64)
     - 3.10 -- 3.15
     - ``macosx_14_0_arm64``

On Windows, 64-bit CPython 3.10 -- 3.15 needs nothing extra: the MSVC C++
runtime (``msvcp140.dll``) and the OpenMP runtime (``vcomp140.dll``) are
bundled in the wheel.

Installing from source
----------------------

On any other platform -- for example musl-based distributions such as Alpine,
32-bit Windows, Intel Macs, macOS older than 14, or PyPy -- ``pip`` falls back to the source distribution and compiles the C++ extension
locally.  This requires:

* CMake 3.22 or newer and a C++17 compiler: Visual Studio Build Tools on
  Windows, the Xcode Command Line Tools on macOS (plus ``brew install libomp``
  for OpenMP support), or GCC / Clang on Linux;
* network access during the build, to fetch the header-only nlohmann/json
  library.

Runtime configuration
---------------------

``QUSPECT_DISABLE_AVX2``
   On x86_64 the wheel ships two extension modules: a portable scalar build
   and an AVX2 + FMA build.  The AVX2 module is loaded only when the CPU and
   operating system support AVX2 and FMA.  Set ``QUSPECT_DISABLE_AVX2=1`` to
   force the scalar kernels.  The variable is read once, the first time the
   native backend is loaded.

``OMP_NUM_THREADS``
   The parallel kernels use OpenMP.  Set ``OMP_NUM_THREADS`` to limit the
   number of threads.

Check which backend is active:

.. code-block:: bash

   python -c "from quspect.native import capabilities; print(capabilities())"

Detecting the native core
-------------------------

Use :func:`quspect.native.load_core` to check whether the C++ native core is
available.  It is the supported, stable public way to detect the native core
and never raises when the extension is simply missing: it returns ``None``
when the native core has not been built, and the loaded extension module
otherwise.  Treat the returned module only as an availability signal; its
name (``quspect._core_scalar`` or ``quspect._core_avx2``) is an implementation
detail.

.. code-block:: python

   from quspect.native import load_core

   if load_core() is None:
       print("QuSpect native core not built; native features are unavailable")
   else:
       print("QuSpect native core is available")

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/QuSpect/quspect.git
   cd quspect
   python -m pip install -e ".[dev]"
   python -m pip install pybind11

The editable install is pure Python and does not compile the C++ extension.
Build it into the local ``quspect/`` package with CMake:

.. code-block:: bash

   cmake -S cpp -B cpp/build -G Ninja \
     -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
   cmake --build cpp/build
   ctest --test-dir cpp/build --output-on-failure

Pass ``-DQSPECT_BUILD_AVX2=OFF`` for a scalar-only build.
