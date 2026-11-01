# CORE-MATH (vendored subset)

Correctly rounded binary64 math functions from the CORE-MATH project
(<https://core-math.gitlabpages.inria.fr/>), used by the deterministic numerics
layer (`cpp/include/quspect_native/detmath/`) so that every platform computes the
same (correctly rounded) result.

- Source: <https://gitlab.inria.fr/core-math/core-math>
- Commit: `0ce0cbc15935e24c656280e5a46c34e89f301b7a` (2026-10-20). Later commits rework
  `sin` / `cos` / `log` (work in progress at the time of import; the head commit
  `5f65574` did not compile), so this earlier commit was taken.
- License: MIT (see `LICENSE`, copied verbatim from the repository; every source
  file carries the same notice).
- Files: `src/binary64/{sin,cos,tan,asin,acos,atan,atan2,exp,log,hypot,pow}/<name>.c`,
  plus the headers they include (`atan2/tint.h`, `log/dint.h`, `pow/{pow,dint,qint}.h`). Directory layout as
  upstream.
- Local modifications (marked `quspect local modification` in the sources): the
  "unexpected worst-case" branches of `atan2/tint.h` (`tint_tod`) and `pow/pow.c` print the
  inputs and call `exit(1)`, which would kill the host process. They now fall through and
  round the high-precision approximation (deterministic; possibly not correctly rounded in such
  a case). The atan2 branch was reached at `y, x = 0x1.7fffffffffffdp-53, 0x1.ffffffffffffap-3`
  while transpiling `qiskit_hrs_multiplier_13` (the fall-through gives the correctly rounded
  `0x1.8000000000002p-51`, checked with mpmath at 2000 bits); not reported upstream yet.

  Review of the atan2 change: the exact value at that input lies only about 2^-152.6 ulp above a
  rounding midpoint (y/x is just above a midpoint and the cubic term of the arctangent almost
  cancels the excess), so the 192-bit approximation cannot prove the rounding direction; it still
  rounds the right way. A search of 32,301 inputs (the input and all of its +-8-ulp neighbours, scaled
  by several powers of two, and about 30,000 structured inputs of the same "3/4 * (1 + tiny)" form,
  with both signs) against the unmodified upstream code found only that ratio (at six scalings)
  reaching the branch, and the modified code returns the correctly rounded value there (mpmath, 6000
  bits); a random sample of the other inputs agrees with mpmath as well. `test_detmath`
  (`CoreMathAtan2WorstCaseIsCorrectlyRounded`) calls `cr_atan2` directly on the input, its +-1-ulp
  neighbours and scaled copies. The detmath `Atan2` wrapper does not reach this code for such inputs
  (it uses a short series for |y/x| <= 2^-30). No input reaching the `pow` branch is known; there the
  256-bit approximation is rounded, which is deterministic but not proven correctly rounded.

Build notes (see `cpp/CMakeLists.txt`):

- compiled as C11 with GNU extensions, `-ffp-contract=off -fno-fast-math`;
- the sources use 128-bit integers (`unsigned __int128` / `_BitInt(128)`) and GCC
  builtins, which MSVC's C compiler lacks: with the MSVC toolchain (Windows) CMake compiles
  them with clang-cl (found on `PATH`, in the VS "C++ Clang tools" or in `C:/Program
  Files/LLVM/bin`) into object files that are linked with the MSVC-built C++ code (same
  ABI; CMake itself cannot mix clang-cl C with MSVC C++). The objects only reference UCRT
  functions (`fma`, `fmin`, `fmax`, `sqrt`, `round`, `fe*`, `printf` / `exit`); `std::fma` must be
  correctly rounded there, which `test_detmath` (`FmaIsCorrectlyRounded`) checks;
- `CORE_MATH_SUPPORT_ERRNO` is not defined (errno is not set).

To update: clone the repository into an empty directory, review the diff of these
files, copy them over and record the new commit here.
