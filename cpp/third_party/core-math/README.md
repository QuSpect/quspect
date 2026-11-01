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
  upstream. **No local modifications.**

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
