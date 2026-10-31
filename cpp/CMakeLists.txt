cmake_minimum_required(VERSION 3.22)
project(quspect_native VERSION 0.1.0 LANGUAGES CXX)

# ---- CTest ----
include(CTest)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Build options ----
option(QSPECT_BUILD_PYTHON "Build pybind11 Python module" ON)
option(QSPECT_BUILD_TESTS "Build gtest test suite" ON)
option(QSPECT_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(QSPECT_REQUIRE_OPENMP "Fail configuration when OpenMP is enabled but not found" OFF)
option(QSPECT_BUILD_AVX2 "Build AVX2 optimized kernels" ON)

# AVX2 内核只能在 x86_64 上构建。其他架构（如 arm64）即使 QSPECT_BUILD_AVX2=ON 也自动关闭。
# 使用普通变量 QSPECT_AVX2_ACTIVE 记录最终结果，不改写用户的 cache 选项。
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" QSPECT_SYSTEM_PROCESSOR_LOWER)
set(QSPECT_AVX2_ACTIVE OFF)
if(QSPECT_BUILD_AVX2)
  if(QSPECT_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64)$")
    set(QSPECT_AVX2_ACTIVE ON)
  else()
    message(STATUS "QSPECT_BUILD_AVX2 is ON but CMAKE_SYSTEM_PROCESSOR is "
                   "'${CMAKE_SYSTEM_PROCESSOR}' (not x86_64/AMD64): AVX2 kernels disabled")
  endif()
endif()
message(STATUS "QuSpect AVX2 kernels: ${QSPECT_AVX2_ACTIVE}")

# AVX2 内核的编译选项（内核依赖 FMA，CPU 检测也同时要求 AVX2 + FMA）
if(MSVC)
  set(QSPECT_AVX2_COMPILE_OPTIONS /arch:AVX2)
else()
  set(QSPECT_AVX2_COMPILE_OPTIONS -mavx2 -mfma)
endif()

# ---- Find Python for pybind11 ----
if(QSPECT_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  # Use the pybind11 installed via pip — detect dynamically
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(pybind11 CONFIG REQUIRED)
endif()

# ---- gtest via FetchContent ----
if(QSPECT_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

# ---- nlohmann/json via FetchContent (Phase 5 QSAL decoder) ----
# 使用官方 release 打包的源码 tarball（约 110 KB，带 SHA256 校验），避免 git clone 整个仓库。
# 离线构建可通过 -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=<本地源码目录> 跳过下载。
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # 解压文件使用当前时间戳，保证源码更新后能触发重新构建
endif()
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# ---- OpenMP ----
#
# QSPECT_ENABLE_OPENMP=OFF      不查找 OpenMP，构建单线程版本
# QSPECT_REQUIRE_OPENMP=ON      找不到 OpenMP 时配置失败（发布 wheel 时使用）
#
# 各平台：
#   Linux (GCC)        -fopenmp + libgomp，FindOpenMP 直接可用
#   Windows (MSVC)     /openmp（只支持 OpenMP 2.0）。显式要求默认运行时，不使用 /openmp:llvm
#   macOS (AppleClang) 编译器不自带 OpenMP 运行时，需要 libomp：
#                      -DOpenMP_ROOT=$(brew --prefix libomp)（或 Homebrew llvm 前缀）
#
# 编译期宏 QSPECT_HAS_OPENMP=1 只在真正链接了 OpenMP 的目标上定义，代码据此 #include <omp.h>。
set(QSPECT_OPENMP_ACTIVE OFF)
if(QSPECT_ENABLE_OPENMP)
  if(MSVC AND NOT DEFINED OpenMP_RUNTIME_MSVC)
    set(OpenMP_RUNTIME_MSVC "default")  # CMake >= 3.30：选择 /openmp 而不是 /openmp:llvm
  endif()

  if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND NOT OpenMP_CXX_FLAGS)
    # AppleClang 不认识 -fopenmp，FindOpenMP 也找不到 Homebrew 的 libomp 头文件。
    # 根据 OpenMP_ROOT（CMake 变量或环境变量）预先给出 FindOpenMP 需要的结果变量。
    set(_qspect_omp_root "${OpenMP_ROOT}")
    if(NOT _qspect_omp_root AND DEFINED ENV{OpenMP_ROOT})
      set(_qspect_omp_root "$ENV{OpenMP_ROOT}")
    endif()
    if(_qspect_omp_root)
      # Homebrew libomp: <root>/include/omp.h；Homebrew llvm: <root>/lib/clang/<ver>/include/omp.h
      file(GLOB _qspect_omp_clang_includes LIST_DIRECTORIES TRUE "${_qspect_omp_root}/lib/clang/*/include")
      find_path(QSPECT_LIBOMP_INCLUDE_DIR omp.h
        HINTS "${_qspect_omp_root}/include" ${_qspect_omp_clang_includes}
        NO_DEFAULT_PATH)
      find_library(QSPECT_LIBOMP_LIBRARY NAMES omp
        HINTS "${_qspect_omp_root}/lib"
        NO_DEFAULT_PATH)
      if(QSPECT_LIBOMP_INCLUDE_DIR AND QSPECT_LIBOMP_LIBRARY)
        set(_qspect_omp_include "${QSPECT_LIBOMP_INCLUDE_DIR}")
        if(_qspect_omp_include MATCHES "/lib/clang/[^/]+/include$")
          # clang 资源目录里还有 stddef.h 等编译器内建头文件，直接 -I 会遮蔽 AppleClang/libc++ 的版本。
          # 只把 omp.h 复制到构建目录中单独使用。
          set(_qspect_omp_include "${CMAKE_BINARY_DIR}/qspect_libomp_include")
          file(COPY "${QSPECT_LIBOMP_INCLUDE_DIR}/omp.h" DESTINATION "${_qspect_omp_include}")
        endif()
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_qspect_omp_include}"
            CACHE STRING "CXX compiler flags for OpenMP parallelization" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "CXX compiler libraries for OpenMP parallelization" FORCE)
        set(OpenMP_omp_LIBRARY "${QSPECT_LIBOMP_LIBRARY}" CACHE FILEPATH "libomp location" FORCE)
      else()
        message(STATUS "libomp not found under OpenMP_ROOT=${_qspect_omp_root}")
      endif()
    endif()
  endif()

  find_package(OpenMP COMPONENTS CXX)

  if(MSVC AND NOT OpenMP_CXX_FOUND)
    # FindOpenMP 在 MSVC 上失败时（原因未知，CI 日志中只有 "missing: OpenMP_CXX_FLAGS"），
    # 先尽力打印 FindOpenMP 的 try_compile 诊断，再自行用 /openmp 直接探测一次。
    set(_qspect_cfg_log "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml")
    if(DEFINED CMAKE_CONFIGURE_LOG AND EXISTS "${CMAKE_CONFIGURE_LOG}")
      set(_qspect_cfg_log "${CMAKE_CONFIGURE_LOG}")
    endif()
    if(EXISTS "${_qspect_cfg_log}")
      file(STRINGS "${_qspect_cfg_log}" _qspect_cfg_lines
        REGEX "OpenMP|omp|[Ee]rror|fatal|warning [CD][0-9]+"
        LIMIT_COUNT 80)
      list(JOIN _qspect_cfg_lines "\n" _qspect_cfg_text)
      message(STATUS "FindOpenMP diagnostics from ${_qspect_cfg_log}:\n${_qspect_cfg_text}")
    endif()

    set(_qspect_omp_probe_dir "${CMAKE_BINARY_DIR}/qspect_openmp_probe")
    set(_qspect_omp_probe_src "${_qspect_omp_probe_dir}/omp_probe.cpp")
    file(WRITE "${_qspect_omp_probe_src}" [=[
#include <omp.h>
int main() {
  int n = 0;
#pragma omp parallel
  {
#pragma omp atomic
    ++n;
  }
  return (n > 0 && omp_get_max_threads() > 0) ? 0 : 1;
}
]=])
    try_compile(_qspect_omp_probe_ok
      "${_qspect_omp_probe_dir}/build"
      "${_qspect_omp_probe_src}"
      COMPILE_DEFINITIONS /openmp
      OUTPUT_VARIABLE _qspect_omp_probe_out)
    if(_qspect_omp_probe_ok)
      # /openmp 会在目标文件中写入 vcomp 的 defaultlib，无需显式链接库
      if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
          INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:/openmp>")
      endif()
      set(OpenMP_CXX_FOUND TRUE)
      set(OpenMP_CXX_FLAGS "/openmp")
      message(STATUS "FindOpenMP failed, but a direct /openmp probe succeeded: using OpenMP::OpenMP_CXX with /openmp")
    else()
      message(STATUS "Direct /openmp probe failed. Compiler output:\n${_qspect_omp_probe_out}")
    endif()
  endif()

  if(OpenMP_CXX_FOUND)
    set(QSPECT_OPENMP_ACTIVE ON)
    if(MSVC AND OpenMP_CXX_FLAGS MATCHES "llvm|experimental")
      message(FATAL_ERROR "QuSpect targets the default MSVC OpenMP runtime (/openmp), got: ${OpenMP_CXX_FLAGS}")
    endif()
  elseif(QSPECT_REQUIRE_OPENMP)
    message(FATAL_ERROR "QSPECT_REQUIRE_OPENMP is ON but OpenMP was not found. "
                        "On macOS install libomp and pass -DOpenMP_ROOT=$(brew --prefix libomp).")
  else()
    message(STATUS "OpenMP not found: building single-threaded kernels")
  endif()
elseif(QSPECT_REQUIRE_OPENMP)
  message(FATAL_ERROR "QSPECT_REQUIRE_OPENMP=ON conflicts with QSPECT_ENABLE_OPENMP=OFF")
endif()
message(STATUS "QuSpect OpenMP: ${QSPECT_OPENMP_ACTIVE} ${OpenMP_CXX_FLAGS}")

# 给目标链接 OpenMP 并定义 QSPECT_HAS_OPENMP（未启用 OpenMP 时为空操作）
function(qspect_use_openmp target scope)
  if(QSPECT_OPENMP_ACTIVE)
    target_link_libraries(${target} ${scope} OpenMP::OpenMP_CXX)
    target_compile_definitions(${target} ${scope} QSPECT_HAS_OPENMP=1)
  endif()
endfunction()

# ---- Include directories ----
set(NATIVE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================
# Source file lists
# ============================================================

# Shared helpers (OpenMP wrappers)
set(COMMON_SRCS
  src/common/parallel.cpp
)

# Scalar kernels
set(SCALAR_SRCS
  src/kernels/scalar/state_init.cpp
  src/kernels/scalar/single_qubit_gate.cpp
  src/kernels/scalar/two_qubit_gate.cpp
  src/kernels/scalar/dense_gate.cpp
  src/kernels/scalar/diagonal_gate.cpp
  src/kernels/scalar/expectation.cpp
  src/kernels/scalar/sampling.cpp
  src/kernels/scalar/collapse.cpp
  src/kernels/scalar/noise_kernels.cpp
)

# AVX2 kernels (placeholder for Phase 6)
set(AVX2_SRCS
  src/kernels/avx2/single_qubit_gate.cpp
  src/kernels/avx2/two_qubit_gate.cpp
  src/kernels/avx2/dense_gate.cpp
  src/kernels/avx2/expectation.cpp
)

# Dispatch layer
set(DISPATCH_SRCS
  src/kernels/dispatch/dispatch.cpp
)

# Runtime layer
set(RUNTIME_SRCS
  src/runtime/state_vector.cpp
  src/runtime/classical_register.cpp
  src/runtime/runtime_engine.cpp
  src/runtime/gate_operation.cpp
  src/runtime/observable.cpp
  src/runtime/program.cpp
)

# IR bridge layer
set(IR_SRCS
  src/ir/qsal_decoder.cpp
  src/ir/program_builder.cpp
  src/ir/validation.cpp
)

# Executor layer
set(EXECUTOR_SRCS
  src/executor/simulation_executor.cpp
  src/executor/execution_result.cpp
  src/executor/gradient_executor.cpp
)

# Dense Ising optimization
set(OPTIMIZATION_SRCS
  src/optimization/ising_solver.cpp
)

# Transpiler kernels (SABRE routing core, CPython-compatible MT19937)
set(TRANSPILER_SRCS
  src/transpiler/cpython_random.cpp
  src/transpiler/cx_orientation.cpp
  src/transpiler/depth_schedule.cpp
  src/transpiler/dilation_layout.cpp
  src/transpiler/layer_routing.cpp
  src/transpiler/one_qubit_gauge.cpp
  src/transpiler/ring_routing.cpp
  src/transpiler/route_lns.cpp
  src/transpiler/sabre_routing.cpp
  src/transpiler/su4_routing.cpp
)
# 路由计分必须与 Python 参考实现逐位一致：禁止把 a + b * c 合并成 FMA（arm64 上 clang 默认会合并）。
if(NOT MSVC)
  set_source_files_properties(${TRANSPILER_SRCS} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
else()
  set_source_files_properties(${TRANSPILER_SRCS} PROPERTIES COMPILE_OPTIONS "/fp:precise")
endif()

# pybind11 module
set(PYBIND_SRCS
  src/pybind/module.cpp
)

# ============================================================
# Library targets
# ============================================================
#
# 目标关系：
#
#   qspect_core_obj (OBJECT)          scalar 内核 + runtime + executor + ir + optimization，
#                                     与 dispatch 无关，只编译一次
#   qspect_kernels_avx2_obj (OBJECT)  AVX2 内核（仅 QSPECT_AVX2_ACTIVE）
#   qspect_dispatch_scalar (OBJECT)   dispatch.cpp，不定义 QSPECT_USE_AVX2
#   qspect_dispatch_avx2 (OBJECT)     dispatch.cpp，定义 QSPECT_USE_AVX2（仅 QSPECT_AVX2_ACTIVE）
#
#   qspect_native_scalar_static = core + dispatch_scalar             （不含任何 AVX2 目标文件）
#   qspect_native_avx2_static   = core + dispatch_avx2 + kernels_avx2
#
# runtime 调用 dispatch 中的 GetDispatchF32/F64，而 dispatch 又调用 scalar 内核。
# 每个变体的所有目标文件都放进同一个静态库，链接器在单个归档内会反复解析符号，
# 因此不存在静态库之间的链接顺序 / 循环依赖问题。

set(QSPECT_CORE_SRCS
  ${COMMON_SRCS}
  ${SCALAR_SRCS}
  ${RUNTIME_SRCS}
  ${EXECUTOR_SRCS}
  ${IR_SRCS}
  ${OPTIMIZATION_SRCS}
  ${TRANSPILER_SRCS}
)

# ── 与 dispatch 无关的公共部分 ──
add_library(qspect_core_obj OBJECT ${QSPECT_CORE_SRCS})
target_include_directories(qspect_core_obj PUBLIC ${NATIVE_INCLUDE_DIR})
target_link_libraries(qspect_core_obj PRIVATE nlohmann_json::nlohmann_json)
qspect_use_openmp(qspect_core_obj PUBLIC)

# ── dispatch: scalar 变体 ──
add_library(qspect_dispatch_scalar OBJECT ${DISPATCH_SRCS})
target_include_directories(qspect_dispatch_scalar PUBLIC ${NATIVE_INCLUDE_DIR})

# ── 静态库：scalar 变体 ──
add_library(qspect_native_scalar_static STATIC
  $<TARGET_OBJECTS:qspect_core_obj>
  $<TARGET_OBJECTS:qspect_dispatch_scalar>
)
target_include_directories(qspect_native_scalar_static PUBLIC ${NATIVE_INCLUDE_DIR})
qspect_use_openmp(qspect_native_scalar_static PUBLIC)

if(QSPECT_AVX2_ACTIVE)
  # ── kernels_avx2 ──
  add_library(qspect_kernels_avx2_obj OBJECT ${AVX2_SRCS})
  target_include_directories(qspect_kernels_avx2_obj PUBLIC ${NATIVE_INCLUDE_DIR})
  target_compile_options(qspect_kernels_avx2_obj PRIVATE ${QSPECT_AVX2_COMPILE_OPTIONS})
  target_compile_definitions(qspect_kernels_avx2_obj PRIVATE QSPECT_USE_AVX2=1)
  qspect_use_openmp(qspect_kernels_avx2_obj PUBLIC)

  # ── dispatch: AVX2 变体（dispatch.cpp 本身不加 AVX2 编译选项，CPU 检测必须能在老 CPU 上运行）──
  add_library(qspect_dispatch_avx2 OBJECT ${DISPATCH_SRCS})
  target_include_directories(qspect_dispatch_avx2 PUBLIC ${NATIVE_INCLUDE_DIR})
  target_compile_definitions(qspect_dispatch_avx2 PRIVATE QSPECT_USE_AVX2=1)

  # ── 静态库：AVX2 变体 ──
  add_library(qspect_native_avx2_static STATIC
    $<TARGET_OBJECTS:qspect_core_obj>
    $<TARGET_OBJECTS:qspect_dispatch_avx2>
    $<TARGET_OBJECTS:qspect_kernels_avx2_obj>
  )
  target_include_directories(qspect_native_avx2_static PUBLIC ${NATIVE_INCLUDE_DIR})
  qspect_use_openmp(qspect_native_avx2_static PUBLIC)
endif()

# ============================================================
# Pulse envelopes (quspect._pulse)
# ============================================================
#
# 超导量子比特控制包络（envelopes-qc 0.1.5 的 C++ 重新实现）。解码结果与原实现逐位一致，
# 因此必须关闭浮点收缩（FMA 融合）；libcerf 3.1（MIT）以 C++ 方式编译。不使用 OpenMP。
set(PULSE_SRCS
  src/pulse/envelopes.cpp
  src/pulse/cerf_erfcx.cpp
  src/pulse/cerf_err_fcts.cpp
  src/pulse/cerf_im_w_of_x.cpp
  src/pulse/cerf_w_of_z.cpp
  src/pulse/cerf_width.cpp
)
add_library(qspect_pulse_static STATIC ${PULSE_SRCS})
target_include_directories(qspect_pulse_static
  PUBLIC ${NATIVE_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libcerf
)
if(MSVC)
  target_compile_options(qspect_pulse_static PRIVATE /fp:precise)
else()
  target_compile_options(qspect_pulse_static PRIVATE -ffp-contract=off -fno-fast-math)
endif()

# 默认变体：测试优先链接 AVX2 变体（CPU 支持时 test_dispatch 断言 backend == "avx2"）
if(QSPECT_AVX2_ACTIVE)
  set(QSPECT_NATIVE_DEFAULT_LIB qspect_native_avx2_static)
else()
  set(QSPECT_NATIVE_DEFAULT_LIB qspect_native_scalar_static)
endif()

# ============================================================
# Python modules
# ============================================================
#
#   quspect._core_scalar  始终构建，只含 scalar 内核，可在任意 CPU 上加载
#   quspect._core_avx2    仅 QSPECT_AVX2_ACTIVE 时构建；Python 加载器确认 CPU 支持
#                         AVX2 + FMA 后才会导入
#
# 非 scikit-build 构建（本地开发）时直接把扩展写进 quspect/ 包目录；
# scikit-build 构建时由 install() 规则安装到 wheel 的 quspect/ 下。
function(qspect_add_python_module target module_name native_lib)
  pybind11_add_module(${target} MODULE ${PYBIND_SRCS})
  target_compile_definitions(${target} PRIVATE QSPECT_MODULE_NAME=${module_name})
  target_link_libraries(${target} PRIVATE ${native_lib})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${module_name})
  if(NOT SKBUILD)
    # $<1:...> 防止多配置生成器（Visual Studio）追加 Release/Debug 子目录
    set(_qspect_pkg_dir "${CMAKE_CURRENT_SOURCE_DIR}/../quspect")
    set_target_properties(${target} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "$<1:${_qspect_pkg_dir}>"
      RUNTIME_OUTPUT_DIRECTORY "$<1:${_qspect_pkg_dir}>"
    )
  endif()
  # Windows 上 .pyd 是 MODULE 库（LIBRARY 产物），这里同时声明 RUNTIME 以防万一
  install(TARGETS ${target}
    LIBRARY DESTINATION quspect
    RUNTIME DESTINATION quspect
  )
endfunction()

if(QSPECT_BUILD_PYTHON)
  qspect_add_python_module(qspect_core_scalar_module _core_scalar qspect_native_scalar_static)
  if(QSPECT_AVX2_ACTIVE)
    qspect_add_python_module(qspect_core_avx2_module _core_avx2 qspect_native_avx2_static)
  endif()

  # quspect._pulse：独立于内核变体的包络模块
  pybind11_add_module(qspect_pulse_module MODULE src/pybind/pulse_module.cpp)
  target_link_libraries(qspect_pulse_module PRIVATE qspect_pulse_static)
  set_target_properties(qspect_pulse_module PROPERTIES OUTPUT_NAME _pulse)
  if(NOT SKBUILD)
    set(_qspect_pkg_dir "${CMAKE_CURRENT_SOURCE_DIR}/../quspect")
    set_target_properties(qspect_pulse_module PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "$<1:${_qspect_pkg_dir}>"
      RUNTIME_OUTPUT_DIRECTORY "$<1:${_qspect_pkg_dir}>"
    )
  endif()
  install(TARGETS qspect_pulse_module
    LIBRARY DESTINATION quspect
    RUNTIME DESTINATION quspect
  )
endif()

# ============================================================
# Tests
# ============================================================
if(QSPECT_BUILD_TESTS)
  # Test sources
  set(TEST_SRCS
    tests/test_dispatch.cpp
    tests/test_state_vector.cpp
    tests/test_single_qubit_gate.cpp
    tests/test_two_qubit_gate.cpp
    tests/test_dense_gate.cpp
    tests/test_expectation.cpp
    tests/test_expectation_multi_flip.cpp
    tests/test_sampling.cpp
    tests/test_classical_register.cpp
    tests/test_executor.cpp
    tests/test_gradient.cpp
    tests/test_circuit_execution.cpp
    tests/test_measurement.cpp
    tests/test_qsal_bridge.cpp
    tests/test_noise.cpp
    tests/test_ising_solver.cpp
    tests/test_bit_ops.cpp
    tests/test_parallel.cpp
    tests/test_cpython_random.cpp
    tests/test_sabre_routing.cpp
    tests/test_route_lns.cpp
    tests/test_layer_routing.cpp
    tests/test_ring_routing.cpp
    tests/test_su4_routing.cpp
    tests/test_depth_schedule.cpp
    tests/test_dilation_layout.cpp
    tests/test_one_qubit_gauge.cpp
    tests/test_cx_orientation.cpp
  )

  foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name}
      PRIVATE ${NATIVE_INCLUDE_DIR}
    )
    target_link_libraries(${test_name}
      PRIVATE gtest_main
      PRIVATE ${QSPECT_NATIVE_DEFAULT_LIB}
    )
    # Register with CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()

  # scalar dispatch 变体：无论 CPU 是否支持 AVX2，backend 都必须是 "scalar"
  add_executable(test_dispatch_scalar tests/test_dispatch_scalar.cpp)
  target_link_libraries(test_dispatch_scalar
    PRIVATE gtest_main
    PRIVATE qspect_native_scalar_static
  )
  add_test(NAME test_dispatch_scalar COMMAND test_dispatch_scalar)

  add_executable(test_pulse_envelopes tests/test_pulse_envelopes.cpp)
  target_link_libraries(test_pulse_envelopes
    PRIVATE gtest_main
    PRIVATE qspect_pulse_static
  )
  add_test(NAME test_pulse_envelopes COMMAND test_pulse_envelopes)

  # AVX2 构建中默认测试链接 AVX2 变体；并行测试再对 scalar 变体跑一遍
  if(QSPECT_AVX2_ACTIVE)
    add_executable(test_parallel_scalar tests/test_parallel.cpp)
    target_link_libraries(test_parallel_scalar
      PRIVATE gtest_main
      PRIVATE qspect_native_scalar_static
    )
    add_test(NAME test_parallel_scalar COMMAND test_parallel_scalar)
  endif()
endif()
