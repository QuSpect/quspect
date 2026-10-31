#!/usr/bin/env bash
# 从 LLVM 官方源码构建 macOS arm64 的 libomp（OpenMP 运行时），并以指定的
# 最低系统版本（deployment target）编译，安装到 --prefix。
#
# 为什么不用 Homebrew 的 libomp：delocate 把 libomp.dylib 打包进 wheel 时，会把 wheel
# 的平台标签提升到所有被打包 dylib 中最大的 minos。Homebrew 的 bottle 按 runner 的
# 系统版本构建（macos-15 上 minos 为 15.0），wheel 会因此变成 macosx_15_0_arm64，
# macOS 14 用户无法安装。这里自己构建，保证 minos 等于 deployment target。
#
# 用法（wheels.yml / pyproject.toml 中 cibuildwheel 的 macOS before-all 调用）：
#   bash tools/build_libomp_macos.sh --prefix <dir> --deployment-target 14.0
#
# - 源码：LLVM 发布页的 openmp-<ver>.src.tar.xz 与 cmake-<ver>.src.tar.xz（独立构建
#   openmp 需要同级的 cmake/ 模块目录），下载后校验 sha256。LLVM 21.1.8 是最后一个
#   提供按子项目拆分源码包的版本（之后只有完整的 llvm-project 源码包）。
# - 安装名（install name）设为 <prefix>/lib/libomp.dylib 的绝对路径（与 Homebrew 相同），
#   delocate 可以直接找到并打包。
# - <prefix>/.quspect-libomp-stamp 记录版本与 deployment target；与本次参数一致且
#   dylib 存在时直接复用（配合 actions/cache 缓存 prefix），否则清空 prefix 重新构建。
# - 安装后检查 libomp.dylib 的 minos 不高于 deployment target，否则失败。
set -euo pipefail

LIBOMP_VERSION="21.1.8"
OPENMP_SHA256="856b023748b41ac7b2c83fd8e9f765ff48a4df2fe6777d2811ef7c7ed8f2f977"
CMAKE_MODULES_SHA256="85735f20fd8c81ecb0a09abb0c267018475420e93b65050cc5b7634eab744de9"
RELEASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LIBOMP_VERSION}"

prefix=""
deployment_target=""

usage() {
  echo "usage: $0 --prefix <install-dir> --deployment-target <macos-version>" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)
      [ $# -ge 2 ] || usage
      prefix="$2"
      shift 2
      ;;
    --deployment-target)
      [ $# -ge 2 ] || usage
      deployment_target="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[ -n "$prefix" ] && [ -n "$deployment_target" ] || usage

if [ "$(uname -s)" != "Darwin" ]; then
  echo "error: this script only runs on macOS" >&2
  exit 1
fi

stamp="libomp ${LIBOMP_VERSION} arm64 macos-${deployment_target}"
stamp_file="${prefix}/.quspect-libomp-stamp"
dylib="${prefix}/lib/libomp.dylib"

# 打印 dylib 的 minos（LC_BUILD_VERSION），并检查不高于 deployment target。
check_minos() {
  local minos
  minos=$(vtool -show-build "$1" | awk '$1 == "minos" { print $2; exit }')
  if [ -z "$minos" ]; then
    echo "error: cannot read minos of $1" >&2
    return 1
  fi
  echo "$1: minos ${minos} (deployment target ${deployment_target})"
  if [ "$(printf '%s\n%s\n' "$minos" "$deployment_target" | sort -V | tail -n 1)" != "$deployment_target" ]; then
    echo "error: $1 requires macOS ${minos}, newer than ${deployment_target}" >&2
    return 1
  fi
}

if [ -f "$stamp_file" ] && [ -f "$dylib" ] && [ "$(cat "$stamp_file")" = "$stamp" ]; then
  echo "Reusing cached ${stamp} in ${prefix}"
  check_minos "$dylib"
  exit 0
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fetch() {
  local name="$1" sha256="$2"
  echo "Downloading ${name}"
  curl -fsSL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 300 \
    -o "${work}/${name}" "${RELEASE_URL}/${name}"
  echo "${sha256}  ${work}/${name}" | shasum -a 256 -c -
}

fetch "openmp-${LIBOMP_VERSION}.src.tar.xz" "$OPENMP_SHA256"
fetch "cmake-${LIBOMP_VERSION}.src.tar.xz" "$CMAKE_MODULES_SHA256"

mkdir -p "${work}/src"
tar -xJf "${work}/openmp-${LIBOMP_VERSION}.src.tar.xz" -C "${work}/src"
tar -xJf "${work}/cmake-${LIBOMP_VERSION}.src.tar.xz" -C "${work}/src"
# openmp 的 CMakeLists.txt 通过 ../cmake/Modules 引用 LLVM 公共 CMake 模块。
mv "${work}/src/openmp-${LIBOMP_VERSION}.src" "${work}/src/openmp"
mv "${work}/src/cmake-${LIBOMP_VERSION}.src" "${work}/src/cmake"

generator=()
if command -v ninja >/dev/null 2>&1; then
  generator=(-G Ninja)
fi

rm -rf "$prefix"
cmake -S "${work}/src/openmp" -B "${work}/build" "${generator[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_INSTALL_NAME_DIR="${prefix}/lib" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
  -DLIBOMP_INSTALL_ALIASES=OFF \
  -DLIBOMP_OMPD_SUPPORT=OFF \
  -DLIBOMP_USE_HWLOC=OFF \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
  -DOPENMP_ENABLE_TESTING=OFF
cmake --build "${work}/build" --parallel
cmake --install "${work}/build"

if [ ! -f "$dylib" ] || [ ! -f "${prefix}/include/omp.h" ]; then
  echo "error: libomp.dylib or omp.h missing under ${prefix}" >&2
  exit 1
fi
otool -D "$dylib"
check_minos "$dylib"
printf '%s\n' "$stamp" >"$stamp_file"
echo "Installed ${stamp} to ${prefix}"
