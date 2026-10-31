#!/usr/bin/env bash
# 在 manylinux_2_28 容器内，用 Intel SDE 模拟 Nehalem（-nhm，无 AVX/AVX2）
# 运行已构建的 manylinux wheel（由 PYTHON_TAG 指定 Python 版本）的冒烟测试与原生测试。
#
# 由 .github/workflows/wheels.yml 的 no-avx2-check 作业通过 docker run 调用：
#   docker run ... quay.io/pypa/manylinux_2_28_x86_64 \
#     bash "$GITHUB_WORKSPACE/tools/sde_manylinux_check.sh"
#
# 为什么在容器里跑：ubuntu-latest（26.04）的 glibc 以 x86-64-v3 为基线构建，
# 在 SDE -nhm 下 ld.so 会直接报 "CPU does not support x86-64-v3" 退出；
# manylinux_2_28（AlmaLinux 8，glibc 2.28）的用户态是基线 x86-64。
#
# 需要的环境变量（工作区与 SDE 目录需按宿主同一路径挂载进容器）：
#   GITHUB_WORKSPACE  仓库检出目录（只读即可），作为 --project-dir 传给冒烟脚本
#   SDE_PATH          setup-sde 解压出的 SDE 目录（内含 sde64）
#   PYTHON_TAG        CPython 标签，如 cp315（wheel 文件名与 /opt/python 目录据此选取）
#   WHEEL_DIR         含 quspect-*-${PYTHON_TAG}-${PYTHON_TAG}-manylinux*_x86_64.whl 的目录
# 可选：
#   PYTHON_BIN        用于创建 venv 的解释器
#                     （默认 manylinux 镜像中的 /opt/python/${PYTHON_TAG}-${PYTHON_TAG}/bin/python）

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must point to the repository checkout}"
: "${SDE_PATH:?SDE_PATH must point to the Intel SDE directory}"
: "${WHEEL_DIR:?WHEEL_DIR must point to the directory holding the wheel}"
: "${PYTHON_TAG:?PYTHON_TAG must name the CPython tag to test, e.g. cp315}"
if [[ ! "$PYTHON_TAG" =~ ^cp3[0-9]+$ ]]; then
  echo "::error::PYTHON_TAG must look like cp3XX, got '$PYTHON_TAG'"
  exit 1
fi
python_bin=${PYTHON_BIN:-/opt/python/${PYTHON_TAG}-${PYTHON_TAG}/bin/python}
if [ ! -x "$python_bin" ]; then
  echo "::error::Python interpreter $python_bin not found in this image"
  exit 1
fi
echo "python: $python_bin ($("$python_bin" -V 2>&1))"

echo "glibc: $(ldd --version 2>&1 | head -n 1)"

shopt -s nullglob
wheels=("$WHEEL_DIR"/quspect-*-"$PYTHON_TAG"-"$PYTHON_TAG"-manylinux*_x86_64.whl)
shopt -u nullglob
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "::error::expected exactly one $PYTHON_TAG manylinux x86_64 wheel in $WHEEL_DIR, found ${#wheels[@]}"
  exit 1
fi
wheel=${wheels[0]}
echo "wheel: $wheel"

sde_bin=$(find "$SDE_PATH" -maxdepth 2 -type f -name sde64 | head -n 1)
if [ -z "$sde_bin" ]; then
  echo "::error::sde64 not found under $SDE_PATH"
  exit 1
fi
export SDE_BIN=$sde_bin

work=$(mktemp -d)
venv=$work/venv
"$python_bin" -m venv "$venv"
export PATH="$venv/bin:$PATH"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "::group::Install the wheel"
python -m pip install "$wheel" pytest
python -m pip list
echo "::endgroup::"

# 测试文件复制到工作区之外的临时目录，避免从源码树导入 quspect。
test_dir=$work/no-avx2
mkdir -p "$test_dir"
cp -R "$GITHUB_WORKSPACE/tests" "$GITHUB_WORKSPACE/tools" "$test_dir/"
cd "$test_dir"

# 解析已安装的包路径（不在 SDE 下运行，也不从源码树导入）。
pkg_dir=$(python -c "import quspect, pathlib; print(pathlib.Path(quspect.__file__).resolve().parent)")
case "$pkg_dir" in
  */site-packages/quspect) ;;
  *)
    echo "::error::quspect resolved to $pkg_dir, expected an installed site-packages copy"
    exit 1
    ;;
esac
echo "quspect package: $pkg_dir"
export QUSPECT_PKG_DIR=$pkg_dir

# 在日志中留下 SDE chip-check 控制项的说明，便于核对拼写。
echo "::group::SDE chip-check knobs"
timeout 120 "$SDE_BIN" -thelp 2>&1 | grep -i chip_check \
  || timeout 120 "$SDE_BIN" -long-help 2>&1 | grep -i -A1 chip_check \
  || true
echo "::endgroup::"

# --expect-backend scalar 同时断言 cpu_supports_avx2() 为 False
# （即 SDE 确实隐藏了 AVX2），且 quspect._core_avx2 没有被导入。
# 两项都跑完再汇总结果，避免第一项失败时看不到第二项的 chip-check 日志。
failed=0
QUSPECT_EXPECT_OPENMP=1 bash tools/sde_chip_check.sh smoke -nhm -- \
  python tools/wheel_smoke.py --expect-backend scalar --project-dir "$GITHUB_WORKSPACE" \
  || failed=1
bash tools/sde_chip_check.sh native-tests -nhm -- \
  python -m pytest tests/native -q -p no:cacheprovider -o addopts="" \
  || failed=1

exit "$failed"
