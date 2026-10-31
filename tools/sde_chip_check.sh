#!/usr/bin/env bash
# 在 Intel SDE 下运行命令，并只对 QuSpect 自身模块中的非法指令判定失败。
#
# 用法：
#   SDE_BIN=/path/to/sde64 QUSPECT_PKG_DIR=/path/to/site-packages/quspect \
#     tools/sde_chip_check.sh <label> <chip-option> -- <command> [args...]
#
# 例如：tools/sde_chip_check.sh smoke -nhm -- python tools/wheel_smoke.py ...
#
# 宿主系统库（ld.so、libc、libpython、numpy、libgomp 等）由宿主 glibc 选择
# 指令集分支，在旧 CPU 模型下会执行不支持的指令（例如 ld.so 中的 vpxor）。
# 因此使用 SDE chip-check 控制项（见 SDE 10.13.1 自带文档 doc/chip_check.html）：
#   -chip_check_die 0        违规只警告，不终止进程
#   -chip_check_emit_file 1  将违规写入文件
#   -chip_check_file <path>  违规输出文件路径（子进程的文件会附加 pid）
#   -chip_check_stderr 0     不把违规刷到 stderr（文件中已有完整记录）
# 运行结束后解析违规文件：
#   - 命令自身失败 → 失败；
#   - 违规文件缺失 → 失败（说明控制项未生效）；
#   - 任何违规的 Image 位于 $QUSPECT_PKG_DIR/ 下，或提及 _core_scalar /
#     _core_avx2 → 失败；
#   - 日志中出现 _core_avx2 → 失败（该模块在无 AVX2 的 CPU 上不应被加载）；
#   - 其余镜像中的违规仅以 notice 汇总。

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
  echo "usage: $0 <label> <chip-option> -- <command> [args...]" >&2
  exit 2
fi

label=$1
chip=$2
shift 3

: "${SDE_BIN:?SDE_BIN must point to sde64}"
: "${QUSPECT_PKG_DIR:?QUSPECT_PKG_DIR must point to the installed quspect package}"

log_root=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sde-chip-check
log_dir=$log_root/$label
rm -rf "$log_dir"
mkdir -p "$log_dir"
log_file=$log_dir/chip-check.txt

status=0
"$SDE_BIN" "$chip" \
  -chip_check_die 0 \
  -chip_check_emit_file 1 \
  -chip_check_file "$log_file" \
  -chip_check_stderr 0 \
  -- "$@" || status=$?

if [ "$status" -ne 0 ]; then
  echo "::error::[$label] command exited with status $status under SDE"
fi

shopt -s nullglob
logs=("$log_dir"/*)
shopt -u nullglob

if [ "${#logs[@]}" -eq 0 ]; then
  echo "::error::[$label] SDE chip-check file not found in $log_dir;" \
    "the -chip_check_emit_file/-chip_check_file knobs were not honored"
  exit 1
fi

failed=0
if [ "$status" -ne 0 ]; then
  failed=1
fi

combined=$log_dir/combined.log
cat "${logs[@]}" > "$combined"

echo "::group::[$label] SDE chip-check files"
for f in "${logs[@]}"; do
  lines=$(wc -l < "$f")
  echo "---- $f ($lines lines, first 80 shown)"
  head -n 80 "$f"
done
echo "::endgroup::"

violations=$(grep -c "not valid for specified chip" "$combined" || true)

echo "[$label] violations by image (expected in host libraries):"
grep -E '^[[:space:]]*Image:' "$combined" \
  | sed -E 's/^[[:space:]]*Image:[[:space:]]*//; s/\+0x[0-9a-fA-F]+.*$//' \
  | sort | uniq -c | sort -rn || true

own_hits=$(grep -E '^[[:space:]]*Image:' "$combined" \
  | grep -F -e "$QUSPECT_PKG_DIR/" -e "_core_scalar" -e "_core_avx2" || true)
if [ -n "$own_hits" ]; then
  echo "::error::[$label] unsupported instructions executed in QuSpect's own modules:"
  echo "$own_hits"
  failed=1
fi

if grep -q "_core_avx2" "$combined"; then
  echo "::error::[$label] _core_avx2 appears in the SDE chip-check log:"
  grep -n -B2 -A2 "_core_avx2" "$combined" | head -n 40
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

echo "::notice::[$label] $violations chip-check violation(s), all outside QuSpect's own modules"
