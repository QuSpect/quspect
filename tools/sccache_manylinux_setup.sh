#!/usr/bin/env bash
# 在 cibuildwheel 的 manylinux 容器内安装 sccache 并启动其服务进程（GitHub Actions
# 缓存后端）。只在 pull_request 触发的 wheel 构建中使用：wheels.yml 通过
# CIBW_BEFORE_ALL_LINUX 调用本脚本，并用 CIBW_ENVIRONMENT_PASS_LINUX 传入下列变量：
#
#   QSPECT_SCCACHE_HOST_BIN  宿主机上 sccache 的路径（mozilla-actions/sccache-action
#                            安装的静态 musl 版本，已校验 sha256；容器内经 /host 访问）
#   QSPECT_SCCACHE_ENV_FILE  宿主机上的环境文件（SCCACHE_GHA_ENABLED、ACTIONS_RESULTS_URL、
#                            ACTIONS_RUNTIME_TOKEN 等）。缓存凭据只交给 sccache 服务进程，
#                            不经 environment-pass 传递，避免 cibuildwheel 在日志中打印。
#
# 编译时 CMake 以 CMAKE_CXX_COMPILER_LAUNCHER=sccache 调用 /usr/local/bin/sccache，
# 客户端连接这里启动的服务进程（SCCACHE_IDLE_TIMEOUT=0，在容器存活期间不退出）。
# 任何一步失败（缺少二进制、缓存后端不可用导致服务启动失败）都不让构建失败：
# 改为安装一个直接执行编译器的透传脚本，本次构建退回为不使用缓存的普通编译。
set -uo pipefail

dest=/usr/local/bin/sccache

install_passthrough() {
  echo "::warning::sccache unavailable ($1); compiling without the cache"
  cat >"$dest" <<'EOF'
#!/bin/sh
# sccache 不可用时的透传脚本：作为编译器 launcher 时直接执行编译器；
# 以选项调用（如 --show-stats）时什么也不做。
case "${1:-}" in
  -*) echo "sccache: disabled (passthrough shim), ignoring: $*" >&2; exit 0 ;;
esac
exec "$@"
EOF
  chmod 0755 "$dest"
}

src="/host${QSPECT_SCCACHE_HOST_BIN:-/nonexistent}"
env_file="/host${QSPECT_SCCACHE_ENV_FILE:-/nonexistent}"

if [ -z "${QSPECT_SCCACHE_HOST_BIN:-}" ] || [ ! -f "$src" ]; then
  install_passthrough "binary not found at $src"
  exit 0
fi
if ! install -m 0755 "$src" "$dest"; then
  install_passthrough "could not install $src"
  exit 0
fi
"$dest" --version

if [ ! -f "$env_file" ]; then
  install_passthrough "environment file not found at $env_file"
  exit 0
fi

# 在子 shell 中导入缓存后端配置并启动服务进程；服务进程继承这些变量后常驻。
if ! (
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
  "$dest" --start-server
); then
  install_passthrough "server failed to start"
  exit 0
fi
"$dest" --show-stats || true
