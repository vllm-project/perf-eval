#!/usr/bin/env bash
# Launch `vllm serve` under Nsight Systems. Runs where vLLM runs: as the
# Docker container's entrypoint, or directly in the native-runtime job pod.
#
# Usage: nsys_serve.sh <report_path_without_ext> <vllm serve args...>
#
# The vLLM images don't ship nsys, so it is installed from the CUDA apt repo
# that the nvidia/cuda base images already configure. Collection only covers
# a cudaProfilerStart/Stop range (vLLM's `--profiler-config.profiler cuda`,
# driven by /start_profile and /stop_profile), and stops after the first
# range so the report is written while the server keeps running.
#
# If nsys can't be installed the server still starts, unprofiled, and
# <report>.unavailable is written so the caller doesn't wait for a report.
set -uo pipefail

report=${1:?usage: $0 <report_path_without_ext> <vllm serve args...>}
shift

find_nsys() {
  local candidate
  for candidate in "$(command -v nsys 2>/dev/null)" \
                   /usr/local/cuda/bin/nsys \
                   /opt/nvidia/nsight-systems/*/bin/nsys; do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

install_nsys() {
  command -v apt-get >/dev/null 2>&1 || { echo "nsys: apt-get not available" >&2; return 1; }
  local sudo=()
  if [[ "$(id -u)" != 0 ]]; then
    sudo -n true 2>/dev/null || { echo "nsys: need root or passwordless sudo to install" >&2; return 1; }
    sudo=(sudo -n)
  fi
  echo "nsys: installing Nsight Systems from the CUDA apt repo" >&2
  "${sudo[@]}" apt-get update -qq >/dev/null || return 1
  local pkg
  pkg=$(apt-cache search --names-only '^nsight-systems-[0-9][0-9.]*$' |
        cut -d' ' -f1 | sort -V | tail -n 1)
  [[ -n "$pkg" ]] || { echo "nsys: no nsight-systems package in apt sources" >&2; return 1; }
  DEBIAN_FRONTEND=noninteractive "${sudo[@]}" \
    apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null || return 1
}

nsys_bin=$(find_nsys) || { install_nsys && nsys_bin=$(find_nsys); } || nsys_bin=""
mkdir -p "$(dirname "$report")"
if [[ -z "$nsys_bin" ]]; then
  echo "nsys: unavailable; serving without profiling" >&2
  touch "${report}.unavailable"
  exec vllm serve "$@"
fi
echo "nsys: $("$nsys_bin" --version 2>/dev/null)" >&2

exec "$nsys_bin" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$report" \
  vllm serve "$@"
