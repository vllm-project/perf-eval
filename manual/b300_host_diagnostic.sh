#!/usr/bin/env bash
set -uo pipefail

WORKLOAD=${1:?usage: $0 <workload.yaml>}
RESULTS=results/b300-host-diagnostic-${BUILDKITE_BUILD_NUMBER:-manual}
mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/diagnostic.log") 2>&1

echo "timestamp=$(date -u +%FT%TZ)"
echo "host=$(hostname)"
echo "workload=$WORKLOAD"
echo "build_url=${BUILDKITE_BUILD_URL:-manual}"

run() {
  echo "--- $*"
  timeout 120 "$@"
  local status=$?
  echo "status=$status"
  return 0
}

run nvidia-smi -L
run nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,utilization.gpu,memory.used \
  --format=csv
run nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
run docker version
run docker info
run docker ps -a --no-trunc
run nvidia-container-cli --version
run nvidia-ctk --version
run bash -lc 'ps -ef | grep -E "nvidia|buildkite|dockerd|containerd" | grep -v grep'
run bash -lc 'journalctl --no-pager -n 200 -u docker -u nvidia-persistenced 2>&1'

echo "--- minimal Docker GPU probe"
timeout 180 docker run --rm --gpus all --entrypoint nvidia-smi \
  inferactinc/public:glm52-ll-8d407ae@sha256:3ea9431a2298950a1aa2b4c07786b18396c756ee6f21b6cb49984620d1ab5413 \
  -L
probe_status=$?
echo "probe_status=$probe_status"

docker ps -aq --filter label=ac-glm52-host-diagnostic | xargs -r docker rm -f || true
exit "$probe_status"
