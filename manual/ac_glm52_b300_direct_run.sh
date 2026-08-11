#!/usr/bin/env bash
set -euo pipefail

CURRENT_IMAGE=${CURRENT_IMAGE:-inferactinc/public:glm52-ll-8d407ae@sha256:3ea9431a2298950a1aa2b4c07786b18396c756ee6f21b6cb49984620d1ab5413}
REBASED_IMAGE=${REBASED_IMAGE:-local/ac-glm52-pr51425:e7709dcf3}
EXPECTED_REBASED_VERSION_FRAGMENT=${EXPECTED_REBASED_VERSION_FRAGMENT:-e7709dcf3}
MODEL_HOST=${MODEL_HOST:-/raid/inf-simon/models/nvidia/GLM-5.2-NVFP4}
CACHE_HOST=${CACHE_HOST:-/raid/inf-simon/glm52-repro/cache}
SERVED_MODEL=nvidia/GLM-5.2-NVFP4
ROOT=$(git rev-parse --show-toplevel)
HARNESS_HOST=$ROOT/manual/ac_glm52_b300_acceptance_v2.py
SUMMARIZER=$ROOT/manual/summarize_ac_glm52.py
PRE_RUN_ACK=${PRE_RUN_ACK:?set PRE_RUN_ACK to the evidence file written after the required pre-run Raft ping}
RUN_ID=${RUN_ID:-direct-$(date -u +%Y%m%dT%H%M%SZ)}
RESULTS=${RESULTS:-/raid/inf-simon/logs/ac-glm52-b300-ablation-20260811/$RUN_ID}
PREFIX=ac-glm52-${RUN_ID//[^a-zA-Z0-9_.-]/-}

mkdir -p "$RESULTS"
RESULTS=$(cd "$RESULTS" && pwd)
exec 9>"$RESULTS/run.lock"
if ! flock -n 9; then
  echo "another sweep holds $RESULTS/run.lock" >&2
  exit 1
fi

if [[ ! -s "$PRE_RUN_ACK" ]]; then
  echo "pre-run gate failed: $PRE_RUN_ACK is absent or empty" >&2
  exit 1
fi
cp "$PRE_RUN_ACK" "$RESULTS/pre-run-raft-ack.txt"

CONTAINERS=()
LOG_PIDS=()
MONITOR_PID=""

stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_servers() {
  stop_monitor
  local name pid
  for name in "${CONTAINERS[@]:-}"; do
    docker stop -t 30 "$name" >/dev/null 2>&1 || \
      docker rm -f "$name" >/dev/null 2>&1 || true
  done
  for pid in "${LOG_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  CONTAINERS=()
  LOG_PIDS=()
  sleep 5
}

capture_final_state() {
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"$RESULTS/final-compute-processes.csv" 2>&1 || true
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,pstate \
    --format=csv,noheader,nounits >"$RESULTS/final-gpus.csv" 2>&1 || true
  docker ps -a --filter "name=^/${PREFIX}" --format '{{.ID}} {{.Names}} {{.Status}}' \
    >"$RESULTS/final-containers.txt" 2>&1 || true
  date -u +%FT%TZ >"$RESULTS/final-state-time-utc.txt"
}

cleanup() {
  local status=$?
  set +e
  stop_servers
  capture_final_state
  if [[ -s "$RESULTS/final-compute-processes.csv" ]]; then
    printf 'cleanup_gate=FAIL active GPU compute processes remain\n' \
      >"$RESULTS/cleanup-status.txt"
    status=1
  elif awk -F, '$3 + 0 != 0 || $4 + 0 > 16 { exit 1 }' \
    "$RESULTS/final-gpus.csv"; then
    printf 'cleanup_gate=PASS\n' >"$RESULTS/cleanup-status.txt"
  else
    printf 'cleanup_gate=FAIL GPU utilization or memory is not idle\n' \
      >"$RESULTS/cleanup-status.txt"
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

probe_host() {
  date -u +%FT%TZ >"$RESULTS/recovery-probe-time-utc.txt"
  hostname >"$RESULTS/hostname.txt"
  uptime >"$RESULTS/uptime.txt"
  timeout 30 nvidia-smi -L >"$RESULTS/nvidia-smi-L.txt"
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,pstate \
    --format=csv,noheader,nounits >"$RESULTS/gpus-before-run.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"$RESULTS/compute-before-run.csv" || true
  journalctl -k -b --no-pager >"$RESULTS/kernel-since-boot.log" 2>&1 || true
  if [[ $(grep -c 'NVIDIA B300' "$RESULTS/nvidia-smi-L.txt") -ne 8 ]]; then
    echo "pre-run gate failed: expected exactly 8 NVIDIA B300 GPUs" >&2
    return 1
  fi
  if [[ -s "$RESULTS/compute-before-run.csv" ]]; then
    echo "pre-run gate failed: GPU compute processes are active" >&2
    return 1
  fi
  if ! awk -F, '$4 + 0 != 0 || $5 + 0 > 16 { exit 1 }' \
    "$RESULTS/gpus-before-run.csv"; then
    echo "pre-run gate failed: GPU utilization or memory is not idle" >&2
    return 1
  fi
  if grep -Eai 'NVRM: Xid|driver rpc error|GPU has fallen off the bus' \
    "$RESULTS/kernel-since-boot.log" >"$RESULTS/kernel-gpu-errors.txt"; then
    echo "pre-run gate failed: kernel GPU errors are present since boot" >&2
    return 1
  fi
  : >"$RESULTS/kernel-gpu-errors.txt"
}

probe_host
[[ -d "$MODEL_HOST" ]] || { echo "missing model: $MODEL_HOST" >&2; exit 1; }
[[ -f "$HARNESS_HOST" ]] || { echo "missing harness: $HARNESS_HOST" >&2; exit 1; }
mkdir -p "$CACHE_HOST"

docker pull "$CURRENT_IMAGE" >"$RESULTS/current-image-pull.txt"
docker image inspect "$CURRENT_IMAGE" >"$RESULTS/current-image-inspect.json"
docker image inspect "$REBASED_IMAGE" >"$RESULTS/rebased-image-inspect.json"
docker run --rm --gpus all --entrypoint nvidia-smi "$CURRENT_IMAGE" -L \
  >"$RESULTS/container-nvidia-smi-L.txt"
if [[ $(grep -c 'NVIDIA B300' "$RESULTS/container-nvidia-smi-L.txt") -ne 8 ]]; then
  echo "container runtime gate failed: expected exactly 8 NVIDIA B300 GPUs" >&2
  exit 1
fi

for spec in "current $CURRENT_IMAGE" "rebased $REBASED_IMAGE"; do
  read -r label image <<<"$spec"
  docker run --rm --entrypoint python3 "$image" -c \
    'import json, platform, torch, transformers, vllm; import flashinfer; print(json.dumps({"python": platform.python_version(), "vllm": vllm.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "flashinfer": flashinfer.__version__}, sort_keys=True))' \
    >"$RESULTS/runtime-${label}.json"
done
if ! grep -q "$EXPECTED_REBASED_VERSION_FRAGMENT" \
  "$RESULTS/runtime-rebased.json"; then
  echo "rebased runtime gate failed: vLLM version does not contain $EXPECTED_REBASED_VERSION_FRAGMENT" >&2
  exit 1
fi

cat >"$RESULTS/manifest.txt" <<EOF
run_id=$RUN_ID
perf_eval_git_commit=$(git rev-parse HEAD)
current_image=$CURRENT_IMAGE
rebased_image=$REBASED_IMAGE
expected_rebased_version_fragment=$EXPECTED_REBASED_VERSION_FRAGMENT
model=$MODEL_HOST
served_model=$SERVED_MODEL
hardware_boundary=exactly 8 NVIDIA B300 GPUs on one standalone node
mix=90% p50 (12500 prompt/10700 intended cached/95 output), 10% p95 (26500/24000/450)
loads_prompt_tpm=2000000,2500000,3000000,3500000,4000000,4500000,5000000
duration_s=120
steady_warmup_fraction=0.2
seed_policy=seed 4242 for all cells; deterministic 1000000 request-ID offset per load bracket, reused across arms
ttft_sla=p50<=1.6s,p95<=3.5s
itl_sla=not supplied; report only
pre_run_ack=$PRE_RUN_ACK
EOF

COMMON_ARGS=(
  --served-model-name "$SERVED_MODEL"
  --host 0.0.0.0
  --quantization modelopt_fp4
  --kv-cache-dtype fp8_e4m3
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --trust-remote-code
  --compilation-config '{"custom_ops":["none","+rms_norm"]}'
  --ir-op-priority.rms_norm=vllm_c
  --ir-op-priority.fused_add_rms_norm=vllm_c
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
)

server_running() {
  local name
  for name in "${CONTAINERS[@]}"; do
    [[ $(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true) == true ]] || return 1
  done
}

wait_healthy() {
  local port=$1 deadline=$((SECONDS + 7200))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    server_running || return 1
    sleep 5
  done
  return 1
}

launch_server() {
  local arm=$1 image=$2 devices=$3 port=$4 tp=$5 maxseq=$6 tokens=$7 util=$8
  shift 8
  local name=${PREFIX}-${arm}-${port}
  local arm_dir=$RESULTS/$arm
  local command=(
    docker run -d --rm
    --name "$name"
    --gpus all
    --ipc host
    --network host
    --ulimit nofile=65536:65536
    -e "CUDA_VISIBLE_DEVICES=$devices"
    -e HF_HUB_OFFLINE=1
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200
    -e FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN=1
    -v "$CACHE_HOST:/root/.cache"
    -v "$MODEL_HOST:/model:ro"
    --entrypoint vllm
    "$image"
    serve /model
    "${COMMON_ARGS[@]}"
    --port "$port"
    --tensor-parallel-size "$tp"
    --max-num-seqs "$maxseq"
    --max-num-batched-tokens "$tokens"
    --gpu-memory-utilization "$util"
    "$@"
  )
  printf '%q ' "${command[@]}" >"$arm_dir/command-${port}.sh"
  printf '\n' >>"$arm_dir/command-${port}.sh"
  "${command[@]}" >"$arm_dir/container-${port}.id"
  CONTAINERS+=("$name")
  docker logs -f "$name" >"$arm_dir/server-${port}.log" 2>&1 &
  LOG_PIDS+=("$!")
}

capture_metrics() {
  local arm_dir=$1 load=$2
  shift 2
  local endpoint port
  for endpoint in "$@"; do
    port=${endpoint#http://127.0.0.1:}
    port=${port%%/*}
    curl -fsS "http://127.0.0.1:${port}/metrics" \
      >"$arm_dir/metrics-${load}-port-${port}.prom" 2>&1 || true
  done
}

severe_server_error() {
  local arm_dir=$1
  grep -Eai 'CUDA out of memory|OutOfMemoryError|device-side assert|EngineCore encountered a fatal|GPU has fallen off|corrupt|invalid probability|nan in' \
    "$arm_dir"/server-*.log >"$arm_dir/severe-errors.txt"
}

verify_gpu_cleanup() {
  local arm_dir=$1 deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits >"$arm_dir/compute-processes-after.csv" 2>/dev/null || true
    if [[ ! -s "$arm_dir/compute-processes-after.csv" ]]; then
      return 0
    fi
    sleep 3
  done
  echo "cleanup gate failed after arm $(basename "$arm_dir")" >&2
  return 1
}

run_sweep() {
  local arm=$1 image=$2
  shift 2
  local endpoints=("$@") arm_dir=$RESULTS/$arm
  local loads=(2000000 2500000 3000000 3500000 4000000 4500000 5000000)
  local endpoint_args=() load index offset out client_name
  local uid gid
  uid=$(id -u)
  gid=$(id -g)
  for endpoint in "${endpoints[@]}"; do
    endpoint_args+=(--endpoint "$endpoint")
  done
  for index in "${!loads[@]}"; do
    load=${loads[$index]}
    offset=$((index * 1000000))
    out=/results/load-${load}.json
    client_name=${PREFIX}-${arm}-client-${load}
    date -u +%FT%TZ >"$arm_dir/load-${load}-start.txt"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
      --format=csv -l 1 >"$arm_dir/gpu-${load}.csv" 2>&1 &
    MONITOR_PID=$!
    docker run --rm --name "$client_name" --network host \
      --user "$uid:$gid" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
      -v "$ROOT/manual:/harness:ro" -v "$MODEL_HOST:/model:ro" \
      -v "$arm_dir:/results" --entrypoint python3 "$image" \
      /harness/ac_glm52_b300_acceptance_v2.py \
      "${endpoint_args[@]}" --model "$SERVED_MODEL" --tokenizer /model \
      --duration 120 --warmup-fraction 0.2 --target-prompt-tpm "$load" \
      --seed 4242 --request-id-offset "$offset" --out "$out" \
      >"$arm_dir/harness-${load}.log" 2>&1
    stop_monitor
    capture_metrics "$arm_dir" "$load" "${endpoints[@]}"
    date -u +%FT%TZ >"$arm_dir/load-${load}-end.txt"
    if ! server_running; then
      echo "server exited during $arm load $load" >&2
      return 1
    fi
    if severe_server_error "$arm_dir"; then
      echo "severe server error during $arm load $load; aborting" >&2
      return 1
    fi
    : >"$arm_dir/severe-errors.txt"
  done
}

run_arm() {
  local arm=$1 image=$2 devices=$3 port=$4 tp=$5 maxseq=$6 tokens=$7 util=$8
  shift 8
  local arm_dir=$RESULTS/$arm
  mkdir -p "$arm_dir"
  stop_servers
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"$arm_dir/compute-processes-before.csv" 2>/dev/null || true
  [[ ! -s "$arm_dir/compute-processes-before.csv" ]] || {
    echo "pre-arm gate failed: GPU processes exist before $arm" >&2
    return 1
  }
  launch_server "$arm" "$image" "$devices" "$port" "$tp" "$maxseq" "$tokens" "$util" "$@"
  if ! wait_healthy "$port"; then
    echo "server_start_failed" >"$arm_dir/status.txt"
    return 1
  fi
  echo "healthy" >"$arm_dir/status.txt"
  curl -fsS "http://127.0.0.1:${port}/v1/models" >"$arm_dir/models.json"
  run_sweep "$arm" "$CURRENT_IMAGE" "http://127.0.0.1:${port}/v1"
  stop_servers
  verify_gpu_cleanup "$arm_dir"
}

run_two_tp4() {
  local arm=G-two-tp4 arm_dir=$RESULTS/G-two-tp4
  mkdir -p "$arm_dir"
  stop_servers
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"$arm_dir/compute-processes-before.csv" 2>/dev/null || true
  [[ ! -s "$arm_dir/compute-processes-before.csv" ]] || {
    echo "pre-arm gate failed: GPU processes exist before $arm" >&2
    return 1
  }
  launch_server "$arm" "$REBASED_IMAGE" 0,1,2,3 8000 4 16 4096 0.95 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  launch_server "$arm" "$REBASED_IMAGE" 4,5,6,7 8001 4 16 4096 0.95 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  if ! wait_healthy 8000 || ! wait_healthy 8001; then
    echo "unsupported_or_server_start_failed" >"$arm_dir/status.txt"
    stop_servers
    verify_gpu_cleanup "$arm_dir"
    return 0
  fi
  echo "healthy" >"$arm_dir/status.txt"
  curl -fsS http://127.0.0.1:8000/v1/models >"$arm_dir/models-8000.json"
  curl -fsS http://127.0.0.1:8001/v1/models >"$arm_dir/models-8001.json"
  run_sweep "$arm" "$CURRENT_IMAGE" \
    http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1
  stop_servers
  verify_gpu_cleanup "$arm_dir"
}

date -u +%FT%TZ >"$RESULTS/start-time-utc.txt"
run_arm A-live "$CURRENT_IMAGE" 0,1,2,3,4,5,6,7 8000 8 64 16384 0.85
run_arm B-mtp3 "$CURRENT_IMAGE" 0,1,2,3,4,5,6,7 8000 8 64 16384 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
run_arm C-seq32-tokens8192 "$CURRENT_IMAGE" 0,1,2,3,4,5,6,7 8000 8 32 8192 0.85
run_arm D-ep "$CURRENT_IMAGE" 0,1,2,3,4,5,6,7 8000 8 64 16384 0.85 \
  --enable-expert-parallel --enable-ep-weight-filter
run_arm E-combined "$CURRENT_IMAGE" 0,1,2,3,4,5,6,7 8000 8 32 8192 0.95 \
  --enable-expert-parallel --enable-ep-weight-filter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
run_arm F-pr51425 "$REBASED_IMAGE" 0,1,2,3,4,5,6,7 8000 8 32 8192 0.95 \
  --enable-expert-parallel --enable-ep-weight-filter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
run_two_tp4

python3 "$SUMMARIZER" "$RESULTS" \
  --json-out "$RESULTS/summary.json" \
  --markdown-out "$RESULTS/summary.md"
date -u +%FT%TZ >"$RESULTS/end-time-utc.txt"
echo "results=$RESULTS"
