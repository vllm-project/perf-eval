#!/usr/bin/env bash
set -euo pipefail

WORKLOAD=${1:?usage: $0 <workload.yaml>}
ROOT=$(git rev-parse --show-toplevel)
GROUP=$(python3 - "$WORKLOAD" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    print(yaml.safe_load(f)["custom_group"])
PY
)
MODEL=/raid/inf-simon/models/nvidia/GLM-5.2-NVFP4
SERVED_MODEL=nvidia/GLM-5.2-NVFP4
RUN_ID=${BUILDKITE_BUILD_NUMBER:-manual}-$(date -u +%Y%m%dT%H%M%SZ)
RESULTS=$ROOT/results/ac-glm52-b300-ablation-${GROUP}-${RUN_ID}
PERSIST=/raid/inf-simon/logs/ac-glm52-b300-ablation-20260810/${GROUP}-${RUN_ID}
HARNESS=$ROOT/manual/ac_glm52_b300_acceptance_v2.py
mkdir -p "$RESULTS"

SERVER_PIDS=()
SERVER_PGIDS=()
MONITOR_PID=""

snapshot_processes() {
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader 2>/dev/null || true
}

stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_servers() {
  stop_monitor
  local pgid pid
  for pgid in "${SERVER_PGIDS[@]}"; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  sleep 5
  mapfile -t leftovers < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u
  )
  for pid in "${leftovers[@]:-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 3
  for pid in "${leftovers[@]:-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -KILL "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
  SERVER_PGIDS=()
}

cleanup() {
  stop_servers
  snapshot_processes >"$RESULTS/final-compute-processes.txt"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
    --format=csv,noheader >"$RESULTS/final-gpus.csv" 2>&1 || true
  mkdir -p "$PERSIST"
  cp -a "$RESULTS/." "$PERSIST/"
}
trap cleanup EXIT

cat >"$RESULTS/manifest.txt" <<EOF
group=$GROUP
run_id=$RUN_ID
git_commit=$(git rev-parse HEAD)
build_url=${BUILDKITE_BUILD_URL:-manual}
image=${VLLM_IMAGE:-from-workload}
remote_evidence=$PERSIST
model=$MODEL
served_model=$SERVED_MODEL
hardware_boundary=exactly 8 NVIDIA B300 GPUs on one standalone node
mix=90% p50 (12500 prompt/10700 intended cached/95 output), 10% p95 (26500/24000/450)
loads_prompt_tpm=2000000,2500000,3000000,3500000,4000000,4500000,5000000
duration_s=120
steady_warmup_fraction=0.2
seed_policy=stable unique seed per load cell, reused across arms
EOF
date -u +%FT%TZ >"$RESULTS/start-time.txt"
uname -a >"$RESULTS/uname.txt"
nvidia-smi -q >"$RESULTS/nvidia-smi-before.txt" 2>&1 || true
python3 - <<'PY' >"$RESULTS/runtime-versions.txt" 2>&1 || true
import flashinfer, torch, transformers, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("flashinfer", flashinfer.__version__)
PY

export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN=1

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

wait_healthy() {
  local port=$1 deadline=$((SECONDS + 7200))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    local pid
    for pid in "${SERVER_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        return 1
      fi
    done
    sleep 5
  done
  return 1
}

launch_server() {
  local arm_dir=$1 devices=$2 port=$3 tp=$4 maxseq=$5 tokens=$6 util=$7
  shift 7
  local log="$arm_dir/server-${port}.log"
  setsid env CUDA_VISIBLE_DEVICES="$devices" \
    vllm serve "$MODEL" \
      "${COMMON_ARGS[@]}" \
      --port "$port" \
      --tensor-parallel-size "$tp" \
      --max-num-seqs "$maxseq" \
      --max-num-batched-tokens "$tokens" \
      --gpu-memory-utilization "$util" \
      "$@" >"$log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  SERVER_PGIDS+=("$pid")
}

run_sweep() {
  local arm=$1
  shift
  local endpoints=("$@")
  local arm_dir="$RESULTS/$arm"
  local loads=(2000000 2500000 3000000 3500000 4000000 4500000 5000000)
  local seeds=(424200 424201 424202 424203 424204 424205 424206)
  local endpoint_args=()
  local endpoint load seed index out
  for endpoint in "${endpoints[@]}"; do
    endpoint_args+=(--endpoint "$endpoint")
  done
  for index in "${!loads[@]}"; do
    load=${loads[$index]}
    seed=${seeds[$index]}
    out="$arm_dir/load-${load}.json"
    echo "--- ${arm}: offered prompt TPM ${load}, seed ${seed}"
    if ! python3 "$HARNESS" \
      "${endpoint_args[@]}" \
      --model "$SERVED_MODEL" \
      --tokenizer "$MODEL" \
      --duration 120 \
      --warmup-fraction 0.2 \
      --target-prompt-tpm "$load" \
      --seed "$seed" \
      --out "$out"; then
      printf 'harness_failed load=%s seed=%s\n' "$load" "$seed" \
        >>"$arm_dir/harness-errors.txt"
    fi
    for endpoint in "${endpoints[@]}"; do
      local port=${endpoint#http://127.0.0.1:}
      port=${port%%/*}
      curl -fsS "http://127.0.0.1:${port}/metrics" \
        >"$arm_dir/metrics-${load}-port-${port}.prom" 2>&1 || true
    done
  done
}

run_arm() {
  local arm=$1 devices=$2 port=$3 tp=$4 maxseq=$5 tokens=$6 util=$7
  shift 7
  local arm_dir="$RESULTS/$arm"
  mkdir -p "$arm_dir"
  stop_servers
  snapshot_processes >"$arm_dir/compute-processes-before.txt"
  printf '%q ' vllm serve "$MODEL" "${COMMON_ARGS[@]}" --port "$port" \
    --tensor-parallel-size "$tp" --max-num-seqs "$maxseq" \
    --max-num-batched-tokens "$tokens" --gpu-memory-utilization "$util" "$@" \
    >"$arm_dir/command.sh"
  printf '\n' >>"$arm_dir/command.sh"
  launch_server "$arm_dir" "$devices" "$port" "$tp" "$maxseq" "$tokens" "$util" "$@"
  if ! wait_healthy "$port"; then
    echo "server_start_failed" >"$arm_dir/status.txt"
    tail -n 300 "$arm_dir/server-${port}.log" >"$arm_dir/server-tail.txt" || true
    stop_servers
    return 0
  fi
  echo "healthy" >"$arm_dir/status.txt"
  curl -fsS "http://127.0.0.1:${port}/v1/models" >"$arm_dir/models.json"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv -l 1 >"$arm_dir/gpu-telemetry.csv" 2>&1 &
  MONITOR_PID=$!
  run_sweep "$arm" "http://127.0.0.1:${port}/v1"
  stop_servers
  grep -Eai 'traceback|engine.*error|out of memory|oom|nan|corrupt|spec.*accept|prefix.*cache|cache.*hit' \
    "$arm_dir/server-${port}.log" >"$arm_dir/log-signals.txt" || true
  snapshot_processes >"$arm_dir/compute-processes-after.txt"
}

run_two_tp4() {
  local arm=G-two-tp4
  local arm_dir="$RESULTS/$arm"
  mkdir -p "$arm_dir"
  stop_servers
  snapshot_processes >"$arm_dir/compute-processes-before.txt"
  # Match arm E's global scheduler envelope: each logical TP4 engine receives
  # half the sequence and token budget, with deterministic client round-robin.
  launch_server "$arm_dir" 0,1,2,3 8000 4 16 4096 0.95 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  launch_server "$arm_dir" 4,5,6,7 8001 4 16 4096 0.95 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  if ! wait_healthy 8000 || ! wait_healthy 8001; then
    echo "unsupported_or_server_start_failed" >"$arm_dir/status.txt"
    stop_servers
    return 0
  fi
  echo "healthy" >"$arm_dir/status.txt"
  curl -fsS http://127.0.0.1:8000/v1/models >"$arm_dir/models-8000.json"
  curl -fsS http://127.0.0.1:8001/v1/models >"$arm_dir/models-8001.json"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv -l 1 >"$arm_dir/gpu-telemetry.csv" 2>&1 &
  MONITOR_PID=$!
  run_sweep "$arm" http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1
  stop_servers
  grep -Eai 'traceback|engine.*error|out of memory|oom|nan|corrupt|spec.*accept|prefix.*cache|cache.*hit' \
    "$arm_dir"/server-*.log >"$arm_dir/log-signals.txt" || true
  snapshot_processes >"$arm_dir/compute-processes-after.txt"
}

case "$GROUP" in
  current)
    run_arm A-live 0,1,2,3,4,5,6,7 8000 8 64 16384 0.85
    run_arm B-mtp3 0,1,2,3,4,5,6,7 8000 8 64 16384 0.85 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    run_arm C-seq32-tokens8192 0,1,2,3,4,5,6,7 8000 8 32 8192 0.85
    run_arm D-ep 0,1,2,3,4,5,6,7 8000 8 64 16384 0.85 \
      --enable-expert-parallel --enable-ep-weight-filter
    run_arm E-combined 0,1,2,3,4,5,6,7 8000 8 32 8192 0.95 \
      --enable-expert-parallel --enable-ep-weight-filter \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    ;;
  rebased)
    run_arm F-pr51425 0,1,2,3,4,5,6,7 8000 8 32 8192 0.95 \
      --enable-expert-parallel --enable-ep-weight-filter \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    run_two_tp4
    ;;
  *)
    echo "unknown custom_group: $GROUP" >&2
    exit 2
    ;;
esac

date -u +%FT%TZ >"$RESULTS/end-time.txt"
echo "results=$RESULTS"
