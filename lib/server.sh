# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime]
#   wait_healthy <port> [timeout_s=1500]
#   stop_server  <container>
#
# `env` is a newline-separated list of KEY=VALUE pairs. For Docker runtime,
# each value is injected into the container with -e. As a special case, HF_HOME
# is also bind-mounted at the same path inside the container so the model cache
# on the host is visible to vLLM. For native runtime, values are exported before
# starting `vllm serve` in the current job container.
# Note: attention backend selection is passed via --attention-backend in
# serve_args, not as an environment variable (vLLM does not support
# VLLM_ATTENTION_BACKEND as an env var).
#
# After start_server, vLLM logs are streamed to stdout (prefixed with `[vllm]`)
# so build output reflects server startup progress in real time. The streamer's
# PID is held in $VLLM_LOGS_PID; stop_server kills it.

start_server() {
  local container=$1 port=$2 image=$3 model=$4 serve_args=$5 env=$6 runtime=${7:-docker}
  echo "--- :rocket: starting vllm: $model"

  if [[ "$runtime" == "native" ]]; then
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      export "$kv"
    done <<< "$env"
    local log_file="/tmp/${container}.log"
    VLLM_LOG_FILE="$log_file"
    # shellcheck disable=SC2086  # serve_args intentionally word-split
    #echo "Server call: vllm server $model --port $port $serve_args"
    #vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
    local -a serve_args_arr
    IFS=' ' read -ra serve_args_arr <<< "$serve_args"
    echo "Server call: vllm serve $model --port $port ${serve_args_arr[*]}"
    vllm serve "$model" --port "$port" "${serve_args_arr[@]}" >"$log_file" 2>&1 &
    VLLM_SERVER_PID=$!
    echo "--- :memo: streaming vllm logs"
    ( tail -f "$log_file" 2>/dev/null | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
    VLLM_LOGS_PID=$!
    return
  fi

  local docker_args=(--gpus all --ipc=host --ulimit nofile=65536:65536
                     -e VLLM_ENGINE_READY_TIMEOUT_S=3600
                     -p "${port}:${port}")
  local hf_home=""
  while IFS= read -r kv; do
    [[ -z "$kv" ]] && continue
    docker_args+=(-e "$kv")
    [[ "$kv" == HF_HOME=* ]] && hf_home="${kv#HF_HOME=}"
  done <<< "$env"
  if [[ -n "$hf_home" ]]; then
    docker_args+=(-v "${hf_home}:${hf_home}")
  fi

  # shellcheck disable=SC2086  # serve_args intentionally word-split
  local -a server_args_arr
  IFS=' ' read -ra serve_args_arr <<< "$serve_args"
  # vllm/vllm-openai's entrypoint takes the model as the first positional
  # arg; do not prepend `vllm` or `serve`.
  echo "docker run -d --rm --name \"$container\" ${docker_args[*]} \"$image\" \"$model\" --port \"$port\" ${serve_args_arr[*]}" >&2
  docker run -d --rm --name "$container" "${docker_args[@]}" \
    "$image" \
  #  "$model" --port "$port" $serve_args
    "$model" --port "$port" "${serve_args_arr[@]}"

  # Install pytest to avoid cupy.testing import failure during torch.compile
  docker exec "$container" pip install -q pytest 2>/dev/null || true

  # Install pytest to avoid cupy.testing import failure during torch.compile
  docker exec "$container" pip install -q pytest 2>/dev/null || true

  echo "--- :memo: streaming vllm logs"
  ( docker logs -f "$container" 2>&1 | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
  VLLM_LOGS_PID=$!
}

wait_healthy() {
  local port=$1 timeout=${2:-3600}
  echo "+++ :hourglass: waiting for /health (timeout ${timeout}s)"
  local now start deadline next_status elapsed
  start=$(date +%s)
  deadline=$(( start + timeout ))
  next_status=$(( start + 60 ))
  while (( $(date +%s) < deadline )); do
    if curl -fs "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "server healthy"
      return 0
    fi
    if [[ -n "${VLLM_SERVER_PID:-}" ]] && ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
      echo "vLLM server exited before becoming healthy" >&2
      [[ -n "${VLLM_LOG_FILE:-}" ]] && tail -n 80 "$VLLM_LOG_FILE" >&2 || true
      return 1
    fi
    now=$(date +%s)
    if (( now >= next_status )); then
      elapsed=$(( now - start ))
      echo "still waiting for /health after ${elapsed}s"
      next_status=$(( now + 60 ))
    fi
    sleep 5
  done
  echo "server never came up" >&2
  return 1
}

stop_server() {
  local container=$1
  if [[ -n "${VLLM_LOGS_PID:-}" ]]; then
    kill "$VLLM_LOGS_PID" 2>/dev/null || true
    pkill -P "$VLLM_LOGS_PID" 2>/dev/null || true
    wait "$VLLM_LOGS_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    kill -SIGINT "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}

# Wait until all GPUs have freed their VRAM below a low watermark, or until
# timeout. Call this between backends so the next vllm serve doesn't OOM on
# memory still held by the dying container's ROCm context.
# rocm-smi --showmeminfo vram emits lines like:
#   GPU[0] : VRAM Total Used Memory (B): 12345678
# timeout default: 120sec, threshold_gib default: 0
drain_gpu() {
  local timeout=${1:-120} threshold_gib=${2:-0}
  local threshold_bytes=$(( threshold_gib * 1024 * 1024 * 1024 ))
  local current_used=0
  echo "--- :hourglass: waiting for GPU VRAM to drain (threshold ${threshold_gib} GiB, timeout ${timeout}s)"
  local deadline
  deadline=$(( $(date +%s) + timeout ))
  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm-smi unavailable; skipping GPU drain check" >&2
    return 0
  fi
  while (( $(date +%s) < deadline )); do
    current_used=$(rocm-smi --showmeminfo vram --noheader 2>/dev/null \
                 | awk '/VRAM Total Used Memory/{if($NF+0>m)m=$NF+0} END{print m+0}') || true
    if [[ -z "$current_used" ]]; then
      echo "rocm-smi returned no data; skipping GPU drain check" >&2
      return 0
    fi
    if (( current_used <= threshold_bytes )); then
      echo "GPU VRAM drained (max used: $(( current_used / 1024 / 1024 )) MiB)"
      return 0
    fi
    echo "GPU still holds $(( current_used / 1024 / 1024 )) MiB VRAM; waiting..."
    sleep 5
  done
  echo "WARNING: **************************************************************************************************************" >&2
  echo "WARNING: GPU(s)  DID NOT DRAIN COMPLETELY                                                                              " >&2
  echo "WARNING: GPU VRAM did not drain within ${timeout}s (max used: $(( current_used / 1024 / 1024 )) MiB); proceeding anyway" >&2
  echo "WARNING: **************************************************************************************************************" >&2
}

# Wait until all GPUs have dropped to 0% utilization, or until timeout.
# Call this after stop_server to confirm the GPU is truly idle before
# starting the next server (complements drain_gpu which checks VRAM).
# rocm-smi --showuse emits lines like:
#   GPU[0]          : GPU use (%): 12
# timeout default: 60s
wait_gpu_idle() {
  local timeout=${1:-60}
  local max_util=0
  echo "--- :hourglass: waiting for GPU utilization to reach 0% (timeout ${timeout}s)"
  local deadline
  deadline=$(( $(date +%s) + timeout ))
  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm-smi unavailable; skipping GPU idle check" >&2
    return 0
  fi
  while (( $(date +%s) < deadline )); do
    max_util=$(rocm-smi --showuse --noheader 2>/dev/null \
                 | awk '/GPU use \(%\)/{if($NF+0>m)m=$NF+0} END{print m+0}') || true
    if [[ -z "$max_util" ]]; then
      echo "rocm-smi returned no data; skipping GPU idle check" >&2
      return 0
    fi
    if (( max_util == 0 )); then
      echo "GPU utilization is 0% on all devices"
      return 0
    fi
    echo "GPU still at ${max_util}% utilization; waiting..."
    sleep 5
  done
  echo "WARNING: GPU utilization did not reach 0% within ${timeout}s (max: ${max_util}%); proceeding anyway" >&2
}