# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime]
#   wait_healthy <base_url-or-port> [timeout_s=1500]
#   stop_server  <container>
#
# `env` is a newline-separated list of KEY=VALUE pairs. For Docker runtime,
# each value is injected into the container with -e. As a special case, HF_HOME
# is also bind-mounted at the same path inside the container so the model cache
# on the host is visible to vLLM. For native runtime, values are exported before
# starting `vllm serve` in the current job container.
#
# After start_server, vLLM logs are streamed to stdout (prefixed with `[vllm]`)
# so build output reflects server startup progress in real time. The streamer's
# PID is held in $VLLM_LOGS_PID; stop_server kills its process group when
# supported so pipeline children such as `tail -F` do not survive teardown.
#
# For `runtime=slurm`, start_server delegates to lib/slurm_launch_vllm.sh by
# default. Custom launchers can be provided with PERF_EVAL_SLURM_LAUNCHER. The
# launcher must submit a long-running server job, write an endpoint env file,
# and print the Slurm job id on stdout.

start_log_stream() {
  local source_cmd=$1
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c "$source_cmd | stdbuf -oL -eL sed 's/^/[vllm] /'" &
    VLLM_LOGS_PID=$!
    VLLM_LOGS_PGID=$VLLM_LOGS_PID
  else
    bash -c "$source_cmd | stdbuf -oL -eL sed 's/^/[vllm] /'" &
    VLLM_LOGS_PID=$!
    VLLM_LOGS_PGID=""
  fi
}

stop_log_stream() {
  if [[ -z "${VLLM_LOGS_PID:-}" ]]; then
    return
  fi
  if [[ -n "${VLLM_LOGS_PGID:-}" ]]; then
    kill -- "-$VLLM_LOGS_PGID" 2>/dev/null || true
  else
    kill "$VLLM_LOGS_PID" 2>/dev/null || true
  fi
  wait "$VLLM_LOGS_PID" 2>/dev/null || true
  VLLM_LOGS_PID=""
  VLLM_LOGS_PGID=""
}

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
    vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
    VLLM_SERVER_PID=$!
    VLLM_BASE_URL="http://127.0.0.1:${port}"
    echo "--- :memo: streaming vllm logs"
    start_log_stream "tail -f $(printf "%q" "$log_file") 2>/dev/null"
    return
  fi

  if [[ "$runtime" == "slurm" || "$runtime" == "slurm_pyxis" || "$runtime" == "srt_slurm" ]]; then
    local results_dir="results/${WORKLOAD_NAME:-$container}"
    local launcher="${PERF_EVAL_SLURM_LAUNCHER:-$DIR/slurm_launch_vllm.sh}"
    local endpoint_file="${PERF_EVAL_SLURM_ENDPOINT_FILE:-${results_dir}/${container}.endpoint}"
    local env_file="${results_dir}/${container}.env"
    mkdir -p "$results_dir"
    printf '%s\n' "$env" > "$env_file"
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      case "$kv" in
        PERF_EVAL_SLURM_*=*)
          export "$kv"
          ;;
      esac
    done <<< "$env"

    [[ -x "$launcher" ]] || { echo "Slurm launcher not executable: $launcher" >&2; return 2; }

    echo "--- :satellite: submitting vLLM Slurm server"
    VLLM_SLURM_JOB_ID="$(
      PERF_EVAL_CONTAINER_NAME="$container" \
      PERF_EVAL_PORT="$port" \
      PERF_EVAL_IMAGE="$image" \
      PERF_EVAL_MODEL="$model" \
      PERF_EVAL_SERVE_ARGS="$serve_args" \
      PERF_EVAL_ENV_FILE="$env_file" \
      PERF_EVAL_ENDPOINT_FILE="$endpoint_file" \
      PERF_EVAL_RESULTS_DIR="$results_dir" \
      PERF_EVAL_NUM_NODES="${WORKLOAD_NUM_NODES:-1}" \
      PERF_EVAL_GPUS_PER_NODE="${WORKLOAD_GPUS_PER_NODE:-${WORKLOAD_NUM_GPUS:-1}}" \
      PERF_EVAL_NUM_GPUS="${WORKLOAD_NUM_GPUS:-}" \
      "$launcher"
    )"
    export VLLM_SLURM_JOB_ID

    local timeout="${PERF_EVAL_SLURM_ENDPOINT_TIMEOUT_S:-600}"
    local deadline=$(( $(date +%s) + timeout ))
    while [[ ! -s "$endpoint_file" ]]; do
      if (( $(date +%s) >= deadline )); then
        echo "Slurm server did not publish endpoint file: $endpoint_file" >&2
        return 1
      fi
      sleep 5
    done
    # shellcheck disable=SC1090
    source "$endpoint_file"
    VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT:-$port}}"
    export VLLM_BASE_URL
    echo "server endpoint: $VLLM_BASE_URL"
    if [[ -n "${VLLM_SLURM_LOG_FILE:-}" ]]; then
      echo "--- :memo: streaming vllm Slurm logs"
      start_log_stream "tail -F $(printf "%q" "$VLLM_SLURM_LOG_FILE") 2>/dev/null"
    fi
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
  # vllm/vllm-openai's entrypoint takes the model as the first positional
  # arg; do not prepend `vllm` or `serve`.
  docker run -d --rm --name "$container" "${docker_args[@]}" \
    "$image" \
    "$model" --port "$port" $serve_args
  VLLM_BASE_URL="http://127.0.0.1:${port}"

  # Install pytest to avoid cupy.testing import failure during torch.compile
  docker exec "$container" pip install -q pytest 2>/dev/null || true

  echo "--- :memo: streaming vllm logs"
  start_log_stream "docker logs -f $(printf "%q" "$container") 2>&1"
}

wait_healthy() {
  local target=$1 timeout=${2:-3600}
  local health_url
  if [[ "$target" == http://* || "$target" == https://* ]]; then
    health_url="${target%/}/health"
  else
    health_url="http://localhost:${target}/health"
  fi
  echo "+++ :hourglass: waiting for ${health_url} (timeout ${timeout}s)"
  local now start deadline next_status elapsed
  start=$(date +%s)
  deadline=$(( start + timeout ))
  next_status=$(( start + 60 ))
  while (( $(date +%s) < deadline )); do
    if curl -fs "$health_url" >/dev/null 2>&1; then
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
  stop_log_stream
  if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_SLURM_JOB_ID:-}" ]]; then
    scancel "$VLLM_SLURM_JOB_ID" >/dev/null 2>&1 || true
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}
