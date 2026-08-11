# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime] [startup_timeout_s]
#   wait_healthy <port> [timeout_s=3600]
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
# supported so pipeline children such as `tail -f` do not survive teardown.

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
  local startup_timeout=${8:-3600}
  echo "--- :rocket: starting vllm: $model"

  if [[ "$runtime" == "native" ]]; then
    export VLLM_ENGINE_READY_TIMEOUT_S="$startup_timeout"
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      export "$kv"
    done <<< "$env"
    local log_file="/tmp/${container}.log"
    VLLM_LOG_FILE="$log_file"
    # shellcheck disable=SC2086  # serve_args intentionally word-split
    vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
    VLLM_SERVER_PID=$!
    echo "--- :memo: streaming vllm logs"
    start_log_stream "tail -f $(printf "%q" "$log_file") 2>/dev/null"
    return
  fi

  local docker_args=(--gpus all --ipc=host --ulimit nofile=65536:65536
                     -e "VLLM_ENGINE_READY_TIMEOUT_S=${startup_timeout}"
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

  # Install pytest to avoid cupy.testing import failure during torch.compile
  docker exec "$container" pip install -q pytest 2>/dev/null || true

  echo "--- :memo: streaming vllm logs"
  start_log_stream "docker logs -f $(printf "%q" "$container") 2>&1"
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
  stop_log_stream
  if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}
