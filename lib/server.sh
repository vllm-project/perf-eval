# shellcheck shell=bash
# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   pick_server_port
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime]
#   wait_healthy <port> [timeout_s=3600] [expected_model]
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
# PID is held in $VLLM_LOGS_PID; stop_server kills it.

pick_server_port() {
  # Native GPU jobs use host networking and multiple jobs can share a node.
  # Derive a stable high port from the Buildkite job ID so one job cannot
  # mistake another job's vLLM server on port 8000 for its own.
  local seed=${BUILDKITE_JOB_ID:-${HOSTNAME:-local}-$$}
  python3 - "$seed" <<'PY'
import hashlib
import socket
import sys

seed = sys.argv[1].encode()
first = 20000 + int.from_bytes(hashlib.sha256(seed).digest()[:4], "big") % 40000
for offset in range(40000):
    port = 20000 + (first - 20000 + offset) % 40000
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit("no free server port in 20000-59999")
PY
}

start_server() {
  local container=$1 port=$2 image=$3 model=$4 serve_args=$5 env=$6 runtime=${7:-docker}
  echo "--- :rocket: starting vllm: $model"

  if [[ "$runtime" == "native" ]]; then
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      # shellcheck disable=SC2163  # kv is an intentional KEY=VALUE assignment
      export "$kv"
    done <<< "$env"
    local log_file="/tmp/${container}.log"
    VLLM_LOG_FILE="$log_file"
    # shellcheck disable=SC2086  # serve_args intentionally word-split
    vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
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
  # vllm/vllm-openai's entrypoint takes the model as the first positional
  # arg; do not prepend `vllm` or `serve`.
  docker run -d --rm --name "$container" "${docker_args[@]}" \
    "$image" \
    "$model" --port "$port" $serve_args

  # Install pytest to avoid cupy.testing import failure during torch.compile
  docker exec "$container" pip install -q pytest 2>/dev/null || true

  echo "--- :memo: streaming vllm logs"
  ( docker logs -f "$container" 2>&1 | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
  VLLM_LOGS_PID=$!
}

server_is_healthy() {
  local port=$1 expected_model=${2:-}
  curl -fs "http://localhost:${port}/health" >/dev/null 2>&1 || return 1
  [[ -z "$expected_model" ]] && return 0
  curl -fs "http://localhost:${port}/v1/models" 2>/dev/null |
    python3 -c '
import json
import sys

expected = sys.argv[1]
models = json.load(sys.stdin).get("data", [])
raise SystemExit(0 if any(model.get("id") == expected for model in models) else 1)
' "$expected_model" 2>/dev/null
}

wait_healthy() {
  local port=$1 timeout=${2:-3600} expected_model=${3:-}
  echo "+++ :hourglass: waiting for /health (timeout ${timeout}s)"
  local now start deadline next_status elapsed
  start=$(date +%s)
  deadline=$(( start + timeout ))
  next_status=$(( start + 60 ))
  while (( $(date +%s) < deadline )); do
    if server_is_healthy "$port" "$expected_model"; then
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

stop_process() {
  local pid=${1:-} timeout=${2:-10}
  [[ -z "$pid" ]] && return 0
  kill "$pid" 2>/dev/null || true
  local start=$SECONDS
  while kill -0 "$pid" 2>/dev/null && (( SECONDS - start < timeout )); do
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

stop_server() {
  local container=$1
  # Stop the server/container before its log follower. If the follower's
  # pipeline does not propagate TERM, stop_process escalates after a bounded
  # wait instead of hanging the job until Buildkite's timeout.
  stop_process "${VLLM_SERVER_PID:-}"
  docker rm -f "$container" >/dev/null 2>&1 || true
  stop_process "${VLLM_LOGS_PID:-}"
}
