# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <hf_home> <serve_args>
#   wait_healthy <port> [timeout_s=1500]
#   stop_server  <container>
#
# After start_server, container logs are streamed to stdout (prefixed with
# `[vllm]`) so build output reflects server startup progress in real time.
# The streamer's PID is held in $VLLM_LOGS_PID; stop_server kills it.

start_server() {
  local container=$1 port=$2 image=$3 model=$4 hf_home=$5 serve_args=$6
  echo "--- :rocket: starting vllm: $model"
  # shellcheck disable=SC2086  # serve_args intentionally word-split
  # vllm/vllm-openai's entrypoint takes the model as the first positional
  # arg; do not prepend `vllm` or `serve`.
  docker run -d --rm --name "$container" \
    --gpus all --ipc=host -p "${port}:${port}" \
    -v "${hf_home}:${hf_home}" \
    -e "HF_HOME=${hf_home}" \
    "$image" \
    "$model" --port "$port" $serve_args

  echo "--- :memo: streaming vllm logs"
  ( docker logs -f "$container" 2>&1 | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
  VLLM_LOGS_PID=$!
}

wait_healthy() {
  local port=$1 timeout=${2:-1500}
  echo "+++ :hourglass: waiting for /health (timeout ${timeout}s)"
  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if curl -fs "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "server healthy"
      return 0
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
    wait "$VLLM_LOGS_PID" 2>/dev/null || true
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}
