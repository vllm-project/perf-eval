# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <hf_home> <serve_args>
#   wait_healthy <port> [timeout_s=1500]
#   stop_server  <container>

start_server() {
  local container=$1 port=$2 image=$3 model=$4 hf_home=$5 serve_args=$6
  echo "--- :rocket: starting vllm: $model"
  # shellcheck disable=SC2086  # serve_args intentionally word-split
  docker run -d --rm --name "$container" \
    --gpus all --ipc=host -p "${port}:${port}" \
    -v "${hf_home}:${hf_home}" \
    -e "HF_HOME=${hf_home}" \
    "$image" \
    vllm serve "$model" --port "$port" $serve_args
}

wait_healthy() {
  local port=$1 timeout=${2:-1500}
  echo "--- :hourglass: waiting for /health (timeout ${timeout}s)"
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
  docker logs --tail 200 "$container" 2>&1 || true
  docker rm -f "$container" >/dev/null 2>&1 || true
}
