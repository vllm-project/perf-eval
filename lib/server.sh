# Server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime] [engine]
#   wait_healthy <port> [timeout_s=1500]
#   stop_server  <container>
#
# `env` is a newline-separated list of KEY=VALUE pairs. For Docker runtime,
# each value is injected into the container with -e. As a special case, HF_HOME
# is also bind-mounted at the same path inside the container so the model cache
# on the host is visible to the server. For native runtime, values are exported
# before starting the server process in the current job container.
#
# After start_server, logs are streamed to stdout so build output reflects
# server startup progress in real time. The streamer's PID is held in
# $VLLM_LOGS_PID; stop_server kills it.

run_rdma_preflight() {
  local attr cmd gid gid_idx hca ib_port_path netdev

  echo "+++ :mag: SGLang RDMA/NVSHMEM preflight"
  uname -a || true
  nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true
  echo "--- GPU/NIC topology"
  nvidia-smi topo -m || true
  echo "--- /dev/infiniband"
  ls -la /dev/infiniband || true
  for cmd in ibv_devices "ibv_devinfo -l" ibv_devinfo "rdma link show"; do
    echo "--- ${cmd}"
    bash -lc "command -v ${cmd%% *} >/dev/null && ${cmd}" || true
  done
  echo "--- HCA to network-device mapping"
  for hca in /sys/class/infiniband/*; do
    [[ -e "$hca" ]] || continue
    printf "%s: " "$(basename "$hca")"
    for netdev in "$hca"/device/net/*; do
      [[ -e "$netdev" ]] && printf "%s " "$(basename "$netdev")"
    done
    echo
  done
  echo "--- HCA port state and GID table"
  for hca in /sys/class/infiniband/*; do
    [[ -e "$hca" ]] || continue
    for ib_port_path in "$hca"/ports/*; do
      [[ -e "$ib_port_path" ]] || continue
      printf "%s port %s:" "$(basename "$hca")" "$(basename "$ib_port_path")"
      for attr in state phys_state rate link_layer; do
        [[ -r "$ib_port_path/$attr" ]] &&
          printf " %s=%s" "$attr" "$(tr -d '\n' < "$ib_port_path/$attr")"
      done
      echo
      for gid in "$ib_port_path"/gids/*; do
        [[ -r "$gid" ]] || continue
        gid_idx=$(basename "$gid")
        printf "  gid[%s]=%s" "$gid_idx" "$(tr -d '\n' < "$gid")"
        [[ -r "$ib_port_path/gid_attrs/types/$gid_idx" ]] &&
          printf " type=%s" "$(tr -d '\n' < "$ib_port_path/gid_attrs/types/$gid_idx")"
        [[ -r "$ib_port_path/gid_attrs/ndevs/$gid_idx" ]] &&
          printf " netdev=%s" "$(tr -d '\n' < "$ib_port_path/gid_attrs/ndevs/$gid_idx")"
        echo
      done
    done
  done
  echo "--- network addresses"
  ip -brief address || true
  echo "--- transport environment"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|NCCL_|NVSHMEM_)' | sort || true
}

start_server() {
  local container=$1 port=$2 image=$3 model=$4 serve_args=$5 env=$6 runtime=${7:-docker} engine=${8:-vllm}
  echo "--- :rocket: starting ${engine}: $model"

  if [[ "$runtime" == "native" ]]; then
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      export "$kv"
    done <<< "$env"
    if [[ "$engine" == "sglang" &&
          "${SGLANG_RDMA_PREFLIGHT:-0}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ ]]; then
      run_rdma_preflight
    fi
    local log_file="/tmp/${container}.log"
    VLLM_LOG_FILE="$log_file"
    case "$engine" in
      vllm)
        # shellcheck disable=SC2086  # serve_args intentionally word-split
        vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
        ;;
      sglang)
        # shellcheck disable=SC2086  # serve_args intentionally word-split
        python3 -m sglang.launch_server --model-path "$model" \
          --host 0.0.0.0 --port "$port" $serve_args >"$log_file" 2>&1 &
        ;;
      *)
        echo "unsupported server engine for native runtime: $engine" >&2
        return 2
        ;;
    esac
    VLLM_SERVER_PID=$!
    echo "--- :memo: streaming ${engine} logs"
    ( tail -f "$log_file" 2>/dev/null | stdbuf -oL -eL sed "s/^/[${engine}] /" ) &
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

  case "$engine" in
    vllm)
      # shellcheck disable=SC2086  # serve_args intentionally word-split
      # vllm/vllm-openai's entrypoint takes the model as the first positional
      # arg; do not prepend `vllm` or `serve`.
      docker run -d --rm --name "$container" "${docker_args[@]}" \
        "$image" \
        "$model" --port "$port" $serve_args

      # Install pytest to avoid cupy.testing import failure during torch.compile
      docker exec "$container" pip install -q pytest 2>/dev/null || true
      ;;
    sglang)
      # shellcheck disable=SC2086  # serve_args intentionally word-split
      docker run -d --rm --name "$container" "${docker_args[@]}" \
        "$image" \
        python3 -m sglang.launch_server --model-path "$model" \
          --host 0.0.0.0 --port "$port" $serve_args
      ;;
    *)
      echo "unsupported server engine for docker runtime: $engine" >&2
      return 2
      ;;
  esac

  echo "--- :memo: streaming ${engine} logs"
  ( docker logs -f "$container" 2>&1 | stdbuf -oL -eL sed "s/^/[${engine}] /" ) &
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
      echo "server exited before becoming healthy" >&2
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
    wait "$VLLM_LOGS_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}
