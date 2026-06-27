#!/usr/bin/env bash
# Submit a long-running vLLM server job to Slurm and print the Slurm job id.
#
# Inputs are supplied by lib/server.sh through PERF_EVAL_* environment vars.
# The default path launches non-Ray `vllm serve` through Slurm. If Pyxis is
# available it can run in a Slurm container; clusters with their own launch
# wrapper can use PERF_EVAL_SLURM_SERVER_COMMAND.
set -euo pipefail

require() {
  local name=$1
  local value=${!name:-}
  [[ -n "$value" ]] || { echo "missing required env $name" >&2; exit 2; }
}

shell_quote() {
  printf "%q" "$1"
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_container_runtime() {
  local runtime=${PERF_EVAL_SLURM_CONTAINER_RUNTIME:-auto}
  if [[ "$runtime" != "auto" ]]; then
    printf '%s\n' "$runtime"
    return
  fi
  if srun --help 2>&1 | grep -q -- "--container-image"; then
    printf 'pyxis\n'
  else
    printf 'none\n'
  fi
}

resolve_container_workdir() {
  if [[ -n "${PERF_EVAL_SLURM_CONTAINER_WORKDIR:-}" ]]; then
    printf '%s\n' "$PERF_EVAL_SLURM_CONTAINER_WORKDIR"
    return
  fi
  [[ -n "${PERF_EVAL_SLURM_CONTAINER_MOUNTS:-}" ]] || return

  local pwd_real
  pwd_real="$(pwd -P)"
  local mount host_path container_path rest
  IFS=',' read -ra mounts <<< "$PERF_EVAL_SLURM_CONTAINER_MOUNTS"
  for mount in "${mounts[@]}"; do
    host_path="${mount%%:*}"
    rest="${mount#*:}"
    [[ "$rest" != "$mount" ]] || continue
    container_path="${rest%%:*}"
    [[ -n "$host_path" && -n "$container_path" ]] || continue
    if [[ "$pwd_real" == "$host_path" ]]; then
      printf '%s\n' "$container_path"
      return
    fi
    if [[ "$pwd_real" == "$host_path"/* ]]; then
      printf '%s%s\n' "$container_path" "${pwd_real#"$host_path"}"
      return
    fi
  done
}

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require PERF_EVAL_CONTAINER_NAME
require PERF_EVAL_PORT
require PERF_EVAL_MODEL
require PERF_EVAL_ENDPOINT_FILE
require PERF_EVAL_RESULTS_DIR

mkdir -p "$PERF_EVAL_RESULTS_DIR"

script="${PERF_EVAL_RESULTS_DIR}/${PERF_EVAL_CONTAINER_NAME}.sbatch.sh"
log_file="${PERF_EVAL_RESULTS_DIR}/${PERF_EVAL_CONTAINER_NAME}-%j.log"
nodes="${PERF_EVAL_NUM_NODES:-1}"
gpus_per_node="${PERF_EVAL_GPUS_PER_NODE:-${PERF_EVAL_NUM_GPUS:-1}}"
slurm_gpus_per_node="${PERF_EVAL_SLURM_GPUS_PER_NODE:-$gpus_per_node}"
ntasks_per_node="${PERF_EVAL_SLURM_NTASKS_PER_NODE:-1}"
job_name="${PERF_EVAL_SLURM_JOB_NAME:-$PERF_EVAL_CONTAINER_NAME}"
container_runtime="$(resolve_container_runtime)"
request_gpus="${PERF_EVAL_SLURM_REQUEST_GPUS:-1}"
slurm_gres="${PERF_EVAL_SLURM_GRES:-}"
container_workdir="$(resolve_container_workdir)"
vllm_distributed_backend="${PERF_EVAL_SLURM_VLLM_DISTRIBUTED_BACKEND:-}"
if [[ -z "$vllm_distributed_backend" ]] && is_integer "$nodes" && (( nodes > 1 )); then
  vllm_distributed_backend=mp
fi

serve_cmd="vllm serve $(shell_quote "$PERF_EVAL_MODEL") --host 0.0.0.0 --port $(shell_quote "$PERF_EVAL_PORT")"
if [[ -n "${PERF_EVAL_SERVE_ARGS:-}" ]]; then
  serve_cmd+=" ${PERF_EVAL_SERVE_ARGS}"
fi

server_command="${PERF_EVAL_SLURM_SERVER_COMMAND:-}"
server_command_block=""
if [[ -z "$server_command" ]]; then
  srun_common_args=()
  if is_truthy "$request_gpus"; then
    if [[ -n "$slurm_gres" ]]; then
      srun_common_args+=(--gres "$slurm_gres")
    elif [[ -n "$slurm_gpus_per_node" ]]; then
      srun_common_args+=(--gpus-per-node "$slurm_gpus_per_node")
    fi
  fi
  case "$container_runtime" in
    pyxis|container)
      require PERF_EVAL_IMAGE
      srun_common_args+=(--container-image "$PERF_EVAL_IMAGE")
      if [[ -n "${PERF_EVAL_SLURM_CONTAINER_MOUNTS:-}" ]]; then
        srun_common_args+=(--container-mounts "$PERF_EVAL_SLURM_CONTAINER_MOUNTS")
      fi
      if [[ -n "$container_workdir" ]]; then
        srun_common_args+=(--container-workdir "$container_workdir")
      fi
      if [[ "${PERF_EVAL_SLURM_NO_CONTAINER_REMAP_ROOT:-1}" == "1" ]]; then
        srun_common_args+=(--no-container-remap-root)
      fi
      ;;
    none|native)
      ;;
    *)
      echo "unsupported PERF_EVAL_SLURM_CONTAINER_RUNTIME=$container_runtime" >&2
      exit 2
      ;;
  esac
  if [[ -n "${PERF_EVAL_SLURM_MPI:-pmix}" ]]; then
    srun_common_args+=(--mpi "${PERF_EVAL_SLURM_MPI:-pmix}")
  fi
  if [[ -n "${PERF_EVAL_SLURM_EXTRA_SRUN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # Buildkite env intentionally supplies words.
    extra_srun_args=($PERF_EVAL_SLURM_EXTRA_SRUN_ARGS)
    srun_common_args+=("${extra_srun_args[@]}")
  fi

  srun_common=""
  for arg in "${srun_common_args[@]}"; do
    srun_common+=" $(shell_quote "$arg")"
  done

  if [[ "$vllm_distributed_backend" == "mp" ]] && is_integer "$nodes" && (( nodes > 1 )); then
    server_command_block=$(cat <<BLOCK
server_pids=()
cleanup_servers() {
  for pid in "\${server_pids[@]:-}"; do
    kill "\$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup_servers EXIT INT TERM

if ((\${#perf_eval_slurm_hosts[@]} < $(shell_quote "$nodes"))); then
  echo "expected $(shell_quote "$nodes") Slurm hosts, got \${#perf_eval_slurm_hosts[@]}" >&2
  exit 2
fi
master_addr="\${perf_eval_slurm_hosts[0]}"
base_serve_cmd=$(shell_quote "$serve_cmd")
for rank in \$(seq 0 $((nodes - 1))); do
  node="\${perf_eval_slurm_hosts[\$rank]}"
  rank_cmd="\$base_serve_cmd --distributed-executor-backend mp --nnodes $(shell_quote "$nodes") --master-addr \$master_addr --node-rank \$rank"
  if ((rank > 0)); then
    rank_cmd+=" --headless"
  fi
  srun --nodes=1 --ntasks=1 --nodelist "\$node"$srun_common bash -lc "\$rank_cmd" &
  server_pids+=("\$!")
done

wait -n "\${server_pids[@]}"
status=\$?
cleanup_servers
wait "\${server_pids[@]}" >/dev/null 2>&1 || true
exit "\$status"
BLOCK
)
  else
    server_command="srun"
    server_command+=" --ntasks=1"
    server_command+="$srun_common"
    server_command+=" bash -lc $(shell_quote "$serve_cmd")"
  fi
fi

cat > "$script" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

env_file=$(shell_quote "${PERF_EVAL_ENV_FILE:-}")
if [[ -n "\$env_file" && -f "\$env_file" ]]; then
  while IFS= read -r kv; do
    [[ -z "\$kv" ]] && continue
    export "\$kv"
  done < "\$env_file"
fi

mapfile -t perf_eval_slurm_hosts < <(scontrol show hostnames "\${SLURM_JOB_NODELIST:-}" 2>/dev/null || true)
host=\$(hostname -f 2>/dev/null || hostname)
if [[ "$(shell_quote "$vllm_distributed_backend")" == "mp" && "${nodes}" != "1" && "\${#perf_eval_slurm_hosts[@]}" -gt 0 ]]; then
  host="\${perf_eval_slurm_hosts[0]}"
fi
log_path=$(shell_quote "$log_file")
if [[ -n "\${SLURM_JOB_ID:-}" ]]; then
  log_path="\${log_path//%j/\$SLURM_JOB_ID}"
fi
tmp="$(shell_quote "$PERF_EVAL_ENDPOINT_FILE").tmp"
{
  printf 'VLLM_SERVER_HOST=%q\\n' "\$host"
  printf 'VLLM_SERVER_PORT=%q\\n' "$(shell_quote "$PERF_EVAL_PORT")"
  printf 'VLLM_BASE_URL=%q\\n' "http://\${host}:$(shell_quote "$PERF_EVAL_PORT")"
  printf 'VLLM_SLURM_LOG_FILE=%q\\n' "\$log_path"
} > "\$tmp"
mv "\$tmp" "$(shell_quote "$PERF_EVAL_ENDPOINT_FILE")"

${server_command_block:-exec $server_command}
SCRIPT
chmod +x "$script"

sbatch_args=(--parsable --job-name "$job_name" --nodes "$nodes" \
  --ntasks-per-node "$ntasks_per_node" --output "$log_file" --error "$log_file")
if is_truthy "$request_gpus"; then
  if [[ -n "$slurm_gres" ]]; then
    sbatch_args+=(--gres "$slurm_gres")
  elif [[ -n "$slurm_gpus_per_node" ]]; then
    sbatch_args+=(--gpus-per-node "$slurm_gpus_per_node")
  fi
fi
if [[ -n "${PERF_EVAL_SLURM_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account "$PERF_EVAL_SLURM_ACCOUNT")
fi
if [[ -n "${PERF_EVAL_SLURM_PARTITION:-}" ]]; then
  sbatch_args+=(--partition "$PERF_EVAL_SLURM_PARTITION")
fi
if [[ -n "${PERF_EVAL_SLURM_QOS:-}" ]]; then
  sbatch_args+=(--qos "$PERF_EVAL_SLURM_QOS")
fi
if [[ -n "${PERF_EVAL_SLURM_RESERVATION:-}" ]]; then
  sbatch_args+=(--reservation "$PERF_EVAL_SLURM_RESERVATION")
fi
if [[ -n "${PERF_EVAL_SLURM_TIME:-}" ]]; then
  sbatch_args+=(--time "$PERF_EVAL_SLURM_TIME")
fi
if [[ -n "${PERF_EVAL_SLURM_EXTRA_SBATCH_ARGS:-}" ]]; then
  # shellcheck disable=SC2206  # Buildkite env intentionally supplies words.
  extra_sbatch_args=($PERF_EVAL_SLURM_EXTRA_SBATCH_ARGS)
  sbatch_args+=("${extra_sbatch_args[@]}")
fi

job_id="$(sbatch "${sbatch_args[@]}" "$script")"
job_id="${job_id%%;*}"
printf '%s\n' "$job_id"
