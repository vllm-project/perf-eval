# Run a single `vllm bench serve` config against the running vLLM container.
# Source this from run.sh.
#
# Usage:
#   run_vllm_bench <container> <base_url> <model> <name> <backend> <dataset> \
#                  <input_len> <output_len> <num_prompts> <max_concurrency> \
#                  <speed_bench_dataset_subset> <speed_bench_category> \
#                  <trust_remote_code> <output_dir>
#
# Docker runtime invokes `vllm bench serve` inside the vllm/vllm-openai
# container via `docker exec`; native runtime invokes it directly. The raw
# JSON lands in "<output_dir>/bench-<name>.json" so ingest_perf.py can pick
# it up.

# vLLM's SpeedBench class expects a local <subset>.jsonl file built by
# NeMo's prepare.py — the bench CLI does not download the dataset itself.
SPEED_BENCH_PREPARE_URL="https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py"

pip_install_quiet() {
  if python3 -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'; then
    python3 -m pip install --quiet "$@"
  else
    PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user --quiet "$@"
  fi
}

resolve_slurm_container_runtime() {
  local runtime=${PERF_EVAL_BENCH_CLIENT_CONTAINER_RUNTIME:-${PERF_EVAL_SLURM_CONTAINER_RUNTIME:-auto}}
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

resolve_slurm_container_workdir() {
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

prepare_speed_bench_dataset() {
  local container=$1 runtime=$2 subset=$3 category=$4 data_dir=$5

  if [[ ! -s "${data_dir}/${subset}.jsonl" ]]; then
    if ! python3 -c 'import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ("datasets","numpy","pandas","tiktoken")) else 1)' 2>/dev/null; then
      echo "--- :python: installing SPEED-Bench prep dependencies"
      pip_install_quiet datasets numpy pandas tiktoken
    fi
    mkdir -p "$data_dir"
    echo "--- :arrow_down: preparing SPEED-Bench ${subset} dataset in ${data_dir}"
    python3 - "$SPEED_BENCH_PREPARE_URL" "$subset" "$category" "$data_dir" <<'PY'
import sys, urllib.request
from pathlib import Path

prepare_url, subset, category, data_dir = sys.argv[1:]
source = urllib.request.urlopen(prepare_url, timeout=60).read()
ns = {"__name__": "speed_bench_prepare", "__file__": "prepare.py"}
exec(compile(source, "prepare.py", "exec"), ns)

dataset = ns["load_dataset"]("nvidia/SPEED-Bench", subset, split="test")
if category:
    dataset = dataset.filter(lambda ex: ex["category"] == category)
dataset = ns["_resolve_external_data"](dataset, subset)
dataset = dataset.map(
    lambda ex: {"messages": [{"role": "user", "content": t} for t in ex["turns"]]},
    remove_columns=["turns"],
)
Path(data_dir).mkdir(parents=True, exist_ok=True)
dataset.to_json(Path(data_dir) / f"{subset}.jsonl")
PY
  fi
  test -s "${data_dir}/${subset}.jsonl"

  # Docker runtime: ship the data into the container and make sure pandas is
  # available there (vLLM's SpeedBench loads the JSONL via pandas).
  if [[ "$runtime" != "native" && "$runtime" != slurm* ]]; then
    docker exec "$container" mkdir -p "$data_dir"
    docker cp "${data_dir}/." "${container}:${data_dir}/"
    if ! docker exec "$container" python3 -c 'import pandas' 2>/dev/null; then
      echo "--- :docker: installing pandas in vLLM container"
      docker exec "$container" bash -lc \
        'PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --quiet pandas \
          || PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user --quiet pandas'
    fi
  fi
}

run_bench_command() {
  local runtime=$1
  shift
  if [[ "$runtime" == slurm* && "${PERF_EVAL_BENCH_CLIENT_RUNTIME:-local}" == "slurm" ]]; then
    local container_runtime
    container_runtime="$(resolve_slurm_container_runtime)"
    local container_workdir
    container_workdir="$(resolve_slurm_container_workdir)"
    local srun_args=(--ntasks=1)
    case "$container_runtime" in
      pyxis|container)
        srun_args+=(--container-image "$WORKLOAD_IMAGE")
        if [[ -n "${PERF_EVAL_SLURM_CONTAINER_MOUNTS:-}" ]]; then
          srun_args+=(--container-mounts "$PERF_EVAL_SLURM_CONTAINER_MOUNTS")
        fi
        if [[ -n "$container_workdir" ]]; then
          srun_args+=(--container-workdir "$container_workdir")
        fi
        if [[ "${PERF_EVAL_SLURM_NO_CONTAINER_REMAP_ROOT:-1}" == "1" ]]; then
          srun_args+=(--no-container-remap-root)
        fi
        ;;
      none|native)
        ;;
      *)
        echo "unsupported PERF_EVAL_BENCH_CLIENT_CONTAINER_RUNTIME=$container_runtime" >&2
        return 2
        ;;
    esac
    if [[ -n "${PERF_EVAL_SLURM_MPI:-pmix}" ]]; then
      srun_args+=(--mpi "${PERF_EVAL_SLURM_MPI:-pmix}")
    fi
    if [[ -n "${PERF_EVAL_BENCH_CLIENT_EXTRA_SRUN_ARGS:-}" ]]; then
      # shellcheck disable=SC2206  # Buildkite env intentionally supplies words.
      extra_srun_args=($PERF_EVAL_BENCH_CLIENT_EXTRA_SRUN_ARGS)
      srun_args+=("${extra_srun_args[@]}")
    fi
    local cmd=""
    for arg in "$@"; do
      cmd+=" $(printf "%q" "$arg")"
    done
    srun "${srun_args[@]}" bash -lc "$cmd"
    return
  fi
  "$@"
}

run_vllm_bench() {
  local container=$1 base_url=$2 model=$3 name=$4 backend=$5 dataset=$6
  local input_len=$7 output_len=$8 num_prompts=$9 max_concurrency=${10}
  local speed_bench_dataset_subset=${11} speed_bench_category=${12}
  local trust_remote_code=${13} outdir=${14}
  local runtime="${WORKLOAD_SERVER_RUNTIME:-docker}"
  local in_container_json="/tmp/bench-${name}.json"
  local host_json="${outdir}/bench-${name}.json"

  [[ "$backend" == "-" ]] && backend=""
  [[ "$speed_bench_dataset_subset" == "-" ]] && speed_bench_dataset_subset=""
  [[ "$speed_bench_category" == "-" ]] && speed_bench_category=""

  echo "--- :stopwatch: vllm bench serve ${name} (dataset=${dataset} isl=${input_len} osl=${output_len} conc=${max_concurrency} n=${num_prompts})"
  mkdir -p "$outdir"

  local cmd=(vllm bench serve)
  if [[ "$runtime" != "native" && "$runtime" != slurm* ]]; then
    cmd=(docker exec "$container" "${cmd[@]}")
  fi

  if [[ -n "$backend" ]]; then
    cmd+=(--backend "$backend" --base-url "$base_url")
    [[ "$backend" == "openai-chat" ]] && cmd+=(--endpoint /v1/chat/completions)
  else
    local host_port="${base_url#http://}"
    host_port="${host_port#https://}"
    host_port="${host_port%%/*}"
    cmd+=(--host "${host_port%:*}" --port "${host_port##*:}")
  fi

  cmd+=(
    --model "$model"
    --dataset-name "$dataset"
    --num-prompts "$num_prompts"
    --max-concurrency "$max_concurrency"
  )
  [[ "$trust_remote_code" == "true" ]] && cmd+=(--trust-remote-code)

  case "$dataset" in
    random)
      # --ignore-eos forces every request to emit the full output_len; without it
      # the model can stop early on the random prompt and decode throughput collapses.
      cmd+=(--random-input-len "$input_len" --random-output-len "$output_len" --ignore-eos)
      ;;
    speed_bench)
      [[ -z "$speed_bench_dataset_subset" ]] && speed_bench_dataset_subset="qualitative"
      local data_dir="${VLLM_SPEED_BENCH_DIR:-/tmp/vllm-speed-bench}/${speed_bench_dataset_subset}"
      [[ -n "$speed_bench_category" ]] && data_dir="${data_dir}-${speed_bench_category}"
      prepare_speed_bench_dataset "$container" "$runtime" \
        "$speed_bench_dataset_subset" "$speed_bench_category" "$data_dir"
      # SPEED-Bench applies the client-side chat template at tokenizer init,
      # which breaks for chat-template-less models — rely on server-side
      # usage accounting instead.
      cmd+=(
        --dataset-path "$data_dir"
        --speed-bench-output-len "$output_len"
        --speed-bench-dataset-subset "$speed_bench_dataset_subset"
        --skip-tokenizer-init
      )
      [[ -n "$speed_bench_category" ]] && cmd+=(--speed-bench-category "$speed_bench_category")
      ;;
    *)
      echo "unsupported vllm_bench dataset: $dataset" >&2
      return 2
      ;;
  esac

  if [[ "$runtime" == "native" || "$runtime" == slurm* ]]; then
    cmd+=(--save-result --result-filename "$host_json")
  else
    cmd+=(--save-result --result-filename "$in_container_json")
  fi

  run_bench_command "$runtime" "${cmd[@]}"

  if [[ "$runtime" != "native" && "$runtime" != slurm* ]]; then
    docker cp "${container}:${in_container_json}" "$host_json"
  fi

  python3 - "$host_json" "$num_prompts" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    result = json.load(f)
def read_int(*keys, default=None):
    for k in keys:
        v = result.get(k)
        if v is not None:
            return int(v)
    if default is not None:
        return default
    raise KeyError(keys[0])
completed = read_int("completed", "successful", "successful_requests")
failed = read_int("failed", "errored", "failed_requests", "num_failed_requests", default=0)
if failed or completed != expected:
    print(f"vllm bench serve incomplete: completed={completed} failed={failed} expected={expected}", file=sys.stderr)
    sys.exit(1)
PY
  echo "  saved $host_json"
}
