# Run a single `vllm bench serve` config against the running vLLM container.
# Source this from run.sh.
#
# Usage:
#   run_vllm_bench <container> <port> <model> <name> <backend> <dataset> \
#                  <input_len> <output_len> <num_prompts> <max_concurrency> \
#                  <speed_bench_dataset_subset> <speed_bench_category> \
#                  <trust_remote_code> <output_dir>
#
# Docker runtime invokes `vllm bench serve` inside the vllm/vllm-openai
# container via `docker exec`. Native runtime invokes it directly in the job
# container. The raw JSON lands in "<output_dir>/bench-<name>.json" on the host
# so the perf-ingest helper can pick it up.

SPEED_BENCH_PREPARE_URL="https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py"

prepare_speed_bench_dataset() {
  local container=$1 runtime=$2 subset=$3 data_dir=$4
  local script='set -euo pipefail
data_dir=$1
subset=$2
prepare_url=$3
mkdir -p "$data_dir"
if [[ ! -s "${data_dir}/${subset}.jsonl" ]]; then
  echo "--- :arrow_down: preparing SPEED-Bench ${subset} dataset in ${data_dir}"
  curl -LsSf "$prepare_url" | python3 - --config "$subset" --output_dir "$data_dir"
fi
test -s "${data_dir}/${subset}.jsonl"'

  if [[ "$runtime" == "native" ]]; then
    bash -lc "$script" _ "$data_dir" "$subset" "$SPEED_BENCH_PREPARE_URL"
  else
    docker exec "$container" bash -lc "$script" _ "$data_dir" "$subset" "$SPEED_BENCH_PREPARE_URL"
  fi
}

run_vllm_bench() {
  local container=$1 port=$2 model=$3 name=$4 backend=$5 dataset=$6
  local input_len=$7 output_len=$8 num_prompts=$9 max_concurrency=${10}
  local speed_bench_dataset_subset=${11} speed_bench_category=${12}
  local trust_remote_code=${13} outdir=${14}
  local in_container_json="/tmp/bench-${name}.json"
  local host_json="${outdir}/bench-${name}.json"
  local runtime="${WORKLOAD_SERVER_RUNTIME:-docker}"
  local speed_bench_dataset_dir=""

  [[ "$backend" == "-" ]] && backend=""
  [[ "$speed_bench_dataset_subset" == "-" ]] && speed_bench_dataset_subset=""
  [[ "$speed_bench_category" == "-" ]] && speed_bench_category=""

  echo "--- :stopwatch: vllm bench serve ${name} (dataset=${dataset} isl=${input_len} osl=${output_len} conc=${max_concurrency} n=${num_prompts})"
  mkdir -p "$outdir"

  local cmd=(vllm bench serve)
  if [[ "$runtime" != "native" ]]; then
    cmd=(docker exec "$container" "${cmd[@]}")
  fi
  if [[ -n "$backend" ]]; then
    cmd+=(--backend "$backend" --base-url "http://127.0.0.1:${port}")
    if [[ "$backend" == "openai-chat" ]]; then
      cmd+=(--endpoint /v1/chat/completions)
    fi
  else
    cmd+=(--host 127.0.0.1 --port "$port")
  fi
  cmd+=(
    --model "$model"
    --dataset-name "$dataset"
    --num-prompts "$num_prompts"
    --max-concurrency "$max_concurrency"
  )
  if [[ "$trust_remote_code" == "true" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ "$dataset" == "random" ]]; then
    cmd+=(--random-input-len "$input_len" --random-output-len "$output_len")
  elif [[ "$dataset" == "speed_bench" ]]; then
    [[ -z "$speed_bench_dataset_subset" ]] && speed_bench_dataset_subset="qualitative"
    speed_bench_dataset_dir="${VLLM_SPEED_BENCH_DIR:-/tmp/vllm-speed-bench}"
    prepare_speed_bench_dataset \
      "$container" "$runtime" "$speed_bench_dataset_subset" "$speed_bench_dataset_dir"
    cmd+=(
      --dataset-path "$speed_bench_dataset_dir"
      --speed-bench-output-len "$output_len"
    )
    if [[ -n "$speed_bench_dataset_subset" ]]; then
      cmd+=(--speed-bench-dataset-subset "$speed_bench_dataset_subset")
    fi
  else
    echo "unsupported vllm_bench dataset: $dataset" >&2
    return 2
  fi
  if [[ -n "$speed_bench_category" ]]; then
    cmd+=(--speed-bench-category "$speed_bench_category")
  fi
  if [[ "$runtime" == "native" ]]; then
    cmd+=(--save-result --result-filename "$host_json")
  else
    cmd+=(--save-result --result-filename "$in_container_json")
  fi

  "${cmd[@]}"

  if [[ "$runtime" != "native" ]]; then
    docker cp "${container}:${in_container_json}" "$host_json"
  fi
  python3 - "$host_json" "$num_prompts" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])

with open(path) as f:
    result = json.load(f)


def read_int(*keys, default=None):
    for key in keys:
        value = result.get(key)
        if value is not None:
            return int(value)
    if default is not None:
        return default
    raise KeyError(keys[0])


completed = read_int("completed", "successful", "successful_requests")
failed = read_int("failed", "errored", "failed_requests", "num_failed_requests", default=0)

if failed or completed != expected:
    print(
        f"vllm bench serve incomplete: completed={completed} "
        f"failed={failed} expected={expected}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
  echo "  saved $host_json"
}
