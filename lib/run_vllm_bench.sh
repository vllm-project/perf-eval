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

ensure_speed_bench_prepare_deps() {
  if ! python3 - <<'PY'
import importlib.util
import sys

missing = [
    module
    for module in ("datasets", "numpy", "pandas", "tiktoken")
    if importlib.util.find_spec(module) is None
]
if missing:
    print("missing SPEED-Bench prep deps: " + ", ".join(missing))
    sys.exit(1)
PY
  then
    echo "--- :python: installing SPEED-Bench prep dependencies"
    if python3 - <<'PY'
import sys
sys.exit(0 if sys.prefix != sys.base_prefix else 1)
PY
    then
      python3 -m pip install --quiet datasets numpy pandas tiktoken
    else
      PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user --quiet datasets numpy pandas tiktoken
    fi
  fi
}

prepare_speed_bench_dataset_local() {
  local subset=$1 category=$2 data_dir=$3
  ensure_speed_bench_prepare_deps
  mkdir -p "$data_dir"
  if [[ ! -s "${data_dir}/${subset}.jsonl" ]]; then
    echo "--- :arrow_down: preparing SPEED-Bench ${subset} dataset in ${data_dir}"
    python3 - "$SPEED_BENCH_PREPARE_URL" "$subset" "$category" "$data_dir" <<'PY'
import sys
import urllib.request
from pathlib import Path

prepare_url, subset, category, data_dir = sys.argv[1:]
source = urllib.request.urlopen(prepare_url, timeout=60).read()
namespace = {"__name__": "speed_bench_prepare", "__file__": "prepare.py"}
exec(compile(source, "prepare.py", "exec"), namespace)

dataset = namespace["load_dataset"]("nvidia/SPEED-Bench", subset, split="test")
if category:
    dataset = dataset.filter(lambda example: example["category"] == category)
dataset = namespace["_resolve_external_data"](dataset, subset)
dataset = dataset.map(
    lambda example: {
        "messages": [
            {"role": "user", "content": turn}
            for turn in example["turns"]
        ]
    },
    remove_columns=["turns"],
)
Path(data_dir).mkdir(parents=True, exist_ok=True)
dataset.to_json(Path(data_dir) / f"{subset}.jsonl")
PY
  fi
  test -s "${data_dir}/${subset}.jsonl"
}

prepare_speed_bench_dataset() {
  local container=$1 runtime=$2 subset=$3 category=$4 data_dir=$5
  prepare_speed_bench_dataset_local "$subset" "$category" "$data_dir"
  if [[ "$runtime" != "native" ]]; then
    docker exec "$container" mkdir -p "$data_dir"
    docker cp "${data_dir}/." "${container}:${data_dir}/"
  fi
}

ensure_speed_bench_runtime_deps() {
  local container=$1 runtime=$2
  if [[ "$runtime" == "native" ]]; then
    if ! python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pandas") else 1)'; then
      echo "--- :python: installing SPEED-Bench runtime dependencies"
      if python3 - <<'PY'
import sys
sys.exit(0 if sys.prefix != sys.base_prefix else 1)
PY
      then
        python3 -m pip install --quiet pandas
      else
        PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user --quiet pandas
      fi
    fi
  else
    if ! docker exec "$container" python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pandas") else 1)'; then
      echo "--- :docker: installing SPEED-Bench runtime dependencies in vLLM container"
      docker exec "$container" bash -lc '
set -e
PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --quiet pandas ||
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user --quiet pandas
'
    fi
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
    speed_bench_dataset_dir="${VLLM_SPEED_BENCH_DIR:-/tmp/vllm-speed-bench}/${speed_bench_dataset_subset}"
    if [[ -n "$speed_bench_category" ]]; then
      speed_bench_dataset_dir="${speed_bench_dataset_dir}-${speed_bench_category}"
    fi
    prepare_speed_bench_dataset \
      "$container" "$runtime" "$speed_bench_dataset_subset" \
      "$speed_bench_category" "$speed_bench_dataset_dir"
    ensure_speed_bench_runtime_deps "$container" "$runtime"
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
