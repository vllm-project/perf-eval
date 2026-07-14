# Run a single `vllm bench serve` config against the running vLLM container.
# Source this from run.sh.
#
# Usage:
#   run_vllm_bench <container> <port> <model> <name> <backend> <dataset> \
#                  <input_len> <output_len> <num_prompts> <max_concurrency> \
#                  <speed_bench_dataset_subset> <speed_bench_category> \
#                  <extra_args_base64> <trust_remote_code> <output_dir>
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

append_bench_args() {
  local encoded=$1
  local -n command_ref=$2
  local extra_args=()
  mapfile -d '' -t extra_args < <(
    python3 - "$encoded" <<'PY'
import base64
import json
import sys

args = json.loads(base64.b64decode(sys.argv[1]))
output = sys.stdout.buffer


def emit(value):
    output.write(str(value).encode() + b"\0")


for name, value in args.items():
    flag = f"--{name}"
    if value is True:
        emit(flag)
    elif value is False or value is None:
        continue
    elif isinstance(value, list):
        for item in value:
            emit(flag)
            emit(json.dumps(item, separators=(",", ":")) if isinstance(item, (dict, list)) else item)
    else:
        emit(flag)
        emit(json.dumps(value, separators=(",", ":")) if isinstance(value, dict) else value)
PY
  )
  command_ref+=("${extra_args[@]}")
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
  if [[ "$runtime" != "native" ]]; then
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

run_vllm_bench() {
  local container=$1 port=$2 model=$3 name=$4 backend=$5 dataset=$6
  local input_len=$7 output_len=$8 num_prompts=$9 max_concurrency=${10}
  local speed_bench_dataset_subset=${11} speed_bench_category=${12}
  local extra_args_base64=${13} trust_remote_code=${14} outdir=${15}
  local runtime="${WORKLOAD_SERVER_RUNTIME:-docker}"
  local in_container_json="/tmp/bench-${name}.json"
  local host_json="${outdir}/bench-${name}.json"

  [[ "$backend" == "-" ]] && backend=""
  [[ "$speed_bench_dataset_subset" == "-" ]] && speed_bench_dataset_subset=""
  [[ "$speed_bench_category" == "-" ]] && speed_bench_category=""

  echo "--- :stopwatch: vllm bench serve ${name} (dataset=${dataset} isl=${input_len} osl=${output_len} conc=${max_concurrency} n=${num_prompts})"
  mkdir -p "$outdir"

  local cmd=(vllm bench serve)
  [[ "$runtime" != "native" ]] && cmd=(docker exec "$container" "${cmd[@]}")

  if [[ -n "$backend" ]]; then
    cmd+=(--backend "$backend" --base-url "http://127.0.0.1:${port}")
    [[ "$backend" == "openai-chat" ]] && cmd+=(--endpoint /v1/chat/completions)
  else
    cmd+=(--host 127.0.0.1 --port "$port")
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

  append_bench_args "$extra_args_base64" cmd

  if [[ "$runtime" == "native" ]]; then
    cmd+=(--save-result --result-filename "$host_json")
  else
    cmd+=(--save-result --result-filename "$in_container_json")
  fi

  "${cmd[@]}"

  [[ "$runtime" != "native" ]] && docker cp "${container}:${in_container_json}" "$host_json"

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
