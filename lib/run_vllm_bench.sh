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
# container via `docker exec`; native runtime invokes it directly. The raw
# JSON lands in "<output_dir>/bench-<name>.json" so ingest_perf.py can pick
# it up.

run_vllm_bench() {
  local container=$1 port=$2 model=$3 name=$4 backend=$5 dataset=$6
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
      cmd+=(--random-input-len "$input_len" --random-output-len "$output_len")
      ;;
    speed_bench)
      cmd+=(--speed-bench-output-len "$output_len")
      [[ -n "$speed_bench_dataset_subset" ]] && cmd+=(--speed-bench-dataset-subset "$speed_bench_dataset_subset")
      [[ -n "$speed_bench_category" ]] && cmd+=(--speed-bench-category "$speed_bench_category")
      ;;
    *)
      echo "unsupported vllm_bench dataset: $dataset" >&2
      return 2
      ;;
  esac

  local result_path
  if [[ "$runtime" == "native" ]]; then
    result_path="$host_json"
  else
    result_path="$in_container_json"
  fi
  cmd+=(--save-result --result-filename "$result_path")

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
