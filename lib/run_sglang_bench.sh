# Run a single `python3 -m sglang.bench_serving` config against the running
# SGLang server. Source this from run.sh after run_vllm_bench.sh so the shared
# append_bench_args helper is available.
#
# Usage:
#   run_sglang_bench <container> <port> <model> <name> <backend> <dataset> \
#                    <input_len> <output_len> <num_prompts> <max_concurrency> \
#                    <extra_args_base64> <output_dir>
#
# The raw JSONL lands in "<output_dir>/bench-<name>.jsonl"; the final appended
# row is also copied to "<output_dir>/bench-<name>.json" for ingest_perf.py.

run_sglang_bench() {
  local container=$1 port=$2 model=$3 name=$4 backend=$5 dataset=$6
  local input_len=$7 output_len=$8 num_prompts=$9 max_concurrency=${10}
  local extra_args_base64=${11} outdir=${12}
  local runtime="${WORKLOAD_SERVER_RUNTIME:-docker}"
  local host_jsonl="${outdir}/bench-${name}.jsonl"
  local host_json="${outdir}/bench-${name}.json"
  local in_container_jsonl="/tmp/bench-${name}.jsonl"
  local output_jsonl="$host_jsonl"

  [[ "$backend" == "-" ]] && backend="sglang"

  echo "--- :stopwatch: sglang bench_serving ${name} (dataset=${dataset} isl=${input_len} osl=${output_len} conc=${max_concurrency} n=${num_prompts})"
  mkdir -p "$outdir"
  rm -f "$host_jsonl" "$host_json"
  if [[ "$runtime" != "native" ]]; then
    output_jsonl="$in_container_jsonl"
    docker exec "$container" rm -f "$in_container_jsonl" >/dev/null 2>&1 || true
  fi

  local cmd=(python3 -m sglang.bench_serving)
  [[ "$runtime" != "native" ]] && cmd=(docker exec "$container" "${cmd[@]}")

  cmd+=(
    --backend "$backend"
    --host 127.0.0.1
    --port "$port"
    --model "$model"
    --dataset-name "$dataset"
    --num-prompts "$num_prompts"
    --max-concurrency "$max_concurrency"
  )

  case "$dataset" in
    random)
      cmd+=(
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio 1.0
        --warmup-requests 64
        --flush-cache
      )
      ;;
    *)
      echo "unsupported sglang_bench dataset: $dataset" >&2
      return 2
      ;;
  esac

  append_bench_args "$extra_args_base64" cmd
  cmd+=(--output-file "$output_jsonl")

  "${cmd[@]}"

  [[ "$runtime" != "native" ]] && docker cp "${container}:${in_container_jsonl}" "$host_jsonl"

  python3 - "$host_jsonl" "$host_json" "$num_prompts" <<'PY'
import json
import sys
from pathlib import Path

jsonl_path, json_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [line for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
if not lines:
    print(f"sglang bench_serving produced no JSONL rows: {jsonl_path}", file=sys.stderr)
    sys.exit(1)
result = json.loads(lines[-1])
completed = int(result.get("completed") or 0)
if completed != expected:
    print(
        f"sglang bench_serving incomplete: completed={completed} expected={expected}",
        file=sys.stderr,
    )
    sys.exit(1)
Path(json_path).write_text(json.dumps(result) + "\n")
PY
  echo "  saved $host_json"
}
