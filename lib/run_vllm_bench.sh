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

run_vllm_bench() {
  local container=$1 port=$2 model=$3 name=$4 backend=$5 dataset=$6
  local input_len=$7 output_len=$8 num_prompts=$9 max_concurrency=${10}
  local speed_bench_dataset_subset=${11} speed_bench_category=${12}
  local trust_remote_code=${13} outdir=${14}
  local in_container_json="/tmp/bench-${name}.json"
  local host_json="${outdir}/bench-${name}.json"
  local runtime="${WORKLOAD_SERVER_RUNTIME:-docker}"

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
    cmd+=(--speed-bench-output-len "$output_len")
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
  echo "  saved $host_json"
}
