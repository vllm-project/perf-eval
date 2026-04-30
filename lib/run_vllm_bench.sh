# Run a single `vllm bench serve` config against the running vLLM container.
# Source this from run.sh.
#
# Usage:
#   run_vllm_bench <container> <port> <model> <name> <dataset> \
#                  <input_len> <output_len> <num_prompts> <max_concurrency> \
#                  <output_dir>
#
# `vllm bench serve` is invoked inside the vllm/vllm-openai container via
# `docker exec` so we don't need vLLM installed on the host. The raw JSON it
# produces is copied back to "<output_dir>/bench-<name>.json" on the host so
# the perf-ingest helper can pick it up.

run_vllm_bench() {
  local container=$1 port=$2 model=$3 name=$4 dataset=$5
  local input_len=$6 output_len=$7 num_prompts=$8 max_concurrency=$9
  local outdir=${10}
  local in_container_json="/tmp/bench-${name}.json"
  local host_json="${outdir}/bench-${name}.json"

  echo "--- :stopwatch: vllm bench serve ${name} (isl=${input_len} osl=${output_len} conc=${max_concurrency} n=${num_prompts})"
  mkdir -p "$outdir"

  # vllm bench serve uses --random-input-len / --random-output-len for the
  # default `random` dataset. Other datasets ignore these flags.
  docker exec "$container" vllm bench serve \
    --host 127.0.0.1 \
    --port "$port" \
    --model "$model" \
    --dataset-name "$dataset" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$max_concurrency" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --save-result \
    --result-filename "$in_container_json"

  docker cp "${container}:${in_container_json}" "$host_json"
  echo "  saved $host_json"
}
