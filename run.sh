#!/usr/bin/env bash
# Run an lm-eval workload defined by a YAML recipe.
#
# Usage: ./run.sh workloads/qwen3_5_h200.yaml
set -euo pipefail

WORKLOAD="${1:?usage: $0 <workload.yaml>}"
[[ -f "$WORKLOAD" ]] || { echo "not found: $WORKLOAD" >&2; exit 2; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/lib/server.sh"
eval "$(python3 "$DIR/lib/parse_workload.py" "$WORKLOAD")"

PORT=8000
CONTAINER="perf-eval-${WORKLOAD_NAME}-$$"
RESULTS_DIR="results/${WORKLOAD_NAME}"
BASE_URL="http://localhost:${PORT}"
mkdir -p "$RESULTS_DIR"

trap 'stop_server "$CONTAINER"' EXIT

start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
             "$WORKLOAD_HF_HOME" "$WORKLOAD_SERVE_ARGS"
wait_healthy "$PORT"

while IFS=$'\t' read -r task fewshot; do
  [[ -z "$task" ]] && continue
  echo "--- :microscope: lm_eval ${task} (${fewshot}-shot)"
  lm_eval --model local-completions \
    --model_args "model=${WORKLOAD_MODEL},base_url=${BASE_URL}/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=64,timeout=5000,max_length=8192" \
    --tasks "$task" \
    --num_fewshot "$fewshot" \
    --log_samples \
    --output_path "${RESULTS_DIR}/${task}"

  python3 "$DIR/lib/ingest.py" \
    --results-dir "${RESULTS_DIR}/${task}" \
    --workload "$WORKLOAD_NAME" \
    --task "$task" \
    ${INGEST_NO_SAMPLES:+--no-samples} || true
done <<< "$WORKLOAD_TASKS_TSV"
