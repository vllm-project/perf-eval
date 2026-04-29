#!/usr/bin/env bash
# Orchestrate a workload: bring up vLLM, then dispatch each task to the
# helper script for its type.
#
# Usage: ./lib/run.sh workloads/qwen3_5_h200.yaml
set -euo pipefail

WORKLOAD="${1:?usage: $0 <workload.yaml>}"
[[ -f "$WORKLOAD" ]] || { echo "not found: $WORKLOAD" >&2; exit 2; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/server.sh"
# shellcheck disable=SC1091
source "$DIR/run_lm_eval.sh"
eval "$(python3 "$DIR/parse_workload.py" "$WORKLOAD")"

PORT=8000
CONTAINER="perf-eval-${WORKLOAD_NAME}-$$"
RESULTS_DIR="results/${WORKLOAD_NAME}"
BASE_URL="http://localhost:${PORT}"
mkdir -p "$RESULTS_DIR"

trap 'stop_server "$CONTAINER"' EXIT

start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
             "$WORKLOAD_SERVE_ARGS" "$WORKLOAD_ENV"
wait_healthy "$PORT"

while IFS=$'\t' read -r task fewshot model_args; do
  [[ -z "$task" ]] && continue
  run_lm_eval "$WORKLOAD_MODEL" "$BASE_URL" "$task" "$fewshot" \
              "$model_args" "$RESULTS_DIR"

  python3 "$DIR/ingest.py" \
    --results-dir "${RESULTS_DIR}/${task}" \
    --workload "$WORKLOAD_NAME" \
    --task "$task" \
    ${INGEST_NO_SAMPLES:+--no-samples} || true
done <<< "$WORKLOAD_LM_EVAL_TASKS_TSV"
