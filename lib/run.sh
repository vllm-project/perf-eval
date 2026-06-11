#!/usr/bin/env bash
# Orchestrate a workload: bring up vLLM, then dispatch each task to the
# helper script for its type.
#
# When vllm.attention_backends is set in the workload YAML, the server is
# started once per backend (with --attention-backend appended to serve_args),
# and the full eval suite runs for each. Results land in
# results/<name>/attn-<backend>/ instead of results/<name>/.
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
# shellcheck disable=SC1091
source "$DIR/run_vllm_bench.sh"
WORKLOAD_EXPORTS="$(python3 "$DIR/parse_workload.py" "$WORKLOAD")"
eval "$WORKLOAD_EXPORTS"
export WORKLOAD_IMAGE WORKLOAD_VLLM_COMMIT WORKLOAD_SERVER_RUNTIME

PORT=8000
BASE_URL="http://localhost:${PORT}"
BENCH_TRUST_REMOTE_CODE=false
if [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code([[:space:]]|$) ]] ||
   [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code=(true|True|1|yes|Yes)([[:space:]]|$) ]]; then
  BENCH_TRUST_REMOTE_CODE=true
fi

# Build the list of attention backends to sweep.  An empty
# WORKLOAD_ATTENTION_BACKENDS means "run once with whatever vLLM picks" and
# we represent that as a single sentinel entry so the loop always executes.
mapfile -t ATTN_BACKENDS <<< "${WORKLOAD_ATTENTION_BACKENDS}"
if [[ "${#ATTN_BACKENDS[@]}" -eq 0 || ( "${#ATTN_BACKENDS[@]}" -eq 1 && -z "${ATTN_BACKENDS[0]}" ) ]]; then
  ATTN_BACKENDS=("default")
fi

for ATTN_BACKEND in "${ATTN_BACKENDS[@]}"; do
  if [[ "$ATTN_BACKEND" == "default" ]]; then
    echo "=== :brain: attention backend: (vLLM default)"
    RESULTS_DIR="results/${WORKLOAD_NAME}"
    EFFECTIVE_SERVE_ARGS="$WORKLOAD_SERVE_ARGS"
  else
    echo "=== :brain: attention backend: ${ATTN_BACKEND}"
    RESULTS_DIR="results/${WORKLOAD_NAME}/attn-${ATTN_BACKEND}"
    # --attention-backend is a vLLM server arg, not an env var.
    EFFECTIVE_SERVE_ARGS="${WORKLOAD_SERVE_ARGS} --attention-backend ${ATTN_BACKEND}"
  fi
  mkdir -p "$RESULTS_DIR"

  CONTAINER="perf-eval-${WORKLOAD_NAME}-${ATTN_BACKEND}-$$"

  trap 'stop_server "$CONTAINER"' EXIT

  start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
               "$EFFECTIVE_SERVE_ARGS" "$WORKLOAD_ENV" "$WORKLOAD_SERVER_RUNTIME"
  wait_healthy "$PORT"

  # vllm bench serve runs first so we can validate perf flow without waiting
  # on a full lm_eval pass. Each config's raw json lands in
  # $RESULTS_DIR/bench-<name>.json and is then transformed and POSTed to the
  # perf dashboard ingest endpoint.
  while IFS=$'\t' read -r bname backend dataset isl osl nprompts conc speed_subset speed_category; do
    [[ -z "$bname" ]] && continue
    run_vllm_bench "$CONTAINER" "$PORT" "$WORKLOAD_MODEL" \
                   "$bname" "$backend" "$dataset" "$isl" "$osl" "$nprompts" \
                   "$conc" "$speed_subset" "$speed_category" \
                   "$BENCH_TRUST_REMOTE_CODE" "$RESULTS_DIR"

    python3 "$DIR/ingest_perf.py" \
      --raw-result "${RESULTS_DIR}/bench-${bname}.json" \
      --device "$WORKLOAD_BENCH_DEVICE" \
      --tp "$WORKLOAD_BENCH_TP" \
      --precision "$WORKLOAD_BENCH_PRECISION" \
      --model "$WORKLOAD_MODEL" \
      --image "$WORKLOAD_IMAGE" \
      --isl "$isl" --osl "$osl" --conc "$conc" || true
  done <<< "$WORKLOAD_VLLM_BENCH_TSV"

  if [[ "${BENCH_ONLY:-}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ ]]; then
    echo "--- :stopwatch: BENCH_ONLY set; skipping lm_eval and bfcl tasks"
    exit 0
  fi

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

  # bfcl function-calling eval
  while IFS=$'\t' read -r category num_threads temperature; do
    [[ -z "$category" ]] && continue
    echo "--- :phone: bfcl ${category}"
    python3 "$DIR/run_bfcl.py" "$WORKLOAD_MODEL" "$BASE_URL" \
      "$category" "$num_threads" "$temperature" "$RESULTS_DIR"

    python3 "$DIR/ingest.py" \
      --results-dir "${RESULTS_DIR}/bfcl-${category}" \
      --workload "$WORKLOAD_NAME" \
      --task "bfcl_${category}" \
      --no-samples || true
  done <<< "$WORKLOAD_BFCL_TSV"

done << "$WORKLOAD_BFCL_TSV"
