#!/usr/bin/env bash
# Attention-backend sweep: run the full eval suite once per backend listed in
# vllm.attention_backends in the workload YAML.  Results land in
# results/<name>/attn-<backend>/ for each backend.
#
# Invoked by run.sh via exec when WORKLOAD_ATTENTION_BACKENDS is non-empty.
# Can also be run directly: ./lib/run_attn_sweep.sh workloads/foo.yaml
#
# Usage: ./lib/run_attn_sweep.sh <workload.yaml>

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

mapfile -t ATTN_BACKENDS <<< "$WORKLOAD_ATTENTION_BACKENDS"

for ATTN_BACKEND in "${ATTN_BACKENDS[@]}"; do
  [[ -z "$ATTN_BACKEND" ]] && continue

  if [[ "$ATTN_BACKEND" == "default" ]]; then
    echo "=== :brain: attention backend: (vLLM default)"
    RESULTS_DIR="results/${WORKLOAD_NAME}/attn-default"
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

  if ! start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
                    "$EFFECTIVE_SERVE_ARGS" "$WORKLOAD_ENV" "$WORKLOAD_SERVER_RUNTIME"; then
    echo "^^^ +++ ERROR: start_server failed for backend ${ATTN_BACKEND}; skipping" >&2
    stop_server "$CONTAINER"
    trap - EXIT
    drain_gpu
    continue
  fi

  if ! wait_healthy "$PORT"; then
    echo "^^^ +++ ERROR: vLLM never became healthy for backend ${ATTN_BACKEND}; skipping" >&2
    stop_server "$CONTAINER"
    trap - EXIT
    drain_gpu
    continue
  fi

  if [[ "$ATTN_BACKEND" == "default" ]]; then
    echo "--- :mag: attention backend selected by vLLM:"
    _backend_lines=""
    if [[ "${WORKLOAD_SERVER_RUNTIME:-docker}" == "native" ]]; then
      _backend_lines=$(grep -E "(Overriding with|Using [A-Z_]+ backend)" \
        "${VLLM_LOG_FILE:-/dev/null}" 2>/dev/null) || true
    else
      _backend_lines=$(docker logs "$CONTAINER" 2>&1 \
        | grep -E "(Overriding with|Using [A-Z_]+ backend)") || true
    fi
    if [[ -n "$_backend_lines" ]]; then
      echo "$_backend_lines" | sed 's/^/  /'
      echo "$_backend_lines" > "${RESULTS_DIR}/attn_backend.txt"
    else
      echo "  (backend selection lines not found in log)"
      echo "unknown" > "${RESULTS_DIR}/attn_backend.txt"
    fi
  else
    echo "$ATTN_BACKEND" > "${RESULTS_DIR}/attn_backend.txt"
  fi

  # vllm bench serve runs first so we can validate perf flow without waiting
  # on a full lm_eval pass. Each config's raw json lands in
  # $RESULTS_DIR/bench-<name>.json and is then transformed and POSTed to the
  # perf dashboard ingest endpoint.
  while IFS=$'\t' read -r bname backend dataset isl osl nprompts conc speed_subset speed_category; do
    [[ -z "$bname" ]] && continue
    if ! run_vllm_bench "$CONTAINER" "$PORT" "$WORKLOAD_MODEL" \
                        "$bname" "$backend" "$dataset" "$isl" "$osl" "$nprompts" \
                        "$conc" "$speed_subset" "$speed_category" \
                        "$BENCH_TRUST_REMOTE_CODE" "$RESULTS_DIR"; then
      echo "^^^ +++ ERROR: run_vllm_bench failed for ${bname} (backend ${ATTN_BACKEND}); skipping run" >&2
      continue
    fi

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
    stop_server "$CONTAINER"
    trap - EXIT
    drain_gpu
    continue
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

  stop_server "$CONTAINER"
  trap - EXIT
  drain_gpu
done
