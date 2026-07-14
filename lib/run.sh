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
# shellcheck disable=SC1091
source "$DIR/run_vllm_bench.sh"
WORKLOAD_EXPORTS="$(python3 "$DIR/parse_workload.py" "$WORKLOAD")"
eval "$WORKLOAD_EXPORTS"
WORKLOAD_SERVING_MODE="${WORKLOAD_SERVING_MODE:-standalone}"
export WORKLOAD_IMAGE WORKLOAD_VLLM_COMMIT WORKLOAD_SERVER_RUNTIME
export WORKLOAD_SERVING_MODE WORKLOAD_SERVING_JSON

PORT=8000
CONTAINER="perf-eval-${WORKLOAD_NAME}-$$"
RESULTS_DIR="results/${WORKLOAD_NAME}"
BASE_URL="http://localhost:${PORT}"
WORKLOAD_BENCH_BASE_URL="$BASE_URL"
BENCH_TRUST_REMOTE_CODE=false
if [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code([[:space:]]|$) ]] ||
   [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code=(true|True|1|yes|Yes)([[:space:]]|$) ]]; then
  BENCH_TRUST_REMOTE_CODE=true
fi
mkdir -p "$RESULTS_DIR"

cleanup() {
  local status=$?
  set +e
  if [[ "$WORKLOAD_SERVING_MODE" == "pd_disagg" ]]; then
    local state_file="${PD_SERVING_STATE_FILE:-}"
    local launcher="${PD_LAUNCHER:-$DIR/slurm_pd_launcher.py}"
    local stopped=false
    if [[ -n "$state_file" && -f "$state_file" ]]; then
      if python3 "$launcher" --state-file "$state_file" stop; then
        stopped=true
      fi
    fi
    if [[ -n "${PD_SUPERVISOR_PID:-}" ]]; then
      if [[ "$stopped" != true ]]; then
        kill "$PD_SUPERVISOR_PID" 2>/dev/null || true
      fi
      wait "$PD_SUPERVISOR_PID" 2>/dev/null || true
    fi
  else
    stop_server "$CONTAINER"
  fi
  return "$status"
}
trap cleanup EXIT

if [[ "$WORKLOAD_SERVING_MODE" == "pd_disagg" ]]; then
  [[ "$WORKLOAD_SERVER_RUNTIME" == "slurm" ]] || {
    echo "pd_disagg requires a GPU profile with server_runtime: slurm" >&2
    exit 2
  }
  PD_LAUNCHER="$DIR/slurm_pd_launcher.py"
  PD_SERVING_STATE_FILE="${RESULTS_DIR}/pd-serving-state.json"
  export PD_LAUNCHER PD_SERVING_STATE_FILE WORKLOAD_BENCH_BASE_URL
  rm -f "$PD_SERVING_STATE_FILE"

  read -r ROUTER_CLIENT_HOST PORT < <(
    python3 -c 'import json,sys; r=json.loads(sys.argv[1])["router"]; print(r["client_host"], r["port"])' \
      "$WORKLOAD_SERVING_JSON"
  )
  BASE_URL="http://${ROUTER_CLIENT_HOST}:${PORT}"
  # exec-client substitutes this with the login node's routable address from
  # state before running vllm bench inside the allocation.
  WORKLOAD_BENCH_BASE_URL='{router_url}'
  export WORKLOAD_BENCH_BASE_URL

  if python3 -c 'import json,sys; c=json.loads(sys.argv[1]); a=c["common_argv"] + sum((r["serve_argv"] for r in c["roles"]), []); sys.exit(0 if "--trust-remote-code" in a else 1)' \
      "$WORKLOAD_SERVING_JSON"; then
    BENCH_TRUST_REMOTE_CODE=true
  fi

  echo "--- :rocket: starting Slurm prefill/decode serving"
  python3 "$PD_LAUNCHER" \
    --config "$WORKLOAD_SERVING_JSON" \
    --state-file "$PD_SERVING_STATE_FILE" \
    supervise &
  PD_SUPERVISOR_PID=$!
  export PD_SUPERVISOR_PID
  python3 "$PD_LAUNCHER" \
    --state-file "$PD_SERVING_STATE_FILE" \
    wait-ready --timeout 3600 --supervisor-pid "$PD_SUPERVISOR_PID"
  kill -0 "$PD_SUPERVISOR_PID" 2>/dev/null || {
    echo "PD supervisor exited immediately after readiness" >&2
    wait "$PD_SUPERVISOR_PID" || true
    exit 1
  }
else
  start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
               "$WORKLOAD_SERVE_ARGS" "$WORKLOAD_ENV" "$WORKLOAD_SERVER_RUNTIME"
  wait_healthy "$PORT"
fi

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

  ingest_args=(
    --raw-result "${RESULTS_DIR}/bench-${bname}.json"
    --device "$WORKLOAD_BENCH_DEVICE"
    --tp "$WORKLOAD_BENCH_TP"
    --gpu-count "$WORKLOAD_BENCH_GPU_COUNT"
    --precision "$WORKLOAD_BENCH_PRECISION"
    --model "$WORKLOAD_MODEL"
    --image "$WORKLOAD_IMAGE"
    --isl "$isl" --osl "$osl" --conc "$conc"
  )
  [[ "$WORKLOAD_BENCH_DISAGG" == "true" ]] && ingest_args+=(--disagg)
  [[ "$WORKLOAD_BENCH_IS_MULTINODE" == "true" ]] && ingest_args+=(--is-multinode)
  python3 "$DIR/ingest_perf.py" "${ingest_args[@]}" || true
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
