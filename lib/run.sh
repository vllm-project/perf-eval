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
# shellcheck disable=SC1091
source "$DIR/run_aiperf.sh"
WORKLOAD_EXPORTS="$(python3 "$DIR/parse_workload.py" "$WORKLOAD")"
eval "$WORKLOAD_EXPORTS"
export WORKLOAD_IMAGE WORKLOAD_VLLM_COMMIT WORKLOAD_SERVER_RUNTIME
echo "image: $WORKLOAD_IMAGE  commit: ${WORKLOAD_VLLM_COMMIT:-unknown}"

PORT="${PERF_EVAL_SERVER_PORT:-$(pick_server_port)}"
CONTAINER="perf-eval-${WORKLOAD_NAME}-$$"
RESULTS_DIR="results/${WORKLOAD_NAME}"
BASE_URL="http://localhost:${PORT}"
BENCH_TRUST_REMOTE_CODE=false
if [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code([[:space:]]|$) ]] ||
   [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code=(true|True|1|yes|Yes)([[:space:]]|$) ]]; then
  BENCH_TRUST_REMOTE_CODE=true
fi
mkdir -p "$RESULTS_DIR"

trap 'stop_server "$CONTAINER"' EXIT

start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
             "$WORKLOAD_SERVE_ARGS" "$WORKLOAD_ENV" "$WORKLOAD_SERVER_RUNTIME"
wait_healthy "$PORT" "$WORKLOAD_SERVER_STARTUP_TIMEOUT" "$WORKLOAD_MODEL"

# vllm bench serve runs first so we can validate perf flow without waiting
# on a full lm_eval pass. Each config's raw json lands in
# $RESULTS_DIR/bench-<name>.json and is then transformed and POSTed to the
# perf dashboard ingest endpoint.
BENCH_ASSERTION_STATUS=0
while IFS=$'\t' read -r bname backend dataset isl osl nprompts conc repetitions speed_subset speed_category extra_args encoded_assertions; do
  [[ -z "$bname" ]] && continue
  run_vllm_bench "$CONTAINER" "$PORT" "$WORKLOAD_MODEL" \
                 "$bname" "$backend" "$dataset" "$isl" "$osl" "$nprompts" \
                 "$conc" "$speed_subset" "$speed_category" "$repetitions" \
                 "$extra_args" \
                 "$BENCH_TRUST_REMOTE_CODE" "$RESULTS_DIR"

  # Do not put run_vllm_bench in an if/|| condition: that would disable
  # Bash errexit inside the function and could hide a failed benchmark.
  assertion_status=0
  if [[ -n "$encoded_assertions" && "$encoded_assertions" != "-" ]]; then
    assertion_report="$RESULTS_DIR/assertions-$bname.json"
    python3 "$DIR/check_perf_assertions.py" \
      --assertions-base64 "$encoded_assertions" \
      --result "$RESULTS_DIR/bench-$bname.json" --repetitions "$repetitions" \
      > "$assertion_report" || assertion_status=$?
    cat "$assertion_report"
    # Python crashes also exit 1; only a valid bound-failure report may upload.
    if ((assertion_status == 1)) && ! PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}" \
      python3 -c 'import sys
from check_perf_assertions import is_failure_report
sys.exit(not is_failure_report(*sys.argv[1:]))' \
        "$assertion_report" "$RESULTS_DIR/bench-$bname.json" "$encoded_assertions"; then
      echo "Checker exited 1 without a valid bound-failure report for $bname." >&2
      assertion_status=2
    fi
    if ((assertion_status > BENCH_ASSERTION_STATUS)); then
      BENCH_ASSERTION_STATUS=$assertion_status
    fi
  fi

  # Valid results remain useful even when they exceed the configured bounds.
  # Invalid inputs or an unexpected checker failure must not enter the dashboard.
  if ((assertion_status > 1)); then
    echo "Skipping dashboard upload for $bname: assertion validation failed (exit $assertion_status); results retained in $RESULTS_DIR." >&2
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

# aiperf profile runs (perf, like vllm_bench). Artifacts are uploaded via the
# Buildkite artifact_paths glob; there is no dashboard ingest for aiperf yet.
while IFS=$'\t' read -r aname aargs; do
  [[ -z "$aname" ]] && continue
  run_aiperf "$CONTAINER" "$PORT" "$WORKLOAD_MODEL" \
             "$aname" "$aargs" "$RESULTS_DIR"
done <<< "$WORKLOAD_AIPERF_TSV"

if [[ "${BENCH_ONLY:-}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ ]]; then
  echo "--- :stopwatch: BENCH_ONLY set; skipping lm_eval and bfcl tasks"
  exit "$BENCH_ASSERTION_STATUS"
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
while IFS=$'\t' read -r category num_threads temperature maximum_step_limit max_test_cases; do
  [[ -z "$category" ]] && continue
  echo "--- :phone: bfcl ${category}"
  python3 "$DIR/run_bfcl.py" "$WORKLOAD_MODEL" "$BASE_URL" \
    "$category" "$num_threads" "$temperature" "$RESULTS_DIR" \
    "$maximum_step_limit" "$max_test_cases"

  manifest="${RESULTS_DIR}/.bfcl_ingest/${category}.txt"
  if [[ -f "$manifest" ]]; then
    while IFS= read -r ingest_category; do
      [[ -z "$ingest_category" ]] && continue
      python3 "$DIR/ingest.py" \
        --results-dir "${RESULTS_DIR}/bfcl-${ingest_category}" \
        --workload "$WORKLOAD_NAME" \
        --task "bfcl_${ingest_category}" \
        --no-samples || true
    done < "$manifest"
  else
    python3 "$DIR/ingest.py" \
      --results-dir "${RESULTS_DIR}/bfcl-${category}" \
      --workload "$WORKLOAD_NAME" \
      --task "bfcl_${category}" \
      --no-samples || true
  fi
done <<< "$WORKLOAD_BFCL_TSV"

# Assertion failures must not suppress later evaluation results. Command
# failures above retain their existing immediate-exit behavior via errexit.
exit "$BENCH_ASSERTION_STATUS"
