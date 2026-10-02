# shellcheck shell=bash
# Nsight Systems profiling of the vLLM server. Source this from run.sh.
#
# Functions:
#   nsys_configure  <runtime> <results_dir> <report_name>
#   run_nsys_profile <container> <port> <model> <bench_tsv_row> \
#                    <trust_remote_code> <results_dir>
#   nsys_finalize   <container>
#
# nsys_configure sets NSYS_REPORT, which start_server (server.sh) uses to run
# `vllm serve` under lib/nsys_serve.sh, and appends the serve args/env vLLM
# needs to hand the capture range to nsys. Nothing is recorded until
# run_nsys_profile issues one `vllm bench serve --profile`, so the measured
# benchmarks before it run untraced. The report is copied to
# <results_dir>/nsys/<report_name>.nsys-rep, which the step's artifact_paths
# glob uploads to Buildkite.
#
# Tunables (env):
#   NSYS_MAX_ITERATIONS    engine iterations to record (default 100; 0 = all)
#   NSYS_DELAY_ITERATIONS  engine iterations to skip first (default 0)
#   NSYS_REPORT_TIMEOUT_S  wait for nsys to write the report (default 900)

NSYS_REPORT=""
NSYS_HOST_DIR=""
NSYS_COLLECTED=false

nsys_configure() {
  local runtime=$1 results_dir=$2 name=$3
  NSYS_HOST_DIR="${results_dir}/nsys"
  mkdir -p "$NSYS_HOST_DIR"
  if [[ "$runtime" == "native" ]]; then
    NSYS_REPORT="$(cd "$NSYS_HOST_DIR" && pwd)/${name}"
  else
    # Written inside the container, then `docker cp`'d out, so the host copy
    # is owned by the job user rather than the container's root.
    NSYS_REPORT="/tmp/perf-eval-nsys/${name}"
  fi
  WORKLOAD_SERVE_ARGS+=" --profiler-config.profiler cuda"
  WORKLOAD_SERVE_ARGS+=" --profiler-config.max_iterations ${NSYS_MAX_ITERATIONS:-100}"
  WORKLOAD_SERVE_ARGS+=" --profiler-config.delay_iterations ${NSYS_DELAY_ITERATIONS:-0}"
  # nsys can't follow fork()ed CUDA workers reliably; vLLM's profiling docs
  # recommend spawn.
  WORKLOAD_ENV+=$'\n'"VLLM_WORKER_MULTIPROC_METHOD=spawn"
  echo "nsys: profiling enabled; report -> ${NSYS_HOST_DIR}/${name}.nsys-rep"
}

# Run a command where the report lives: in the container, or locally.
_nsys_where() {
  local container=$1; shift
  if [[ "${WORKLOAD_SERVER_RUNTIME:-docker}" == "native" ]]; then
    "$@"
  else
    docker exec "$container" "$@"
  fi
}

# Wait until nsys has finished writing the report: present, non-empty, and
# the same size across consecutive polls.
wait_for_nsys_report() {
  local container=$1 timeout=${NSYS_REPORT_TIMEOUT_S:-900}
  local rep="${NSYS_REPORT}.nsys-rep" deadline size last="" stable=0
  deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if _nsys_where "$container" test -e "${NSYS_REPORT}.unavailable" 2>/dev/null; then
      echo "nsys: was not available in the server environment; no report" >&2
      return 1
    fi
    size=$(_nsys_where "$container" stat -c %s "$rep" 2>/dev/null || true)
    if [[ -n "$size" && "$size" != 0 && "$size" == "$last" ]]; then
      (( ++stable >= 2 )) && return 0
    else
      stable=0
    fi
    last=$size
    sleep 5
  done
  echo "nsys: report not written within ${timeout}s; retrying at shutdown" >&2
  return 1
}

collect_nsys_report() {
  local container=$1
  if [[ "${WORKLOAD_SERVER_RUNTIME:-docker}" != "native" ]]; then
    docker cp "${container}:${NSYS_REPORT}.nsys-rep" "${NSYS_HOST_DIR}/" || return 1
  fi
  [[ -s "${NSYS_HOST_DIR}/$(basename "$NSYS_REPORT").nsys-rep" ]] || return 1
  NSYS_COLLECTED=true
  echo "nsys: saved ${NSYS_HOST_DIR}/$(basename "$NSYS_REPORT").nsys-rep"
}

# One `vllm bench serve --profile` pass using a vllm_bench TSV row (see
# run.sh): repetitions forced to 1 and num_prompts to max_concurrency, so the
# recorded window is a single full-concurrency wave. Failures are reported
# but don't fail the workload — the trace is a diagnostic, not a result.
run_nsys_profile() {
  local container=$1 port=$2 model=$3 row=$4 trust_remote_code=$5 outdir=$6
  local bname backend dataset isl osl conc speed_subset speed_category
  local extra_args status
  # num_prompts and repetitions (the 6th and 8th fields) are overridden.
  IFS=$'\t' read -r bname backend dataset isl osl _ conc _ \
    speed_subset speed_category extra_args <<< "$row"
  extra_args=$(python3 - "$extra_args" <<'PY'
import base64, json, sys
args = json.loads(base64.b64decode(sys.argv[1]))
args["profile"] = True
print(base64.b64encode(json.dumps(args, separators=(",", ":")).encode()).decode())
PY
  )

  echo "--- :mag: nsys profile (vllm bench serve ${bname} --profile)"
  # Subshell outside any `if` so `set -e` still aborts it on the first error.
  set +e
  (
    set -e
    run_vllm_bench "$container" "$port" "$model" "nsys-${bname}" \
                   "$backend" "$dataset" "$isl" "$osl" "$conc" "$conc" \
                   "$speed_subset" "$speed_category" 1 "$extra_args" \
                   "$trust_remote_code" "$outdir"
  )
  status=$?
  set -e
  if (( status != 0 )); then
    echo "nsys: profiled bench run failed (exit ${status}); continuing" >&2
    return 0
  fi
  if wait_for_nsys_report "$container"; then
    collect_nsys_report "$container" || echo "nsys: failed to copy report" >&2
  fi
  return 0
}

# Called on exit, before stop_server. If the report wasn't collected after
# the profiled run, stop nsys with SIGINT (Ctrl-C), which makes it write any
# pending report, and copy that out.
nsys_finalize() {
  local container=$1 timeout=${NSYS_REPORT_TIMEOUT_S:-900}
  [[ -n "$NSYS_REPORT" && "$NSYS_COLLECTED" != true ]] || return 0
  echo "--- :mag: nsys: stopping server to finalize report"
  if [[ "${WORKLOAD_SERVER_RUNTIME:-docker}" == "native" ]]; then
    if [[ -n "${VLLM_SERVER_PID:-}" ]] && kill -INT "$VLLM_SERVER_PID" 2>/dev/null; then
      local start=$SECONDS
      while kill -0 "$VLLM_SERVER_PID" 2>/dev/null && (( SECONDS - start < timeout )); do
        sleep 1
      done
    fi
  else
    # The container runs with --stop-signal SIGINT and without --rm, so it
    # can still be copied from once stopped.
    docker stop -t "$timeout" "$container" >/dev/null 2>&1 || true
  fi
  collect_nsys_report "$container" 2>/dev/null || echo "nsys: no report produced" >&2
}
