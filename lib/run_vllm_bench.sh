# Run a single vllm-bench config against the running vLLM server.
# Source this from run.sh.
#
# Usage:
#   run_vllm_bench <container> <port> <model> <name> <backend> <dataset> \
#                  <input_len> <output_len> <num_prompts> <max_concurrency> \
#                  <trust_remote_code> <output_dir>
#
# Uses the standalone vllm-bench Rust CLI from github.com/vllm-project/vllm-bench
# (drop-in for `vllm bench serve` with the same flags and identical result-JSON
# schema). The binary runs on the host and talks HTTP to the served port, so
# the <container> arg is unused — kept for the existing call-site signature.
# The raw JSON lands in "<output_dir>/bench-<name>.json" so ingest_perf.py can
# pick it up.

# Pinned upstream tag for the prebuilt binary; bump as new releases land.
VLLM_BENCH_VERSION="${VLLM_BENCH_VERSION:-v0.1.0}"
VLLM_BENCH_BIN="${VLLM_BENCH_BIN:-}"

ensure_vllm_bench() {
  if [[ -n "$VLLM_BENCH_BIN" && -x "$VLLM_BENCH_BIN" ]]; then
    return
  fi
  if command -v vllm-bench >/dev/null 2>&1; then
    VLLM_BENCH_BIN="$(command -v vllm-bench)"
    return
  fi
  local cache="${HOME}/.cache/perf-eval"
  local bin="${cache}/vllm-bench-${VLLM_BENCH_VERSION}"
  if [[ ! -x "$bin" ]]; then
    mkdir -p "$cache"
    local arch; arch="$(uname -m)"
    local url="https://github.com/vllm-project/vllm-bench/releases/download/${VLLM_BENCH_VERSION}/vllm-bench-${arch}-linux-musl"
    echo "--- :arrow_down: downloading vllm-bench ${VLLM_BENCH_VERSION} (${arch})"
    curl -fsSL "$url" -o "${bin}.tmp"
    chmod +x "${bin}.tmp"
    mv "${bin}.tmp" "$bin"
  fi
  VLLM_BENCH_BIN="$bin"
}

run_vllm_bench() {
  local _container=$1 port=$2 model=$3 name=$4 backend=$5 dataset=$6
  local input_len=$7 output_len=$8 num_prompts=$9 max_concurrency=${10}
  local trust_remote_code=${11} outdir=${12}
  local host_json="${outdir}/bench-${name}.json"

  [[ "$backend" == "-" ]] && backend=""

  if [[ "$dataset" != "random" ]]; then
    echo "unsupported vllm_bench dataset: $dataset" >&2
    return 2
  fi

  ensure_vllm_bench

  echo "--- :stopwatch: vllm-bench ${name} (isl=${input_len} osl=${output_len} conc=${max_concurrency} n=${num_prompts})"
  mkdir -p "$outdir"

  local cmd=("$VLLM_BENCH_BIN")

  if [[ -n "$backend" ]]; then
    cmd+=(--backend "$backend" --base-url "http://127.0.0.1:${port}")
    [[ "$backend" == "openai-chat" ]] && cmd+=(--endpoint /v1/chat/completions)
  else
    cmd+=(--host 127.0.0.1 --port "$port")
  fi

  cmd+=(
    --model "$model"
    --dataset-name random
    --num-prompts "$num_prompts"
    --max-concurrency "$max_concurrency"
    # --ignore-eos forces every request to emit the full output_len; without it
    # the model can stop early on the random prompt and decode throughput collapses.
    --random-input-len "$input_len"
    --random-output-len "$output_len"
    --ignore-eos
    --save-result
    --result-filename "$host_json"
  )
  [[ "$trust_remote_code" == "true" ]] && cmd+=(--trust-remote-code)

  "${cmd[@]}"

  python3 - "$host_json" "$num_prompts" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    result = json.load(f)
def read_int(*keys, default=None):
    for k in keys:
        v = result.get(k)
        if v is not None:
            return int(v)
    if default is not None:
        return default
    raise KeyError(keys[0])
completed = read_int("completed", "successful", "successful_requests")
failed = read_int("failed", "errored", "failed_requests", "num_failed_requests", default=0)
if failed or completed != expected:
    print(f"vllm-bench incomplete: completed={completed} failed={failed} expected={expected}", file=sys.stderr)
    sys.exit(1)
PY
  echo "  saved $host_json"
}
