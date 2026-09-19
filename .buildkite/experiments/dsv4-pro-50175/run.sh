#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$(pwd)/results/dsv4-pro-50175"
mkdir -p "$RESULTS"
[[ "${BUILDKITE_AGENT_NAME:-}" == h200-ci-1-1 ]]
[[ "$(hostname)" == h200-ci-1 ]]

BASE_COMMIT=7c5dc571cbd1064ecc8a9b1045637ff647aa22cb
BASE_DIGEST=sha256:312219a8b02e951a741629d25ab9344da454b9c81a1c23968eabc3e60b311ee4
BASE_IMAGE="936637512419.dkr.ecr.us-west-2.amazonaws.com/vllm-ci-pull-through-cache/q9t5s3a7/vllm-release-repo:${BASE_COMMIT}-x86_64"
MODEL=deepseek-ai/DeepSeek-V4-Pro
MODEL_REV=b5968e9190ef611bbf34a7229255be88a0e937c1
HF_CACHE=/mnt/shared/hf-models
TOKENIZER="${HF_CACHE}/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/${MODEL_REV}"
PORT=38175
PREFIX="dsv4-ab-${BUILDKITE_JOB_ID}"
CLIENT="${PREFIX}-client"
SERVER="${PREFIX}-server"
LOG_PID=""
MONITOR_PID=""

stop_server() {
  docker rm -f "$SERVER" >/dev/null 2>&1 || true
  if [[ -n "$LOG_PID" ]]; then kill "$LOG_PID" 2>/dev/null || true; wait "$LOG_PID" 2>/dev/null || true; LOG_PID=""; fi
  if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; MONITOR_PID=""; fi
}
cleanup() {
  stop_server
  docker rm -f "$CLIENT" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

assert_idle() {
  local processes
  processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [[ -n "$processes" ]]; then
    echo "GPU processes are already active; refusing to interfere: $processes" >&2
    return 1
  fi
}
assert_idle
if ss -ltnH | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "Experiment port $PORT is occupied" >&2; exit 1
fi
nvidia-smi -q > "$RESULTS/gpu-before.txt"
nvidia-smi topo -m > "$RESULTS/gpu-topology.txt"
cp "$DIR/manifest.json" "$RESULTS/manifest.json"
echo "--- Pull and verify common image"
docker pull "$BASE_IMAGE"
docker image inspect --format '{{json .RepoDigests}}' "$BASE_IMAGE" | tee "$RESULTS/base-digests.json" | grep -q "$BASE_DIGEST"
BASE_ID=$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")
printf '%s\n' "$BASE_ID" > "$RESULTS/base-image-id.txt"
docker tag "$BASE_ID" "${PREFIX}:base"
for arm in client A B; do
  docker build --build-arg "BASE_IMAGE=${PREFIX}:base" --build-arg "ARM=$arm" \
    -t "${PREFIX}:${arm}" -f "$DIR/Dockerfile" "$DIR"
  docker run --rm --entrypoint cat "${PREFIX}:${arm}" /opt/dsv4-ab/identity.json > "$RESULTS/identity-${arm}.json"
done
docker run -d --name "$CLIENT" --network host \
  -v "$HF_CACHE:$HF_CACHE" -v "$RESULTS:/results" -e "HF_HOME=$HF_CACHE" \
  --entrypoint sleep "${PREFIX}:client" infinity
docker exec "$CLIENT" /opt/dsv4-ab/.venv/bin/python -c '
import json
from pathlib import Path
identities=[json.loads(Path("/results/identity-"+arm+".json").read_text()) for arm in ("client","A","B")]
assert all(i["packages"] == identities[0]["packages"] for i in identities)
assert all(i["native_extensions"] == identities[0]["native_extensions"] for i in identities)
print("Verified identical dependency versions and native-extension hashes")
'

for session in A1 B1 B2 A2; do
  arm=${session:0:1}
  mkdir -p "$RESULTS/$session"
  for ((attempt=0; attempt<60; attempt++)); do
    if assert_idle; then break; fi
    sleep 1
  done
  assert_idle
  echo "+++ Session $session: fresh server, five repetitions"
  docker run -d --name "$SERVER" --gpus all --network host --ipc=host \
    --ulimit nofile=65536:65536 -v "$HF_CACHE:$HF_CACHE" \
    -e "HF_HOME=$HF_CACHE" -e VLLM_DEEP_GEMM_WARMUP=skip \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    --entrypoint /opt/dsv4-ab/.venv/bin/python "${PREFIX}:${arm}" \
    -m vllm.entrypoints.cli.main serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --revision "$MODEL_REV" --tokenizer-revision "$MODEL_REV" \
    --data-parallel-size 8 --enable-expert-parallel --max-model-len 32768 \
    --max-num-seqs 512 --max-num-batched-tokens 512 --kv-cache-dtype fp8 \
    --block-size 256 --tokenizer-mode deepseek_v4 --gpu-memory-utilization 0.95 \
    --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice --trust-remote-code --no-enable-flashinfer-autotune \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
  docker logs -f "$SERVER" > "$RESULTS/$session/server.log" 2>&1 &
  LOG_PID=$!
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 5 > "$RESULTS/$session/gpu.csv" &
  MONITOR_PID=$!
  ready=0
  for ((attempt=0; attempt<720; attempt++)); do
    if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null; then ready=1; break; fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "$SERVER")" != true ]]; then
      tail -100 "$RESULTS/$session/server.log"; exit 1
    fi
    if (( attempt % 12 == 0 )); then echo "Waiting for session $session startup ($((attempt * 5))s)"; fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { tail -100 "$RESULTS/$session/server.log"; exit 1; }
  curl -fs "http://127.0.0.1:${PORT}/v1/models" > "$RESULTS/$session/models.json"
  for rep in 1 2 3 4 5; do
    echo "--- Session $session repetition $rep/5"
    docker exec "$CLIENT" /opt/dsv4-ab/.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
      --backend openai --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" \
      --tokenizer "$TOKENIZER" --trust-remote-code --dataset-name random \
      --random-input-len 8192 --random-output-len 1024 --ignore-eos \
      --num-prompts 512 --max-concurrency 128 --num-warmups 128 --seed 0 \
      --save-result --save-detailed --result-dir "/results/$session" \
      --result-filename "run-${rep}.json" 2>&1 | tee "$RESULTS/$session/run-${rep}.log"
    docker exec "$CLIENT" /opt/dsv4-ab/.venv/bin/python -c '
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get("completed") == 512 and not d.get("failed",0), d
' "/results/$session/run-${rep}.json"
  done
  stop_server
done
docker exec "$CLIENT" /opt/dsv4-ab/.venv/bin/python /opt/dsv4-ab/summarize.py /results | tee "$RESULTS/summary.log"
nvidia-smi -q > "$RESULTS/gpu-after.txt"
