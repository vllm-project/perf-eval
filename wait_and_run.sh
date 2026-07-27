#!/usr/bin/env bash
# Usage: wait_and_run.sh <workload.yaml> [vllm_image]
# Example: wait_and_run.sh workloads/moe_sweep_amd_gpt_oss_120b_mi355x.yaml vllm/vllm-openai-rocm:nightly
set -euo pipefail

WORKLOAD="${1:?usage: $0 <workload.yaml> [vllm_image]}"
VLLM_IMAGE="${2:-${VLLM_IMAGE:-vllm/vllm-openai-rocm:nightly}}"
STEM=$(basename "$WORKLOAD" .yaml)
LOG="/home/sroberts/perf-eval/results/${STEM}-run.log"

THRESHOLD_MiB=500
REQUIRED_CLEAN_S=60
POLL_INTERVAL=15

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Workload: $WORKLOAD"
log "Image:    $VLLM_IMAGE"
log "Log:      $LOG"

log "Polling GPU VRAM every ${POLL_INTERVAL}s — need ${REQUIRED_CLEAN_S}s clean (<${THRESHOLD_MiB} MiB) before launching..."

clean_since=0

while true; do
  max_mib=0
  while IFS= read -r line; do
    bytes=$(echo "$line" | grep -oE '[0-9]+' | tail -1)
    mib=$(( bytes / 1024 / 1024 ))
    (( mib > max_mib )) && max_mib=$mib
  done < <(rocm-smi --showmeminfo vram | grep "Used Memory")

  if (( max_mib < THRESHOLD_MiB )); then
    if (( clean_since == 0 )); then
      clean_since=$(date +%s)
      log "GPUs idle (max ${max_mib} MiB) — starting ${REQUIRED_CLEAN_S}s clean window..."
    else
      elapsed=$(( $(date +%s) - clean_since ))
      log "Still clean (max ${max_mib} MiB) — ${elapsed}s / ${REQUIRED_CLEAN_S}s"
      if (( elapsed >= REQUIRED_CLEAN_S )); then
        log "Clean for ${elapsed}s — launching moe sweep"
        break
      fi
    fi
  else
    if (( clean_since != 0 )); then
      log "VRAM spiked to ${max_mib} MiB — resetting clean window"
      clean_since=0
    else
      log "GPUs busy (max ${max_mib} MiB) — waiting..."
    fi
  fi

  sleep $POLL_INTERVAL
done

cd /home/sroberts/perf-eval
export VLLM_IMAGE
export HF_HUB_CACHE=/shareddata/hf_hub_cache
export WORKLOAD_SERVER_RUNTIME=docker
export PATH=/home/sroberts/perf-eval/.venv/bin:$PATH

mkdir -p results
nohup ./lib/run.sh "$WORKLOAD" > "$LOG" 2>&1 &
log "Launched run.sh (PID $!) — logging to $LOG"
