#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: $0 <provenance/manifest.json>}"
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVENANCE_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"
readarray -t FIELDS < <(python3 - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    manifest = json.load(file)
source = manifest.get("source") or {}
build = manifest.get("build") or {}
print(source.get("repository", ""))
print(source.get("commit", ""))
print(json.dumps(build.get("args") or {}, separators=(",", ":")))
print(source.get("context_subdirectory", "."))
PY
)
REPOSITORY="${FIELDS[0]}"
COMMIT="${FIELDS[1]}"
BUILD_ARGS_JSON="${FIELDS[2]}"
CONTEXT_SUBDIRECTORY="${FIELDS[3]}"

[[ -n "$REPOSITORY" && -n "$COMMIT" ]] || {
  echo "manifest does not contain build source metadata" >&2
  exit 2
}
if [[ "$BUILD_ARGS_JSON" == *'"<redacted>"'* ]]; then
  echo "manifest contains redacted build arguments and cannot be replayed unattended" >&2
  exit 2
fi

REPLAY_DIR="$(mktemp -d)"
trap 'rm -rf "$REPLAY_DIR"' EXIT
git clone --no-checkout "$REPOSITORY" "$REPLAY_DIR/source"
git -C "$REPLAY_DIR/source" checkout --detach "$COMMIT"
BUILD_CONTEXT="$REPLAY_DIR/source"
if [[ "$CONTEXT_SUBDIRECTORY" != "." ]]; then
  BUILD_CONTEXT="${BUILD_CONTEXT}/${CONTEXT_SUBDIRECTORY}"
fi
[[ -d "$BUILD_CONTEXT" ]] || { echo "recorded build context not found: $BUILD_CONTEXT" >&2; exit 2; }
IMAGE="perf-eval-replay:${COMMIT:0:12}"
python3 "$DIR/provenance.py" build \
  --image "$IMAGE" \
  --dockerfile "$PROVENANCE_DIR/docker/Dockerfile" \
  --context "$BUILD_CONTEXT" \
  --args-json "$BUILD_ARGS_JSON" >/dev/null
REPLAY_WORKLOAD="$REPLAY_DIR/workload.yaml"
python3 - "$PROVENANCE_DIR/workload.yaml" "$REPLAY_WORKLOAD" "$IMAGE" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as file:
    workload = yaml.safe_load(file)
workload["vllm"].pop("build", None)
workload["vllm"]["image"] = sys.argv[3]
with open(sys.argv[2], "w") as file:
    yaml.safe_dump(workload, file, sort_keys=False)
PY
"$DIR/run.sh" "$REPLAY_WORKLOAD"
