#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: $0 <provenance/manifest.json>}"
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVENANCE_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"
VALIDATED_FIELDS="$(python3 - "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def fail(message):
    raise ValueError(message)


def validate_record(root, label, record, expected_path):
    if not isinstance(record, dict):
        fail(f"{label} record is missing")
    content = record.get("content")
    checksum = record.get("sha256")
    if not isinstance(content, str) or not isinstance(checksum, str):
        fail(f"{label} record is incomplete")
    actual_checksum = hashlib.sha256(content.encode()).hexdigest()
    if actual_checksum != checksum:
        fail(f"{label} checksum mismatch in manifest")
    path = (root / expected_path).resolve()
    if root not in path.parents:
        fail(f"invalid {label} path: {expected_path}")
    try:
        file_content = path.read_text()
    except OSError as error:
        fail(f"cannot read captured {label}: {error}")
    if file_content != content or hashlib.sha256(file_content.encode()).hexdigest() != checksum:
        fail(f"{label} checksum mismatch")


try:
    manifest_path = Path(sys.argv[1]).resolve()
    root = manifest_path.parent
    with manifest_path.open() as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        fail("manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        fail(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    workload = manifest.get("workload") or {}
    workload_path = workload.get("path")
    if not isinstance(workload_path, str) or not workload_path:
        fail("workload path is missing")
    validate_record(root, "workload", workload, workload_path)
    build = manifest.get("build") or {}
    dockerfile = build.get("dockerfile")
    if not isinstance(dockerfile, str) or not dockerfile:
        fail("manifest does not contain a captured Dockerfile")
    if build.get("context") != "empty":
        fail("unsupported build context")
    validate_record(root, "Dockerfile", build.get("dockerfile_record"), dockerfile)
    image = manifest.get("image") or {}
    if not isinstance(image, dict):
        fail("image record is invalid")
    image_id = image.get("id", "")
    if not isinstance(image_id, str):
        fail("recorded image ID is invalid")
    print(workload_path)
    print(dockerfile)
    print(image_id)
except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
    print(f"replay: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
)" || exit $?
readarray -t FIELDS <<< "$VALIDATED_FIELDS"
WORKLOAD_PATH="${FIELDS[0]}"
DOCKERFILE="${FIELDS[1]}"
EXPECTED_IMAGE_ID="${FIELDS[2]}"

REPLAY_DIR="$(mktemp -d)"
trap 'rm -rf "$REPLAY_DIR"' EXIT
IMAGE="perf-eval-replay:$(printf '%s' "$EXPECTED_IMAGE_ID" | sha256sum | cut -c1-12)"
REBUILT_IMAGE_ID="$(python3 "$DIR/provenance.py" build \
  --image "$IMAGE" \
  --dockerfile "$PROVENANCE_DIR/$DOCKERFILE")"
if [[ -n "$EXPECTED_IMAGE_ID" && "$REBUILT_IMAGE_ID" != "$EXPECTED_IMAGE_ID" ]]; then
  echo "rebuilt image ID does not match the recorded image ID" >&2
  echo "recorded: $EXPECTED_IMAGE_ID" >&2
  echo "rebuilt:  $REBUILT_IMAGE_ID" >&2
  exit 2
fi
REPLAY_WORKLOAD="$REPLAY_DIR/workload.yaml"
python3 - "$PROVENANCE_DIR/$WORKLOAD_PATH" "$REPLAY_WORKLOAD" "$IMAGE" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as file:
    workload = yaml.safe_load(file)
workload["vllm"].pop("build", None)
workload["vllm"]["image"] = sys.argv[3]
with open(sys.argv[2], "w") as file:
    yaml.safe_dump(workload, file, sort_keys=False)
PY
PERF_EVAL_PROFILES_FILE="$DIR/gpu_profiles.yaml" bash "$DIR/run.sh" "$REPLAY_WORKLOAD"
