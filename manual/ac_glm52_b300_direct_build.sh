#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA=${EXPECTED_SHA:-e7709dcf3ee0696239d8d79a8d8ca5effd41576f}
SOURCE_DIR=${1:?usage: $0 <vllm-source-dir> <evidence-dir> [local-image-tag]}
EVIDENCE_DIR=${2:?usage: $0 <vllm-source-dir> <evidence-dir> [local-image-tag]}
IMAGE_TAG=${3:-local/ac-glm52-pr51425:e7709dcf3}
CUDA_VERSION=13.0.1
UBUNTU_VERSION=22.04
BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
CACHE_REF=inferactinc/dev:buildcache-v1-x86_64-cu13-0-1-ubuntu22-04
BUILDER_NAME=ac-glm52-direct
MOONCAKE_WHEEL_AARCH64=https://vllm-wheels.s3.amazonaws.com/mooncake/mooncake_transfer_engine-0.3.10.post2-0da9dfea3-cp312-cp312-manylinux_2_35_aarch64.whl
MOONCAKE_WHEEL_X86_64=https://vllm-wheels.s3.amazonaws.com/mooncake/mooncake_transfer_engine-0.3.10.post2-0da9dfea3-cp312-cp312-manylinux_2_35_x86_64.whl

mkdir -p "$EVIDENCE_DIR"
EVIDENCE_DIR=$(cd "$EVIDENCE_DIR" && pwd)
SOURCE_DIR=$(cd "$SOURCE_DIR" && pwd)

exec 9>"$EVIDENCE_DIR/build.lock"
if ! flock -n 9; then
  echo "another direct build holds $EVIDENCE_DIR/build.lock" >&2
  exit 1
fi

capture_gpu_probe() {
  date -u +%FT%TZ >"$EVIDENCE_DIR/probe-time-utc.txt"
  hostname >"$EVIDENCE_DIR/hostname.txt"
  uptime >"$EVIDENCE_DIR/uptime.txt"
  timeout 30 nvidia-smi -L >"$EVIDENCE_DIR/nvidia-smi-L.txt"
  nvidia-smi \
    --query-gpu=index,name,uuid,utilization.gpu,memory.used,pstate \
    --format=csv,noheader,nounits >"$EVIDENCE_DIR/gpus-before-build.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"$EVIDENCE_DIR/compute-before-build.csv" || true
  journalctl -k -b --no-pager >"$EVIDENCE_DIR/kernel-since-boot.log" 2>&1 || true
}

capture_gpu_probe

if [[ $(grep -c 'NVIDIA B300' "$EVIDENCE_DIR/nvidia-smi-L.txt") -ne 8 ]]; then
  echo "pre-build gate failed: expected exactly 8 NVIDIA B300 GPUs" >&2
  exit 1
fi
if [[ -s "$EVIDENCE_DIR/compute-before-build.csv" ]]; then
  echo "pre-build gate failed: GPU compute processes are active" >&2
  exit 1
fi
if awk -F, '$4 + 0 != 0 || $5 + 0 > 16 { exit 1 }' \
  "$EVIDENCE_DIR/gpus-before-build.csv"; then
  :
else
  echo "pre-build gate failed: GPU utilization or memory is not idle" >&2
  exit 1
fi
if grep -Eai 'NVRM: Xid|driver rpc error|GPU has fallen off the bus' \
  "$EVIDENCE_DIR/kernel-since-boot.log" >"$EVIDENCE_DIR/kernel-gpu-errors.txt"; then
  echo "pre-build gate failed: kernel GPU errors are present since boot" >&2
  exit 1
else
  : >"$EVIDENCE_DIR/kernel-gpu-errors.txt"
fi

actual_sha=$(git -C "$SOURCE_DIR" rev-parse HEAD)
git -C "$SOURCE_DIR" status --porcelain=v1 >"$EVIDENCE_DIR/source-status.txt"
git -C "$SOURCE_DIR" show -s --format=fuller HEAD >"$EVIDENCE_DIR/source-commit.txt"
git -C "$SOURCE_DIR" diff --stat "$EXPECTED_SHA^" "$EXPECTED_SHA" \
  >"$EVIDENCE_DIR/source-change-stat.txt"
git -C "$SOURCE_DIR" submodule status --recursive \
  >"$EVIDENCE_DIR/source-submodules.txt" 2>&1 || true
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  echo "source gate failed: got $actual_sha, expected $EXPECTED_SHA" >&2
  exit 1
fi
if [[ -s "$EVIDENCE_DIR/source-status.txt" ]]; then
  echo "source gate failed: worktree is dirty" >&2
  exit 1
fi

docker version >"$EVIDENCE_DIR/docker-version.txt"
docker buildx version >"$EVIDENCE_DIR/buildx-version.txt"
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx create --name "$BUILDER_NAME" --driver docker-container >/dev/null
fi
docker buildx inspect "$BUILDER_NAME" --bootstrap \
  >"$EVIDENCE_DIR/buildx-inspect.txt"

use_sccache=0
secret_args=()
if [[ -r "$HOME/.aws/credentials" ]]; then
  use_sccache=1
  secret_args=(--secret "id=aws-credentials,src=$HOME/.aws/credentials")
fi

cache_args=()
if docker buildx imagetools inspect "$CACHE_REF" \
  >"$EVIDENCE_DIR/cache-inspect.txt" 2>&1; then
  cache_args=(--cache-from "type=registry,ref=$CACHE_REF")
fi

build_command=(
  docker buildx build
  --builder "$BUILDER_NAME"
  --build-arg max_jobs=64
  --build-arg "USE_SCCACHE=$use_sccache"
  --build-arg "CUDA_VERSION=$CUDA_VERSION"
  --build-arg INSTALL_KV_CONNECTORS=true
  --build-arg SCCACHE_BUCKET_NAME=inferact-sccache
  --build-arg SCCACHE_REGION_NAME=us-west-2
  --build-arg "BUILD_BASE_IMAGE=$BUILD_BASE_IMAGE"
  --build-arg "MOONCAKE_WHEEL_AARCH64=$MOONCAKE_WHEEL_AARCH64"
  --build-arg "MOONCAKE_WHEEL_X86_64=$MOONCAKE_WHEEL_X86_64"
  "${secret_args[@]}"
  "${cache_args[@]}"
  --load
  --tag "$IMAGE_TAG"
  --target vllm-openai
  --progress plain
  --metadata-file "$EVIDENCE_DIR/build-metadata.json"
  -f "$SOURCE_DIR/docker/Dockerfile"
  "$SOURCE_DIR"
)

printf '%q ' "${build_command[@]}" >"$EVIDENCE_DIR/build-command.sh"
printf '\n' >>"$EVIDENCE_DIR/build-command.sh"
printf 'source_sha=%s\nimage_tag=%s\ncuda_version=%s\nubuntu_version=%s\nuse_sccache=%s\ncache_ref=%s\n' \
  "$actual_sha" "$IMAGE_TAG" "$CUDA_VERSION" "$UBUNTU_VERSION" \
  "$use_sccache" "$CACHE_REF" >"$EVIDENCE_DIR/build-manifest.txt"

"${build_command[@]}" 2>&1 | tee "$EVIDENCE_DIR/build.log"

docker image inspect "$IMAGE_TAG" >"$EVIDENCE_DIR/image-inspect.json"
docker image inspect --format '{{.Id}}' "$IMAGE_TAG" >"$EVIDENCE_DIR/image-id.txt"
docker run --rm --entrypoint python3 "$IMAGE_TAG" -c \
  'import json, platform, torch, transformers, vllm; import flashinfer; print(json.dumps({"python": platform.python_version(), "vllm": vllm.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "flashinfer": flashinfer.__version__}, sort_keys=True))' \
  >"$EVIDENCE_DIR/runtime-versions.json"
if ! grep -q "${EXPECTED_SHA:0:9}" "$EVIDENCE_DIR/runtime-versions.json"; then
  echo "runtime gate failed: vLLM version does not contain ${EXPECTED_SHA:0:9}" >&2
  exit 1
fi
docker run --rm --entrypoint sh "$IMAGE_TAG" -c \
  'cat /usr/local/cuda/version.json 2>/dev/null || true' \
  >"$EVIDENCE_DIR/cuda-version.json"
date -u +%FT%TZ >"$EVIDENCE_DIR/build-complete-time-utc.txt"

echo "image=$IMAGE_TAG"
echo "image_id=$(cat "$EVIDENCE_DIR/image-id.txt")"
echo "evidence=$EVIDENCE_DIR"
