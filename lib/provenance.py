#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SENSITIVE_NAME = re.compile(r"(?:token|secret|password|passwd|api[_-]?key|credential)", re.IGNORECASE)


def sanitize_environment(environment: str) -> dict:
    values = {}
    for entry in environment.splitlines():
        if not entry:
            continue
        name, separator, value = entry.partition("=")
        if not separator:
            continue
        values[name] = "<redacted>" if SENSITIVE_NAME.search(name) else value
    return values


def file_record(path: Path) -> dict:
    content = path.read_text()
    return {
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def image_metadata(
    image: str,
    image_id: str = "",
    runtime: str = "docker",
    container: str = "",
) -> dict:
    if runtime != "docker":
        return {"reference": image, "id": image_id, "repo_digests": []}
    if container:
        image_id = subprocess.run(
            ["docker", "container", "inspect", container, "--format", "{{.Image}}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    elif not image_id:
        raise RuntimeError("docker provenance requires a running container or built image ID")
    result = subprocess.run(
        ["docker", "image", "inspect", image_id, "--format", "{{json .RepoDigests}}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return {"reference": image, "id": image_id, "repo_digests": json.loads(result.stdout)}


def build_image(image: str, dockerfile: Path) -> str:
    with tempfile.TemporaryDirectory() as context:
        subprocess.run(
            ["docker", "build", "--tag", image, "--file", str(dockerfile), context],
            check=True,
            stdout=sys.stderr,
        )
    return subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def capture(
    workload: Path,
    results_dir: Path,
    image: str,
    image_id: str,
    dockerfile: Path | None,
    runtime: str,
    environment: str = "",
    container: str = "",
) -> Path:
    provenance_dir = results_dir / "provenance"
    docker_dir = provenance_dir / "docker"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    if docker_dir.exists():
        shutil.rmtree(docker_dir)
    workload_copy = provenance_dir / "workload.yaml"
    shutil.copyfile(workload, workload_copy)

    manifest = {
        "schema_version": 1,
        "workload": {"path": "workload.yaml", **file_record(workload_copy)},
        "image": image_metadata(image, image_id, runtime, container),
        "runtime": runtime,
        "environment": sanitize_environment(environment),
    }
    if dockerfile is not None:
        docker_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_copy = docker_dir / "Dockerfile"
        shutil.copyfile(dockerfile, dockerfile_copy)
        manifest["build"] = {
            "dockerfile": "docker/Dockerfile",
            "dockerfile_record": file_record(dockerfile_copy),
            "context": "empty",
        }

    manifest_path = provenance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--image", required=True)
    build_parser.add_argument("--dockerfile", required=True, type=Path)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--workload", required=True, type=Path)
    capture_parser.add_argument("--results-dir", required=True, type=Path)
    capture_parser.add_argument("--image", required=True)
    capture_parser.add_argument("--image-id", default="")
    capture_parser.add_argument("--dockerfile", type=Path)
    capture_parser.add_argument("--runtime", required=True)
    capture_parser.add_argument("--environment", default="")
    capture_parser.add_argument("--container", default="")

    args = parser.parse_args()
    try:
        if args.command == "build":
            print(build_image(args.image, args.dockerfile))
        else:
            print(
                capture(
                    args.workload,
                    args.results_dir,
                    args.image,
                    args.image_id,
                    args.dockerfile,
                    args.runtime,
                    args.environment,
                    args.container,
                )
            )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"provenance: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
