#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SENSITIVE_NAME = re.compile(r"(?:token|secret|password|passwd|api[_-]?key|credential)", re.IGNORECASE)


def run_git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout


def source_metadata(path: Path) -> dict:
    try:
        repository = run_git(path, "config", "--get", "remote.origin.url").strip()
        commit = run_git(path, "rev-parse", "HEAD").strip()
        root = Path(run_git(path, "rev-parse", "--show-toplevel").strip())
        subdirectory = str(path.resolve().relative_to(root.resolve()))
    except (subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError(f"build context is not a readable git repository: {path}") from error
    return {
        "repository": repository,
        "commit": commit,
        "context_subdirectory": subdirectory or ".",
    }


def sanitize_build_args(build_args: dict) -> dict:
    return {
        str(name): "<redacted>" if SENSITIVE_NAME.search(str(name)) else value
        for name, value in build_args.items()
    }


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


def image_metadata(image: str, image_id: str = "", runtime: str = "docker") -> dict:
    if runtime != "docker":
        return {"reference": image, "id": image_id, "repo_digests": []}
    inspect_command = [
        "docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"
    ]
    result = subprocess.run(
        inspect_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        subprocess.run(["docker", "pull", image], check=True)
        result = subprocess.run(
            inspect_command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    digests = json.loads(result.stdout)
    if not image_id:
        image_id = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    return {"reference": image, "id": image_id, "repo_digests": digests}


def build_image(image: str, dockerfile: Path, context: Path, build_args: dict) -> str:
    command = ["docker", "build", "--tag", image, "--file", str(dockerfile)]
    for name, value in build_args.items():
        command.extend(["--build-arg", f"{name}={value}"])
    command.append(str(context))
    subprocess.run(command, check=True, stdout=sys.stderr)
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
    build_context: Path | None,
    build_args: dict,
    runtime: str,
    environment: str = "",
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
        "image": image_metadata(image, image_id, runtime),
        "runtime": runtime,
        "environment": sanitize_environment(environment),
    }
    if dockerfile is not None and build_context is not None:
        docker_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_copy = docker_dir / "Dockerfile"
        shutil.copyfile(dockerfile, dockerfile_copy)
        manifest["build"] = {
            "dockerfile": "docker/Dockerfile",
            "dockerfile_record": file_record(dockerfile_copy),
            "args": sanitize_build_args(build_args),
        }
        manifest["source"] = source_metadata(build_context)

    manifest_path = provenance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_json_map(value: str) -> dict:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("build args must be a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--image", required=True)
    build_parser.add_argument("--dockerfile", required=True, type=Path)
    build_parser.add_argument("--context", required=True, type=Path)
    build_parser.add_argument("--args-json", default="{}")

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--workload", required=True, type=Path)
    capture_parser.add_argument("--results-dir", required=True, type=Path)
    capture_parser.add_argument("--image", required=True)
    capture_parser.add_argument("--image-id", default="")
    capture_parser.add_argument("--dockerfile", type=Path)
    capture_parser.add_argument("--context", type=Path)
    capture_parser.add_argument("--args-json", default="{}")
    capture_parser.add_argument("--runtime", required=True)
    capture_parser.add_argument("--environment", default="")

    args = parser.parse_args()
    try:
        build_args = parse_json_map(args.args_json)
        if args.command == "build":
            print(build_image(args.image, args.dockerfile, args.context, build_args))
        else:
            print(
                capture(
                    args.workload,
                    args.results_dir,
                    args.image,
                    args.image_id,
                    args.dockerfile,
                    args.context,
                    build_args,
                    args.runtime,
                    args.environment,
                )
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"provenance: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
