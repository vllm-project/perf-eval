#!/usr/bin/env python3

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = load_module("provenance", ROOT / "lib" / "provenance.py")

    def test_source_metadata_records_repository_without_inspecting_worktree(self):
        completed = {
            ("config", "--get", "remote.origin.url"): "git@example.test:vllm.git\n",
            ("rev-parse", "HEAD"): "abc123\n",
            ("rev-parse", "--show-toplevel"): "/src\n",
        }

        def run_git(path, *args):
            return completed[args]

        with mock.patch.object(self.provenance, "run_git", side_effect=run_git):
            source = self.provenance.source_metadata(Path("/src/vllm"))

        self.assertEqual(source["repository"], "git@example.test:vllm.git")
        self.assertEqual(source["commit"], "abc123")
        self.assertEqual(source["context_subdirectory"], "vllm")
        self.assertNotIn("dirty", source)

    def test_native_image_metadata_does_not_require_docker(self):
        with mock.patch.object(self.provenance.subprocess, "run") as run:
            metadata = self.provenance.image_metadata("registry/image:tag", runtime="native")
        self.assertEqual(metadata, {"reference": "registry/image:tag", "id": "", "repo_digests": []})
        run.assert_not_called()

    def test_build_image_stdout_contains_only_image_id(self):
        completed = [mock.Mock(stdout=""), mock.Mock(stdout="sha256:123\n")]
        with mock.patch.object(
            self.provenance.subprocess, "run", side_effect=completed
        ) as run:
            image_id = self.provenance.build_image(
                "local/test:dev", Path("Dockerfile"), Path("."), {}
            )
        self.assertEqual(image_id, "sha256:123")
        self.assertIs(run.call_args_list[0].kwargs["stdout"], self.provenance.sys.stderr)

    def test_sensitive_build_args_are_redacted(self):
        sanitized = self.provenance.sanitize_build_args(
            {"CUDA_ARCH": "90", "HF_TOKEN": "secret", "api-key": "secret"}
        )
        self.assertEqual(sanitized["CUDA_ARCH"], "90")
        self.assertEqual(sanitized["HF_TOKEN"], "<redacted>")
        self.assertEqual(sanitized["api-key"], "<redacted>")

    def test_manifest_copies_inputs_and_is_self_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload = root / "workload.yaml"
            dockerfile = root / "Dockerfile.custom"
            results = root / "results"
            workload.write_text("name: test\n")
            dockerfile.write_text("FROM scratch\n")
            with mock.patch.object(
                self.provenance,
                "source_metadata",
                return_value={"repository": "repo", "commit": "abc"},
            ), mock.patch.object(
                self.provenance,
                "image_metadata",
                return_value={
                    "reference": "local/test:dev",
                    "id": "sha256:123",
                    "repo_digests": [],
                },
            ):
                manifest_path = self.provenance.capture(
                    workload=workload,
                    results_dir=results,
                    image="local/test:dev",
                    image_id="sha256:123",
                    dockerfile=dockerfile,
                    build_context=root,
                    build_args={"MODE": "dev"},
                    runtime="docker",
                    environment="CUDA_VISIBLE_DEVICES=0\nHF_TOKEN=secret",
                )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["image"]["id"], "sha256:123")
            self.assertNotIn("dirty", manifest["source"])
            self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(manifest["environment"]["HF_TOKEN"], "<redacted>")
            self.assertEqual(manifest["build"]["dockerfile"], "docker/Dockerfile")
            self.assertEqual((results / "provenance" / "workload.yaml").read_text(), "name: test\n")
            self.assertEqual(
                (results / "provenance" / "docker" / "Dockerfile").read_text(),
                "FROM scratch\n",
            )


class ParserBuildTests(unittest.TestCase):
    def run_parser(self, workload: dict, extra_env=None):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "workload.yaml"
            path.write_text(yaml.safe_dump(workload))
            env = {**os.environ, "BENCH_ONLY": "1", **(extra_env or {})}
            return subprocess.run(
                ["python3", str(ROOT / "lib" / "parse_workload.py"), str(path)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

    def workload(self):
        return {
            "name": "local-build",
            "gpu": "H200",
            "vllm": {
                "model": "example/model",
                "image": "local/vllm:test",
                "build": {
                    "dockerfile": "Dockerfile",
                    "context": ".",
                    "args": {"CUDA_ARCH": "90"},
                },
            },
            "vllm_bench": {
                "configs": [
                    {
                        "name": "smoke",
                        "input_len": 1,
                        "output_len": 1,
                        "num_prompts": 1,
                        "max_concurrency": 1,
                    }
                ]
            },
        }

    def test_build_config_is_exported(self):
        result = self.run_parser(self.workload())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WORKLOAD_BUILD_DOCKERFILE=", result.stdout)
        self.assertIn("WORKLOAD_BUILD_CONTEXT=", result.stdout)
        self.assertIn("WORKLOAD_BUILD_ARGS_JSON=", result.stdout)

    def test_build_requires_explicit_image_tag(self):
        workload = self.workload()
        del workload["vllm"]["image"]
        result = self.run_parser(workload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vllm.image", result.stderr)

    def test_build_rejects_image_override(self):
        result = self.run_parser(self.workload(), {"VLLM_IMAGE": "override:test"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot be combined", result.stderr)


if __name__ == "__main__":
    unittest.main()
