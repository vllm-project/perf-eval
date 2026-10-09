#!/usr/bin/env python3

import hashlib
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

    def test_native_image_metadata_does_not_require_docker(self):
        with mock.patch.object(self.provenance.subprocess, "run") as run:
            metadata = self.provenance.image_metadata("registry/image:tag", runtime="native")
        self.assertEqual(metadata, {"reference": "registry/image:tag", "id": "", "repo_digests": []})
        run.assert_not_called()

    def test_docker_image_metadata_uses_running_container_without_pull(self):
        completed = [
            mock.Mock(stdout="sha256:123\n"),
            mock.Mock(stdout='["registry/image@sha256:abc"]\n'),
        ]
        with mock.patch.object(
            self.provenance.subprocess, "run", side_effect=completed
        ) as run:
            metadata = self.provenance.image_metadata(
                "registry/image:tag", runtime="docker", container="running-vllm"
            )
        self.assertEqual(
            metadata,
            {
                "reference": "registry/image:tag",
                "id": "sha256:123",
                "repo_digests": ["registry/image@sha256:abc"],
            },
        )
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["docker", "container", "inspect", "running-vllm", "--format", "{{.Image}}"],
        )
        self.assertTrue(all("pull" not in call.args[0] for call in run.call_args_list))

    def test_build_image_uses_an_empty_context(self):
        completed = [mock.Mock(stdout=""), mock.Mock(stdout="sha256:123\n")]
        with mock.patch.object(
            self.provenance.subprocess, "run", side_effect=completed
        ) as run:
            image_id = self.provenance.build_image(
                "local/test:dev", Path("Dockerfile")
            )
        self.assertEqual(image_id, "sha256:123")
        build_command = run.call_args_list[0].args[0]
        self.assertEqual(build_command[:5], ["docker", "build", "--tag", "local/test:dev", "--file"])
        self.assertNotEqual(Path(build_command[-1]).resolve(), ROOT)
        self.assertTrue(Path(build_command[-1]).name.startswith("tmp"))
        self.assertIs(run.call_args_list[0].kwargs["stdout"], self.provenance.sys.stderr)

    def test_capture_runs_after_server_start(self):
        script = (ROOT / "lib" / "run.sh").read_text()
        self.assertLess(
            script.index('start_server "$CONTAINER"'),
            script.index('python3 "$DIR/provenance.py" "${PROVENANCE_ARGS[@]}"'),
        )

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
                    runtime="docker",
                    environment="CUDA_VISIBLE_DEVICES=0\nHF_TOKEN=secret",
                )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["image"]["id"], "sha256:123")
            self.assertNotIn("source", manifest)
            self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(manifest["environment"]["HF_TOKEN"], "<redacted>")
            self.assertEqual(manifest["build"]["dockerfile"], "docker/Dockerfile")
            self.assertEqual((results / "provenance" / "workload.yaml").read_text(), "name: test\n")
            self.assertEqual(
                (results / "provenance" / "docker" / "Dockerfile").read_text(),
                "FROM scratch\n",
            )


class ReplayTests(unittest.TestCase):
    def create_replay_fixture(self, root: Path, image_id="sha256:123") -> tuple[Path, Path]:
        lib = root / "lib"
        provenance_dir = root / "bundle"
        docker_dir = provenance_dir / "docker"
        lib.mkdir()
        docker_dir.mkdir(parents=True)
        (lib / "replay.sh").write_text((ROOT / "lib" / "replay.sh").read_text())
        (lib / "gpu_profiles.yaml").write_text("H200: {}\n")
        (lib / "provenance.py").write_text(
            "#!/usr/bin/env python3\nimport os\nprint(os.environ['REBUILT_IMAGE_ID'])\n"
        )
        (lib / "run.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "cp \"$1\" \"${REPLAY_RUN_RECORD}.workload\"\n"
            "printf '%s\\n' \"$PERF_EVAL_PROFILES_FILE\" \"${REPLAY_RUN_RECORD}.workload\" > \"$REPLAY_RUN_RECORD\"\n"
        )
        (lib / "provenance.py").chmod(0o755)
        (lib / "run.sh").chmod(0o755)
        (lib / "replay.sh").chmod(0o755)
        workload = (
            "name: replay-test\n"
            "gpu: H200\n"
            "vllm:\n"
            "  model: example/model\n"
            "  image: local/test:dev\n"
            "  build:\n"
            "    dockerfile: Dockerfile\n"
            "vllm_bench:\n"
            "  configs: []\n"
        )
        dockerfile = "FROM scratch\n"
        (provenance_dir / "workload.yaml").write_text(workload)
        (docker_dir / "Dockerfile").write_text(dockerfile)
        manifest = {
            "schema_version": 1,
            "workload": {
                "path": "workload.yaml",
                "content": workload,
                "sha256": hashlib.sha256(workload.encode()).hexdigest(),
            },
            "image": {"reference": "local/test:dev", "id": image_id, "repo_digests": []},
            "runtime": "docker",
            "environment": {},
            "build": {
                "dockerfile": "docker/Dockerfile",
                "dockerfile_record": {
                    "content": dockerfile,
                    "sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
                },
                "context": "empty",
            },
        }
        manifest_path = provenance_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path, root / "run-record"

    def run_replay(self, manifest: Path, record: Path, rebuilt_image_id="sha256:123"):
        env = {
            **os.environ,
            "REBUILT_IMAGE_ID": rebuilt_image_id,
            "REPLAY_RUN_RECORD": str(record),
        }
        return subprocess.run(
            ["bash", str(manifest.parents[1] / "lib" / "replay.sh"), str(manifest)],
            env=env,
            text=True,
            capture_output=True,
        )

    def test_replay_validates_and_runs_captured_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, record = self.create_replay_fixture(Path(tmp))
            result = self.run_replay(manifest, record)
            self.assertEqual(result.returncode, 0, result.stderr)
            profiles, workload = record.read_text().splitlines()
            self.assertEqual(profiles, str(Path(tmp) / "lib" / "gpu_profiles.yaml"))
            replayed = yaml.safe_load(Path(workload).read_text())
            self.assertNotIn("build", replayed["vllm"])
            self.assertTrue(replayed["vllm"]["image"].startswith("perf-eval-replay:"))

    def test_replay_rejects_rebuilt_image_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, record = self.create_replay_fixture(Path(tmp))
            result = self.run_replay(manifest, record, rebuilt_image_id="sha256:different")
            self.assertEqual(result.returncode, 2)
            self.assertIn("does not match", result.stderr)
            self.assertFalse(record.exists())

    def test_replay_rejects_schema_and_checksum_errors_before_build(self):
        cases = (("schema_version", 2, "unsupported schema_version"),)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, record = self.create_replay_fixture(root)
            data = json.loads(manifest.read_text())
            for field, value, message in cases:
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(data))
                    changed[field] = value
                    manifest.write_text(json.dumps(changed))
                    result = self.run_replay(manifest, record)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(record.exists())
            changed = json.loads(json.dumps(data))
            changed["workload"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(changed))
            result = self.run_replay(manifest, record)
            self.assertEqual(result.returncode, 2)
            self.assertIn("workload checksum mismatch", result.stderr)
            self.assertFalse(record.exists())
            changed = json.loads(json.dumps(data))
            changed["build"]["dockerfile_record"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(changed))
            result = self.run_replay(manifest, record)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Dockerfile checksum mismatch", result.stderr)
            self.assertFalse(record.exists())


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
        self.assertNotIn("WORKLOAD_BUILD_CONTEXT=", result.stdout)
        self.assertNotIn("WORKLOAD_BUILD_ARGS_JSON=", result.stdout)

    def test_build_requires_explicit_image_tag(self):
        workload = self.workload()
        del workload["vllm"]["image"]
        result = self.run_parser(workload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vllm.image", result.stderr)

    def test_build_rejects_image_override(self):
        for variable in ("VLLM_IMAGE", "VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM", "VLLM_COMMIT"):
            with self.subTest(variable=variable):
                result = self.run_parser(self.workload(), {variable: "override"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot be combined", result.stderr)

    def test_build_rejects_pin_image(self):
        workload = self.workload()
        workload["vllm"]["pin_image"] = True
        result = self.run_parser(workload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pin_image", result.stderr)

    def test_build_rejects_local_context_and_args(self):
        for field, value in (("context", "."), ("args", {"MODE": "dev"})):
            with self.subTest(field=field):
                workload = self.workload()
                workload["vllm"]["build"][field] = value
                result = self.run_parser(workload)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsupported fields", result.stderr)

    def test_rejects_credentials_in_workload_environment(self):
        workload = self.workload()
        workload["vllm"]["env"] = {"HF_TOKEN": "secret"}
        result = self.run_parser(workload)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("credentials must not be stored", result.stderr)

    def test_profile_override_supports_replay_workloads_outside_repo(self):
        workload = self.workload()
        del workload["vllm"]["build"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.yaml"
            profiles = Path(tmp) / "profiles.yaml"
            path.write_text(yaml.safe_dump(workload))
            profiles.write_text(yaml.safe_dump({"H200": {}}))
            env = {
                **os.environ,
                "BENCH_ONLY": "1",
                "PERF_EVAL_PROFILES_FILE": str(profiles),
            }
            result = subprocess.run(
                ["python3", str(ROOT / "lib" / "parse_workload.py"), str(path)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
