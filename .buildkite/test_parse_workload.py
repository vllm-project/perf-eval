import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parse_workload", ROOT / "lib" / "parse_workload.py"
)
assert SPEC and SPEC.loader
parse_workload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parse_workload)

IMAGE_VARS = ("VLLM_IMAGE", "VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM", "VLLM_COMMIT")


@pytest.fixture(autouse=True)
def build_env(monkeypatch):
    """Resolve images from what the test sets, not what the build's env pins."""
    for var in IMAGE_VARS:
        monkeypatch.delenv(var, raising=False)


def write_workload(tmp_path: Path, startup_timeout_s: object) -> Path:
    (tmp_path / "lib").mkdir()
    (tmp_path / "workloads").mkdir()
    (tmp_path / "lib" / "gpu_profiles.yaml").write_text(yaml.safe_dump({"H200": {}}))
    workload = tmp_path / "workloads" / "workload.yaml"
    workload.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "gpu": "H200",
                "vllm": {
                    "model": "test/model",
                    "startup_timeout_s": startup_timeout_s,
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
        )
    )
    return workload


def test_emits_server_startup_timeout(tmp_path, capsys):
    parse_workload.main(str(write_workload(tmp_path, 7200)))

    assert "WORKLOAD_SERVER_STARTUP_TIMEOUT=7200" in capsys.readouterr().out


@pytest.mark.parametrize("value", [0, -1, True, "7200"])
def test_rejects_invalid_server_startup_timeout(tmp_path, value):
    with pytest.raises(SystemExit, match="must be a positive integer"):
        parse_workload.main(str(write_workload(tmp_path, value)))


def test_cuda_release_image_does_not_select_rocm_commit(monkeypatch):
    """A CUDA release image embeds a commit, but AMD has no build of it: fall
    back to the ROCm nightly rather than a same-commit image that never existed.
    """
    monkeypatch.setenv(
        "VLLM_IMAGE", "public.ecr.aws/example/release:abc123def456-x86_64"
    )

    image, commit = parse_workload.resolve_image(
        {}, {"image_repo": "vllm/vllm-openai-rocm"}
    )

    assert image == "vllm/vllm-openai-rocm:nightly"
    assert commit == ""
