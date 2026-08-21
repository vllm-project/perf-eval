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
