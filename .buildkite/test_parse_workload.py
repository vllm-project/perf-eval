import importlib.util
import shlex
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


@pytest.mark.parametrize(
    ("profile", "env", "expected"),
    [
        ({"nsys": True}, None, True),
        ({}, None, False),
        ({"nsys": True}, "0", False),
        ({}, "1", True),
        ({"nsys": True, "image_repo": "vllm/vllm-openai-rocm"}, "1", False),
    ],
)
def test_nsys_follows_profile_with_build_override(monkeypatch, profile, env, expected):
    if env is None:
        monkeypatch.delenv("NSYS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("NSYS_PROFILE", env)

    assert parse_workload.nsys_enabled(profile) is expected


FASTSAFETENSORS = {"--load-format": "fastsafetensors"}


@pytest.mark.parametrize(
    ("serve_args", "defaults", "expected"),
    [
        ("--tensor-parallel-size 8", FASTSAFETENSORS,
         "--tensor-parallel-size 8 --load-format fastsafetensors"),
        ("--load-format auto", FASTSAFETENSORS, "--load-format auto"),
        ("--load-format=dummy", FASTSAFETENSORS, "--load-format=dummy"),
        ("", {"--enforce-eager": True}, "--enforce-eager"),
        ("--max-num-seqs 8", None, "--max-num-seqs 8"),
    ],
)
def test_default_serve_args_apply_unless_the_workload_sets_them(
    serve_args, defaults, expected
):
    assert parse_workload.apply_default_serve_args(serve_args, defaults) == expected


KVV_REF = "7b938c38bbd89c2404a9ed5e7cb40da8079206b4"


def write_kvv_workload(tmp_path: Path, kvv: dict, serve_args: str = "") -> Path:
    (tmp_path / "lib").mkdir(exist_ok=True)
    (tmp_path / "workloads").mkdir(exist_ok=True)
    (tmp_path / "lib" / "gpu_profiles.yaml").write_text(yaml.safe_dump({"H200": {}}))
    workload = tmp_path / "workloads" / "kvv.yaml"
    workload.write_text(
        yaml.safe_dump(
            {
                "name": "kvv",
                "gpu": "H200",
                "vllm": {"model": "test/model", "serve_args": serve_args},
                "kvv": kvv,
            }
        )
    )
    return workload


def parse_exports(out: str) -> dict:
    return dict(tok.partition("=")[::2] for tok in shlex.split(out))


def test_kvv_only_workload_emits_job_rows(tmp_path, capsys):
    kvv = {"ref": KVV_REF, "jobs": ["verify_parameters", "ocrbench_thinking_effort:max"]}
    parse_workload.main(str(write_kvv_workload(tmp_path, kvv)))

    exports = parse_exports(capsys.readouterr().out)
    assert exports["WORKLOAD_KVV_REF"] == KVV_REF
    assert exports["WORKLOAD_KVV_THINKING_EFFORT"] == "true"
    assert exports["WORKLOAD_KVV_TSV"].splitlines() == [
        "verify_parameters\tkvv-verify_parameters\tkvv_verify_parameters",
        "ocrbench_thinking_effort:max\tkvv-ocrbench_thinking_effort-max"
        "\tkvv_ocrbench_thinking_effort_max",
    ]


def test_kvv_thinking_effort_can_be_disabled(tmp_path, capsys):
    kvv = {"ref": KVV_REF, "jobs": ["AIME_thinking"], "thinking_effort": False}
    parse_workload.main(str(write_kvv_workload(tmp_path, kvv)))

    assert "WORKLOAD_KVV_THINKING_EFFORT=false" in capsys.readouterr().out


def test_workload_without_kvv_emits_empty_kvv_rows(tmp_path, capsys):
    parse_workload.main(str(write_workload(tmp_path, 3600)))

    assert "WORKLOAD_KVV_TSV=''" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("kvv", "message"),
    [
        ({"ref": "main", "jobs": ["verify_parameters"]}, "40-character"),
        ({"ref": KVV_REF[:12], "jobs": ["verify_parameters"]}, "40-character"),
        ({"ref": KVV_REF, "jobs": []}, "non-empty jobs"),
        ({"ref": KVV_REF, "jobs": ["all"]}, "'all' is not supported"),
        ({"ref": KVV_REF, "jobs": ["nope"]}, "unknown kvv job"),
        ({"ref": KVV_REF, "jobs": ["ocrbench_thinking_effort"]}, "needs an effort"),
        ({"ref": KVV_REF, "jobs": ["ocrbench_thinking_effort:huge"]}, "effort must be"),
        ({"ref": KVV_REF, "jobs": ["verify_parameters:max"]}, "does not take an effort"),
        ({"ref": KVV_REF, "jobs": ["AIME_thinking", "AIME_thinking"]}, "duplicate"),
        ({"ref": KVV_REF, "jobs": [{"AIME_thinking": 1}]}, "must be a string"),
        ({"ref": KVV_REF, "jobs": ["ocrbench_thinking_effort:"]}, "effort must be"),
        ({"ref": KVV_REF, "jobs": ["AIME_thinking"], "thinking_effort": "yes"},
         "true or false"),
        ({"ref": KVV_REF, "jobs": ["AIME_thinking"], "extra": 1}, "unsupported fields"),
        ({"ref": KVV_REF, "jobs": ["AIME_thinking"], "answers_from": "x"},
         "only valid with the beam_judge"),
        ({"ref": KVV_REF, "jobs": ["beam_judge"]}, "requires answers_from"),
        ({"ref": KVV_REF, "jobs": ["beam_generate", "beam_judge"], "answers_from": "x"},
         "separate workloads"),
    ],
)
def test_rejects_invalid_kvv_block(tmp_path, kvv, message):
    serve_args = "--served-model-name test/model kimi-k2.6"
    with pytest.raises(SystemExit, match=message):
        parse_workload.main(str(write_kvv_workload(tmp_path, kvv, serve_args)))


@pytest.mark.parametrize(
    "serve_args",
    ["", "--served-model-name kimi-k2.6", "--served-model-name=kimi-k2.6"],
)
def test_kvv_judge_must_serve_both_names(tmp_path, serve_args):
    kvv = {"ref": KVV_REF, "jobs": ["beam_judge"], "answers_from": "gen"}
    with pytest.raises(SystemExit, match="--served-model-name test/model kimi-k2.6"):
        parse_workload.main(str(write_kvv_workload(tmp_path, kvv, serve_args)))


def test_kvv_judge_accepts_both_served_names(tmp_path, capsys):
    kvv = {"ref": KVV_REF, "jobs": ["beam_judge"], "answers_from": "gen"}
    serve_args = "--served-model-name test/model kimi-k2.6 --reasoning-parser kimi_k2"
    parse_workload.main(str(write_kvv_workload(tmp_path, kvv, serve_args)))

    assert "kvv-beam_judge" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("serve_args", "expected"),
    [
        ("--served-model-name a b --tp 8", ["a", "b"]),
        ("--served-model-name=a", ["a"]),
        ("--tp 8", []),
        # vLLM's argparse keeps the last occurrence and expands no abbreviations.
        ("--served-model-name a --served-model-name b c", ["b", "c"]),
        ("--served a", []),
    ],
)
def test_served_model_names(serve_args, expected):
    assert parse_workload.served_model_names(serve_args) == expected
