#!/usr/bin/env python3
"""Stdlib-only regression tests for generate_pipeline.py.

Run with ``python3 .buildkite/test_generate_pipeline.py`` (needs only pyyaml,
which the pipeline already installs). No pytest / GPU / network required.

Guards the HF-cache volume behaviour: the AMD k8s plugin must NOT emit a
root-disk hostPath by default (that leaked model caches onto node root disks),
and the volume source must be overridable per-cluster.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "generate_pipeline", os.path.join(HERE, "generate_pipeline.py")
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def _amd_volumes(profile, gpu="MI300X"):
    plugin = g.amd_k8s_plugin("img", 8, profile, gpu)
    patch = plugin["kubernetes"]["podSpecPatch"]
    vols = {v["name"]: v for v in patch["volumes"]}
    return patch, vols


def test_default_hf_cache_is_emptydir_not_hostpath():
    """Default (no override, no profile field) must be an emptyDir, never a
    hostPath — a hostPath on an unmounted node path is what filled root disks."""
    _, vols = _amd_volumes({})
    hf = vols["hf-cache"]
    assert "emptyDir" in hf, f"expected emptyDir default, got {hf}"
    assert "hostPath" not in hf, f"default must not be a hostPath: {hf}"


def test_hf_home_mount_matches_env():
    """Whatever the volume source, the mount path and HF_HOME must agree so vLLM
    finds its cache at the advertised location."""
    patch, vols = _amd_volumes({"hf_home": "/root/.cache/huggingface"})
    c = patch["containers"][0]
    mount = next(m for m in c["volumeMounts"] if m["name"] == "hf-cache")
    hf_home = next(e for e in c["env"] if e["name"] == "HF_HOME")
    assert mount["mountPath"] == hf_home["value"] == "/root/.cache/huggingface"


def test_run_command_preserves_injected_hf_home():
    """Kubernetes injects HF_HOME for its mounted cache. The run command must
    retain that value instead of redirecting a large download into the checkout.
    The double dollar survives Buildkite's pipeline-upload interpolation."""
    rendered = g.RUN_TEMPLATE.format(path="workloads/example.yaml")
    assert 'HF_HOME="$${HF_HOME:-$(pwd)/.hf-cache}"' in rendered


def test_profile_field_overrides_source():
    """A profile-level hf_cache_volume sets the source but keeps the volume name."""
    pvc = {"persistentVolumeClaim": {"claimName": "hf-cache-pvc"}}
    _, vols = _amd_volumes({"hf_cache_volume": pvc})
    assert vols["hf-cache"] == {"name": "hf-cache", **pvc}


def test_env_override_wins_over_profile_and_default():
    """{GPU}_HF_CACHE_VOLUME (per-cluster) overrides the profile and default."""
    key = "MI355X_HF_CACHE_VOLUME"
    prev = os.environ.get(key)
    os.environ[key] = '{"persistentVolumeClaim":{"claimName":"buildkite-hf-cache"}}'
    try:
        _, vols = _amd_volumes(
            {"hf_cache_volume": {"emptyDir": {}}}, gpu="MI355X"
        )
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    assert vols["hf-cache"] == {
        "name": "hf-cache",
        "persistentVolumeClaim": {"claimName": "buildkite-hf-cache"},
    }


def test_env_override_is_scoped_per_gpu():
    """An override for one GPU key must not leak into another GPU's volume."""
    key = "MI300X_HF_CACHE_VOLUME"
    prev = os.environ.get(key)
    os.environ[key] = '{"persistentVolumeClaim":{"claimName":"only-mi300"}}'
    try:
        _, vols = _amd_volumes({}, gpu="MI355X")
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    assert "emptyDir" in vols["hf-cache"], vols["hf-cache"]


def test_shipped_amd_profiles_have_no_rootdisk_hostpath():
    """The committed MI300X/MI355X profiles must not reintroduce an hf_home under
    /mnt/shared (the concrete path that leaked onto root disks)."""
    profiles = g.load_profiles()
    for gpu in ("MI300X", "MI355X"):
        hf_home = (profiles.get(gpu) or {}).get("hf_home") or ""
        assert not hf_home.startswith("/mnt/shared"), (
            f"{gpu} hf_home={hf_home!r} would land on the node root disk"
        )


def test_b300_profile_uses_standalone_docker_plugin():
    profiles = g.load_profiles()
    profile = profiles["B300"]
    plugin = g.nvidia_docker_plugin("example/vllm:test", 8, profile, "B300")
    docker = plugin["docker#v5.2.0"]
    assert profile["queue"] == "b300-8"
    assert docker["entrypoint"] == ""
    assert docker["shell"] == ["/bin/sh", "-e", "-c"]
    assert docker["propagate-uid-gid"] is True
    assert docker["gpus"] == "all"
    assert docker["network"] == "host"
    assert docker["ipc"] == "host"
    assert "memlock=-1" in docker["ulimits"]
    assert "/raid:/raid" in docker["volumes"]
    assert "HF_HOME=/raid/buildkite/hf-cache" in docker["environment"]
    assert "HOME=/raid/buildkite/home" in docker["environment"]
    assert "XDG_CACHE_HOME=/raid/buildkite/cache" in docker["environment"]
    assert not any("flashinfer-cubins" in volume for volume in docker["volumes"])


def test_b300_workload_renders_docker_plugin_step():
    profiles = g.load_profiles()
    step = g.make_step(
        "workloads/b300_runner_smoke.yaml",
        {
            "name": "b300-runner-smoke",
            "gpu": "B300",
            "num_gpus": 1,
            "bench_only": True,
            "vllm": {"model": "facebook/opt-125m", "image": "example/vllm:test"},
            "vllm_bench": {"configs": []},
        },
        profiles,
    )
    assert step["agents"] == {"queue": "b300-8"}
    assert step["plugins"][0]["docker#v5.2.0"]["image"] == "example/vllm:test"


def test_custom_ablation_workload_uses_system_packages_and_script():
    profiles = g.load_profiles()
    path = "workloads/ac_glm52_b300_ablation_current.yaml"
    with open(path) as f:
        data = g.yaml.safe_load(f)
    step = g.make_step(path, data, profiles)
    assert "--system-site-packages" in step["commands"][0]
    assert (
        "bash manual/ac_glm52_b300_ablation.sh "
        "workloads/ac_glm52_b300_ablation_current.yaml"
    ) in step["commands"][-1]
    assert 'HF_HOME="$${HF_HOME:-$(pwd)/.hf-cache}"' in step["commands"][-1]
    docker = step["plugins"][0]["docker#v5.2.0"]
    assert docker["image"].endswith(
        "@sha256:3ea9431a2298950a1aa2b4c07786b18396c756ee6f21b6cb49984620d1ab5413"
    )


def test_glm_b300_vllm_uses_matched_real_spec_shape():
    path = os.path.join(
        os.path.dirname(HERE), "workloads", "glm_5_2_b200.yaml"
    )
    with open(path) as f:
        data = g.yaml.safe_load(f)
    vllm = data["vllm"]
    args = vllm["serve_args"]
    for expected in (
        "--tensor-parallel-size 1",
        "--data-parallel-size 8",
        "--enable-expert-parallel",
        "--all2all-backend allgather_reducescatter",
        "--moe-backend triton",
        "--kv-cache-dtype fp8_e4m3",
        "--max-model-len 32768",
        "--max-num-batched-tokens 32768",
        "--long-prefill-token-threshold 4096",
        "--max-num-seqs 256",
        "--gpu-memory-utilization 0.85",
        "--no-enable-prefix-caching",
        "--speculative-config.method mtp",
        "--speculative-config.num_speculative_tokens 1",
    ):
        assert expected in args
    assert vllm["env"]["NVSHMEM_DISABLE_IB"] == 1
    configs = data["vllm_bench"]["configs"]
    assert [(c["num_prompts"], c["max_concurrency"]) for c in configs] == [
        (128, 64),
        (512, 256),
    ]
    assert all(c["args"]["num_warmups"] == 64 for c in configs)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
