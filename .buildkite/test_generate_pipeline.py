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


def _b200_patch(image="img", num_gpus=8):
    plugin = g.b200_k8s_plugin(image, num_gpus, {}, "B200")
    return plugin["kubernetes"]["podSpecPatch"]


def test_b200_uses_podspec_patch_to_override_agent_container():
    """The B200 Kubernetes stack runs commands in its pre-created build
    container, so the workload image must patch container-0 instead of adding a
    separate unnamed container that the agent ignores."""
    patch = _b200_patch("lmsysorg/sglang:latest", 8)
    c = patch["containers"][0]
    assert c["name"] == "container-0"
    assert c["image"] == "lmsysorg/sglang:latest"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert patch["hostIPC"] is True
    assert c["securityContext"]["privileged"] is True
    assert "SYS_ADMIN" in c["securityContext"]["capabilities"]["add"]
    mounts = {m["name"]: m["mountPath"] for m in c["volumeMounts"]}
    volumes = {v["name"]: v for v in patch["volumes"]}
    assert mounts["infiniband"] == "/dev/infiniband"
    assert volumes["infiniband"]["hostPath"] == {
        "path": "/dev/infiniband",
        "type": "Directory",
    }
    assert "nodeSelector" not in patch


def test_b200_node_name_override_adds_hostname_selector():
    key = "B200_NODE_NAME"
    prev = os.environ.get(key)
    os.environ[key] = "dgxB200-12"
    try:
        patch = _b200_patch("lmsysorg/sglang:latest", 8)
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    assert patch["nodeSelector"] == {"kubernetes.io/hostname": "dgxB200-12"}


def test_native_runtime_venv_keeps_image_packages_visible():
    """Native runtimes use packages already installed in the workload image
    (`vllm`, `sglang`). The helper venv is only for pyyaml/lm-eval, so it must
    include system site packages."""
    profiles = {"B200": g.load_profiles()["B200"]}
    step = g.make_step(
        "workloads/glm_5_2_sglang_b200.yaml",
        {
            "name": "glm_5_2-sglang-b200",
            "gpu": "B200",
            "num_gpus": 8,
            "bench_only": True,
            "sglang": {"model": "zai-org/GLM-5.2-FP8"},
            "sglang_bench": {"configs": []},
        },
        profiles,
    )
    assert "python3 -m venv --system-site-packages .venv" in step["commands"][0]


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
    plugin = g.nvidia_docker_plugin("example/sglang:test", 8, profile, "B300")
    docker = plugin["docker#v5.2.0"]
    assert profile["queue"] == "b300-8"
    assert docker["entrypoint"] == ""
    assert docker["shell"] == ["/bin/sh", "-e", "-c"]
    assert docker["propagate-uid-gid"] is True
    assert docker["gpus"] == "all"
    assert docker["network"] == "host"
    assert docker["ipc"] == "host"
    assert docker["add-caps"] == ["SYS_PTRACE"]
    assert "memlock=-1" in docker["ulimits"]
    assert "/raid:/raid" in docker["volumes"]
    assert "HF_HOME=/raid/buildkite/hf-cache" in docker["environment"]
    assert "HOME=/raid/buildkite/home" in docker["environment"]
    assert "XDG_CACHE_HOME=/raid/buildkite/cache" in docker["environment"]
    assert (
        "/raid/buildkite/flashinfer-cubins-sglang:"
        "/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins"
    ) in docker["volumes"]
    assert "/etc/passwd:/etc/passwd:ro" in docker["volumes"]
    assert "/etc/group:/etc/group:ro" in docker["volumes"]


def test_run_command_preserves_injected_hf_home():
    rendered = g.RUN_TEMPLATE.format(path="workloads/example.yaml")
    assert 'HF_HOME="$${HF_HOME:-$(pwd)/.hf-cache}"' in rendered


def test_b300_sglang_workload_renders_docker_plugin_step():
    profiles = g.load_profiles()
    step = g.make_step(
        "workloads/glm_5_2_sglang_b200.yaml",
        {
            "name": "glm_5_2-sglang-b300",
            "gpu": "B300",
            "num_gpus": 8,
            "bench_only": True,
            "sglang": {
                "model": "/raid/inf-simon/models/zai-org/GLM-5.2-FP8",
                "image": "example/sglang:test",
            },
            "sglang_bench": {"configs": []},
        },
        profiles,
    )
    assert step["agents"] == {"queue": "b300-8"}
    assert step["plugins"][0]["docker#v5.2.0"]["image"] == "example/sglang:test"


def test_glm_b300_sglang_uses_matched_real_eagle_shape():
    path = os.path.join(
        os.path.dirname(HERE), "workloads", "glm_5_2_sglang_b200.yaml"
    )
    with open(path) as f:
        data = g.yaml.safe_load(f)
    sglang = data["sglang"]
    args = sglang["serve_args"]
    for expected in (
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend flashinfer",
        "--moe-runner-backend flashinfer_cutlass",
        "--kv-cache-dtype fp8_e4m3",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--context-length 32768",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--max-running-requests 256",
        "--disable-radix-cache",
    ):
        assert expected in args
    assert sglang["env"]["NVSHMEM_DISABLE_IB"] == 1
    assert "SGLANG_SIMULATE_ACC_LEN" not in sglang["env"]
    configs = data["sglang_bench"]["configs"]
    assert [(c["num_prompts"], c["max_concurrency"]) for c in configs] == [
        (128, 64),
        (512, 256),
    ]


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
