#!/usr/bin/env python3
"""Stdlib-only regression tests for generate_pipeline.py.

Run with ``python3 .buildkite/test_generate_pipeline.py`` (needs only pyyaml,
which the pipeline already installs). No pytest / GPU / network required.

Guards workload discovery/selection and the HF-cache volume behaviour: nested
model directories must remain discoverable without breaking existing
WORKLOADS selectors, and the AMD k8s plugin must not emit a root-disk hostPath
by default.
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


def _select(workloads, value):
    previous = os.environ.get("WORKLOADS")
    os.environ["WORKLOADS"] = value
    try:
        return g.select_workloads(workloads)
    finally:
        if previous is None:
            os.environ.pop("WORKLOADS", None)
        else:
            os.environ["WORKLOADS"] = previous


def test_load_workloads_discovers_nested_model_directories():
    workloads = g.load_workloads()
    paths = {w["path"] for w in workloads}
    assert "workloads/qwen3_5/h200.yaml" in paths
    assert "workloads/minimax_m3/b200.yaml" in paths
    assert all(os.path.dirname(path) != "workloads" for path in paths), paths


def test_select_workloads_accepts_nested_and_legacy_aliases():
    workload = {
        "path": "workloads/qwen3_5/h200.yaml",
        "data": {"name": "qwen3_5-h200", "nightly": True},
    }
    selectors = (
        "workloads/qwen3_5/h200.yaml",
        "workloads/qwen3_5/h200",
        "qwen3_5/h200.yaml",
        "qwen3_5/h200",
        "qwen3_5-h200",
        "qwen3_5_h200",
        "workloads/qwen3_5_h200.yaml",
    )
    for selector in selectors:
        selected = _select([workload], selector)
        assert selected == [workload], (selector, selected)


def test_select_workloads_preserves_irregular_legacy_alias():
    workload = {
        "path": "workloads/deepseek_v4_pro/h200.yaml",
        "data": {"name": "deepseek_v4_pro-h200", "nightly": True},
    }
    assert _select([workload], "deepseek_v4_pro_5_h200") == [workload]
    assert _select([workload], "workloads/deepseek_v4_pro_5_h200.yaml") == [workload]


def test_select_workloads_rejects_ambiguous_yaml_names():
    workloads = [
        {"path": "workloads/alpha/h200.yaml", "data": {"name": "duplicate"}},
        {"path": "workloads/beta/h200.yaml", "data": {"name": "duplicate"}},
    ]
    try:
        _select(workloads, "duplicate")
    except SystemExit as exc:
        assert "ambiguous" in str(exc), exc
    else:
        raise AssertionError("ambiguous WORKLOADS selector was accepted")


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
