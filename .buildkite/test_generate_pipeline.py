#!/usr/bin/env python3
"""Stdlib-only regression tests for generate_pipeline.py.

Run with ``python3 .buildkite/test_generate_pipeline.py`` (needs only pyyaml,
which the pipeline already installs). No pytest / GPU / network required.

Guards the HF-cache volume behaviour: the AMD k8s plugin must NOT emit a
root-disk hostPath by default (that leaked model caches onto node root disks),
and the volume source must be overridable per-cluster. Also guards the
per-platform image pins, which have to reach the platform they name and only
that one.
"""

import contextlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "generate_pipeline", os.path.join(HERE, "generate_pipeline.py")
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


IMAGE_VARS = ("VLLM_IMAGE", "VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM", "VLLM_COMMIT")

CUDA = {"queue": "H200"}
ROCM = {"queue": "mi355_perf_eval", "image_repo": "vllm/vllm-openai-rocm"}


@contextlib.contextmanager
def build_env(**pins):
    """Run a test with exactly `pins` set, so the host env can't leak in."""
    saved = {k: os.environ.pop(k, None) for k in IMAGE_VARS}
    os.environ.update(pins)
    try:
        yield
    finally:
        for k in IMAGE_VARS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


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


def test_b200_cache_is_on_raid_not_root_disk():
    """The shipped B200 profile keeps the HF cache on /raid and mounts nothing
    under /mnt/shared, which is a plain directory on the B200 nodes' root disk."""
    profile = g.load_profiles()["B200"]
    spec = g.b200_k8s_plugin("img", 8, profile, "B200")["kubernetes"]["podSpec"]
    c = spec["containers"][0]
    hf_home = next(e["value"] for e in c["env"] if e["name"] == "HF_HOME")
    assert hf_home == profile["hf_home"] == "/raid/hf_cache", hf_home
    paths = [m["mountPath"] for m in c["volumeMounts"]]
    paths += [v["hostPath"]["path"] for v in spec["volumes"] if "hostPath" in v]
    assert not any(p.startswith("/mnt/shared") for p in paths), paths
    moved = g.b200_k8s_plugin("img", 8, {"hf_home": "/raid/other"}, "B200")
    env = moved["kubernetes"]["podSpec"]["containers"][0]["env"]
    assert {"name": "HF_HOME", "value": "/raid/other"} in env, env


def test_nvidia_plugin_profiles_set_hf_home():
    """b200_k8s_plugin reads hf_home from the profile with no default, so every
    shipped profile that uses it must set one."""
    for gpu, profile in g.load_profiles().items():
        if profile.get("k8s_plugin") == "nvidia":
            assert profile.get("hf_home"), f"{gpu} uses k8s_plugin nvidia but sets no hf_home"


def test_platform_pins_reach_their_own_platform():
    """The release-candidate case: two images that share nothing in their names,
    each reaching only the workloads of its platform."""
    with build_env(
        VLLM_IMAGE_CUDA="myrepo/vllm:v0.12.0rc2",
        VLLM_IMAGE_ROCM="other.registry/amd-vllm:rc2-final",
        VLLM_COMMIT="abc1234def5678",
    ):
        assert g.resolved_image({}, CUDA) == "myrepo/vllm:v0.12.0rc2"
        assert g.resolved_image({}, ROCM) == "other.registry/amd-vllm:rc2-final"


def test_platform_pin_wins_over_other_selection():
    """A platform pin overrides VLLM_IMAGE, VLLM_COMMIT and the workload's own
    image — it is the one thing that knows what that platform should run."""
    with build_env(
        VLLM_IMAGE="vllm/vllm-openai:nightly-abc1234",
        VLLM_IMAGE_ROCM="myrepo/rocm:pin",
        VLLM_COMMIT="abc1234",
    ):
        data = {"vllm": {"image": "vllm/vllm-openai-rocm:some-pin"}}
        assert g.resolved_image(data, ROCM) == "myrepo/rocm:pin"


def test_platform_without_a_pin_is_skipped():
    """A build that pins one platform and not the other has nothing to run on
    the other: skip it rather than benchmark an image the build didn't name."""
    with build_env(VLLM_IMAGE_ROCM="myrepo/rocm:pin", VLLM_COMMIT="abc1234def5678"):
        assert g.resolved_image({}, CUDA) == ""
        step = g.make_step(
            "workloads/test.yaml", {"name": "t", "gpu": "H200"}, g.load_profiles()
        )
    assert step["skip"] == "no CUDA image: set VLLM_IMAGE_CUDA"
    assert len(step["skip"]) <= 70, "Buildkite caps skip reasons at 70 chars"
    assert "plugins" not in step, "a skipped step must not name an image to pull"


def test_generic_image_covers_the_platform_without_a_pin():
    """VLLM_IMAGE is the fallback that keeps the unpinned platform running — for
    AMD, via the ROCm nightly of the same commit, as it does without any pins."""
    with build_env(
        VLLM_IMAGE="vllm/vllm-openai:nightly-abc1234def5678",
        VLLM_IMAGE_CUDA="myrepo/vllm:v0.12.0rc2",
        VLLM_COMMIT="abc1234def5678",
    ):
        assert g.resolved_image({}, ROCM) == (
            "vllm/vllm-openai-rocm:nightly-abc1234def5678"
        )


def test_no_pins_still_runs_every_platform():
    """The nightly build sets no per-platform pins, and nothing about it changes."""
    with build_env(
        VLLM_IMAGE="vllm/vllm-openai:nightly-abc1234def5678",
        VLLM_COMMIT="abc1234def5678",
    ):
        assert g.resolved_image({}, CUDA) == "vllm/vllm-openai:nightly-abc1234def5678"
        assert g.resolved_image({}, ROCM) == (
            "vllm/vllm-openai-rocm:nightly-abc1234def5678"
        )


def test_platform_pins_are_passed_to_the_step():
    """The job re-resolves the image, so it needs to see the pins the generator
    saw."""
    with build_env(VLLM_IMAGE_CUDA="myrepo/vllm:v0.12.0rc2"):
        step = g.make_step(
            "workloads/test.yaml", {"name": "t", "gpu": "H200"}, g.load_profiles()
        )
    assert step["env"]["VLLM_IMAGE_CUDA"] == "myrepo/vllm:v0.12.0rc2"


def test_summary_names_the_image_of_every_platform():
    """The generate-steps log has to name both platforms, including the one that
    resolved to nothing — that is the case someone needs to notice."""
    profiles = {"H200": CUDA, "MI355X": ROCM}
    selected = [
        {"path": "workloads/a.yaml", "data": {"name": "a", "gpu": "H200"}},
        {"path": "workloads/b.yaml", "data": {"name": "b", "gpu": "MI355X"}},
        {"path": "workloads/c.yaml", "data": {"name": "c", "gpu": "MI355X"}},
    ]
    with build_env(VLLM_IMAGE_CUDA="myrepo/vllm:v0.12.0rc2"):
        counts = g.images_by_platform(selected, profiles)
    assert counts == {("CUDA", "myrepo/vllm:v0.12.0rc2"): 1, ("ROCM", ""): 2}


def _pod_image(step):
    """The image the k8s plugin tells the pod to pull, either podSpec shape."""
    k8s = step["plugins"][0]["kubernetes"]
    spec = k8s.get("podSpec") or k8s["podSpecPatch"]
    return spec["containers"][0]["image"]


def test_amd_pod_pulls_public_ecr_directly():
    """AMD clusters hold no credentials for the pull-through cache, so their
    image must reach the pod as the public ECR ref, unrewritten."""
    ecr = "public.ecr.aws/q9t5s3a7/vllm-release-repo:abc1234def5678-rocm"
    with build_env(VLLM_IMAGE_ROCM=ecr):
        step = g.make_step(
            "workloads/test.yaml", {"name": "t", "gpu": "MI355X"}, g.load_profiles()
        )
    assert _pod_image(step) == ecr


def test_nvidia_pod_pulls_through_the_cache():
    """NVIDIA clusters renew credentials for the cache, and still use it."""
    ecr = "public.ecr.aws/q9t5s3a7/vllm-release-repo:abc1234def5678-x86_64"
    with build_env(VLLM_IMAGE_CUDA=ecr):
        step = g.make_step(
            "workloads/test.yaml", {"name": "t", "gpu": "B200"}, g.load_profiles()
        )
    assert _pod_image(step) == (
        g.ECR_PULL_THROUGH_CACHE + "q9t5s3a7/vllm-release-repo:abc1234def5678-x86_64"
    )


def test_non_ecr_images_are_never_rewritten():
    """A Docker Hub ref has no cache equivalent; both platforms take it as-is."""
    with build_env(
        VLLM_IMAGE_CUDA="vllm/vllm-openai:nightly-abc1234def5678",
        VLLM_IMAGE_ROCM="vllm/vllm-openai-rocm:nightly-abc1234def5678",
    ):
        profiles = g.load_profiles()
        b200 = g.make_step("workloads/t.yaml", {"name": "t", "gpu": "B200"}, profiles)
        mi355x = g.make_step("workloads/t.yaml", {"name": "t", "gpu": "MI355X"}, profiles)
    assert _pod_image(b200) == "vllm/vllm-openai:nightly-abc1234def5678"
    assert _pod_image(mi355x) == "vllm/vllm-openai-rocm:nightly-abc1234def5678"


def test_pin_image_outranks_a_platform_pin():
    """`pin_image` means the model runs nowhere else, so it beats a platform pin
    and keeps its workload out of the skip path the pins would otherwise trigger.
    """
    data = {"name": "t", "gpu": "MI355X",
            "vllm": {"image": "myrepo/only-here:v1", "pin_image": True}}
    with build_env(VLLM_IMAGE_ROCM="other.registry/amd-vllm:rc2-final"):
        assert g.resolved_image(data, ROCM) == "myrepo/only-here:v1"
    with build_env(VLLM_IMAGE_CUDA="myrepo/vllm:v0.12.0rc2"):
        step = g.make_step("workloads/t.yaml", data, g.load_profiles())
    assert "skip" not in step, "a pinned workload needs no image from the build"
    assert _pod_image(step) == "myrepo/only-here:v1"


def test_run_command_fetches_ingest_token_before_workload():
    command = g.RUN_TEMPLATE.format(path="workloads/example.yaml")
    get_secret = "buildkite-agent secret get INGEST_BEARER_TOKEN"
    assert get_secret in command
    assert command.index(get_secret) < command.index("./lib/run.sh")


def test_shipped_amd_profiles_pull_public_ecr_directly():
    image = "public.ecr.aws/example/release:rocm"
    profiles = g.load_profiles()
    for gpu in ("MI300X", "MI355X"):
        assert g.route_ecr_image(image, profiles[gpu]) == image


def test_ecr_pull_through_remains_default():
    image = "public.ecr.aws/example/release:cuda"
    assert g.route_ecr_image(image, {}) == (
        f"{g.ECR_PULL_THROUGH_CACHE}example/release:cuda"
    )


def test_cuda_release_image_does_not_select_rocm_commit():
    """A CUDA release image embeds a commit, but AMD has no build of it: fall
    back to the ROCm nightly rather than a same-commit image that never existed.
    """
    with build_env(VLLM_IMAGE="public.ecr.aws/example/release:abc123def456-x86_64"):
        assert g.resolved_image({}, ROCM) == "vllm/vllm-openai-rocm:nightly"


def test_step_has_automatic_retry_by_default():
    """Every step gets automatic retry with limit 2 unless the workload overrides it."""
    step = g.make_step(
        "workloads/test.yaml", {"name": "t", "gpu": "H200"}, g.load_profiles()
    )
    assert step["retry"] == {"automatic": {"limit": 2}}


def test_step_retry_is_configurable_per_workload():
    """A workload can override the retry policy, e.g. to disable it or change the limit."""
    step = g.make_step(
        "workloads/test.yaml",
        {"name": "t", "gpu": "H200", "retry": {"automatic": {"limit": 3}}},
        g.load_profiles(),
    )
    assert step["retry"] == {"automatic": {"limit": 3}}


def test_step_retry_can_be_disabled():
    """A workload can disable retry entirely by setting it to None or False."""
    step = g.make_step(
        "workloads/test.yaml",
        {"name": "t", "gpu": "H200", "retry": None},
        g.load_profiles(),
    )
    assert step["retry"] is None


KVV_REF = "7b938c38bbd89c2404a9ed5e7cb40da8079206b4"


def _kvv_workload(path, jobs, **kvv):
    return {
        "path": path,
        "data": {
            "name": g.workload_stem(path).replace("_", "-"),
            "gpu": "MI355X",
            "kvv": {"ref": KVV_REF, "jobs": jobs, **kvv},
        },
    }


def _kvv_steps(selected):
    with build_env():
        profiles = g.load_profiles()
        return [g.make_step(w["path"], w["data"], profiles) for w in selected]


def test_kvv_step_installs_uv_without_lm_eval():
    """KVV builds its own environment with uv; it needs nothing from lm_eval."""
    w = _kvv_workload("workloads/k.yaml", ["verify_parameters"])
    setup = _kvv_steps([w])[0]["commands"][0]
    assert "pip install --upgrade pyyaml uv\n" in setup, setup
    assert "lm-eval" not in setup, setup


def test_kvv_step_with_lm_eval_installs_both():
    w = _kvv_workload("workloads/k.yaml", ["verify_parameters"])
    w["data"]["lm_eval"] = {"tasks": [{"name": "gsm8k"}]}
    setup = _kvv_steps([w])[0]["commands"][0]
    assert "pip install --upgrade 'lm-eval[api]' pyyaml uv\n" in setup, setup


def test_kvv_step_with_bfcl_adds_uv_to_bfcl_packages():
    w = _kvv_workload("workloads/k.yaml", ["verify_parameters"])
    w["data"]["bfcl"] = {"test_categories": ["simple_python"]}
    setup = _kvv_steps([w])[0]["commands"][0]
    assert "bfcl-eval soundfile 'pyarrow>=24,<26' uv\n" in setup, setup


def test_kvv_judge_waits_for_and_downloads_generated_answers():
    """The judge runs after the generate step, even a failed one, and is told
    where run_kvv.py put that step's answers."""
    selected = [
        _kvv_workload("workloads/judge.yaml", ["beam_judge"], answers_from="gen_beam"),
        _kvv_workload("workloads/other.yaml", ["verify_parameters"]),
        _kvv_workload("workloads/gen_beam.yaml", ["beam_generate"]),
    ]
    steps = _kvv_steps(selected)
    g.link_kvv_answers(selected, steps)
    judge, other, gen = steps

    assert gen["key"] == "kvv-gen_beam", gen.get("key")
    assert judge["depends_on"] == [{"step": "kvv-gen_beam", "allow_failure": True}]
    assert judge["env"]["KVV_ANSWERS_STEP"] == "kvv-gen_beam"
    assert judge["env"]["KVV_ANSWERS_ARTIFACT"] == (
        "results/gen-beam/kvv-beam_generate/answers.jsonl"
    ), judge["env"]
    assert "key" not in other and "depends_on" not in other, other


def _kvv_suite():
    """A main KVV workload, a BEAM generate/judge pair, and an unrelated one."""
    main = _kvv_workload("workloads/main.yaml", ["verify_parameters"])
    main["data"]["groups"] = ["kvv", "kvv-main"]
    gen = _kvv_workload("workloads/gen_beam.yaml", ["beam_generate"])
    gen["data"]["groups"] = ["kvv", "kvv-beam"]
    judge = _kvv_workload("workloads/judge.yaml", ["beam_judge"], answers_from="gen_beam")
    judge["data"]["groups"] = ["kvv", "kvv-beam"]
    other = {"path": "workloads/other.yaml", "data": {"gpu": "H200", "nightly": True}}
    return [main, gen, judge, other]


def _select(workloads, value):
    """Stems main() would run with WORKLOADS=value, or its exit message."""
    saved = os.environ.get("WORKLOADS")
    os.environ["WORKLOADS"] = value
    try:
        selected = g.with_kvv_beam_partners(g.select_workloads(workloads), workloads)
        return [g.workload_stem(w["path"]) for w in selected]
    except SystemExit as e:
        return str(e)
    finally:
        if saved is None:
            os.environ.pop("WORKLOADS")
        else:
            os.environ["WORKLOADS"] = saved


def test_group_selects_every_workload_in_it():
    suite = _kvv_suite()
    assert _select(suite, "kvv") == ["main", "gen_beam", "judge"]
    assert _select(suite, "kvv-main") == ["main"]
    assert _select(suite, "kvv-beam") == ["gen_beam", "judge"]


def test_groups_mix_with_names_and_run_each_workload_once():
    suite = _kvv_suite()
    assert _select(suite, "kvv-beam,kvv,other") == ["gen_beam", "judge", "main", "other"]


def test_selecting_either_half_of_a_beam_pair_selects_both():
    suite = _kvv_suite()
    assert _select(suite, "gen_beam") == ["gen_beam", "judge"]
    assert _select(suite, "judge") == ["judge", "gen_beam"]


def test_nightly_selection_also_completes_beam_pairs():
    suite = _kvv_suite()
    suite[2]["data"]["nightly"] = True
    assert _select(suite, "") == ["judge", "other", "gen_beam"]


def test_group_named_like_a_workload_is_rejected():
    suite = _kvv_suite()
    suite[0]["data"]["groups"] = ["other"]
    err = _select(suite, "main")
    assert "group 'other' is also a workload name" in err, err


def test_groups_must_be_a_list_of_names():
    suite = _kvv_suite()
    suite[0]["data"]["groups"] = "kvv"
    err = _select(suite, "main")
    assert "groups must be a list" in err, err


def test_unknown_workloads_entry_is_rejected():
    err = _select(_kvv_suite(), "nope")
    assert "'nope' did not match any file in workloads/ or any workload's groups" in err, err


def test_kvv_judge_answers_from_must_name_a_workload():
    suite = _kvv_suite()
    suite[2]["data"]["kvv"]["answers_from"] = "gone"
    err = _select(suite, "judge")
    assert "kvv.answers_from 'gone' is not a workload" in err, err


def test_kvv_judge_answers_workload_must_generate():
    suite = _kvv_suite()
    suite[1]["data"]["kvv"]["jobs"] = ["verify_parameters"]
    err = _select(suite, "judge")
    assert "has no beam_generate job" in err, err


def test_shipped_kvv_groups():
    """The KVV workloads in workloads/ are wired up as the README says."""
    cwd = os.getcwd()
    os.chdir(os.path.join(HERE, ".."))
    try:
        workloads = g.load_workloads()
    finally:
        os.chdir(cwd)
    main = "kimi_k3_1_pro_mi355x_kvv"
    beam = ["kimi_k3_1_pro_mi355x_kvv_beam", "kimi_k2_6_mi355x_kvv_beam_judge"]
    assert _select(workloads, "kvv-main") == [main]
    assert sorted(_select(workloads, "kvv-beam")) == sorted(beam)
    assert sorted(_select(workloads, "kvv")) == sorted([main, *beam])


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
