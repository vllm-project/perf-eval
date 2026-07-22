---
name: vllm-pd-disagg-model-onboarding
description: Add, adapt, debug, and validate a model-specific vLLM prefill/decode-disaggregated serving workload in perf-eval, especially multi-node Slurm/Pyxis deployments using NIXL and vllm-router. Use this whenever a user asks to onboard another model to PD disaggregation, change prefill/decode topology (DEP, TEP, TP, DP, or EP), port an internal serving recipe into perf-eval, or diagnose a PD-disaggregated Buildkite/Slurm bring-up. Do not use for ordinary single-server workloads that omit the serving block.
compatibility: Requires the perf-eval repository. Real bring-up requires SSH access to a Slurm login node with Pyxis, shared storage, the model checkpoint, and a compatible vllm-router build; Buildkite testing requires an online queue agent.
---

# vLLM PD-disaggregated model onboarding

Turn a model serving recipe into a reviewable, reproducible perf-eval workload, prove it on the Slurm cluster, and only then exercise the Buildkite path.

Read [references/model-onboarding.md](references/model-onboarding.md) before editing. It contains the field-by-field checklist, topology formulas, staged validation commands, and failure guide distilled from the first Kimi K2.5 GB300 deployment.

## 1. Establish scope and preserve the worktree

1. Read `AGENTS.md`, `CLAUDE.md`, and the PD section of `README.md`.
2. Inspect `git status`, the current branch, and active worktrees. Preserve unrelated and untracked changes. Prefer a dedicated branch or existing task worktree.
3. Treat these files as the current implementation contract:
   - `workloads/kimi_k2_5_gb300_pd.yaml`
   - `lib/parse_workload.py`
   - `lib/slurm_pd_launcher.py`
   - `lib/run.sh` and `lib/run_vllm_bench.sh`
   - `lib/gpu_profiles.yaml`
   - `.buildkite/generate_pipeline.py`
   - `tests/test_parse_workload_serving.py`
   - `tests/test_slurm_pd_launcher.py`
4. Determine whether the task needs only a new workload or a schema/launcher change. Prefer a workload-only change when the existing schema can express the model and topology.

## 2. Extract a model contract

Translate the source recipe into an explicit contract before writing YAML:

- model/checkpoint path and how compute nodes see it;
- exact vLLM image or commit known to support the model;
- precision or quantization and the KV-cache dtype actually safe for correctness;
- common model flags, plus prefill-only and decode-only flags;
- prefill and decode counts, nodes per instance, GPUs per node, TP, derived local/global DP, and EP intent;
- NIXL, UCX, NCCL, Gloo, InfiniBand device, and socket-interface requirements;
- router revision, binary path, policy, endpoint expansion, and ports;
- first-start compilation/autotuning time and persistent cache requirements;
- minimal smoke benchmark and the later representative performance cases.

Do not silently guess a correctness-sensitive field. If the source does not establish KV-scale support, quantization compatibility, network devices, or the router revision, inspect the cluster/model artifacts or call out the uncertainty.

## 3. Create the smallest viable workload

1. Copy the closest PD workload and give the new recipe a model/hardware-specific name.
2. Keep initial bring-up opt-in with `nightly: false`, `bench_only: true`, and a generous timeout.
3. Put model flags shared by both roles in `serving.common_serve_args`; put scheduling flags in the relevant role's `serve_args`.
4. Let the launcher own host, port, TP/DP ranks and addresses, hybrid load balancing, and KV-transfer JSON. Never duplicate its managed flags in workload arguments.
5. Mount the model read-only, `/dev/infiniband`, and a persistent FlashInfer cache. Keep the Pyxis container writable when runtime compilation needs it.
6. Use environment-overridable mount sources for site-specific paths.
7. Add a tiny random smoke config before expensive throughput configs. The smoke case should be cheap enough to rerun while debugging startup and result ingestion.
8. Set `vllm_bench.metadata` deliberately: device, precision, effective topology label, total serving GPUs, `disagg: true`, and `is_multinode: true`.

Keep model-specific compromises documented next to the field. For example, if FP8 KV scales are unavailable or unreliable for the model loader, use a correct dtype such as BF16 and explain why.

## 4. Prove topology and generated commands without GPUs

Before allocating nodes:

1. Verify the topology arithmetic and router expansion by hand using the reference formulas.
2. Run the parser tests and launcher tests.
3. Parse the workload with a stubbed lm-eval registry when needed.
4. Feed the emitted normalized serving JSON to `lib/slurm_pd_launcher.py dry-run` and inspect every role instance, node command, mount, environment variable, endpoint, and allocation request.
5. Add focused tests for any new topology or schema behavior. Assert generated commands and invariants, not implementation trivia.
6. Run shell syntax checks for modified shell entry points.

Do not start a GPU allocation while parser or dry-run validation is failing.

## 5. Bring up locally on the Slurm cluster

Use SSH on the login node before involving Buildkite.

1. Confirm `sinfo`, `squeue`, `salloc`, `srun`, `scontrol`, Pyxis, shared checkout/results storage, model mount, router binary, and network devices.
2. Export site-specific model/cache paths and Enroot cache/data/runtime paths.
3. Start with the smoke workload via `./lib/run.sh <workload>` outside an existing allocation.
4. Watch allocation state and the labeled prefill/decode step logs. Confirm all role health checks, router readiness, one successful request, raw benchmark JSON, and ingestion handling.
5. Keep the first run alive long enough for kernel autotuning. Align `VLLM_ENGINE_READY_TIMEOUT_S`, role health deadlines, launcher wait time, Slurm time limit, and the persistent cache mount.
6. On failure, gather the earliest causal error from the role or router that failed. Fix one layer at a time: allocation, container/mounts, model load, collectives, NIXL, role readiness, router, request, benchmark, ingestion.
7. Confirm cleanup removes router and `srun` children and releases the allocation.

Record the exact image digest/tag, router revision, topology, overrides, job ID, and outcome so the Buildkite run is reproducible.

## 6. Validate performance and correctness separately

- Treat server health and one successful response as bring-up, not correctness proof.
- Run the smoke benchmark first, then one representative workload, then longer/high-concurrency cases.
- Inspect warnings about KV scales, quantization fallback, unsupported kernels, collectives, and tokenizer/remote-code behavior.
- Do not label numbers authoritative when the benchmark client shares serving resources or when the model uses an unvalidated cache dtype.
- Preserve every benchmark configuration row and raw artifact; verify multiple configs do not collapse into one ingestion result.

## 7. Exercise Buildkite only after local success

1. If new hardware or a queue is required, update `lib/gpu_profiles.yaml`; do not hard-code queues in pipeline generation.
2. Keep Slurm steps serialized per resolved queue because fixed router/DP RPC ports can collide.
3. Ensure a one-worker Buildkite agent is online on the login-node queue and its environment hook exposes the cluster's Slurm configuration.
4. Run repository tests and pre-commit checks, commit with the required AI co-author trailer, and push the feature branch.
5. Trigger `vllm/perf-eval` for the exact pushed SHA and only the new workload. Pass both `VLLM_IMAGE` and `VLLM_COMMIT` as required by repository guidance.
6. Report the build URL immediately. Tail failing job logs, diagnose the first causal failure, patch, revalidate locally where possible, push, and rerun once.
7. Avoid duplicate GPU builds for the same commit.

## 8. Finish the change

Update `README.md` whenever the workload schema, repo layout, prerequisites, commands, queue setup, or supported topology changes. Keep reusable explanations generic and leave model-specific caveats in the workload.

Hand off with:

- files changed and whether the launcher/schema changed;
- final topology and arithmetic;
- local tests and Slurm job result;
- Buildkite build URL/result, if run;
- image/model/router revisions;
- remaining correctness or measurement caveats;
- branch, commit, and PR status.

Include the repository-required AI-assistance disclosure in commit and PR metadata.
