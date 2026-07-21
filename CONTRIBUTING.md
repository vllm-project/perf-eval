# Contributing

How to add or change a workload recipe without breaking the nightly.

## Setup

Install pre-commit so recipes are linted on commit:

```bash
pip install pre-commit
pre-commit install
```

## Add or change a recipe

1. Copy an existing workload that targets the same GPU (e.g.
   `workloads/minimax_m3_b200.yaml` for B200) into `workloads/<model>_<hw>.yaml`.
2. Edit the fields for your model and tasks. The recipe schema is documented in
   [README.md](./README.md). Set `nightly: true` to include it in the nightly
   schedule; leave it off for opt-in recipes.
3. Lint locally (see below), open a PR, and validate on hardware before merge.

## Lint (local + CI)

`scripts/lint_workload.py` runs the same structural checks the pipeline relies
on, with no GPU and no model download:

- the recipe passes `lib/parse_workload.py` (required fields, known GPU profile,
  well-formed `lm_eval` / `vllm_bench` / `bfcl` blocks, and, when `lm_eval` is
  installed, task names against the registry);
- `.buildkite/generate_pipeline.py` emits a buildkite step for it.

```bash
scripts/lint_workload.py workloads/<your>_<hw>.yaml   # one or more paths
scripts/lint_workload.py --changed                    # recipes changed vs origin/main
scripts/lint_workload.py --all                        # every recipe
```

This is what catches the "commit garbage" cases: a typo'd field, an unknown
GPU, a malformed `serve_args`, or a task that generates no step.

## What runs on your PR

- **DCO**: every commit needs a `Signed-off-by` line (`git commit -s`).
- **lint workloads**: the buildkite pipeline runs `lint_workload.py --all` on
  every PR and fails the build if any recipe is structurally invalid.

The lint does **not** prove accuracy. It only proves the recipe is well-formed
and will produce a runnable step.

## Validate on hardware (before merge)

Prove the recipe actually serves and scores by triggering a build scoped to
just your recipe(s), against a known-good vLLM image:

```bash
bk build create \
  --pipeline perf-eval \
  --commit "<your perf-eval branch SHA>" \
  --branch "<your branch>" \
  --env "VLLM_IMAGE=<full image URI>" \
  --env "VLLM_COMMIT=<vLLM SHA>" \
  --env "WORKLOADS=workloads/<your>_<hw>.yaml"
```

Link the passing build in your PR so reviewers can see it ran on the target GPU.
A build takes ~30-90 minutes; GPU queues are shared, so don't trigger duplicate
builds for the same commit.

## AI assistance

See [CLAUDE.md](./CLAUDE.md) for the disclosure convention (PR body line plus a
commit trailer for non-trivial agent-authored changes).
