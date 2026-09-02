# Model onboarding reference

Use this reference while adapting a model to the schema-v1 Slurm PD path. `lib/parse_workload.py` and `lib/slurm_pd_launcher.py` remain the source of truth when this reference and code differ.

## Topology arithmetic

For each role:

```text
local_dp_size = gpus_per_node / tensor_parallel_size
dp_size        = nodes_per_instance * local_dp_size
role_nodes     = count * nodes_per_instance
role_gpus      = count * nodes_per_instance * gpus_per_node
```

Across roles:

```text
total_nodes = sum(role_nodes)
total_gpus  = sum(role_gpus)
num_gpus    = total_gpus
```

`tensor_parallel_size` must divide `gpus_per_node` and currently stays within one node. Both roles use the same `gpus_per_node` inside one allocation.

The Kimi baseline demonstrates asymmetric local DP:

```text
prefill: count 1 * one 4-GPU node, TP1 => local DP4 => DEP4
decode:  count 2 * one 4-GPU node, TP4 => local DP1 => TEP4 x2
total:   3 nodes, 12 GPUs
```

EP is enabled by vLLM serve arguments. The schema derives TP and DP; do not infer EP merely from the topology fields.

## Router mapping

Schema v1 uses:

- `prefill_endpoints: all_instances`
- `decode_endpoints: all_nodes`

vllm-router v0.1.12 has one global `intra_node_data_parallel_size`. Set it to `1` when roles have different local DP sizes. This registers endpoint-level URLs: vLLM balances the prefill endpoint across its local DP ranks, while each TP decode replica remains one router worker.

Only use a router DP expansion value above 1 when every role has the same local DP size. Otherwise the router expands prefill and decode URLs uniformly and invents invalid workers.

Use fixed ports only with queue-level serialization. Check router, metrics, NIXL side-channel, and DP RPC port ranges for collisions.

## Field checklist

### Top level

- `name`: unique model/hardware/PD identifier.
- `gpu`: key in `lib/gpu_profiles.yaml`.
- `num_gpus`: exact role-derived total.
- `nightly: false`: keep experimental bring-up manual.
- `bench_only: true`: skip accuracy suites while proving serving/perf flow.
- `timeout_in_minutes`: cover queueing, image/model load, first compile, smoke benchmark, and cleanup.

### `vllm`

- `model`: in-container path or model ID visible to every role and the benchmark client.
- `image`: pin the first known-compatible image. A newer tag is not automatically compatible with quantization, NIXL, router protocol, or the target GPU.
- `env`: merge model and site runtime variables over the hardware profile.
- Leave legacy `vllm.serve_args` empty for `pd_disagg`; use the serving fields below.

### `serving.common_serve_args`

Typical model decisions include:

- expert parallelism and EP weight filtering;
- remote code and language-model-only mode;
- attention and MoE backends;
- block size, max model length, memory utilization, prefix caching, and chunked prefill policy;
- KV-cache dtype supported correctly by the checkpoint and loader;
- eager mode versus CUDA graphs;
- loading strategy and access logging.

The launcher rejects orchestration flags it owns: host, port, TP/DP settings and ranks, hybrid load balancing, and KV-transfer config.

### Role fields

- `role`: exactly one `prefill` and one `decode` role object.
- `count`: number of independent instances/replicas.
- `nodes_per_instance`: nodes owned by each instance.
- `gpus_per_node`: same for both roles.
- `tensor_parallel_size`: divides GPUs per node.
- `base_port`: node-local API port.
- `kv_role`: `kv_producer` for prefill, `kv_consumer` for decode.
- `serve_args`: scheduling and role-specific behavior only.
- `health_check`: path, polling interval, and timeout large enough for first startup.

### Slurm and Pyxis

- `partition` and `time_limit` match the target cluster.
- `grace_period_s` lets the controller stop router and role steps before forced cancellation.
- Model mount is readable on every node and normally read-only in the container.
- `/dev/infiniband` is exposed to the container for NIXL RDMA.
- FlashInfer cache is persistent and mounted at `/root/.cache/flashinfer`.
- Container is writable when vLLM/FlashInfer needs runtime compilation.
- Checkout and results path are shared at the identical path because the benchmark client bind-mounts the repository.

### Networking

- Discover the actual IP interface and InfiniBand device on the target nodes; do not copy another cluster's values.
- Keep NCCL and Gloo socket interfaces consistent with the routable fabric.
- Use a UCX transport allow-list such as `rc,cuda_copy` when appropriate. A literal `^cuda_ipc` in an allow-list position can be resolved as a transport name instead of excluding it.
- Set `UCX_MEMTYPE_CACHE=n` only when required by the tested stack.
- Expose a unique `VLLM_NIXL_SIDE_CHANNEL_PORT` and let the launcher set the per-node side-channel host.
- Validate that firewalls and cluster routing permit role endpoints and the login-node router path used by the benchmark client.

### KV transfer

- Schema v1 supports `NixlConnector`.
- Use `load_failure_policy: fail` during bring-up so broken KV transfer cannot silently become a misleading benchmark.
- Tune `extra_config.num_threads` only after correctness and basic transfer work.

### Benchmark metadata

- `device`: dashboard hardware label.
- `precision`: model weight precision, not automatically the KV-cache dtype.
- `tp`: document the effective decoder parallel/replica degree used for comparisons; explain non-obvious labels in a comment.
- `total_gpus`: all serving GPUs participating in the end-to-end result.
- `disagg: true` and `is_multinode: true`.

## Staged validation

### 1. Repository-only checks

Run the focused suites first:

```bash
python3 -m unittest \
  tests.test_parse_workload_serving \
  tests.test_slurm_pd_launcher \
  tests.test_run_vllm_bench \
  tests.test_generate_pipeline \
  tests.test_ingest_perf
bash -n lib/run.sh
bash -n lib/server.sh
bash -n lib/run_lm_eval.sh
bash -n lib/run_vllm_bench.sh
```

The repository's parser smoke-test pattern in `AGENTS.md` can stub `lm_eval.tasks.TaskManager`. Include every lm-eval task named by the target workload. Bench-only PD recipes may use an empty task registry.

Capture normalized serving JSON without allowing shell evaluation of arbitrary output. One convenient inspection flow is:

```bash
python3 lib/parse_workload.py workloads/<model>_<hardware>_pd.yaml
```

Extract `WORKLOAD_SERVING_JSON` from the emitted shell assignments, then run:

```bash
python3 lib/slurm_pd_launcher.py --config '<normalized-json>' dry-run
```

Inspect:

- `salloc` node/task/GPU request;
- role instance ordering and node assignment;
- TP/local DP/global DP and DP RPC ports;
- per-node NIXL host and producer/consumer config;
- Pyxis image, mounts, writable mode, workdir, and env;
- router worker URLs and expansion;
- benchmark client placement and router URL substitution.

### 2. Login-node preflight

Verify before consuming GPUs:

```bash
sinfo
squeue -u "$USER"
command -v salloc
command -v srun
command -v scontrol
test -r "$MODEL_PATH"
test -x "$ROUTER_REPO/target/release/vllm-router"
```

Also inspect the network interfaces and InfiniBand devices on an allocated node if the site values are not already established.

### 3. Smoke run

Export model, FlashInfer, and Enroot paths, then run the workload from the login node outside an allocation. Success means:

1. one allocation with the expected nodes/GPUs;
2. every prefill/decode instance becomes healthy;
3. router becomes ready with the expected workers;
4. one request completes through the router;
5. the smoke benchmark writes valid JSON;
6. ingestion is attempted with disaggregated/multinode metadata;
7. cleanup releases all steps and the allocation.

### 4. Buildkite

- Push the exact commit before triggering.
- Scope `WORKLOADS` to the new workload stem.
- Pass the repository-required `VLLM_IMAGE` and `VLLM_COMMIT` values.
- Verify the generated step targets the Slurm queue and is serialized by resolved queue.
- Report the build URL, then use tail/search logs on only the failed job.

## Failure guide

| Symptom | Check first | Typical fix |
| --- | --- | --- |
| Allocation never starts | partition, account/QOS, requested nodes, queue state | Correct site allocation fields or wait; do not change serving flags. |
| Pyxis container exits immediately | image pull, Enroot paths, mounts, writable mode | Fix image access/cache paths or the missing host path. |
| Model load fails | checkpoint format, vLLM version, remote code, quantization kernels | Pin a compatible image and model flags; avoid unrelated topology changes. |
| First health check times out | compilation/autotuning logs and cache mount | Persist cache and align engine, role, launcher, Slurm, and Buildkite timeouts. |
| NCCL/Gloo bootstrap fails | socket interface and address reachability | Use the cluster's routable interface consistently. |
| NIXL/UCX cannot initialize | `/dev/infiniband`, UCX device/transports, side-channel host/port | Expose devices and correct site-specific fabric settings. |
| Router shows wrong workers | role counts, endpoint modes, global router DP expansion | Use endpoint-level expansion (`1`) for asymmetric local DP. |
| Requests hang after prefill | producer/consumer KV roles, connector failure policy, NIXL logs | Fix transfer config before tuning performance. |
| Output is numerically suspect | KV-scale warnings, quantization fallback, dtype support | Use a validated KV dtype or calibrated scale path and rerun correctness. |
| Only first bench config runs | stdin consumed by Slurm/Pyxis child | Keep benchmark TSV on a dedicated file descriptor, as `lib/run.sh` does. |
| Multiple rows overwrite/collapse | config name/artifact path/ingest metadata | Preserve a distinct raw result and ingestion call per config. |
| Concurrent jobs collide | router or DP RPC ports shared on one queue | Serialize generated Slurm steps and keep the login-node agent at one worker. |

## Change boundaries

Prefer these scopes in order:

1. New workload only.
2. Workload plus existing hardware profile.
3. Parser/schema extension with tests and README update.
4. Launcher behavior change with plan tests, lifecycle tests, README update, and a regression build.

Do not generalize the launcher from one model's unexplained flag. First decide whether the behavior is a model concern, cluster concern, topology concern, or truly orchestration-wide.
