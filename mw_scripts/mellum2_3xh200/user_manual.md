# Mellum2 Architecture Variants: H200 Continued Pretraining

This pipeline contains eight Base cases and six Thinking cases. The Base-only
v3_2 and v3_3 cases are the scalar and nonlinear cross-injection experiments.
All cases except v1_1 train the original parameters; v1_1 freezes the source
model and trains only its new correction branches. The default
target is GPU node 32 (one 8xH200 node); `MELLUM_NODES` still controls topology.

Select one of the fourteen cases with `--variant`:

Human-facing labels use dots (v1.1, v3.1, v3.2, v3.3), while YAML filenames,
registry ids, checkpoint metadata, and run directories use underscores. The
launcher accepts either spelling and canonicalizes it to the underscore form.

| Case | Graph | Initial HF checkpoint |
|---|---|---|
| `base-v0` | original sequential baseline | JetBrains Base |
| `base-v1` | parallel attention/MoE | local byte-identical v1 derivative |
| `base-v2` | shifted adjacent-layer schedule | local byte-identical v2 derivative |
| `base-v3` | adjacent decoders in parallel | local byte-identical v3 derivative |
| `base-v3_1` | parallel decoders with a merged expert pool | local byte-identical v3_1 derivative |
| `base-v1_1` | parallel attention/MoE plus Ivan-style trainable correction | local v1_1 derivative with deterministic random correction metadata |
| `base-v3_2` | adjacent parallel decoders with scalar cross-attention injection | local v3_2 derivative; zero-initialized scalar gates |
| `base-v3_3` | adjacent parallel decoders with nonlinear cross-attention injection | local v3_3 derivative; zero-output nonlinear gates |
| `thinking-v0` | original sequential baseline | JetBrains Thinking |
| `thinking-v1` | parallel attention/MoE | local byte-identical v1 derivative |
| `thinking-v2` | shifted adjacent-layer schedule | local byte-identical v2 derivative |
| `thinking-v3` | adjacent decoders in parallel | local byte-identical v3 derivative |
| `thinking-v3_1` | parallel decoders with a merged expert pool | local byte-identical v3_1 derivative |
| `thinking-v1_1` | parallel attention/MoE plus Ivan-style trainable correction | local v1_1 derivative with deterministic random correction metadata |

The derived checkpoints live under
`/cb/ml-olympus/codesign_pretrain/checkpoints`. Every inherited tensor remains
byte-identical to its source; `architecture_surgery.json` selects the graph and,
for v1_1, specifies the deterministic correction-branch initialization. See
`mw_scripts/model_surgeon/README.md` for the exact equations and invariants.

## Files and Configuration Entry Points

- `config/*.yaml`: the fourteen complete case configs. v3_1, v3_2 and v3_3 inherit v3's measured
  starting point until its own throughput sweep is complete.
- `config.yaml`: retained as a legacy editable example; the launcher does not select it by default.
- `configuration.py`: YAML validation, tokenizer ID checks, and shard weight resolution.
- `train.py`: training configuration, HF weight loading, training, and checkpoint resume.
- `launch.sh`: SSH coordinator running on the current host; no local PyTorch installation is required.
- `node.sh`: container environment, torch distributed launch, and PID management.
- `network_test.py`: cross-node NCCL correctness and bandwidth smoke test.
- `validate.py`: weight round trips, logits, and cross-entropy comparisons for a toy or real checkpoint.
- `wandb_logging.py`: W&B destination, authentication check, and persistent run identity.

Model code lives in `src/megatron/bridge/models/mellum/`, and the base recipe lives in
`src/megatron/bridge/recipes/mellum/`. Model parameters are derived from the HF config;
tensor shapes are not hardcoded to the 12B model.
Model support is limited to the all-MoE Mellum2 architecture with SiLU,
normalized top-k routing, and identical local/global RoPE bases.
Unsupported variants raise errors. Parallel attention/MoE cannot be enabled by changing YAML alone.

The fourteen case configs use the same 100%-normalized mixture and 10B-token budget so
convergence comparisons differ only by source checkpoint and graph.
They save every 100 optimizer steps and set `run.keep_last_checkpoints: 3`.
Bridge deletes older
`iter_*` directories only after a newer checkpoint saves successfully, and it
also saves the final checkpoint after normal completion.
All cases use peak/min LR `3e-5`/`3e-6`. Current v1_1 matches Ivan's normalized,
non-gated compensatory branch and freezes every source parameter; the older
gated/zero-output v1_1 and its low-LR recovery results are retired.
All one-node cases use BF16, TP=1, PP=1, CP=1, and ETP=1. The measured baseline
uses EP=8; surgery variants use EP=4 with all-to-all dispatch and permute fusion.
v0, v1, v2, v3, v3_1, v3_2, and v3_3 use MBS1 and GBS192
(24 gradient-accumulation steps). The current v1_1
trains only its 297M correction parameters and uses MBS2 with the same GBS192
(12 accumulation steps); keeping GBS fixed preserves the optimizer batch size.
v2/v3/v3_1/v3_2/v3_3 require PP=1 and do not support expert-parallel overlap. Before a long
v3_1 run, use a short real-EP smoke test and confirm both merged expert pools receive
router gradients and `merged_pool_frac_0` is neither pinned nor collapsed.

### Historical artifact names

The completed 2100-step experiments retain their original storage and W&B
provenance. Readers normalize these names automatically:

| Historical artifact id | Canonical id |
|---|---|
| `v4` | `v1_1` |
| `v5_1` | `v3_2` |
| `v5_2` | `v3_3` |

Do not rename the historical `runs/mellum2_base-v4_10B`, `-v5_1_10B`, or
`-v5_2_10B` directories. New launches use canonical run names. The local
Thinking-derived checkpoints were intentionally removed; their YAML files are
templates until those checkpoints are generated again.

## Docker and File Paths

Run these commands on the host containing the current checkout:

```bash
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --prepare-containers
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --preflight
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --prepare
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --dry-run
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --train
```

Actions and `--variant` may appear in either order. `MELLUM_CONFIG` remains an
escape hatch for custom experiments, but it cannot be combined with `--variant`.

Container preparation creates a persistent `mellum2_3xh200` container on each selected node,
using `nemo:26.06`. An existing container with the same name is never deleted or recreated.
If it is stopped, inspect it first and start it explicitly with `docker start`.
Containers remain available after use for inspecting logs and the environment.
The writable-output containers now use the original name `mellum2_3xh200`.
Clear any temporary `MELLUM_CONTAINER=mellum2_3xh200_rw` override, or explicitly set
`MELLUM_CONTAINER=mellum2_3xh200`, to select the current default.
Preparation rejects existing containers without the required read-only data/read-write output mounts.
The previous containers were stopped and deleted at the user's request; the replacements were
renamed without recreation. Shared data, HF cache, and run directories were not deleted.

Preparation uses non-interactive SSH and `docker exec` without a terminal. Ctrl+C on the
coordinator is not a reliable way to stop a remote preparation process: it can leave the
container process running. `launch.sh --stop` manages training/network-test PID files, not
preparation. Before rerunning an interrupted preparation, inspect `docker top mellum2_3xh200`
on the first node. Stopping the container terminates all of its processes, not just preparation;
only do this when no other work in that container needs to continue.

Mount mappings:

| Host | Container |
|---|---|
| Shared `ws` directory containing the checkout | `/workspace/ws` |
| `/cb/ml-olympus`, read-only | `/workspace/ml-olympus` |
| `/cb/ml-olympus/codesign_pretrain/runs`, read-write | `/workspace/ml-olympus/codesign_pretrain/runs` |
| `/dev/infiniband` | `/dev/infiniband` |

The nested output mount lets checkpoints, logs, and dataset caches use the large shared storage
without permitting writes to tokenized training data. Its host directory is created by the SSH user.
Preflight creates and removes a tiny temporary file in the configured run directory on each node
to test actual write access before training. Changing YAML alone cannot make a read-only mount writable.

The original Docker example's `/wokspace/ml-olympus` typo is corrected to `/workspace/ml-olympus`.
In addition to `--network=host`, the container must expose the RDMA devices.
The image includes EFA libfabric and the AWS OFI NCCL plugin.
The scripts default to `NCCL_NET_PLUGIN=ofi`, `FI_PROVIDER=efa`, and `NCCL_NET="AWS Libfabric"`.
This prevents another NCCL network plugin with the same name in the image from taking precedence.

Training processes run with the SSH user's UID/GID so they can write to shared NFS;
root may not have write access to this NFS mount.
`uv` uses the image's preinstalled `/opt/venv/bin/python`, without automatically syncing project
dependencies or changing the lockfile.
The HF cache defaults to this directory's `.cache/huggingface`, shared by the selected nodes.

Environment overrides:

```bash
export MELLUM_NODES="172.31.210.225"
export MELLUM_CONTAINER=mellum2_3xh200
export MELLUM_MASTER_PORT=29541
# Match the memory budget granted for the allocation (new containers only):
export MELLUM_MEMORY_GIB=1024
# If needed, choose a socket bootstrap interface reachable from the other selected nodes:
# export NCCL_SOCKET_IFNAME=eth0
```

Additional overrides include `MELLUM_IMAGE`, `MELLUM_HOST_WS`, `MELLUM_HOST_DATA`,
`MELLUM_HOST_RUNS`, `MELLUM_REMOTE_RUNS`, `MELLUM_REMOTE_REPO`, `MELLUM_MASTER_ADDR`,
`MELLUM_HF_HOME`, `MELLUM_PYTHON`, and `OFI_NCCL_LIB`.
All SSH calls use `-F "$HOME/.ssh/config"`. Ensure non-interactive SSH works for every selected node.
To reuse an existing container, set `MELLUM_CONTAINER=megatron_bridge_mw`, but first verify its
shared mounts and RDMA device access.

## Switching the Number of Nodes

Change only one environment variable. All prepare, dry-run, network-test, train, status,
and stop operations use the same node list.
The current default connects only to GPU node 32. Override stale multi-node settings
that remain in your shell:

```bash
export MELLUM_NODES="172.31.210.225"
bash mw_scripts/mellum2_3xh200/launch.sh --help    # Show nodes and GPU count locally; no SSH connections
# On first use, complete the container and data preparation described below before proceeding:
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --dry-run
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --network-test
bash mw_scripts/mellum2_3xh200/launch.sh --variant thinking-v2 --train
```

If a third node is allocated later, append its IP to `MELLUM_NODES` and repeat the same checks
and launch commands. Each node must still have eight available GPUs, the same container setup,
and the same shared mounts. The first IP is the default master.
`nnodes`, `node_rank`, and `world_size` are generated automatically;
no Python, shell, or YAML edits are needed.
The historical directory and container name `mellum2_3xh200` is retained for path compatibility
and does not restrict the actual number of nodes.

With TP=PP=CP=ETP=1, MBS=1, and GBS=192:

| Nodes | GPUs / Dense DP | Expert DP (EP8 / EP4) | Gradient Accumulation Steps per Update |
|---|---|---|---|
| 1 | 8 | 1 / 2 | 24 |
| 2 | 16 | 2 / 4 | 12 |
| 3 | 24 | 3 / 6 | 8 |

Both layouts retain `192 × 8192 = 1,572,864` tokens per update, so the training iteration count,
warmup, and LR schedule remain unchanged. No changes to the selected case YAML are required.
Reducing DP increases each GPU's share of optimizer state; memory usage and performance still
need smoke testing. Dry-run checks batch divisibility and expert mesh compatibility for other
node counts or parallelism settings; not every combination is valid.

This is a **stop-and-relaunch** topology change, not automatic elastic shrinking during training.
Use the old node list to stop a job running on the old topology, then change the list.
Do not connect to or stop jobs on nodes that have already been handed over to someone else.
If training checkpoints already exist, optimizer/RNG resume across world sizes needs separate
validation. Same-topology resume tests do not establish that 24 → 16 GPU resume works.
Use a separate `run.directory` for new experiments.

## Data Mixture

Each YAML key is a **full path inside the container**. Its value is a positive relative weight;
weights do not need to sum to one:

```yaml
dataset:
  eod_id: 0
  num_workers: 2
  subsets:
    /workspace/ml-olympus/codesign_pretrain/tokenized/mellum2/stack-v3-train/data: 8
    /workspace/ml-olympus/codesign_pretrain/tokenized/mellum2/Nemotron-CC-v2/High-Quality-Synthetic/data: 2
```

Each supplied `config/*.yaml` lists every directory directly containing `.bin/.idx` files,
including the final `/data` component. This lets you independently tune categories such as
`Nemotron-CC-v2/High-Quality-Synthetic/data` and `Nemotron-CC-Math-v1/3/data`.
Each listed directory initially has example weight `1`; equal leaf-directory weights do not
preserve the earlier equal-weight mixture across top-level dataset families.
The list includes `uspto/data` and reflects the directories currently containing shard files.
If tokenization creates additional directories later, add them explicitly in a new run's YAML.

Directories are searched recursively for `.idx/.bin` pairs. You can also specify a single
Megatron dataset prefix, omitting the `.idx/.bin` suffix.
A directory's weight is distributed among its shards in proportion to their token counts,
avoiding oversampling subsets containing many small shards.
Sampling weights apply to fixed-length training sequences; the actual number of effective
loss tokens may depend on EOD handling and other processing.
Duplicate prefixes, overlapping directories, empty directories, missing bin files,
and truncated index or bin files raise errors.

Each shard must have a `manifest.json` from this project's tokenization pipeline in an ancestor directory.
Preparation verifies the tokenizer JSON checksum and checks source token IDs against the target checkpoint.
The Base and Thinking models currently share their ordinary vocabulary, but six Base reserved tokens
(IDs 27–32) were renamed to special markers in Thinking.
The reserved-ID scan policy is explicit in YAML:

```yaml
dataset:
  reserved_token_check: skip
```

- `strict`: preparation **scans each selected shard's entire bin file** for the renamed IDs.
  If any occur, preparation fails; existing data is not modified.
- `skip`: do not read bin contents for this check. Accept that the renamed IDs may occur and
  their absence is **unverified**. This is the supplied configuration's setting for the
  original-versus-parallel attention/MoE convergence experiment, not a compatibility guarantee.
  Preparation logs a warning for each affected tokenizer manifest; dry-run/training warns again.
- Omitting the field preserves the older strict behavior.

`metadata/resolved_mixture.json` records the policy, actually checked IDs in
`reserved_ids_checked_absent`, and unverified IDs in `reserved_ids_not_checked`.
Skipped IDs are never recorded as having passed a scan. Tokenizer checksum, ordinary token-ID
mapping, vocabulary bounds, EOD, and bin/idx completeness checks remain enabled in both modes.
Any other token-ID mismatch fails immediately. The first scan of a large dataset can take a long
time in strict mode because it reads all selected data. Skip mode still reads index files and
tokenizer metadata, but avoids this full bin-content scan.
The dataset's EOD=0 is the document boundary written by the Base tokenizer;
Thinking's generation EOS=28 is a different concept.
Training uses a `NullTokenizer` with the dataset EOD and does not change existing token 0 to 28.

After interrupting preparation before its first snapshot was published, rerun
`bash launch.sh --prepare` from this directory with the updated YAML. The HF download cache is reused.
There is no per-shard scan checkpoint. Changing the policy after a snapshot exists requires a
new `run.directory`; explicit `strict` and an omitted policy are equivalent.
For architecture comparisons, freeze and reuse exactly the same shard list for both variants;
do not let newly completed tokenization shards change only one side of the comparison.

If a subset has not produced complete shards, remove it from YAML for now. After tokenization
finishes, use it in a **new run directory**.
The first preparation freezes the shard list. Newly added shards are not automatically included
in that run, keeping the mixture unchanged on resume.
Do not replace, modify, or delete frozen bin/idx files. Launch and resume also verify frozen
file sizes/mtimes and tokenizer checksums.
Tokenization may continue creating new shard prefixes, but selected files and tokenizer metadata
must already be complete and remain unchanged. An `.idx` still being written can fail preparation.

## Training Parameters

```text
DP = world_size / (TP × PP × CP)
gradient accumulation = global_batch_size / (micro_batch_size × DP)
tokens per optimizer step = global_batch_size × sequence_length
train_iters = ceil(train_tokens / tokens per optimizer step)
```

The default one-node setup uses DP=8, MBS=1, and GBS=192, accumulating 24 micro batches per update.
EP shards the experts; it does not introduce another division of the dense DP used for the global batch.
With 64 experts, EP=8 holds eight experts per rank while EP=4 holds sixteen.
GBS=128 is valid on 16 GPUs (eight accumulation steps), but is not divisible by MBS×DP=24 on 24 GPUs.
Keeping GBS=192 lets you switch between these two layouts without changing tokens per update.

The default sequence length is 8192, giving approximately 1.57M tokens per update.
The 1000 warmup iterations correspond to approximately 1.57B tokens.
Warmup increases LR linearly from zero to the peak, followed by cosine decay to the minimum LR;
the decay horizon includes warmup.
Changing GBS or sequence length changes the iteration count and warmup token count.
Check GPU memory before increasing the micro batch size.

## Preparation, Network Validation, and Launch

```bash
# Edit the YAML mixture, budget, LR, and a separate run.directory before running:
bash mw_scripts/mellum2_3xh200/launch.sh --prepare
bash mw_scripts/mellum2_3xh200/launch.sh --dry-run
bash mw_scripts/mellum2_3xh200/launch.sh --network-test
```

`--prepare` downloads the full HF checkpoint on the first node, validates the data,
and saves `metadata/resolved_mixture.json`.
The network test does not load the model. It checks all-reduce using every GPU on the selected
nodes (8 GPUs by default).
Inspect `logs/network-test_node_*.log` for `NET/OFI`, `efa`, `AWS Libfabric`, and `NCCL PASS`.
Seeing only Socket does not establish that EFA is active.
Use `REQUIRE_EFA=0` for a diagnostic comparison if needed, but retain the default of 1 for production runs.
AWS plugin selection reference: [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl/blob/master/src/Makefile.am).

Preparation or launch fails if configuration or data is inconsistent.
Training stays in the foreground; running it inside tmux is recommended:

```bash
bash mw_scripts/mellum2_3xh200/launch.sh --train
```

Alternatively, run it in the background on the host while keeping a coordinator log:

```bash
nohup bash mw_scripts/mellum2_3xh200/launch.sh --train > train.log 2>&1 < /dev/null &
```

The coordinator launches one torch distributed agent per node, with eight GPU workers per agent.
If any node fails, it stops the other node processes owned by that launch.
Use a new `run.directory` to isolate different experiments.

Before a production long run, use a `smoke.local.yaml` with a separate `run.directory` and
`exit_interval: 3`, while retaining the production LR and training budget.
It saves a checkpoint and exits after the third optimizer update.
The YAML schema requires all fields to be explicit and rejects misspelled fields;
arbitrary hidden overrides are not supported.

The pipeline disables dataset-side attention mask creation and keeps the existing
backend-generated causal attention path. Otherwise, each DataLoader worker can cache
an unused dense mask for every visited shard (64 MiB per shard at sequence length 8192),
eventually exhausting host RAM. Tokens, labels, loss masks, position IDs, and sampling
indices are unchanged. Existing prepared mixtures and dataset caches can be reused;
this fix does not require another `--prepare`. After restarting, monitor host memory
and verify that training progresses beyond the previous failure point.

### CPU memory policy

The Mellum pipeline uses `CompactGPTDatasetConfig`, a Bridge dataset provider.
It still constructs ordinary MCore `GPTDataset` objects, so dataset identities,
mixture weights, sample order, and sampling cache keys do not change.
After each shard has been built or loaded from cache, the builder releases its
build-only `indices` array before loading subsequent shards and before creating
DataLoader workers. For the current 11.65 billion sequences, those arrays would
otherwise retain about 43.38 GiB per rank, or 347 GiB across eight ranks on a node.
One shard's temporary construction arrays can still be allocated; this is not a
guarantee about total peak RAM. Deferred index loading is intentionally disabled
because it needs these arrays for length/epoch calculations.

Token `.bin` files use bounded file reads rather than long-lived mmap mappings.
Sampling `.npy` indices still use read-only mmap. This may change storage I/O
performance, but does not change token values or invalidate the prepared mixture
or existing sampling cache. No new `--prepare` is needed.

New containers have a **1024 GiB CPU memory limit**, with no additional swap allowance.
This leaves substantial headroom on the current 2 TiB nodes. Set `MELLUM_MEMORY_GIB`
to the budget approved for your allocation before `--prepare-containers`.
Existing containers are not automatically resized or replaced. GPU launch refuses
an unlimited-memory container; recreate it explicitly if necessary. A container
limit is a safety boundary, not a resource reservation or authorization to use a node.
Use the allocation/launch process required by IT before restarting training.

During training, `logs/cpu_memory_node_<rank>.jsonl` records cgroup CPU usage,
anonymous memory, file cache, shared memory, page tables, OOM counters, and host
`MemAvailable` every 30 seconds, tagged by launch ID. Warnings above 85% of the
container limit appear in `logs/memory_monitor_node_<rank>.log`.
Counters such as `file`, `shmem`, and `file_mapped` overlap and must not be summed.
Read the JSONL from the host using the `/cb/ml-olympus/...` path:

```bash
tail -n 1 /cb/ml-olympus/codesign_pretrain/runs/mellum2_thinking_first_run_50B_1007/logs/cpu_memory_node_1.jsonl | jq
```

After restarting, check that memory reaches a sustainable plateau through dataset
initialization, DataLoader startup, several hundred steps, and a checkpoint save.
Passing correctness tests is not a substitute for this runtime memory validation.

## Logs, Stop, and Resume

Output follows YAML `run.directory`. The current configuration writes under shared large storage:
`/workspace/ml-olympus/codesign_pretrain/runs/mellum2_thinking_first_run_50B_1007`
inside containers, or `/cb/ml-olympus/codesign_pretrain/runs/mellum2_thinking_first_run_50B_1007`
on the coordinator/hosts. Checkpoints are always in the `checkpoints/` subdirectory:

```text
metadata/resolved_mixture.json   Frozen data, tokenizer checksums, and configuration
metadata/ConfigContainer.yaml   Complete Bridge configuration
metadata/wandb_run.json         Non-secret W&B destination and persistent run ID
logs/train_node_0.log           First node; node 1 is the second, node 2 exists for three-node runs
logs/cpu_memory_node_0.jsonl    CPU memory categories and host headroom, every 30 seconds
logs/memory_monitor_node_0.log CPU memory warnings or monitoring failures
logs/network-test_node_0.log    Network diagnostic log
checkpoints/                   torch_dist weights, optimizer, RNG, and training state
dataset_cache/                 Data index cache shared by selected nodes
tensorboard/                   Training metrics
wandb/                         Local W&B run files
```

```bash
bash mw_scripts/mellum2_3xh200/launch.sh --status
bash mw_scripts/mellum2_3xh200/launch.sh --stop
tail -F /cb/ml-olympus/codesign_pretrain/runs/mellum2_thinking_first_run_50B_1007/logs/train_node_0.log
```

Stop targets only the process groups recorded for this run.
A forced interruption does not guarantee a checkpoint for the current iteration;
the checkpoint interval and `exit_interval` determine the available resume points.

To resume, set `run.resume_checkpoint` to the checkpoint **root directory** containing
`latest_checkpointed_iteration.txt`. Keep the original training budget, mixture, parallelism,
optimizer, and scheduler settings.
If the smoke test exits at step 3, change `exit_interval` to 6 and run again to test resume to step 6.

```yaml
run:
  directory: /workspace/ml-olympus/codesign_pretrain/runs/mellum2_thinking_first_run_50B_1007
  save_interval: 100
  keep_last_checkpoints: 3
  log_interval: 10
  resume_checkpoint: /workspace/ml-olympus/codesign_pretrain/runs/mellum2_thinking_first_run_50B_1007/checkpoints
  # Leave omitted/false for a normal resume.
  override_checkpoint_scheduler: false
  # Numerical recovery only: retain weights/RNG/data position but reset optimizer moments.
  reset_optimizer_on_resume: false
  exit_interval: null
```

The initial run loads HF weights and creates a new optimizer.
A resumed run loads optimizer/RNG state, data position, and the scheduler;
it does not overwrite trained weights with the HF checkpoint again.
For numerical recovery only, set `override_checkpoint_scheduler: true` to keep the
checkpoint's optimizer state and completed-step count while applying the optimizer and
scheduler values from YAML. This must be paired with `resume_checkpoint` and requires
rerunning `--prepare` after changing an optimizer or scheduler value.
If the optimizer moments may also be contaminated by an unstable trajectory, additionally set
`reset_optimizer_on_resume: true`. The scheduler is advanced to the checkpoint's consumed-sample
position, but Adam moments are freshly initialized. This option requires
`override_checkpoint_scheduler: true`.
An output directory containing existing checkpoints raises an error unless resume is explicitly requested.
Changing the mixture or hyperparameters within the same run is not allowed;
use a new directory for a new experiment.

## Weights & Biases

The supplied YAML enables the private W&B instance with these non-secret settings:

```yaml
wandb:
  enabled: true
  base_url: https://cerebras.wandb.io
  entity: team-inferenceml
  project: draft_pretrain
  run_name: mellum2_thinking_cpt
```

The project is created by the first successful online run if it does not already exist,
provided your account has permission to create projects in this team.
Project URL: [draft_pretrain](https://cerebras.wandb.io/team-inferenceml/draft_pretrain).
The launcher does not create a project merely by reading YAML or running dry-run.
Megatron Bridge initializes W&B only on the last global rank (rank 7 with the default 8 GPUs).
It uploads the training scalar metrics and complete Bridge configuration, not 16 separate runs.
The YAML `base_url` is authoritative. If `WANDB_BASE_URL` or `WANDB_HOST` is provided,
it must match the YAML destination (a trailing slash is allowed), or authentication/training fails.

### Option A: Automatic Coordinator `.env` Loading

Place a `.env` file beside `launch.sh` on the coordinator VM:

```dotenv
WANDB_API_KEY=YOUR_PRIVATE_API_KEY
WANDB_BASE_URL=https://cerebras.wandb.io
WANDB_HOST=https://cerebras.wandb.io
```

Every launcher invocation automatically reads this file, independently of the working directory.
Existing shell environment variables take precedence over file values, including explicitly empty values.
Only these three W&B variables are loaded; other variable assignments are ignored.
Blank lines, comment-only lines, surrounding whitespace, optional `export`, and surrounding
single/double quotes are supported. Use one assignment per line; inline comments and multiline
values are not supported. Values are literal: no variable expansion, command substitution, or
shell code is executed. Invalid assignments, unmatched quotes, and duplicate W&B assignments
fail with a line number only, without displaying the contents.

The file is Git-ignored. Restrict its permissions with `chmod 600 .env` from this directory.
Do not commit it or use shell tracing while credentials are set; the launcher disables its own
xtrace before loading/forwarding credentials. Credentials and URL values are forwarded over SSH
stdin only for `--wandb-check` and `--train`, never as command-line values or training metadata.
No manual container login or coordinator `wandb login` is required with this method.

`MELLUM_ENV_FILE=/absolute/path/to/private.env` selects another file;
`MELLUM_ENV_FILE=''` disables loading. An explicitly selected missing file raises an error;
a missing default `.env` is allowed for the other authentication methods below.

After creating the file, run `bash launch.sh --wandb-check` from this directory to verify
authentication on the selected nodes without starting training or creating a W&B run.

### Option B: Interactive Container Login

Logging in on the coordinator host alone does not automatically authenticate SSH/Docker workers.
For this pipeline, log in inside a selected container as the same UID/GID used by training.
The credential file below is under the ignored, shared `.cache/` directory, so one login is
available to both nodes. The launcher passes this `NETRC` path to the training containers.

From the coordinator:

```bash
ssh -F "$HOME/.ssh/config" -t 172.31.223.220 \
  'docker exec -it --user "$(id -u):$(id -g)" -w /workspace/ws/Megatron-Bridge mellum2_3xh200 bash'
```

Then, inside that container:

```bash
export USER=mellum
export UV_CACHE_DIR=/tmp/mellum-wandb-uv
export NETRC="$PWD/mw_scripts/mellum2_3xh200/.cache/wandb-auth.netrc"
umask 077
mkdir -p "${NETRC%/*}"
/usr/local/bin/uv run --no-project --python /opt/venv/bin/python \
  wandb login --host=https://cerebras.wandb.io
```

Enter your API key at the interactive prompt. Do not pass the key as a command argument,
paste it into YAML, or print/read the credential file in logs.
Use `MELLUM_WANDB_NETRC` to override the **container-side** credential-file path if needed.
Restrict access to this file and shared directory. Container deletion does not delete this
shared credential file; revoke the key and remove the file explicitly when decommissioning access.

### Option C: Ephemeral Environment Authentication

If you do not want a persistent login file, enter the key in the coordinator's shell without
putting it in shell history:

```bash
read -r -s -p 'W&B API key: ' WANDB_API_KEY
printf '\n'
export WANDB_API_KEY
bash mw_scripts/mellum2_3xh200/launch.sh --wandb-check
# After container/data preparation and smoke validation:
bash mw_scripts/mellum2_3xh200/launch.sh --train
unset WANDB_API_KEY
```

The launcher forwards this key over SSH stdin and passes it to Docker by environment name;
the value is not included in SSH/docker command arguments or written by the launcher to a file.
Environment-based credentials remain visible to processes with sufficient host/container access.
Do not use shell tracing (`set -x` / `bash -x`) while credentials are set.

### Authentication Check and Resume

After any authentication method, run from the coordinator:

```bash
bash mw_scripts/mellum2_3xh200/launch.sh --wandb-check
```

This checks SDK authentication on both selected nodes without loading the model, scanning
training data, occupying GPUs, or creating a run/project. It does not prove project write permission;
the first online run checks that permission. TensorBoard remains enabled alongside W&B.
Both current containers already include the W&B SDK; no project dependency change is required.

The logging rank saves the current run ID in `metadata/wandb_run.json`:

- **Fresh HF start (`run.resume_checkpoint: null`):** every training launch generates a new ID,
  even if `run_name` and `run.directory` are unchanged. Any previous identity is backed up to
  `metadata/wandb_run.previous.<timestamp>.<new-id>.json`. The launcher sets `WANDB_RESUME=never`
  so this launch cannot append to an existing W&B run.
- **Explicit checkpoint resume:** reuse the saved ID with `WANDB_RESUME=allow`, including after
  a node-count change. The saved entity/project/base URL must still match. If the identity file
  is missing, create a new tracking run and log a warning; the model still loads the checkpoint.

`--prepare`, `--dry-run`, and `--wandb-check` do not create or rotate run IDs.
Changing only the display name does not change the ID during checkpoint resume.
W&B history resume does not roll back metrics already uploaded after an older checkpoint.
If W&B reports that the saved run was deleted, it cannot reuse that ID. A fresh HF launch now
rotates it automatically. For checkpoint resume, stop all workers and rename the identity file
to a backup before relaunching to start a separate W&B history with the same model checkpoint.
ID rotation does not change the prepared mixture, dataset cache, or checkpoint files and requires
no additional `--prepare`. Existing checkpoint safeguards still prohibit a fresh HF start in a
directory containing saved checkpoints; use explicit resume or a separate experiment directory.

Set `wandb.enabled: false` to disable W&B without disabling TensorBoard.
Older YAML files without a `wandb` section remain supported and default to W&B disabled.
Tracking settings are excluded from the immutable training/data digest, so adding W&B does not
invalidate a previously prepared mixture. API keys are never included in YAML or run metadata.
Generated toy training fixtures explicitly disable W&B to avoid creating production tracking runs.

SDK references: [environment variables](https://github.com/wandb/wandb/blob/main/wandb/env.py),
[run identity and resume settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py),
and [login CLI](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py).

## Model Bring-Up Validation

Run inside the container on the first node. First set the same paths as the launcher and writable caches:

```bash
cd /workspace/ws/Megatron-Bridge
export USER=mellum
export UV_CACHE_DIR=/tmp/mellum-validation-uv
export HF_HOME=$PWD/mw_scripts/mellum2_3xh200/.cache/huggingface
source mw_scripts/container_env.sh
```

Single-GPU toy model validation, with a sequence extending past the sliding-window boundary:

```bash
/usr/local/bin/uv run --no-project --python /opt/venv/bin/python python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=29542 mw_scripts/mellum2_3xh200/validate.py \
  --toy-dir /tmp/mellum-toy --output /tmp/mellum-toy-parity.json
```

To validate toy TP/EP sharding with two GPUs, add `--tp 2 --ep 2` and change `--nproc_per_node` to 2.
Single-GPU validation of the real Thinking checkpoint defaults to 1152 tokens,
extending past its 1024-token sliding-window boundary:

```bash
/usr/local/bin/uv run --no-project --python /opt/venv/bin/python python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=29542 mw_scripts/mellum2_3xh200/validate.py \
  --output mw_scripts/mellum2_3xh200/runs/validation/thinking-parity.json
```

The script checks that all exported weights match exactly, then compares logits cosine similarity
(default threshold ≥0.9999) and next-token cross entropy (loss difference <0.02).
It also records the maximum logit difference and argmax agreement.
The HF-compatible reference uses SDPA, executes experts individually, and has no KV cache,
so it is slower; Megatron training uses TE/grouped GEMM.
Routing preserves HF's BF16 gate projection → FP32 softmax/top-k → normalized BF16 weights.
See the [HF Mellum implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mellum/modeling_mellum.py).
When official native HF Mellum support is available, registration gives precedence to that implementation.
Add `--generate-tokens 64` to real-checkpoint validation to generate a code sample using
Megatron full-prefix greedy decoding as a sanity check.
This is not an optimized inference service or a KV-cache throughput benchmark.

Unit tests: `uv run --no-project --python /opt/venv/bin/python python -m pytest tests/unit_tests/models/mellum -q`.

For an end-to-end toy training/resume smoke test through the launcher, use a separate shared directory:

```bash
/usr/local/bin/uv run --no-project --python /opt/venv/bin/python python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=29542 mw_scripts/mellum2_3xh200/validate.py \
  --toy-dir /workspace/ws/Megatron-Bridge/mw_scripts/mellum2_3xh200/runs/validation/toy \
  --toy-training-fixture --output /tmp/mellum-toy-parity.json
```

This option creates a separate toy HF checkpoint, synthetic dataset, and `smoke.local.yaml`;
it does not use real training data.
Back on the host, point `MELLUM_CONFIG` at the generated YAML, then run `--prepare` and `--train`.
After it exits at step 2, set the generated YAML's `resume_checkpoint` to its checkpoint root
and `exit_interval: 4`. Run `--train` again to validate resume.
Do not regenerate the fixture in a toy directory while training is active.

## Future Model Surgery

Copy a new experimental YAML and choose a new run directory first.
Architecture changes require a new layer spec. Place implementations under
`src/megatron/bridge/models/mellum/modeling_mellum/`, rather than modifying the Megatron-LM submodule.
Parallel attention/MoE requires explicit definitions of the inputs read by both branches,
normalization and residual rules, and initialization from the original weights.
After architecture changes, validate against an HF/reference implementation of the new architecture,
not against the original Thinking model's logits.
Parallel HF model directories must carry `architecture_surgery.json`; unknown architecture names are rejected.
