# Air-Gapped GPT-OSS Training Smoke Tests

This runbook validates an existing Megatron-Bridge container environment, a
small randomly initialized GPT-OSS-style MoE model, the official GPT-OSS-20B
checkpoint conversion, and one full-model continual-pretraining step.

It assumes that rootless Docker/CDI is already working. See
`mw_scripts/ROOTLESS_DOCKER_CDI.md` for the one-time host-side GPU setup.

The GPU-host commands and training tests run offline. Image staging, repository
cloning, and the optional checkpoint download must run on a separate connected
engineering machine before their artifacts cross the air gap. The tests use
the Python environment already installed in the container and do not download
or synchronize packages.

## 1. Prepare the repository and start the GPU container

### 1.1 Stage the NeMo 26.06 image

Pull the base image on a connected engineering machine, not on the air-gapped
GPU host:

```bash
docker pull nvcr.io/nvidia/nemo:26.06
```

Mirror that image into the approved internal registry, or transfer it with the
site-approved offline image workflow. On the GPU host, set `CONTAINER_IMAGE`
to the resulting local or internal image tag and confirm that it exists:

```bash
export CONTAINER_IMAGE="nvcr.io/nvidia/nemo:26.06"
docker image inspect "$CONTAINER_IMAGE" >/dev/null
```

### 1.2 Clone the Bridge release baseline

Do this on a connected engineering machine rather than an `oly-dev` host.
Replace `YOUR_GITHUB_ORG_OR_USER` with the owner of the development fork:

```bash
cd "$HOME/ws"

git clone --recurse-submodules \
  git@github.com:YOUR_GITHUB_ORG_OR_USER/Megatron-Bridge.git

cd Megatron-Bridge

git remote add upstream \
  https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
git remote set-url --push upstream DISABLED

git fetch upstream --tags
git switch -c gptoss-dev v0.5.0
git submodule update --init --recursive
```

Megatron Bridge `v0.5.0` is the reproducible baseline for this runbook because
the [official NeMo 26.06 software matrix](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.5.0/docs/releases/software-versions.md)
pairs it with Megatron Core 0.18.0 and Transformer Engine 2.16. It includes
GPT-OSS support and the 26.06 MoE and GB200 performance stack. Avoid creating
the development branch from the fork's default branch, because that may
silently select a newer source and dependency combination.

Transfer or synchronize the resulting repository to the air-gapped GPU
workspace through the approved mechanism.

### 1.3 Start the container as container root

Run the following on the GPU host. Confirm that the host GPU and rootless
Docker/CDI setup work before creating the container:

```bash
nvidia-smi -L
docker context show

export HOST_USER="$(id -un)"
export HOST_WS_ROOT="$(readlink -f "$HOME/ws")"
export CONTAINER_NAME="megatron_bridge_mw"

test -d "$HOST_WS_ROOT/Megatron-Bridge"
```

`docker context show` should print `rootless`. Start the container with native
CDI device injection:

```bash
docker run -it \
  --network=host \
  --runtime=runc \
  --device=nvidia.com/gpu=all \
  --name "$CONTAINER_NAME" \
  --shm-size=64g \
  -e HOME=/workspace \
  -e USER="$HOST_USER" \
  -e PYTHONNOUSERSITE=1 \
  -e NCCL_DEBUG=WARN \
  -e XDG_CACHE_HOME=/tmp/mbridge-cache \
  -e HF_HOME=/tmp/mbridge-hf \
  -e PIP_CACHE_DIR=/tmp/mbridge-pip-cache \
  -e FLASHINFER_WORKSPACE_BASE=/tmp/mbridge-flashinfer \
  -e MPLCONFIGDIR=/tmp/mbridge-matplotlib \
  -e MEGATRON_CONFIG_LOCK_DIR=/tmp/mbridge-locks \
  -v "$HOME:/workspace" \
  -v "$HOST_WS_ROOT:/workspace/ws" \
  -w /workspace \
  "$CONTAINER_IMAGE" \
  bash
```

Rootless Docker on the validated host rejected attempts to raise `memlock` and
`stack` limits, so this command deliberately omits `--ulimit memlock=-1` and
`--ulimit stack=67108864`.

The prompt identifies this shell as `root`, but under rootless Docker it is
root only inside the container's user namespace. Its bind-mount writes map
back to the host user. Do not create or select a container user with the host's
large directory-service UID: that UID may be outside the rootless namespace
mapping and fail with `cannot setuid to unmapped uid`.

There is therefore no need to detach, create `/etc/passwd` entries, and enter
again as the host UID. Continue directly to Section 2 in this shell.

To leave the container running, detach with `Ctrl-p`, then `Ctrl-q`. Re-enter
the existing container later with:

```bash
export CONTAINER_NAME="megatron_bridge_gptoss"

docker start "$CONTAINER_NAME" >/dev/null

docker exec -it   -e USER="$(id -un)"   $CONTAINER_NAME   bash
```

## 2. Environment and import smoke test

The launchers source `mw_scripts/container_env.sh`, which configures the
working directory, local-source `PYTHONPATH`, and writable cache directories
automatically. Source the same helper before running ad-hoc Python or debugging
commands interactively. It configures the current shell and does not install
packages. The air-gapped smoke tests use the Python environment already present
in the NeMo image with `uv run --no-sync`, so do not run `uv sync` unless all
required artifacts are available offline.

Set the four input paths to the locations mounted into the container:

```bash
export MBRIDGE_ROOT="/workspace/ws/Megatron-Bridge"
export GPTOSS_DATASET_ROOT="/workspace/ws/data/ultrafineweb_l1_en_hq_gptoss"
export GPTOSS_HF_PATH="/workspace/ws/checkpoints/gpt-oss-20b-hf"
export GPTOSS_MCORE_PATH="/workspace/ws/checkpoints/gpt-oss-20b-megatron-ep8"

source "$MBRIDGE_ROOT/mw_scripts/container_env.sh"
```

Verify Python, the mounted Bridge source tree, the local Megatron-LM submodule,
PyTorch CUDA, and a real GPU allocation:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python - <<'PY'
import os
import sys

import torch
import megatron.bridge
import megatron.core

bridge_root = os.environ["MBRIDGE_ROOT"]

print("Python:", sys.executable)
print("Bridge:", megatron.bridge.__file__)
print("MCore:", megatron.core.__file__)
print("Torch:", torch.__version__)
print("Container CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("CUDA allocation:", torch.ones(1, device="cuda"))

assert f"{bridge_root}/src/" in megatron.bridge.__file__
assert f"{bridge_root}/3rdparty/Megatron-LM/" in megatron.core.__file__
assert torch.cuda.is_available()

print("Megatron-Bridge environment: OK")
PY
```

Success criterion:

```text
Megatron-Bridge environment: OK
```

Warnings from optional TVM, Triton, NIXL, ModelOpt, or telemetry components can
appear during import. They are not a failure when the final assertion passes.

## 3. Random-init custom GPT-OSS MoE training and throughput

This test runs the custom GPT-OSS-style MoE architecture defined in
`mw_scripts/mini_gptoss.py` from random initialization. Despite the script
name, this is no longer a tiny model. Its main architecture is:

- 16 Transformer layers with hidden size 2880.
- 32 attention heads, one KV head, and head dimension 128.
- 64 routed experts per layer, top-16 routing, and expert intermediate size
  1440.
- One shared expert per layer with intermediate size 1440.
- Sliding-window attention in every layer with a 32768-token window.
- YaRN scaling factor 64 with an original context length of 4096 and maximum
  configured context length of 262144.
- Untied input embeddings and output weights over a vocabulary of 201088
  tokens.

MCore currently requires `add_bias_linear=False` whenever shared experts are
enabled. This throughput model therefore preserves the requested attention
bias on QKV projections with `add_qkv_bias=True`, but omits attention-output
and routed/shared-expert linear biases. It also disables bias-activation fusion
so the shared-expert quick-GELU path works without a bias tensor. Supporting
the exact mixed bias layout would require a model-specific shared-expert
implementation rather than the stock MCore module.

The routed-expert matrices alone contain approximately 12.7 billion
parameters. The default launch therefore uses all eight GPUs with expert
parallelism of eight. It initializes the model randomly and does not load the
official GPT-OSS checkpoint.

Preflight the real tokenized dataset:

```bash
ls -lh \
  "$GPTOSS_DATASET_ROOT/manifest.json" \
  "$GPTOSS_DATASET_ROOT/data_prefixes.txt"

head "$GPTOSS_DATASET_ROOT/data_prefixes.txt"
```

### 3.1 BF16 baseline

Run 2000 BF16 training steps at sequence length 8192:

```bash
cd "$MBRIDGE_ROOT"
set -o pipefail

BF16_RUN_DIR="${BF16_RUN_DIR:-/scratch/$USER/gptoss-custom-random-init-bf16}"
BF16_TENSORBOARD_DIR="$BF16_RUN_DIR/tb_logs"
mkdir -p "$BF16_TENSORBOARD_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  mw_scripts/mini_gptoss.py \
  --hf-model "$GPTOSS_HF_PATH" \
  --dataset-root "$GPTOSS_DATASET_ROOT" \
  --train-iters 2000 \
  --seq-length 8192 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --tensor-parallel 1 \
  --pipeline-parallel 1 \
  --expert-parallel 8 \
  --expert-tensor-parallel 1 \
  --precision bf16 \
  --recompute \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --lr-warmup-iters 200 \
  --lr-decay-iters 2000 \
  --lr-decay-style cosine \
  --log-interval 10 \
  --tensorboard-dir "$BF16_TENSORBOARD_DIR" \
  --num-workers 2 \
  2>&1 | tee "$BF16_RUN_DIR/train.log"

echo "training_exit_code=${PIPESTATUS[0]}"
```

### 3.2 FP8 current-scaling training

Use this command on Hopper or newer NVIDIA GPUs with a compatible Transformer
Engine build. It keeps the optimizer and numerically sensitive operations in
the recipe's BF16/FP32 precision while running supported Transformer Engine
linear operations in FP8. The model script maps `--precision fp8-current` to
`bf16_with_fp8_current_scaling_mixed` and enables MoE router padding for FP8
alignment.

Run the same 2000-step workload in FP8, using a separate output directory so
its TensorBoard events do not overwrite or mix with the BF16 baseline:

```bash
cd "$MBRIDGE_ROOT"
set -o pipefail

FP8_RUN_DIR="${FP8_RUN_DIR:-/scratch/$USER/gptoss-custom-random-init-fp8-current}"
FP8_TENSORBOARD_DIR="$FP8_RUN_DIR/tb_logs"
mkdir -p "$FP8_TENSORBOARD_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  mw_scripts/mini_gptoss.py \
  --hf-model "$GPTOSS_HF_PATH" \
  --dataset-root "$GPTOSS_DATASET_ROOT" \
  --train-iters 2000 \
  --seq-length 8192 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --tensor-parallel 1 \
  --pipeline-parallel 1 \
  --expert-parallel 8 \
  --expert-tensor-parallel 1 \
  --precision fp8-current \
  --recompute \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --lr-warmup-iters 200 \
  --lr-decay-iters 2000 \
  --lr-decay-style cosine \
  --log-interval 10 \
  --tensorboard-dir "$FP8_TENSORBOARD_DIR" \
  --num-workers 2 \
  2>&1 | tee "$FP8_RUN_DIR/train.log"

echo "training_exit_code=${PIPESTATUS[0]}"
```

Before committing to the full run, it is reasonable to change
`--train-iters`, `--lr-warmup-iters`, and `--lr-decay-iters` to 20, 2, and 20
respectively for a short FP8 smoke test. Check that Transformer Engine reports
FP8 execution and that the loss and gradient norms remain finite.

On Blackwell GPUs, `mini_gptoss.py` also supports MXFP8 block scaling. To test
it, use a distinct run directory and replace only:

```bash
FP8_RUN_DIR="/scratch/$USER/gptoss-custom-random-init-mxfp8"
# In the mini_gptoss.py arguments:
--precision mxfp8
```

MXFP8 requires Blackwell-generation hardware and a Transformer Engine build
with MXFP8 support. Do not use `mxfp8` as a fallback on Hopper; use
`fp8-current` there.

With TP=1 and PP=1, the dense data-parallel size is eight. Therefore
`micro_batch_size=1` and `global_batch_size=8` use one microbatch per optimizer
step without gradient accumulation. The run processes 131,072,000 training
tokens in total. The learning rate warms up linearly from zero to `3e-4` over
the first 200 steps, then follows cosine decay to `3e-5` by step 2000.

`BF16_TENSORBOARD_DIR` and `FP8_TENSORBOARD_DIR` store only TensorBoard event
files; checkpoint saving and loading remain disabled. From a second shell on
the training host, compare the BF16 and FP8 runs in one TensorBoard process
with:

```bash
uv run --no-sync tensorboard \
  --logdir_spec \
    bf16:/scratch/$USER/gptoss-custom-random-init-bf16/tb_logs,fp8:/scratch/$USER/gptoss-custom-random-init-fp8-current/tb_logs \
  --host 0.0.0.0 \
  --port 6006
```

If the training host is remote, forward port 6006 through SSH and open
`http://localhost:6006` locally. The dashboard includes loss, learning rate,
batch size, iteration timers, and per-GPU throughput. TensorBoard files do not
require a checkpoint directory.

Full activation recomputation is enabled because the 8192-token sequence and
top-16 MoE routing create substantial activation pressure. Expandable CUDA
allocator segments reduce the chance of a borderline failure caused by memory
fragmentation.

For a model-only throughput measurement that removes dataset I/O from the
comparison, omit `--dataset-root` and `--num-workers`; the recipe then uses mock
data. A sequence length of 8192 does not exercise the 32768-token window
truncation because the complete sequence fits inside the sliding window.

Verify and record the following before treating the run as a throughput
baseline:

- All 2000 iterations complete with exit code zero and without NaN or skipped
  iterations.
- For FP8, the logs confirm that the requested FP8 recipe is active; do not
  infer FP8 execution only from a faster iteration time.
- Initial language-model loss is near `ln(201088)`, approximately 12.21, as
  expected for random predictions. The router auxiliary loss is configured
  separately with coefficient 0.9.
- The logged model configuration reports 16 layers, 64 experts, top-16 routing,
  one shared expert, and the requested parallelism.
- Record peak allocated/reserved memory per GPU.
- Ignore the first 5 to 10 iterations when calculating steady-state throughput
  because model initialization, kernel loading, and compilation can dominate
  early steps.
- Record steady-state `elapsed time per iteration (ms)` and `throughput per GPU
  (TFLOP/s/GPU)`. Compute aggregate token throughput as
  `global_batch_size * sequence_length / iteration_time_seconds`.
- Compare BF16 and FP8 loss, gradient norms, skipped iterations, peak memory,
  and steady-state tokens/second using the same data, seed, batch size, and
  sequence length.

No validated timing or memory baseline is recorded here yet; populate those
values only after running this exact architecture on the target GPU type.

## 4. Validate the official GPT-OSS-20B HF checkpoint

The HF directory should contain `config.json`, tokenizer files,
`model.safetensors.index.json`, and three root-level safetensors shards. The
original validated MXFP4 checkpoint occupied approximately 13 GB.

Verify that every shard referenced by the index exists:

```bash
uv run --no-sync python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["GPTOSS_HF_PATH"])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file()]

print("tensor entries:", len(index["weight_map"]))
print("weight shards:", shards)
print("missing shards:", missing)
assert not missing
print("HF checkpoint structure: OK")
PY
```

Confirm that the checkpoint contains the MXFP4 block/scale pairs expected by
`GPTOSSBridge` without loading the tensor payloads:

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

from safetensors import safe_open

root = Path(os.environ["GPTOSS_HF_PATH"])
block_keys = []
scale_keys = []

for shard in sorted(root.glob("model-*.safetensors")):
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        block_keys.extend(key for key in keys if key.endswith("_blocks"))
        scale_keys.extend(key for key in keys if key.endswith("_scales"))
        print(shard.name, "tensor count:", len(keys))

print("MXFP4 block tensors:", len(block_keys))
print("MXFP4 scale tensors:", len(scale_keys))
print("example block:", block_keys[0] if block_keys else None)
print("example scale:", scale_keys[0] if scale_keys else None)

assert block_keys
assert len(block_keys) == len(scale_keys)
print("GPT-OSS MXFP4 structure: OK")
PY
```

The validated checkpoint contained 459 indexed tensors and 48 MXFP4
block/scale pairs.

## 5. Convert official MXFP4 weights to a BF16 Megatron checkpoint

The conversion uses eight GPUs with `EP=8`. `GPTOSSBridge` dequantizes MXFP4
expert weights to BF16 while mapping the HF tensors to Megatron.

The output directory must not already exist:

```bash
test ! -e "$GPTOSS_MCORE_PATH" \
  && echo "output path is ready" \
  || echo "STOP: output path already exists"
```

Run the conversion only when the preflight prints `output path is ready`:

```bash
cd "$MBRIDGE_ROOT"
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  examples/conversion/convert_checkpoints_multi_gpu.py import \
  --hf-model "$GPTOSS_HF_PATH" \
  --megatron-path "$GPTOSS_MCORE_PATH" \
  --tp 1 \
  --pp 1 \
  --ep 8 \
  --etp 1 \
  --torch-dtype bfloat16 \
  --distributed-timeout-minutes 60 \
  2>&1 | tee /tmp/gptoss-hf-to-megatron.log

echo "conversion_exit_code=${PIPESTATUS[0]}"
```

Inspect the result:

```bash
du -sh "$GPTOSS_MCORE_PATH"

find "$GPTOSS_MCORE_PATH" \
  -maxdepth 2 \
  -type f \
  | sort \
  | head -30
```

Observed success evidence from the validated conversion:

- `GPTOSSBridge` loaded and mapped 603 of 603 conversion items.
- The log printed `Import complete` and the exit code was zero.
- The BF16 Megatron checkpoint occupied approximately 40 GB.
- The checkpoint contained eight `.distcp` shards plus `.metadata`,
  `run_config.yaml`, `train_state.pt`, and latest-checkpoint tracker files.

## 6. Full GPT-OSS-20B continual pretraining for 2000 steps

This validated run loads the converted BF16 GPT-OSS-20B checkpoint, creates a
fresh optimizer and RNG state, and continues pretraining on the real
UltraFineWeb Megatron dataset. It uses eight H100 GPUs, BF16 mixed precision,
`TP=2`, `PP=1`, `EP=8`, sequence length 512, and full activation
recomputation.

The wrapper owns the full launch lifecycle: it runs an iteration-zero baseline
evaluation for a fresh run, starts training with `nohup`, writes TensorBoard
events, saves every 500 steps, and resumes model, optimizer, scheduler, RNG,
and consumed-sample state when a valid checkpoint already exists.

```bash
cd "$MBRIDGE_ROOT"

export CPT_RUN_NAME="gpt-oss-20b-ultrafineweb-cpt-lr2e-6-pilot-2k"
export CPT_RUN_DIR="/scratch/$USER/checkpoints/$CPT_RUN_NAME"

bash mw_scripts/continue_gptoss.sh
tail -f mw_scripts/train.log
```

The corresponding training configuration in `continue_gptoss.sh` is:

- 2000 optimizer steps with `micro_batch_size=1` and `global_batch_size=16`.
- Sequence length 512, for 16,384,000 training tokens in total.
- Peak learning rate `2e-6`, 500 linear warmup steps, cosine decay, and minimum
  learning rate `2e-7`.
- Evaluation every 100 steps with 20 evaluation iterations.
- Checkpoints at steps 500, 1000, 1500, and 2000 under
  `$CPT_RUN_DIR/checkpoints`.
- TensorBoard events under `$CPT_RUN_DIR/tb_logs`.

Start TensorBoard from a second shell:

```bash
cd "$MBRIDGE_ROOT"

uv run --no-sync tensorboard \
  --logdir "/scratch/$USER/checkpoints/gpt-oss-20b-ultrafineweb-cpt-lr2e-6-pilot-2k/tb_logs" \
  --host 127.0.0.1 \
  --port 6006 \
  --reload_interval 30
```

Forward port 6006 through SSH when viewing from another machine. The training
log for the validated run is `mw_scripts/train.log`.

Observed success evidence from the run completed on August 31, 2026:

- The iteration-zero baseline evaluation completed with exit code zero;
  validation loss was `4.131616` and test loss was `4.234321`.
- Iteration 2000 completed after consuming 32,000 samples, with training LM
  loss `1.977195`, grad norm `4.749`, and no skipped or NaN iterations.
- The final validation loss was `1.953116` and final test loss was `1.930499`.
- Steady-state iterations near the end took approximately 1.94--2.03 seconds.
- The distributed checkpoints at steps 500, 1000, 1500, and 2000 all saved
  successfully; the step-2000 save completed before the final evaluation.
- The run ended with `training_exit_code=0`.

For a fresh run, choose a new `CPT_RUN_NAME`. If the run directory exists but
does not contain `checkpoints/latest_checkpointed_iteration.txt`, the wrapper
stops instead of overwriting or silently mixing run state.

## Optional: download the HF checkpoint on a connected host

For an air-gapped deployment, run this on a connected host and transfer the
result through the approved mechanism. Excluding both `original/` and `metal/`
avoids downloading duplicate 13.8 GB weight representations.

```bash
export HF_HOME="$HOME/hf-download-cache"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export GPTOSS_HF_DOWNLOAD="$HOME/gpt-oss-20b-hf"

mkdir -p "$HF_HOME" "$GPTOSS_HF_DOWNLOAD"

hf download openai/gpt-oss-20b \
  --local-dir "$GPTOSS_HF_DOWNLOAD" \
  --exclude "original/*" \
  --exclude "metal/*" \
  --dry-run

hf download openai/gpt-oss-20b \
  --local-dir "$GPTOSS_HF_DOWNLOAD" \
  --exclude "original/*" \
  --exclude "metal/*"
```

## Final validation summary

The complete validated flow is:

```text
container Python/CUDA/import check
  -> random-init 234M GPT-OSS-style MoE on real data
  -> official GPT-OSS-20B MXFP4 validation
  -> distributed BF16 Megatron checkpoint conversion
  -> full GPT-OSS-20B real-data continual-pretraining optimizer step
```

All stages were validated in an air-gapped environment, making the workflow a
direct starting point for similarly isolated GB200 clusters.
