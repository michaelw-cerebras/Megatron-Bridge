#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

# shellcheck source=mw_scripts/container_env.sh
source "$REPO_ROOT/mw_scripts/container_env.sh"

usage() {
  cat <<'EOF'
Train the official GPT-OSS 20B B200 recipe on the RTJ 50B-token mixture.

Usage:
  mw_scripts/gptoss_20b_rtj_50b/gptoss_20B_rtj_50B_train.sh              # background (default)
  mw_scripts/gptoss_20b_rtj_50b/gptoss_20B_rtj_50B_train.sh --foreground
  mw_scripts/gptoss_20b_rtj_50b/gptoss_20B_rtj_50B_train.sh --status
  mw_scripts/gptoss_20b_rtj_50b/gptoss_20B_rtj_50B_train.sh --preflight
  mw_scripts/gptoss_20b_rtj_50b/gptoss_20B_rtj_50B_train.sh --dryrun

Training defaults match the earlier RTJ run: 50B tokens, sequence length 8192,
MBS 1, GBS 128, cosine LR 3e-4 -> 3e-5, 1000 warmup iterations, and the
checked-in Stack25/StackEdu10 mixture. Model/system settings come from the
official gpt_oss_20b B200 FP8-MX v1 performance recipe.

Useful environment overrides:
  GPTOSS_HF_PATH, GPTOSS_DATASET_ROOT, MIXTURE_CONFIG
  RUN_NAME, RUN_DIR, TRAIN_TOKENS, SEQ_LENGTH
  MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE, LR, MIN_LR, LR_WARMUP_ITERS
  SAVE_INTERVAL, LOG_INTERVAL, NUM_WORKERS, ENABLE_NCCL_UB, NCCL_UB_REGISTRATION
  NCCL_NVLS_ENABLE, ALLOW_EXPERIMENTAL_NCCL_UB
  NUMA_BIND_LOCAL_RANK (default: 1 for this two-socket B200 host)

NCCL user-buffer modes:
  ENABLE_NCCL_UB=1 NCCL_UB_REGISTRATION=symmetric  # official registration mode
  ENABLE_NCCL_UB=1 NCCL_UB_REGISTRATION=local      # avoid NCCL window registration

NCCL user buffers require a static allocator. When ENABLE_NCCL_UB=1, this
script deliberately unsets PYTORCH_CUDA_ALLOC_CONF and NCCL_GRAPH_REGISTER.
On this standalone B200 host, leave NCCL_NVLS_ENABLE at its default 0 during
torch.distributed initialization; setting it to 1 hangs there. Megatron-Core
enables NVLS later when it initializes its user-buffer allocator.

This current container/MCore combination also hangs during that later
user-buffer initialization. ENABLE_NCCL_UB=1 is therefore guarded; use
ALLOW_EXPERIMENTAL_NCCL_UB=1 only to re-test after changing the container,
driver, or Megatron-Core revision.
EOF
}

action="background"
case "${1:-}" in
  "") ;;
  --foreground) action="foreground" ;;
  --status) action="status" ;;
  --preflight) action="preflight" ;;
  --dryrun) action="dryrun" ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname -- "$MBRIDGE_ROOT")}"
GPTOSS_HF_PATH="${GPTOSS_HF_PATH:-$WORKSPACE_ROOT/checkpoints/gpt-oss-20b-hf}"
GPTOSS_DATASET_ROOT="${GPTOSS_DATASET_ROOT:-$WORKSPACE_ROOT/data/rtj_pt_50b_gptoss}"
MIXTURE_CONFIG="${MIXTURE_CONFIG:-$SCRIPT_DIR/rtj_pt_50b_stack25_stackedu10.json}"

RUN_NAME="${RUN_NAME:-gptoss_20B_rtj_50B_official_b200}"
RUN_DIR="${RUN_DIR:-$WORKSPACE_ROOT/training_runs/$RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
LOG_DIR="$RUN_DIR/logs"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/train.log}"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-$LOG_DIR/bootstrap.log}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tb_logs}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$RUN_DIR/dataset_cache}"
METADATA_DIR="$RUN_DIR/metadata"
DATA_ARGS_PATH="$METADATA_DIR/megatron_data_args.txt"
RESOLVED_MIXTURE_PATH="$METADATA_DIR/resolved_mixture.json"
MIXTURE_SNAPSHOT_PATH="$METADATA_DIR/mixture_config.json"
RUN_CONFIG_PATH="$METADATA_DIR/launch_config.env"
CONFIG_OUTPUT="$METADATA_DIR/ConfigContainer.yaml"
PID_FILE="${PID_FILE:-$RUN_DIR/train.pid}"

NPROC_PER_NODE=8
TRAIN_TOKENS="${TRAIN_TOKENS:-50000000000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ENABLE_NCCL_UB="${ENABLE_NCCL_UB:-0}"
NCCL_UB_REGISTRATION="${NCCL_UB_REGISTRATION:-symmetric}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
ALLOW_EXPERIMENTAL_NCCL_UB="${ALLOW_EXPERIMENTAL_NCCL_UB:-0}"
NUMA_BIND_LOCAL_RANK="${NUMA_BIND_LOCAL_RANK:-1}"

process_is_this_run() {
  local process_id="$1"
  [[ "$process_id" =~ ^[0-9]+$ ]] \
    && kill -0 "$process_id" 2>/dev/null \
    && ps -p "$process_id" -o args= | grep -Fq "$SCRIPT_PATH"
}

if [[ "$action" == "status" ]]; then
  if [[ -f "$PID_FILE" ]] && process_is_this_run "$(<"$PID_FILE")"; then
    echo "RUNNING: PID=$(<"$PID_FILE")"
    echo "RUN_DIR: $RUN_DIR"
    echo "LOG: $TRAIN_LOG"
    exit 0
  fi
  echo "NOT RUNNING"
  echo "RUN_DIR: $RUN_DIR"
  echo "LOG: $TRAIN_LOG"
  exit 1
fi

if [[ "$action" == "background" ]]; then
  mkdir -p "$LOG_DIR"
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(<"$PID_FILE")"
    if process_is_this_run "$existing_pid"; then
      echo "ALREADY RUNNING: PID=$existing_pid" >&2
      echo "LOG: $TRAIN_LOG" >&2
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  nohup "$SCRIPT_PATH" --foreground >"$BOOTSTRAP_LOG" 2>&1 </dev/null &
  background_pid=$!
  printf '%s\n' "$background_pid" >"$PID_FILE"

  echo "STARTED: PID=$background_pid"
  echo "RUN_DIR: $RUN_DIR"
  echo "CHECKPOINTS: $CHECKPOINT_DIR"
  echo "LOG: $TRAIN_LOG"
  echo "BOOTSTRAP_LOG: $BOOTSTRAP_LOG"
  echo "TENSORBOARD: $TENSORBOARD_DIR"
  echo "FOLLOW: tail -F '$TRAIN_LOG'"
  exit 0
fi

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR" "$DATASET_CACHE_DIR" "$METADATA_DIR"

cleanup_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

for numeric_value in \
  "$TRAIN_TOKENS" \
  "$SEQ_LENGTH" \
  "$MICRO_BATCH_SIZE" \
  "$GLOBAL_BATCH_SIZE" \
  "$LR_WARMUP_ITERS" \
  "$SAVE_INTERVAL" \
  "$LOG_INTERVAL" \
  "$NUM_WORKERS"; do
  if [[ ! "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "STOP: expected a nonnegative integer, got: $numeric_value" >&2
    exit 2
  fi
done
if ((
  TRAIN_TOKENS < 1 || SEQ_LENGTH < 1 || MICRO_BATCH_SIZE < 1
  || GLOBAL_BATCH_SIZE < 1 || SAVE_INTERVAL < 1 || LOG_INTERVAL < 1
)); then
  echo "STOP: token, sequence, batch, save, and log values must be positive." >&2
  exit 2
fi
if [[ "$ENABLE_NCCL_UB" != "0" && "$ENABLE_NCCL_UB" != "1" ]]; then
  echo "STOP: ENABLE_NCCL_UB must be 0 or 1." >&2
  exit 2
fi
if [[ "$NCCL_UB_REGISTRATION" != "symmetric" && "$NCCL_UB_REGISTRATION" != "local" ]]; then
  echo "STOP: NCCL_UB_REGISTRATION must be symmetric or local." >&2
  exit 2
fi
if [[ "$NCCL_NVLS_ENABLE" != "0" && "$NCCL_NVLS_ENABLE" != "1" ]]; then
  echo "STOP: NCCL_NVLS_ENABLE must be 0 or 1." >&2
  exit 2
fi
if [[ "$ALLOW_EXPERIMENTAL_NCCL_UB" != "0" && "$ALLOW_EXPERIMENTAL_NCCL_UB" != "1" ]]; then
  echo "STOP: ALLOW_EXPERIMENTAL_NCCL_UB must be 0 or 1." >&2
  exit 2
fi
if [[ "$NUMA_BIND_LOCAL_RANK" != "0" && "$NUMA_BIND_LOCAL_RANK" != "1" ]]; then
  echo "STOP: NUMA_BIND_LOCAL_RANK must be 0 or 1." >&2
  exit 2
fi
if [[ "$NUMA_BIND_LOCAL_RANK" == "1" && ! -x "$REPO_ROOT/mw_scripts/numa_local_rank_exec.sh" ]]; then
  echo "STOP: NUMA binding wrapper is missing or not executable: $REPO_ROOT/mw_scripts/numa_local_rank_exec.sh" >&2
  exit 2
fi
if [[ "$ENABLE_NCCL_UB" == "1" && "$ALLOW_EXPERIMENTAL_NCCL_UB" != "1" \
  && "$action" != "preflight" && "$action" != "dryrun" ]]; then
  echo "STOP: NCCL user buffers hang on this standalone B200 with the current container/MCore." >&2
  echo "Keep ENABLE_NCCL_UB=0. To deliberately re-test after an environment change, set" >&2
  echo "ALLOW_EXPERIMENTAL_NCCL_UB=1 as well." >&2
  exit 2
fi
if [[ ! -f "$GPTOSS_HF_PATH/config.json" ]]; then
  echo "STOP: GPT-OSS config not found: $GPTOSS_HF_PATH/config.json" >&2
  exit 1
fi
if [[ ! -f "$GPTOSS_DATASET_ROOT/manifest.json" ]]; then
  echo "STOP: tokenized-data manifest not found: $GPTOSS_DATASET_ROOT/manifest.json" >&2
  exit 1
fi
if [[ ! -f "$MIXTURE_CONFIG" ]]; then
  echo "STOP: mixture config not found: $MIXTURE_CONFIG" >&2
  exit 1
fi

export GPT_OSS_20B_HF_MODEL="$GPTOSS_HF_PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NVLINK_DOMAIN_SIZE="${NVLINK_DOMAIN_SIZE:-8}"
export USE_MNNVL="${USE_MNNVL:-0}"
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-1}"
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API="${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-128}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-0}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-CRITICAL}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::UserWarning}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [[ "$ENABLE_NCCL_UB" == "1" ]]; then
  # Megatron-Core's NCCL allocator owns these static pools. Dynamic allocator
  # settings conflict with NCCL user-buffer registration and are removed by
  # the official performance launcher as well.
  unset PYTORCH_CUDA_ALLOC_CONF
  unset NCCL_GRAPH_REGISTER
  export NCCL_CTA_POLICY=1
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,graph_capture_record_stream_reuse:True}"
  export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
fi
export NCCL_NVLS_ENABLE

torchrun_python_mode=()
worker_command=(python)
if [[ "$NUMA_BIND_LOCAL_RANK" == 1 ]]; then
  torchrun_python_mode=(--no-python)
  worker_command=("$REPO_ROOT/mw_scripts/numa_local_rank_exec.sh" python)
fi

uv run --no-sync python - "$GPTOSS_HF_PATH" "$ENABLE_NCCL_UB" "$NCCL_UB_REGISTRATION" "$NCCL_NVLS_ENABLE" <<'PY'
import sys
from importlib import metadata

import torch
import transformer_engine as te
from deep_ep import Buffer, HybridEPBuffer  # noqa: F401
from megatron.core.full_cuda_graph import FullCudaGraphWrapper  # noqa: F401
from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401

model_root = sys.argv[1]
enable_nccl_ub = sys.argv[2] == "1"
nccl_ub_registration = sys.argv[3]
nccl_nvls_enable = sys.argv[4]
names = [torch.cuda.get_device_name(rank) for rank in range(torch.cuda.device_count())]
if len(names) != 8 or any("B200" not in name or "GB200" in name for name in names):
    raise SystemExit(f"Expected exactly 8 NVIDIA B200 GPUs, found: {names}")
if torch.cuda.get_device_capability(0) != (10, 0):
    raise SystemExit(f"Expected SM100, found: {torch.cuda.get_device_capability(0)}")
if enable_nccl_ub:
    if torch.cuda.nccl.version() < (2, 27, 0):
        raise SystemExit(f"NCCL >= 2.27 is required for user-buffer registration; found {torch.cuda.nccl.version()}")
    if not hasattr(torch.cuda, "MemPool"):
        raise SystemExit("This PyTorch build does not expose torch.cuda.MemPool required by NCCL user buffers")
if not hasattr(te.common.recipe, "MXFP8BlockScaling"):
    raise SystemExit("Transformer Engine does not provide MXFP8BlockScaling")

print("Preflight passed")
print(f"  GPUs: 8 x {names[0]} (SM100)")
print(f"  PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; NCCL: {torch.cuda.nccl.version()}")
print(f"  Transformer Engine: {te.__version__}; DeepEP: {metadata.version('deep-ep')}")
print(f"  Local model config: {model_root}/config.json")
if enable_nccl_ub:
    print(
        "  NCCL user buffers: enabled "
        f"({nccl_ub_registration} registration; NVLS at communicator init={nccl_nvls_enable})"
    )
    print("  PyTorch allocator settings: unset for static NCCL user-buffer pools")
else:
    print("  NCCL user buffers: disabled; expandable PyTorch allocator enabled")
PY

if [[ "$action" == "preflight" ]]; then
  exit 0
fi

if [[ "$action" != "dryrun" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  BUSY_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$BUSY_GPU_PROCESSES" ]]; then
    echo "Refusing to start because GPU compute processes are already running:" >&2
    echo "$BUSY_GPU_PROCESSES" >&2
    echo "Stop the existing training run first, or set ALLOW_BUSY_GPUS=1 intentionally." >&2
    exit 3
  fi
fi

resume_run=0
if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" || -f "$CHECKPOINT_DIR/latest_train_state.pt" ]]; then
  resume_run=1
fi

mixture_sha256="$(sha256sum "$MIXTURE_CONFIG" | awk '{print $1}')"
dataset_manifest_sha256="$(sha256sum "$GPTOSS_DATASET_ROOT/manifest.json" | awk '{print $1}')"
current_run_config="$(mktemp "$METADATA_DIR/.launch_config.XXXXXX")"
{
  printf 'SYSTEM_RECIPE=%s\n' 'gpt_oss_20b/b200/fp8_mx/v1'
  printf 'GPTOSS_HF_PATH=%s\n' "$GPTOSS_HF_PATH"
  printf 'GPTOSS_DATASET_ROOT=%s\n' "$GPTOSS_DATASET_ROOT"
  printf 'MIXTURE_SHA256=%s\n' "$mixture_sha256"
  printf 'DATASET_MANIFEST_SHA256=%s\n' "$dataset_manifest_sha256"
  printf 'TRAIN_TOKENS=%s\n' "$TRAIN_TOKENS"
  printf 'SEQ_LENGTH=%s\n' "$SEQ_LENGTH"
  printf 'MICRO_BATCH_SIZE=%s\n' "$MICRO_BATCH_SIZE"
  printf 'GLOBAL_BATCH_SIZE=%s\n' "$GLOBAL_BATCH_SIZE"
  printf 'LR=%s\n' "$LR"
  printf 'MIN_LR=%s\n' "$MIN_LR"
  printf 'LR_WARMUP_ITERS=%s\n' "$LR_WARMUP_ITERS"
  printf 'SAVE_INTERVAL=%s\n' "$SAVE_INTERVAL"
  printf 'LOG_INTERVAL=%s\n' "$LOG_INTERVAL"
  printf 'NUM_WORKERS=%s\n' "$NUM_WORKERS"
  printf 'ENABLE_NCCL_UB=%s\n' "$ENABLE_NCCL_UB"
  printf 'NCCL_UB_REGISTRATION=%s\n' "$NCCL_UB_REGISTRATION"
  printf 'NCCL_NVLS_ENABLE=%s\n' "$NCCL_NVLS_ENABLE"
  printf 'ALLOW_EXPERIMENTAL_NCCL_UB=%s\n' "$ALLOW_EXPERIMENTAL_NCCL_UB"
} >"$current_run_config"

if (( resume_run )); then
  if [[ ! -f "$RUN_CONFIG_PATH" ]]; then
    echo "STOP: checkpoint exists but the frozen launch config is missing: $RUN_CONFIG_PATH" >&2
    rm -f "$current_run_config"
    exit 1
  fi
  if ! cmp -s "$RUN_CONFIG_PATH" "$current_run_config"; then
    echo "STOP: current settings differ from the checkpointed run:" >&2
    diff -u "$RUN_CONFIG_PATH" "$current_run_config" || true
    rm -f "$current_run_config"
    exit 1
  fi
  rm -f "$current_run_config"
elif [[ -f "$RUN_CONFIG_PATH" ]]; then
  if ! cmp -s "$RUN_CONFIG_PATH" "$current_run_config"; then
    echo "STOP: current settings differ from the existing run metadata:" >&2
    diff -u "$RUN_CONFIG_PATH" "$current_run_config" || true
    rm -f "$current_run_config"
    exit 1
  fi
  rm -f "$current_run_config"
else
  mv -f "$current_run_config" "$RUN_CONFIG_PATH"
  cp -f "$MIXTURE_CONFIG" "$MIXTURE_SNAPSHOT_PATH"
fi

uv run --no-sync python -m mw_scripts.gptoss_16l_rtj_50b.build_megatron_mixture \
  --dataset-root "$GPTOSS_DATASET_ROOT" \
  --mixture-config "$MIXTURE_CONFIG" \
  --train-tokens "$TRAIN_TOKENS" \
  --output-data-args "$DATA_ARGS_PATH" \
  --output-manifest "$RESOLVED_MIXTURE_PATH"

tokens_per_iteration=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
train_iters=$(((TRAIN_TOKENS + tokens_per_iteration - 1) / tokens_per_iteration))
scheduled_tokens=$((train_iters * tokens_per_iteration))

echo "RUN_DIR: $RUN_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "TRAIN_LOG: $TRAIN_LOG"
echo "TENSORBOARD_DIR: $TENSORBOARD_DIR"
echo "MIXTURE_CONFIG: $MIXTURE_CONFIG"
echo "RESOLVED_MIXTURE: $RESOLVED_MIXTURE_PATH"
echo "DATA_ARGS: $DATA_ARGS_PATH"
echo "TRAIN_TOKENS_REQUESTED: $TRAIN_TOKENS"
echo "TRAIN_ITERS: $train_iters"
echo "TRAIN_TOKENS_SCHEDULED: $scheduled_tokens"
echo "SYSTEM_RECIPE: gpt_oss_20b/b200/fp8_mx/v1"
echo "NCCL_UB: $ENABLE_NCCL_UB"
echo "NCCL_UB_REGISTRATION: $NCCL_UB_REGISTRATION"
echo "NCCL_NVLS_ENABLE_AT_COMM_INIT: $NCCL_NVLS_ENABLE"
echo "NUMA_BIND_LOCAL_RANK: $NUMA_BIND_LOCAL_RANK"
echo "MODE: $([[ "$resume_run" == 1 ]] && echo resume || echo fresh-random-init)"

train_args=(
  --hf-model "$GPTOSS_HF_PATH"
  --dataset-root "$GPTOSS_DATASET_ROOT"
  --data-args-path "$DATA_ARGS_PATH"
  --train-tokens "$TRAIN_TOKENS"
  --seq-length "$SEQ_LENGTH"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --dataset-cache-dir "$DATASET_CACHE_DIR"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --lr-warmup-iters "$LR_WARMUP_ITERS"
  --save-dir "$CHECKPOINT_DIR"
  --save-interval "$SAVE_INTERVAL"
  --tensorboard-dir "$TENSORBOARD_DIR"
  --log-interval "$LOG_INTERVAL"
  --config-output "$CONFIG_OUTPUT"
)
if (( resume_run )); then
  train_args+=(--resume-checkpoint "$CHECKPOINT_DIR")
fi
if [[ "$ENABLE_NCCL_UB" == "1" ]]; then
  train_args+=(--enable-nccl-ub)
  if [[ "$NCCL_UB_REGISTRATION" == "local" ]]; then
    train_args+=(--disable-symmetric-nccl-ub-registration)
  fi
fi
if [[ "$action" == "dryrun" ]]; then
  train_args+=(--dryrun)
fi

if [[ "$action" == "dryrun" ]]; then
  uv run --no-sync python -m mw_scripts.gptoss_20b_rtj_50b.train "${train_args[@]}"
  exit 0
fi

if (( resume_run )); then
  exec >>"$TRAIN_LOG" 2>&1
  echo
  echo "===== RESUME $(date --iso-8601=seconds) ====="
else
  exec >"$TRAIN_LOG" 2>&1
  echo "===== FRESH RANDOM-INIT RUN $(date --iso-8601=seconds) ====="
fi

set +e
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${torchrun_python_mode[@]}" \
  "${worker_command[@]}" \
  -m mw_scripts.gptoss_20b_rtj_50b.train \
  "${train_args[@]}" \
  2>&1 \
  | sed -u '/^\[WARNING\] Field .* duplicates an ancestor field in /d'
pipeline_status=("${PIPESTATUS[@]}")
set -e

training_exit_code="${pipeline_status[0]}"
if (( training_exit_code == 0 && pipeline_status[1] != 0 )); then
  training_exit_code="${pipeline_status[1]}"
fi

echo "TRAINING_EXIT_CODE: $training_exit_code"
echo "FINISHED_AT: $(date --iso-8601=seconds)"
exit "$training_exit_code"
