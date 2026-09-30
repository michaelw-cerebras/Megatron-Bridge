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
Continue GPT-OSS 20B pretraining on a weighted CodeSign data mixture.

Before the first launch, edit codesign_mixture.json and set one or more
source_weights above zero. Positive values are relative weights and are
normalized automatically; zero disables a source.

Usage:
  mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh              # background
  mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --foreground
  mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --preflight
  mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --dryrun
  mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --status

Defaults:
  8 x B200, official gpt_oss_20b B200 FP8-MX v1 recipe
  50B training tokens, sequence length 8192, MBS 1, GBS 128
  cosine LR 2e-6 -> 2e-7 with 1000 warmup iterations
  initialize from /workspace/ws/checkpoints/gpt-oss-20b-megatron-ep8
  save every 1000 iterations and retain the latest 2 checkpoints

Useful overrides:
  GPTOSS_HF_PATH, GPTOSS_MCORE_PATH, MIXTURE_CONFIG
  RUN_NAME, RUN_DIR, TRAIN_TOKENS, SEQ_LENGTH
  MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE, LR, MIN_LR, LR_WARMUP_ITERS
  SAVE_INTERVAL, KEEP_LAST_CHECKPOINTS, LOG_INTERVAL, NUM_WORKERS
  ENABLE_NCCL_UB, NCCL_UB_REGISTRATION, ALLOW_EXPERIMENTAL_NCCL_UB
  NUMA_BIND_LOCAL_RANK, ALLOW_BUSY_GPUS
EOF
}

action="background"
case "${1:-}" in
  "") ;;
  --foreground) action="foreground" ;;
  --preflight) action="preflight" ;;
  --dryrun) action="dryrun" ;;
  --status) action="status" ;;
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

GPTOSS_HF_PATH="${GPTOSS_HF_PATH:-/workspace/ws/checkpoints/gpt-oss-20b-hf}"
GPTOSS_MCORE_PATH="${GPTOSS_MCORE_PATH:-/workspace/ws/checkpoints/gpt-oss-20b-megatron-ep8}"
MIXTURE_CONFIG="${MIXTURE_CONFIG:-$SCRIPT_DIR/codesign_mixture.json}"

RUN_NAME="${RUN_NAME:-gptoss_20B_codesign_cpt}"
RUN_DIR="${RUN_DIR:-/scratch/$RUNTIME_USER/training_runs/$RUN_NAME}"
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
LR="${LR:-2e-6}"
MIN_LR="${MIN_LR:-2e-7}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
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
  "$KEEP_LAST_CHECKPOINTS" \
  "$LOG_INTERVAL" \
  "$NUM_WORKERS"; do
  if [[ ! "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "STOP: expected a nonnegative integer, got: $numeric_value" >&2
    exit 2
  fi
done
if ((
  TRAIN_TOKENS < 1 || SEQ_LENGTH < 1 || MICRO_BATCH_SIZE < 1
  || GLOBAL_BATCH_SIZE < 1 || SAVE_INTERVAL < 1
  || KEEP_LAST_CHECKPOINTS < 1 || LOG_INTERVAL < 1
)); then
  echo "STOP: token, sequence, batch, save, retention, and log values must be positive." >&2
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
if [[ "$ALLOW_EXPERIMENTAL_NCCL_UB" != "0" && "$ALLOW_EXPERIMENTAL_NCCL_UB" != "1" ]]; then
  echo "STOP: ALLOW_EXPERIMENTAL_NCCL_UB must be 0 or 1." >&2
  exit 2
fi
if [[ "$NUMA_BIND_LOCAL_RANK" != "0" && "$NUMA_BIND_LOCAL_RANK" != "1" ]]; then
  echo "STOP: NUMA_BIND_LOCAL_RANK must be 0 or 1." >&2
  exit 2
fi
if [[ "$NUMA_BIND_LOCAL_RANK" == "1" && ! -x "$REPO_ROOT/mw_scripts/numa_local_rank_exec.sh" ]]; then
  echo "STOP: missing executable NUMA wrapper: $REPO_ROOT/mw_scripts/numa_local_rank_exec.sh" >&2
  exit 2
fi
if [[ "$ENABLE_NCCL_UB" == "1" && "$ALLOW_EXPERIMENTAL_NCCL_UB" != "1" \
  && "$action" != "preflight" && "$action" != "dryrun" ]]; then
  echo "STOP: NCCL user buffers are experimental on this standalone B200 setup." >&2
  echo "Keep ENABLE_NCCL_UB=0, or also set ALLOW_EXPERIMENTAL_NCCL_UB=1 intentionally." >&2
  exit 2
fi
if [[ ! -f "$GPTOSS_HF_PATH/config.json" || ! -f "$GPTOSS_HF_PATH/tokenizer.json" ]]; then
  echo "STOP: incomplete GPT-OSS HF directory: $GPTOSS_HF_PATH" >&2
  exit 1
fi
if [[ ! -f "$GPTOSS_MCORE_PATH/latest_checkpointed_iteration.txt" \
  && ! -f "$GPTOSS_MCORE_PATH/latest_train_state.pt" ]]; then
  echo "STOP: pretrained Megatron checkpoint is incomplete: $GPTOSS_MCORE_PATH" >&2
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
export NCCL_NVLS_ENABLE
if [[ "$ENABLE_NCCL_UB" == "1" ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
  unset NCCL_GRAPH_REGISTER
  export NCCL_CTA_POLICY=1
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,graph_capture_record_stream_reuse:True}"
  export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
fi

resume_run=0
if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" || -f "$CHECKPOINT_DIR/latest_train_state.pt" ]]; then
  resume_run=1
fi

if [[ "$action" == "foreground" ]]; then
  if (( resume_run )); then
    exec >>"$TRAIN_LOG" 2>&1
    echo
    echo "===== RESUME $(date --iso-8601=seconds) ====="
  else
    exec >"$TRAIN_LOG" 2>&1
    echo "===== FRESH CONTINUAL-PRETRAINING RUN $(date --iso-8601=seconds) ====="
  fi
fi

mixture_sha256="$(sha256sum "$MIXTURE_CONFIG" | awk '{print $1}')"
current_run_config="$(mktemp "$METADATA_DIR/.launch_config.XXXXXX")"
{
  printf 'SYSTEM_RECIPE=%s\n' 'gpt_oss_20b/b200/fp8_mx/v1'
  printf 'GPTOSS_HF_PATH=%s\n' "$GPTOSS_HF_PATH"
  printf 'GPTOSS_MCORE_PATH=%s\n' "$GPTOSS_MCORE_PATH"
  printf 'MIXTURE_SHA256=%s\n' "$mixture_sha256"
  printf 'TRAIN_TOKENS=%s\n' "$TRAIN_TOKENS"
  printf 'SEQ_LENGTH=%s\n' "$SEQ_LENGTH"
  printf 'MICRO_BATCH_SIZE=%s\n' "$MICRO_BATCH_SIZE"
  printf 'GLOBAL_BATCH_SIZE=%s\n' "$GLOBAL_BATCH_SIZE"
  printf 'LR=%s\n' "$LR"
  printf 'MIN_LR=%s\n' "$MIN_LR"
  printf 'LR_WARMUP_ITERS=%s\n' "$LR_WARMUP_ITERS"
  printf 'SAVE_INTERVAL=%s\n' "$SAVE_INTERVAL"
  printf 'KEEP_LAST_CHECKPOINTS=%s\n' "$KEEP_LAST_CHECKPOINTS"
  printf 'LOG_INTERVAL=%s\n' "$LOG_INTERVAL"
  printf 'NUM_WORKERS=%s\n' "$NUM_WORKERS"
  printf 'ENABLE_NCCL_UB=%s\n' "$ENABLE_NCCL_UB"
  printf 'NCCL_UB_REGISTRATION=%s\n' "$NCCL_UB_REGISTRATION"
} >"$current_run_config"

if (( resume_run )); then
  if [[ ! -f "$RUN_CONFIG_PATH" ]]; then
    echo "STOP: checkpoint exists but frozen launch configuration is missing: $RUN_CONFIG_PATH" >&2
    rm -f "$current_run_config"
    exit 1
  fi
  if ! cmp -s "$RUN_CONFIG_PATH" "$current_run_config"; then
    echo "STOP: current settings differ from the checkpointed run:" >&2
    diff -u "$RUN_CONFIG_PATH" "$current_run_config" || true
    rm -f "$current_run_config"
    echo "Use a new RUN_NAME/RUN_DIR for a different mixture or training setup." >&2
    exit 1
  fi
  rm -f "$current_run_config"
  if [[ ! -f "$DATA_ARGS_PATH" || ! -f "$RESOLVED_MIXTURE_PATH" ]]; then
    echo "STOP: checkpoint exists but frozen mixture artifacts are missing from $METADATA_DIR" >&2
    exit 1
  fi
else
  uv run --no-sync python -m mw_scripts.gptoss_20b_codesign_cpt.build_megatron_mixture \
    --mixture-config "$MIXTURE_CONFIG" \
    --train-tokens "$TRAIN_TOKENS" \
    --output-data-args "$DATA_ARGS_PATH" \
    --output-manifest "$RESOLVED_MIXTURE_PATH"
  mv -f "$current_run_config" "$RUN_CONFIG_PATH"
  cp -f "$MIXTURE_CONFIG" "$MIXTURE_SNAPSHOT_PATH"
fi

uv run --no-sync python - "$GPTOSS_HF_PATH" "$GPTOSS_MCORE_PATH" <<'PY'
import sys
from importlib import metadata
from pathlib import Path

import torch
import transformer_engine as te
from deep_ep import Buffer, HybridEPBuffer  # noqa: F401
from megatron.core.full_cuda_graph import FullCudaGraphWrapper  # noqa: F401
from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401

hf_root = Path(sys.argv[1])
mcore_root = Path(sys.argv[2])
names = [torch.cuda.get_device_name(rank) for rank in range(torch.cuda.device_count())]
if len(names) != 8 or any("B200" not in name or "GB200" in name for name in names):
    raise SystemExit(f"Expected exactly 8 NVIDIA B200 GPUs, found: {names}")
if torch.cuda.get_device_capability(0) != (10, 0):
    raise SystemExit(f"Expected SM100, found: {torch.cuda.get_device_capability(0)}")
if not hasattr(te.common.recipe, "MXFP8BlockScaling"):
    raise SystemExit("Transformer Engine does not provide MXFP8BlockScaling")
if not (hf_root / "tokenizer.json").is_file():
    raise SystemExit(f"Missing tokenizer: {hf_root / 'tokenizer.json'}")
if not (mcore_root / "latest_checkpointed_iteration.txt").is_file():
    raise SystemExit(f"Missing checkpoint tracker: {mcore_root / 'latest_checkpointed_iteration.txt'}")

print("Preflight passed")
print(f"  GPUs: 8 x {names[0]} (SM100)")
print(f"  PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; NCCL: {torch.cuda.nccl.version()}")
print(f"  Transformer Engine: {te.__version__}; DeepEP: {metadata.version('deep-ep')}")
print(f"  HF model: {hf_root}")
print(f"  pretrained Megatron checkpoint: {mcore_root}")
PY

if [[ "$action" == "preflight" ]]; then
  echo "MIXTURE: $RESOLVED_MIXTURE_PATH"
  echo "DATA_ARGS: $DATA_ARGS_PATH"
  exit 0
fi

if [[ "$action" != "dryrun" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  busy_gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$busy_gpu_processes" ]]; then
    echo "Refusing to start because GPU compute processes are already running:" >&2
    echo "$busy_gpu_processes" >&2
    echo "Stop them first, or set ALLOW_BUSY_GPUS=1 intentionally." >&2
    exit 3
  fi
fi

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
echo "MODE: $([[ "$resume_run" == 1 ]] && echo full-state-resume || echo initialize-from-pretrained)"

train_args=(
  --hf-model "$GPTOSS_HF_PATH"
  --data-args-path "$DATA_ARGS_PATH"
  --resolved-mixture "$RESOLVED_MIXTURE_PATH"
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
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --tensorboard-dir "$TENSORBOARD_DIR"
  --log-interval "$LOG_INTERVAL"
  --config-output "$CONFIG_OUTPUT"
)
if (( resume_run )); then
  train_args+=(--resume-checkpoint "$CHECKPOINT_DIR")
else
  train_args+=(--pretrained-checkpoint "$GPTOSS_MCORE_PATH")
fi
if [[ "$ENABLE_NCCL_UB" == "1" ]]; then
  train_args+=(--enable-nccl-ub)
  if [[ "$NCCL_UB_REGISTRATION" == "local" ]]; then
    train_args+=(--disable-symmetric-nccl-ub-registration)
  fi
fi
if [[ "$action" == "dryrun" ]]; then
  train_args+=(--dryrun)
  uv run --no-sync python -m mw_scripts.gptoss_20b_codesign_cpt.train "${train_args[@]}"
  exit 0
fi

torchrun_python_mode=()
worker_command=(python)
if [[ "$NUMA_BIND_LOCAL_RANK" == "1" ]]; then
  torchrun_python_mode=(--no-python)
  worker_command=("$REPO_ROOT/mw_scripts/numa_local_rank_exec.sh" python)
fi

set +e
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${torchrun_python_mode[@]}" \
  "${worker_command[@]}" \
  -m mw_scripts.gptoss_20b_codesign_cpt.train \
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
