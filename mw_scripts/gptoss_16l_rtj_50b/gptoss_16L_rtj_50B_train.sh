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
Start or resume the custom 16-layer GPT-OSS 50B-token run.

Usage:
  mw_scripts/gptoss_16l_rtj_50b/gptoss_16L_rtj_50B_train.sh              # background (default)
  mw_scripts/gptoss_16l_rtj_50b/gptoss_16L_rtj_50B_train.sh --foreground
  mw_scripts/gptoss_16l_rtj_50b/gptoss_16L_rtj_50B_train.sh --status

The default run uses 8 GPUs, sequence length 8192, GBS 128, EP 8, FP8 current
scaling, full activation recomputation, and the checked-in RTJ source mix.
Set environment variables below before launch to override defaults:

  GPTOSS_HF_PATH, GPTOSS_DATASET_ROOT, MIXTURE_CONFIG
  RUN_NAME, RUN_DIR, NPROC_PER_NODE, CUDA_VISIBLE_DEVICES
  TRAIN_TOKENS, SEQ_LENGTH, MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE
  TENSOR_PARALLEL, PIPELINE_PARALLEL, EXPERT_PARALLEL
  EXPERT_TENSOR_PARALLEL, PRECISION, NO_FULL_RECOMPUTE
  LR, MIN_LR, LR_WARMUP_ITERS, SAVE_INTERVAL, LOG_INTERVAL, NUM_WORKERS
EOF
}

action="background"
case "${1:-}" in
  "") ;;
  --foreground) action="foreground" ;;
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

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname -- "$MBRIDGE_ROOT")}"
GPTOSS_HF_PATH="${GPTOSS_HF_PATH:-$WORKSPACE_ROOT/checkpoints/gpt-oss-20b-hf}"
GPTOSS_DATASET_ROOT="${GPTOSS_DATASET_ROOT:-$WORKSPACE_ROOT/data/rtj_pt_50b_gptoss}"
MIXTURE_CONFIG="${MIXTURE_CONFIG:-$SCRIPT_DIR/rtj_pt_50b_stack25_stackedu10.json}"

RUN_NAME="${RUN_NAME:-gptoss_16L_rtj_50B}"
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
PID_FILE="${PID_FILE:-$RUN_DIR/train.pid}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_TOKENS="${TRAIN_TOKENS:-50000000000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-8}"
EXPERT_TENSOR_PARALLEL="${EXPERT_TENSOR_PARALLEL:-1}"
PRECISION="${PRECISION:-fp8-current}"
NO_FULL_RECOMPUTE="${NO_FULL_RECOMPUTE:-0}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"

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
  echo "FOLLOW: tail -f '$TRAIN_LOG'"
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
  "$NPROC_PER_NODE" \
  "$TRAIN_TOKENS" \
  "$SEQ_LENGTH" \
  "$MICRO_BATCH_SIZE" \
  "$GLOBAL_BATCH_SIZE" \
  "$TENSOR_PARALLEL" \
  "$PIPELINE_PARALLEL" \
  "$EXPERT_PARALLEL" \
  "$EXPERT_TENSOR_PARALLEL" \
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
  NPROC_PER_NODE < 1 || TRAIN_TOKENS < 1 || SEQ_LENGTH < 1
  || MICRO_BATCH_SIZE < 1 || GLOBAL_BATCH_SIZE < 1
  || TENSOR_PARALLEL < 1 || PIPELINE_PARALLEL < 1
  || EXPERT_PARALLEL < 1 || EXPERT_TENSOR_PARALLEL < 1
  || SAVE_INTERVAL < 1 || LOG_INTERVAL < 1
)); then
  echo "STOP: GPU, token, batch, parallelism, save, and log values must be positive." >&2
  exit 2
fi
if [[ "$NO_FULL_RECOMPUTE" != 0 && "$NO_FULL_RECOMPUTE" != 1 ]]; then
  echo "STOP: NO_FULL_RECOMPUTE must be 0 or 1." >&2
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

resume_run=0
if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" || -f "$CHECKPOINT_DIR/latest_train_state.pt" ]]; then
  resume_run=1
fi

if (( resume_run )); then
  exec >>"$TRAIN_LOG" 2>&1
  echo
  echo "===== RESUME $(date --iso-8601=seconds) ====="
else
  exec >"$TRAIN_LOG" 2>&1
  echo "===== FRESH RANDOM-INIT RUN $(date --iso-8601=seconds) ====="
fi

mixture_sha256="$(sha256sum "$MIXTURE_CONFIG" | awk '{print $1}')"
dataset_manifest_sha256="$(sha256sum "$GPTOSS_DATASET_ROOT/manifest.json" | awk '{print $1}')"
current_run_config="$(mktemp "$METADATA_DIR/.launch_config.XXXXXX")"
{
  printf 'GPTOSS_HF_PATH=%s\n' "$GPTOSS_HF_PATH"
  printf 'GPTOSS_DATASET_ROOT=%s\n' "$GPTOSS_DATASET_ROOT"
  printf 'MIXTURE_SHA256=%s\n' "$mixture_sha256"
  printf 'DATASET_MANIFEST_SHA256=%s\n' "$dataset_manifest_sha256"
  printf 'NPROC_PER_NODE=%s\n' "$NPROC_PER_NODE"
  printf 'TRAIN_TOKENS=%s\n' "$TRAIN_TOKENS"
  printf 'SEQ_LENGTH=%s\n' "$SEQ_LENGTH"
  printf 'MICRO_BATCH_SIZE=%s\n' "$MICRO_BATCH_SIZE"
  printf 'GLOBAL_BATCH_SIZE=%s\n' "$GLOBAL_BATCH_SIZE"
  printf 'TENSOR_PARALLEL=%s\n' "$TENSOR_PARALLEL"
  printf 'PIPELINE_PARALLEL=%s\n' "$PIPELINE_PARALLEL"
  printf 'EXPERT_PARALLEL=%s\n' "$EXPERT_PARALLEL"
  printf 'EXPERT_TENSOR_PARALLEL=%s\n' "$EXPERT_TENSOR_PARALLEL"
  printf 'PRECISION=%s\n' "$PRECISION"
  printf 'NO_FULL_RECOMPUTE=%s\n' "$NO_FULL_RECOMPUTE"
  printf 'LR=%s\n' "$LR"
  printf 'MIN_LR=%s\n' "$MIN_LR"
  printf 'LR_WARMUP_ITERS=%s\n' "$LR_WARMUP_ITERS"
  printf 'SAVE_INTERVAL=%s\n' "$SAVE_INTERVAL"
  printf 'LOG_INTERVAL=%s\n' "$LOG_INTERVAL"
  printf 'NUM_WORKERS=%s\n' "$NUM_WORKERS"
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
echo "MODE: $([[ "$resume_run" == 1 ]] && echo resume || echo fresh-random-init)"
echo "WARNING: this 16-layer variant starts from random weights; it does not load the public 24-layer checkpoint."

train_args=(
  --hf-model "$GPTOSS_HF_PATH"
  --dataset-root "$GPTOSS_DATASET_ROOT"
  --data-args-path "$DATA_ARGS_PATH"
  --train-tokens "$TRAIN_TOKENS"
  --seq-length "$SEQ_LENGTH"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --tensor-parallel "$TENSOR_PARALLEL"
  --pipeline-parallel "$PIPELINE_PARALLEL"
  --expert-parallel "$EXPERT_PARALLEL"
  --expert-tensor-parallel "$EXPERT_TENSOR_PARALLEL"
  --num-workers "$NUM_WORKERS"
  --dataset-cache-dir "$DATASET_CACHE_DIR"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --lr-warmup-iters "$LR_WARMUP_ITERS"
  --lr-decay-style cosine
  --precision "$PRECISION"
  --save-dir "$CHECKPOINT_DIR"
  --save-interval "$SAVE_INTERVAL"
  --tensorboard-dir "$TENSORBOARD_DIR"
  --log-interval "$LOG_INTERVAL"
)
if (( resume_run )); then
  train_args+=(--resume-checkpoint "$CHECKPOINT_DIR")
fi
if [[ "$NO_FULL_RECOMPUTE" == 1 ]]; then
  train_args+=(--no-full-recompute)
fi

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-CRITICAL}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::UserWarning}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# NVLS is disabled by default for the current B200 bring-up. Override only after validating the fabric.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

set +e
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m mw_scripts.gptoss_16l_rtj_50b.train \
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
