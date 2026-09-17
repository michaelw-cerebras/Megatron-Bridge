#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"

# shellcheck source=mw_scripts/container_env.sh
source "$SCRIPT_DIR/container_env.sh"

GPTOSS_HF_PATH="${GPTOSS_HF_PATH:-/workspace/ws/checkpoints/gpt-oss-20b-hf}"
GPTOSS_MCORE_PATH="${GPTOSS_MCORE_PATH:-/workspace/ws/checkpoints/gpt-oss-20b-megatron-ep8}"
GPTOSS_DATASET_ROOT="${GPTOSS_DATASET_ROOT:-/workspace/ws/data/ultrafineweb_l1_en_hq_gptoss}"

CPT_RUN_NAME="${CPT_RUN_NAME:-gpt-oss-20b-ultrafineweb-cpt-lr2e-6-pilot-2k}"
CPT_RUN_DIR="${CPT_RUN_DIR:-/scratch/$RUNTIME_USER/checkpoints/$CPT_RUN_NAME}"
CPT_SAVE_DIR="$CPT_RUN_DIR/checkpoints"
CPT_LOG_DIR="$CPT_RUN_DIR/logs"
CPT_TENSORBOARD_DIR="$CPT_RUN_DIR/tb_logs"
# CPT_LOG="$CPT_LOG_DIR/train.log"
CPT_LOG="$MBRIDGE_ROOT/mw_scripts/train.log"
CPT_PID_FILE="${CPT_PID_FILE:-$CPT_RUN_DIR.pid}"

if [[ "${CPT_BACKGROUND:-0}" != 1 ]]; then
  mkdir -p "$(dirname -- "$CPT_PID_FILE")"

  if [[ -f "$CPT_PID_FILE" ]]; then
    existing_pid="$(<"$CPT_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] \
      && kill -0 "$existing_pid" 2>/dev/null \
      && ps -p "$existing_pid" -o args= | grep -Fq "$SCRIPT_PATH"; then
      echo "ALREADY RUNNING: PID=$existing_pid"
      echo "LOG: $CPT_LOG"
      exit 1
    fi
  fi

  nohup env CPT_BACKGROUND=1 "$SCRIPT_PATH" \
    >/dev/null \
    2>&1 \
    </dev/null &

  background_pid=$!
  printf '%s\n' "$background_pid" >"$CPT_PID_FILE"
  echo "STARTED: PID=$background_pid"
  echo "LOG: $CPT_LOG"
  exit 0
fi

checkpoint_source_args=()
run_baseline=0
if [[ -f "$CPT_SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
  checkpoint_source_args=(--resume-checkpoint "$CPT_SAVE_DIR")
  run_mode="RESUME: $CPT_SAVE_DIR"
elif [[ -e "$CPT_RUN_DIR" ]]; then
  mkdir -p "$CPT_LOG_DIR"
  exec >>"$CPT_LOG" 2>&1
  echo "STOP: run directory exists without a resumable checkpoint: $CPT_RUN_DIR"
  exit 1
else
  checkpoint_source_args=(--pretrained-checkpoint "$GPTOSS_MCORE_PATH")
  run_baseline=1
  run_mode="FRESH RUN: $CPT_RUN_DIR"
fi

mkdir -p "$CPT_LOG_DIR" "$CPT_TENSORBOARD_DIR"
exec >>"$CPT_LOG" 2>&1
echo "$run_mode"

common_training_args=(
  --hf-model "$GPTOSS_HF_PATH" \
  --dataset-root "$GPTOSS_DATASET_ROOT" \
  --train-iters 2000 \
  --seq-length 512 \
  --micro-batch-size 1 \
  --global-batch-size 16 \
  --tensor-parallel 2 \
  --pipeline-parallel 1 \
  --expert-parallel 8 \
  --expert-tensor-parallel 1 \
  --num-workers 0 \
  --lr 2e-6 \
  --min-lr 2e-7 \
  --lr-warmup-iters 500 \
  --eval-interval 100 \
  --eval-iters 20 \
  --log-interval 10 \
  --tensorboard-dir "$CPT_TENSORBOARD_DIR"
)

run_distributed() {
  VLLM_LOGGING_LEVEL=CRITICAL \
  PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  uv run --no-sync python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    mw_scripts/continue_gpt_oss.py \
    "${common_training_args[@]}" \
    "$@" \
    2>&1 \
    | sed -u \
        '/^\[WARNING\] Field .* duplicates an ancestor field in /d'

  local pipeline_status=("${PIPESTATUS[@]}")
  if ((pipeline_status[0] != 0)); then
    return "${pipeline_status[0]}"
  fi
  return "${pipeline_status[1]}"
}

set +e
if ((run_baseline)); then
  echo "BASELINE EVAL: pretrained checkpoint at iteration 0"
  run_distributed \
    --pretrained-checkpoint "$GPTOSS_MCORE_PATH" \
    --eval-only
  baseline_exit_code=$?
  echo "baseline_exit_code=$baseline_exit_code"
  if ((baseline_exit_code != 0)); then
    exit "$baseline_exit_code"
  fi
fi

run_distributed \
  "${checkpoint_source_args[@]}" \
  --save-dir "$CPT_SAVE_DIR" \
  --save-interval 500
training_exit_code=$?
set -e

echo "training_exit_code=$training_exit_code"
if ((training_exit_code != 0)); then
  exit "$training_exit_code"
fi
exit 0
