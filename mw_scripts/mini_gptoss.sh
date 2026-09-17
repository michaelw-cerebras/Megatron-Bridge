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

# shellcheck source=mw_scripts/container_env.sh
source "$SCRIPT_DIR/container_env.sh"

GPTOSS_HF_PATH="${GPTOSS_HF_PATH:-/workspace/ws/checkpoints/gpt-oss-20b-hf}"
GPTOSS_DATASET_ROOT="${GPTOSS_DATASET_ROOT:-/workspace/ws/data/ultrafineweb_l1_en_hq_gptoss}"

# MINI_RUN_NAME="${MINI_RUN_NAME:-gptoss-custom-random-init-bf16-2k}"
MINI_RUN_NAME="${MINI_RUN_NAME:-gptoss-custom-random-init-fp8-current-2k}"
MINI_RUN_DIR="${MINI_RUN_DIR:-/scratch/$RUNTIME_USER/$MINI_RUN_NAME}"
MINI_LOG_DIR="$MINI_RUN_DIR/logs"
MINI_TENSORBOARD_DIR="$MINI_RUN_DIR/tb_logs"
# MINI_LOG="$MINI_LOG_DIR/train.log"
MINI_LOG="$MBRIDGE_ROOT/mw_scripts/mini_train_fp8.log"
MINI_PID_FILE="${MINI_PID_FILE:-$MINI_RUN_DIR.pid}"

if [[ "${MINI_BACKGROUND:-0}" != 1 ]]; then
  mkdir -p "$(dirname -- "$MINI_PID_FILE")"

  if [[ -f "$MINI_PID_FILE" ]]; then
    existing_pid="$(<"$MINI_PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] \
      && kill -0 "$existing_pid" 2>/dev/null \
      && ps -p "$existing_pid" -o args= | grep -Fq "$SCRIPT_PATH"; then
      echo "ALREADY RUNNING: PID=$existing_pid"
      echo "LOG: $MINI_LOG"
      exit 1
    fi
  fi

  if [[ -e "$MINI_RUN_DIR" ]]; then
    echo "STOP: run directory already exists: $MINI_RUN_DIR" >&2
    echo "Set MINI_RUN_NAME or MINI_RUN_DIR to start a distinct random-init run." >&2
    exit 1
  fi

  nohup env MINI_BACKGROUND=1 "$SCRIPT_PATH" \
    >/dev/null \
    2>&1 \
    </dev/null &

  background_pid=$!
  printf '%s\n' "$background_pid" >"$MINI_PID_FILE"
  echo "STARTED: PID=$background_pid"
  echo "LOG: $MINI_LOG"
  echo "TENSORBOARD: $MINI_TENSORBOARD_DIR"
  exit 0
fi

if [[ -e "$MINI_RUN_DIR" ]]; then
  exit 1
fi

mkdir -p "$MINI_LOG_DIR" "$MINI_TENSORBOARD_DIR"
# A fresh run owns this log, so discard output left by earlier runs.
exec >"$MINI_LOG" 2>&1
echo "FRESH RANDOM-INIT RUN: $MINI_RUN_DIR"

set +e
VLLM_LOGGING_LEVEL=CRITICAL \
PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# B200 only setting
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
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
  --recompute \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --lr-warmup-iters 200 \
  --lr-decay-iters 2000 \
  --lr-decay-style cosine \
  --log-interval 10 \
  --tensorboard-dir "$MINI_TENSORBOARD_DIR" \
  --num-workers 2 \
  --precision fp8-current \
  2>&1 \
  | sed -u \
      '/^\[WARNING\] Field .* duplicates an ancestor field in /d'

# Can set --precision bf16 or --precision fp8-current \

pipeline_status=("${PIPESTATUS[@]}")
set -e

training_exit_code="${pipeline_status[0]}"
if ((training_exit_code == 0 && pipeline_status[1] != 0)); then
  training_exit_code="${pipeline_status[1]}"
fi

echo "training_exit_code=$training_exit_code"
exit "$training_exit_code"
