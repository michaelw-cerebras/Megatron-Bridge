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
REPO_ROOT="$SCRIPT_DIR/.."

# shellcheck source=mw_scripts/container_env.sh
source "$SCRIPT_DIR/container_env.sh"

usage() {
  cat <<'EOF'
Run the official GPT-OSS 20B B200 FP8-MX v1 performance recipe on mock data.

Usage:
  mw_scripts/gptoss_20b_official_b200_mock_perf.sh              # background
  mw_scripts/gptoss_20b_official_b200_mock_perf.sh --foreground
  mw_scripts/gptoss_20b_official_b200_mock_perf.sh --status
  mw_scripts/gptoss_20b_official_b200_mock_perf.sh --dryrun

Defaults are the published B200 FP8-MX workload shape: 8 GPUs, mock data,
vocab 128256, sequence length 8192, MBS 2, GBS 16, and 100 steps. NCCL user
buffers are disabled by default; enable them only for a controlled bring-up.

Useful environment overrides:
  GPTOSS_HF_PATH, RUN_NAME, RUN_DIR, MAX_STEPS, LOG_INTERVAL, WARMUP_ITERS
  MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE, SEQ_LENGTH, CUDA_VISIBLE_DEVICES
  EXPERT_MODEL_PARALLEL_SIZE (default: 1), MOE_DISPATCHER (recipe, alltoall, deepep)
  MOE_DEEPEP_NUM_SMS (default: 20; used only with MOE_DISPATCHER=deepep)
  NUMA_BIND_LOCAL_RANK (default: 1 for this two-socket B200 host)
  ENABLE_NCCL_UB (default: 0), NCCL_UB_REGISTRATION (symmetric or local)
  NCCL_NVLS_ENABLE (default: 0), CUDA_GRAPH_IMPL (local or none; default: local)
  CUDA_GRAPH_SCOPE (comma-separated Transformer Engine graph scopes)
EOF
}

action="background"
case "${1:-}" in
  "") ;;
  --foreground) action="foreground" ;;
  --status) action="status" ;;
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
RUN_NAME="${RUN_NAME:-gptoss_20B_official_b200_mock_fp8mx}"
RUN_DIR="${RUN_DIR:-$WORKSPACE_ROOT/training_runs/$RUN_NAME}"
NUMA_BIND_LOCAL_RANK="${NUMA_BIND_LOCAL_RANK:-1}"
ENABLE_NCCL_UB="${ENABLE_NCCL_UB:-0}"
NCCL_UB_REGISTRATION="${NCCL_UB_REGISTRATION:-symmetric}"
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
CUDA_GRAPH_IMPL="${CUDA_GRAPH_IMPL:-local}"
CUDA_GRAPH_SCOPE="${CUDA_GRAPH_SCOPE:-}"
LOG_DIR="$RUN_DIR/logs"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/train.log}"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-$LOG_DIR/bootstrap.log}"
METADATA_DIR="$RUN_DIR/metadata"
CONFIG_OUTPUT="$METADATA_DIR/ConfigContainer.yaml"
PID_FILE="${PID_FILE:-$RUN_DIR/train.pid}"

NPROC_PER_NODE=8
MAX_STEPS="${MAX_STEPS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
MOE_DISPATCHER="${MOE_DISPATCHER:-recipe}"
MOE_DEEPEP_NUM_SMS="${MOE_DEEPEP_NUM_SMS:-20}"

process_is_this_run() {
  local process_id="$1"
  [[ "$process_id" =~ ^[0-9]+$ ]] \
    && kill -0 "$process_id" 2>/dev/null \
    && ps -p "$process_id" -o args= | grep -Fq "$SCRIPT_PATH"
}

if [[ "$action" == "status" ]]; then
  if [[ -f "$PID_FILE" ]] && process_is_this_run "$(<"$PID_FILE")"; then
    echo "RUNNING: PID=$(<"$PID_FILE")"
    echo "LOG: $TRAIN_LOG"
    exit 0
  fi
  echo "NOT RUNNING"
  echo "LOG: $TRAIN_LOG"
  exit 1
fi

if [[ "$action" == "background" ]]; then
  mkdir -p "$LOG_DIR"
  if [[ -f "$PID_FILE" ]] && process_is_this_run "$(<"$PID_FILE")"; then
    echo "ALREADY RUNNING: PID=$(<"$PID_FILE")" >&2
    echo "LOG: $TRAIN_LOG" >&2
    exit 1
  fi
  if [[ -e "$TRAIN_LOG" ]]; then
    echo "STOP: log already exists for this run: $TRAIN_LOG" >&2
    echo "Choose a new RUN_NAME rather than mixing benchmark results." >&2
    exit 1
  fi

  nohup "$SCRIPT_PATH" --foreground >"$BOOTSTRAP_LOG" 2>&1 </dev/null &
  background_pid=$!
  printf '%s\n' "$background_pid" >"$PID_FILE"
  echo "STARTED: PID=$background_pid"
  echo "RUN_DIR: $RUN_DIR"
  echo "LOG: $TRAIN_LOG"
  echo "FOLLOW: tail -F '$TRAIN_LOG'"
  exit 0
fi

for numeric_value in "$MAX_STEPS" "$LOG_INTERVAL" "$WARMUP_ITERS" \
  "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$SEQ_LENGTH" \
  "$EXPERT_MODEL_PARALLEL_SIZE" "$MOE_DEEPEP_NUM_SMS"; do
  if [[ ! "$numeric_value" =~ ^[0-9]+$ ]] || (( numeric_value < 1 )); then
    echo "STOP: all numeric settings must be positive integers, got: $numeric_value" >&2
    exit 2
  fi
done
if (( WARMUP_ITERS >= MAX_STEPS )); then
  echo "STOP: WARMUP_ITERS must be smaller than MAX_STEPS." >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * NPROC_PER_NODE) != 0 )); then
  echo "STOP: GLOBAL_BATCH_SIZE must divide by MICRO_BATCH_SIZE * $NPROC_PER_NODE." >&2
  exit 2
fi
if (( NPROC_PER_NODE % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
  echo "STOP: EXPERT_MODEL_PARALLEL_SIZE must divide $NPROC_PER_NODE." >&2
  exit 2
fi
if [[ "$MOE_DISPATCHER" != "recipe" && "$MOE_DISPATCHER" != "alltoall" \
  && "$MOE_DISPATCHER" != "deepep" ]]; then
  echo "STOP: MOE_DISPATCHER must be recipe, alltoall, or deepep." >&2
  exit 2
fi
if [[ ! -f "$GPTOSS_HF_PATH/config.json" ]]; then
  echo "STOP: GPT-OSS config not found: $GPTOSS_HF_PATH/config.json" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$METADATA_DIR"

cleanup_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
# Keep the direct launcher aligned with the performance executor defaults for
# GPT-OSS on B200. Full-iteration CUDA graphs require NCCL record streams,
# while this model's recipe intentionally retains cuDNN LayerNorm kernels.
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-0}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export NVTE_NORM_FWD_USE_CUDNN="${NVTE_NORM_FWD_USE_CUDNN:-1}"
export NVTE_NORM_BWD_USE_CUDNN="${NVTE_NORM_BWD_USE_CUDNN:-1}"
if [[ "$ENABLE_NCCL_UB" == 1 ]]; then
  # NCCL user-buffer pools must be statically allocated and registered.
  unset PYTORCH_CUDA_ALLOC_CONF
  unset NCCL_GRAPH_REGISTER
  export NCCL_CTA_POLICY=1
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,graph_capture_record_stream_reuse:True}"
  export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
fi
export NCCL_NVLS_ENABLE
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GPT_OSS_20B_HF_MODEL="$GPTOSS_HF_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "$NUMA_BIND_LOCAL_RANK" != 0 && "$NUMA_BIND_LOCAL_RANK" != 1 ]]; then
  echo "STOP: NUMA_BIND_LOCAL_RANK must be 0 or 1." >&2
  exit 2
fi
if [[ "$ENABLE_NCCL_UB" != 0 && "$ENABLE_NCCL_UB" != 1 ]]; then
  echo "STOP: ENABLE_NCCL_UB must be 0 or 1." >&2
  exit 2
fi
if [[ "$NCCL_UB_REGISTRATION" != "symmetric" && "$NCCL_UB_REGISTRATION" != "local" ]]; then
  echo "STOP: NCCL_UB_REGISTRATION must be symmetric or local." >&2
  exit 2
fi
if [[ "$NCCL_NVLS_ENABLE" != 0 && "$NCCL_NVLS_ENABLE" != 1 ]]; then
  echo "STOP: NCCL_NVLS_ENABLE must be 0 or 1." >&2
  exit 2
fi
if [[ "$CUDA_GRAPH_IMPL" != "none" && "$CUDA_GRAPH_IMPL" != "local" && "$CUDA_GRAPH_IMPL" != "transformer_engine" ]]; then
  echo "STOP: CUDA_GRAPH_IMPL must be none, local, or transformer_engine." >&2
  exit 2
fi
if [[ "$CUDA_GRAPH_IMPL" == "transformer_engine" && -z "$CUDA_GRAPH_SCOPE" ]]; then
  echo "STOP: CUDA_GRAPH_SCOPE is required when CUDA_GRAPH_IMPL=transformer_engine." >&2
  exit 2
fi
if [[ "$NUMA_BIND_LOCAL_RANK" == 1 && ! -x "$SCRIPT_DIR/numa_local_rank_exec.sh" ]]; then
  echo "STOP: NUMA binding wrapper is missing or not executable: $SCRIPT_DIR/numa_local_rank_exec.sh" >&2
  exit 2
fi

torchrun_python_mode=()
worker_command=()
if [[ "$NUMA_BIND_LOCAL_RANK" == 1 ]]; then
  torchrun_python_mode=(--no-python)
  worker_command=("$SCRIPT_DIR/numa_local_rank_exec.sh" python)
fi

run_args=(
  -m gpt_oss
  -mr gpt_oss_20b
  --task pretrain
  -g b200
  -c fp8_mx
  -ng "$NPROC_PER_NODE"
  -ms "$MAX_STEPS"
  -gb "$GLOBAL_BATCH_SIZE"
  -mb "$MICRO_BATCH_SIZE"
  -sl "$SEQ_LENGTH"
  --data mock
  --tokenizer_type NullTokenizer
  --vocab_size 128256
  --offline
  -cv v1
  --cuda_graph_impl "$CUDA_GRAPH_IMPL"
  --warmup_iters "$WARMUP_ITERS"
  --save_config_filepath "$CONFIG_OUTPUT"
  "ddp.nccl_ub=$([[ "$ENABLE_NCCL_UB" == 1 ]] && echo true || echo false)"
  "ddp.disable_symmetric_registration=$([[ "$NCCL_UB_REGISTRATION" == local ]] && echo true || echo false)"
  "ddp.average_in_collective=$([[ "$ENABLE_NCCL_UB" == 1 ]] && echo false || echo true)"
  checkpoint.load=null
  checkpoint.save=null
  logger.log_interval="$LOG_INTERVAL"
  logger.log_throughput=true
  logger.log_throughput_to_tensorboard=false
  validation.eval_interval=null
  validation.eval_iters=0
)
if [[ -n "$CUDA_GRAPH_SCOPE" ]]; then
  run_args+=(--cuda_graph_scope "$CUDA_GRAPH_SCOPE")
fi
if [[ "$EXPERT_MODEL_PARALLEL_SIZE" != 1 ]]; then
  run_args+=(--expert_model_parallel_size "$EXPERT_MODEL_PARALLEL_SIZE")
fi
case "$MOE_DISPATCHER" in
  recipe) ;;
  alltoall)
    run_args+=(
      model.moe_token_dispatcher_type=alltoall
      model.moe_flex_dispatcher_backend=null
      model.moe_expert_rank_capacity_factor=null
    )
    ;;
  deepep)
    run_args+=(
      --moe_flex_dispatcher_backend deepep
      "model.moe_deepep_num_sms=$MOE_DEEPEP_NUM_SMS"
      model.moe_expert_rank_capacity_factor=null
      model.moe_router_force_load_balancing=true
    )
    ;;
esac
if [[ "$action" == "dryrun" ]]; then
  uv run --no-sync python scripts/performance/run_script.py "${run_args[@]}" --dryrun
  exit 0
fi

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  busy_gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$busy_gpu_processes" ]]; then
    echo "STOP: GPU compute processes are already running:" >&2
    echo "$busy_gpu_processes" >&2
    echo "Stop them first, or set ALLOW_BUSY_GPUS=1 intentionally." >&2
    exit 3
  fi
fi

exec >"$TRAIN_LOG" 2>&1
echo "===== OFFICIAL B200 MOCK FP8-MX START $(date --iso-8601=seconds) ====="
echo "RECIPE: gpt_oss_20b/b200/fp8_mx/v1"
echo "MOCK_DATA: true"
echo "VOCAB_SIZE: 128256"
echo "NCCL_UB: $ENABLE_NCCL_UB"
echo "NCCL_UB_REGISTRATION: $NCCL_UB_REGISTRATION"
echo "NCCL_NVLS_ENABLE_AT_COMM_INIT: $NCCL_NVLS_ENABLE"
echo "EXPERT_MODEL_PARALLEL_SIZE: $EXPERT_MODEL_PARALLEL_SIZE"
echo "MOE_DISPATCHER: $MOE_DISPATCHER"
echo "MOE_DEEPEP_NUM_SMS: $MOE_DEEPEP_NUM_SMS"
echo "TORCH_NCCL_AVOID_RECORD_STREAMS: $TORCH_NCCL_AVOID_RECORD_STREAMS"
echo "TORCH_NCCL_HIGH_PRIORITY: $TORCH_NCCL_HIGH_PRIORITY"
echo "NVTE_NORM_FWD_USE_CUDNN: $NVTE_NORM_FWD_USE_CUDNN"
echo "NVTE_NORM_BWD_USE_CUDNN: $NVTE_NORM_BWD_USE_CUDNN"
echo "CUDA_GRAPH_IMPL: $CUDA_GRAPH_IMPL"
echo "CUDA_GRAPH_SCOPE: ${CUDA_GRAPH_SCOPE:-<none>}"
echo "NUMA_BIND_LOCAL_RANK: $NUMA_BIND_LOCAL_RANK"
echo "MAX_STEPS: $MAX_STEPS"
echo "CONFIG_OUTPUT: $CONFIG_OUTPUT"

set +e
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${torchrun_python_mode[@]}" \
  "${worker_command[@]}" \
  scripts/performance/run_script.py \
  "${run_args[@]}" \
  2>&1 \
  | sed -u '/^\[WARNING\] Field .* duplicates an ancestor field in /d'
pipeline_status=("${PIPESTATUS[@]}")
set -e

exit_code="${pipeline_status[0]}"
if (( exit_code == 0 && pipeline_status[1] != 0 )); then
  exit_code="${pipeline_status[1]}"
fi
echo "TRAINING_EXIT_CODE: $exit_code"
echo "FINISHED_AT: $(date --iso-8601=seconds)"
exit "$exit_code"
