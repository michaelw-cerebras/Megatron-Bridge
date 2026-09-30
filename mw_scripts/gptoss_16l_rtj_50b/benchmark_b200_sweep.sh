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
Run a controlled real-RTJ B200 throughput sweep for the custom GPT-OSS 16L model.

Usage:
  mw_scripts/gptoss_16l_rtj_50b/benchmark_b200_sweep.sh              # background
  mw_scripts/gptoss_16l_rtj_50b/benchmark_b200_sweep.sh --foreground
  mw_scripts/gptoss_16l_rtj_50b/benchmark_b200_sweep.sh --status

The sweep keeps architecture, data mixture, sequence length (8192), global
batch size (128), EP (8), and random initialization fixed. It varies only
B200 runtime choices: MXFP8, micro-batch size, DeepEP SM allocation, TE graphs,
and safe EP overlap (without delayed wgrad).

Optional environment variables:
  BENCHMARK_ITERS (default: 120), BENCHMARK_WARMUP_ITERS (default: 20)
  BENCHMARK_PRECISION (default: mxfp8)
  SWEEP_NAME (default: gptoss_16L_b200_realdata_sweep)
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

SWEEP_NAME="${SWEEP_NAME:-gptoss_16L_b200_realdata_sweep}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname -- "$MBRIDGE_ROOT")}"
SWEEP_DIR="${SWEEP_DIR:-$WORKSPACE_ROOT/training_runs/$SWEEP_NAME}"
SWEEP_LOG="$SWEEP_DIR/sweep.log"
RESULTS_TSV="$SWEEP_DIR/results.tsv"
PID_FILE="$SWEEP_DIR/sweep.pid"
BENCHMARK_ITERS="${BENCHMARK_ITERS:-120}"
BENCHMARK_WARMUP_ITERS="${BENCHMARK_WARMUP_ITERS:-20}"
BENCHMARK_PRECISION="${BENCHMARK_PRECISION:-mxfp8}"
TRAIN_SCRIPT="$SCRIPT_DIR/gptoss_16L_rtj_50B_train.sh"

if [[ ! "$BENCHMARK_ITERS" =~ ^[0-9]+$ ]] || (( BENCHMARK_ITERS < 20 )); then
  echo "STOP: BENCHMARK_ITERS must be an integer of at least 20." >&2
  exit 2
fi
if [[ ! "$BENCHMARK_WARMUP_ITERS" =~ ^[0-9]+$ ]] || (( BENCHMARK_WARMUP_ITERS < 1 || BENCHMARK_WARMUP_ITERS >= BENCHMARK_ITERS )); then
  echo "STOP: BENCHMARK_WARMUP_ITERS must be positive and less than BENCHMARK_ITERS." >&2
  exit 2
fi
if [[ "$BENCHMARK_PRECISION" != "fp8-current" && "$BENCHMARK_PRECISION" != "mxfp8" ]]; then
  echo "STOP: BENCHMARK_PRECISION must be fp8-current or mxfp8." >&2
  exit 2
fi

process_is_this_sweep() {
  local process_id="$1"
  [[ "$process_id" =~ ^[0-9]+$ ]] \
    && kill -0 "$process_id" 2>/dev/null \
    && ps -p "$process_id" -o args= | grep -Fq "$SCRIPT_PATH"
}

if [[ "$action" == "status" ]]; then
  if [[ -f "$PID_FILE" ]] && process_is_this_sweep "$(<"$PID_FILE")"; then
    echo "RUNNING: PID=$(<"$PID_FILE")"
  else
    echo "NOT RUNNING"
  fi
  echo "SWEEP_DIR: $SWEEP_DIR"
  echo "RESULTS: $RESULTS_TSV"
  exit 0
fi

if [[ "$action" == "background" ]]; then
  mkdir -p "$SWEEP_DIR"
  if [[ -f "$PID_FILE" ]] && process_is_this_sweep "$(<"$PID_FILE")"; then
    echo "ALREADY RUNNING: PID=$(<"$PID_FILE")" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
  nohup "$SCRIPT_PATH" --foreground >"$SWEEP_LOG" 2>&1 </dev/null &
  background_pid=$!
  printf '%s\n' "$background_pid" >"$PID_FILE"
  echo "STARTED: PID=$background_pid"
  echo "SWEEP_LOG: $SWEEP_LOG"
  echo "RESULTS: $RESULTS_TSV"
  exit 0
fi

mkdir -p "$SWEEP_DIR"
cleanup_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

tokens_per_iteration=$((8192 * 128))
train_tokens=$((BENCHMARK_ITERS * tokens_per_iteration))
printf 'case\tstatus\tmean_step_ms\tmean_tokens_per_second\tmeasurements\tlog\n' >"$RESULTS_TSV"

summarize_case() {
  local log_path="$1"
  uv run --no-sync python - "$log_path" "$BENCHMARK_WARMUP_ITERS" "$tokens_per_iteration" <<'PY'
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
warmup = int(sys.argv[2])
tokens_per_iteration = int(sys.argv[3])
pattern = re.compile(r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)")
samples = [float(match.group(2)) for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if (match := pattern.search(line)) and int(match.group(1)) > warmup]
if not samples:
    raise SystemExit("No post-warmup iteration timings found")
mean_ms = statistics.fmean(samples)
sys.stdout.write(f"{mean_ms:.2f}\t{tokens_per_iteration / (mean_ms / 1000):.1f}\t{len(samples)}")
PY
}

run_case() {
  local case_name="$1"
  local micro_batch_size="$2"
  local deepep_num_sms="$3"
  local disable_te_cuda_graphs="$4"
  local enable_ep_overlap="$5"
  local disable_deepep="$6"
  local disable_te_op_fuser="$7"
  local enable_moe_permute_fusion="$8"
  local run_name="${SWEEP_NAME}_${case_name}"
  local run_dir="$WORKSPACE_ROOT/training_runs/$run_name"
  local train_log="$run_dir/logs/train.log"

  echo "===== CASE $case_name $(date --iso-8601=seconds) ====="
  if RUN_NAME="$run_name" \
    TRAIN_TOKENS="$train_tokens" \
    LR_WARMUP_ITERS="$BENCHMARK_WARMUP_ITERS" \
    DISABLE_CHECKPOINTING=1 \
    LOG_INTERVAL=1 \
    MICRO_BATCH_SIZE="$micro_batch_size" \
    GLOBAL_BATCH_SIZE=128 \
    PRECISION="$BENCHMARK_PRECISION" \
    NO_FULL_RECOMPUTE=1 \
    B200_OPTIMIZED=1 \
    DEEP_EP_NUM_SMS="$deepep_num_sms" \
    DISABLE_TE_CUDA_GRAPHS="$disable_te_cuda_graphs" \
    DISABLE_DEEPEP="$disable_deepep" \
    DISABLE_TE_OP_FUSER="$disable_te_op_fuser" \
    ENABLE_EP_OVERLAP="$enable_ep_overlap" \
    ENABLE_MOE_PERMUTE_FUSION="$enable_moe_permute_fusion" \
    "$TRAIN_SCRIPT" --foreground; then
    if summary="$(summarize_case "$train_log")"; then
      printf '%s\tok\t%s\t%s\n' "$case_name" "$summary" "$train_log" | tee -a "$RESULTS_TSV"
    else
      printf '%s\tno_timing\t-\t-\t-\t%s\n' "$case_name" "$train_log" | tee -a "$RESULTS_TSV"
    fi
  else
    printf '%s\tfailed\t-\t-\t-\t%s\n' "$case_name" "$train_log" | tee -a "$RESULTS_TSV"
  fi
}

# Router padding makes MXFP8 valid for this top-k-16, dropless model. The
# cases preserve model architecture, routing objective, data, and GBS, while
# measuring dispatcher, graph, fusion, overlap, SM allocation, and MBS choices.
# At the default length this is sized for roughly six to seven hours including
# process initialization and the real-data input path.
run_case mbs1_alltoall_no_fuser_no_graphs 1 20 1 0 1 1 0
run_case mbs1_alltoall_fuser_no_graphs 1 20 1 0 1 0 0
run_case mbs1_deepep_no_fuser_no_graphs 1 20 1 0 0 1 0
run_case mbs1_deepep_fuser_no_graphs 1 20 1 0 0 0 0
run_case mbs1_deepep_fuser_graphs 1 20 0 0 0 0 0
run_case mbs2_deepep_fuser_no_graphs 2 20 1 0 0 0 0
run_case mbs2_deepep_fuser_graphs 2 20 0 0 0 0 0
run_case mbs4_deepep_fuser_graphs 4 20 0 0 0 0 0
run_case mbs2_deepep_sm16_graphs 2 16 0 0 0 0 0
run_case mbs2_deepep_sm24_graphs 2 24 0 0 0 0 0
run_case mbs2_deepep_sm32_graphs 2 32 0 0 0 0 0
run_case mbs1_deepep_fuser_graphs_ep_overlap 1 20 0 1 0 0 0
run_case mbs2_deepep_fuser_graphs_ep_overlap 2 20 0 1 0 0 0
run_case mbs4_deepep_fuser_graphs_ep_overlap 4 20 0 1 0 0 0
run_case mbs2_deepep_fuser_graphs_permute 2 20 0 0 0 0 1
run_case mbs2_deepep_fuser_graphs_permute_ep_overlap 2 20 0 1 0 0 1

echo "===== SWEEP COMPLETE $(date --iso-8601=seconds) ====="
column -t -s $'\t' "$RESULTS_TSV" || cat "$RESULTS_TSV"
