#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export UV_CACHE_DIR="${MELLUM_UV_CACHE_DIR:-/tmp/mellum-uv-$(id -u)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/mellum-inductor-$(id -u)}"
source "$SCRIPT_DIR/../container_env.sh"
export VLLM_LOGGING_LEVEL=ERROR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Each shard retains several index/mmap handles; use the granted hard limit.
ulimit -Sn "$(ulimit -Hn)"

# Select AWS OFI only when explicitly requested. A visible EFA device alone
# does not prove NCCL is using EFA. The preflight reports the provider/plugin.
if [[ "${REQUIRE_EFA:-1}" == 1 ]]; then
  OFI_NCCL_LIB="${OFI_NCCL_LIB:-/opt/amazon/aws-ofi-nccl/lib}"
  [[ -f "$OFI_NCCL_LIB/libnccl-net-ofi.so" ]] || { echo "Missing AWS OFI NCCL plugin" >&2; exit 1; }
  export LD_LIBRARY_PATH="$OFI_NCCL_LIB:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
  export FI_PROVIDER=efa
  export NCCL_NET_PLUGIN=ofi
  export NCCL_NET="AWS Libfabric"
fi
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then unset NCCL_SOCKET_IFNAME; fi

action="${1:?Expected preflight/prepare/dry-run/train/status/stop/network-test}"
config="${2:?Expected absolute YAML path}"
shift 2
python_cmd=(/usr/local/bin/uv run --no-project --python "${MELLUM_PYTHON:-/opt/venv/bin/python}" python)
run_dir="$("${python_cmd[@]}" "$SCRIPT_DIR/train.py" --config "$config" --launch-info)"
rank="${1:-0}"
pid_file="$run_dir/node_${rank}.pid"
nnodes="${2:?Expected node count from launch.sh}"

require_memory_limit() {
  local limit cgroup_dir="${1:-/sys/fs/cgroup}"
  if [[ ! -r "$cgroup_dir/memory.max" ]]; then
    echo "A container-private cgroup-v2 memory limit is required" >&2; return 1
  fi
  limit="$(< "$cgroup_dir/memory.max")"
  if [[ ! "$limit" =~ ^[0-9]+$ || "$limit" == 0 ]]; then
    echo "Refusing GPU launch in an unlimited-memory container. Recreate it with MELLUM_MEMORY_GIB set to your allocation budget." >&2
    return 1
  fi
  echo "node $rank: container CPU memory limit: $limit bytes"
}

case "$action" in
  preflight)
    require_memory_limit
    mkdir -p -- "$run_dir"
    write_probe="$(mktemp "$run_dir/.write-probe-node_${rank}.XXXXXX")"
    unlink -- "$write_probe"
    echo "node $rank: run directory is writable: $run_dir"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    test -d /dev/infiniband
    /opt/amazon/efa/bin/fi_info -p efa > /tmp/mellum_efa_info.txt
    head -12 /tmp/mellum_efa_info.txt
    ldconfig -p | grep -E 'libnccl-net|libfabric' || true
    "${python_cmd[@]}" -c 'import torch, yaml; from megatron.bridge.models.mellum import MellumBridge; assert torch.cuda.device_count() == 8'
    ;;
  prepare|dry-run)
    "${python_cmd[@]}" "$SCRIPT_DIR/train.py" --config "$config" "--$action" --world-size "$((nnodes * 8))"
    ;;
  wandb-check)
    "${python_cmd[@]}" "$SCRIPT_DIR/train.py" --config "$config" --wandb-check
    ;;
  status|stop)
    if [[ ! -f "$pid_file" ]]; then echo "node $rank: NOT RUNNING"; exit 0; fi
    read -r pid owner < "$pid_file"
    if [[ -n "${MELLUM_LAUNCH_ID:-}" && "${MELLUM_LAUNCH_ID}" != "$owner" ]]; then exit 0; fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
      && ps -p "$pid" -o args= | grep -Fq -- "$config"; then
      if [[ "$action" == stop ]]; then kill -TERM -- "-$pid"; else echo "node $rank: RUNNING pid=$pid"; fi
    else echo "node $rank: stale PID file"; fi
    ;;
  train|network-test)
    require_memory_limit
    master="${3:?Expected master address}"
    port="${4:?Expected master port}"
    mkdir -p "$run_dir/logs"
    exec 9> "$run_dir/node_${rank}.lock"
    flock -n 9 || { echo "Run already locked on node $rank" >&2; exit 1; }
    if [[ -f "$pid_file" ]]; then
      read -r pid owner < "$pid_file"
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Run already active on node $rank: $pid_file" >&2; exit 1
      fi
    fi
    program=("$SCRIPT_DIR/train.py" --config "$config")
    if [[ "$action" == network-test ]]; then
      program=("$SCRIPT_DIR/network_test.py" --config "$config")
      export NCCL_DEBUG=INFO
    fi
    # setsid gives this launch an owned process group; stop never targets other jobs.
    log_file="$run_dir/logs/${action}_node_${rank}.log"
    if [[ -f "$log_file" ]]; then
      mv -- "$log_file" "$run_dir/logs/${action}_node_${rank}_$(date -u +%Y%m%dT%H%M%S)_$$.previous.log"
    fi
    setsid "${python_cmd[@]}" -m torch.distributed.run \
      --nnodes="$nnodes" --nproc_per_node=8 --node_rank="$rank" \
      --master_addr="$master" --master_port="$port" --max_restarts=0 \
      "${program[@]}" > "$log_file" 2>&1 &
    pid=$!
    monitor_pid=""
    printf '%s %s\n' "$pid" "${MELLUM_LAUNCH_ID:-manual}" > "$pid_file"
    cleanup() {
      if [[ -n "$monitor_pid" ]]; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
      fi
      if [[ -f "$pid_file" ]]; then
        read -r recorded_pid owner < "$pid_file"
        if [[ "$recorded_pid" == "$pid" ]]; then unlink "$pid_file"; fi
      fi
    }
    terminate() { kill -TERM -- "-$pid" 2>/dev/null || true; }
    trap terminate TERM INT HUP
    trap 'terminate; cleanup' EXIT
    if [[ "$action" == train ]]; then
      "${python_cmd[@]}" "$SCRIPT_DIR/memory_monitor.py" \
        --output "$run_dir/logs/cpu_memory_node_${rank}.jsonl" \
        --node-rank "$rank" --launch-id "${MELLUM_LAUNCH_ID:-manual}" \
        > "$run_dir/logs/memory_monitor_node_${rank}.log" 2>&1 &
      monitor_pid=$!
    fi
    wait "$pid"
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
