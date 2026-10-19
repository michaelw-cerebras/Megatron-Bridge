#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run this on the SSH coordinator host; no Python/GPU packages are needed here.
# Never trace credential parsing or forwarding, even when invoked with bash -x.
set +x
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

load_wandb_env() {
  local env_file="${MELLUM_ENV_FILE-$SCRIPT_DIR/.env}"
  [[ -n "$env_file" ]] || return 0
  if [[ ! -e "$env_file" ]]; then
    if [[ -v MELLUM_ENV_FILE ]]; then
      echo "Configured MELLUM_ENV_FILE does not exist" >&2; return 1
    fi
    return 0
  fi
  [[ -f "$env_file" && -r "$env_file" ]] || {
    echo "W&B env file must be a readable regular file" >&2; return 1;
  }
  local line key value line_number=0
  local -A seen=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ ! "$line" =~ ^(export[[:space:]]+)?([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      echo "Invalid env assignment at line $line_number (contents hidden)" >&2; return 1
    fi
    key="${BASH_REMATCH[2]}"
    value="${BASH_REMATCH[3]}"
    case "$key" in WANDB_API_KEY|WANDB_BASE_URL|WANDB_HOST) ;; *) continue ;; esac
    if [[ -v seen[$key] ]]; then
      echo "Duplicate W&B env assignment at line $line_number (contents hidden)" >&2; return 1
    fi
    seen[$key]=1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "$value" == \"* || "$value" == \'* ]]; then
      if [[ ${#value} -lt 2 || "${value:0:1}" != "${value: -1}" ]]; then
        echo "Unmatched env quotes at line $line_number (contents hidden)" >&2; return 1
      fi
      value="${value:1:${#value}-2}"
    fi
    # No source/eval/interpolation: values are literal, including dollar signs.
    # A variable already set in the caller's shell wins, even when empty.
    if [[ ! -v $key ]]; then export "$key=$value"; fi
  done < "$env_file"
}

load_wandb_env
# The historical folder/container name does not fix the number of nodes.
# Current throughput configs target the one available 8xH200 node. Override
# MELLUM_NODES to scale a compatible config to more nodes.
read -r -a nodes <<< "${MELLUM_NODES:-172.31.210.225}"
# Container preparation validates mounts and never replaces existing containers automatically.
CONTAINER="${MELLUM_CONTAINER:-mellum2_3xh200}"
IMAGE="${MELLUM_IMAGE:-nvcr.io/nvidia/nemo:26.06}"
# On these 2 TiB nodes, a 1 TiB cap leaves substantial system headroom.
# Set this explicitly to match the memory budget granted for the allocation.
MEMORY_GIB="${MELLUM_MEMORY_GIB-1024}"
if [[ ! "$MEMORY_GIB" =~ ^[1-9][0-9]{0,4}$ ]]; then
  echo "MELLUM_MEMORY_GIB must be a positive integer (GiB)" >&2; exit 2
fi
REMOTE_REPO="${MELLUM_REMOTE_REPO:-/workspace/ws/Megatron-Bridge}"
HOST_WS="${MELLUM_HOST_WS:-$(dirname -- "$REPO_ROOT")}"
HOST_DATA="${MELLUM_HOST_DATA:-/cb/ml-olympus}"
HOST_RUNS="${MELLUM_HOST_RUNS:-$HOST_DATA/codesign_pretrain/runs}"
REMOTE_RUNS="${MELLUM_REMOTE_RUNS:-/workspace/ml-olympus/codesign_pretrain/runs}"
MASTER_ADDR="${MELLUM_MASTER_ADDR:-${nodes[0]}}"
MASTER_PORT="${MELLUM_MASTER_PORT:-29541}"
ACTION=""
VARIANT=""
while (( $# )); do
  case "$1" in
    --variant)
      [[ $# -ge 2 ]] || { echo "--variant requires BASE-vN or THINKING-vN" >&2; exit 2; }
      VARIANT="$2"; shift 2
      ;;
    --prepare-containers|--preflight|--prepare|--dry-run|--network-test|--wandb-check|--train|--status|--stop|--help|-h)
      [[ -z "$ACTION" ]] || { echo "Specify exactly one action" >&2; exit 2; }
      ACTION="$1"; shift
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
ACTION="${ACTION:---help}"
if [[ -n "$VARIANT" ]]; then
  if [[ -v MELLUM_CONFIG ]]; then
    echo "Use either --variant or MELLUM_CONFIG, not both" >&2; exit 2
  fi
  VARIANT="${VARIANT,,}"
  # Dotted names are the human-facing labels; filenames and registry ids use underscores.
  VARIANT="${VARIANT//./_}"
  case "$VARIANT" in
    base-v0|base-v1|base-v2|base-v3|base-v3_1|base-v1_1|base-v3_2|base-v3_3|thinking-v0|thinking-v1|thinking-v2|thinking-v3|thinking-v3_1|thinking-v1_1) ;;
    *) echo "Unknown variant case: $VARIANT" >&2; exit 2 ;;
  esac
  CONFIG="$REMOTE_REPO/mw_scripts/mellum2_3xh200/config/$VARIANT.yaml"
else
  CONFIG="${MELLUM_CONFIG:-$REMOTE_REPO/mw_scripts/mellum2_3xh200/config/thinking-v0.yaml}"
fi

if [[ "$HOST_RUNS" != /* || "$HOST_RUNS" == / || "$HOST_RUNS" == *,* || \
      "$REMOTE_RUNS" != /* || "$REMOTE_RUNS" == / || "$REMOTE_RUNS" == *,* ]]; then
  echo "Output mount paths must be absolute non-root paths without commas" >&2; exit 2
fi

remote() {
  local node="$1"
  shift
  local command
  printf -v command '%q ' "$@"
  ssh -F "$HOME/.ssh/config" -o BatchMode=yes -o ConnectTimeout=15 "$node" "$command"
}

node_action() {
  local node="$1" action="$2" rank="$3"
  local identity
  identity="$(remote "$node" id -u):$(remote "$node" id -g)"
  local credential_options=()
  local forward_credentials=0
  if [[ ( "$action" == train || "$action" == wandb-check ) && \
        ( -n "${WANDB_API_KEY:-}" || -n "${WANDB_BASE_URL:-}" || -n "${WANDB_HOST:-}" ) ]]; then
    credential_options=(-e WANDB_API_KEY -e WANDB_BASE_URL -e WANDB_HOST)
    forward_credentials=1
  fi
  local command=(docker exec --user "$identity" \
    -e "REQUIRE_EFA=${REQUIRE_EFA:-1}" -e "OFI_NCCL_LIB=${OFI_NCCL_LIB:-}" \
    -e "MELLUM_LAUNCH_ID=${MELLUM_LAUNCH_ID:-}" \
    -e "MELLUM_PYTHON=${MELLUM_PYTHON:-/opt/venv/bin/python}" \
    -e "HF_HOME=${MELLUM_HF_HOME:-$REMOTE_REPO/mw_scripts/mellum2_3xh200/.cache/huggingface}" \
    -e "USER=${USER:-mellum}" \
    -e "NETRC=${MELLUM_WANDB_NETRC:-$REMOTE_REPO/mw_scripts/mellum2_3xh200/.cache/wandb-auth.netrc}" \
    "${credential_options[@]}" \
    -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}" \
    -w "$REMOTE_REPO" "$CONTAINER" bash "$REMOTE_REPO/mw_scripts/mellum2_3xh200/node.sh" \
    "$action" "$CONFIG" "$rank" "${#nodes[@]}" "$MASTER_ADDR" "$MASTER_PORT")
  if (( forward_credentials )); then
    # Values travel over encrypted SSH stdin, never in SSH/docker argv.
    # The remote Docker client passes them by environment name.
    printf '%s\0' "${WANDB_API_KEY:-}" "${WANDB_BASE_URL:-}" "${WANDB_HOST:-}" | remote "$node" bash -c \
      'IFS= read -r -d "" WANDB_API_KEY; IFS= read -r -d "" WANDB_BASE_URL; IFS= read -r -d "" WANDB_HOST; export WANDB_API_KEY WANDB_BASE_URL WANDB_HOST; exec "$@"' \
      bash "${command[@]}"
  else
    remote "$node" "${command[@]}"
  fi
}

validate_container_mounts() {
  local node="$1" writable_mounts readonly_mounts
  writable_mounts="$(remote "$node" docker inspect -f \
    '{{range .Mounts}}{{if .RW}}{{println .Source .Destination}}{{end}}{{end}}' "$CONTAINER")"
  readonly_mounts="$(remote "$node" docker inspect -f \
    '{{range .Mounts}}{{if not .RW}}{{println .Source .Destination}}{{end}}{{end}}' "$CONTAINER")"
  if ! grep -Fqx -- "$HOST_RUNS $REMOTE_RUNS" <<< "$writable_mounts" \
    || ! grep -Fqx -- "$HOST_DATA /workspace/ml-olympus" <<< "$readonly_mounts"; then
    echo "$node: container $CONTAINER must mount data read-only and $REMOTE_RUNS read-write" >&2
    echo "Existing containers are never changed automatically; select a new MELLUM_CONTAINER and prepare it." >&2
    return 1
  fi
}

case "$ACTION" in
  --prepare-containers)
    for node in "${nodes[@]}"; do
      if remote "$node" docker inspect "$CONTAINER" >/dev/null 2>&1; then
        if [[ "$(remote "$node" docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]]; then
          echo "$node: $CONTAINER exists but is stopped; start it explicitly with docker start" >&2; exit 1
        fi
        echo "$node: using existing container $CONTAINER"
      else
        # Create output storage as the SSH user, not Docker root on a root-squashed NFS mount.
        remote "$node" mkdir -p -- "$HOST_RUNS"
        remote "$node" docker run -d --network=host --gpus all --name "$CONTAINER" \
          --label megatron-bridge.mellum-bringup=true \
          --memory "${MEMORY_GIB}g" --memory-swap "${MEMORY_GIB}g" --cgroupns=private \
          --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=262144:262144 \
          --device=/dev/infiniband -e HOME=/workspace -e PYTHONNOUSERSITE=1 \
          -e PIP_CACHE_DIR=/tmp/pip-cache -v "$HOST_WS:/workspace/ws" \
          -v "$HOST_DATA:/workspace/ml-olympus:ro" \
          --mount "type=bind,src=$HOST_RUNS,dst=$REMOTE_RUNS" \
          -w "$REMOTE_REPO" "$IMAGE" sleep infinity
      fi
      validate_container_mounts "$node"
    done
    ;;
  --preflight|--dry-run|--status|--stop|--wandb-check)
    for rank in "${!nodes[@]}"; do node_action "${nodes[$rank]}" "${ACTION#--}" "$rank"; done
    ;;
  --prepare)
    node_action "${nodes[0]}" prepare 0
    ;;
  --train|--network-test)
    export MELLUM_LAUNCH_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
    # Complete configuration validation on every node before occupying GPUs.
    if [[ "$ACTION" == --train ]]; then
      validation_jobs=()
      for rank in "${!nodes[@]}"; do
        node_action "${nodes[$rank]}" dry-run "$rank" &
        validation_jobs+=("$!")
      done
      validation_failed=0
      for job in "${validation_jobs[@]}"; do wait "$job" || validation_failed=1; done
      if (( validation_failed )); then exit 1; fi
    fi
    jobs=()
    cleanup() {
      for rank in "${!nodes[@]}"; do node_action "${nodes[$rank]}" stop "$rank" || true; done
      for job in "${jobs[@]}"; do kill "$job" 2>/dev/null || true; done
    }
    trap 'cleanup; exit 130' INT TERM HUP
    for rank in "${!nodes[@]}"; do
      node_action "${nodes[$rank]}" "${ACTION#--}" "$rank" &
      jobs+=("$!")
    done
    failed=0
    for (( remaining=${#jobs[@]}; remaining>0; remaining-- )); do
      if ! wait -n; then failed=1; cleanup; break; fi
    done
    exit "$failed"
    ;;
  --help|-h)
    echo "Usage: bash launch.sh --variant CASE ACTION"
    echo "Cases: base-v{0,1,1.1,2,3,3.1,3.2,3.3}; thinking-v{0,1,1.1,2,3,3.1}"
    echo "Dotted labels and underscore ids are both accepted (for example base-v3.3 or base-v3_3)."
    echo "Actions: --prepare-containers|--preflight|--prepare|--dry-run|--network-test|--wandb-check|--train|--status|--stop"
    echo "Overrides: MELLUM_NODES MELLUM_CONTAINER MELLUM_CONFIG MELLUM_HOST_WS MELLUM_HOST_DATA MELLUM_HOST_RUNS MELLUM_REMOTE_RUNS MELLUM_MASTER_PORT"
    echo "Container: $CONTAINER; writable output: $HOST_RUNS -> $REMOTE_RUNS (data stays read-only)"
    echo "New container memory cap: $MEMORY_GIB GiB; override with MELLUM_MEMORY_GIB to match your allocation."
    echo "Selected nodes (${#nodes[@]} nodes, $((${#nodes[@]} * 8)) GPUs): ${nodes[*]}"
    echo "Selected config: $CONFIG"
    echo "W&B: automatically read .env beside launch.sh; existing environment variables take precedence."
    echo "MELLUM_ENV_FILE overrides that path; set it empty to disable .env loading. Never put credentials in YAML."
    echo "Training stays in the foreground. Use tmux or nohup as described in user_manual.md."
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
