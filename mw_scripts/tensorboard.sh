 #!/usr/bin/env bash

  set -euo pipefail

  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

  # shellcheck source=mw_scripts/container_env.sh
  source "$SCRIPT_DIR/container_env.sh"

  WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname -- "$MBRIDGE_ROOT")}"
  RUN_DIR="${RUN_DIR:-$WORKSPACE_ROOT/training_runs/gptoss_16L_rtj_50B}"

  declare -Ar TENSORBOARD_RUNS=(
    ["gptoss_16L_rtj_50B"]="$RUN_DIR/tb_logs"
  )

  TB_HOST="${TB_HOST:-127.0.0.1}"
  TB_PORT="${TB_PORT:-6006}"
  TB_RELOAD_INTERVAL="${TB_RELOAD_INTERVAL:-30}"
  TB_STATE_DIR="${TB_STATE_DIR:-$RUN_DIR}"
  TB_LOG="${TB_LOG:-$TB_STATE_DIR/tensorboard-gptoss.log}"
  TB_PID_FILE="${TB_PID_FILE:-$TB_STATE_DIR/tensorboard-gptoss.pid}"

tensorboard_pid=""

build_logdir_spec() {
  local logdir_spec=""
  local run_name

  for run_name in "${!TENSORBOARD_RUNS[@]}"; do
    if [[ -n "$logdir_spec" ]]; then
      logdir_spec+=","
    fi
    logdir_spec+="$run_name:${TENSORBOARD_RUNS[$run_name]}"
  done

  printf '%s\n' "$logdir_spec"
}

tensorboard_is_running() {
  tensorboard_pid=""
  if [[ ! -f "$TB_PID_FILE" ]]; then
    return 1
  fi

  tensorboard_pid="$(<"$TB_PID_FILE")"
  [[ "$tensorboard_pid" =~ ^[0-9]+$ ]] \
    && kill -0 "$tensorboard_pid" 2>/dev/null \
    && ps -p "$tensorboard_pid" -o args= | grep -Fq "tensorboard"
}

print_runs() {
  local run_name

  echo "RUNS:"
  for run_name in "${!TENSORBOARD_RUNS[@]}"; do
    echo "  $run_name: ${TENSORBOARD_RUNS[$run_name]}"
  done
}

start_tensorboard() {
  local logdir_spec

  if tensorboard_is_running; then
    echo "ALREADY RUNNING: PID=$tensorboard_pid"
    echo "URL: http://$TB_HOST:$TB_PORT"
    echo "Use '$0 restart' after editing TENSORBOARD_RUNS."
    return 1
  fi

  mkdir -p "$TB_STATE_DIR"
  logdir_spec="$(build_logdir_spec)"

  nohup uv run --no-sync tensorboard \
    --logdir_spec "$logdir_spec" \
    --host "$TB_HOST" \
    --port "$TB_PORT" \
    --reload_interval "$TB_RELOAD_INTERVAL" \
    >"$TB_LOG" \
    2>&1 \
    </dev/null &

  tensorboard_pid=$!
  printf '%s\n' "$tensorboard_pid" >"$TB_PID_FILE"
  echo "STARTED: PID=$tensorboard_pid"
  echo "URL: http://$TB_HOST:$TB_PORT"
  echo "LOG: $TB_LOG"
  print_runs
}

stop_tensorboard() {
  local attempt

  if ! tensorboard_is_running; then
    rm -f "$TB_PID_FILE"
    echo "NOT RUNNING"
    return 0
  fi

  kill "$tensorboard_pid"
  for ((attempt = 0; attempt < 50; attempt++)); do
    if ! kill -0 "$tensorboard_pid" 2>/dev/null; then
      rm -f "$TB_PID_FILE"
      echo "STOPPED: PID=$tensorboard_pid"
      return 0
    fi
    sleep 0.2
  done

  echo "STOP TIMEOUT: PID=$tensorboard_pid is still running" >&2
  return 1
}

show_status() {
  if tensorboard_is_running; then
    echo "RUNNING: PID=$tensorboard_pid"
    echo "URL: http://$TB_HOST:$TB_PORT"
  else
    echo "NOT RUNNING"
  fi
  echo "LOG: $TB_LOG"
  print_runs
}

if (($# > 1)); then
  echo "Usage: $0 {start|stop|restart|status}" >&2
  exit 2
fi

action="${1:-start}"
case "$action" in
  start)
    start_tensorboard
    ;;
  stop)
    stop_tensorboard
    ;;
  restart)
    stop_tensorboard
    start_tensorboard
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
