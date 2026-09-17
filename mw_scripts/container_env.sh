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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file from a launcher instead of executing it directly." >&2
  exit 2
fi

_MBRIDGE_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export MBRIDGE_ROOT="${MBRIDGE_ROOT:-$(cd -- "$_MBRIDGE_ENV_DIR/.." && pwd -P)}"
cd "$MBRIDGE_ROOT"

export RUNTIME_USER="${USER:-$(id -un)}"

_mbridge_prepend_pythonpath() {
  local entry="$1"

  case ":${PYTHONPATH:-}:" in
    *":$entry:"*) ;;
    *) PYTHONPATH="$entry${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
}

# Iterate in reverse because each entry is prepended.
_mbridge_prepend_pythonpath "$MBRIDGE_ROOT/3rdparty/Megatron-LM"
_mbridge_prepend_pythonpath "$MBRIDGE_ROOT/src"
export PYTHONPATH
unset -f _mbridge_prepend_pythonpath

export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/$RUNTIME_USER-cache}"
export HF_HOME="${HF_HOME:-/tmp/$RUNTIME_USER-hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/$RUNTIME_USER-pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$RUNTIME_USER-uv-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/$RUNTIME_USER-flashinfer}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$RUNTIME_USER-matplotlib}"
export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/tmp/$RUNTIME_USER-megatron-locks}"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" \
  "$MPLCONFIGDIR" \
  "$MEGATRON_CONFIG_LOCK_DIR"

unset _MBRIDGE_ENV_DIR
