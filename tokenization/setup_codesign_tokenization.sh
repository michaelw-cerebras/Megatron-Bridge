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
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

export INPUT_ROOT=/cb/ml-cold/michaelw/codesign_pretrain/processed_hf
export OUTPUT_ROOT=/cb/ml-cold/michaelw/codesign_pretrain/tokenized
export DATATROVE_ROOT=/cb/ws/michaelw/datatrove
export TOKENIZATION_PYTHON=/opt/venv/bin/python
export TOKENIZER=/workspace/ws/checkpoints/gpt-oss-20b-hf/tokenizer.json
export EOS_TOKEN='<|return|>'
export STACK_V3_SEED=1234
export STACK_V3_MAX_CHARS=1048576
export STACK_V3_WORKERS=96
export STACK_V3_TASKS=192
export STACK_V3_READER_BATCH_SIZE=1
export STACK_V3_TOKENIZER_BATCH_SIZE=1
export WORKERS=96
export CPUSET=0-95
export RUN_LOG="$SCRIPT_DIR/tokenization.log"
export PID_FILE="$OUTPUT_ROOT/tokenization.pid"
export PYTHONPATH="$DATATROVE_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARTIFACTORY_INDEX='https://cerebras-repo.nova-cerebras.aws.internal/artifactory/api/pypi/olympus-pypi-dev/simple/'
ARTIFACTORY_HOST='cerebras-repo.nova-cerebras.aws.internal'

usage() {
  cat <<'EOF'
Usage:
  tokenization/setup_codesign_tokenization.sh --check
  tokenization/setup_codesign_tokenization.sh --dry-run
  tokenization/setup_codesign_tokenization.sh --background

The default action is --dry-run. The script uses the fixed CodeSign paths for
michaelw's NeMo container, installs humanize if missing, and runs a preflight.
Completed datasets are skipped. Stack v3 uses 96 workers with bounded 1 Mi-char
documents; all workers are restricted to physical CPU cores 0-95.
EOF
}

action="${1:---dry-run}"
case "$action" in
  --check|--dry-run|--background) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

for directory in "$INPUT_ROOT" "$DATATROVE_ROOT/src/datatrove"; do
  if [[ ! -d "$directory" ]]; then
    echo "STOP: required directory does not exist: $directory" >&2
    exit 1
  fi
done
for file in "$TOKENIZATION_PYTHON" "$TOKENIZER"; do
  if [[ ! -f "$file" ]]; then
    echo "STOP: required file does not exist: $file" >&2
    exit 1
  fi
done
if [[ ! -w /cb/ml-cold/michaelw/codesign_pretrain ]]; then
  echo "STOP: /cb/ml-cold/michaelw/codesign_pretrain is not writable" >&2
  exit 1
fi
if ! command -v taskset >/dev/null 2>&1; then
  echo "STOP: taskset is required to bind workers to the 96 physical cores." >&2
  exit 1
fi

if ! "$TOKENIZATION_PYTHON" -c 'import humanize' >/dev/null 2>&1; then
  echo "Installing missing DataTrove dependency: humanize"
  uv pip install \
    --no-config \
    --python "$TOKENIZATION_PYTHON" \
    --index-url "$ARTIFACTORY_INDEX" \
    --allow-insecure-host "$ARTIFACTORY_HOST" \
    humanize
fi

"$TOKENIZATION_PYTHON" - "$TOKENIZER" "$EOS_TOKEN" "$STACK_V3_MAX_CHARS" <<'PY'
import sys

import humanize
import orjson
import pyarrow
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.tokens import MegatronDocumentTokenizer
from mw_scripts.datatrove_text_reader import StackV3ParquetReader, TextJsonlReader, TextParquetReader
from mw_scripts.mapped_megatron_tokenizer import MappedMegatronDocumentTokenizer
from tokenization.stack_v3_repo_to_text_27_variants import repo_to_text_chunks, source_row_seed
from tokenizers import Tokenizer

tokenizer_path, eos_token, max_chars_raw = sys.argv[1:]
max_chars = int(max_chars_raw)
tokenizer = Tokenizer.from_file(tokenizer_path)
eos_id = tokenizer.token_to_id(eos_token)
if eos_id is None:
    raise RuntimeError(f"EOS token {eos_token!r} is missing from {tokenizer_path}")

example = {
    "repo_path": "preflight/example",
    "files": [{"file_path": "README.md", "content": "# Preflight\n"}],
}
seed = source_row_seed(1234, "data/example.parquet", 0)
first = list(repo_to_text_chunks(example, seed=seed, max_chars=max_chars))
second = list(repo_to_text_chunks(example, seed=seed, max_chars=max_chars))
if first != second or not first or any(len(chunk) > max_chars for chunk in first):
    raise RuntimeError("Stack v3 renderer is not deterministic")

print(
    f"Preflight OK: pyarrow={pyarrow.__version__} "
    f"humanize={humanize.__version__} EOS_ID={eos_id}"
)
PY

if [[ "$action" == "--check" ]]; then
  exit 0
fi

exec taskset --cpu-list "$CPUSET" "$SCRIPT_DIR/tokenize_codesign_pretrain.sh" "$action"
