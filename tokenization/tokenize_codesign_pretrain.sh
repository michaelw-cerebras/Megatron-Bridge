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

# shellcheck source=mw_scripts/container_env.sh
source "$REPO_ROOT/mw_scripts/container_env.sh"

usage() {
  cat <<'EOF'
Tokenize the local CodeSign pretraining corpora with DataTrove.

Usage:
  tokenization/tokenize_codesign_pretrain.sh [--background | --dry-run]

The launcher preserves source subdirectories as independently weighted output
datasets, skips outputs with a valid completion marker, and resumes incomplete
DataTrove tasks automatically.

Environment overrides:
  INPUT_ROOT               Source root
  OUTPUT_ROOT              Tokenized output root
  TOKENIZER                Local tokenizer directory/tokenizer.json, or HF ID
  TOKENIZER_REVISION       Tokenizer revision (default: main)
  TOKENIZATION_PYTHON      Python executable from the DataTrove environment
  DATATROVE_ROOT           Optional local DataTrove checkout; adds its src/ to PYTHONPATH
  EOS_TOKEN                End-of-document token (default: <|return|>)
  WORKERS                  Concurrent DataTrove workers (default: available CPUs)
  READER_BATCH_SIZE        Parquet reader batch size (default: 1000)
  TOKENIZER_BATCH_SIZE     Tokenizer batch size (default: 1000)
  STACK_V3_SEED            Stable Stack v3 augmentation seed (default: 1234)
  STACK_V3_MAX_CHARS       Maximum rendered characters per Stack document (default: 1048576)
  STACK_V3_WORKERS         Stack-only workers (default: 96)
  STACK_V3_TASKS           Stable Stack output shard count (default: 192)
  STACK_V3_READER_BATCH_SIZE     Stack-only Parquet batch size (default: 1)
  STACK_V3_TOKENIZER_BATCH_SIZE  Stack-only tokenizer batch size (default: 1)
  ALLOW_INCOMPLETE_INPUT   Set to 1 to ignore source .tmp files (default: 0)
  RUN_LOG                  Background log path
  PID_FILE                 Background PID path
EOF
}

action="run"
case "${1:-}" in
  "") ;;
  --background) action="background" ;;
  --dry-run) action="dry-run" ;;
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

INPUT_ROOT="${INPUT_ROOT:-/cb/ml-cold/michaelw/codesign_pretrain/processed_hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cb/ml-cold/michaelw/codesign_pretrain/tokenized}"
TOKENIZER="${TOKENIZER:-openai/gpt-oss-20b}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-main}"
EOS_TOKEN="${EOS_TOKEN:-<|return|>}"
WORKERS="${WORKERS:-$(nproc)}"
READER_BATCH_SIZE="${READER_BATCH_SIZE:-1000}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-1000}"
STACK_V3_SEED="${STACK_V3_SEED:-1234}"
STACK_V3_MAX_CHARS="${STACK_V3_MAX_CHARS:-1048576}"
STACK_V3_WORKERS="${STACK_V3_WORKERS:-96}"
STACK_V3_TASKS="${STACK_V3_TASKS:-192}"
STACK_V3_READER_BATCH_SIZE="${STACK_V3_READER_BATCH_SIZE:-1}"
STACK_V3_TOKENIZER_BATCH_SIZE="${STACK_V3_TOKENIZER_BATCH_SIZE:-1}"
ALLOW_INCOMPLETE_INPUT="${ALLOW_INCOMPLETE_INPUT:-0}"
TOKENIZATION_PYTHON="${TOKENIZATION_PYTHON:-}"
DATATROVE_ROOT="${DATATROVE_ROOT:-}"
RUN_LOG="${RUN_LOG:-$SCRIPT_DIR/tokenization.log}"
PID_FILE="${PID_FILE:-$OUTPUT_ROOT/tokenization.pid}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for numeric_value in \
  "$WORKERS" \
  "$READER_BATCH_SIZE" \
  "$TOKENIZER_BATCH_SIZE" \
  "$STACK_V3_MAX_CHARS" \
  "$STACK_V3_WORKERS" \
  "$STACK_V3_TASKS" \
  "$STACK_V3_READER_BATCH_SIZE" \
  "$STACK_V3_TOKENIZER_BATCH_SIZE"; do
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "STOP: expected a positive integer, got: $numeric_value" >&2
    exit 2
  fi
done
if [[ ! "$STACK_V3_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "STOP: STACK_V3_SEED must be an integer, got: $STACK_V3_SEED" >&2
  exit 2
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "STOP: input root does not exist: $INPUT_ROOT" >&2
  exit 1
fi
if [[ -n "$DATATROVE_ROOT" ]]; then
  datatrove_source="$DATATROVE_ROOT/src"
  if [[ ! -d "$datatrove_source/datatrove" ]]; then
    echo "STOP: DataTrove package not found under: $datatrove_source" >&2
    exit 1
  fi
  export PYTHONPATH="$datatrove_source${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -n "$TOKENIZATION_PYTHON" ]]; then
  if [[ ! -x "$TOKENIZATION_PYTHON" ]]; then
    echo "STOP: TOKENIZATION_PYTHON is not executable: $TOKENIZATION_PYTHON" >&2
    exit 1
  fi
  python_command=("$TOKENIZATION_PYTHON")
elif command -v uv >/dev/null 2>&1; then
  python_command=(uv run --no-sync python)
else
  echo "STOP: set TOKENIZATION_PYTHON to an offline environment's Python, or make uv available." >&2
  exit 1
fi
if [[ "$ALLOW_INCOMPLETE_INPUT" != "0" && "$ALLOW_INCOMPLETE_INPUT" != "1" ]]; then
  echo "STOP: ALLOW_INCOMPLETE_INPUT must be 0 or 1." >&2
  exit 2
fi

temporary_file_count="$(find "$INPUT_ROOT" -type f -name '*.tmp' -print | wc -l)"
if (( temporary_file_count > 0 )); then
  if [[ "$action" == "dry-run" ]]; then
    echo "WARNING: found $temporary_file_count source .tmp files; a real run will stop by default." >&2
  elif [[ "$ALLOW_INCOMPLETE_INPUT" != "1" ]]; then
    echo "STOP: found $temporary_file_count source .tmp files; input downloads appear incomplete." >&2
    echo "Wait for downloads to finish, or set ALLOW_INCOMPLETE_INPUT=1 to process only completed files." >&2
    exit 1
  else
    echo "WARNING: ignoring $temporary_file_count source .tmp files by explicit request." >&2
  fi
fi

dataset_specs=(
  "Nemotron-CC-Code-v1|text|*.parquet|identity"
  "Nemotron-CC-Math-v1|text|*.parquet|identity"
  "Nemotron-CC-v2|text|*.parquet|identity"
  "Nemotron-CC-v2.1|text|*.parquet|identity"
  "Nemotron-Pretraining-Legal-v1|text|*.parquet|identity"
  "Nemotron-Pretraining-Specialized-v1|text|*.parquet|identity"
  "Nemotron-Pretraining-Specialized-v1.1|text|*.parquet|identity"
  "Nemotron-Pretraining-Specialized-v1.2|text|*.parquet|identity"
  "Ultra-FineWeb|content|*.parquet|identity"
  "Ultra-FineWeb-L3|content|*.parquet|identity"
  "finepdfs|text|*.parquet|identity"
  "peS2o|text|*.json.gz|identity"
  "pubmed|text|*.jsonl.gz|identity"
  "uspto|text|*.parquet|identity"
  "stack-v3-train|text|*.parquet|stack-v3"
)

discover_dataset_units() {
  local input_folder="$1"
  local path_glob="$2"
  "${python_command[@]}" - "$input_folder" "$path_glob" <<'PY'
import fnmatch
from pathlib import Path
import sys

input_folder = Path(sys.argv[1])
path_glob = sys.argv[2]
paths = sorted(
    path.relative_to(input_folder).as_posix()
    for path in input_folder.rglob("*")
    if path.is_file() and fnmatch.fnmatch(path.relative_to(input_folder).as_posix(), path_glob)
)
if not paths:
    raise SystemExit(1)

directories = [Path(path).parts[:-1] for path in paths]
common_prefix = []
for components in zip(*directories):
    if len(set(components)) != 1:
        break
    common_prefix.append(components[0])

branch_by_path = {}
for path, directory in zip(paths, directories):
    branch_by_path[path] = directory[len(common_prefix)] if len(directory) > len(common_prefix) else ""
branches = sorted(set(branch_by_path.values()))
if "" in branches or len(branches) <= 1:
    sys.stdout.write(".\t\n")
else:
    for branch in branches:
        source_prefix = "/".join((*common_prefix, branch))
        sys.stdout.write(f"{branch}\t{source_prefix}\n")
PY
}

dataset_output_is_complete() {
  local output_folder="$1"
  "${python_command[@]}" - "$output_folder" <<'PY'
import json
from pathlib import Path
import sys

output_folder = Path(sys.argv[1])
manifest_path = output_folder / "manifest.json"
result_path = output_folder / "result.json"
prefixes_path = output_folder / "data_prefixes.txt"
if not (manifest_path.is_file() and prefixes_path.is_file()):
    raise SystemExit(1)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("storage") == "filtered-index-with-shared-bin-symlink":
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise SystemExit(2)
    prefixes = [line for line in prefixes_path.read_text(encoding="utf-8").splitlines() if line]
    expected_prefixes = [shard.get("prefix") for shard in shards if isinstance(shard, dict)]
    valid = (
        len(expected_prefixes) == len(shards)
        and prefixes == expected_prefixes
        and int(manifest.get("documents", -1)) == sum(int(shard.get("documents", -1)) for shard in shards)
        and int(manifest.get("tokens", -1)) == sum(int(shard.get("tokens", -1)) for shard in shards)
        and all(
            (output_folder / f"{prefix}.idx").is_file()
            and (output_folder / f"{prefix}.bin").is_file()
            and (output_folder / f"{prefix}.bin").is_symlink()
            for prefix in prefixes
        )
    )
    raise SystemExit(0 if valid else 2)

if not result_path.is_file():
    raise SystemExit(1)
result = json.loads(result_path.read_text(encoding="utf-8"))
tasks = int(manifest["pipeline"]["tasks"])
data_dir = output_folder / manifest["output"]["data_dir"]
save_filename = manifest["output"]["save_filename"]
index_paths = sorted(data_dir.glob(f"{save_filename}_*_tokens.idx"))
binary_paths = sorted(data_dir.glob(f"{save_filename}_*_tokens.bin"))
mapping_enabled = bool(manifest["output"].get("document_mapping", {}).get("enabled", False))
mapping_paths = sorted(data_dir.glob(f"{save_filename}_*_tokens.mapping.parquet"))
prefixes = [line for line in prefixes_path.read_text(encoding="utf-8").splitlines() if line]
expected_prefixes = [str(path.with_suffix("").relative_to(output_folder)) for path in index_paths]
counted_paths = [*index_paths, *binary_paths, *(mapping_paths if mapping_enabled else [])]
actual_bytes = sum(path.stat().st_size for path in counted_paths)
expected_bytes = int(result["result"]["output_bytes"])
result_shards = int(result["result"]["shards"])
mapping_valid = (
    len(mapping_paths) == tasks
    and int(result["result"].get("mapping_rows", -1)) == int(result["result"]["documents"])
    and int(result["result"].get("mapping_bytes", -1)) == sum(path.stat().st_size for path in mapping_paths)
) if mapping_enabled else not mapping_paths

valid = (
    len(index_paths) == tasks
    and len(binary_paths) == tasks
    and result_shards == tasks
    and prefixes == expected_prefixes
    and actual_bytes == expected_bytes
    and mapping_valid
    and all(path.with_suffix(".bin").is_file() for path in index_paths)
)
raise SystemExit(0 if valid else 2)
PY
}

if [[ "$action" == "background" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(<"$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "STOP: tokenization is already running with PID $existing_pid" >&2
      exit 1
    fi
  fi
  nohup "$SCRIPT_DIR/tokenize_codesign_pretrain.sh" >"$RUN_LOG" 2>&1 </dev/null &
  background_pid=$!
  printf '%s\n' "$background_pid" >"$PID_FILE"
  echo "Tokenization started: PID=$background_pid log=$RUN_LOG"
  exit 0
fi

echo "INPUT_ROOT: $INPUT_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "TOKENIZER: $TOKENIZER@$TOKENIZER_REVISION"
echo "TOKENIZATION_PYTHON: ${TOKENIZATION_PYTHON:-uv run --no-sync python}"
if [[ -n "$DATATROVE_ROOT" ]]; then
  echo "DATATROVE_ROOT: $DATATROVE_ROOT"
fi
echo "WORKERS: $WORKERS"
echo "STACK_V3_SEED: $STACK_V3_SEED"
echo "STACK_V3: workers=$STACK_V3_WORKERS tasks=$STACK_V3_TASKS max_chars=$STACK_V3_MAX_CHARS batches=$STACK_V3_READER_BATCH_SIZE/$STACK_V3_TOKENIZER_BATCH_SIZE"

for spec in "${dataset_specs[@]}"; do
  IFS='|' read -r dataset_name text_column path_glob text_renderer <<<"$spec"
  input_folder="$INPUT_ROOT/$dataset_name"
  if [[ ! -d "$input_folder" ]]; then
    echo "STOP: expected dataset directory does not exist: $input_folder" >&2
    exit 1
  fi
  first_input="$(find "$input_folder" -type f -name "$path_glob" -print -quit)"
  if [[ -z "$first_input" ]]; then
    echo "STOP: no files matched $path_glob under $input_folder" >&2
    exit 1
  fi

  mapfile -t dataset_units < <(discover_dataset_units "$input_folder" "$path_glob")
  if (( ${#dataset_units[@]} == 0 )); then
    echo "STOP: failed to discover tokenization units under $input_folder" >&2
    exit 1
  fi
  for unit in "${dataset_units[@]}"; do
    IFS=$'\t' read -r output_name source_config <<<"$unit"
    output_folder="$OUTPUT_ROOT/$dataset_name"
    display_name="$dataset_name"
    config_arguments=()
    if [[ "$output_name" != "." ]]; then
      output_folder="$output_folder/$output_name"
      display_name="$display_name/$output_name"
      config_arguments=(--config "$source_config")
    fi

    dataset_workers="$WORKERS"
    dataset_reader_batch_size="$READER_BATCH_SIZE"
    dataset_tokenizer_batch_size="$TOKENIZER_BATCH_SIZE"
    extra_arguments=()
    if [[ "$text_renderer" == "stack-v3" ]]; then
      dataset_workers="$STACK_V3_WORKERS"
      dataset_reader_batch_size="$STACK_V3_READER_BATCH_SIZE"
      dataset_tokenizer_batch_size="$STACK_V3_TOKENIZER_BATCH_SIZE"
      extra_arguments=(
        --tasks "$STACK_V3_TASKS"
        --renderer-seed "$STACK_V3_SEED"
        --renderer-max-chars "$STACK_V3_MAX_CHARS"
      )
    fi

    if dataset_output_is_complete "$output_folder"; then
      echo "SKIP: $display_name (complete; hierarchy and output files validated)"
      continue
    fi
    if [[ -f "$output_folder/result.json" ]]; then
      echo "STOP: completion marker exists but output validation failed: $output_folder" >&2
      exit 1
    fi

    command=(
      "${python_command[@]}" "$REPO_ROOT/mw_scripts/process_datatrove.py"
      --input-folder "$input_folder"
      --path-glob "$path_glob"
      --text-column "$text_column"
      --text-renderer "$text_renderer"
      --output-folder "$output_folder"
      --tokenizer "$TOKENIZER"
      --tokenizer-revision "$TOKENIZER_REVISION"
      --eos-token "$EOS_TOKEN"
      --workers "$dataset_workers"
      --reader-batch-size "$dataset_reader_batch_size"
      --tokenizer-batch-size "$dataset_tokenizer_batch_size"
      "${config_arguments[@]}"
      "${extra_arguments[@]}"
    )
    if [[ -f "$output_folder/manifest.json" ]]; then
      if [[ ! -f "$output_folder/result.json" ]] \
        && grep -q 'filtered-index-with-shared-bin-symlink' "$output_folder/manifest.json"; then
        echo "STOP: zero-copy subset output failed validation and cannot be resumed: $output_folder" >&2
        exit 1
      fi
      command+=(--resume)
    elif [[ -d "$output_folder" ]]; then
      first_output="$(find "$output_folder" -mindepth 1 -print -quit)"
      if [[ -n "$first_output" ]]; then
        echo "STOP: non-empty output has no resumable manifest: $output_folder" >&2
        exit 1
      fi
    fi

    if [[ "$action" == "dry-run" ]]; then
      printf 'DRY RUN:'
      printf ' %q' "${command[@]}"
      printf '\n'
      continue
    fi

    echo "START: $display_name ($path_glob, text_column=$text_column, text_renderer=$text_renderer)"
    "${command[@]}"
    echo "DONE: $display_name"
  done
done

if [[ "$action" == "dry-run" ]]; then
  echo "DRY RUN COMPLETE: no output directories or tokenized files were created."
else
  echo "TOKENIZATION COMPLETE: all configured datasets finished."
fi
