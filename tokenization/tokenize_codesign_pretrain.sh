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

The launcher processes the reviewed CodeSign source-to-target units, skips
matching completed outputs, and resumes incomplete DataTrove tasks automatically.

Environment overrides:
  HEURISTICS_ROOT          Heuristics-filtered source root
  DEDUPED_ROOT             Deduplicated source root
  RAW_HF_ROOT              Materialized Hugging Face dataset root
  OUTPUT_ROOT              Tokenized output root
  TOKENIZER                Local tokenizer directory/tokenizer.json, or HF ID
  TOKENIZER_REVISION       Tokenizer revision (default: main)
  TOKENIZATION_PYTHON      Python executable from the DataTrove environment
                           (default: sibling envs/datatrove, then uv)
  DATATROVE_ROOT           Optional local DataTrove checkout; adds its src/ to PYTHONPATH
  EOS_TOKEN                End-of-document token (default: <|endoftext|>)
  WORKERS                  Concurrent DataTrove workers (default: min(96, available CPUs))
  FINEPDFS_WORKERS         finepdfs-only workers (default: min(20, WORKERS))
  READER_BATCH_SIZE        Parquet reader batch size (default: 1000)
  TOKENIZER_BATCH_SIZE     Tokenizer batch size (default: 1000)
  STACK_V3_SEED            Stable Stack v3 augmentation seed (default: 1234)
  STACK_V3_MAX_CHARS       Maximum rendered characters per Stack document (default: 1048576)
  STACK_V3_WORKERS         Stack-only workers (default: min(16, WORKERS))
  STACK_V3_TASKS           Stable Stack output shard count (default: 192)
  STACK_V3_READER_BATCH_SIZE     Stack-only Parquet batch size (default: 1)
  STACK_V3_TOKENIZER_BATCH_SIZE  Stack-only tokenizer batch size (default: 1)
  ALLOW_INCOMPLETE_INPUT   Set to 1 to ignore source .tmp files (default: 0)
  RUN_LOG                  Background log path
  PID_FILE                 Background PID path
  CODESIGN_START_INDEX     First dataset index to process (default: 0)
  CODESIGN_END_INDEX       Last dataset index to process (default: last)
  CODESIGN_REVERSE         Process selected datasets in reverse order (default: 0)
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

HEURISTICS_ROOT="${HEURISTICS_ROOT:-/cb/ml-olympus/codesign_pretrain/heuristics_filtered_datasets}"
DEDUPED_ROOT="${DEDUPED_ROOT:-/cb/ml-olympus/codesign_pretrain/deduped_datasets}"
RAW_HF_ROOT="${RAW_HF_ROOT:-/cb/ml-olympus/codesign_pretrain/raw_hf_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cb/ml-olympus/codesign_pretrain/tokenized/mellum2}"
TOKENIZER="${TOKENIZER:-JetBrains/Mellum2-12B-A2.5B-Base}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-main}"
EOS_TOKEN="${EOS_TOKEN:-<|endoftext|>}"
available_cpus="$(nproc)"
default_workers=$(( available_cpus < 96 ? available_cpus : 96 ))
WORKERS="${WORKERS:-$default_workers}"
default_finepdfs_workers=$(( WORKERS < 20 ? WORKERS : 20 ))
FINEPDFS_WORKERS="${FINEPDFS_WORKERS:-$default_finepdfs_workers}"
READER_BATCH_SIZE="${READER_BATCH_SIZE:-1000}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-1000}"
STACK_V3_SEED="${STACK_V3_SEED:-1234}"
STACK_V3_MAX_CHARS="${STACK_V3_MAX_CHARS:-1048576}"
default_stack_workers=$(( WORKERS < 16 ? WORKERS : 16 ))
STACK_V3_WORKERS="${STACK_V3_WORKERS:-$default_stack_workers}"
STACK_V3_TASKS="${STACK_V3_TASKS:-192}"
STACK_V3_READER_BATCH_SIZE="${STACK_V3_READER_BATCH_SIZE:-1}"
STACK_V3_TOKENIZER_BATCH_SIZE="${STACK_V3_TOKENIZER_BATCH_SIZE:-1}"
ALLOW_INCOMPLETE_INPUT="${ALLOW_INCOMPLETE_INPUT:-0}"
TOKENIZATION_PYTHON="${TOKENIZATION_PYTHON:-}"
if [[ -z "$TOKENIZATION_PYTHON" && -x "$REPO_ROOT/../envs/datatrove/bin/python" ]]; then
  TOKENIZATION_PYTHON="$REPO_ROOT/../envs/datatrove/bin/python"
fi
DATATROVE_ROOT="${DATATROVE_ROOT:-}"
RUN_LOG="${RUN_LOG:-$SCRIPT_DIR/tokenization.log}"
PID_FILE="${PID_FILE:-$OUTPUT_ROOT/tokenization.pid}"
CODESIGN_START_INDEX="${CODESIGN_START_INDEX:-0}"
CODESIGN_END_INDEX="${CODESIGN_END_INDEX:-}"
CODESIGN_REVERSE="${CODESIGN_REVERSE:-0}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for numeric_value in \
  "$WORKERS" \
  "$FINEPDFS_WORKERS" \
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
for input_root in "$HEURISTICS_ROOT" "$DEDUPED_ROOT" "$RAW_HF_ROOT"; do
  if [[ ! -d "$input_root" ]]; then
    echo "STOP: input root does not exist: $input_root" >&2
    exit 1
  fi
done
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

temporary_file_count="$(find "$HEURISTICS_ROOT" "$DEDUPED_ROOT" "$RAW_HF_ROOT" -type f -name '*.tmp' -print | wc -l)"
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
if [[ "$action" != "dry-run" ]] \
  && ! "${python_command[@]}" -c 'import datatrove, huggingface_hub, humanize, orjson, pyarrow, tokenizers' >/dev/null 2>&1; then
  echo "STOP: tokenization Python is missing a DataTrove dependency." >&2
  echo "Use a prepared container/environment and set TOKENIZATION_PYTHON if needed." >&2
  exit 1
fi

# One entry per reviewed source-to-target tokenization unit.
# Fields: source root | source-relative folder | output-relative folder | text column | file glob | renderer.
dataset_specs=(
  # "heuristics|Nemotron-CC-v2/Diverse-QA|Nemotron-CC-v2/Diverse-QA|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2/High-Quality|Nemotron-CC-v2/High-Quality|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2/High-Quality-Synthetic|Nemotron-CC-v2/High-Quality-Synthetic|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2/Medium-High-Quality|Nemotron-CC-v2/Medium-High-Quality|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2/Translated-Diverse-QA|Nemotron-CC-v2/Translated-Diverse-QA|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2.1/High-Quality|Nemotron-CC-v2.1/High-Quality|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2.1/High-Quality-Synthetic|Nemotron-CC-v2.1/High-Quality-Synthetic|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2.1/High-Quality-Translated-To-English|Nemotron-CC-v2.1/High-Quality-Translated-To-English|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2.1/High-Quality-Translated-To-English-Synthetic|Nemotron-CC-v2.1/High-Quality-Translated-To-English-Synthetic|text|*.parquet|identity"
  # "heuristics|Nemotron-CC-v2.1/Medium-High-Quality|Nemotron-CC-v2.1/Medium-High-Quality|text|*.parquet|identity"
  "heuristics|Nemotron-CC-v2.1/Medium-High-Quality-Synthetic|Nemotron-CC-v2.1/Medium-High-Quality-Synthetic|text|*.parquet|identity"
  "heuristics|Nemotron-CC-v2.1/Medium-High-Quality-Translated-To-English|Nemotron-CC-v2.1/Medium-High-Quality-Translated-To-English|text|*.parquet|identity"
  "heuristics|Nemotron-CC-Code-v1/data|Nemotron-CC-Code-v1|text|*.parquet|identity"
  "heuristics|Nemotron-CC-Math-v1/3|Nemotron-CC-Math-v1/3|text|*.parquet|identity"
  "heuristics|Nemotron-CC-Math-v1/4plus|Nemotron-CC-Math-v1/4plus|text|*.parquet|identity"
  "heuristics|Nemotron-CC-Math-v1/4plus_MIND|Nemotron-CC-Math-v1/4plus_MIND|text|*.parquet|identity"
  "heuristics|Ultra-FineWeb-L3/data/ultrafineweb_en_l3|Ultra-FineWeb-L3/ultrafineweb_en_l3|content|*.parquet|identity"
  "heuristics|Ultra-FineWeb-L3/data/ultrafineweb_zh_l3|Ultra-FineWeb-L3/ultrafineweb_zh_l3|content|*.parquet|identity"
  "heuristics|finepdfs|finepdfs|text|*.parquet|identity"
  "heuristics|stack-v3-train/data|stack-v3-train|text|*.parquet|stack-v3"
  "raw_hf|Puro-2B/phase2|Puro-2B/phase2|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Legal-v1|Nemotron-Pretraining-Legal-v1|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks|Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-STEM-SFT|Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-STEM-SFT|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding|Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Wiki-Rewrite|Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Wiki-Rewrite|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts|Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Economics|Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Economics|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Formal-Logic|Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Formal-Logic|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Multiple-Choice|Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Multiple-Choice|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Fact-Seeking|Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Fact-Seeking|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Generative|Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Generative|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Moral-Scenarios|Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Moral-Scenarios|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Multiple-Choice|Nemotron-Pretraining-Specialized-v1.2/Nemotron-Pretraining-Multiple-Choice|text|*.parquet|identity"
  # "deduped|puffer_exact_specialized_v1/peS2o|peS2o|text|*.json.gz|identity"
  # "deduped|puffer_exact_specialized_v1/pubmed/data|pubmed|text|*.jsonl.gz|identity"
  # "deduped|puffer_exact_specialized_v1/uspto|uspto|text|*.parquet|identity"
  # "deduped|puffer_minhash_ufw_en_v1_4_p128_b8_lsh_only_v1/Ultra-FineWeb/data/ultrafineweb_en_v1_4|Ultra-FineWeb/ultrafineweb_en_v1_4|content|*.parquet|identity"
  # "deduped|puffer_minhash_ufw_en_v1_4_p128_b8_lsh_only_v1/Ultra-FineWeb/data/ultrafineweb_l1_en_hq|Ultra-FineWeb/ultrafineweb_l1_en_hq|content|*.parquet|identity"
  # "deduped|puffer_minhash_ufw_en_v1_4_p128_b8_lsh_only_v1/Ultra-FineWeb/data/ultrafineweb_zh|Ultra-FineWeb/ultrafineweb_zh|content|*.parquet|identity"
)
if [[ -z "$CODESIGN_END_INDEX" ]]; then
  CODESIGN_END_INDEX=$((${#dataset_specs[@]} - 1))
fi
if [[ ! "$CODESIGN_START_INDEX" =~ ^[0-9]+$ || ! "$CODESIGN_END_INDEX" =~ ^[0-9]+$ ]]; then
  echo "STOP: CODESIGN_START_INDEX and CODESIGN_END_INDEX must be nonnegative integers." >&2
  exit 2
fi
CODESIGN_START_INDEX=$((10#$CODESIGN_START_INDEX))
CODESIGN_END_INDEX=$((10#$CODESIGN_END_INDEX))
if (( CODESIGN_START_INDEX > CODESIGN_END_INDEX || CODESIGN_END_INDEX >= ${#dataset_specs[@]} )); then
  echo "STOP: dataset indexes must satisfy 0 <= start <= end < ${#dataset_specs[@]}." >&2
  exit 2
fi
if [[ "$CODESIGN_REVERSE" != "0" && "$CODESIGN_REVERSE" != "1" ]]; then
  echo "STOP: CODESIGN_REVERSE must be 0 or 1." >&2
  exit 2
fi
if [[ "$action" != "dry-run" ]] && \
  (( CODESIGN_START_INDEX != 0 || CODESIGN_END_INDEX != ${#dataset_specs[@]}-1 || CODESIGN_REVERSE == 1 )); then
  if ! command -v findmnt >/dev/null 2>&1; then
    echo "STOP: findmnt is needed to verify cross-machine output locking." >&2
    exit 1
  fi
  output_mount_options="$(findmnt -T "$OUTPUT_ROOT" -no OPTIONS)"
  if [[ ",$output_mount_options," != *",flock,"* ]]; then
    echo "STOP: output filesystem must be mounted with cluster-wide flock: $OUTPUT_ROOT" >&2
    exit 1
  fi
fi
selected_count=$((CODESIGN_END_INDEX - CODESIGN_START_INDEX + 1))
dataset_specs=("${dataset_specs[@]:CODESIGN_START_INDEX:selected_count}")
if [[ "$CODESIGN_REVERSE" == "1" ]]; then
  reversed_specs=()
  for ((spec_index=${#dataset_specs[@]}-1; spec_index>=0; spec_index--)); do
    reversed_specs+=("${dataset_specs[spec_index]}")
  done
  dataset_specs=("${reversed_specs[@]}")
fi

dataset_output_is_complete() {
  local output_folder="$1"
  local input_folder="$2"
  local path_glob="$3"
  local text_column="$4"
  local text_renderer="$5"
  "${python_command[@]}" - "$output_folder" "$input_folder" "$path_glob" "$text_column" "$text_renderer" "$TOKENIZER" "$EOS_TOKEN" <<'PY'
import json
from pathlib import Path
import sys

output_folder = Path(sys.argv[1])
input_folder = Path(sys.argv[2]).resolve()
path_glob, text_column, text_renderer, tokenizer_id, eos_token = sys.argv[3:]
manifest_path = output_folder / "manifest.json"
result_path = output_folder / "result.json"
prefixes_path = output_folder / "data_prefixes.txt"
if not (manifest_path.is_file() and prefixes_path.is_file()):
    raise SystemExit(1)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
dataset = manifest.get("dataset", {})
pipeline = manifest.get("pipeline", {})
tokenizer = manifest.get("tokenizer", {})
renderer = manifest.get("renderer", {})
if (
    dataset.get("root") != str(input_folder)
    or dataset.get("path_glob") != path_glob
    or pipeline.get("text_column") != text_column
    or tokenizer.get("id") != tokenizer_id
    or tokenizer.get("eos_token") != eos_token
    or renderer.get("name") != text_renderer
):
    raise SystemExit(2)
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
  nohup bash "$SCRIPT_DIR/tokenize_codesign_pretrain.sh" >"$RUN_LOG" 2>&1 </dev/null &
  background_pid=$!
  printf '%s\n' "$background_pid" >"$PID_FILE"
  echo "Tokenization started: PID=$background_pid log=$RUN_LOG"
  exit 0
fi

echo "HEURISTICS_ROOT: $HEURISTICS_ROOT"
echo "DEDUPED_ROOT: $DEDUPED_ROOT"
echo "RAW_HF_ROOT: $RAW_HF_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "TOKENIZER: $TOKENIZER@$TOKENIZER_REVISION"
echo "EOS_TOKEN: $EOS_TOKEN"
echo "TOKENIZATION_PYTHON: ${TOKENIZATION_PYTHON:-uv run --no-sync python}"
if [[ -n "$DATATROVE_ROOT" ]]; then
  echo "DATATROVE_ROOT: $DATATROVE_ROOT"
fi
echo "WORKERS: $WORKERS"
echo "FINEPDFS_WORKERS: $FINEPDFS_WORKERS"
echo "STACK_V3_SEED: $STACK_V3_SEED"
echo "STACK_V3: workers=$STACK_V3_WORKERS tasks=$STACK_V3_TASKS max_chars=$STACK_V3_MAX_CHARS batches=$STACK_V3_READER_BATCH_SIZE/$STACK_V3_TOKENIZER_BATCH_SIZE"

for spec in "${dataset_specs[@]}"; do
  IFS='|' read -r source_root_name source_relative output_relative text_column path_glob text_renderer <<<"$spec"
  case "$source_root_name" in
    heuristics) source_root="$HEURISTICS_ROOT" ;;
    deduped) source_root="$DEDUPED_ROOT" ;;
    raw_hf) source_root="$RAW_HF_ROOT" ;;
    *) echo "STOP: unknown source root: $source_root_name" >&2; exit 2 ;;
  esac
  input_folder="$source_root/$source_relative"
  output_folder="$OUTPUT_ROOT/$output_relative"
  display_name="$output_relative"
  if [[ ! -d "$input_folder" ]]; then
    echo "STOP: expected source directory does not exist: $input_folder" >&2
    exit 1
  fi
  first_input="$(find "$input_folder" -type f -name "$path_glob" -print -quit)"
  if [[ -z "$first_input" ]]; then
    echo "STOP: no files matched $path_glob under $input_folder" >&2
    exit 1
  fi

  dataset_workers="$WORKERS"
  dataset_reader_batch_size="$READER_BATCH_SIZE"
  dataset_tokenizer_batch_size="$TOKENIZER_BATCH_SIZE"
  extra_arguments=()
  if [[ "$output_relative" == "finepdfs" ]]; then
    dataset_workers="$FINEPDFS_WORKERS"
  elif [[ "$text_renderer" == "stack-v3" ]]; then
    dataset_workers="$STACK_V3_WORKERS"
    dataset_reader_batch_size="$STACK_V3_READER_BATCH_SIZE"
    dataset_tokenizer_batch_size="$STACK_V3_TOKENIZER_BATCH_SIZE"
    extra_arguments=(
      --tasks "$STACK_V3_TASKS"
      --renderer-seed "$STACK_V3_SEED"
      --renderer-max-chars "$STACK_V3_MAX_CHARS"
    )
  fi

  if [[ -f "$output_folder/manifest.json" ]] \
    && dataset_output_is_complete "$output_folder" "$input_folder" "$path_glob" "$text_column" "$text_renderer"; then
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

if [[ "$action" == "dry-run" ]]; then
  echo "DRY RUN COMPLETE: no output directories or tokenized files were created."
else
  echo "TOKENIZATION COMPLETE: all configured datasets finished."
fi
