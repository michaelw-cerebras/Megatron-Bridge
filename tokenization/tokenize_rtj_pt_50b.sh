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
Tokenize the local RTJ PT 50B JSONL corpus into Megatron mmap datasets.

Usage:
  tokenization/tokenize_rtj_pt_50b.sh [--background | --dry-run | --finalize-only]

The default invocation is resumable and processes every part-*.jsonl shard.
The output preserves the source/shard hierarchy and writes both global and
per-source data-prefix lists for configuring later data mixtures.

Environment overrides:
  RAW_ROOT                  Input root (default: sibling raw_PT_50B_tokens)
  TOKENIZER_MODEL           Local GPT-OSS tokenizer/model directory
  OUTPUT_ROOT               Tokenized output root
  RUN_LOG                   Background master log (default: OUTPUT_ROOT/logs/tokenization.log)
  PID_FILE                  Background PID file (default: OUTPUT_ROOT/tokenization.pid)
  PARALLEL_JOBS             Concurrent input shards (default: half available CPUs)
  TOKENIZER_WORKERS_PER_JOB Tokenizer workers per shard (default: 1)
  EXPECTED_SHARDS           Expected JSONL shard count (default: 771; 0 disables)
  EXPECTED_SOURCES          Expected source count (default: 9; 0 disables)
  LOG_INTERVAL              Documents between progress messages (default: 10000)
  TOKENIZATION_CUDA_VISIBLE_DEVICES
                             GPU visibility for tokenizer workers (default: empty)
EOF
}

action="run"
case "${1:-}" in
  "") ;;
  --background) action="background" ;;
  --dry-run) action="dry-run" ;;
  --finalize-only) action="finalize-only" ;;
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
RAW_ROOT="${RAW_ROOT:-$WORKSPACE_ROOT/raw_PT_50B_tokens}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$WORKSPACE_ROOT/checkpoints/gpt-oss-20b-hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_ROOT/data/rtj_pt_50b_gptoss}"
RUN_LOG="${RUN_LOG:-$OUTPUT_ROOT/logs/tokenization.log}"
PID_FILE="${PID_FILE:-$OUTPUT_ROOT/tokenization.pid}"
TOKENIZER_ID="${TOKENIZER_ID:-openai/gpt-oss-20b}"
EOS_TOKEN="${EOS_TOKEN:-<|return|>}"
EOS_ID="${EOS_ID:-200002}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-771}"
EXPECTED_SOURCES="${EXPECTED_SOURCES:-9}"
TOKENIZER_WORKERS_PER_JOB="${TOKENIZER_WORKERS_PER_JOB:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"

available_cpus="$(nproc)"
default_parallel_jobs=$((available_cpus / 2))
if (( default_parallel_jobs < 1 )); then
  default_parallel_jobs=1
fi
PARALLEL_JOBS="${PARALLEL_JOBS:-$default_parallel_jobs}"

PREPROCESS_SCRIPT="$MBRIDGE_ROOT/3rdparty/Megatron-LM/tools/preprocess_data.py"
TOKENIZER_JSON="$TOKENIZER_MODEL/tokenizer.json"

for numeric_value in \
  "$PARALLEL_JOBS" \
  "$TOKENIZER_WORKERS_PER_JOB" \
  "$EXPECTED_SHARDS" \
  "$EXPECTED_SOURCES" \
  "$LOG_INTERVAL"; do
  if [[ ! "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "STOP: expected a nonnegative integer, got: $numeric_value" >&2
    exit 2
  fi
done
if (( PARALLEL_JOBS < 1 || TOKENIZER_WORKERS_PER_JOB < 1 || LOG_INTERVAL < 1 )); then
  echo "STOP: PARALLEL_JOBS, TOKENIZER_WORKERS_PER_JOB, and LOG_INTERVAL must be positive." >&2
  exit 2
fi

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "STOP: raw-data root does not exist: $RAW_ROOT" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_JSON" ]]; then
  echo "STOP: tokenizer.json does not exist: $TOKENIZER_JSON" >&2
  exit 1
fi
if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
  echo "STOP: Megatron preprocessing script does not exist: $PREPROCESS_SCRIPT" >&2
  exit 1
fi

export RAW_ROOT TOKENIZER_MODEL OUTPUT_ROOT TOKENIZER_ID EOS_TOKEN EOS_ID
export PREPROCESS_SCRIPT TOKENIZER_WORKERS_PER_JOB LOG_INTERVAL
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${TOKENIZATION_CUDA_VISIBLE_DEVICES:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cleanup_pid_file() {
  if [[ -n "${TOKENIZATION_PID_FILE:-}" && -f "$TOKENIZATION_PID_FILE" ]]; then
    stored_pid="$(<"$TOKENIZATION_PID_FILE")"
    if [[ "$stored_pid" == "$$" ]]; then
      rm -f "$TOKENIZATION_PID_FILE"
    fi
  fi
}
trap cleanup_pid_file EXIT

if [[ "$action" == "background" ]]; then
  mkdir -p "$OUTPUT_ROOT/logs"
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(<"$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "STOP: tokenization is already running with PID $existing_pid" >&2
      echo "Log: $RUN_LOG" >&2
      exit 1
    fi
    rm -f "$PID_FILE"
  fi

  nohup env TOKENIZATION_PID_FILE="$PID_FILE" \
    "$SCRIPT_DIR/tokenize_rtj_pt_50b.sh" \
    >"$RUN_LOG" 2>&1 </dev/null &
  background_pid=$!
  printf '%s\n' "$background_pid" >"$PID_FILE"

  echo "Tokenization started in the background."
  echo "PID: $background_pid"
  echo "OUTPUT_ROOT: $OUTPUT_ROOT"
  echo "RUN_LOG: $RUN_LOG"
  echo "Follow progress: tail -f '$RUN_LOG'"
  exit 0
fi

uv run --no-sync python - "$RAW_ROOT" "$TOKENIZER_JSON" "$EOS_TOKEN" "$EOS_ID" \
  "$EXPECTED_SHARDS" "$EXPECTED_SOURCES" <<'PY'
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

raw_root = Path(sys.argv[1])
tokenizer_json = Path(sys.argv[2])
eos_token = sys.argv[3]
expected_eos_id = int(sys.argv[4])
expected_shards = int(sys.argv[5])
expected_sources = int(sys.argv[6])

sources = sorted(path for path in raw_root.iterdir() if path.is_dir() and path.name != "_state")
shards = sorted(path for source in sources for path in source.glob("part-*.jsonl"))

if expected_sources and len(sources) != expected_sources:
    raise RuntimeError(f"Expected {expected_sources} sources, found {len(sources)}")
if expected_shards and len(shards) != expected_shards:
    raise RuntimeError(f"Expected {expected_shards} JSONL shards, found {len(shards)}")

state_dir = raw_root / "_state"
for source in sources:
    state_path = state_dir / f"{source.name}.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("completed") is not True:
        raise RuntimeError(f"Source is not marked complete: {source.name}")

tokenizer = Tokenizer.from_file(str(tokenizer_json))
actual_eos_id = tokenizer.token_to_id(eos_token)
if actual_eos_id != expected_eos_id:
    raise RuntimeError(
        f"EOS mismatch for {eos_token!r}: expected {expected_eos_id}, tokenizer returned {actual_eos_id}"
    )

temporary_files = sorted(raw_root.glob("*/*.jsonl.*"))
print(
    f"PREFLIGHT PASS: sources={len(sources)} shards={len(shards)} "
    f"eos={eos_token!r}:{actual_eos_id} ignored_temporary_files={len(temporary_files)}"
)
PY

echo "RAW_ROOT: $RAW_ROOT"
echo "TOKENIZER_MODEL: $TOKENIZER_MODEL"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "AVAILABLE_CPUS: $available_cpus"
echo "PARALLEL_JOBS: $PARALLEL_JOBS"
echo "TOKENIZER_WORKERS_PER_JOB: $TOKENIZER_WORKERS_PER_JOB"

if [[ "$action" == "dry-run" ]]; then
  echo "DRY RUN COMPLETE: no output directories or tokenized files were created."
  exit 0
fi

mkdir -p \
  "$OUTPUT_ROOT/data" \
  "$OUTPUT_ROOT/logs" \
  "$OUTPUT_ROOT/completed" \
  "$OUTPUT_ROOT/metadata" \
  "$OUTPUT_ROOT/data_prefixes_by_source"

input_paths="$OUTPUT_ROOT/metadata/input_paths.txt"
input_paths_tmp="$(mktemp "$OUTPUT_ROOT/metadata/input_paths.XXXXXX")"
find "$RAW_ROOT" \
  -mindepth 2 \
  -maxdepth 2 \
  -type f \
  -name 'part-*.jsonl' \
  -printf '%P\n' \
  | LC_ALL=C sort >"$input_paths_tmp"

if [[ -f "$input_paths" ]] && ! cmp -s "$input_paths_tmp" "$input_paths"; then
  echo "STOP: current raw-data shard list differs from the existing run: $input_paths" >&2
  rm -f "$input_paths_tmp"
  exit 1
fi
mv -f "$input_paths_tmp" "$input_paths"

input_paths_sha256="$(sha256sum "$input_paths" | awk '{print $1}')"
tokenizer_sha256="$(sha256sum "$TOKENIZER_JSON" | awk '{print $1}')"
run_fingerprint="$(
  printf '%s\n' \
    "$input_paths_sha256" \
    "$tokenizer_sha256" \
    "$TOKENIZER_ID" \
    "$EOS_TOKEN" \
    "$EOS_ID" \
  | sha256sum \
  | awk '{print $1}'
)"
fingerprint_path="$OUTPUT_ROOT/metadata/run_fingerprint.sha256"
if [[ -f "$fingerprint_path" ]]; then
  stored_fingerprint="$(<"$fingerprint_path")"
  if [[ "$stored_fingerprint" != "$run_fingerprint" ]]; then
    echo "STOP: tokenizer/input settings differ from the existing output run." >&2
    echo "Existing fingerprint: $stored_fingerprint" >&2
    echo "Current fingerprint:  $run_fingerprint" >&2
    exit 1
  fi
else
  printf '%s\n' "$run_fingerprint" >"$fingerprint_path"
fi

process_one() {
  local relative_input="$1"
  local input_path="$RAW_ROOT/$relative_input"
  local relative_stem="${relative_input%.jsonl}"
  local output_prefix="$OUTPUT_ROOT/data/$relative_stem"
  local binary_path="${output_prefix}_text_document.bin"
  local index_path="${output_prefix}_text_document.idx"
  local log_path="$OUTPUT_ROOT/logs/${relative_stem}.log"
  local completed_path="$OUTPUT_ROOT/completed/${relative_stem}.done"

  mkdir -p \
    "$(dirname -- "$output_prefix")" \
    "$(dirname -- "$log_path")" \
    "$(dirname -- "$completed_path")"

  if [[ -f "$completed_path" && -s "$binary_path" && -s "$index_path" ]]; then
    echo "SKIP: $relative_input"
    return 0
  fi

  echo "START: $relative_input"
  if uv run --no-sync python "$PREPROCESS_SCRIPT" \
    --input "$input_path" \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --tokenizer-hf-no-include-special-tokens \
    --append-eod \
    --workers "$TOKENIZER_WORKERS_PER_JOB" \
    --log-interval "$LOG_INTERVAL" \
    --output-prefix "$output_prefix" \
    >"$log_path" 2>&1; then
    if [[ ! -s "$binary_path" || ! -s "$index_path" ]]; then
      echo "FAIL: preprocessing returned success without complete outputs: $relative_input" >&2
      return 1
    fi
    printf 'completed_at=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1 >"${completed_path}.tmp.$$"
    mv -f "${completed_path}.tmp.$$" "$completed_path"
    echo "DONE: $relative_input"
  else
    status=$?
    echo "FAIL: $relative_input (exit=$status, log=$log_path)" >&2
    return "$status"
  fi
}
export -f process_one

if [[ "$action" == "run" ]]; then
  echo "Starting tokenization. Re-running this script resumes completed shards."
  if ! find "$RAW_ROOT" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name 'part-*.jsonl' \
    -printf '%s\t%P\0' \
    | sort -zrn \
    | cut -z -f2- \
    | xargs -0 --no-run-if-empty -n1 -P "$PARALLEL_JOBS" bash -c 'set -euo pipefail; process_one "$1"' _; then
    echo "STOP: one or more shards failed. Fix the reported error and rerun to resume." >&2
    exit 1
  fi
fi

uv run --no-sync python - <<'PY'
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mw_scripts.process_datatrove import indexed_dataset_totals


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output_file:
        temporary_path = Path(output_file.name)
        output_file.write(value)
    temporary_path.replace(path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


output_root = Path(os.environ["OUTPUT_ROOT"])
raw_root = Path(os.environ["RAW_ROOT"])
input_paths_path = output_root / "metadata/input_paths.txt"
relative_inputs = [Path(line) for line in input_paths_path.read_text(encoding="utf-8").splitlines() if line]

prefixes_by_source: dict[str, list[Path]] = {}
for relative_input in relative_inputs:
    relative_prefix = Path("data") / relative_input.parent / f"{relative_input.stem}_text_document"
    prefixes_by_source.setdefault(relative_input.parent.as_posix(), []).append(relative_prefix)

expected_indexes = {output_root / f"{prefix}.idx" for prefixes in prefixes_by_source.values() for prefix in prefixes}
actual_indexes = set((output_root / "data").glob("*/*_text_document.idx"))
if actual_indexes != expected_indexes:
    missing = sorted(str(path) for path in expected_indexes - actual_indexes)
    unexpected = sorted(str(path) for path in actual_indexes - expected_indexes)
    raise RuntimeError(f"Output index mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")

all_prefixes: list[Path] = []
source_results: dict[str, dict[str, object]] = {}
target_tokens_total = 0
for source, prefixes in sorted(prefixes_by_source.items()):
    prefixes = sorted(prefixes)
    all_prefixes.extend(prefixes)
    prefix_file = output_root / "data_prefixes_by_source" / f"{source}.txt"
    write_text_atomic(prefix_file, "".join(f"{prefix.as_posix()}\n" for prefix in prefixes))

    totals = indexed_dataset_totals([output_root / f"{prefix}.idx" for prefix in prefixes])
    state_path = raw_root / "_state" / f"{source}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    target_tokens = int(state["target_tokens"])
    target_tokens_total += target_tokens
    source_results[source] = {
        "data_prefixes_file": str(prefix_file.relative_to(output_root)),
        "input_shards": len(prefixes),
        "original_target_tokens": target_tokens,
        "tokenized": asdict(totals),
    }

all_prefixes = sorted(all_prefixes)
write_text_atomic(
    output_root / "data_prefixes.txt",
    "".join(f"{prefix.as_posix()}\n" for prefix in all_prefixes),
)

all_totals = indexed_dataset_totals([output_root / f"{prefix}.idx" for prefix in all_prefixes])
for source_result in source_results.values():
    source_result["original_target_ratio"] = source_result["original_target_tokens"] / target_tokens_total
    source_result["tokenized_ratio"] = source_result["tokenized"]["tokens"] / all_totals.tokens

tokenizer_json = Path(os.environ["TOKENIZER_MODEL"]) / "tokenizer.json"
tokenizer_sha256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
input_paths_sha256 = hashlib.sha256(input_paths_path.read_bytes()).hexdigest()
completed_at = datetime.now(UTC).isoformat()

manifest = {
    "completed_at": completed_at,
    "input": {
        "input_paths_file": str(input_paths_path.relative_to(output_root)),
        "input_paths_sha256": input_paths_sha256,
        "root": str(raw_root),
        "shards": len(relative_inputs),
        "sources": len(prefixes_by_source),
    },
    "output": {
        "data_dir": "data",
        "data_prefixes_by_source_dir": "data_prefixes_by_source",
        "data_prefixes_file": "data_prefixes.txt",
    },
    "result": asdict(all_totals),
    "schema_version": 1,
    "sources": source_results,
    "tokenizer": {
        "eos_id": int(os.environ["EOS_ID"]),
        "eos_token": os.environ["EOS_TOKEN"],
        "id": os.environ["TOKENIZER_ID"],
        "json_path": str(tokenizer_json),
        "json_sha256": tokenizer_sha256,
    },
}
write_json_atomic(output_root / "manifest.json", manifest)
write_json_atomic(
    output_root / "mixture_manifest.json",
    {
        "created_at": completed_at,
        "original_target_tokens": target_tokens_total,
        "sources": source_results,
        "tokenized_tokens": all_totals.tokens,
    },
)
write_json_atomic(
    output_root / "result.json",
    {
        "completed_at": completed_at,
        "result": asdict(all_totals),
        "tool": "megatron-preprocess-data",
    },
)

print(
    "TOKENIZATION COMPLETE: "
    f"sources={len(prefixes_by_source)} shards={all_totals.shards} "
    f"documents={all_totals.documents} tokens={all_totals.tokens} "
    f"output_bytes={all_totals.output_bytes}"
)
PY
