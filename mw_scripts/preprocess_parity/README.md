# GPT-OSS preprocessing parity check

This check uses all configurations and all rows in `nvidia/Nemotron-Pretraining-Dataset-sample`. The preparation step pins immutable dataset and tokenizer revisions but does not download or persist dataset Parquet shards.

DataTrove uses the production-shaped streaming path:

```text
Hugging Face Parquet shards → HuggingFaceDatasetReader(streaming=True)
                            → MegatronDocumentTokenizer → .bin/.idx
```

Megatron-LM's checked-in `tools/preprocess_data.py` accepts JSONL only. Its wrapper streams the same pinned Hugging Face configs into a temporary JSONL under `/tmp`, preprocesses it, and deletes the JSONL automatically.

## 1. Pin the inputs in the DataTrove venv

```bash
PARITY_DIR="$HOME/ws/gpt_oss_data_parity"
SCRIPT_DIR="$HOME/ws/Megatron-Bridge/scripts/data_preprocessing/gpt_oss_parity"

python "$SCRIPT_DIR/prepare_input.py" \
  --work-dir "$PARITY_DIR"
```

This writes a small `manifest.json` and downloads only the GPT-OSS tokenizer snapshot. The manifest contains all 10 configs and the exact dataset commit.

## 2. Stream directly into DataTrove

Use one task for parity so the result is a single deterministically ordered shard:

```bash
python "$SCRIPT_DIR/tokenize_datatrove.py" \
  --manifest "$PARITY_DIR/manifest.json" \
  --tokenizer-json "$PARITY_DIR/tokenizer/tokenizer.json" \
  --output-dir "$PARITY_DIR/datatrove" \
  --tasks 1 \
  --workers 1
```

The resulting prefix is:

```text
$PARITY_DIR/datatrove/datatrove_00000_tokens
```

For a production throughput run, increase `--tasks` and `--workers`. Each task streams its assigned source shards or rows and writes an independent Megatron indexed-dataset shard.

If a multi-task run is interrupted, rerun the same command with `--resume`. DataTrove will skip ranks recorded as complete and rerun incomplete ranks.

## 3. Run Megatron-LM in the NeMo container

Run this command from the Megatron Bridge checkout mounted into the container:

```bash
python "$SCRIPT_DIR/tokenize_megatron.py" \
  --manifest "$PARITY_DIR/manifest.json" \
  --tokenizer-dir "$PARITY_DIR/tokenizer" \
  --preprocess-script "$HOME/ws/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py" \
  --output-dir "$PARITY_DIR/megatron" \
  --temp-dir /tmp \
  --workers 8
```

The wrapper disables automatic Hugging Face special-token insertion and enables Megatron `--append-eod`. DataTrove explicitly appends the GPT-OSS EOS token `<|return|>`, so both sides implement `encode(text, no automatic BOS/EOS) + EOS`.

## 4. Compare the results

The comparison script only requires NumPy and can run in either environment:

```bash
python "$SCRIPT_DIR/compare_indexed_datasets.py" \
  --datatrove-prefix "$PARITY_DIR/datatrove/datatrove_00000_tokens" \
  --megatron-prefix "$PARITY_DIR/megatron/megatron_text_document"
```

A successful check exits with status 0 and logs `PARITY PASS`. A mismatch exits with status 1 and reports the first differing token position and a small token window from each output.

## What streaming means

Streaming avoids keeping the entire source corpus on local disk. Parquet bytes still travel over the network, but only active buffers and row groups are held locally while documents flow through the pipeline. The final `.bin/.idx` output must still be stored.

For a 12T-token production run, direct remote streaming trades local input capacity for network dependency. A robust deployment normally uses many resumable DataTrove tasks, writes independent output shards, and may use a bounded local cache for active Parquet shards. It does not require all source Parquet data and all tokenized output to coexist on one node.
