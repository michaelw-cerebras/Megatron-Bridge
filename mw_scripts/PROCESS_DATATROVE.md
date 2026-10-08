# Production Hugging Face → Megatron preprocessing

`process_datatrove.py` streams local or Hugging Face shards and writes Megatron mmap indexed datasets
(`.bin` + `.idx`). It does not create JSONL or copy the source corpus into the output directory. Hugging Face
inputs support Parquet; local inputs support Parquet, JSONL, `jsonl.gz`, and line-delimited `json.gz`.

The script is intended for datasets whose documents live in a `text` column. It reads only that column by default
and generates stable IDs from the source file path and row number. IDs do not alter the tokenized Megatron output.

## Environment

Use the same DataTrove 0.9.0 environment as the throughput benchmark. The required packages are DataTrove,
PyArrow, Hugging Face Hub, fsspec, and tokenizers. Private datasets use the normal `HF_TOKEN` environment variable.
For air-gapped local input, pass a local tokenizer directory or `tokenizer.json` to `--tokenizer`; no Hub request is
made when that path exists.

## Process an entire dataset repository

This selects every Parquet file in the pinned dataset revision:

```bash
python mw_scripts/process_datatrove.py \
  --dataset nvidia/Nemotron-Pretraining-Legal-v1 \
  --output-folder /path/to/nemotron_legal_gptoss \
  --workers 96
```

The default tokenizer is `openai/gpt-oss-20b`, and the default end-of-document token is `<|return|>`.
The script resolves both `main` revisions to immutable commit SHAs before processing.

## Process local files

Local Parquet and compressed JSONL are read in place:

```bash
python mw_scripts/process_datatrove.py \
  --input-folder /path/to/local/dataset \
  --path-glob '*.jsonl.gz' \
  --output-folder /path/to/tokenized/dataset \
  --workers 96
```

Each invocation must select one input format. Use separate output folders when a corpus mixes Parquet and JSONL.

## Select one or more configs

For repositories that store configurations in top-level directories:

```bash
python mw_scripts/process_datatrove.py \
  --dataset nvidia/Nemotron-Pretraining-Legal-v1 \
  --config Nemotron-Pretraining-Legal-eCFR \
  --config Nemotron-Pretraining-Legal-eCFR-QA \
  --output-folder /path/to/nemotron_legal_ecfr_gptoss \
  --workers 96
```

Use `--path-glob` for repositories with a different file layout. The pattern is matched against the full
repository-relative path. For example, `--path-glob 'data/train-*.parquet'`.

## Text-only datasets

No special argument is needed when the dataset has only a `text` column. For other schemas:

```bash
python mw_scripts/process_datatrove.py \
  --dataset org/dataset \
  --text-column content \
  --id-column uuid \
  --output-folder /path/to/output \
  --workers 96
```

## Resume

Rerun the identical command with `--resume`:

```bash
python mw_scripts/process_datatrove.py \
  --dataset nvidia/Nemotron-Pretraining-Legal-v1 \
  --output-folder /path/to/nemotron_legal_gptoss \
  --workers 96 \
  --resume
```

The immutable dataset file list and tokenizer are reused from `manifest.json`; the Hub is not re-resolved.
Do not change `--tasks`, config selection, columns, tokenizer, EOS, batch sizes, or filename prefix while resuming.
The worker count may change.

## Output layout

```text
OUTPUT_FOLDER/
├── data/
│   ├── dataset_00000_tokens.bin
│   ├── dataset_00000_tokens.idx
│   └── ...
├── logs/
├── metadata/
│   ├── input_paths.txt
│   └── tokenizer/tokenizer.json
├── data_prefixes.txt
├── manifest.json
└── result.json
```

`manifest.json` pins the dataset and tokenizer commits, exact Parquet path-list hash, EOS ID, columns, and task
count. `result.json` is written only after every `.idx` is parsed and every `.bin` size is validated.

Each input Parquet file is assigned to one task. With the default settings, tasks are
`min(number of Parquet files, 8 × workers)`, keeping each task shorter so preemption replays less work.
If the dataset contains fewer Parquet files than requested workers, the effective worker count is reduced.
Existing runs keep the task count pinned in their manifests; changing that count requires starting a new output.
Re-sharding the input is required to obtain more file-level parallelism.

For Megatron training, each pair under `data/` is a valid mmap indexed-dataset prefix. `data_prefixes.txt` lists
all prefixes relative to the output directory, so the complete directory can be moved without rewriting the list.
For example, the corresponding `mini_gptoss.py` setup can load the paths with:

```python
from pathlib import Path

dataset_root = Path("/path/to/nemotron_legal_gptoss")
cfg.dataset.mock = False
cfg.dataset.data_path = [
    str(dataset_root / prefix)
    for prefix in (dataset_root / "data_prefixes.txt").read_text().splitlines()
]
```
