# GPT-OSS preprocessing throughput benchmark

This benchmark measures DataTrove and Megatron-LM preprocessing on the same pinned GPT-OSS tokenizer and text
corpus. It is separate from the exact-token parity check.

The default input is the `Nemotron-Pretraining-Legal-eCFR-QA` configuration from
`nvidia/Nemotron-Pretraining-Legal-v1` (about 593M source-reported tokens). The source has only four Parquet
files, which cannot keep a 96-core DataTrove run busy. The preparation step therefore streams it once and writes
192 balanced local Parquet shards. This preparation is not part of the timed benchmark.

The preparation step also writes one reusable JSONL file because the checked-in Megatron-LM
`tools/preprocess_data.py` only accepts JSONL.

## 1. Prepare the input on the development VM

Run in the DataTrove environment. The input directory is self-contained and uses relative paths in its manifest,
so it can be copied to node-local storage later.

```bash
THROUGHPUT_DIR="$HOME/ws/gpt_oss_throughput_data"
SCRIPT_DIR="$HOME/ws/Megatron-Bridge/scripts/data_preprocessing/gpt_oss_throughput"

python "$SCRIPT_DIR/prepare_benchmark_data.py" \
  --work-dir "$THROUGHPUT_DIR" \
  --num-shards 192
```

The outputs are:

```text
gpt_oss_throughput_data/
├── input.jsonl
├── input_parquet/
│   ├── part_00000.parquet
│   └── ... 192 balanced shards total
├── manifest.json
└── tokenizer/
```

Preparation refuses to overwrite an existing manifest or input directory. The manifest pins immutable dataset and
tokenizer revisions and records row counts, byte sizes, and shard balance.

## 2. Stage onto the reserved node

For a CPU throughput measurement, place the self-contained input directory on node-local SSD/NVMe when one is
available. Reading input or writing output on NFS/EFS can turn the experiment into a shared-filesystem benchmark.

Use a unique persistent directory for benchmark results. If results are written under node-local temporary storage,
copy each `result.json` out before releasing the reservation.

Set `TOKENIZERS_PARALLELISM=false` consistently. Both runners set it when the variable is not already defined.

## 3. Benchmark DataTrove

Run in the DataTrove environment. Keep `tasks=192` fixed across the sweep and vary only `workers`:

```bash
python "$SCRIPT_DIR/benchmark_datatrove.py" \
  --manifest "$THROUGHPUT_DIR/manifest.json" \
  --run-dir "$THROUGHPUT_DIR/runs/datatrove_w32" \
  --tasks 192 \
  --workers 32 \
  --discard-output
```

Suggested sweep:

```text
workers = 1, 8, 16, 32, 48, 64, 96
tasks   = 192 for every run
```

The timed scope is:

```text
local Parquet read → batched GPT-OSS tokenization → Megatron .bin/.idx write
```

`--discard-output` removes only the new `RUN_DIR/output` directory after its indexes have been validated and metrics
have been recorded. `logs/` and `result.json` remain. Omit the flag to retain tokenized output.

## 4. Benchmark Megatron-LM

Run inside the NeMo/Megatron container. The default `tokenization-only` mode uses the reusable JSONL from the
preparation step:

```bash
python "$SCRIPT_DIR/benchmark_megatron.py" \
  --manifest "$THROUGHPUT_DIR/manifest.json" \
  --preprocess-script "$HOME/ws/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py" \
  --run-dir "$THROUGHPUT_DIR/runs/megatron_tokenize_w32" \
  --workers 32 \
  --mode tokenization-only \
  --discard-output
```

Use the same worker sweep as DataTrove. Megatron does not have a separate `tasks` setting: one coordinator reads the
JSONL and dispatches documents to its tokenizer worker pool.

To measure the additional format-conversion cost imposed by Megatron's JSONL-only entry point, run the best worker
configuration once in `end-to-end` mode. Point `--temp-dir` at node-local storage:

```bash
python "$SCRIPT_DIR/benchmark_megatron.py" \
  --manifest "$THROUGHPUT_DIR/manifest.json" \
  --preprocess-script "$HOME/ws/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py" \
  --run-dir "$THROUGHPUT_DIR/runs/megatron_e2e_w32" \
  --workers 32 \
  --mode end-to-end \
  --temp-dir /tmp \
  --discard-output
```

The result separates:

- `materialization_seconds`: local Parquet to temporary JSONL;
- `tokenization_seconds`: Megatron JSONL tokenization and indexed-dataset writing;
- `wall_seconds`: the complete selected benchmark scope.

## Result schema

Each run writes `RUN_DIR/result.json` with:

- document, sequence, token, input-byte, and output-byte counts;
- wall time, documents/s, and tokens/s;
- worker/task/batch settings;
- CPU affinity count, hostname, Python version, and relevant package versions;
- the exact manifest used for the run.

The runner parses every generated `.idx` file and validates its corresponding `.bin` size before reporting success.
Run the two tools on an otherwise idle node, alternate tool order between repetitions, and repeat the winning
configurations at least three times before drawing conclusions.
