# CodeSign tokenization quick start

This is the shortest known-working setup for tokenizing the local CodeSign
corpora with DataTrove in the NeMo 26.06 container. It reads Parquet,
`.json.gz`, and `.jsonl.gz` directly and writes Megatron `.bin/.idx` files to
`/cb/ml-cold/michaelw/codesign_pretrain/tokenized`.

`stack-v3-train` uses its repo-to-text renderer while tokenizing, so it does not
materialize rendered repository text as an intermediate dataset.

## 1. Start the Docker container on the GPU node

```bash
docker run -it \
  --network=host \
  --runtime=runc \
  --device=nvidia.com/gpu=all \
  --name megatron_bridge_mw \
  --shm-size=64g \
  --cap-add SYS_NICE \
  -e HOME=/workspace \
  -e USER=michaelw \
  -e PYTHONNOUSERSITE=1 \
  -e NCCL_DEBUG=WARN \
  -e XDG_CACHE_HOME=/tmp/mbridge-cache \
  -e HF_HOME=/tmp/mbridge-hf \
  -e PIP_CACHE_DIR=/tmp/mbridge-pip-cache \
  -e FLASHINFER_WORKSPACE_BASE=/tmp/mbridge-flashinfer \
  -e MPLCONFIGDIR=/tmp/mbridge-matplotlib \
  -e MEGATRON_CONFIG_LOCK_DIR=/tmp/mbridge-locks \
  -v "$HOME:/workspace" \
  -v /cb/ws/michaelw:/cb/ws/michaelw \
  -v /scratch/michaelw:/scratch/michaelw \
  -v /cb/ml-cold:/cb/ml-cold:rw \
  -w /workspace \
  nvcr.io/nvidia/nemo:26.06
```

The `/cb/ml-cold` mount is required. The GPU is not used by tokenization, but
keeping the standard NeMo command makes this container usable for training too.

If the named container already exists, start it instead:

```bash
docker start -ai megatron_bridge_mw
```

## 2. Set up and check everything inside the container

```bash
cd /cb/ws/michaelw/Megatron-Bridge
bash tokenization/setup_codesign_tokenization.sh --check
```

The helper uses these fixed paths:

```text
input:      /cb/ml-cold/michaelw/codesign_pretrain/processed_hf
output:     /cb/ml-cold/michaelw/codesign_pretrain/tokenized
DataTrove:  /cb/ws/michaelw/datatrove
Python:     /opt/venv/bin/python
tokenizer:  /workspace/ws/checkpoints/gpt-oss-20b-hf/tokenizer.json
workers:    96 (one per physical core)
CPU set:    0-95 (avoids the sibling SMT threads)
EOS token:  <|return|>
```

If `humanize` is missing, the helper installs it automatically from the internal
Artifactory using `uv`. The certificate bypass is restricted to that exact
internal hostname. The install uses `--no-config` so the NVIDIA package index in
the repository's `pyproject.toml` cannot override the internal index.

## 3. Dry run, then start

```bash
bash tokenization/setup_codesign_tokenization.sh --dry-run
bash tokenization/setup_codesign_tokenization.sh --background
tail -f tokenization/tokenization.log
```

The job is resumable. If Docker or the process stops, start the container again
and rerun the `--background` command. Existing dataset manifests cause completed
DataTrove tasks to be skipped.

Normal startup looks like:

```text
Processing 2334 parquet files with tasks=192 workers=96
Launching pipeline for rank=...
Tokenizing in "dataset_....._tokens"...
```

One dataset is complete when its output contains `result.json` and
`data_prefixes.txt`, and the master log prints `DONE: <dataset>`.

## 4. Leave the container running

Do not run `exit` in the main container shell. Detach with `Ctrl-p`, then
`Ctrl-q`. Re-enter with:

```bash
docker exec -it megatron_bridge_mw bash
```

`nohup` keeps the job alive after detaching, but cannot keep it alive if the
container itself stops.

## 5. Quick monitoring

```bash
tail -f /cb/ws/michaelw/Megatron-Bridge/tokenization/tokenization.log
```

```bash
watch -n 30 '
du -sh /cb/ml-cold/michaelw/codesign_pretrain/tokenized/Nemotron-CC-Code-v1
find /cb/ml-cold/michaelw/codesign_pretrain/tokenized/Nemotron-CC-Code-v1/data \
  -name "*.idx" | wc -l
'
```

For CPU and I/O utilization:

```bash
mpstat -P ALL 2
free -h
vmstat 2
nfsiostat 2
```

The helper binds the job to CPU `0-95`, which selects one hardware thread from
each of the machine's 96 physical cores. If I/O wait is high, more workers will
probably not help.

## 6. Build zero-copy subset views

The launcher discovers sibling source directories and tokenizes each one into a
separate output directory. For example:

```text
processed_hf/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-STEM-SFT
    -> tokenized/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-STEM-SFT
```

Older merged outputs can be converted into zero-copy subset views with:

```bash
/opt/venv/bin/python tokenization/split_tokenized_by_subset.py \
  --source-root /cb/ml-cold/michaelw/codesign_pretrain/processed_hf \
  --tokenized-root /cb/ml-cold/michaelw/codesign_pretrain/tokenized \
  --workers 32
```

The command is resumable and does not copy or retokenize documents. It writes a
filtered `.idx` per subset and creates a relative `.bin` symlink to the original
token shard. Finalize the clean, flat hierarchy and remove superseded merged
indexes and run metadata with:

```bash
/opt/venv/bin/python tokenization/finalize_tokenized_hierarchy.py \
  --tokenized-root /cb/ml-cold/michaelw/codesign_pretrain/tokenized
```

Each subset is placed directly under `<dataset>/<subset>/` and has its own
`data_prefixes.txt` and `manifest.json`. Shared token binaries live under
`<dataset>/_shared_bin/`; the subset `.bin` entries are relative symlinks.

When assigning one mixture weight to a subset, distribute that weight among its
prefixes in proportion to the per-shard token counts in `manifest.json`. Do not
infer subset tokens from symlinked `.bin` sizes: every subset symlink exposes
the full parent binary shard even though its filtered index selects only that
subset's documents.

## Common failures

- `No module named datatrove`: `/cb/ws/michaelw/datatrove` is missing or was not
  mounted.
- `No module named humanize`: rerun the helper; it installs the package before
  preflight.
- `UnknownIssuer`: the helper already uses the working hostname-scoped
  Artifactory bypass.
- `Failed to download pyarrow --offline`: do not run a full DataTrove `uv sync`.
  The NeMo environment already has a compatible PyArrow.
- A log path beginning with `$/workspace`: remove the extra `$`; the helper uses
  `tokenization/tokenization.log` automatically.
