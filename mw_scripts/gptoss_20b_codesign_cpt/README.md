# GPT-OSS 20B CodeSign continual pretraining

This folder launches the official 8×B200 GPT-OSS 20B FP8-MX recipe on a
weighted mixture of the tokenized CodeSign datasets.

## 1. Set the mixture

Edit `codesign_mixture.json`. Every key in `source_weights` is a complete
tokenized source/subset path:

- `0` disables that source.
- Any positive number enables it.
- Positive values are relative weights and are normalized automatically, so
  `2, 1, 1` becomes `0.5, 0.25, 0.25`.
- Keep `finepdfs/test` at `0` unless you intentionally want test data in the
  training mixture.

The checked-in config intentionally has every weight set to zero. The launcher
will stop until at least one source is enabled, which prevents an accidental
run with an arbitrary default mixture.

## 2. Validate, then launch

Inside the NeMo container:

```bash
cd /cb/ws/michaelw/Megatron-Bridge

bash mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --preflight
bash mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh
```

The container must mount `/cb/ml-cold` at the same path. The default run and
dataset-index cache are placed under
`/scratch/$USER/training_runs/gptoss_20B_codesign_cpt`.

Follow progress or inspect status with:

```bash
tail -F /scratch/$USER/training_runs/gptoss_20B_codesign_cpt/logs/train.log
bash mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --status
```

For a config-only test that does not start distributed training:

```bash
bash mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh --dryrun
```

## Checkpoint behavior

On the first run, model weights are initialized from
`/workspace/ws/checkpoints/gpt-oss-20b-megatron-ep8`; optimizer, scheduler, and
RNG state start fresh. Once this run has a checkpoint, the launcher restores
its model, optimizer, scheduler, RNG, iteration, and consumed-sample state.

The mixture and launch settings are frozen in the run's `metadata/` directory.
Changing them after a resumable checkpoint exists is rejected; use a new
`RUN_NAME` for a different experiment. Only the newest two training
checkpoints are retained by default to limit scratch usage.

Common overrides:

```bash
RUN_NAME=my_codesign_mix \
TRAIN_TOKENS=10000000000 \
LR=2e-6 \
bash mw_scripts/gptoss_20b_codesign_cpt/gptoss_20B_codesign_cpt_train.sh
```
