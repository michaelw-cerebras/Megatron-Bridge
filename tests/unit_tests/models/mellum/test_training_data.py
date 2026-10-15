"""Regression coverage for Mellum's dataset-side attention mask memory policy."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers.text.libraries.null_tokenizer import NullTokenizer


SCRIPTS = Path(__file__).resolve().parents[4] / "mw_scripts/mellum2_3xh200"
pytestmark = pytest.mark.unit


def test_scheduler_recovery_overrides_checkpoint_param_group_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    training = importlib.import_module("train")
    optimizer = SimpleNamespace(
        param_groups=[
            {"max_lr": 3e-6, "min_lr": 3e-7, "lr": 6e-7, "is_decoupled_lr": False},
            {"max_lr": 3e-6, "min_lr": 3e-7, "lr": 6e-7, "is_decoupled_lr": False},
        ]
    )
    scheduler = Mock(max_lr=3e-8, min_lr=3e-9, num_steps=0)

    training._override_optimizer_param_group_schedule(
        optimizer, scheduler, max_lr=3e-8, min_lr=3e-9, resume_num_steps=96_000
    )

    assert all(group["max_lr"] == 3e-8 for group in optimizer.param_groups)
    assert all(group["min_lr"] == 3e-9 for group in optimizer.param_groups)
    scheduler.step.assert_called_once_with(96_000)


@pytest.mark.parametrize("world_size", [16, 24])
@pytest.mark.parametrize("resume", [False, True])
def test_pipeline_disables_unused_masks_without_changing_training_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world_size: int, resume: bool
) -> None:
    from megatron.bridge.models.gpt_provider import GPTModelProvider
    from megatron.bridge.recipes.mellum import mellum as recipe

    monkeypatch.syspath_prepend(str(SCRIPTS))
    configuration = importlib.import_module("configuration")
    training = importlib.import_module("train")
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
    settings["run"]["resume_checkpoint"] = None
    if resume:
        checkpoint = tmp_path / "checkpoints"
        checkpoint.mkdir()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("5000")
        settings["run"]["resume_checkpoint"] = str(checkpoint)
    digest = configuration.config_digest(settings)
    mixture = {
        "config_digest": digest,
        "hf_snapshot": str(tmp_path / "hf"),
        "shards": [{"prefix": str(tmp_path / "tokens"), "weight": 1.0}],
    }
    provider = GPTModelProvider(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        num_moe_experts=64,
        vocab_size=98304,
        seq_length=8192,
    )
    bridge = Mock()
    bridge.to_megatron_provider.return_value = provider
    monkeypatch.setattr(recipe.AutoBridge, "from_hf_pretrained", Mock(return_value=bridge))

    cfg = training.build_config(settings, mixture, world_size=world_size)

    assert cfg.dataset.create_attention_mask is False
    assert cfg.dataset.skip_getting_attention_mask_from_dataset is True
    assert cfg.dataset.reset_attention_mask is False
    assert cfg.dataset.reset_position_ids is False
    assert cfg.dataset.eod_mask_loss is False
    assert cfg.dataset.mmap_bin_files is False
    from megatron.bridge.training.config import DatasetProvider

    assert isinstance(cfg.dataset, DatasetProvider)
    assert cfg.dataset.num_workers == settings["dataset"]["num_workers"]
    assert cfg.dataset.sequence_length == settings["train"]["sequence_length"]
    assert cfg.dataset.random_seed == settings["train"]["seed"]
    assert cfg.dataset.path_to_cache == str(tmp_path / "dataset_cache")
    assert cfg.dataset.data_path == ["1.0", str(tmp_path / "tokens")]
    assert cfg.train.global_batch_size == settings["train"]["global_batch_size"]
    assert cfg.train.micro_batch_size == settings["train"]["micro_batch_size"]
    assert cfg.optimizer.lr == settings["optimizer"]["lr"]
    assert cfg.scheduler.lr_decay_style == settings["scheduler"]["lr_decay_style"]
    assert cfg.scheduler.lr_warmup_iters == settings["scheduler"]["lr_warmup_iters"]
    assert configuration.config_digest(settings) == digest
    bridge.to_megatron_provider.assert_called_once_with(load_weights=not resume)
    assert cfg.checkpoint.load == settings["run"]["resume_checkpoint"]
    assert cfg.checkpoint.load_optim is resume
    assert cfg.checkpoint.load_rng is resume
    assert cfg.scheduler.use_checkpoint_opt_param_scheduler is resume
    assert sorted(path.name for path in tmp_path.iterdir()) == (["checkpoints"] if resume else [])
    cfg.to_yaml(tmp_path / "ConfigContainer.yaml")
    assert "CompactGPTDatasetConfig" in (tmp_path / "ConfigContainer.yaml").read_text()


def test_maskless_dataset_preserves_samples_and_reuses_cached_indices(tmp_path: Path) -> None:
    prefix = str(tmp_path / "tokens")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    builder.add_item(torch.arange(1, 65, dtype=torch.int64))
    builder.end_document()
    builder.finalize(prefix + ".idx")
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=16,
        tokenizer=NullTokenizer(98304, eod_id=0),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        split="1,0,0",
        path_to_cache=str(tmp_path / "cache"),
        create_attention_mask=True,
    )
    indexed = IndexedDataset(prefix)
    indices = np.arange(1, dtype=np.int32)
    dense = GPTDataset(indexed, prefix, indices, 2, Split.train, config)
    cache = Path(config.path_to_cache)
    before = {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in cache.iterdir()}
    assert before

    maskless = GPTDataset(indexed, prefix, indices, 2, Split.train, replace(config, create_attention_mask=False))

    assert maskless.unique_description_hash == dense.unique_description_hash
    assert before == {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in cache.iterdir()}
    for field in ("document_index", "sample_index", "shuffle_index"):
        np.testing.assert_array_equal(getattr(dense, field), getattr(maskless, field))
    # Repeated access exercises the per-shard mask cache as well as initial creation.
    for index in (0, 1, 0):
        original = dense[index]
        actual = maskless[index]
        assert set(actual) == set(original) - {"attention_mask"}
        for field in actual:
            assert torch.equal(actual[field], original[field]), field
        assert maskless.cached_attention_mask is None
    assert dense.cached_attention_mask.shape == (1, 16, 16)


@pytest.mark.parametrize("warm_cache", [True, False])
def test_compact_builder_preserves_blend_cache_and_resumed_worker_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warm_cache: bool
) -> None:
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder

    from megatron.bridge.data.samplers import build_pretraining_data_loader
    from megatron.bridge.data.utils import get_dataset_provider
    from megatron.bridge.training.config import GPTDatasetConfig as BridgeGPTDatasetConfig

    monkeypatch.syspath_prepend(str(SCRIPTS))
    compact = importlib.import_module("compact_data")
    prefixes = [str(tmp_path / f"tokens_{index}") for index in range(2)]
    for prefix in prefixes:
        builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
        for document in range(32):
            builder.add_item(torch.arange(1, 65, dtype=torch.int64) + document)
            builder.end_document()
        builder.finalize(prefix + ".idx")
    config = BridgeGPTDatasetConfig(
        random_seed=1234,
        sequence_length=16,
        tokenizer=NullTokenizer(98304, eod_id=0),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        split="1,0,0",
        blend=(prefixes, [2.0, 1.0]),
        path_to_cache=str(tmp_path / "original_cache"),
    )
    config.finalize()
    # More than one epoch exercises construction as well as the cache-hit path.
    reference = BlendedMegatronDatasetBuilder(GPTDataset, [1000, 0, 0], lambda: True, config).build()[0]
    assert all(dataset.indices is not None for dataset in reference.datasets)
    before = {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in Path(config.path_to_cache).iterdir()}
    optimized_config = compact.compact_dataset_config(config)
    optimized_config.mmap_bin_files = False
    if not warm_cache:
        optimized_config.path_to_cache = str(tmp_path / "new_cache")
    optimized_config.finalize()
    provider = get_dataset_provider(optimized_config)
    actual, valid, test = provider([1000, 0, 0], optimized_config)

    assert valid is None and test is None
    assert actual.unique_description_hash == reference.unique_description_hash
    assert len(actual) == len(reference)
    assert all(dataset.indices is None for dataset in actual.datasets)
    for old, new in zip(reference.datasets, actual.datasets):
        assert new.unique_description_hash == old.unique_description_hash
        assert new.dataset.mmap is False
        for field in ("document_index", "sample_index", "shuffle_index"):
            np.testing.assert_array_equal(getattr(new, field), getattr(old, field))
    after = {
        file.name: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in Path(optimized_config.path_to_cache).iterdir()
    }
    assert {name: data[0] for name, data in after.items()} == {name: data[0] for name, data in before.items()}
    if warm_cache:
        assert after == before  # No sampling indices were rewritten.
    for index in (0, 1, 503, 999):
        old, new = reference[index], actual[index]
        assert old.keys() == new.keys()
        for field in new:
            if isinstance(new[field], torch.Tensor):
                assert torch.equal(new[field], old[field]), field
            else:
                assert new[field] == old[field]
    # A nonzero consumed_samples simulates checkpoint resume; workers receive
    # identical samples without inheriting build-only ID arrays or bin mappings.
    loader_options = dict(
        consumed_samples=500,
        dataloader_type="single",
        micro_batch_size=1,
        num_workers=2,
        data_sharding=True,
        pin_memory=False,
        persistent_workers=False,
        data_parallel_rank=1,
        data_parallel_size=2,
    )
    old_iterator = iter(build_pretraining_data_loader(reference, **loader_options))
    new_iterator = iter(build_pretraining_data_loader(actual, **loader_options))
    try:
        for _ in range(8):
            old, new = next(old_iterator), next(new_iterator)
            for field in new:
                assert torch.equal(new[field], old[field]), field
    finally:
        old_iterator._shutdown_workers()
        new_iterator._shutdown_workers()


def test_compact_builder_rejects_deferred_length_calculations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.bridge.training.config import GPTDatasetConfig as BridgeGPTDatasetConfig

    monkeypatch.syspath_prepend(str(SCRIPTS))
    compact = importlib.import_module("compact_data")
    config = BridgeGPTDatasetConfig(
        random_seed=1234, sequence_length=16, defer_npy_index_mmap=True, blend=(["unused"], [1.0])
    )
    builder = compact.CompactIndicesBuilder(GPTDataset, [1, 0, 0], lambda: True, config)
    with pytest.raises(ValueError, match="defer_npy_index_mmap=False"):
        builder._build_megatron_dataset_splits(None, [(0.0, 1.0), None, None], [1, 0, 0])
