"""Regression coverage for Mellum's dataset-side attention mask memory policy."""

import importlib
from dataclasses import replace
from pathlib import Path
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


@pytest.mark.parametrize("world_size", [16, 24])
def test_pipeline_disables_unused_masks_without_changing_training_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    from megatron.bridge.models.gpt_provider import GPTModelProvider
    from megatron.bridge.recipes.mellum import mellum as recipe

    monkeypatch.syspath_prepend(str(SCRIPTS))
    configuration = importlib.import_module("configuration")
    training = importlib.import_module("train")
    settings = configuration.read_config(SCRIPTS / "config.yaml")
    settings["run"]["directory"] = str(tmp_path)
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
    bridge.to_megatron_provider.assert_called_once_with(load_weights=True)
    assert not list(tmp_path.iterdir())


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
