# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release build-only document IDs without changing Megatron's dataset identity."""

import logging
from dataclasses import fields

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider, GPTDatasetConfig


LOGGER = logging.getLogger(__name__)


class CompactIndicesBuilder(BlendedMegatronDatasetBuilder):
    """Keep the standard GPTDataset and cache keys, dropping only build-time IDs."""

    def _build_megatron_dataset_splits(
        self,
        dataset_path: str | None,
        split: list[tuple[float, float] | None],
        sizes: list[int | None],
        synchronize_ranks: bool = True,
    ) -> list[GPTDataset | None]:
        # Lazy index loading calculates lengths from self.indices; retain it in
        # that mode rather than silently breaking length/epoch calculations.
        if self.config.defer_npy_index_mmap:
            raise ValueError("CompactIndicesBuilder requires defer_npy_index_mmap=False")
        datasets = super()._build_megatron_dataset_splits(dataset_path, split, sizes, synchronize_ranks)
        for dataset in datasets:
            if dataset is not None:
                # __len__ and __getitem__ use document/sample/shuffle_index,
                # not this arange. Release before subsequent shards are built
                # and before DataLoader workers inherit the dataset objects.
                dataset.indices = None
        return datasets


class CompactGPTDatasetConfig(GPTDatasetConfig, DatasetProvider):
    """Bridge dataset provider that bounds retained document-ID memory."""

    def build_datasets(self, context: DatasetBuildContext) -> tuple[object | None, object | None, object | None]:
        """Build ordinary GPT datasets with unchanged sampling and cache identities.

        Args:
            context: Bridge sample counts and tokenizer initialization context.

        Returns:
            Train, validation and test datasets from the standard blending algorithm.
        """
        if self.mock:
            raise ValueError("CompactGPTDatasetConfig requires real indexed pretraining data")
        LOGGER.info("Building compact GPT datasets; build-only document IDs will not be retained")
        result = CompactIndicesBuilder(
            GPTDataset,
            [context.train_samples, context.valid_samples, context.test_samples],
            lambda: True,
            self,
        ).build()
        train, valid, test = result
        return train, valid, test


def compact_dataset_config(config: GPTDatasetConfig) -> CompactGPTDatasetConfig:
    """Copy a recipe's GPT configuration into the compact dataset provider.

    Args:
        config: Unfinalized recipe dataset configuration.

    Returns:
        Equivalent GPT configuration with a custom builder and no registry patches.
    """
    options = {field.name: getattr(config, field.name) for field in fields(config) if field.init}
    return CompactGPTDatasetConfig(
        skip_getting_attention_mask_from_dataset=config.skip_getting_attention_mask_from_dataset,
        **options,
    )
