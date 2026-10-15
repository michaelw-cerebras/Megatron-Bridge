# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reverify exact weight preservation for a derived surgery checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from surgeon import resolve_hf_source, verify_checkpoint


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Resolve the source and verify a derived surgery checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original Hugging Face model ID or local directory")
    parser.add_argument("--derived", "--parallel", dest="derived", required=True, help="Derived HF directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    derived_dir = Path(args.derived)
    source_dir = resolve_hf_source(args.source, target_dir=derived_dir, cache_dir=None, revision=None)
    result = verify_checkpoint(source_dir, derived_dir)
    LOGGER.info(
        "PASS %s: %d tensors across %d exact weight shards (%d same-inode, %d SHA-256)",
        result.variant_id,
        result.parameter_tensors,
        result.weight_shards,
        result.same_inode_shards,
        result.sha256_shards,
    )


if __name__ == "__main__":
    main()
