# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Compare two Megatron mmap indexed datasets document by document.

Usage:
python compare_indexed_datasets.py \
  --datatrove-prefix raw_data/datatrove/datatrove_00000_tokens \
  --megatron-prefix raw_data/megatron/megatron_text_document
"""

import argparse
import logging
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


logger = logging.getLogger(__name__)
INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPES = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.float64),
    7: np.dtype(np.float32),
    8: np.dtype(np.uint16),
}


@dataclass(frozen=True)
class IndexedDataset:
    """Metadata and memory-mapped tokens for one indexed dataset."""

    prefix: Path
    dtype: np.dtype
    sequence_lengths: np.ndarray
    sequence_pointers: np.ndarray
    document_indices: np.ndarray
    tokens: np.memmap

    def sequence(self, index: int) -> np.ndarray:
        """Return one token sequence."""
        start = int(self.sequence_pointers[index]) // self.dtype.itemsize
        length = int(self.sequence_lengths[index])
        return self.tokens[start : start + length]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datatrove-prefix", type=Path, required=True, help="Prefix without .bin/.idx.")
    parser.add_argument("--megatron-prefix", type=Path, required=True, help="Prefix without .bin/.idx.")
    parser.add_argument("--max-mismatches", type=int, default=10, help="Maximum mismatches to report.")
    return parser.parse_args()


def read_exact(input_file: BinaryIO, size: int) -> bytes:
    """Read exactly size bytes or fail with a useful error."""
    value = input_file.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of index file; wanted {size} bytes, got {len(value)}")
    return value


def load_indexed_dataset(prefix: Path) -> IndexedDataset:
    """Load Megatron mmap index metadata and memory-map its token data."""
    index_path = prefix.with_suffix(".idx")
    binary_path = prefix.with_suffix(".bin")
    if not index_path.is_file() or not binary_path.is_file():
        raise FileNotFoundError(f"Expected both {index_path} and {binary_path}")

    with index_path.open("rb") as input_file:
        header = read_exact(input_file, len(INDEX_HEADER))
        if header != INDEX_HEADER:
            raise ValueError(f"Invalid indexed-dataset header in {index_path}: {header!r}")
        version = struct.unpack("<Q", read_exact(input_file, 8))[0]
        if version != 1:
            raise ValueError(f"Unsupported indexed-dataset version {version} in {index_path}")
        dtype_code = struct.unpack("<B", read_exact(input_file, 1))[0]
        if dtype_code not in DTYPES:
            raise ValueError(f"Unsupported dtype code {dtype_code} in {index_path}")
        sequence_count = struct.unpack("<Q", read_exact(input_file, 8))[0]
        document_count = struct.unpack("<Q", read_exact(input_file, 8))[0]
        sequence_lengths = np.frombuffer(read_exact(input_file, sequence_count * 4), dtype="<i4").copy()
        sequence_pointers = np.frombuffer(read_exact(input_file, sequence_count * 8), dtype="<i8").copy()
        document_indices = np.frombuffer(read_exact(input_file, document_count * 8), dtype="<i8").copy()
        if input_file.read(1):
            raise ValueError(f"Unexpected trailing bytes in {index_path}")

    dtype = DTYPES[dtype_code]
    tokens = np.memmap(binary_path, mode="r", dtype=dtype)
    expected_tokens = int(sequence_lengths.sum())
    if tokens.size != expected_tokens:
        raise ValueError(f"{binary_path} contains {tokens.size} tokens; index declares {expected_tokens}")
    return IndexedDataset(prefix, dtype, sequence_lengths, sequence_pointers, document_indices, tokens)


def compare(left: IndexedDataset, right: IndexedDataset, max_mismatches: int) -> bool:
    """Compare indexed datasets and log actionable mismatch details."""
    matches = True
    if left.dtype != right.dtype:
        logger.error("Dtype mismatch: DataTrove=%s Megatron=%s", left.dtype, right.dtype)
        matches = False
    if not np.array_equal(left.document_indices, right.document_indices):
        logger.error(
            "Document-index mismatch: DataTrove shape=%s Megatron shape=%s",
            left.document_indices.shape,
            right.document_indices.shape,
        )
        matches = False
    if not np.array_equal(left.sequence_lengths, right.sequence_lengths):
        logger.error(
            "Sequence-length mismatch: DataTrove sequences=%d Megatron sequences=%d",
            len(left.sequence_lengths),
            len(right.sequence_lengths),
        )
        matches = False

    mismatch_count = 0
    common_sequences = min(len(left.sequence_lengths), len(right.sequence_lengths))
    for sequence_index in range(common_sequences):
        left_tokens = left.sequence(sequence_index)
        right_tokens = right.sequence(sequence_index)
        if np.array_equal(left_tokens, right_tokens):
            continue
        matches = False
        mismatch_count += 1
        if mismatch_count <= max_mismatches:
            common_tokens = min(len(left_tokens), len(right_tokens))
            unequal = np.flatnonzero(left_tokens[:common_tokens] != right_tokens[:common_tokens])
            first_position = int(unequal[0]) if unequal.size else common_tokens
            window_start = max(0, first_position - 5)
            window_end = first_position + 6
            logger.error(
                "Sequence %d differs at token %d: lengths=(%d, %d), DataTrove=%s, Megatron=%s",
                sequence_index,
                first_position,
                len(left_tokens),
                len(right_tokens),
                left_tokens[window_start:window_end].tolist(),
                right_tokens[window_start:window_end].tolist(),
            )

    if mismatch_count > max_mismatches:
        logger.error("Suppressed %d additional sequence mismatches", mismatch_count - max_mismatches)
    if matches:
        logger.info(
            "PARITY PASS: %d documents, %d sequences, %d tokens",
            len(left.document_indices) - 1,
            len(left.sequence_lengths),
            int(left.sequence_lengths.sum()),
        )
    else:
        logger.error("PARITY FAIL: %d token-sequence mismatches", mismatch_count)
    return matches


def main() -> None:
    """Load both outputs and return a nonzero status on any difference."""
    args = parse_args()
    if args.max_mismatches < 0:
        raise ValueError("--max-mismatches cannot be negative")
    datatrove_dataset = load_indexed_dataset(args.datatrove_prefix)
    megatron_dataset = load_indexed_dataset(args.megatron_prefix)
    if not compare(datatrove_dataset, megatron_dataset, args.max_mismatches):
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
