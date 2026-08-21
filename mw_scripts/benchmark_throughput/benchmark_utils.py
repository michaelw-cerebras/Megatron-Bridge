# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for GPT-OSS preprocessing throughput benchmarks."""

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO


INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_ITEM_SIZES = {
    1: 1,  # uint8
    2: 1,  # int8
    3: 2,  # int16
    4: 4,  # int32
    5: 8,  # int64
    6: 8,  # float64
    7: 4,  # float32
    8: 2,  # uint16
}


@dataclass(frozen=True)
class IndexedDatasetTotals:
    """Aggregate metadata from one or more Megatron indexed datasets."""

    documents: int
    sequences: int
    tokens: int
    output_bytes: int
    shards: int


def read_exact(input_file: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes from a binary file.

    Args:
        input_file: Open binary input stream.
        size: Number of bytes to read.

    Returns:
        Bytes read from the stream.

    Raises:
        ValueError: If the stream ends before ``size`` bytes are read.
    """
    value = input_file.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of index file; wanted {size} bytes, got {len(value)}")
    return value


def indexed_dataset_totals(index_paths: list[Path]) -> IndexedDatasetTotals:
    """Aggregate counts and validate binary sizes for indexed datasets.

    Args:
        index_paths: Paths to Megatron ``.idx`` files.

    Returns:
        Aggregate document, sequence, token, byte, and shard counts.

    Raises:
        ValueError: If an index is malformed or inconsistent with its binary file.
    """
    documents = 0
    sequences = 0
    tokens = 0
    output_bytes = 0
    for index_path in sorted(index_paths):
        binary_path = index_path.with_suffix(".bin")
        if not binary_path.is_file():
            raise FileNotFoundError(binary_path)
        with index_path.open("rb") as input_file:
            if read_exact(input_file, len(INDEX_HEADER)) != INDEX_HEADER:
                raise ValueError(f"Invalid indexed-dataset header in {index_path}")
            version = struct.unpack("<Q", read_exact(input_file, 8))[0]
            if version != 1:
                raise ValueError(f"Unsupported indexed-dataset version {version} in {index_path}")
            dtype_code = struct.unpack("<B", read_exact(input_file, 1))[0]
            if dtype_code not in DTYPE_ITEM_SIZES:
                raise ValueError(f"Unsupported dtype code {dtype_code} in {index_path}")
            sequence_count = struct.unpack("<Q", read_exact(input_file, 8))[0]
            document_index_count = struct.unpack("<Q", read_exact(input_file, 8))[0]
            lengths_bytes = read_exact(input_file, sequence_count * 4)
            sequence_token_count = sum(
                value[0] for value in struct.iter_unpack("<i", lengths_bytes)
            )
            read_exact(input_file, sequence_count * 8)
            read_exact(input_file, document_index_count * 8)
            if input_file.read(1):
                raise ValueError(f"Unexpected trailing bytes in {index_path}")

        expected_binary_bytes = sequence_token_count * DTYPE_ITEM_SIZES[dtype_code]
        if binary_path.stat().st_size != expected_binary_bytes:
            raise ValueError(
                f"{binary_path} has {binary_path.stat().st_size} bytes; expected {expected_binary_bytes}"
            )
        documents += max(0, document_index_count - 1)
        sequences += sequence_count
        tokens += sequence_token_count
        output_bytes += index_path.stat().st_size + binary_path.stat().st_size

    return IndexedDatasetTotals(
        documents=documents,
        sequences=sequences,
        tokens=tokens,
        output_bytes=output_bytes,
        shards=len(index_paths),
    )


def load_manifest(path: Path) -> dict[str, object]:
    """Load a benchmark manifest.

    Args:
        path: Manifest JSON path.

    Returns:
        Parsed manifest object.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If required fields are absent.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset",
        "dataset_revision",
        "num_shards",
        "parquet_dir",
        "tokenizer_dir",
        "tokenizer_revision",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    return manifest


def resolve_manifest_path(manifest_path: Path, value: object) -> Path:
    """Resolve a manifest-relative path.

    Args:
        manifest_path: Path to the manifest containing the value.
        value: Relative path value from the manifest.

    Returns:
        Absolute resolved filesystem path.
    """
    if not isinstance(value, str):
        raise TypeError(f"Expected a string path in manifest, found {type(value).__name__}")
    return (manifest_path.resolve().parent / value).resolve()


def directory_size(path: Path) -> int:
    """Return the total byte size of regular files below a directory.

    Args:
        path: Directory to inspect.

    Returns:
        Total file size in bytes.
    """
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_cpu_count() -> int:
    """Return CPUs available to the current process affinity mask.

    Returns:
        Number of CPUs available to the process.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def package_versions(package_names: list[str]) -> dict[str, str]:
    """Return installed versions for selected Python distributions.

    Args:
        package_names: Distribution names to query.

    Returns:
        Mapping from distribution name to installed version or ``not-installed``.
    """
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def environment_metadata(package_names: list[str]) -> dict[str, object]:
    """Collect reproducibility metadata for a benchmark run.

    Args:
        package_names: Python distributions whose versions should be recorded.

    Returns:
        Host, CPU, Python, platform, and package-version metadata.
    """
    return {
        "available_cpus": available_cpu_count(),
        "hostname": socket.gethostname(),
        "packages": package_versions(package_names),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically write a JSON file.

    Args:
        path: Destination JSON path.
        value: JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output_file:
        temporary_path = Path(output_file.name)
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)


def totals_as_dict(totals: IndexedDatasetTotals) -> dict[str, int]:
    """Convert indexed-dataset totals to a JSON-ready dictionary.

    Args:
        totals: Aggregate indexed-dataset metadata.

    Returns:
        Dictionary representation of the totals.
    """
    return asdict(totals)
