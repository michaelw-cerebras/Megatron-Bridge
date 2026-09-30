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

"""Flatten zero-copy subset views and remove superseded merged metadata."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path


LOGGER = logging.getLogger(__name__)
HIERARCHY_MANIFEST = "hierarchy_manifest.json"
SHARED_BIN_DIRECTORY = "_shared_bin"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move <dataset>/subsets/<subset> views directly below <dataset> and remove merged metadata."
    )
    parser.add_argument("--tokenized-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to finalize; repeat to select several. Defaults to every dataset with subset views.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without modifying the filesystem.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _replace_symlink(*, link_path: Path, target_path: Path) -> None:
    relative_target = Path(os.path.relpath(target_path, start=link_path.parent))
    if link_path.is_symlink() and Path(os.readlink(link_path)) == relative_target:
        return
    if link_path.exists() and not link_path.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink binary: {link_path}")
    temporary_link = link_path.parent / f".{link_path.name}.{os.getpid()}.tmp"
    try:
        temporary_link.symlink_to(relative_target)
        os.replace(temporary_link, link_path)
    finally:
        temporary_link.unlink(missing_ok=True)


def _selected_datasets(tokenized_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(set(requested))
    return sorted(
        path.name
        for path in tokenized_root.iterdir()
        if path.is_dir()
        and ((path / "subsets" / "manifest.json").is_file() or (path / HIERARCHY_MANIFEST).is_file())
    )


def _validate_complete_layout(dataset_root: Path) -> tuple[int, int]:
    hierarchy_path = dataset_root / HIERARCHY_MANIFEST
    hierarchy = _read_json(hierarchy_path)
    raw_subsets = hierarchy.get("subsets")
    if not isinstance(raw_subsets, dict) or not raw_subsets:
        raise ValueError(f"Missing subsets in {hierarchy_path}")

    documents = 0
    tokens = 0
    for subset, raw_metadata in raw_subsets.items():
        if not isinstance(subset, str) or not isinstance(raw_metadata, dict):
            raise ValueError(f"Malformed subset entry in {hierarchy_path}: {subset!r}")
        subset_root = dataset_root / subset
        manifest_path = subset_root / "manifest.json"
        prefixes_path = subset_root / "data_prefixes.txt"
        if not manifest_path.is_file() or not prefixes_path.is_file():
            raise FileNotFoundError(f"Incomplete subset output: {subset_root}")
        manifest = _read_json(manifest_path)
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError(f"Missing shards in {manifest_path}")
        prefixes = [line for line in prefixes_path.read_text(encoding="utf-8").splitlines() if line]
        expected_prefixes = []
        for shard in raw_shards:
            if not isinstance(shard, dict) or not isinstance(shard.get("prefix"), str):
                raise ValueError(f"Malformed shard in {manifest_path}")
            prefix = str(shard["prefix"])
            expected_prefixes.append(prefix)
            index_path = subset_root / f"{prefix}.idx"
            binary_path = subset_root / f"{prefix}.bin"
            if not index_path.is_file() or not binary_path.is_symlink() or not binary_path.is_file():
                raise FileNotFoundError(f"Invalid zero-copy prefix: {subset_root / prefix}")
        if prefixes != expected_prefixes:
            raise ValueError(f"Prefix list does not match manifest shards: {subset_root}")
        documents += int(manifest["documents"])
        tokens += int(manifest["tokens"])

    if documents != int(hierarchy["documents"]) or tokens != int(hierarchy["tokens"]):
        raise ValueError(f"Hierarchy totals do not reconcile in {hierarchy_path}")
    return documents, tokens


def _build_hierarchy_manifest(
    *,
    dataset_root: Path,
    subset_manifest: dict[str, object],
    parent_manifest: dict[str, object],
) -> dict[str, object]:
    hierarchy = dict(subset_manifest)
    hierarchy["layout"] = "flat-subset-directories"
    hierarchy["shared_bin_directory"] = SHARED_BIN_DIRECTORY
    hierarchy["tokenizer"] = parent_manifest.get("tokenizer")
    hierarchy["source_dataset"] = parent_manifest.get("dataset")
    hierarchy["parent_dataset"] = dataset_root.name
    return hierarchy


def _validate_before_migration(dataset_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    subset_manifest_path = dataset_root / "subsets" / "manifest.json"
    parent_manifest_path = dataset_root / "manifest.json"
    parent_result_path = dataset_root / "result.json"
    if not subset_manifest_path.is_file():
        raise FileNotFoundError(subset_manifest_path)
    if not parent_manifest_path.is_file() or not parent_result_path.is_file():
        raise FileNotFoundError(f"Missing parent tokenization metadata under {dataset_root}")

    subset_manifest = _read_json(subset_manifest_path)
    parent_manifest = _read_json(parent_manifest_path)
    parent_result = _read_json(parent_result_path)
    raw_result = parent_result.get("result")
    raw_subsets = subset_manifest.get("subsets")
    if not isinstance(raw_result, dict) or not isinstance(raw_subsets, dict) or not raw_subsets:
        raise ValueError(f"Malformed hierarchy or parent result under {dataset_root}")
    if int(subset_manifest["documents"]) != int(raw_result["documents"]):
        raise ValueError(f"Document total mismatch under {dataset_root}")
    if int(subset_manifest["tokens"]) != int(raw_result["tokens"]):
        raise ValueError(f"Token total mismatch under {dataset_root}")

    reserved_names = {"data", "logs", "metadata", "subsets", SHARED_BIN_DIRECTORY}
    conflicts = sorted(set(raw_subsets).intersection(reserved_names))
    if conflicts:
        raise ValueError(f"Subset names conflict with reserved output directories under {dataset_root}: {conflicts}")
    for subset in raw_subsets:
        source_subset = dataset_root / "subsets" / subset
        destination_subset = dataset_root / subset
        if not source_subset.is_dir() and not destination_subset.is_dir():
            raise FileNotFoundError(f"Missing subset directory for {subset!r} under {dataset_root}")
        if source_subset.is_dir() and destination_subset.exists():
            raise FileExistsError(f"Both nested and flat subset directories exist for {subset!r} under {dataset_root}")
    return subset_manifest, parent_manifest


def _move_parent_bins(dataset_root: Path, *, dry_run: bool) -> int:
    parent_data_dir = dataset_root / "data"
    shared_bin_dir = dataset_root / SHARED_BIN_DIRECTORY
    parent_bins = sorted(parent_data_dir.glob("*.bin")) if parent_data_dir.is_dir() else []
    existing_bins = sorted(shared_bin_dir.glob("*.bin")) if shared_bin_dir.is_dir() else []
    if not parent_bins and not existing_bins:
        raise FileNotFoundError(f"No parent binary shards found under {dataset_root}")
    if dry_run:
        return len(parent_bins) + len(existing_bins)
    shared_bin_dir.mkdir(exist_ok=True)
    for source_bin in parent_bins:
        destination_bin = shared_bin_dir / source_bin.name
        if destination_bin.exists():
            raise FileExistsError(destination_bin)
        os.replace(source_bin, destination_bin)
    return len(list(shared_bin_dir.glob("*.bin")))


def _move_subsets_and_relink(
    *,
    dataset_root: Path,
    subset_manifest: dict[str, object],
    dry_run: bool,
) -> tuple[int, int]:
    raw_subsets = subset_manifest["subsets"]
    if not isinstance(raw_subsets, dict):
        raise TypeError("subsets must be an object")
    subset_count = 0
    prefix_count = 0
    for subset in sorted(raw_subsets):
        source_subset = dataset_root / "subsets" / subset
        destination_subset = dataset_root / subset
        if dry_run:
            subset_root = source_subset if source_subset.is_dir() else destination_subset
        else:
            if source_subset.is_dir():
                os.replace(source_subset, destination_subset)
            subset_root = destination_subset
        manifest = _read_json(subset_root / "manifest.json")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list):
            raise ValueError(f"Missing shards in {subset_root / 'manifest.json'}")
        for shard in raw_shards:
            if not isinstance(shard, dict) or not isinstance(shard.get("prefix"), str):
                raise ValueError(f"Malformed shard in {subset_root / 'manifest.json'}")
            prefix = Path(str(shard["prefix"]))
            link_path = subset_root / prefix.with_suffix(".bin")
            target_path = dataset_root / SHARED_BIN_DIRECTORY / link_path.name
            if not dry_run:
                if not target_path.is_file():
                    raise FileNotFoundError(target_path)
                _replace_symlink(link_path=link_path, target_path=target_path)
            prefix_count += 1
        subset_count += 1
    return subset_count, prefix_count


def _remove_superseded_metadata(dataset_root: Path) -> tuple[int, int]:
    parent_data_dir = dataset_root / "data"
    removed_indexes = 0
    removed_index_bytes = 0
    for index_path in parent_data_dir.glob("*.idx"):
        removed_index_bytes += index_path.stat().st_size
        index_path.unlink()
        removed_indexes += 1
    unexpected_data_files = list(parent_data_dir.iterdir()) if parent_data_dir.is_dir() else []
    if unexpected_data_files:
        raise RuntimeError(f"Unexpected files remain in superseded data directory: {unexpected_data_files[:5]}")
    if parent_data_dir.is_dir():
        parent_data_dir.rmdir()

    for directory_name in ("logs", "metadata"):
        directory = dataset_root / directory_name
        if directory.is_dir():
            shutil.rmtree(directory)
    for filename in ("data_prefixes.txt", "manifest.json", "result.json"):
        (dataset_root / filename).unlink(missing_ok=True)

    nested_root = dataset_root / "subsets"
    (nested_root / "manifest.json").unlink(missing_ok=True)
    if nested_root.is_dir():
        unexpected_nested_files = list(nested_root.iterdir())
        if unexpected_nested_files:
            raise RuntimeError(f"Unexpected files remain in nested subset directory: {unexpected_nested_files[:5]}")
        nested_root.rmdir()
    return removed_indexes, removed_index_bytes


def _finalize_dataset(dataset_root: Path, *, dry_run: bool) -> tuple[int, int, int, int]:
    hierarchy_path = dataset_root / HIERARCHY_MANIFEST
    nested_manifest_path = dataset_root / "subsets" / "manifest.json"
    if hierarchy_path.is_file() and not nested_manifest_path.exists():
        documents, tokens = _validate_complete_layout(dataset_root)
        LOGGER.info("SKIP %s: flat hierarchy is already complete (%d documents, %d tokens)", dataset_root.name, documents, tokens)
        hierarchy = _read_json(hierarchy_path)
        raw_subsets = hierarchy["subsets"]
        if not isinstance(raw_subsets, dict):
            raise TypeError("subsets must be an object")
        return len(raw_subsets), 0, 0, 0

    subset_manifest, parent_manifest = _validate_before_migration(dataset_root)
    bin_count = _move_parent_bins(dataset_root, dry_run=dry_run)
    subset_count, prefix_count = _move_subsets_and_relink(
        dataset_root=dataset_root,
        subset_manifest=subset_manifest,
        dry_run=dry_run,
    )
    if dry_run:
        index_paths = list((dataset_root / "data").glob("*.idx"))
        return subset_count, prefix_count, len(index_paths), sum(path.stat().st_size for path in index_paths)

    hierarchy = _build_hierarchy_manifest(
        dataset_root=dataset_root,
        subset_manifest=subset_manifest,
        parent_manifest=parent_manifest,
    )
    _atomic_write_json(hierarchy_path, hierarchy)
    removed_indexes, removed_index_bytes = _remove_superseded_metadata(dataset_root)
    _validate_complete_layout(dataset_root)
    LOGGER.info(
        "DONE %s: %d subsets, %d prefixes, %d shared bins; removed %d merged indexes (%.2f GiB)",
        dataset_root.name,
        subset_count,
        prefix_count,
        bin_count,
        removed_indexes,
        removed_index_bytes / 2**30,
    )
    return subset_count, prefix_count, removed_indexes, removed_index_bytes


def main() -> None:
    """Flatten all selected post-tokenization subset views."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tokenized_root = args.tokenized_root.expanduser().resolve()
    if not tokenized_root.is_dir():
        raise NotADirectoryError(tokenized_root)

    datasets = _selected_datasets(tokenized_root, args.datasets)
    total_subsets = 0
    total_prefixes = 0
    total_removed_indexes = 0
    total_removed_bytes = 0
    for dataset in datasets:
        dataset_root = tokenized_root / dataset
        subsets, prefixes, removed_indexes, removed_bytes = _finalize_dataset(
            dataset_root,
            dry_run=args.dry_run,
        )
        total_subsets += subsets
        total_prefixes += prefixes
        total_removed_indexes += removed_indexes
        total_removed_bytes += removed_bytes

    action = "DRY RUN" if args.dry_run else "COMPLETE"
    LOGGER.info(
        "%s: %d datasets, %d subsets, %d prefixes; merged indexes removable/removed: %d (%.2f GiB)",
        action,
        len(datasets),
        total_subsets,
        total_prefixes,
        total_removed_indexes,
        total_removed_bytes / 2**30,
    )


if __name__ == "__main__":
    main()
