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

"""
Stack v3 repository row -> bounded text serializer with 27 augmentation variants.

Axes:
1) Header: repo_name_only / flat_directory_manifest / compact_ascii_tree
2) Ordering: full_random / lexical / randomized_tree
3) Serialization: starcoder / directory_grouped / xml
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


HEADER_VARIANTS = (
    "repo_name_only",
    "flat_directory_manifest",
    "compact_ascii_tree",
)

ORDERING_VARIANTS = (
    "full_random",
    "lexical",
    "randomized_tree",
)

SERIALIZATION_VARIANTS = (
    "starcoder",
    "directory_grouped",
    "xml",
)


@dataclass(frozen=True)
class RepoTextConfig:
    """Probabilities and delimiters for repository serialization."""

    header_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    ordering_probs: tuple[float, float, float] = (0.15, 0.50, 0.35)
    serialization_probs: tuple[float, float, float] = (0.60, 0.25, 0.15)

    tree_max_depth: int = 3
    tree_max_entries: int = 400
    tree_include_files: bool = False

    repo_name_token: str = "<repo_name>"
    repo_directories_open: str = "<repo_directories>"
    repo_directories_close: str = "</repo_directories>"
    repo_tree_open: str = "<repo_tree>"
    repo_tree_close: str = "</repo_tree>"
    file_sep_token: str = "<file_sep>"
    directory_token: str = "<directory>"
    file_token: str = "<file>"

    def __post_init__(self) -> None:
        _validate_probs(self.header_probs, "header_probs")
        _validate_probs(self.ordering_probs, "ordering_probs")
        _validate_probs(self.serialization_probs, "serialization_probs")
        if self.tree_max_depth < 1:
            raise ValueError("tree_max_depth must be >= 1")
        if self.tree_max_entries < 1:
            raise ValueError("tree_max_entries must be >= 1")


def repo_to_text(
    row: Mapping[str, Any],
    *,
    seed: int | None = None,
    config: RepoTextConfig | None = None,
    header_variant: str | None = None,
    ordering_variant: str | None = None,
    serialization_variant: str | None = None,
) -> str:
    """Serialize one Stack v3 repository row into one training document."""
    prepared = _prepare_repo_text(
        row,
        seed=seed,
        config=config,
        header_variant=header_variant,
        ordering_variant=ordering_variant,
        serialization_variant=serialization_variant,
    )
    if prepared is None:
        return ""
    cfg, header, ordered, serialization = prepared
    body = serialize_body(ordered, serialization, cfg)
    return (header + "\n" + body).rstrip() + "\n"


def repo_to_text_chunks(
    row: Mapping[str, Any],
    *,
    max_chars: int,
    seed: int | None = None,
    config: RepoTextConfig | None = None,
    header_variant: str | None = None,
    ordering_variant: str | None = None,
    serialization_variant: str | None = None,
) -> Iterator[str]:
    """Serialize one repository into deterministic, bounded training documents.

    Repositories whose rendered form fits in ``max_chars`` retain the exact
    single-document representation returned by :func:`repo_to_text`. Larger
    repositories are split on file boundaries where possible. An individual
    oversized file is split near newline boundaries without dropping content.

    Args:
        row: Stack v3 source row.
        max_chars: Maximum number of Python characters in each output document.
        seed: Stable per-source-row augmentation seed.
        config: Optional serialization configuration.
        header_variant: Optional fixed header variant.
        ordering_variant: Optional fixed ordering variant.
        serialization_variant: Optional fixed serialization variant.

    Returns:
        Ordered rendered chunks. Empty repositories produce no chunks.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    prepared = _prepare_repo_text(
        row,
        seed=seed,
        config=config,
        header_variant=header_variant,
        ordering_variant=ordering_variant,
        serialization_variant=serialization_variant,
    )
    if prepared is None:
        return
    cfg, header, ordered, serialization = prepared

    complete_length = len(header) + 2 + _serialized_body_length(ordered, serialization, cfg)
    if complete_length <= max_chars:
        body = serialize_body(ordered, serialization, cfg)
        yield (header + "\n" + body).rstrip() + "\n"
        return

    body_budget = max_chars - len(header) - 2
    if body_budget <= 0:
        raise ValueError(f"Repository header is too large for max_chars={max_chars}")

    pending_blocks: list[str] = []
    pending_length = 0

    for file in ordered:
        prefix, content, suffix = _file_block(file, serialization, cfg)
        block_length = len(prefix) + len(content) + len(suffix)
        separator_length = 1 if pending_blocks else 0
        if block_length <= body_budget - pending_length - separator_length:
            pending_blocks.append(prefix + content + suffix)
            pending_length += separator_length + block_length
            continue

        if pending_blocks:
            yield _combine_chunk(header, "\n".join(pending_blocks), max_chars)
            pending_blocks = []
            pending_length = 0
        if block_length <= body_budget:
            pending_blocks.append(prefix + content + suffix)
            pending_length = block_length
            continue

        content_budget = body_budget - len(prefix) - len(suffix)
        if content_budget <= 0:
            raise ValueError(f"File wrapper for {file['path']!r} is too large for max_chars={max_chars}")
        for content_piece in _split_text(content, content_budget):
            yield _combine_chunk(header, prefix + content_piece + suffix, max_chars)

    if pending_blocks:
        yield _combine_chunk(header, "\n".join(pending_blocks), max_chars)


def _prepare_repo_text(
    row: Mapping[str, Any],
    *,
    seed: int | None,
    config: RepoTextConfig | None,
    header_variant: str | None,
    ordering_variant: str | None,
    serialization_variant: str | None,
) -> tuple[RepoTextConfig, str, list[dict[str, str]], str] | None:
    """Resolve deterministic renderer choices without materializing file contents."""
    cfg = config or RepoTextConfig()
    rng = random.Random(seed)

    repo_name = extract_repo_name(row)
    files = extract_files(row)
    if not files:
        return None

    h = _resolve_variant(header_variant, HEADER_VARIANTS, cfg.header_probs, rng)
    o = _resolve_variant(ordering_variant, ORDERING_VARIANTS, cfg.ordering_probs, rng)
    s = _resolve_variant(serialization_variant, SERIALIZATION_VARIANTS, cfg.serialization_probs, rng)

    ordered = order_files(files, o, rng)
    header = serialize_repo_header(repo_name, files, h, cfg)
    return cfg, header, ordered, s


def source_row_seed(global_seed: int, source_file: str, source_row_index: int) -> int:
    """Derive a stable per-record seed for deterministic resume behavior.

    Args:
        global_seed: User-selected seed shared by the complete preprocessing run.
        source_file: Source-root-relative Parquet path.
        source_row_index: Zero-based row index within the Parquet file.

    Returns:
        Unsigned 64-bit seed derived from the source document key.
    """
    key = f"{global_seed}\0{source_file}\0{source_row_index}".encode()
    digest = hashlib.blake2b(key, digest_size=8, person=b"stack-v3").digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def extract_repo_name(row: Mapping[str, Any]) -> str:
    """Extract the best available repository name from a source row."""
    for key in ("repo_path", "repo_name", "repository", "name"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "<unknown_repo>"


def extract_files(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Extract serializable file paths and contents from a source row."""
    out: list[dict[str, str]] = []
    for item in row.get("files") or []:
        if not isinstance(item, Mapping):
            continue
        path = item.get("file_path") or item.get("path") or item.get("filename") or ""
        content = item.get("content")
        if not isinstance(content, str):
            continue
        out.append({"path": _normalize_path(str(path)), "content": content})
    return out


def serialize_repo_header(
    repo_name: str,
    files: Sequence[Mapping[str, str]],
    variant: str,
    cfg: RepoTextConfig,
) -> str:
    """Serialize the selected repository-level header."""
    if variant == "repo_name_only":
        return f"{cfg.repo_name_token}{repo_name}"

    if variant == "flat_directory_manifest":
        dirs = collect_directories(
            [f["path"] for f in files],
            max_depth=cfg.tree_max_depth,
            max_entries=cfg.tree_max_entries,
        )
        return "\n".join(
            [
                f"{cfg.repo_name_token}{repo_name}",
                cfg.repo_directories_open,
                *dirs,
                cfg.repo_directories_close,
            ]
        )

    if variant == "compact_ascii_tree":
        tree = build_compact_repo_tree(
            [f["path"] for f in files],
            max_depth=cfg.tree_max_depth,
            max_entries=cfg.tree_max_entries,
            include_files=cfg.tree_include_files,
        )
        return "\n".join(
            [
                f"{cfg.repo_name_token}{repo_name}",
                cfg.repo_tree_open,
                tree,
                cfg.repo_tree_close,
            ]
        )

    raise ValueError(variant)


def collect_directories(paths: Sequence[str], *, max_depth: int, max_entries: int) -> list[str]:
    """Collect a bounded lexical directory manifest."""
    dirs: set[str] = set()
    for raw in paths:
        parts = [p for p in PurePosixPath(raw).parts if p not in ("", "/")]
        parents = parts[:-1]
        for depth in range(1, min(len(parents), max_depth) + 1):
            dirs.add("/".join(parents[:depth]) + "/")
    return sorted(dirs)[:max_entries]


def build_compact_repo_tree(
    paths: Sequence[str], *, max_depth: int = 3, max_entries: int = 400, include_files: bool = False
) -> str:
    """Build a bounded compact ASCII tree for repository paths."""
    tree: dict[str, Any] = {}

    for raw in sorted(set(paths)):
        parts = [p for p in PurePosixPath(raw).parts if p not in ("", "/")]
        if not include_files:
            parts = parts[:-1]
        if not parts:
            continue

        node = tree
        for depth, part in enumerate(parts, start=1):
            if depth > max_depth:
                break
            node = node.setdefault(part, {})

    lines = ["repo"]
    emitted = 0
    truncated = False

    def render(node: Mapping[str, Any], prefix: str = "") -> None:
        nonlocal emitted, truncated
        names = sorted(node)
        for idx, name in enumerate(names):
            if emitted >= max_entries:
                truncated = True
                return
            last = idx == len(names) - 1
            branch = "└── " if last else "├── "
            child = node[name]
            suffix = "/" if child or not include_files else ""
            lines.append(f"{prefix}{branch}{name}{suffix}")
            emitted += 1
            if child:
                render(child, prefix + ("    " if last else "│   "))
            if truncated:
                return

    render(tree)
    if truncated:
        lines.append("...")
    return "\n".join(lines)


def order_files(files: Sequence[Mapping[str, str]], variant: str, rng: random.Random) -> list[dict[str, str]]:
    """Order repository files with the selected augmentation."""
    items = [dict(f) for f in files]

    if variant == "full_random":
        rng.shuffle(items)
        return items
    if variant == "lexical":
        return sorted(items, key=lambda x: x["path"])
    if variant == "randomized_tree":
        return randomized_tree_traversal(items, rng)

    raise ValueError(variant)


def randomized_tree_traversal(files: Sequence[Mapping[str, str]], rng: random.Random) -> list[dict[str, str]]:
    """
    Preserve directory locality while randomizing:
    - sibling-directory order
    - file order within each directory

    Files directly under the same directory are always emitted as one
    contiguous block.
    """
    root: dict[str, Any] = {"files": [], "children": {}}

    for f in files:
        parts = [p for p in PurePosixPath(f["path"]).parts if p not in ("", "/")]
        dirs = parts[:-1]

        node: dict[str, Any] = root
        for d in dirs:
            node = node["children"].setdefault(d, {"files": [], "children": {}})
        node["files"].append(dict(f))

    output: list[dict[str, str]] = []

    def visit(node: Mapping[str, Any]) -> None:
        local_files = [dict(f) for f in node["files"]]
        rng.shuffle(local_files)

        children = list(node["children"])
        rng.shuffle(children)

        blocks = [("files", None)] + [("dir", name) for name in children]
        rng.shuffle(blocks)

        for kind, value in blocks:
            if kind == "files":
                output.extend(local_files)
            else:
                visit(node["children"][value])

    visit(root)
    return output


def serialize_body(files: Sequence[Mapping[str, str]], variant: str, cfg: RepoTextConfig) -> str:
    """Serialize ordered repository files with the selected format."""
    if variant == "starcoder":
        return "\n".join(f"{cfg.file_sep_token}{f['path']}\n{f['content']}" for f in files)

    if variant == "directory_grouped":
        parts = []
        current_dir = None
        for f in files:
            p = PurePosixPath(f["path"])
            directory = str(p.parent)
            if directory != current_dir:
                parts.append(f"{cfg.directory_token}{directory}")
                current_dir = directory
            parts.append(f"{cfg.file_token}{p.name}\n{f['content']}")
        return "\n".join(parts)

    if variant == "xml":
        return "\n".join(f'<file path="{_xml_escape_attr(f["path"])}">\n{f["content"]}\n</file>' for f in files)

    raise ValueError(variant)


def _file_block(
    file: Mapping[str, str],
    variant: str,
    cfg: RepoTextConfig,
) -> tuple[str, str, str]:
    """Return the wrapper prefix, content, and suffix for one self-contained file."""
    path = file["path"]
    content = file["content"]
    if variant == "starcoder":
        return f"{cfg.file_sep_token}{path}\n", content, ""
    if variant == "directory_grouped":
        parsed_path = PurePosixPath(path)
        prefix = f"{cfg.directory_token}{parsed_path.parent}\n{cfg.file_token}{parsed_path.name}\n"
        return prefix, content, ""
    if variant == "xml":
        return f'<file path="{_xml_escape_attr(path)}">\n', content, "\n</file>"
    raise ValueError(variant)


def _serialized_body_length(
    files: Sequence[Mapping[str, str]],
    variant: str,
    cfg: RepoTextConfig,
) -> int:
    """Return the exact serialized body length without copying file contents."""
    if not files:
        return 0
    if variant in ("starcoder", "xml"):
        block_lengths = [sum(map(len, _file_block(file, variant, cfg))) for file in files]
        return sum(block_lengths) + len(block_lengths) - 1
    if variant == "directory_grouped":
        length = 0
        part_count = 0
        current_dir = None
        for file in files:
            path = PurePosixPath(file["path"])
            directory = str(path.parent)
            if directory != current_dir:
                length += len(cfg.directory_token) + len(directory)
                part_count += 1
                current_dir = directory
            length += len(cfg.file_token) + len(path.name) + 1 + len(file["content"])
            part_count += 1
        return length + part_count - 1
    raise ValueError(variant)


def _split_text(text: str, max_chars: int) -> Iterator[str]:
    """Split text without loss, preferring the last newline in each bounded window."""
    if not text:
        yield ""
        return
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline >= start:
                end = newline + 1
        yield text[start:end]
        start = end


def _combine_chunk(header: str, body: str, max_chars: int) -> str:
    """Combine a repeated repository header and chunk body without stripping content."""
    rendered = header + "\n" + body
    if not rendered.endswith("\n"):
        rendered += "\n"
    if len(rendered) > max_chars:
        raise AssertionError(f"Rendered chunk has {len(rendered)} chars; limit is {max_chars}")
    return rendered


def all_variant_combinations() -> list[tuple[str, str, str]]:
    """Return all 27 header, ordering, and serialization combinations."""
    return [(h, o, s) for h in HEADER_VARIANTS for o in ORDERING_VARIANTS for s in SERIALIZATION_VARIANTS]


def variant_id(h: str, o: str, s: str) -> str:
    """Return a stable human-readable identifier for a variant combination."""
    return f"header={h}__ordering={o}__serialization={s}"


def _resolve_variant(
    requested: str | None, variants: Sequence[str], probs: Sequence[float], rng: random.Random
) -> str:
    if requested is not None:
        if requested not in variants:
            raise ValueError(f"{requested!r} not in {tuple(variants)}")
        return requested

    x = rng.random()
    cumulative = 0.0
    for v, p in zip(variants, probs):
        cumulative += p
        if x <= cumulative:
            return v
    return variants[-1]


def _validate_probs(probs: Sequence[float], name: str) -> None:
    if len(probs) != 3:
        raise ValueError(f"{name} must have exactly 3 entries")
    if any(p < 0 for p in probs):
        raise ValueError(f"{name} contains a negative probability")
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1.0")


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    return path or "<unknown_file>"


def _xml_escape_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    example = {
        "repo_path": "acme/demo",
        "files": [
            {"file_path": "src/model/attention.py", "content": "class Attention:\n    pass\n"},
            {"file_path": "src/model/mlp.py", "content": "class MLP:\n    pass\n"},
            {"file_path": "src/utils/io.py", "content": "def load():\n    pass\n"},
            {"file_path": "tests/model/test_attention.py", "content": "def test_attention():\n    pass\n"},
            {"file_path": "docs/api.md", "content": "# API\n"},
            {"file_path": "README.md", "content": "# Demo\n"},
        ],
    }

    combos = all_variant_combinations()
    assert len(combos) == 27

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Total variants: %d", len(combos))
    for i, (h, o, s) in enumerate(combos, 1):
        text = repo_to_text(
            example,
            seed=123,
            header_variant=h,
            ordering_variant=o,
            serialization_variant=s,
        )
        assert text
        logger.info("%02d: %s -> %d chars", i, variant_id(h, o, s), len(text))

    logger.info(
        "\nExample:\n\n%s",
        repo_to_text(
            example,
            seed=7,
            header_variant="compact_ascii_tree",
            ordering_variant="randomized_tree",
            serialization_variant="starcoder",
        ),
    )
