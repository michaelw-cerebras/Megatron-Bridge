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

"""Megatron document tokenizer with source-key mapping sidecars."""

import os
from pathlib import Path

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer, MegatronTokenizedFile
from datatrove.utils.batching import batched


MAPPING_COLUMNS = (
    "dataset",
    "source_file",
    "source_row_index",
    "chunk_index",
    "target_document_index",
)


class _ParquetMappingWriter:
    """Write an atomic, buffered source-to-target document mapping."""

    def __init__(self, output_path: Path, *, row_group_size: int = 100_000) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.output_path = output_path
        self.temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        self.temporary_path.unlink(missing_ok=True)
        self.row_group_size = row_group_size
        self.schema = pa.schema(
            [
                pa.field("dataset", pa.string(), nullable=False),
                pa.field("source_file", pa.string(), nullable=False),
                pa.field("source_row_index", pa.int64(), nullable=False),
                pa.field("chunk_index", pa.int32(), nullable=False),
                pa.field("target_document_index", pa.int64(), nullable=False),
            ]
        )
        self.writer = pq.ParquetWriter(
            self.temporary_path,
            self.schema,
            compression="zstd",
            use_dictionary=["dataset", "source_file"],
            write_statistics=True,
        )
        self.columns: dict[str, list[object]] = {name: [] for name in MAPPING_COLUMNS}

    def append(self, document: Document, target_document_index: int) -> None:
        """Append one document mapping, validating its source identity."""
        metadata = document.metadata
        missing = [name for name in MAPPING_COLUMNS[:4] if name not in metadata]
        if missing:
            raise ValueError(f"Document {document.id!r} is missing mapping metadata: {missing}")
        self.columns["dataset"].append(str(metadata["dataset"]))
        self.columns["source_file"].append(str(metadata["source_file"]))
        self.columns["source_row_index"].append(int(metadata["source_row_index"]))
        self.columns["chunk_index"].append(int(metadata["chunk_index"]))
        self.columns["target_document_index"].append(target_document_index)
        if len(self.columns["dataset"]) >= self.row_group_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered mapping rows as one Parquet row group."""
        import pyarrow as pa

        if not self.columns["dataset"]:
            return
        table = pa.Table.from_pydict(self.columns, schema=self.schema)
        self.writer.write_table(table)
        self.columns = {name: [] for name in MAPPING_COLUMNS}

    def close(self) -> None:
        """Finish the sidecar and atomically expose it to validators."""
        self.flush()
        self.writer.close()
        os.replace(self.temporary_path, self.output_path)

    def abort(self) -> None:
        """Close and remove an incomplete sidecar after a failed task."""
        self.writer.close()
        self.temporary_path.unlink(missing_ok=True)


class MappedMegatronDocumentTokenizer(MegatronDocumentTokenizer):
    """Write standard Megatron shards plus an aligned source mapping Parquet.

    The ``target_document_index`` column is zero-based within the paired
    ``.bin/.idx`` shard. Mapping row order exactly matches Megatron document
    order; the binary and index formats themselves are unchanged.
    """

    def __init__(self, *, mapping_output_folder: Path, **kwargs: object) -> None:
        """Initialize the standard tokenizer and local mapping destination."""
        super().__init__(**kwargs)
        self.mapping_output_folder = mapping_output_folder

    def write_tokens(self, data: DocumentsPipeline, filename: str) -> MegatronTokenizedFile:
        """Tokenize documents and write their source keys in matching order."""
        from tokenizers import Encoding

        tokenized_file = MegatronTokenizedFile(
            self.output_folder,
            filename,
            upload_block_size=self.upload_block_size,
            token_size=self.token_size,
        )
        mapping_writer = _ParquetMappingWriter(self.mapping_output_folder / f"{filename}.mapping.parquet")
        target_document_index = 0
        try:
            for batch in batched(data, self.batch_size):
                with self.track_time(unit="batch"):
                    encoded_batch: list[Encoding] = self.tokenizer.encode_batch([document.text for document in batch])
                    for document, encoded in zip(batch, encoded_batch, strict=True):
                        tokens = encoded.ids
                        tokenized_file.write(tokens)
                        mapping_writer.append(document, target_document_index)
                        target_document_index += 1
                        self.stat_update("tokens", value=len(tokens))
            tokenized_file.close()
            mapping_writer.close()
        except BaseException:
            tokenized_file.bin_file.close()
            mapping_writer.abort()
            raise
        return tokenized_file
