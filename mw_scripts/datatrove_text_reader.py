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

"""DataTrove readers that load only text and an optional ID field."""

import logging

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


GENERATED_ID_KEY = "__datatrove_generated_document_id__"
logger = logging.getLogger(__name__)


class TextParquetReader(BaseDiskReader):
    """Read only the columns needed for Megatron document tokenization.

    Args:
        data_folder: Local or fsspec-backed input directory.
        paths_file: File containing one input path per line.
        batch_size: PyArrow record-batch size.
        text_key: Parquet text column.
        id_key: Optional Parquet ID column. Source path and row number are used when omitted.
    """

    name = "Text-only Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        *,
        paths_file: DataFileLike,
        batch_size: int,
        text_key: str,
        id_key: str | None,
    ) -> None:
        """Initialize the text-only Parquet reader."""
        super().__init__(
            data_folder=data_folder,
            paths_file=paths_file,
            text_key=text_key,
            id_key=id_key or GENERATED_ID_KEY,
            add_file_path=False,
        )
        self.batch_size = batch_size
        self.columns = [text_key] if id_key is None else [text_key, id_key]

    def read_file(self, filepath: str) -> DocumentsPipeline:
        """Yield documents from a single Parquet file.

        Args:
            filepath: Path relative to ``data_folder``.

        Yields:
            DataTrove documents with source-stable IDs.
        """
        import pyarrow.parquet as pq

        with self.data_folder.open(filepath, "rb") as input_file:
            with pq.ParquetFile(input_file) as parquet_file:
                row_number = 0
                for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns):
                    documents = []
                    with self.track_time("batch"):
                        for row in batch.to_pylist():
                            document = self.get_document_from_dict(row, filepath, row_number)
                            row_number += 1
                            if document is not None:
                                documents.append(document)
                    yield from documents


class StackV3ParquetReader(BaseDiskReader):
    """Render Stack v3 repository rows into text while reading Parquet.

    Args:
        data_folder: Local or fsspec-backed input directory.
        paths_file: File containing one input path per line.
        batch_size: PyArrow record-batch size.
        dataset_name: Stable dataset component of the quality-score key.
        seed: Global augmentation seed. A stable per-row seed is derived from
            this value and the source filename/row index.
        max_chars: Maximum rendered characters per tokenized document.
    """

    name = "Stack v3 repository Parquet"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        *,
        paths_file: DataFileLike,
        batch_size: int,
        dataset_name: str,
        seed: int,
        max_chars: int,
    ) -> None:
        """Initialize the Stack v3 repository reader."""
        super().__init__(
            data_folder=data_folder,
            paths_file=paths_file,
            text_key="text",
            id_key=GENERATED_ID_KEY,
            add_file_path=False,
        )
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.seed = seed
        self.max_chars = max_chars
        self.columns = ["repo_path", "files"]

    def read_file(self, filepath: str) -> DocumentsPipeline:
        """Yield deterministic bounded documents from repository rows.

        Args:
            filepath: Path relative to ``data_folder``.

        Yields:
            DataTrove documents containing rendered repository text.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from tokenization.stack_v3_repo_to_text_27_variants import repo_to_text_chunks, source_row_seed

        with self.data_folder.open(filepath, "rb") as input_file:
            with pq.ParquetFile(input_file) as parquet_file:
                row_number = 0
                next_memory_release = 256
                for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns):
                    rows = batch.to_pylist()
                    with self.track_time("batch"):
                        for row in rows:
                            chunk_iterator = repo_to_text_chunks(
                                row,
                                max_chars=self.max_chars,
                                seed=source_row_seed(self.seed, filepath, row_number),
                            )
                            for chunk_index, rendered in enumerate(chunk_iterator):
                                metadata = {
                                    "dataset": self.dataset_name,
                                    "source_file": filepath,
                                    "source_row_index": row_number,
                                    "chunk_index": chunk_index,
                                }
                                document = self.get_document_from_dict(
                                    {"text": rendered, "metadata": metadata},
                                    filepath,
                                    f"{row_number}/{chunk_index}",
                                )
                                if document is not None:
                                    yield document
                            row_number += 1
                    if rows:
                        del row
                    del rows, batch
                    if row_number >= next_memory_release:
                        # Large nested repository rows leave reusable Arrow pages in the pool.
                        # Return them before they accumulate across thousands of rows.
                        pa.default_memory_pool().release_unused()
                        next_memory_release = row_number + 256


class TextJsonlReader(BaseDiskReader):
    """Read only text and an optional ID from line-delimited JSON files.

    Args:
        data_folder: Local or fsspec-backed input directory.
        paths_file: File containing one input path per line.
        batch_size: Unused compatibility argument shared with ``TextParquetReader``.
        text_key: JSON field containing document text.
        id_key: Optional JSON ID field. Source path and row number are used when omitted.
    """

    name = "Text-only JSONL"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        *,
        paths_file: DataFileLike,
        batch_size: int,
        text_key: str,
        id_key: str | None,
    ) -> None:
        """Initialize the text-only JSONL reader."""
        del batch_size
        super().__init__(
            data_folder=data_folder,
            paths_file=paths_file,
            text_key=text_key,
            id_key=id_key or GENERATED_ID_KEY,
            add_file_path=False,
        )
        self.source_id_key = id_key

    def read_file(self, filepath: str) -> DocumentsPipeline:
        """Yield documents from a plain or gzip-compressed JSONL file.

        Args:
            filepath: Path relative to ``data_folder``.

        Yields:
            DataTrove documents with source-stable IDs.
        """
        import orjson
        from orjson import JSONDecodeError

        with self.data_folder.open(filepath, "r", compression="infer") as input_file:
            for row_number, line in enumerate(input_file):
                try:
                    row = orjson.loads(line)
                except (EOFError, JSONDecodeError) as error:
                    logger.warning("Skipping malformed JSON in %s at row %d: %s", filepath, row_number, error)
                    continue
                if not isinstance(row, dict):
                    logger.warning("Skipping non-object JSON in %s at row %d", filepath, row_number)
                    continue
                selected = {self.text_key: row.get(self.text_key)}
                if self.source_id_key is not None:
                    selected[self.source_id_key] = row.get(self.source_id_key)
                document = self.get_document_from_dict(selected, filepath, row_number)
                if document is not None:
                    yield document
