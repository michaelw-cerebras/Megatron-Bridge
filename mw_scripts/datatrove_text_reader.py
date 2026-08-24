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

"""A DataTrove Parquet reader that loads only text and an optional ID column."""

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


GENERATED_ID_KEY = "__datatrove_generated_document_id__"


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
