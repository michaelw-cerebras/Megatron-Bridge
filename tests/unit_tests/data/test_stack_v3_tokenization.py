from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from mw_scripts.datatrove_text_reader import StackV3ParquetReader
from mw_scripts.mapped_megatron_tokenizer import MAPPING_COLUMNS, MappedMegatronDocumentTokenizer
from mw_scripts.process_datatrove import validate_outputs
from tokenization.stack_v3_repo_to_text_27_variants import (
    all_variant_combinations,
    repo_to_text,
    repo_to_text_chunks,
)


def test_small_repositories_retain_all_27_serializations() -> None:
    row = {
        "repo_path": "acme/example",
        "files": [
            {"file_path": "src/main.py", "content": "print('hello')\n"},
            {"file_path": "README.md", "content": "# Example\n"},
        ],
    }

    for header, ordering, serialization in all_variant_combinations():
        expected = repo_to_text(
            row,
            seed=7,
            header_variant=header,
            ordering_variant=ordering,
            serialization_variant=serialization,
        )
        actual = list(
            repo_to_text_chunks(
                row,
                max_chars=10_000,
                seed=7,
                header_variant=header,
                ordering_variant=ordering,
                serialization_variant=serialization,
            )
        )
        assert actual == [expected]


def test_large_repository_chunks_are_bounded_and_deterministic() -> None:
    row = {
        "repo_path": "acme/large",
        "files": [{"file_path": "large.py", "content": "a = 1\n" * 1_000}],
    }

    first = list(repo_to_text_chunks(row, max_chars=256, seed=17))
    second = list(repo_to_text_chunks(row, max_chars=256, seed=17))

    assert first == second
    assert len(first) > 1
    assert all(len(chunk) <= 256 for chunk in first)


def _write_test_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "<eos>": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_megatron_mapping_matches_document_order(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    source_path = source_dir / "part.parquet"
    rows = [
        {
            "repo_path": "acme/small",
            "files": [{"file_path": "small.py", "content": "small\n"}],
        },
        {
            "repo_path": "acme/large",
            "files": [{"file_path": "large.py", "content": "large line\n" * 200}],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), source_path)
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("part.parquet\n", encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_test_tokenizer(tokenizer_path)

    reader = StackV3ParquetReader(
        data_folder=str(source_dir),
        paths_file=str(paths_file),
        batch_size=1,
        dataset_name="stack-v3-train",
        seed=1234,
        max_chars=256,
    )
    tokenizer = MappedMegatronDocumentTokenizer(
        mapping_output_folder=output_dir,
        output_folder=str(output_dir),
        save_filename="dataset",
        tokenizer_name_or_path=str(tokenizer_path),
        eos_token="<eos>",
        batch_size=1,
    )
    tokenizer.write_tokens(reader.read_file("part.parquet"), "dataset_00000_tokens")
    tokenizer.write_tokens(reader.read_file("part.parquet"), "dataset_00000_tokens")

    totals = validate_outputs(
        data_dir=output_dir,
        save_filename="dataset",
        tasks=1,
        require_mapping=True,
    )
    mapping = pq.read_table(output_dir / "dataset_00000_tokens.mapping.parquet").to_pydict()

    assert set(mapping) == set(MAPPING_COLUMNS)
    assert totals.documents == totals.mapping_rows == len(mapping["dataset"])
    assert mapping["target_document_index"] == list(range(totals.documents))
    assert mapping["dataset"] == ["stack-v3-train"] * totals.documents
    assert mapping["source_file"] == ["part.parquet"] * totals.documents
    assert mapping["source_row_index"][0] == 0
    assert mapping["source_row_index"][1:] == [1] * (totals.documents - 1)
    assert mapping["chunk_index"][0] == 0
    assert mapping["chunk_index"][1:] == list(range(totals.documents - 1))
