"""Tests for BM25Index."""

from pathlib import Path

from anchor.ingestion.chunker import Chunk
from anchor.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_preserves_underscores_and_digits() -> None:
    assert tokenize("my_var has value 42") == ["my_var", "has", "value", "42"]


def test_tokenize_strips_punctuation() -> None:
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_empty_index_returns_no_results(tmp_path: Path) -> None:
    index = BM25Index(persist_path=tmp_path / "bm25.pkl")
    assert index.query("anything") == []


def test_index_query_returns_relevant_chunks(tmp_path: Path) -> None:
    index = BM25Index(persist_path=tmp_path / "bm25.pkl")
    chunks = [
        Chunk(doc_id="d1", chunk_index=0, text="Python is a programming language", token_count=5),
        Chunk(doc_id="d1", chunk_index=1, text="JavaScript runs in browsers", token_count=4),
        Chunk(doc_id="d1", chunk_index=2, text="Python supports async and await", token_count=5),
    ]
    index.add(chunks)

    results = index.query("python async", top_k=2)
    assert len(results) >= 1
    # The chunk specifically about Python+async should rank above the JS one.
    top_chunk_ids = [r.chunk_id for r in results]
    assert "d1::chunk-0002" in top_chunk_ids


def test_index_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    a = BM25Index(persist_path=path)
    a.add([Chunk(doc_id="d1", chunk_index=0, text="hello world", token_count=2)])
    a.save()

    b = BM25Index(persist_path=path)
    results = b.query("hello", top_k=1)
    assert len(results) == 1
    assert results[0].text == "hello world"


def test_re_adding_same_chunk_id_replaces_not_duplicates(tmp_path: Path) -> None:
    index = BM25Index(persist_path=tmp_path / "bm25.pkl")
    chunk = Chunk(doc_id="d1", chunk_index=0, text="original text", token_count=2)
    index.add([chunk])

    updated = Chunk(doc_id="d1", chunk_index=0, text="updated text", token_count=2)
    index.add([updated])

    results = index.query("updated", top_k=5)
    assert len(results) == 1
    assert "updated" in results[0].text
