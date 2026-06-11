"""Tests for TokenChunker."""

import pytest

from anchor.ingestion.chunker import TokenChunker


def test_chunk_short_text_produces_single_chunk() -> None:
    chunker = TokenChunker(chunk_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk("doc1", "Hello world. This is a short document.")

    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count > 0


def test_chunk_long_text_produces_multiple_chunks() -> None:
    chunker = TokenChunker(chunk_tokens=20, overlap_tokens=4)
    long_text = " ".join(f"sentence{i}" for i in range(200))
    chunks = chunker.chunk("doc1", long_text)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.doc_id == "doc1"


def test_chunk_ids_are_unique_and_stable() -> None:
    chunker = TokenChunker(chunk_tokens=20, overlap_tokens=4)
    text = " ".join(f"token{i}" for i in range(200))
    chunks_a = chunker.chunk("doc1", text)
    chunks_b = chunker.chunk("doc1", text)

    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]

    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)


def test_chunk_id_format_is_globally_unique_across_docs() -> None:
    chunker = TokenChunker(chunk_tokens=20, overlap_tokens=4)
    text = " ".join(f"token{i}" for i in range(50))
    chunks_doc1 = chunker.chunk("doc1", text)
    chunks_doc2 = chunker.chunk("doc2", text)

    ids = {c.chunk_id for c in chunks_doc1} | {c.chunk_id for c in chunks_doc2}
    assert len(ids) == len(chunks_doc1) + len(chunks_doc2)


def test_empty_text_yields_no_chunks() -> None:
    chunker = TokenChunker(chunk_tokens=20, overlap_tokens=4)
    assert chunker.chunk("doc1", "") == []


def test_invalid_chunk_tokens_raises() -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_tokens=0)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        TokenChunker(chunk_tokens=10, overlap_tokens=-1)


def test_overlap_produces_overlapping_content() -> None:
    """With overlap, consecutive chunks should share some tokens."""
    chunker = TokenChunker(chunk_tokens=20, overlap_tokens=8)
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunker.chunk("doc1", text)

    assert len(chunks) >= 2
    # The last few words of chunk 0 should appear at the start of chunk 1.
    first_tail = chunks[0].text.split()[-3:]
    second_head = chunks[1].text.split()[:10]
    overlap = set(first_tail) & set(second_head)
    assert overlap, "Expected token overlap between consecutive chunks"
