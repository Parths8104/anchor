"""Tests for HybridRetriever RRF fusion logic."""

from anchor.retrieval.hybrid import RRF_K, HybridRetriever
from anchor.retrieval.vector_store import RetrievedChunk


def _chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        chunk_index=int(chunk_id.split("-")[-1]),
        text=f"text for {chunk_id}",
        score=score,
    )


def test_rrf_empty_lists() -> None:
    assert HybridRetriever._reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_order() -> None:
    dense = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
    fused = HybridRetriever._reciprocal_rank_fusion([dense])
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_rrf_combines_two_lists() -> None:
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    bm25 = [_chunk("c"), _chunk("a"), _chunk("d")]
    fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])

    fused_ids = [c.chunk_id for c in fused]
    assert set(fused_ids) == {"a", "b", "c", "d"}

    # 'a' is rank-1 in dense AND rank-2 in bm25 — should beat 'c' which
    # is rank-3 in dense and rank-1 in bm25 (similar but slightly less).
    scores = {c.chunk_id: c.score for c in fused}
    expected_a = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
    expected_c = 1 / (RRF_K + 3) + 1 / (RRF_K + 1)
    assert scores["a"] == round(expected_a, 6)
    assert scores["c"] == round(expected_c, 6)


def test_rrf_dedupes_chunks() -> None:
    dense = [_chunk("a"), _chunk("b")]
    bm25 = [_chunk("a"), _chunk("c")]
    fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])

    ids = [c.chunk_id for c in fused]
    assert ids.count("a") == 1
    assert set(ids) == {"a", "b", "c"}


def test_rrf_chunk_appearing_in_both_outranks_chunk_in_one() -> None:
    """A chunk in both lists should score higher than a chunk in only one,
    even if its individual ranks are mediocre."""
    dense = [_chunk("a"), _chunk("b"), _chunk("shared")]
    bm25 = [_chunk("c"), _chunk("d"), _chunk("shared")]
    fused = HybridRetriever._reciprocal_rank_fusion([dense, bm25])

    ids = [c.chunk_id for c in fused]
    # 'shared' is rank 3 in both lists. Its RRF score = 2 * 1/(K+3).
    # Compare to 'a' which is rank 1 in dense only: 1/(K+1).
    # 1/(K+1) ≈ 0.0164, 2/(K+3) ≈ 0.0317 — shared should win.
    assert ids.index("shared") < ids.index("a")
