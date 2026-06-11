"""Hybrid retrieval: dense + BM25 fused via Reciprocal Rank Fusion (RRF).

Why RRF over weighted-sum: RRF only depends on rank position, not on
absolute score magnitudes. Dense (cosine) and BM25 scores live on
different scales — weighted sums are sensitive to that calibration. RRF
sidesteps the problem entirely.

  RRF_score(d) = sum over each ranker r of:  1 / (k + rank_r(d))

where rank_r(d) is the 1-indexed rank of doc d in ranker r's results.
k=60 is the value from the original RRF paper (Cormack et al., 2009).
"""

from collections import defaultdict
from dataclasses import dataclass

from anchor.config import get_settings
from anchor.ingestion.embedder import Embedder
from anchor.logging_config import get_logger
from anchor.retrieval.bm25 import BM25Index
from anchor.retrieval.vector_store import RetrievedChunk, VectorStore

log = get_logger(__name__)

RRF_K = 60


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieval response: the chunks, plus diagnostics for debugging."""

    chunks: list[RetrievedChunk]
    dense_count: int
    bm25_count: int


class HybridRetriever:
    """Hybrid retriever combining dense (vector) and sparse (BM25) results via RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25 or BM25Index()
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Run hybrid retrieval and return the top-k fused results."""
        settings = get_settings()
        k = top_k or settings.rerank_top_k
        retrieval_k = settings.retrieval_top_k

        # Dense: embed once, query vector store.
        query_embedding = self.embedder.embed([query])[0]
        dense_results = self.vector_store.query(embedding=query_embedding, top_k=retrieval_k)

        # Sparse: BM25 over the same query string.
        bm25_results = self.bm25.query(text=query, top_k=retrieval_k)

        fused = self._reciprocal_rank_fusion([dense_results, bm25_results])
        top = fused[:k]

        log.debug(
            "hybrid_retrieval",
            query_len=len(query),
            dense=len(dense_results),
            bm25=len(bm25_results),
            fused=len(fused),
            returned=len(top),
        )

        return RetrievalResult(
            chunks=top,
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[RetrievedChunk]],
    ) -> list[RetrievedChunk]:
        """Fuse multiple ranked lists into a single ordering via RRF."""
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_by_id: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank)
                # Keep the highest-quality copy (dense usually wins; first
                # writer with non-zero original score wins on ties).
                if chunk.chunk_id not in chunk_by_id:
                    chunk_by_id[chunk.chunk_id] = chunk

        # Build the final ordered list with the fused score replacing the
        # original ranker-specific score.
        fused: list[RetrievedChunk] = []
        for chunk_id, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            base = chunk_by_id[chunk_id]
            fused.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    doc_id=base.doc_id,
                    chunk_index=base.chunk_index,
                    text=base.text,
                    score=round(fused_score, 6),
                    source_path=base.source_path,
                )
            )

        return fused
