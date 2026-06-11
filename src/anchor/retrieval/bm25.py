"""BM25 sparse retrieval, persisted to disk.

Why BM25 alongside dense retrieval: dense embeddings capture semantic
similarity but miss exact keyword matches (acronyms, code identifiers,
proper nouns). BM25 catches those. Fusing both gives noticeably better
recall on technical documentation than either alone — see ADR-0001.
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from anchor.config import get_settings
from anchor.ingestion.chunker import Chunk
from anchor.logging_config import get_logger
from anchor.retrieval.vector_store import RetrievedChunk

log = get_logger(__name__)

# Simple whitespace + lowercase tokenizer. For technical text this is
# usually preferable to heavy NLP tokenization that splits on punctuation
# inside identifiers (e.g. snake_case, function.name).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index with disk persistence."""

    def __init__(self, persist_path: Path | None = None) -> None:
        settings = get_settings()
        self.persist_path = persist_path or settings.bm25_index_path
        self._corpus: list[list[str]] = []
        self._chunk_meta: list[dict[str, object]] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def add(self, chunks: list[Chunk]) -> None:
        """Append chunks to the BM25 corpus. Re-fits the index in place."""
        if not chunks:
            return

        # Replace any existing entries for these chunks (idempotency).
        chunk_ids = {c.chunk_id for c in chunks}
        keep_indices = [
            i for i, meta in enumerate(self._chunk_meta) if meta["chunk_id"] not in chunk_ids
        ]
        self._corpus = [self._corpus[i] for i in keep_indices]
        self._chunk_meta = [self._chunk_meta[i] for i in keep_indices]

        for chunk in chunks:
            self._corpus.append(tokenize(chunk.text))
            self._chunk_meta.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                }
            )

        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def query(self, text: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return top-k chunks by BM25 score for the query text."""
        if self._bm25 is None or not self._corpus:
            return []

        tokens = tokenize(text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # argsort descending without numpy dependency leak in interface
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        # Filter out zero-score hits (no token overlap).
        top = [(idx, score) for idx, score in top if score > 0]
        if not top:
            return []

        # Normalize scores to [0, 1] using the max in the result set so they
        # play well with cosine similarities during fusion.
        max_score = top[0][1]

        out: list[RetrievedChunk] = []
        for idx, score in top:
            meta = self._chunk_meta[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=str(meta["chunk_id"]),
                    doc_id=str(meta["doc_id"]),
                    chunk_index=int(meta["chunk_index"]),  # type: ignore[arg-type]
                    text=str(meta["text"]),
                    score=float(score) / float(max_score),
                )
            )
        return out

    def save(self) -> None:
        """Persist the corpus and metadata to disk."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("wb") as f:
            pickle.dump(
                {"corpus": self._corpus, "meta": self._chunk_meta},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log.debug("bm25_saved", path=str(self.persist_path), size=len(self._corpus))

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with self.persist_path.open("rb") as f:
                state = pickle.load(f)  # noqa: S301 — we control this file
            self._corpus = state["corpus"]
            self._chunk_meta = state["meta"]
            self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
            log.debug("bm25_loaded", path=str(self.persist_path), size=len(self._corpus))
        except (OSError, pickle.UnpicklingError, KeyError) as e:
            log.warning("bm25_load_failed", path=str(self.persist_path), error=str(e))
