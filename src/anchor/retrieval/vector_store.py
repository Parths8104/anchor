"""Vector store wrapper over ChromaDB.

Why ChromaDB: embedded, single-file persistence, no separate server
process. Good for portfolio / demo deployments. The interface here is
narrow enough that swapping in Pinecone, Weaviate, or pgvector later is
a focused change rather than a rewrite.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from anchor.config import get_settings
from anchor.ingestion.chunker import Chunk
from anchor.logging_config import get_logger

log = get_logger(__name__)

COLLECTION_NAME = "anchor_chunks"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from retrieval with its score."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: float
    source_path: str = ""


class VectorStore:
    """Dense vector store backed by ChromaDB (persistent local mode)."""

    def __init__(self, persist_path: str | None = None) -> None:
        settings = get_settings()
        path = persist_path or str(settings.vector_db_path)
        self.client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # We provide embeddings ourselves rather than letting Chroma call
        # the embedding API. This keeps the embedding model under the
        # caller's control and makes the pipeline testable.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source_path: str = "",
    ) -> None:
        """Upsert chunks + embeddings into the store."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "source_path": source_path,
                }
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int = 10) -> list[RetrievedChunk]:
        """Return top-k nearest chunks for a query embedding."""
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
        )

        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        # Chroma returns cosine distance; convert to similarity in [0, 1].
        distances = result["distances"][0]

        out: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=True):
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta["doc_id"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=doc,
                    score=1.0 - float(dist),
                    source_path=str(meta.get("source_path", "")),
                )
            )
        return out

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove all chunks for a given doc_id (used for idempotent re-ingest)."""
        self.collection.delete(where={"doc_id": doc_id})

    def list_by_doc(self, doc_id: str) -> list[Chunk]:
        """Return all chunks indexed for a doc_id (diagnostic)."""
        result = self.collection.get(where={"doc_id": doc_id})
        chunks: list[Chunk] = []
        for doc, meta in zip(result["documents"], result["metadatas"], strict=True):
            chunks.append(
                Chunk(
                    doc_id=str(meta["doc_id"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=doc,
                    token_count=int(meta.get("token_count", 0)),
                )
            )
        return chunks

    def count(self) -> int:
        return self.collection.count()
