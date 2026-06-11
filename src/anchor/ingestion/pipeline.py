"""End-to-end ingestion pipeline.

Reads markdown / text files, chunks them, embeds them, and writes to both
the dense (vector) and sparse (BM25) indexes. Idempotent over doc_id:
re-ingesting a doc replaces its existing chunks rather than duplicating.
"""

import hashlib
from pathlib import Path

from anchor.config import get_settings
from anchor.ingestion.chunker import Chunk, TokenChunker
from anchor.ingestion.embedder import Embedder
from anchor.logging_config import get_logger
from anchor.retrieval.bm25 import BM25Index
from anchor.retrieval.vector_store import VectorStore

log = get_logger(__name__)


def doc_id_for(path: Path) -> str:
    """Deterministic doc_id derived from the file's relative path.

    Using a hash makes IDs filesystem-safe (no path separators) and stable
    across runs as long as the path doesn't change.
    """
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:12]
    return f"{path.stem}-{digest}"


class IngestionPipeline:
    """Coordinates chunking, embedding, and index writes."""

    def __init__(
        self,
        chunker: TokenChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        settings = get_settings()
        self.chunker = chunker or TokenChunker(
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25 or BM25Index()

    def ingest_file(self, path: Path) -> int:
        """Ingest a single file. Returns number of chunks written."""
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(doc_id=doc_id_for(path), text=text, source_path=str(path))

    def ingest_directory(self, root: Path, glob: str = "**/*.md") -> int:
        """Ingest every file under `root` matching `glob`. Returns total chunks."""
        total = 0
        for path in sorted(root.glob(glob)):
            if path.is_file():
                count = self.ingest_file(path)
                total += count
                log.info("ingested_file", path=str(path), chunks=count)
        return total

    def ingest_text(self, doc_id: str, text: str, source_path: str = "") -> int:
        """Ingest raw text under a given doc_id. Returns number of chunks."""
        chunks = self.chunker.chunk(doc_id, text)
        if not chunks:
            log.warning("empty_document", doc_id=doc_id)
            return 0

        # Idempotency: remove any existing chunks for this doc_id first.
        self.vector_store.delete_by_doc(doc_id)

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add(chunks=chunks, embeddings=embeddings, source_path=source_path)
        self.bm25.add(chunks)
        self.bm25.save()

        log.info("ingested_doc", doc_id=doc_id, chunks=len(chunks))
        return len(chunks)

    def list_chunks(self, doc_id: str) -> list[Chunk]:
        """Diagnostic: return all chunks indexed for a given doc_id."""
        return self.vector_store.list_by_doc(doc_id)
