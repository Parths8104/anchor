"""Token-aware document chunking with sliding-window overlap.

Why token-based (not character or sentence): downstream embedding and
generation models have token limits, and embedding quality degrades near
those limits. Token-based chunking gives us deterministic, model-aware
chunk sizes.

Why overlap: pure non-overlapping chunks lose context at chunk boundaries.
A small overlap (~12% of chunk size) catches cross-boundary references
without significantly increasing index size.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class Chunk:
    """An immutable chunk of text plus enough metadata to cite back to source."""

    doc_id: str
    chunk_index: int
    text: str
    token_count: int

    @property
    def chunk_id(self) -> str:
        """Stable, globally unique identifier for this chunk."""
        return f"{self.doc_id}::chunk-{self.chunk_index:04d}"


class TokenChunker:
    """Chunks text by token count with optional sliding-window overlap.

    Uses tiktoken's cl100k_base encoding by default (the encoding used by
    GPT-4 / GPT-4o / text-embedding-3-* models).
    """

    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be in [0, chunk_tokens)")

        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        """Split text into a list of Chunks. Materialized for downstream batching."""
        return list(self._chunk_iter(doc_id, text))

    def _chunk_iter(self, doc_id: str, text: str) -> Iterator[Chunk]:
        tokens = self.encoder.encode(text)
        if not tokens:
            return

        stride = self.chunk_tokens - self.overlap_tokens
        chunk_index = 0
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_tokens, len(tokens))
            window = tokens[start:end]
            chunk_text = self.encoder.decode(window)

            yield Chunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(window),
            )

            chunk_index += 1
            if end == len(tokens):
                break
            start += stride
