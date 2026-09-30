"""In-memory semantic cache for query → answer pairs.

Why semantic (not exact-match): the same user often phrases the same
question slightly differently across sessions. Exact string matching
misses these; embedding-based similarity catches them without
false-matching genuinely different questions if the threshold is set
conservatively (default 0.95).

Why in-memory: the cache is a latency + cost optimization, not a
correctness requirement. Losing it on process restart is acceptable.
For multi-instance deployments, this would move behind Redis — the
interface below (`get` / `put`) is deliberately narrow so that swap
is a focused change.
"""

from collections import OrderedDict
from dataclasses import dataclass


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class CacheEntry:
    """A single (query embedding, cached answer) pair."""

    query_text: str
    embedding: list[float]
    answer: str


@dataclass(frozen=True)
class CacheHit:
    """A successful cache lookup with the matched entry and its similarity."""

    entry: CacheEntry
    similarity: float


class SemanticCache:
    """LRU cache keyed on embedding cosine similarity.

    Not thread-safe. Fine for the single-process FastAPI worker Anchor
    ships with; would need a lock (or a swap to Redis) for multi-worker.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.95,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")

        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold

        # OrderedDict preserves insertion order, which we use for LRU:
        # move_to_end on access, popitem(last=False) on eviction.
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, query_embedding: list[float]) -> CacheHit | None:
        """Return the best matching entry if similarity >= threshold."""

        best_key: str | None = None
        best_similarity = -1.0

        for key, entry in self._entries.items():
            sim = _cosine_similarity(
                query_embedding,
                entry.embedding,
            )

            if sim > best_similarity:
                best_similarity = sim
                best_key = key

        if best_key is None or best_similarity < self.similarity_threshold:
            self._misses += 1
            return None

        self._entries.move_to_end(best_key)
        self._hits += 1

        return CacheHit(
            entry=self._entries[best_key],
            similarity=best_similarity,
        )

    def put(
        self,
        query_text: str,
        query_embedding: list[float],
        answer: str,
    ) -> None:
        """Store an answer and evict the least-recently-used entry if full."""

        entry = CacheEntry(
            query_text=query_text,
            embedding=query_embedding,
            answer=answer,
        )

        if query_text in self._entries:
            del self._entries[query_text]

        self._entries[query_text] = entry

        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, int | float]:
        """Return process-lifetime cache usage statistics."""

        total_lookups = self._hits + self._misses

        hit_rate = (
            self._hits / total_lookups
            if total_lookups
            else 0.0
        )

        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()
