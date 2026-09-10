"""Tests for the SemanticCache LRU + similarity behaviour."""

import pytest

from anchor.cache import CacheHit, SemanticCache


# Deterministic little vectors. Cosine similarity of a unit vector with
# itself is 1.0; with an orthogonal vector, 0.0.
V1 = [1.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0]
V3 = [0.9, 0.1, 0.0]  # very close to V1


def test_empty_cache_returns_none() -> None:
    cache = SemanticCache()
    assert cache.get(V1) is None
    assert len(cache) == 0


def test_put_then_get_exact_match_hits() -> None:
    cache = SemanticCache()
    cache.put("q1", V1, "answer 1")

    hit = cache.get(V1)
    assert isinstance(hit, CacheHit)
    assert hit.entry.answer == "answer 1"
    assert hit.similarity == pytest.approx(1.0)


def test_put_then_get_similar_hits_when_above_threshold() -> None:
    cache = SemanticCache(similarity_threshold=0.9)
    cache.put("q1", V1, "answer 1")

    hit = cache.get(V3)
    assert hit is not None
    assert hit.entry.answer == "answer 1"
    assert hit.similarity > 0.9


def test_get_returns_none_when_below_threshold() -> None:
    cache = SemanticCache(similarity_threshold=0.99)
    cache.put("q1", V1, "answer 1")

    # V3 is close but not close enough at 0.99 threshold.
    assert cache.get(V3) is None


def test_orthogonal_query_never_matches() -> None:
    cache = SemanticCache(similarity_threshold=0.5)
    cache.put("q1", V1, "answer 1")

    assert cache.get(V2) is None


def test_repeated_put_same_key_updates_in_place() -> None:
    cache = SemanticCache()
    cache.put("q1", V1, "answer 1")
    cache.put("q1", V1, "answer 2")

    assert len(cache) == 1
    hit = cache.get(V1)
    assert hit is not None
    assert hit.entry.answer == "answer 2"


def test_lru_evicts_oldest_when_full() -> None:
    cache = SemanticCache(max_entries=2, similarity_threshold=0.99)
    cache.put("q1", V1, "answer 1")
    cache.put("q2", V2, "answer 2")
    cache.put("q3", [0.0, 0.0, 1.0], "answer 3")

    # q1 should have been evicted (oldest).
    assert len(cache) == 2
    assert cache.get(V1) is None
    assert cache.get(V2) is not None
    assert cache.get([0.0, 0.0, 1.0]) is not None


def test_get_marks_entry_as_most_recently_used() -> None:
    cache = SemanticCache(max_entries=2, similarity_threshold=0.99)
    cache.put("q1", V1, "answer 1")
    cache.put("q2", V2, "answer 2")

    # Touch q1 so q2 becomes the oldest.
    cache.get(V1)

    # Adding a third entry should evict q2, not q1.
    cache.put("q3", [0.0, 0.0, 1.0], "answer 3")
    assert cache.get(V1) is not None
    assert cache.get(V2) is None


def test_clear_removes_everything() -> None:
    cache = SemanticCache()
    cache.put("q1", V1, "answer 1")
    cache.put("q2", V2, "answer 2")
    cache.clear()

    assert len(cache) == 0
    assert cache.get(V1) is None


def test_invalid_max_entries_raises() -> None:
    with pytest.raises(ValueError):
        SemanticCache(max_entries=0)


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        SemanticCache(similarity_threshold=1.5)
    with pytest.raises(ValueError):
        SemanticCache(similarity_threshold=-0.1)


def test_vector_length_mismatch_raises() -> None:
    cache = SemanticCache()
    cache.put("q1", V1, "answer 1")

    with pytest.raises(ValueError):
        cache.get([1.0, 0.0])  # length 2 vs stored length 3
