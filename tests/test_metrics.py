"""Tests for citation-coverage and similarity metrics."""

from anchor.eval.metrics import citation_coverage, cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_citation_coverage_refusal_returns_one() -> None:
    answer = "I don't have enough information in the provided context to answer that."
    assert citation_coverage(answer) == 1.0


def test_citation_coverage_all_cited() -> None:
    answer = (
        "FastAPI uses dependency injection through Depends [1]. "
        "Dependencies are cached per request by default [2]."
    )
    assert citation_coverage(answer) == 1.0


def test_citation_coverage_no_citations() -> None:
    answer = (
        "FastAPI uses dependency injection through Depends. "
        "Dependencies are cached per request by default."
    )
    assert citation_coverage(answer) == 0.0


def test_citation_coverage_partial() -> None:
    answer = (
        "FastAPI uses dependency injection through Depends [1]. "
        "It is a popular framework with great documentation."
    )
    # 1 of 2 sentences cited.
    assert citation_coverage(answer) == 0.5


def test_citation_coverage_ignores_short_fragments() -> None:
    answer = "Yes. FastAPI uses dependency injection through Depends [1]."
    # 'Yes.' is too short to count — only the long sentence matters.
    assert citation_coverage(answer) == 1.0
