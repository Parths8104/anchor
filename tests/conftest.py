"""Shared pytest fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never accidentally hit the real OpenAI API.

    Individual tests that exercise OpenAI integration should mock the
    client explicitly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-do-not-use")


@pytest.fixture()
def tmp_vector_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolated ChromaDB path per test (avoids cross-test contamination)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "chroma"
        monkeypatch.setenv("ANCHOR_VECTOR_DB_PATH", str(path))
        # Clear settings cache so the env override takes effect.
        from anchor import config

        config.get_settings.cache_clear()
        yield path
        config.get_settings.cache_clear()


@pytest.fixture()
def tmp_bm25(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolated BM25 index path per test."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bm25.pkl"
        monkeypatch.setenv("ANCHOR_BM25_INDEX_PATH", str(path))
        from anchor import config

        config.get_settings.cache_clear()
        yield path
        config.get_settings.cache_clear()
