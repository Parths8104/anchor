"""Central configuration via Pydantic Settings (env-driven, type-safe)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration sourced from environment / .env file.

    All settings prefixed with ANCHOR_ are user-tunable. Defaults are
    chosen to be sensible for development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    generation_model: str = Field(
        default="gpt-4o-mini",
        alias="ANCHOR_GENERATION_MODEL",
        description="LLM used to generate the final answer.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="ANCHOR_EMBEDDING_MODEL",
        description="Embedding model for dense retrieval.",
    )
    judge_model: str = Field(
        default="gpt-4o-mini",
        alias="ANCHOR_JUDGE_MODEL",
        description="LLM used to judge groundedness in eval harness.",
    )

    chunk_tokens: int = Field(default=512, alias="ANCHOR_CHUNK_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="ANCHOR_CHUNK_OVERLAP_TOKENS")

    retrieval_top_k: int = Field(default=10, alias="ANCHOR_RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=4, alias="ANCHOR_RERANK_TOP_K")
    dense_weight: float = Field(
        default=0.5,
        alias="ANCHOR_DENSE_WEIGHT",
        description="Weight of dense scores in hybrid fusion (BM25 weight = 1 - dense_weight).",
    )

    vector_db_path: Path = Field(default=Path("./.chroma"), alias="ANCHOR_VECTOR_DB_PATH")
    bm25_index_path: Path = Field(default=Path("./.bm25.pkl"), alias="ANCHOR_BM25_INDEX_PATH")

    log_level: str = Field(default="INFO", alias="ANCHOR_LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Caching ensures we don't repeatedly parse env vars and that downstream
    code sees a consistent config snapshot.
    """
    return Settings()
