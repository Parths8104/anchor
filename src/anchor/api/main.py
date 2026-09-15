"""FastAPI application exposing Anchor as a service.

Endpoints:
  POST /query           — ask a question, get a cited answer
  POST /ingest/text     — index a raw text payload under a doc_id
  GET  /health          — basic readiness probe

Design choices:
  - Singleton pipeline/retriever/generator instantiated at startup. This
    keeps ChromaDB and BM25 in-memory state warm across requests.
  - All errors mapped to FastAPI HTTPExceptions; structlog handles
    contextual logging.
"""

from contextlib import asynccontextmanager
from typing import cast

from fastapi import Depends, FastAPI, HTTPException

from anchor.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from anchor.cache import SemanticCache
from anchor.config import get_settings
from anchor.generation.generator import Generator
from anchor.ingestion.pipeline import IngestionPipeline
from anchor.logging_config import configure_logging, get_logger
from anchor.retrieval.hybrid import HybridRetriever

log = get_logger(__name__)


class AppState:
    """Container for singletons. Attached to app.state at startup."""

    pipeline: IngestionPipeline
    retriever: HybridRetriever
    generator: Generator
    cache: SemanticCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook: build singletons once, share across requests."""
    configure_logging()
    log.info("starting_anchor")

    state = AppState()
    state.pipeline = IngestionPipeline()
    # The retriever shares the same VectorStore + BM25 instances created
    # inside the pipeline so that newly ingested docs are immediately
    # queryable without a restart.
    state.retriever = HybridRetriever(
        vector_store=state.pipeline.vector_store,
        bm25=state.pipeline.bm25,
        embedder=state.pipeline.embedder,
    )
    state.generator = Generator()

    # Semantic cache — check cache before retrieval, populate after generation.
    # Config knobs live in Settings; disabled cache still creates the object
    # with max_entries=1 so downstream code doesn't need None-checks.
    settings = get_settings()
    state.cache = SemanticCache(
        max_entries=settings.cache_max_entries if settings.cache_enabled else 1,
        similarity_threshold=settings.cache_similarity_threshold,
    )

    app.state.singletons = state
    log.info("anchor_ready", indexed_chunks=state.pipeline.vector_store.count())

    yield

    log.info("shutting_down")


app = FastAPI(
    title="Anchor",
    description="Production-grade RAG with citation tracing.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_state(app_instance: FastAPI = Depends(lambda: app)) -> AppState:  # noqa: B008
    return cast(AppState, app_instance.state.singletons)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    state = cast(AppState, app.state.singletons)
    return HealthResponse(
        status="ok",
        indexed_chunks=state.pipeline.vector_store.count(),
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(req: QueryRequest) -> QueryResponse:
    state = cast(AppState, app.state.singletons)
    settings = get_settings()

    try:
        # Embed the question once — reused for cache lookup AND (on miss)
        # for dense retrieval, so we don't pay for two embed calls.
        question_embedding = state.pipeline.embedder.embed([req.question])[0]

        # Cache check. On hit, return the stored answer without running
        # retrieval or generation.
        if settings.cache_enabled:
            hit = state.cache.get(question_embedding)
            if hit is not None:
                log.info(
                    "cache_hit",
                    similarity=round(hit.similarity, 4),
                    cached_query=hit.entry.query_text,
                )
                return QueryResponse(
                    answer=hit.entry.answer,
                    citations=[],  # cached answers don't re-derive citations
                    diagnostics={
                        "cache_hit": "true",
                        "cache_similarity": f"{hit.similarity:.4f}",
                        "cached_query": hit.entry.query_text,
                    },
                )

        # Cache miss: run the full pipeline.
        retrieval = state.retriever.retrieve(req.question, top_k=req.top_k)
        answer = state.generator.generate(req.question, retrieval.chunks)

        # Populate cache so subsequent similar queries hit.
        if settings.cache_enabled:
            state.cache.put(
                query_text=req.question,
                query_embedding=question_embedding,
                answer=answer.answer,
            )
    except Exception as e:
        log.error("query_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Query failed") from e

    return QueryResponse(
        answer=answer.answer,
        citations=[
            CitationOut(
                index=c.index,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                source_path=c.source_path,
            )
            for c in answer.citations
        ],
        diagnostics={
            "cache_hit": "false",
            "dense_retrieved": str(retrieval.dense_count),
            "bm25_retrieved": str(retrieval.bm25_count),
            "chunks_used": str(len(retrieval.chunks)),
            "prompt_tokens": str(answer.prompt_tokens),
            "completion_tokens": str(answer.completion_tokens),
            "model": answer.model,
        },
    )

@app.post("/ingest/text", response_model=IngestResponse, tags=["ingestion"])
async def ingest_text(req: IngestTextRequest) -> IngestResponse:
    state = cast(AppState, app.state.singletons)
    try:
        count = state.pipeline.ingest_text(
            doc_id=req.doc_id,
            text=req.text,
            source_path=req.source_path,
        )
    except Exception as e:
        log.error("ingest_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Ingest failed") from e

    return IngestResponse(doc_id=req.doc_id, chunks_written=count)
