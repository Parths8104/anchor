# Architecture

This document describes how Anchor is structured, the responsibilities of
each module, and the data flow through the system.

## High-level diagram

```
            ┌─────────────────────────────────────────────────────────┐
            │                         API                              │
            │            FastAPI · Pydantic schemas · structlog        │
            └────────────┬────────────────────────────┬───────────────┘
                         │                            │
                  /ingest/text                     /query
                         │                            │
            ┌────────────▼─────────────┐  ┌───────────▼───────────────┐
            │      Ingestion           │  │       Retrieval           │
            │                          │  │                           │
            │  TokenChunker (tiktoken) │  │  HybridRetriever          │
            │  Embedder    (OpenAI)    │  │    ├─ dense  → VectorStore│
            │  Pipeline                │  │    └─ sparse → BM25Index  │
            │                          │  │  Fusion via RRF           │
            └────────┬─────────────────┘  └────────────┬──────────────┘
                     │                                  │
                     ▼                                  ▼
            ┌──────────────────────────┐  ┌────────────────────────────┐
            │ VectorStore (ChromaDB)   │  │       Generation           │
            │ BM25Index   (rank_bm25)  │  │                            │
            │                          │  │  Generator (OpenAI Chat)   │
            │  persistent local state  │  │  prompts.py                │
            └──────────────────────────┘  │  Citation parser           │
                                          └────────────┬───────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────────┐
                                          │    Eval Harness            │
                                          │                            │
                                          │  - answer similarity       │
                                          │  - groundedness (judge)    │
                                          │  - citation coverage       │
                                          │  - latency                 │
                                          └────────────────────────────┘
```

## Module responsibilities

### `anchor.config`
Single source of truth for runtime configuration. All env vars are
declared with explicit types via Pydantic Settings; downstream code
never reads `os.environ` directly.

### `anchor.logging_config`
Configures structlog for JSON-structured logs in production (auto-detected
from `isatty`) and pretty console logs in development. Every module
imports `get_logger(__name__)`.

### `anchor.ingestion`
- **`chunker.TokenChunker`** — token-aware sliding-window chunking via
  tiktoken. Token-based (not char-based) so chunk sizes are model-aware.
- **`embedder.Embedder`** — batched embedding requests to OpenAI with
  exponential-backoff retries for transient errors.
- **`pipeline.IngestionPipeline`** — coordinates chunker → embedder →
  vector store + BM25 writes. Idempotent over `doc_id`: re-ingesting a
  doc replaces its existing chunks rather than duplicating.

### `anchor.retrieval`
- **`vector_store.VectorStore`** — wrapper over ChromaDB. Persistent
  local mode (single SQLite file + HNSW index).
- **`bm25.BM25Index`** — sparse retrieval over a tokenized corpus.
  Persisted as a pickle file. Tokenizer keeps `snake_case` and digits
  intact, which matters for technical content.
- **`hybrid.HybridRetriever`** — runs both retrievers in parallel,
  fuses results via Reciprocal Rank Fusion. See
  [ADR-0001](./decisions/0001-hybrid-retrieval.md).

### `anchor.generation`
- **`prompts.py`** — system prompt instructing the LLM to cite passages
  and to refuse rather than hallucinate. The refusal instruction is
  essential — without it, models will invent content to fill gaps.
- **`generator.Generator`** — calls the chat completions API with the
  retrieved context and parses bracketed citations like `[1, 3]` back
  into structured `Citation` objects.

### `anchor.eval`
- **`metrics.py`** — three signals: embedding cosine similarity for
  answer quality, LLM-as-judge for groundedness, regex-based citation
  coverage.
- **`harness.EvalHarness`** — runs all cases from `evals/cases/`,
  produces per-case results and an aggregate summary report.

### `anchor.api`
- **`main.py`** — FastAPI app with three endpoints: `/health`, `/query`,
  `/ingest/text`. Singletons are wired in `lifespan()` so ChromaDB and
  BM25 stay warm across requests.

## Data flow: ingestion

```
text file
   │
   ▼ TokenChunker.chunk(doc_id, text)
list[Chunk]
   │
   ├──→ Embedder.embed(texts)        ──→ list[list[float]]
   │                                          │
   ▼                                          ▼
BM25Index.add(chunks)               VectorStore.add(chunks, embeddings)
   │                                          │
   ▼                                          ▼
.bm25.pkl on disk                  .chroma/ SQLite + HNSW on disk
```

## Data flow: query

```
question
   │
   ▼ Embedder.embed([q])
query embedding
   │
   ├──→ VectorStore.query(emb, k=10)  ──→ dense_results
   │
   └──→ BM25Index.query(text, k=10)   ──→ bm25_results
                              │
                              ▼  RRF fusion
                       fused_top_k chunks
                              │
                              ▼ Generator.generate(question, chunks)
                       GeneratedAnswer
                              │
                              ▼ parse [1], [2] citations
                       Citation[] resolved to chunks
                              │
                              ▼ API response
                       QueryResponse{ answer, citations, diagnostics }
```

## Failure modes and how the system handles them

| Failure | Where caught | Behavior |
|---|---|---|
| Empty corpus query | `VectorStore.query` / `BM25Index.query` | Returns `[]` — generator returns refusal answer |
| Transient OpenAI error | `Embedder._embed_with_retry` | Exponential backoff retry, up to 4 attempts |
| Permanent OpenAI error (4xx) | `Embedder._embed_with_retry` | Fail fast, propagate |
| LLM hallucinated citation index | `Generator._parse_citations` | Logged as warning, omitted from results |
| LLM ignores instructions and answers without context | Eval harness | Caught by groundedness judge → `passed = False` |
| BM25 index file corrupted | `BM25Index._load` | Warning logged, starts with empty index |
| Re-ingesting the same doc | `IngestionPipeline.ingest_text` | Old chunks deleted from both stores before re-write |

## Where this would extend in production

- **Reranking** — cross-encoder reranker (e.g.
  `cross-encoder/ms-marco-MiniLM`) over the top-N fused results.
  Sketched in `retrieval/` as a future module.
- **Caching** — semantic cache layer over `/query` keyed on embedding
  similarity of the question. Big cost saver for repeat queries.
- **Observability** — OpenTelemetry tracing across retrieval +
  generation; per-stage latency histograms exported to Prometheus.
- **Multi-tenancy** — `doc_id` would gain a `tenant_id` prefix; Chroma
  collections sharded per tenant or filtered via metadata.
- **Async ingestion** — large doc sets ingested via background workers
  (Celery / Arq) rather than blocking the API.
