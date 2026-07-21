# Anchor

> Production-grade RAG with citation tracing and evaluation-first design.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

## Architecture at a glance

```mermaid
flowchart TB
    subgraph Ingestion["📥 Ingestion"]
        A[Markdown / Text Docs] --> B[TokenChunker<br/>tiktoken · 512 tokens · 64 overlap]
        B --> C[Embedder<br/>OpenAI text-embedding-3-small]
    end

    C --> D[(Vector Store<br/>ChromaDB)]
    B --> E[(BM25 Index<br/>rank-bm25)]

    subgraph Retrieval["🔍 Retrieval"]
        F[User Question] --> G[Embed query]
        G --> H[Dense search<br/>cosine similarity]
        F --> I[Sparse search<br/>BM25 scoring]
        H --> J[Reciprocal Rank Fusion<br/>k=60]
        I --> J
    end

    D -.retrieves.-> H
    E -.retrieves.-> I

    subgraph Generation["✍️ Generation"]
        J --> K[Format context<br/>with numbered passages]
        K --> L[LLM Call<br/>gpt-4o-mini]
        L --> M[Citation Parser<br/>bracketed indices]
        M --> N[Cited Answer]
    end

    N --> O{Eval Harness}
    O -.scores.-> P[Similarity]
    O -.scores.-> Q[Groundedness<br/>LLM-as-judge]
    O -.scores.-> R[Citation Coverage]

    style A fill:#e1f5ff,stroke:#333,color:#000
    style N fill:#d4edda,stroke:#333,color:#000
    style D fill:#fff3cd,stroke:#333,color:#000
    style E fill:#fff3cd,stroke:#333,color:#000
```

## How a query flows through the system

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant API as FastAPI
    participant R as HybridRetriever
    participant V as VectorStore
    participant B as BM25Index
    participant G as Generator
    participant L as OpenAI LLM

    U->>API: POST /query { question }
    API->>R: retrieve(question)
    R->>R: embed(question)
    par Dense retrieval
        R->>V: query(embedding, k=10)
        V-->>R: top chunks by cosine
    and Sparse retrieval
        R->>B: query(question, k=10)
        B-->>R: top chunks by BM25
    end
    R->>R: RRF fusion → top 4
    R-->>API: 4 fused chunks
    API->>G: generate(question, chunks)
    G->>L: chat completion with cited context
    L-->>G: answer with [1], [2] citations
    G->>G: parse citations → structured refs
    G-->>API: answer + citations + tokens
    API-->>U: JSON { answer, citations, diagnostics }
```

## Why hybrid retrieval?

Dense embeddings catch semantic similarity but miss exact terms — acronyms, function names, error strings. BM25 catches those. Fusing both via RRF gives noticeably better recall on technical docs than either alone. Full rationale in [ADR-0001](./docs/decisions/0001-hybrid-retrieval.md).

| Query type | Dense-only wins | BM25-only wins | Hybrid wins |
|---|:---:|:---:|:---:|
| Paraphrased semantic | ✅ | ❌ | ✅ |
| Exact keyword / code identifier | ❌ | ✅ | ✅ |
| Mixed (concept + specific term) | ⚠️ | ⚠️ | ✅ |

Anchor is a retrieval-augmented generation system that answers questions from your documents and **cites the exact passages it used**. Every claim in the answer maps back to a source chunk; claims that can't be grounded trigger an explicit refusal rather than a hallucination.

The system is designed around three principles I wish more RAG implementations took seriously:

1. **Hybrid retrieval beats dense-only retrieval.** Embedding similarity misses acronyms, code identifiers, and exact-keyword queries. BM25 catches those. Fusing both via Reciprocal Rank Fusion is robust and parameter-free. ([ADR-0001](./docs/decisions/0001-hybrid-retrieval.md))
2. **Citations must be parseable and verifiable.** Inline `[1, 2]` references are easy for LLMs to emit reliably and easy to programmatically check. ([ADR-0002](./docs/decisions/0002-citation-format.md))
3. **Evaluation is part of the system, not an afterthought.** A built-in harness scores answers on similarity, groundedness, and citation coverage — so prompt changes ship with evidence, not vibes.

## What's in here

```
anchor/
├── src/anchor/
│   ├── api/           FastAPI service + Pydantic schemas
│   ├── ingestion/     token-aware chunking · batched embedding · pipeline
│   ├── retrieval/     ChromaDB vector store · BM25 index · hybrid fusion
│   ├── generation/    prompt templates · grounded generator · citation parser
│   ├── eval/          metrics · LLM-as-judge · harness
│   └── config.py      env-driven Pydantic settings
├── evals/
│   ├── cases/         JSON test cases
│   └── run.py         eval runner CLI
├── data/              sample documents (Python & FastAPI fundamentals)
├── tests/             pytest unit tests
├── scripts/           ingest.py · query.py CLIs
├── docs/decisions/    Architecture Decision Records
├── ARCHITECTURE.md    deep dive on system structure & data flow
└── Dockerfile         containerized deployment
```

## Quick start

### 1. Install

```bash
git clone https://github.com/Parths8104/anchor.git
cd anchor
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Ingest sample documents

```bash
python scripts/ingest.py data/
```

You should see structured log output like:

```
[info] ingested_doc doc_id=python_basics-a1b2... chunks=4
[info] ingested_doc doc_id=fastapi_intro-c3d4... chunks=6
[info] ingestion_complete total_chunks=10
```

### 4. Ask a question from the CLI

```bash
python scripts/query.py "How does dependency injection work in FastAPI?"
```

Sample output:

```
========================================================================
QUESTION: How does dependency injection work in FastAPI?
========================================================================

ANSWER:
FastAPI provides dependency injection through the Depends() function [1].
You declare a parameter with Depends(callable) and FastAPI will call that
callable for each request, passing the result into your endpoint [1].
Dependencies can themselves declare further dependencies, forming a tree
that FastAPI resolves automatically [2]. Dependencies are cached per
request by default so the same callable isn't invoked twice in a single
request [2].

CITATIONS:
  [1] fastapi_intro-c3d4...::chunk-0002: FastAPI provides dependency...
  [2] fastapi_intro-c3d4...::chunk-0003: Dependencies can themselves...

DIAGNOSTICS:
  dense retrieved : 10
  bm25 retrieved  : 6
  chunks used     : 4
  prompt tokens   : 412
  completion tok. : 98
```

### 5. Or run it as a service

```bash
uvicorn anchor.api.main:app --reload
```

Then `POST` to `/query`:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "When should I use asyncio in Python?"}'
```

Interactive docs at `http://localhost:8000/docs`.

### 6. Run the eval harness

```bash
python evals/run.py
```

Output:

```
EVAL SUMMARY
============================================================
              cases : 3
             passed : 3
          pass_rate : 1.0
    mean_similarity : 0.87
      mean_coverage : 0.91
      grounded_rate : 1.0
    mean_latency_ms : 1843.2

Report: evals/reports/report-20260601T120000Z.json
```

## Running with Docker

```bash
docker build -t anchor .
docker run --rm -p 8000:8000 --env-file .env anchor
```

Or with `docker compose`:

```bash
docker compose up --build
```

## Running the tests

```bash
pytest tests/ -v
```

Includes unit tests for chunking, BM25, RRF fusion logic, and metrics. CI runs the test suite on Python 3.10, 3.11, and 3.12 via GitHub Actions ([workflow](./.github/workflows/ci.yml)).

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full diagram, module responsibilities, data flows for ingestion and query, and how failure modes are handled.

Key design decisions are recorded as ADRs:
- [ADR-0001 — Hybrid retrieval via Reciprocal Rank Fusion](./docs/decisions/0001-hybrid-retrieval.md)
- [ADR-0002 — Inline bracketed citations](./docs/decisions/0002-citation-format.md)

## What's deliberately not in v1

These are things a production deployment would want, sketched out in [ARCHITECTURE.md](./ARCHITECTURE.md) for the path forward:

- **Cross-encoder reranking** — the top-N fused results would benefit from a reranker pass before generation. Skipped here to keep dependencies light.
- **Semantic caching** — query embeddings hashed against an LRU cache to skip repeat work.
- **OpenTelemetry tracing** — per-stage spans with latency histograms.
- **Multi-tenant isolation** — `doc_id` prefixed with `tenant_id`, per-tenant Chroma collections.
- **Async batched ingestion** — large doc sets processed via background workers rather than blocking the API.

## License

MIT — see [LICENSE](./LICENSE).

## Author

Built by [Parth Sojitra](https://github.com/Parths8104). Reach me at psojitraswe@gmail.com or via [LinkedIn](https://www.linkedin.com/in/parth-sojitra-84098427b/).
