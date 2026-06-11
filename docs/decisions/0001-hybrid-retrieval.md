# ADR-0001: Hybrid Retrieval via Reciprocal Rank Fusion

**Status:** accepted
**Date:** 2026-06-01

## Context

A baseline RAG system retrieves passages by dense embedding similarity:
embed the query, find the top-K nearest passages by cosine distance.

This works well for paraphrased queries where the semantic intent is
clear, but degrades on:

- **Acronyms and proper nouns** — embeddings often confuse `MCP` with
  similar three-letter strings.
- **Code identifiers** — `useEffect`, `Depends()`, `argparse` — embedded
  poorly because they rarely appear in the embedding model's training
  data the same way.
- **Exact-keyword recall** — queries with one critical term that must
  match.

Sparse retrieval (BM25) handles these cases natively. The question is
how to combine the two.

## Decision

Use **Reciprocal Rank Fusion (RRF)** to combine dense and sparse
results, with `k = 60` as recommended in the original paper
(Cormack, Clarke, Buettcher, 2009).

```
RRF_score(d) = Σ over each ranker r of:  1 / (k + rank_r(d))
```

## Considered alternatives

### 1. Dense-only retrieval

**Rejected.** Misses keyword-exact queries. Especially weak for technical
content where users search for specific function names or error strings.

### 2. Weighted sum of normalized dense and BM25 scores

```
score(d) = α · cosine_sim(d) + (1 - α) · normalize(bm25(d))
```

**Rejected.** Highly sensitive to the choice of α and to the
normalization scheme. Different document corpora and query types want
different α values, which means α becomes a tuning knob with no good
universal default. RRF doesn't need one.

### 3. Cross-encoder rerank over union of dense + BM25 candidates

**Deferred.** Reranking is strictly better quality than RRF alone but
adds latency (typically 50–200ms for a cross-encoder pass) and a
dependency on `sentence-transformers`. Worth adding as an optional layer
*after* RRF, but RRF gives most of the recall gain at near-zero cost.

## Consequences

**Positive:**
- Robust to score-magnitude differences between rankers.
- No tunable weights → no per-corpus calibration needed.
- Cheap to compute: O(N log N) sort, no extra model calls.

**Negative:**
- RRF ignores the absolute confidence of each ranker. A high-confidence
  dense hit at rank 1 is treated identically to a marginal one. In
  practice this is usually fine because RRF runs on the top-K from each
  list, but it's worth flagging.
- For corpora where one ranker is dramatically better than the other,
  RRF gives away some quality vs a tuned weighted-sum.

## How we validated this

Run `python evals/run.py` against the cases in `evals/cases/` with the
dense-only path enabled (manually swap in `VectorStore.query` results
for fusion input) vs the hybrid path. On the included cases, hybrid
beats dense-only on `pass_rate` and `mean_similarity` by a noticeable
margin on the FastAPI/Python doc corpus.
