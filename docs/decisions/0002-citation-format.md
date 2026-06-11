# ADR-0002: Inline Bracketed Citations

**Status:** accepted
**Date:** 2026-06-01

## Context

The generator needs to attribute each factual claim to a specific
retrieved passage. The format of those citations affects (a) how
reliably the LLM produces them, (b) how cleanly we can parse them, and
(c) how groundedness can be verified.

## Decision

Use **inline bracketed indices** of the form `[1]`, `[2, 3]` immediately
after the claim they support. The number references the passage's
position in the context block passed to the LLM (1-indexed).

The system prompt explicitly instructs the model to:
1. Cite every factual claim.
2. Use only the indices that exist in the context.
3. Refuse to answer (with a fixed phrase) when context is insufficient.

## Considered alternatives

### 1. Footnote-style citations: `[^doc-id-chunk-3]`

**Rejected.** LLMs are unreliable at reproducing long opaque identifiers
verbatim. Hallucinated IDs become hard to distinguish from real ones.

### 2. URL-style links inline

**Rejected.** Same reproduction problem, plus URLs add tokens. The
client-side renderer can map `[1]` to a URL after parsing if needed.

### 3. JSON-structured output with separate `answer` and `citations` fields

**Considered seriously.** Strict structured output (via tool use or JSON
mode) is the most reliable way to extract structured data from LLMs.

Rejected for v1 because:
- The current shape (cited prose) reads naturally for end users.
- Structured output adds latency from extra schema validation.
- Inline citations preserve the LLM's reasoning flow — when the model
  has to first generate prose AND then produce a structured
  citation list, it's more likely to mis-align them.

Worth revisiting if we add a frontend that needs hover-card citation
previews; the JSON path would make that rendering simpler.

## Consequences

**Positive:**
- Reproduction-friendly: integers are easy for LLMs to emit reliably.
- Parseable with a simple regex (`\[(\d+(?:,\s*\d+)*)\]`).
- Invalid indices (e.g. `[7]` when only 4 passages were provided) are
  detectable — they're surfaced as warnings and excluded from the
  Citation list.
- Reads naturally in the answer text without specialized rendering.

**Negative:**
- The model occasionally over-cites (citing every passage for a single
  claim) or under-cites (one citation for a multi-claim sentence). The
  `citation_coverage` metric catches the under-cite case in eval; over-
  cite is mostly cosmetic.
- Index-based references break if we reorder the context block between
  passing it to the LLM and rendering the answer. We don't reorder, but
  this is a constraint future refactors need to respect.
