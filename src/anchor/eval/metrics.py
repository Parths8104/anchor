"""Evaluation metrics for grounded RAG outputs.

Three signals:

1. ANSWER SIMILARITY — embedding cosine similarity between the generated
   answer and a reference answer. Robust to phrasing differences in a way
   that exact-match never is.

2. GROUNDEDNESS — does the answer rely ONLY on facts in the cited
   chunks? Implemented as LLM-as-judge against the answer + cited
   passages.

3. CITATION COVERAGE — what fraction of factual sentences in the answer
   have at least one citation? Sentences without a citation can't be
   verified.

A test "passes" when all three signals clear configurable thresholds.
"""

import re
from dataclasses import dataclass

from openai import OpenAI

from anchor.config import get_settings
from anchor.generation.generator import Citation
from anchor.logging_config import get_logger

log = get_logger(__name__)


# A "factual sentence" is a sentence that isn't a refusal or boilerplate.
# We strip the standard refusal phrase before counting sentences.
REFUSAL_MARKER = "i don't have enough information"


@dataclass(frozen=True)
class EvalMetrics:
    """Per-case metrics from the eval harness."""

    answer_similarity: float
    grounded: bool
    citation_coverage: float
    latency_ms: float
    passed: bool


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_text(text: str, client: OpenAI | None = None) -> list[float]:
    """Embed a single piece of text for similarity scoring."""
    settings = get_settings()
    cli = client or OpenAI(api_key=settings.openai_api_key)
    resp = cli.embeddings.create(model=settings.embedding_model, input=text)
    return resp.data[0].embedding


GROUNDEDNESS_JUDGE_PROMPT = """\
You are a strict fact-checker. Below is an ANSWER and the CITED PASSAGES
the answer claims to be based on.

Decide whether the ANSWER relies ONLY on facts present in the CITED
PASSAGES. If the answer introduces any non-trivial claim not supported
by the passages, reply NO. If it sticks strictly to the passages (or
correctly refuses to answer), reply YES.

CITED PASSAGES:
{passages}

ANSWER:
{answer}

Reply with exactly one word: YES or NO.\
"""


def judge_groundedness(answer: str, citations: list[Citation]) -> bool:
    """LLM-as-judge: is the answer grounded in the cited passages?"""
    settings = get_settings()
    if answer.strip().lower().startswith(REFUSAL_MARKER):
        # Correctly refusing is by definition grounded.
        return True
    if not citations:
        # Non-refusal answer with no citations is not grounded.
        return False

    passages = "\n\n".join(f"[{c.index}] {c.text}" for c in citations)
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.judge_model,
        messages=[
            {
                "role": "user",
                "content": GROUNDEDNESS_JUDGE_PROMPT.format(
                    passages=passages,
                    answer=answer,
                ),
            }
        ],
        temperature=0,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    return verdict.startswith("YES")


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CITATION_IN_SENTENCE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def citation_coverage(answer: str) -> float:
    """Fraction of non-trivial sentences in the answer that contain at least one citation.

    Returns 1.0 for refusal answers (no claims need citation), and for
    answers with no detectable sentences.
    """
    if answer.strip().lower().startswith(REFUSAL_MARKER):
        return 1.0

    sentences = [s.strip() for s in SENTENCE_SPLIT.split(answer) if s.strip()]
    sentences = [s for s in sentences if len(s) > 20]  # skip short fragments
    if not sentences:
        return 1.0

    cited = sum(1 for s in sentences if CITATION_IN_SENTENCE.search(s))
    return cited / len(sentences)
