"""Grounded answer generation with citation parsing.

Given a question and the retrieved chunks, prompts an LLM to answer with
inline bracketed citations, then parses those citations into structured
references back to source chunks.
"""

import re
from dataclasses import dataclass

from openai import OpenAI

from anchor.config import get_settings
from anchor.generation.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_context_passages,
)
from anchor.logging_config import get_logger
from anchor.retrieval.vector_store import RetrievedChunk

log = get_logger(__name__)

# Matches bracketed citations like [1], [2, 3], [4,5,6]
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    """A single citation linking an answer span to a source chunk."""

    index: int  # 1-indexed as cited in the answer
    chunk_id: str
    doc_id: str
    text: str
    source_path: str = ""


@dataclass(frozen=True)
class GeneratedAnswer:
    """Final answer plus parsed citations and raw token usage."""

    answer: str
    citations: list[Citation]
    used_chunks: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    model: str


class Generator:
    """Calls the LLM with the grounded prompt and parses out citations."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.generation_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        """Generate a grounded answer for the question given retrieved chunks."""
        if not chunks:
            return GeneratedAnswer(
                answer="I don't have enough information in the provided context to answer that.",
                citations=[],
                used_chunks=[],
                prompt_tokens=0,
                completion_tokens=0,
                model=self.model,
            )

        context = format_context_passages([c.text for c in chunks])
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        answer_text = resp.choices[0].message.content or ""
        citations = self._parse_citations(answer_text, chunks)

        usage = resp.usage
        log.info(
            "generation_complete",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            citations=len(citations),
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            used_chunks=chunks,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )

    @staticmethod
    def _parse_citations(
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        """Extract unique citation indices from the answer and resolve to chunks."""
        cited_indices: set[int] = set()
        for match in CITATION_PATTERN.finditer(answer):
            for part in match.group(1).split(","):
                cited_indices.add(int(part.strip()))

        citations: list[Citation] = []
        for idx in sorted(cited_indices):
            # idx is 1-indexed; chunks list is 0-indexed.
            if 1 <= idx <= len(chunks):
                chunk = chunks[idx - 1]
                citations.append(
                    Citation(
                        index=idx,
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        source_path=chunk.source_path,
                    )
                )
            else:
                # Model hallucinated a citation index — log it; surfaced
                # later via the eval harness as a groundedness signal.
                log.warning("invalid_citation_index", index=idx, chunks_available=len(chunks))

        return citations
