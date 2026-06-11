"""Pydantic request/response schemas for the FastAPI service.

Keeping schemas in their own module (rather than inline in main.py) makes
them easy to share with clients via tools like openapi-codegen.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    text: str
    source_path: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    diagnostics: dict[str, int | str]


class IngestTextRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)
    source_path: str = ""


class IngestResponse(BaseModel):
    doc_id: str
    chunks_written: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
