from pydantic import BaseModel, Field
from typing import Optional


# ─── Request Models ───────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is multi-head attention?",
                "session_id": "user_abc123",
                "top_k": 5
            }
        }


# ─── Response Models ──────────────────────────────────

class CitationModel(BaseModel):
    source: str
    page: int | str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    citation_count: int
    session_id: str
    status: str
    rewrite_count: int
    generation_count: int
    answer_confidence: float
    hallucination_detected: bool
    latency_seconds: float
    is_followup: bool


class UploadResponse(BaseModel):
    message: str
    files_uploaded: list[str]
    total_chunks: int
    status: str


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexes_loaded: bool
    active_sessions: int
    documents_indexed: int
    total_chunks: int


class TraceResponse(BaseModel):
    session_id: str
    trace: list[dict]
    trace_formatted: str


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool
    message: str
