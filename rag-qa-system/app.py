import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api_models import (
    QueryRequest, QueryResponse, CitationModel,
    UploadResponse, DocumentsResponse, DocumentInfo,
    HealthResponse, TraceResponse, SessionDeleteResponse
)
from src.session_manager import SessionManager
from src.orchestrator import run_agentic_pipeline, format_trace_for_display, get_graph
from src.graph_nodes import get_models, reload_indexes


# ─── App lifespan: runs on startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — pre-loading models and indexes...")
    get_graph()      # compile LangGraph once
    get_models()     # load FAISS + BM25 + embedding model
    yield
    print("Shutting down")


app = FastAPI(
    title="Agentic RAG Q&A API",
    description="Multi-document Q&A with LangGraph agentic orchestration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global session manager
session_manager = SessionManager(max_turns=4)

# PDF upload directory
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── POST /upload ──────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDF files.
    Saves them to the data/ folder, then rebuilds FAISS + BM25 indexes.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_names = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF"
            )
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded_names.append(file.filename)

    # Rebuild indexes with new documents
    from src.document_loader import load_pdfs_from_folder
    from src.text_chunker import chunk_documents
    from src.embedder import load_embedding_model, generate_embeddings
    from src.vector_store import build_index, save_index
    from src.bm25_retriever import build_bm25_index, save_bm25_index

    print(f"Rebuilding indexes with {len(uploaded_names)} new file(s)...")

    docs = load_pdfs_from_folder(UPLOAD_DIR)
    chunks = chunk_documents(docs)
    embed_model = load_embedding_model()
    embeddings = generate_embeddings(chunks, embed_model)
    index = build_index(embeddings)
    save_index(index, chunks)
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    # Force graph_nodes to reload from disk on next request
    reload_indexes()

    return UploadResponse(
        message=f"Successfully indexed {len(uploaded_names)} file(s)",
        files_uploaded=uploaded_names,
        total_chunks=len(chunks),
        status="success"
    )


# ─── POST /query ───────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Run the full agentic RAG pipeline for a question.
    Returns answer, citations, agent trace metadata, and confidence score.
    """
    session_id = request.session_id
    memory = session_manager.get_or_create(session_id)
    conversation_history = memory.format_history_for_prompt()
    is_followup = memory.is_followup_question(request.question)

    result = run_agentic_pipeline(
        question=request.question,
        session_id=session_id,
        conversation_history=conversation_history,
        top_k=request.top_k
    )

    # Store turn in memory
    memory.add_turn(
        request.question,
        result["answer"],
        citations=result.get("citations", [])
    )

    # Store trace for /agent/trace endpoint
    session_manager.store_trace(session_id, result.get("agent_trace", []))

    citations = [
        CitationModel(
            source=c.get("source", "unknown"),
            page=c.get("page", "?")
        )
        for c in result.get("citations", [])
    ]

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        citations=citations,
        citation_count=len(citations),
        session_id=session_id,
        status=result["status"],
        rewrite_count=result["rewrite_count"],
        generation_count=result["generation_count"],
        answer_confidence=result["answer_confidence"],
        hallucination_detected=result["hallucination_detected"],
        latency_seconds=result["latency_seconds"],
        is_followup=is_followup
    )


# ─── GET /health ───────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns system status: whether indexes are loaded,
    how many sessions are active, and how many documents are indexed.
    """
    from src.graph_nodes import _faiss_index, _chunks
    indexes_loaded = _faiss_index is not None
    total_chunks = len(_chunks) if _chunks else 0
    doc_names = set(c.get("source", "") for c in (_chunks or []))

    return HealthResponse(
        status="ok",
        indexes_loaded=indexes_loaded,
        active_sessions=session_manager.active_session_count(),
        documents_indexed=len(doc_names),
        total_chunks=total_chunks
    )


# ─── GET /documents ────────────────────────────────────

@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """
    Returns list of all indexed documents with their chunk counts.
    """
    from src.graph_nodes import _chunks
    chunks = _chunks or []

    # Count chunks per source file
    doc_chunks: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        doc_chunks[source] = doc_chunks.get(source, 0) + 1

    documents = [
        DocumentInfo(filename=name, chunk_count=count)
        for name, count in doc_chunks.items()
    ]

    return DocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=len(chunks)
    )


# ─── GET /agent/trace/{session_id} ────────────────────

@app.get("/agent/trace/{session_id}", response_model=TraceResponse)
async def get_agent_trace(session_id: str):
    """
    Returns the full agent reasoning trace for the last query
    made by this session. This is your unique agentic feature —
    exposes exactly what decisions the agent made and why.
    """
    trace = session_manager.get_trace(session_id)

    return TraceResponse(
        session_id=session_id,
        trace=trace,
        trace_formatted=format_trace_for_display(trace)
    )


# ─── DELETE /session/{session_id} ─────────────────────

@app.delete("/session/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    """
    Clears conversation memory for a session.
    Use this to start a fresh conversation.
    """
    deleted = session_manager.delete(session_id)
    return SessionDeleteResponse(
        session_id=session_id,
        deleted=deleted,
        message="Session cleared" if deleted else "Session not found"
    )


# ─── Entry point ──────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
