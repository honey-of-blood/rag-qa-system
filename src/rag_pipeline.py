def run(
    self,
    question: str,
    memory,
    top_k: int = 5
) -> dict:
    """
    Runs the full AGENTIC RAG pipeline for a question.
    Replaces the old straight pipeline with LangGraph orchestration.
    """
    from src.orchestrator import run_agentic_pipeline

    if not self.faiss_index or not self.bm25_index:
        return {
            "query": question,
            "answer": "No documents uploaded yet. Please upload PDFs first.",
            "citations": [],
            "citation_count": 0,
            "chunks_used": 0,
            "latency_seconds": 0.0,
            "status": "no_documents",
            "is_followup": False,
            "agent_trace": [],
            "rewrite_count": 0,
            "generation_count": 0,
            "answer_confidence": 0.0,
            "hallucination_detected": False
        }

    # Get conversation history from memory
    conversation_history = ""
    if memory and len(memory.turns) > 0:
        conversation_history = memory.format_history_for_prompt()

    # Run agentic pipeline
    result = run_agentic_pipeline(
        question=question,
        session_id="pipeline",
        conversation_history=conversation_history,
        top_k=top_k
    )

    # Store in memory
    if memory:
        memory.add_turn(
            question,
            result["answer"],
            citations=result.get("citations", [])
        )

    # Map to format expected by FastAPI
    return {
        "query": question,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "citation_count": len(result.get("citations", [])),
        "chunks_used": top_k,
        "latency_seconds": result["latency_seconds"],
        "status": result["status"],
        "is_followup": memory.is_followup_question(question) if memory else False,
        "agent_trace": result.get("agent_trace", []),
        "rewrite_count": result.get("rewrite_count", 0),
        "generation_count": result.get("generation_count", 0),
        "answer_confidence": result.get("answer_confidence", 0.0),
        "hallucination_detected": result.get("hallucination_detected", False)
    }
