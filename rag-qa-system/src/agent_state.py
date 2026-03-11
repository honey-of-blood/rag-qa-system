from typing import TypedDict


class AgentState(TypedDict):
    """
    Shared state dict that flows through every node in the LangGraph pipeline.
    Every node reads from this and returns a partial dict of updates.
    """
    # Input
    original_question: str
    session_id: str
    conversation_history: str
    top_k: int

    # Query Planning
    sub_queries: list[str]
    current_query: str
    is_complex_question: bool

    # Retrieval
    retrieved_chunks: list[dict]

    # Retrieval Grading
    graded_chunks: list[dict]
    rejected_chunks: list[dict]
    retrieval_pass_rate: float
    needs_rewrite: bool

    # Query Rewriting
    rewrite_count: int
    rewrite_history: list[str]

    # Reranking
    reranked_chunks: list[dict]

    # Answer Generation
    answer: str
    citations: list[dict]
    generation_count: int
    strict_mode: bool

    # Answer Grading
    answer_passes: bool
    answer_confidence: float
    hallucination_detected: bool
    problematic_claims: list[str]
    answer_grade_reason: str
    should_regenerate: bool

    # Trace
    agent_trace: list[dict]

    # Final Output
    final_answer: str
    final_citations: list[dict]
    status: str


def create_initial_state(
    question: str,
    session_id: str = "default",
    conversation_history: str = "",
    top_k: int = 5
) -> AgentState:
    """Creates a fresh state for a new question with safe defaults."""
    return AgentState(
        original_question=question,
        session_id=session_id,
        conversation_history=conversation_history,
        top_k=top_k,
        sub_queries=[],
        current_query=question,
        is_complex_question=False,
        retrieved_chunks=[],
        graded_chunks=[],
        rejected_chunks=[],
        retrieval_pass_rate=0.0,
        needs_rewrite=False,
        rewrite_count=0,
        rewrite_history=[],
        reranked_chunks=[],
        answer="",
        citations=[],
        generation_count=0,
        strict_mode=False,
        answer_passes=False,
        answer_confidence=0.0,
        hallucination_detected=False,
        problematic_claims=[],
        answer_grade_reason="",
        should_regenerate=False,
        agent_trace=[],
        final_answer="",
        final_citations=[],
        status="running"
    )


def add_trace_event(
    state: AgentState,
    node_name: str,
    event: str,
    details: dict = None
) -> list[dict]:
    """
    Appends a trace event and returns the updated trace list.
    Call inside every node to build the reasoning trace.
    """
    import time
    trace = list(state.get("agent_trace", []))
    trace.append({
        "node": node_name,
        "event": event,
        "details": details or {},
        "timestamp": round(time.time(), 3)
    })
    return trace


if __name__ == "__main__":
    state = create_initial_state("What is multi-head attention?", "test_session")
    print("Initial state created successfully")
    print(f"Fields: {list(state.keys())}")
    trace = add_trace_event(state, "Test Node", "test_event", {"key": "value"})
    print(f"Trace entry added: {trace[0]}")
    print("agent_state.py OK")
