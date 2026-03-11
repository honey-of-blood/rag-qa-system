from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    Shared state object that flows through every node
    in the LangGraph agentic pipeline.

    Every node reads from this dict and writes updates back.
    LangGraph merges the updates automatically.
    """

    # ─── Input ────────────────────────────────────────
    original_question: str
    session_id: str
    conversation_history: str
    top_k: int

    # ─── Query Planning ───────────────────────────────
    sub_queries: list[str]
    current_query: str
    is_complex_question: bool

    # ─── Retrieval ────────────────────────────────────
    retrieved_chunks: list[dict]

    # ─── Retrieval Grading ────────────────────────────
    graded_chunks: list[dict]
    rejected_chunks: list[dict]
    retrieval_pass_rate: float
    needs_rewrite: bool

    # ─── Query Rewriting ──────────────────────────────
    rewrite_count: int
    rewrite_history: list[str]

    # ─── Reranking ────────────────────────────────────
    reranked_chunks: list[dict]

    # ─── Answer Generation ────────────────────────────
    answer: str
    citations: list[dict]
    generation_count: int

    # ─── Answer Grading ───────────────────────────────
    answer_passes: bool
    answer_confidence: float
    hallucination_detected: bool
    problematic_claims: list[str]
    answer_grade_reason: str
    should_regenerate: bool

    # ─── Trace (for UI display) ───────────────────────
    agent_trace: list[dict]

    # ─── Final Output ─────────────────────────────────
    final_answer: str
    final_citations: list[dict]
    status: str
    total_latency: float


def create_initial_state(
    question: str,
    session_id: str = "default",
    conversation_history: str = "",
    top_k: int = 5
) -> AgentState:
    """
    Creates a fresh initial state for a new question.
    All fields set to safe defaults.
    """
    return AgentState(
        # Input
        original_question=question,
        session_id=session_id,
        conversation_history=conversation_history,
        top_k=top_k,

        # Query Planning
        sub_queries=[],
        current_query=question,
        is_complex_question=False,

        # Retrieval
        retrieved_chunks=[],

        # Retrieval Grading
        graded_chunks=[],
        rejected_chunks=[],
        retrieval_pass_rate=0.0,
        needs_rewrite=False,

        # Query Rewriting
        rewrite_count=0,
        rewrite_history=[],

        # Reranking
        reranked_chunks=[],

        # Answer Generation
        answer="",
        citations=[],
        generation_count=0,

        # Answer Grading
        answer_passes=False,
        answer_confidence=0.0,
        hallucination_detected=False,
        problematic_claims=[],
        answer_grade_reason="",
        should_regenerate=False,

        # Trace
        agent_trace=[],

        # Final Output
        final_answer="",
        final_citations=[],
        status="running",
        total_latency=0.0
    )


def add_trace_event(
    state: AgentState,
    node_name: str,
    event: str,
    details: dict = None
) -> list[dict]:
    """
    Adds a trace event to the agent trace log.
    Returns updated trace list.
    Call this inside every node to build the reasoning trace.
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
    state = create_initial_state(
        question="What is the attention mechanism?",
        session_id="test_123"
    )
    print("Initial state created:")
    for key, value in state.items():
        print(f"  {key}: {repr(value)[:60]}")
