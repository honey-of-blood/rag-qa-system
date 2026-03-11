from src.agent_state import AgentState

MAX_REWRITE_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 2


def route_after_retrieval(state: AgentState) -> str:
    """
    After Retrieval node:
    - No documents in system → finalize immediately
    - Chunks retrieved → go to Retrieval Grader
    """
    status = state.get("status", "running")
    retrieved = state.get("retrieved_chunks", [])

    if status == "no_documents" or not retrieved:
        print("  → No documents found, routing to Finalizer")
        return "no_documents"

    print(f"  → {len(retrieved)} chunks retrieved, routing to Grader")
    return "grade"


def route_after_grading(state: AgentState) -> str:
    """
    After Retrieval Grader node:
    - Too few relevant chunks AND retries left → Query Rewriter
    - Otherwise → Reranker
    """
    needs_rewrite = state.get("needs_rewrite", False)
    rewrite_count = state.get("rewrite_count", 0)

    if needs_rewrite and rewrite_count < MAX_REWRITE_ATTEMPTS:
        print(f"  → Routing to Query Rewriter (attempt {rewrite_count + 1})")
        return "rewrite"

    if rewrite_count >= MAX_REWRITE_ATTEMPTS:
        print(f"  → Max rewrites reached, routing to Reranker")
    else:
        print(f"  → Chunks passed grading, routing to Reranker")
    return "rerank"


def route_after_answer_grading(state: AgentState) -> str:
    """
    After Answer Grader node:
    - Answer failed AND retries left → regenerate
    - Otherwise → finalize
    """
    should_regenerate = state.get("should_regenerate", False)
    generation_count = state.get("generation_count", 0)

    if should_regenerate and generation_count < MAX_GENERATION_ATTEMPTS:
        print(f"  → Answer failed, routing back to Generator (attempt {generation_count + 1})")
        return "regenerate"

    if generation_count >= MAX_GENERATION_ATTEMPTS:
        print(f"  → Max generations reached, routing to Finalizer")
    else:
        print(f"  → Answer passed grading, routing to Finalizer")
    return "finalize"


if __name__ == "__main__":
    print("Testing routing functions...")

    assert route_after_retrieval({"status": "running", "retrieved_chunks": [{"id": 1}]}) == "grade"
    assert route_after_retrieval({"status": "no_documents", "retrieved_chunks": []}) == "no_documents"
    assert route_after_retrieval({"status": "running", "retrieved_chunks": []}) == "no_documents"

    assert route_after_grading({"needs_rewrite": True, "rewrite_count": 0}) == "rewrite"
    assert route_after_grading({"needs_rewrite": True, "rewrite_count": 3}) == "rerank"
    assert route_after_grading({"needs_rewrite": False, "rewrite_count": 0}) == "rerank"

    assert route_after_answer_grading({"should_regenerate": True, "generation_count": 1}) == "regenerate"
    assert route_after_answer_grading({"should_regenerate": True, "generation_count": 2}) == "finalize"
    assert route_after_answer_grading({"should_regenerate": False, "generation_count": 1}) == "finalize"

    print("✅ All routing functions correct")
