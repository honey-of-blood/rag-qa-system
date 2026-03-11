from src.agent_state import AgentState


# Maximum retry limits
MAX_REWRITE_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 2


def route_after_grading(state: AgentState) -> str:
    """
    Called after Retrieval Grader node.

    Decision:
    - If chunks failed grading AND we haven't hit max retries
      → go to Query Rewriter
    - If chunks passed OR we've used all retries
      → go to Reranker
    """
    needs_rewrite = state.get("needs_rewrite", False)
    rewrite_count = state.get("rewrite_count", 0)

    if needs_rewrite and rewrite_count < MAX_REWRITE_ATTEMPTS:
        print(f"  → Routing to: Query Rewriter (attempt {rewrite_count + 1})")
        return "rewrite"
    else:
        if rewrite_count >= MAX_REWRITE_ATTEMPTS:
            print(f"  → Max rewrites reached ({MAX_REWRITE_ATTEMPTS}), routing to Reranker")
        else:
            print(f"  → Chunks passed grading, routing to Reranker")
        return "rerank"


def route_after_answer_grading(state: AgentState) -> str:
    """
    Called after Answer Grader node.

    Decision:
    - If answer failed AND we haven't hit max generations
      → go back to Answer Generator
    - If answer passed OR we've used all generation attempts
      → go to Finalizer
    """
    should_regenerate = state.get("should_regenerate", False)
    generation_count = state.get("generation_count", 0)

    if should_regenerate and generation_count < MAX_GENERATION_ATTEMPTS:
        print(
            f"  → Answer failed grading, routing back to Generator "
            f"(attempt {generation_count + 1})"
        )
        return "regenerate"
    else:
        if generation_count >= MAX_GENERATION_ATTEMPTS:
            print(f"  → Max generations reached ({MAX_GENERATION_ATTEMPTS}), routing to Finalizer")
        else:
            print(f"  → Answer passed grading, routing to Finalizer")
        return "finalize"


def route_after_retrieval(state: AgentState) -> str:
    """
    Called after Retrieval node.

    Decision:
    - If no documents in system → skip to finalizer
    - Otherwise → go to Retrieval Grader
    """
    status = state.get("status", "running")
    retrieved = state.get("retrieved_chunks", [])

    if status == "no_documents" or not retrieved:
        print("  → No documents found, routing to Finalizer")
        return "no_documents"
    else:
        print(f"  → {len(retrieved)} chunks retrieved, routing to Grader")
        return "grade"


if __name__ == "__main__":
    # Test routing logic
    print("Testing routing functions...")

    # Test route_after_grading
    state_needs_rewrite = {"needs_rewrite": True, "rewrite_count": 0}
    assert route_after_grading(state_needs_rewrite) == "rewrite"

    state_max_rewrites = {"needs_rewrite": True, "rewrite_count": 3}
    assert route_after_grading(state_max_rewrites) == "rerank"

    state_passed = {"needs_rewrite": False, "rewrite_count": 0}
    assert route_after_grading(state_passed) == "rerank"

    # Test route_after_answer_grading
    state_regen = {"should_regenerate": True, "generation_count": 1}
    assert route_after_answer_grading(state_regen) == "regenerate"

    state_max_gen = {"should_regenerate": True, "generation_count": 2}
    assert route_after_answer_grading(state_max_gen) == "finalize"

    state_passed_ans = {"should_regenerate": False, "generation_count": 1}
    assert route_after_answer_grading(state_passed_ans) == "finalize"

    print("✅ All routing functions working correctly")
