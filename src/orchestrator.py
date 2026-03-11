import time
from langgraph.graph import StateGraph, END
from src.agent_state import AgentState, create_initial_state
from src.graph_nodes import (
    plan_query_node,
    retrieve_node,
    grade_retrieval_node,
    rewrite_query_node,
    rerank_node,
    generate_answer_node,
    grade_answer_node,
    finalize_node,
    reload_indexes
)
from src.graph_edges import (
    route_after_retrieval,
    route_after_grading,
    route_after_answer_grading
)


def build_rag_graph():
    """
    Assembles and compiles the full agentic RAG graph.

    Graph structure:
      plan_query → retrieve →(conditional)→ grade_retrieval OR finalize
      grade_retrieval →(conditional)→ rewrite_query OR rerank
      rewrite_query → retrieve
      rerank → generate_answer → grade_answer
      grade_answer →(conditional)→ generate_answer OR finalize
      finalize → END
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("plan_query", plan_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("finalize", finalize_node)

    # Entry point
    graph.set_entry_point("plan_query")

    # Fixed edges
    graph.add_edge("plan_query", "retrieve")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")
    graph.add_edge("finalize", END)

    # Conditional edges
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"grade": "grade_retrieval", "no_documents": "finalize"}
    )
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {"rewrite": "rewrite_query", "rerank": "rerank"}
    )
    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grading,
        {"regenerate": "generate_answer", "finalize": "finalize"}
    )

    return graph.compile()


# Singleton graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        print("Building LangGraph pipeline...")
        _graph = build_rag_graph()
        print("Graph compiled")
    return _graph


def run_agentic_pipeline(
    question: str,
    session_id: str = "default",
    conversation_history: str = "",
    top_k: int = 5
) -> dict:
    """
    Main entry point. Runs the full agentic pipeline from question to answer.

    Returns:
        question, answer, citations, agent_trace, status,
        rewrite_count, generation_count, answer_confidence,
        hallucination_detected, latency_seconds
    """
    start = time.time()

    initial_state = create_initial_state(
        question=question,
        session_id=session_id,
        conversation_history=conversation_history,
        top_k=top_k
    )

    graph = get_graph()

    print(f"\n{'='*55}")
    print(f"PIPELINE START: '{question[:60]}'")
    print(f"{'='*55}")

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"Pipeline error: {e}")
        return {
            "question": question,
            "answer": f"Pipeline error: {str(e)}",
            "citations": [],
            "agent_trace": [],
            "status": "error",
            "rewrite_count": 0,
            "generation_count": 0,
            "answer_confidence": 0.0,
            "hallucination_detected": False,
            "latency_seconds": round(time.time() - start, 2)
        }

    return {
        "question": question,
        "answer": final_state.get("final_answer", ""),
        "citations": final_state.get("final_citations", []),
        "agent_trace": final_state.get("agent_trace", []),
        "status": final_state.get("status", "unknown"),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "generation_count": final_state.get("generation_count", 0),
        "answer_confidence": final_state.get("answer_confidence", 0.0),
        "hallucination_detected": final_state.get("hallucination_detected", False),
        "latency_seconds": round(time.time() - start, 2)
    }


def format_trace_for_display(agent_trace: list[dict]) -> str:
    """Formats agent trace into a readable string for UI display."""
    if not agent_trace:
        return "No trace available"

    lines = []
    for event in agent_trace:
        node = event.get("node", "Unknown")
        details = event.get("details", {})

        if node == "Query Planner":
            count = details.get("query_count", 1)
            tag = " (complex)" if details.get("is_complex") else ""
            lines.append(f"🧠 Query Planner{tag}: {count} quer{'ies' if count > 1 else 'y'} planned")
        elif node == "Retrieval":
            lines.append(f"🔍 Retrieval: {details.get('total_chunks', 0)} chunks found")
        elif node == "Retrieval Grader":
            passed = details.get("passed", 0)
            total = details.get("total", 0)
            tag = " ⚠️ rewrite triggered" if details.get("needs_rewrite") else " ✅"
            lines.append(f"✅ Retrieval Grader: {passed}/{total} relevant{tag}")
        elif node == "Query Rewriter":
            lines.append(f"✏️  Query Rewriter (attempt {details.get('attempt', 1)}): '{details.get('rewritten', '')[:50]}' [{details.get('strategy', '')}]")
        elif node == "Reranker":
            lines.append(f"📊 Reranker: {details.get('input_chunks', 0)} → top {details.get('output_chunks', 0)} chunks")
        elif node == "Answer Generator":
            lines.append(f"💬 Answer Generator (attempt {details.get('attempt', 1)}): {details.get('citation_count', 0)} citations, {details.get('latency', 0)}s")
        elif node == "Answer Grader":
            status = "✅ PASS" if details.get("passes") else "❌ FAIL"
            halluc = " 🚨 hallucination" if details.get("hallucination_detected") else ""
            lines.append(f"🔎 Answer Grader: {status} (confidence: {details.get('confidence', 0):.2f}){halluc}")
        elif node == "Finalizer":
            lines.append(f"🏁 Done: {details.get('status', '')} | rewrites={details.get('rewrite_count', 0)} | generations={details.get('generation_count', 0)}")

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_agentic_pipeline(
        question="What is the attention mechanism in transformers?",
        session_id="test_phase7"
    )

    print(f"\n{'='*55}")
    print("FINAL RESULT")
    print(f"{'='*55}")
    print(f"Status:        {result['status']}")
    print(f"Latency:       {result['latency_seconds']}s")
    print(f"Rewrites:      {result['rewrite_count']}")
    print(f"Generations:   {result['generation_count']}")
    print(f"Confidence:    {result['answer_confidence']:.2f}")
    print(f"Hallucination: {result['hallucination_detected']}")
    print(f"\nAnswer:\n{result['answer'][:400]}")
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  📄 {c.get('source')} — Page {c.get('page')}")
    print(f"\nAgent Trace:")
    print(format_trace_for_display(result["agent_trace"]))
