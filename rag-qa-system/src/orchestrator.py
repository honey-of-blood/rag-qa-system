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
    route_after_grading,
    route_after_retrieval,
    route_after_answer_grading
)


def build_rag_graph() -> StateGraph:
    """
    Assembles the full agentic RAG graph.

    Nodes:
        plan_query       → analyze and plan search queries
        retrieve         → hybrid search FAISS + BM25
        grade_retrieval  → filter irrelevant chunks
        rewrite_query    → improve failed queries
        rerank           → cross-encoder reranking
        generate_answer  → LLM answer with citations
        grade_answer     → validate answer quality
        finalize         → package final response

    Edges:
        plan_query → retrieve
        retrieve →(conditional)→ grade_retrieval OR finalize
        grade_retrieval →(conditional)→ rewrite_query OR rerank
        rewrite_query → retrieve
        rerank → generate_answer
        generate_answer → grade_answer
        grade_answer →(conditional)→ generate_answer OR finalize
        finalize → END
    """
    graph = StateGraph(AgentState)

    # ─── Add all nodes ────────────────────────────────
    graph.add_node("plan_query", plan_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("finalize", finalize_node)

    # ─── Set entry point ──────────────────────────────
    graph.set_entry_point("plan_query")

    # ─── Fixed edges ──────────────────────────────────
    graph.add_edge("plan_query", "retrieve")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")
    graph.add_edge("finalize", END)

    # ─── Conditional edges ────────────────────────────
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "grade": "grade_retrieval",
            "no_documents": "finalize"
        }
    )

    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "rewrite": "rewrite_query",
            "rerank": "rerank"
        }
    )

    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grading,
        {
            "regenerate": "generate_answer",
            "finalize": "finalize"
        }
    )

    return graph.compile()


# ─── Singleton graph instance ─────────────────────────
_graph = None


def get_graph():
    """
    Returns singleton compiled graph.
    Built once and reused for all requests.
    """
    global _graph
    if _graph is None:
        print("Building LangGraph agentic pipeline...")
        _graph = build_rag_graph()
        print("Graph compiled successfully")
    return _graph


def run_agentic_pipeline(
    question: str,
    session_id: str = "default",
    conversation_history: str = "",
    top_k: int = 5
) -> dict:
    """
    Main entry point for the agentic RAG pipeline.

    Runs the full graph from question to final answer.

    Returns dict with:
    - question: original question
    - answer: final answer text
    - citations: list of source citations
    - agent_trace: full reasoning trace
    - status: success / no_documents / completed_with_warnings
    - rewrite_count: how many query rewrites were needed
    - generation_count: how many answer generations were needed
    - answer_confidence: confidence score from answer grader
    - hallucination_detected: whether hallucination was found
    - latency_seconds: total pipeline time
    """
    start = time.time()

    # Create initial state
    initial_state = create_initial_state(
        question=question,
        session_id=session_id,
        conversation_history=conversation_history,
        top_k=top_k
    )

    # Get compiled graph
    graph = get_graph()

    # Run the graph
    print(f"\n{'='*55}")
    print(f"AGENTIC PIPELINE: '{question[:60]}'")
    print(f"{'='*55}")

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"Pipeline error: {e}")
        return {
            "question": question,
            "answer": f"Pipeline encountered an error: {str(e)}",
            "citations": [],
            "agent_trace": [],
            "status": "error",
            "rewrite_count": 0,
            "generation_count": 0,
            "answer_confidence": 0.0,
            "hallucination_detected": False,
            "latency_seconds": round(time.time() - start, 2)
        }

    latency = round(time.time() - start, 2)

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
        "latency_seconds": latency
    }


def format_trace_for_display(agent_trace: list[dict]) -> str:
    """
    Formats the agent trace into a readable string for UI display.
    """
    if not agent_trace:
        return "No trace available"

    lines = []
    for event in agent_trace:
        node = event.get("node", "Unknown")
        ev = event.get("event", "")
        details = event.get("details", {})

        # Format each node nicely
        if node == "Query Planner":
            count = details.get("query_count", 1)
            complex_tag = " (complex)" if details.get("is_complex") else ""
            lines.append(f"🧠 Query Planner{complex_tag}: {count} search quer{'ies' if count > 1 else 'y'} planned")

        elif node == "Retrieval":
            total = details.get("total_chunks", 0)
            queries = details.get("queries_run", 1)
            lines.append(f"🔍 Retrieval: {total} chunks found across {queries} quer{'ies' if queries > 1 else 'y'}")

        elif node == "Retrieval Grader":
            passed = details.get("passed", 0)
            total = details.get("total", 0)
            rate = details.get("pass_rate", 0)
            needs = details.get("needs_rewrite", False)
            tag = " ⚠️ triggering rewrite" if needs else " ✅"
            lines.append(f"✅ Retrieval Grader: {passed}/{total} chunks relevant ({rate:.0%}){tag}")

        elif node == "Query Rewriter":
            attempt = details.get("attempt", 1)
            strategy = details.get("strategy", "")
            rewritten = details.get("rewritten", "")
            lines.append(f"✏️  Query Rewriter (attempt {attempt}): '{rewritten[:50]}' [{strategy}]")

        elif node == "Reranker":
            inp = details.get("input_chunks", 0)
            out = details.get("output_chunks", 0)
            lines.append(f"📊 Reranker: {inp} → top {out} chunks selected")

        elif node == "Answer Generator":
            attempt = details.get("attempt", 1)
            cites = details.get("citation_count", 0)
            latency = details.get("latency", 0)
            lines.append(f"💬 Answer Generator (attempt {attempt}): {cites} citations, {latency}s")

        elif node == "Answer Grader":
            passes = details.get("passes", False)
            conf = details.get("confidence", 0)
            halluc = details.get("hallucination_detected", False)
            status = "✅ PASS" if passes else "❌ FAIL"
            halluc_tag = " 🚨 hallucination detected" if halluc else ""
            lines.append(f"🔎 Answer Grader: {status} (confidence: {conf:.2f}){halluc_tag}")

        elif node == "Finalizer":
            status = details.get("status", "")
            rewrites = details.get("rewrite_count", 0)
            gens = details.get("generation_count", 0)
            lines.append(f"🏁 Finalizer: {status} | rewrites={rewrites} | generations={gens}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing LangGraph Orchestrator...")

    result = run_agentic_pipeline(
        question="What is the attention mechanism in transformers?",
        session_id="test_phase7"
    )

    print(f"\n{'='*55}")
    print("FINAL RESULT")
    print(f"{'='*55}")
    print(f"Status:      {result['status']}")
    print(f"Latency:     {result['latency_seconds']}s")
    print(f"Rewrites:    {result['rewrite_count']}")
    print(f"Generations: {result['generation_count']}")
    print(f"Confidence:  {result['answer_confidence']:.2f}")
    print(f"Hallucination: {result['hallucination_detected']}")
    print(f"\nAnswer:\n{result['answer'][:400]}...")
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  📄 {c.get('source')} — Page {c.get('page')}")
    print(f"\nAgent Trace:")
    print(format_trace_for_display(result["agent_trace"]))
