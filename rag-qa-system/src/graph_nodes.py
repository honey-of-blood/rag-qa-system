from src.agent_state import AgentState, add_trace_event
from src.agents.query_planner import plan_query
from src.agents.retrieval_grader import grade_chunks
from src.agents.query_rewriter import rewrite_query
from src.agents.answer_grader import grade_answer
from src.hybrid_retriever import hybrid_search
from src.reranker import rerank_chunks
from src.answer_generator import generate_answer
from src.embedder import load_embedding_model
from src.vector_store import load_index
from src.bm25_retriever import load_bm25_index


# ─── Lazy-loaded model singletons ─────────────────────
_embed_model = None
_faiss_index = None
_chunks = None
_bm25_index = None


def get_models():
    """Loads all indexes once, returns cached instances on subsequent calls."""
    global _embed_model, _faiss_index, _chunks, _bm25_index

    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = load_embedding_model()

    if _faiss_index is None:
        try:
            print("Loading FAISS index...")
            _faiss_index, _chunks = load_index()
        except FileNotFoundError:
            print("No FAISS index found")
            _faiss_index = None
            _chunks = []

    if _bm25_index is None:
        try:
            print("Loading BM25 index...")
            _bm25_index = load_bm25_index()
        except FileNotFoundError:
            print("No BM25 index found")
            _bm25_index = None

    return _embed_model, _faiss_index, _chunks, _bm25_index


def reload_indexes():
    """Forces reload of FAISS and BM25 — call after new PDFs are ingested."""
    global _faiss_index, _chunks, _bm25_index
    _faiss_index = None
    _chunks = None
    _bm25_index = None
    get_models()
    print("Indexes reloaded")


# ─── Node 1: Query Planner ────────────────────────────

def plan_query_node(state: AgentState) -> dict:
    print(f"\n[NODE] Query Planner")
    question = state["original_question"]

    result = plan_query(question)

    trace = add_trace_event(
        state, "Query Planner", "query_analyzed",
        {
            "is_complex": result["is_complex"],
            "sub_queries": result["sub_queries"],
            "query_count": result["query_count"]
        }
    )

    print(f"  Is complex: {result['is_complex']}")
    print(f"  Sub-queries: {result['sub_queries']}")

    return {
        "sub_queries": result["sub_queries"],
        "current_query": result["sub_queries"][0],
        "is_complex_question": result["is_complex"],
        "agent_trace": trace
    }


# ─── Node 2: Retrieval ────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    print(f"\n[NODE] Retrieval")

    embed_model, faiss_index, chunks, bm25_index = get_models()

    if faiss_index is None or bm25_index is None:
        trace = add_trace_event(state, "Retrieval", "no_index", {"error": "No indexes found"})
        return {"retrieved_chunks": [], "agent_trace": trace, "status": "no_documents"}

    sub_queries = state["sub_queries"]
    top_k = state.get("top_k", 5)
    all_chunks = []
    seen_ids = set()

    for query in sub_queries:
        print(f"  Searching: '{query[:60]}'")
        results = hybrid_search(
            query, faiss_index, bm25_index, chunks, embed_model, top_k=top_k * 2
        )
        for chunk in results:
            cid = chunk.get("chunk_id", id(chunk))
            if cid not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(cid)

    print(f"  Total unique chunks: {len(all_chunks)}")

    trace = add_trace_event(
        state, "Retrieval", "chunks_retrieved",
        {"queries_run": len(sub_queries), "total_chunks": len(all_chunks)}
    )

    return {"retrieved_chunks": all_chunks, "agent_trace": trace}


# ─── Node 3: Retrieval Grader ─────────────────────────

def grade_retrieval_node(state: AgentState) -> dict:
    print(f"\n[NODE] Retrieval Grader")

    question = state["original_question"]
    chunks = state["retrieved_chunks"]

    if not chunks:
        trace = add_trace_event(state, "Retrieval Grader", "no_chunks", {})
        return {
            "graded_chunks": [], "needs_rewrite": True,
            "retrieval_pass_rate": 0.0, "agent_trace": trace
        }

    result = grade_chunks(question, chunks)

    trace = add_trace_event(
        state, "Retrieval Grader", "chunks_graded",
        {
            "total": result["total_graded"],
            "passed": result["passed_count"],
            "failed": result["failed_count"],
            "pass_rate": round(result["pass_rate"], 2),
            "needs_rewrite": result["needs_rewrite"]
        }
    )

    print(f"  Passed: {result['passed_count']}/{result['total_graded']}")
    print(f"  Needs rewrite: {result['needs_rewrite']}")

    return {
        "graded_chunks": result["relevant_chunks"],
        "rejected_chunks": result["rejected_chunks"],
        "retrieval_pass_rate": result["pass_rate"],
        "needs_rewrite": result["needs_rewrite"],
        "agent_trace": trace
    }


# ─── Node 4: Query Rewriter ───────────────────────────

def rewrite_query_node(state: AgentState) -> dict:
    print(f"\n[NODE] Query Rewriter")

    question = state["original_question"]
    rewrite_count = state.get("rewrite_count", 0) + 1
    conversation_history = state.get("conversation_history", "")
    rewrite_history = list(state.get("rewrite_history", []))

    result = rewrite_query(
        original_query=question,
        conversation_context=conversation_history,
        attempt_number=rewrite_count
    )

    new_query = result["rewritten_query"]
    rewrite_history.append(new_query)

    print(f"  Attempt {rewrite_count}: '{new_query}'")
    print(f"  Strategy: {result['strategy']}")

    trace = add_trace_event(
        state, "Query Rewriter", "query_rewritten",
        {
            "attempt": rewrite_count,
            "original": question,
            "rewritten": new_query,
            "strategy": result["strategy"]
        }
    )

    return {
        "current_query": new_query,
        "sub_queries": [new_query],
        "rewrite_count": rewrite_count,
        "rewrite_history": rewrite_history,
        "needs_rewrite": False,
        "agent_trace": trace
    }


# ─── Node 5: Reranker ─────────────────────────────────

def rerank_node(state: AgentState) -> dict:
    print(f"\n[NODE] Reranker")

    question = state["original_question"]
    chunks = state["graded_chunks"]
    top_k = state.get("top_k", 5)

    if not chunks:
        trace = add_trace_event(state, "Reranker", "no_chunks", {})
        return {"reranked_chunks": [], "agent_trace": trace}

    reranked = rerank_chunks(question, chunks, top_k=top_k)
    print(f"  {len(chunks)} → top {len(reranked)} chunks")

    trace = add_trace_event(
        state, "Reranker", "chunks_reranked",
        {
            "input_chunks": len(chunks),
            "output_chunks": len(reranked),
            "top_score": round(reranked[0].get("rerank_score", 0), 3) if reranked else 0
        }
    )

    return {"reranked_chunks": reranked, "agent_trace": trace}


# ─── Node 6: Answer Generator ─────────────────────────

def generate_answer_node(state: AgentState) -> dict:
    print(f"\n[NODE] Answer Generator")

    question = state["original_question"]
    chunks = state["reranked_chunks"]
    generation_count = state.get("generation_count", 0) + 1
    conversation_history = state.get("conversation_history", "")
    strict_mode = generation_count > 1  # stricter prompt on retry

    if not chunks:
        trace = add_trace_event(
            state, "Answer Generator", "no_context",
            {"generation_attempt": generation_count}
        )
        return {
            "answer": "I could not find sufficient information in the provided documents to answer this question.",
            "citations": [],
            "generation_count": generation_count,
            "agent_trace": trace,
            "status": "no_context"
        }

    if strict_mode:
        print(f"  Regeneration attempt {generation_count} — using strict prompt")

    result = generate_answer(
        query=question,
        reranked_chunks=chunks,
        conversation_history=conversation_history,
        strict_mode=strict_mode
    )

    print(f"  Answer: {result['answer'][:80]}...")
    print(f"  Citations: {result['citation_count']} | Latency: {result['latency_seconds']}s")

    trace = add_trace_event(
        state, "Answer Generator", "answer_generated",
        {
            "attempt": generation_count,
            "citation_count": result.get("citation_count", 0),
            "latency": result.get("latency_seconds", 0),
            "chunks_used": result.get("chunks_used", 0)
        }
    )

    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "generation_count": generation_count,
        "agent_trace": trace
    }


# ─── Node 7: Answer Grader ────────────────────────────

def grade_answer_node(state: AgentState) -> dict:
    print(f"\n[NODE] Answer Grader")

    question = state["original_question"]
    answer = state["answer"]
    chunks = state["reranked_chunks"]

    result = grade_answer(question, answer, chunks)

    print(f"  Passes: {result['passes']} | Confidence: {result['confidence']:.2f}")
    print(f"  Hallucination: {result['hallucination_detected']}")

    trace = add_trace_event(
        state, "Answer Grader", "answer_graded",
        {
            "passes": result["passes"],
            "confidence": round(result["confidence"], 2),
            "hallucination_detected": result["hallucination_detected"],
            "should_regenerate": result["should_regenerate"],
            "reason": result["reason"]
        }
    )

    final_answer = answer if result["passes"] else ""
    final_citations = state["citations"] if result["passes"] else []

    return {
        "answer_passes": result["passes"],
        "answer_confidence": result["confidence"],
        "hallucination_detected": result["hallucination_detected"],
        "problematic_claims": result["problematic_claims"],
        "answer_grade_reason": result["reason"],
        "should_regenerate": result["should_regenerate"],
        "final_answer": final_answer,
        "final_citations": final_citations,
        "agent_trace": trace
    }


# ─── Node 8: Finalizer ────────────────────────────────

def finalize_node(state: AgentState) -> dict:
    print(f"\n[NODE] Finalizer")

    final_answer = state.get("final_answer", "")
    final_citations = state.get("final_citations", [])

    # If no passing answer, use best attempt rather than returning empty
    if not final_answer:
        final_answer = state.get("answer", "")
        final_citations = state.get("citations", [])
        status = "completed_with_warnings"
    else:
        status = "success"

    trace = add_trace_event(
        state, "Finalizer", "pipeline_complete",
        {
            "status": status,
            "rewrite_count": state.get("rewrite_count", 0),
            "generation_count": state.get("generation_count", 0),
            "final_confidence": state.get("answer_confidence", 0)
        }
    )

    print(f"  Status: {status}")
    print(f"  Rewrites: {state.get('rewrite_count', 0)} | Generations: {state.get('generation_count', 0)}")

    return {
        "final_answer": final_answer,
        "final_citations": final_citations,
        "status": status,
        "agent_trace": trace
    }
