import time
from src.orchestrator import run_agentic_pipeline, format_trace_for_display, get_graph
from src.graph_edges import route_after_retrieval, route_after_grading, route_after_answer_grading
from src.memory import ConversationMemory


def test_graph_builds():
    print("\n[1] Graph compilation...")
    graph = get_graph()
    assert graph is not None
    print("  ✅ LangGraph compiled successfully")


def test_routing_logic():
    print("\n[2] Routing logic...")

    assert route_after_retrieval({"status": "running", "retrieved_chunks": [{"id": 1}]}) == "grade"
    assert route_after_retrieval({"status": "no_documents", "retrieved_chunks": []}) == "no_documents"
    assert route_after_retrieval({"status": "running", "retrieved_chunks": []}) == "no_documents"
    print("  ✅ Retrieval routing correct")

    assert route_after_grading({"needs_rewrite": True, "rewrite_count": 0}) == "rewrite"
    assert route_after_grading({"needs_rewrite": True, "rewrite_count": 3}) == "rerank"
    assert route_after_grading({"needs_rewrite": False, "rewrite_count": 0}) == "rerank"
    print("  ✅ Grading routing correct")

    assert route_after_answer_grading({"should_regenerate": True, "generation_count": 1}) == "regenerate"
    assert route_after_answer_grading({"should_regenerate": True, "generation_count": 2}) == "finalize"
    assert route_after_answer_grading({"should_regenerate": False, "generation_count": 1}) == "finalize"
    print("  ✅ Answer grading routing correct")


def test_simple_question():
    print("\n[3] Simple question pipeline...")
    result = run_agentic_pipeline(
        question="What is the attention mechanism?",
        session_id="test_simple"
    )
    assert "answer" in result
    assert "agent_trace" in result
    assert len(result["agent_trace"]) > 0
    assert result["status"] in ["success", "completed_with_warnings", "no_documents", "no_context", "error"]
    print(f"  ✅ Completed — status: {result['status']}")
    print(f"     Latency: {result['latency_seconds']}s | Rewrites: {result['rewrite_count']} | Generations: {result['generation_count']}")
    print(f"     Confidence: {result['answer_confidence']:.2f} | Hallucination: {result['hallucination_detected']}")
    return result


def test_complex_question():
    print("\n[4] Complex question pipeline...")
    result = run_agentic_pipeline(
        question="Compare how attention works in transformers vs RNNs and explain the tradeoffs",
        session_id="test_complex"
    )
    assert "answer" in result
    print(f"  ✅ Completed — status: {result['status']} | Latency: {result['latency_seconds']}s")
    return result


def test_trace_formatting():
    print("\n[5] Trace formatting...")
    result = run_agentic_pipeline(
        question="What is positional encoding?",
        session_id="test_trace"
    )
    trace_text = format_trace_for_display(result["agent_trace"])
    assert len(trace_text) > 0
    print(f"  ✅ Trace formatted")
    print(f"\n  Agent Reasoning Trace:")
    for line in trace_text.split("\n"):
        print(f"    {line}")


def test_multiturn_memory():
    print("\n[6] Multi-turn with ConversationMemory...")
    memory = ConversationMemory(max_turns=4)

    questions = [
        "What is multi-head attention?",
        "How many heads does it typically use?",
        "Why are they run in parallel?"
    ]

    for i, q in enumerate(questions):
        history = memory.format_history_for_prompt() if memory.turns else ""
        result = run_agentic_pipeline(
            question=q,
            session_id="test_multiturn",
            conversation_history=history
        )
        memory.add_turn(q, result["answer"], citations=result["citations"])
        print(f"  Turn {i+1}: '{q}'")
        print(f"    Status: {result['status']} | Memory turns: {len(memory)}")

    assert len(memory) == 3
    assert memory.is_followup_question("Why does it do that?")
    print("  ✅ Multi-turn conversation working")


def run_phase7_test():
    print("=" * 55)
    print("PHASE 7 TEST: LangGraph Orchestration")
    print("=" * 55)

    start = time.time()

    test_graph_builds()
    test_routing_logic()
    test_simple_question()
    test_complex_question()
    test_trace_formatting()
    test_multiturn_memory()

    elapsed = round(time.time() - start, 1)

    print("\n" + "=" * 55)
    print("PHASE 7 COMPLETE")
    print(f"All tests passed in {elapsed}s")
    print("Ready for Phase 8: FastAPI Backend")
    print("=" * 55)


if __name__ == "__main__":
    run_phase7_test()
