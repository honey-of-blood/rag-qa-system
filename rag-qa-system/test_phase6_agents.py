from src.agents.query_planner import plan_query
from src.agents.retrieval_grader import grade_chunks
from src.agents.query_rewriter import rewrite_query, rewrite_with_retries
from src.agents.answer_grader import grade_answer
import time


def test_query_planner():
    print("\n" + "=" * 55)
    print("TEST 1: Query Planner Agent")
    print("=" * 55)

    simple = plan_query("What is the attention mechanism?")
    assert simple["is_complex"] == False
    assert len(simple["sub_queries"]) == 1
    print(f"✅ Simple query passed through: {simple['sub_queries']}")

    complex_q = plan_query(
        "Compare how transformers and RNNs handle long sequences "
        "and explain the performance tradeoffs"
    )
    assert complex_q["is_complex"] == True
    assert len(complex_q["sub_queries"]) >= 2
    print(f"✅ Complex query split into {complex_q['query_count']} sub-queries:")
    for q in complex_q["sub_queries"]:
        print(f"   - {q}")


def test_retrieval_grader():
    print("\n" + "=" * 55)
    print("TEST 2: Retrieval Grader Agent")
    print("=" * 55)

    query = "What is multi-head attention?"
    chunks = [
        {
            "chunk_id": 0,
            "text": "Multi-head attention allows the model to attend to information from different representation subspaces.",
            "source": "paper.pdf", "page": 4
        },
        {
            "chunk_id": 1,
            "text": "We used Adam optimizer with learning rate 0.001.",
            "source": "paper.pdf", "page": 7
        },
        {
            "chunk_id": 2,
            "text": "The scaled dot-product attention computes queries, keys and values.",
            "source": "paper.pdf", "page": 4
        }
    ]

    result = grade_chunks(query, chunks)
    assert result["total_graded"] == 3
    assert result["passed_count"] >= 1
    print(f"✅ Graded {result['total_graded']} chunks")
    print(f"   Passed: {result['passed_count']}")
    print(f"   Failed: {result['failed_count']}")
    print(f"   Needs rewrite: {result['needs_rewrite']}")


def test_query_rewriter():
    print("\n" + "=" * 55)
    print("TEST 3: Query Rewriter Agent")
    print("=" * 55)

    result = rewrite_query(
        "How does it remember things?",
        conversation_context="User was asking about RNN hidden states"
    )
    assert result["rewritten_query"] != "How does it remember things?"
    print(f"✅ Query rewritten successfully")
    print(f"   Original:  '{result['original_query']}'")
    print(f"   Rewritten: '{result['rewritten_query']}'")
    print(f"   Strategy:  {result['strategy']}")

    rewrites = rewrite_with_retries(
        "Tell me about it",
        max_attempts=3
    )
    assert len(rewrites) == 3
    print(f"\n✅ Generated {len(rewrites)} different rewrites:")
    for r in rewrites:
        print(f"   Attempt {r['attempt_number']}: '{r['rewritten_query']}'")


def test_answer_grader():
    print("\n" + "=" * 55)
    print("TEST 4: Answer Grader Agent")
    print("=" * 55)

    context = [
        {
            "text": "Multi-head attention allows joint attention to different representation subspaces.",
            "source": "paper.pdf", "page": 4
        }
    ]

    # Test 1: Good answer should pass
    good = grade_answer(
        "What is multi-head attention?",
        "Multi-head attention allows the model to attend to different representation subspaces simultaneously.",
        context
    )
    assert good["passes"] == True
    print(f"✅ Good answer correctly passed: confidence {good['confidence']:.2f}")

    # Test 2: Hallucinated answer should fail
    hallucinated = grade_answer(
        "What is multi-head attention?",
        "Multi-head attention uses 512 heads by default and was invented in 2015 by Google Brain.",
        context
    )
    assert hallucinated["hallucination_detected"] == True
    print(f"✅ Hallucinated answer correctly failed")
    print(f"   Problematic claims: {hallucinated['problematic_claims']}")


def run_phase6_test():
    print("=" * 55)
    print("PHASE 6 TEST: All Four Agents")
    print("=" * 55)

    start = time.time()

    test_query_planner()
    test_retrieval_grader()
    test_query_rewriter()
    test_answer_grader()

    elapsed = round(time.time() - start, 1)

    print("\n" + "=" * 55)
    print("PHASE 6 COMPLETE")
    print(f"All 4 agents working correctly in {elapsed}s")
    print("Ready for Phase 7: LangGraph Orchestration")
    print("=" * 55)


if __name__ == "__main__":
    run_phase6_test()
