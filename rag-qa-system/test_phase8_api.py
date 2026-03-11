import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n[1] GET /health")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"
    print(f"  ✅ Health check passed")
    print(f"     indexes_loaded: {data['indexes_loaded']}")
    print(f"     active_sessions: {data['active_sessions']}")
    print(f"     documents_indexed: {data['documents_indexed']}")
    print(f"     total_chunks: {data['total_chunks']}")


def test_documents():
    print("\n[2] GET /documents")
    r = requests.get(f"{BASE_URL}/documents")
    assert r.status_code == 200
    data = r.json()
    print(f"  ✅ Documents endpoint working")
    print(f"     total_documents: {data['total_documents']}")
    print(f"     total_chunks: {data['total_chunks']}")
    for doc in data["documents"]:
        print(f"     - {doc['filename']}: {doc['chunk_count']} chunks")


def test_query():
    print("\n[3] POST /query")
    payload = {
        "question": "What is the attention mechanism?",
        "session_id": "test_session_phase8",
        "top_k": 5
    }
    r = requests.post(f"{BASE_URL}/query", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()

    assert "answer" in data
    assert "citations" in data
    assert "status" in data
    assert "answer_confidence" in data
    assert "rewrite_count" in data
    assert "generation_count" in data
    assert "hallucination_detected" in data
    assert "latency_seconds" in data

    print(f"  ✅ Query endpoint working")
    print(f"     status: {data['status']}")
    print(f"     latency: {data['latency_seconds']}s")
    print(f"     rewrite_count: {data['rewrite_count']}")
    print(f"     generation_count: {data['generation_count']}")
    print(f"     confidence: {data['answer_confidence']:.2f}")
    print(f"     hallucination_detected: {data['hallucination_detected']}")
    print(f"     citations: {data['citation_count']}")
    print(f"     answer preview: {data['answer'][:120]}...")
    return data


def test_agent_trace(session_id: str):
    print(f"\n[4] GET /agent/trace/{session_id}")
    r = requests.get(f"{BASE_URL}/agent/trace/{session_id}")
    assert r.status_code == 200
    data = r.json()
    assert "trace" in data
    assert "trace_formatted" in data
    print(f"  ✅ Agent trace endpoint working")
    print(f"     trace events: {len(data['trace'])}")
    print(f"\n     Formatted trace:")
    for line in data["trace_formatted"].split("\n"):
        print(f"       {line}")


def test_multiturn():
    print("\n[5] Multi-turn conversation via API")
    session_id = "test_multiturn_phase8"

    questions = [
        "What is multi-head attention?",
        "How many heads does it use?",
        "Why are they run in parallel?"
    ]

    for i, q in enumerate(questions):
        r = requests.post(f"{BASE_URL}/query", json={
            "question": q,
            "session_id": session_id,
            "top_k": 5
        })
        assert r.status_code == 200
        data = r.json()
        print(f"  Turn {i+1}: '{q}'")
        print(f"    is_followup: {data['is_followup']} | status: {data['status']}")

    print("  ✅ Multi-turn conversation working via API")


def test_session_delete():
    print("\n[6] DELETE /session/{session_id}")
    session_id = "test_session_phase8"

    r = requests.delete(f"{BASE_URL}/session/{session_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["deleted"] == True
    print(f"  ✅ Session deleted: {data['message']}")

    # Deleting non-existent session should still return 200
    r2 = requests.delete(f"{BASE_URL}/session/does_not_exist")
    assert r2.status_code == 200
    assert r2.json()["deleted"] == False
    print(f"  ✅ Non-existent session handled gracefully")


def test_invalid_requests():
    print("\n[7] Invalid request handling")

    # Empty question
    r = requests.post(f"{BASE_URL}/query", json={
        "question": "",
        "session_id": "test"
    })
    assert r.status_code == 422
    print(f"  ✅ Empty question rejected (422)")

    # Non-PDF upload
    r2 = requests.post(
        f"{BASE_URL}/upload",
        files=[("files", ("test.txt", b"not a pdf", "text/plain"))]
    )
    assert r2.status_code == 400
    print(f"  ✅ Non-PDF upload rejected (400)")


def run_phase8_test():
    print("=" * 55)
    print("PHASE 8 TEST: FastAPI Backend")
    print("=" * 55)
    print(f"Testing against: {BASE_URL}")
    print("Make sure app.py is running before starting this test.")

    start = time.time()

    test_health()
    test_documents()
    query_result = test_query()
    test_agent_trace("test_session_phase8")
    test_multiturn()
    test_session_delete()
    test_invalid_requests()

    elapsed = round(time.time() - start, 1)

    print("\n" + "=" * 55)
    print("PHASE 8 COMPLETE")
    print(f"All 7 API tests passed in {elapsed}s")
    print("Ready for Phase 9: Gradio Frontend")
    print("=" * 55)


if __name__ == "__main__":
    run_phase8_test()
