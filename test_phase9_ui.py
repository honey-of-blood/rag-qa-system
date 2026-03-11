import requests
import time

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:7860"


def test_backend_reachable():
    print("\n[1] Backend reachable from frontend perspective...")
    r = requests.get(f"{BACKEND_URL}/health", timeout=10)
    assert r.status_code == 200
    print(f"  ✅ Backend up: {r.json()['status']}")


def test_frontend_reachable():
    print("\n[2] Frontend reachable...")
    r = requests.get(FRONTEND_URL, timeout=10)
    assert r.status_code == 200
    print(f"  ✅ Gradio UI serving at {FRONTEND_URL}")


def test_query_flow():
    print("\n[3] Full query flow through backend...")
    r = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": "What is the attention mechanism?",
            "session_id": "phase9_test",
            "top_k": 5
        },
        timeout=120
    )
    assert r.status_code == 200
    data = r.json()

    # Verify all fields the frontend needs are present
    required_fields = [
        "answer", "citations", "agent_trace", "status",
        "latency_seconds", "answer_confidence", "rewrite_count",
        "generation_count", "hallucination_detected", "is_followup"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    print(f"  ✅ All required response fields present")
    print(f"     status: {data['status']}")
    print(f"     latency: {data['latency_seconds']}s")
    print(f"     trace events: {len(data['agent_trace'])}")
    print(f"     citations: {data['citation_count']}")
    print(f"     confidence: {data['answer_confidence']:.2f}")


def test_trace_visible():
    print("\n[4] Agent trace visible after query...")
    r = requests.get(
        f"{BACKEND_URL}/agent/trace/phase9_test",
        timeout=10
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["trace"]) > 0
    assert len(data["trace_formatted"]) > 0
    print(f"  ✅ Trace has {len(data['trace'])} events")
    print(f"\n  Trace preview:")
    for line in data["trace_formatted"].split("\n"):
        print(f"    {line}")


def test_session_isolation():
    print("\n[5] Session isolation — two sessions don't share memory...")
    session_a = "phase9_session_a"
    session_b = "phase9_session_b"

    requests.post(f"{BACKEND_URL}/query", json={
        "question": "What is multi-head attention?",
        "session_id": session_a, "top_k": 5
    }, timeout=120)

    r_b = requests.post(f"{BACKEND_URL}/query", json={
        "question": "What is positional encoding?",
        "session_id": session_b, "top_k": 5
    }, timeout=120)

    assert r_b.status_code == 200
    assert r_b.json()["is_followup"] == False

    # Clean up
    requests.delete(f"{BACKEND_URL}/session/{session_a}")
    requests.delete(f"{BACKEND_URL}/session/{session_b}")

    print(f"  ✅ Sessions are isolated correctly")


def test_clear_chat():
    print("\n[6] Clear chat / session delete...")
    r = requests.delete(
        f"{BACKEND_URL}/session/phase9_test",
        timeout=10
    )
    assert r.status_code == 200
    assert r.json()["deleted"] == True
    print(f"  ✅ Session cleared successfully")


def run_phase9_test():
    print("=" * 55)
    print("PHASE 9 TEST: Gradio Frontend")
    print("=" * 55)
    print("Make sure BOTH app.py and frontend.py are running.")

    start = time.time()

    test_backend_reachable()
    test_frontend_reachable()
    test_query_flow()
    test_trace_visible()
    test_session_isolation()
    test_clear_chat()

    elapsed = round(time.time() - start, 1)

    print("\n" + "=" * 55)
    print("PHASE 9 COMPLETE")
    print(f"All 6 tests passed in {elapsed}s")
    print("Ready for Phase 10: Docker")
    print("=" * 55)


if __name__ == "__main__":
    run_phase9_test()
