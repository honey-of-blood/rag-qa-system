import requests
import time
import sys

# Replace these with your actual Render URLs after deployment
BACKEND_URL = "https://rag-backend-a8vo.onrender.com"
FRONTEND_URL = "https://rag-frontend-xhio.onrender.com"


def test_backend_live():
    print("\n[1] Backend live on Render...")
    r = requests.get(f"{BACKEND_URL}/health", timeout=30)
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"  ✅ Backend live: {BACKEND_URL}")
    print(f"     indexes_loaded: {data['indexes_loaded']}")
    print(f"     total_chunks: {data['total_chunks']}")


def test_frontend_live():
    print("\n[2] Frontend live on Render...")
    r = requests.get(FRONTEND_URL, timeout=60)
    assert r.status_code == 200
    print(f"  ✅ Frontend live: {FRONTEND_URL}")

def test_query_live():
    print("\n[3] End-to-end query on live deployment...")
    r = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": "What is the attention mechanism?",
            "session_id": "render_test",
            "top_k": 5
        },
        timeout=120
    )
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    data = r.json()
    assert "answer" in data
    assert "agent_trace" in data
    assert len(data["agent_trace"]) > 0
    print(f"  ✅ Query completed on live deployment")
    print(f"     status: {data['status']}")
    print(f"     latency: {data['latency_seconds']}s")
    print(f"     confidence: {data['answer_confidence']:.2f}")
    print(f"     trace events: {len(data['agent_trace'])}")
    print(f"     answer preview: {data['answer'][:120]}...")


def test_agent_trace_live():
    print("\n[4] Agent trace endpoint live...")
    r = requests.get(
        f"{BACKEND_URL}/agent/trace/render_test",
        timeout=30
    )
    assert r.status_code == 200
    data = r.json()
    assert "trace_formatted" in data
    print(f"  ✅ Agent trace endpoint working on live deployment")
    print(f"\n  Live trace:")
    for line in data["trace_formatted"].split("\n"):
        print(f"    {line}")


def test_documents_endpoint_live():
    print("\n[5] Documents endpoint live...")
    r = requests.get(f"{BACKEND_URL}/documents", timeout=30)
    assert r.status_code == 200
    data = r.json()
    print(f"  ✅ Documents endpoint live")
    print(f"     total_documents: {data['total_documents']}")
    print(f"     total_chunks: {data['total_chunks']}")


def test_autodocs_live():
    print("\n[6] Auto-generated API docs live...")
    r = requests.get(f"{BACKEND_URL}/docs", timeout=30)
    assert r.status_code == 200
    print(f"  ✅ API docs live at {BACKEND_URL}/docs")


def run_phase11_test():
    print("=" * 55)
    print("PHASE 11 TEST: Render Deployment")
    print("=" * 55)
    print(f"Backend:  {BACKEND_URL}")
    print(f"Frontend: {FRONTEND_URL}")
    print("Note: Render free tier may be slow on first request")
    print("(services sleep after 15min inactivity — first request wakes them)")

    start = time.time()

    test_backend_live()
    test_frontend_live()
    test_query_live()
    test_agent_trace_live()
    test_documents_endpoint_live()
    test_autodocs_live()

    elapsed = round(time.time() - start, 1)

    print("\n" + "=" * 55)
    print("PHASE 11 COMPLETE")
    print(f"All 6 live deployment tests passed in {elapsed}s")
    print(f"\nLive URLs:")
    print(f"  Backend:  {BACKEND_URL}")
    print(f"  Frontend: {FRONTEND_URL}")
    print(f"  API Docs: {BACKEND_URL}/docs")
    print("\nReady for Phase 12: Evaluation & Resume Metrics")
    print("=" * 55)


if __name__ == "__main__":
    run_phase11_test()
