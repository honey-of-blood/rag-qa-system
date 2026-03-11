import gradio as gr
import requests
import uuid
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ─── API HELPERS ──────────────────────────────────────

def api_query(question: str, session_id: str, top_k: int = 5) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "session_id": session_id, "top_k": top_k},
            timeout=120
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Make sure app.py is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The pipeline took too long."}
    except Exception as e:
        return {"error": str(e)}


def api_upload(files) -> dict:
    try:
        file_tuples = []
        for f in files:
            file_tuples.append(
                ("files", (os.path.basename(f.name), open(f.name, "rb"), "application/pdf"))
            )

        r = requests.post(
            f"{BACKEND_URL}/upload",
            files=file_tuples,
            timeout=120
        )

        r.raise_for_status()
        return r.json()

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend."}

    except Exception as e:
        return {"error": str(e)}


def api_health() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_documents() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_delete_session(session_id: str) -> dict:
    try:
        r = requests.delete(f"{BACKEND_URL}/session/{session_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── FORMAT HELPERS ───────────────────────────────────

def format_citations(citations):
    if not citations:
        return "No citations"

    lines = []
    seen = set()

    for c in citations:
        key = f"{c.get('source')}:{c.get('page')}"
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 **{c.get('source')}** — Page {c.get('page')}")

    return "\n".join(lines)


def format_status_bar(result):

    if "error" in result:
        return f"❌ Error: {result['error']}"

    parts = [
        f"Status: **{result.get('status')}**",
        f"Latency: **{result.get('latency_seconds')}s**",
        f"Confidence: **{result.get('answer_confidence',0):.2f}**",
        f"Rewrites: **{result.get('rewrite_count')}**",
        f"Generations: **{result.get('generation_count')}**",
    ]

    if result.get("hallucination_detected"):
        parts.append("🚨 **Hallucination detected**")

    if result.get("is_followup"):
        parts.append("🔁 Follow-up question detected")

    return " | ".join(parts)


def format_system_status():

    health = api_health()

    if "error" in health:
        return f"❌ Backend unreachable: {health['error']}"

    docs = api_documents()

    if "documents" in docs and docs["documents"]:
        doc_list = "\n".join(
            f"• {d['filename']} ({d['chunk_count']} chunks)"
            for d in docs["documents"]
        )
    else:
        doc_list = "No documents indexed yet."

    return (
        f"🟢 Backend connected | "
        f"Sessions: {health.get('active_sessions')} | "
        f"Chunks: {health.get('total_chunks')}\n\n"
        f"**Indexed documents:**\n{doc_list}"
    )


# ─── EVENT HANDLERS ───────────────────────────────────

def handle_upload(files):

    if not files:
        return "Please upload at least one PDF.", format_system_status()

    result = api_upload(files)

    if "error" in result:
        return f"❌ Upload failed: {result['error']}", format_system_status()

    uploaded = ", ".join(result.get("files_uploaded", []))

    return (
        f"✅ Uploaded: {uploaded}\n📦 Total chunks: {result.get('total_chunks')}",
        format_system_status()
    )


def handle_query(question, chat_history, session_id, top_k):

    if not question.strip():
        return chat_history, "", "", "Please enter a question.", ""

    result = api_query(question, session_id, top_k)

    if "error" in result:

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": f"❌ {result['error']}"})

        return chat_history, "", "", result["error"], ""

    answer = result.get("answer", "")

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    citations_text = format_citations(result.get("citations"))
    status_text = format_status_bar(result)

    trace_lines = []
    trace = result.get("agent_trace", [])

    for e in trace:

        node = e.get("node")
        details = e.get("details", {})

        if node == "Query Planner":
            trace_lines.append(f"🧠 Query Planner: {details.get('query_count')} query planned")

        elif node == "Retrieval":
            trace_lines.append(f"🔍 Retrieval: {details.get('total_chunks')} chunks found")

        elif node == "Retrieval Grader":
            trace_lines.append(
                f"✅ Retrieval Grader: {details.get('passed')}/{details.get('total')} relevant"
            )

        elif node == "Query Rewriter":
            trace_lines.append(
                f"✏️ Query Rewriter attempt {details.get('attempt')}"
            )

        elif node == "Reranker":
            trace_lines.append(
                f"📊 Reranker: {details.get('input_chunks')} → {details.get('output_chunks')}"
            )

        elif node == "Answer Generator":
            trace_lines.append(
                f"💬 Generator attempt {details.get('attempt')}"
            )

        elif node == "Answer Grader":
            trace_lines.append(
                f"🔎 Answer Grader confidence {details.get('confidence')}"
            )

        elif node == "Finalizer":
            trace_lines.append(
                f"🏁 Done: {details.get('status')}"
            )

    trace_text = "\n".join(trace_lines)

    return chat_history, trace_text, citations_text, status_text, ""


def handle_clear(session_id):
    api_delete_session(session_id)
    return [], "", "No citations yet", "Chat cleared", ""


# ─── BUILD UI ─────────────────────────────────────────

def build_ui():

    with gr.Blocks(title="Agentic RAG Q&A") as demo:

        session_id = gr.State(str(uuid.uuid4()))

        gr.Markdown("# 🤖 Agentic RAG Multi-Document Q&A")

        with gr.Row():

            file_upload = gr.File(
                label="Upload PDFs",
                file_types=[".pdf"],
                file_count="multiple"
            )

            upload_btn = gr.Button("Index Documents")

        upload_status = gr.Textbox(label="Upload Status")

        system_status = gr.Markdown(format_system_status())

        chatbot = gr.Chatbot(
            height=400
        )

        with gr.Row():

            question_input = gr.Textbox(
                placeholder="Ask a question about your documents..."
            )

            send_btn = gr.Button("Send 🚀")

        top_k = gr.Slider(1, 10, value=5, label="Top K chunks")

        status_bar = gr.Markdown("Ask a question to see pipeline stats.")

        with gr.Row():

            trace_display = gr.Textbox(label="Agent Trace")

            citations_display = gr.Markdown("No citations yet")

        clear_btn = gr.Button("Clear Chat")

        upload_btn.click(
            handle_upload,
            inputs=file_upload,
            outputs=[upload_status, system_status]
        )

        send_btn.click(
            handle_query,
            inputs=[question_input, chatbot, session_id, top_k],
            outputs=[chatbot, trace_display, citations_display, status_bar, question_input]
        )

        question_input.submit(
            handle_query,
            inputs=[question_input, chatbot, session_id, top_k],
            outputs=[chatbot, trace_display, citations_display, status_bar, question_input]
        )

        clear_btn.click(
            handle_clear,
            inputs=session_id,
            outputs=[chatbot, trace_display, citations_display, status_bar, question_input]
        )

    return demo


if __name__ == "__main__":

    demo = build_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
