import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=api_key)
    return _client


def build_context_block(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Formats reranked chunks into a context string for the LLM.
    Also builds citation list from chunk metadata.
    Returns (context_text, citations).
    """
    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{text}"
        )

        # Build citation entry
        already_cited = any(
            c["source"] == source and c["page"] == page
            for c in citations
        )
        if not already_cited:
            citations.append({
                "source": source,
                "page": page,
                "chunk_index": i
            })

    return "\n\n".join(context_parts), citations


ANSWER_SYSTEM_PROMPT = """You are a precise document Q&A assistant.

RULES:
- Answer ONLY using information from the provided context chunks
- If the context does not contain enough information, say so clearly
- Do NOT make up facts, numbers, or claims not in the context
- Be concise and direct — 2 to 5 sentences unless the question requires more
- Do not reference chunk numbers or source labels in your answer
- Write in plain prose, not bullet points
"""


def generate_answer(
    query: str,
    reranked_chunks: list[dict],
    conversation_history: str = "",
    strict_mode: bool = False
) -> dict:
    """
    Generates a grounded answer from the reranked chunks.

    Args:
        query: the user's question
        reranked_chunks: top chunks from reranker
        conversation_history: formatted prior conversation turns
        strict_mode: if True, uses a stricter prompt (for regeneration attempts)

    Returns dict with:
        answer, citations, citation_count, chunks_used, latency_seconds
    """
    if not reranked_chunks:
        return {
            "answer": "I could not find sufficient information in the documents to answer this question.",
            "citations": [],
            "citation_count": 0,
            "chunks_used": 0,
            "latency_seconds": 0.0
        }

    context_text, citations = build_context_block(reranked_chunks)

    history_section = ""
    if conversation_history:
        history_section = f"\nConversation history:\n{conversation_history}\n"

    strict_note = ""
    if strict_mode:
        strict_note = "\nIMPORTANT: This is a retry. Only include claims you can directly trace to the context. When in doubt, omit.\n"

    user_prompt = f"""{history_section}{strict_note}
Context documents:
{context_text}

Question: {query}

Answer using only the context above:"""

    client = get_groq_client()
    start = time.time()

    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    latency = round(time.time() - start, 2)
    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "citations": citations,
        "citation_count": len(citations),
        "chunks_used": len(reranked_chunks),
        "latency_seconds": latency
    }


if __name__ == "__main__":
    print("Testing answer_generator.py...")

    mock_chunks = [
        {
            "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
            "source": "attention_paper.pdf",
            "page": 4,
            "rerank_score": 0.92
        },
        {
            "text": "The scaled dot-product attention computes a weighted sum of values, where weights come from the compatibility of queries with keys.",
            "source": "attention_paper.pdf",
            "page": 4,
            "rerank_score": 0.87
        }
    ]

    result = generate_answer(
        query="What is multi-head attention?",
        reranked_chunks=mock_chunks
    )

    print(f"Answer: {result['answer']}")
    print(f"Citations: {result['citations']}")
    print(f"Latency: {result['latency_seconds']}s")
    print("answer_generator.py OK")
