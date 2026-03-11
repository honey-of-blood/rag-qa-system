from src.agents.agent_utils import call_agent_llm, parse_json_response


RETRIEVAL_GRADER_SYSTEM_PROMPT = """You are a retrieval grading agent for a document Q&A system.

Your job is to judge whether a retrieved document chunk is relevant
to a given question.

RULES:
- Grade ONLY based on whether the chunk contains information useful
  for answering the question
- Do NOT require the chunk to fully answer the question
- A chunk is RELEVANT if it contains related concepts, definitions,
  explanations, or supporting information
- A chunk is NOT RELEVANT if it is completely unrelated to the question
- Be generous but not sloppy — partial relevance counts as relevant
- Return ONLY valid JSON, nothing else

OUTPUT FORMAT:
{
  "relevant": true or false,
  "reason": "one sentence explaining your decision",
  "confidence": 0.0 to 1.0
}

EXAMPLES:

Question: "What is multi-head attention?"
Chunk: "Multi-head attention allows the model to attend to information
from different representation subspaces at different positions."
Output: {"relevant": true, "reason": "Directly defines multi-head attention", "confidence": 0.98}

Question: "What is multi-head attention?"
Chunk: "The authors used the Adam optimizer with beta1=0.9 and beta2=0.98"
Output: {"relevant": false, "reason": "About optimizer settings, unrelated to attention", "confidence": 0.95}

Question: "What is multi-head attention?"
Chunk: "The model consists of an encoder and decoder, each with attention layers"
Output: {"relevant": true, "reason": "Mentions attention layers in model architecture", "confidence": 0.72}
"""


def grade_chunk(query: str, chunk: dict) -> dict:
    """
    Grades a single chunk for relevance to the query.

    Returns dict with:
    - chunk_id: original chunk id
    - relevant: bool
    - reason: explanation
    - confidence: float 0-1
    - original_chunk: the chunk dict
    """
    chunk_text = chunk.get("text", "")
    chunk_id = chunk.get("chunk_id", -1)

    user_prompt = f"""Grade this chunk for relevance to the question.

Question: {query}

Chunk content:
{chunk_text}

Return JSON only."""

    response = call_agent_llm(
        system_prompt=RETRIEVAL_GRADER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0
    )

    parsed = parse_json_response(response)

    # Handle parsing failures gracefully
    if "error" in parsed:
        return {
            "chunk_id": chunk_id,
            "relevant": True,  # default to keeping chunk if parsing fails
            "reason": "Grading failed — kept by default",
            "confidence": 0.5,
            "original_chunk": chunk
        }

    return {
        "chunk_id": chunk_id,
        "relevant": bool(parsed.get("relevant", True)),
        "reason": parsed.get("reason", "No reason provided"),
        "confidence": float(parsed.get("confidence", 0.5)),
        "original_chunk": chunk
    }


def grade_chunks(query: str, chunks: list[dict]) -> dict:
    """
    Grades all retrieved chunks and filters to relevant ones.

    Returns dict with:
    - relevant_chunks: list of chunks that passed grading
    - rejected_chunks: list of chunks that failed grading
    - total_graded: int
    - passed_count: int
    - failed_count: int
    - pass_rate: float
    - needs_rewrite: bool (True if too few chunks passed)
    """
    print(f"Grading {len(chunks)} chunks for query: '{query[:60]}...'")

    relevant_chunks = []
    rejected_chunks = []

    for chunk in chunks:
        grade = grade_chunk(query, chunk)
        if grade["relevant"]:
            relevant_chunks.append(grade["original_chunk"])
        else:
            rejected_chunks.append({
                "chunk": grade["original_chunk"],
                "reason": grade["reason"]
            })

    total = len(chunks)
    passed = len(relevant_chunks)
    failed = len(rejected_chunks)
    pass_rate = passed / total if total > 0 else 0.0

    # Trigger rewrite if less than 2 relevant chunks found
    needs_rewrite = passed < 2

    print(f"Grading complete: {passed}/{total} chunks passed ({pass_rate:.0%})")
    if needs_rewrite:
        print("⚠️  Too few relevant chunks — Query Rewriter will be triggered")

    return {
        "relevant_chunks": relevant_chunks,
        "rejected_chunks": rejected_chunks,
        "total_graded": total,
        "passed_count": passed,
        "failed_count": failed,
        "pass_rate": pass_rate,
        "needs_rewrite": needs_rewrite
    }


if __name__ == "__main__":
    print("Testing Retrieval Grader Agent...")
    print("=" * 50)

    test_query = "What is multi-head attention?"

    test_chunks = [
        {
            "chunk_id": 0,
            "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
            "source": "attention_paper.pdf",
            "page": 4
        },
        {
            "chunk_id": 1,
            "text": "We used the Adam optimizer with a learning rate of 0.0001 and batch size of 32 for all experiments.",
            "source": "attention_paper.pdf",
            "page": 7
        },
        {
            "chunk_id": 2,
            "text": "The encoder consists of a stack of 6 identical layers, each with a multi-head self-attention mechanism.",
            "source": "attention_paper.pdf",
            "page": 3
        },
        {
            "chunk_id": 3,
            "text": "Table 3 shows BLEU scores on the English-German translation task across different model configurations.",
            "source": "attention_paper.pdf",
            "page": 9
        },
        {
            "chunk_id": 4,
            "text": "Attention scores are computed as scaled dot products of queries and keys, then softmax normalized.",
            "source": "attention_paper.pdf",
            "page": 4
        }
    ]

    result = grade_chunks(test_query, test_chunks)

    print(f"\nResults:")
    print(f"Total graded: {result['total_graded']}")
    print(f"Passed: {result['passed_count']}")
    print(f"Failed: {result['failed_count']}")
    print(f"Pass rate: {result['pass_rate']:.0%}")
    print(f"Needs rewrite: {result['needs_rewrite']}")

    print(f"\nRelevant chunks:")
    for c in result["relevant_chunks"]:
        print(f"  ✅ [{c['chunk_id']}] {c['text'][:80]}...")

    print(f"\nRejected chunks:")
    for r in result["rejected_chunks"]:
        print(f"  ❌ [{r['chunk']['chunk_id']}] Reason: {r['reason']}")
