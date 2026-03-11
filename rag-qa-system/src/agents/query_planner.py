from src.agents.agent_utils import call_agent_llm, parse_json_list_response


QUERY_PLANNER_SYSTEM_PROMPT = """You are a query planning agent for a document Q&A system.

Your job is to analyze a user's question and decide:
1. Is this a SIMPLE question that can be answered with one search?
2. Is this a COMPLEX question that needs to be broken into sub-questions?

SIMPLE questions examples:
- "What is attention mechanism?"
- "What is positional encoding?"
- "How does BM25 work?"

COMPLEX questions examples:
- "Compare attention in transformers vs RNNs and explain tradeoffs"
- "What are the differences between BERT and GPT and when to use each?"
- "Explain how RAG works and why it reduces hallucinations"

RULES:
- For SIMPLE questions: return a list with just the original question
- For COMPLEX questions: break into 2-3 focused sub-questions
- Each sub-question must be self-contained and searchable
- Sub-questions must be shorter and more specific than the original
- Return ONLY a valid JSON array of strings, nothing else
- Never return more than 3 sub-questions

EXAMPLES:

Input: "What is attention?"
Output: ["What is attention mechanism?"]

Input: "Compare transformers and RNNs and explain which is better for long sequences"
Output: [
  "How does the transformer attention mechanism work?",
  "How do RNNs process sequential data?",
  "What are the limitations of RNNs for long sequences compared to transformers?"
]
"""


def plan_query(question: str) -> dict:
    """
    Analyzes a question and returns search sub-queries.

    Returns dict with:
    - original_question: the input question
    - sub_queries: list of search queries to run
    - is_complex: whether question was broken down
    - query_count: how many sub-queries generated
    """
    user_prompt = f"""Analyze this question and return the appropriate JSON array of search queries:

Question: {question}

Remember: Return ONLY a valid JSON array of strings."""

    response = call_agent_llm(
        system_prompt=QUERY_PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0
    )

    sub_queries = parse_json_list_response(response)

    # Fallback: if parsing failed or empty, use original question
    if not sub_queries:
        sub_queries = [question]

    # Safety: never more than 3 sub-queries
    sub_queries = sub_queries[:3]

    # Safety: make sure all are strings
    sub_queries = [str(q) for q in sub_queries if q]

    is_complex = len(sub_queries) > 1

    return {
        "original_question": question,
        "sub_queries": sub_queries,
        "is_complex": is_complex,
        "query_count": len(sub_queries)
    }


if __name__ == "__main__":
    print("Testing Query Planner Agent...")
    print("=" * 50)

    test_questions = [
        # Simple questions
        "What is the attention mechanism?",
        "What is positional encoding?",

        # Complex questions
        "Compare how attention works in transformers vs RNNs and explain the tradeoffs",
        "What is RAG, how does it work, and why does it reduce hallucinations?",
        "Explain the difference between encoder only and decoder only transformers"
    ]

    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        result = plan_query(question)
        print(f"Is complex: {result['is_complex']}")
        print(f"Sub-queries ({result['query_count']}):")
        for i, q in enumerate(result["sub_queries"]):
            print(f"  [{i+1}] {q}")
        print()
