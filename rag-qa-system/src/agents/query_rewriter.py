from src.agents.agent_utils import call_agent_llm, parse_json_response


QUERY_REWRITER_SYSTEM_PROMPT = """You are a query rewriting agent for a document search system.

Your job is to rewrite a failed search query into a better one
that will retrieve more relevant documents.

WHY QUERIES FAIL:
- Too conversational ("How does it work?")
- Uses pronouns without context ("What does it do?")
- Too vague ("Tell me about the thing")
- Too long and complex for keyword matching
- Uses different terminology than the documents

HOW TO REWRITE:
- Use specific technical terms and keywords
- Remove pronouns — be explicit about what "it" refers to
- Use search-engine style language: nouns and key terms
- Add related synonyms or technical variants
- Keep it concise — 5 to 12 words is ideal
- Think about what words would appear in a relevant paper

RULES:
- Return ONLY valid JSON, nothing else
- The rewritten query must be different from the original
- Never just add "definition of" to the front — be creative

OUTPUT FORMAT:
{
  "rewritten_query": "the improved search query",
  "changes_made": "brief explanation of what you changed and why",
  "strategy": "keyword_expansion | clarification | terminology_shift | decomposition"
}

EXAMPLES:

Original: "How does it remember things?"
Conversation context: User was asking about RNNs
Output: {
  "rewritten_query": "RNN hidden state memory sequential information retention",
  "changes_made": "Replaced pronoun with RNN, added technical terms for memory mechanism",
  "strategy": "clarification"
}

Original: "What are the benefits?"
Conversation context: User was asking about transformer attention
Output: {
  "rewritten_query": "transformer attention mechanism advantages over recurrent networks",
  "changes_made": "Added subject, specified comparison to make query more precise",
  "strategy": "clarification"
}

Original: "Explain the whole architecture in detail with all components"
Output: {
  "rewritten_query": "transformer encoder decoder architecture components layers",
  "changes_made": "Stripped conversational framing, extracted key technical terms",
  "strategy": "keyword_expansion"
}
"""


def rewrite_query(
    original_query: str,
    conversation_context: str = "",
    attempt_number: int = 1
) -> dict:
    """
    Rewrites a failed search query into a better one.

    Returns dict with:
    - original_query: the input query
    - rewritten_query: the improved query
    - changes_made: explanation of changes
    - strategy: rewriting strategy used
    - attempt_number: which retry attempt this is
    """
    context_section = ""
    if conversation_context:
        context_section = f"""
Previous conversation context:
{conversation_context}
"""

    attempt_instruction = ""
    if attempt_number == 2:
        attempt_instruction = "\nThis is a second attempt — try a completely different approach than before."
    elif attempt_number >= 3:
        attempt_instruction = "\nThis is a final attempt — use the most general possible search terms."

    user_prompt = f"""Rewrite this failed search query to retrieve better results.
{context_section}
Original query: {original_query}
{attempt_instruction}

Return JSON only."""

    response = call_agent_llm(
        system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3  # slight creativity for different rewrites
    )

    parsed = parse_json_response(response)

    if "error" in parsed:
        # Fallback: simple keyword extraction
        keywords = " ".join([
            w for w in original_query.split()
            if len(w) > 4 and w.lower() not in
            ["what", "how", "does", "the", "and", "for", "with", "that", "this"]
        ])
        fallback_query = keywords if keywords else original_query
        return {
            "original_query": original_query,
            "rewritten_query": fallback_query,
            "changes_made": "Fallback: extracted keywords",
            "strategy": "keyword_expansion",
            "attempt_number": attempt_number
        }

    return {
        "original_query": original_query,
        "rewritten_query": parsed.get("rewritten_query", original_query),
        "changes_made": parsed.get("changes_made", ""),
        "strategy": parsed.get("strategy", "unknown"),
        "attempt_number": attempt_number
    }


def rewrite_with_retries(
    original_query: str,
    conversation_context: str = "",
    max_attempts: int = 3
) -> list[dict]:
    """
    Generates multiple different rewrites of a failed query.
    Returns list of rewrite results — one per attempt.
    Used by the orchestrator to try multiple search strategies.
    """
    rewrites = []
    for attempt in range(1, max_attempts + 1):
        rewrite = rewrite_query(
            original_query,
            conversation_context,
            attempt_number=attempt
        )
        rewrites.append(rewrite)
        print(
            f"Rewrite attempt {attempt}: "
            f"'{rewrite['rewritten_query']}' "
            f"(strategy: {rewrite['strategy']})"
        )
    return rewrites


if __name__ == "__main__":
    print("Testing Query Rewriter Agent...")
    print("=" * 50)

    test_cases = [
        {
            "query": "How does it remember things?",
            "context": "User was asking about RNN hidden states"
        },
        {
            "query": "What are the benefits?",
            "context": "User was asking about transformer attention mechanism"
        },
        {
            "query": "Tell me more about the training",
            "context": ""
        },
        {
            "query": "Explain everything about how it all works together",
            "context": ""
        }
    ]

    for tc in test_cases:
        print(f"\nOriginal: '{tc['query']}'")
        if tc["context"]:
            print(f"Context: '{tc['context']}'")

        result = rewrite_query(
            tc["query"],
            conversation_context=tc["context"]
        )

        print(f"Rewritten: '{result['rewritten_query']}'")
        print(f"Strategy: {result['strategy']}")
        print(f"Changes: {result['changes_made']}")

    print("\n" + "=" * 50)
    print("Testing multi-attempt rewriting...")
    rewrites = rewrite_with_retries(
        "How does it work?",
        conversation_context="User was asking about RAG systems",
        max_attempts=3
    )
    print(f"\nGenerated {len(rewrites)} different rewrites:")
    for r in rewrites:
        print(f"  Attempt {r['attempt_number']}: '{r['rewritten_query']}'")
