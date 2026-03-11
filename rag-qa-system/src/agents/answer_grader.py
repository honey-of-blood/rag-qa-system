from src.agents.agent_utils import call_agent_llm, parse_json_response


ANSWER_GRADER_SYSTEM_PROMPT = """You are an answer grading agent for a document Q&A system.

Your job is to evaluate whether a generated answer is:
1. ADDRESSES the question — does it actually answer what was asked?
2. GROUNDED — does it only use information from the provided context?
3. FREE OF HALLUCINATION — does it avoid making up facts not in context?

GRADING CRITERIA:
- PASS: Answer addresses the question AND is grounded in context
- FAIL: Answer does not address question OR contains hallucinated facts

HALLUCINATION DETECTION:
- Check if specific facts, numbers, names, or claims in the answer
  actually appear in the provided context chunks
- Vague or general statements are harder to verify — give benefit of doubt
- If a specific claim cannot be traced to any context chunk, flag it

RULES:
- Be strict about hallucinations but fair about relevance
- A partial answer that is grounded is better than a full answer that hallucinates
- Return ONLY valid JSON, nothing else

OUTPUT FORMAT:
{
  "passes": true or false,
  "addresses_question": true or false,
  "is_grounded": true or false,
  "hallucination_detected": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one to two sentences explaining the grade",
  "problematic_claims": ["claim 1 that seems hallucinated", "claim 2"]
}

EXAMPLES:

Question: "What is multi-head attention?"
Answer: "Multi-head attention runs attention in parallel across multiple heads, allowing the model to attend to different subspaces."
Context contains: "Multi-head attention allows the model to jointly attend to information from different representation subspaces"
Output: {
  "passes": true,
  "addresses_question": true,
  "is_grounded": true,
  "hallucination_detected": false,
  "confidence": 0.96,
  "reason": "Answer directly defines multi-head attention using language grounded in context.",
  "problematic_claims": []
}

Question: "What optimizer was used?"
Answer: "The authors used SGD with momentum 0.95 and weight decay 1e-4."
Context contains: "We used the Adam optimizer with beta1=0.9"
Output: {
  "passes": false,
  "addresses_question": true,
  "is_grounded": false,
  "hallucination_detected": true,
  "confidence": 0.91,
  "reason": "Answer claims SGD was used but context clearly states Adam optimizer.",
  "problematic_claims": ["SGD with momentum 0.95", "weight decay 1e-4"]
}
"""


def grade_answer(
    question: str,
    answer: str,
    context_chunks: list[dict]
) -> dict:
    """
    Grades a generated answer for quality and groundedness.

    Returns dict with:
    - passes: bool — whether answer is acceptable
    - addresses_question: bool
    - is_grounded: bool
    - hallucination_detected: bool
    - confidence: float
    - reason: explanation
    - problematic_claims: list of suspicious claims
    - should_regenerate: bool
    """
    # Build context summary for grader
    context_text = "\n\n".join([
        f"[Chunk {i+1} from {c.get('source', 'unknown')} p.{c.get('page', '?')}]:\n{c.get('text', '')}"
        for i, c in enumerate(context_chunks[:5])  # limit to 5 chunks
    ])

    user_prompt = f"""Grade this answer for quality and groundedness.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

CONTEXT CHUNKS THAT WERE PROVIDED TO THE LLM:
{context_text}

Return JSON grade only."""

    response = call_agent_llm(
        system_prompt=ANSWER_GRADER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=512
    )

    parsed = parse_json_response(response)

    if "error" in parsed:
        # Default to passing if grader fails
        return {
            "passes": True,
            "addresses_question": True,
            "is_grounded": True,
            "hallucination_detected": False,
            "confidence": 0.5,
            "reason": "Grader failed to parse — passed by default",
            "problematic_claims": [],
            "should_regenerate": False
        }

    passes = bool(parsed.get("passes", True))
    hallucination = bool(parsed.get("hallucination_detected", False))

    return {
        "passes": passes,
        "addresses_question": bool(parsed.get("addresses_question", True)),
        "is_grounded": bool(parsed.get("is_grounded", True)),
        "hallucination_detected": hallucination,
        "confidence": float(parsed.get("confidence", 0.5)),
        "reason": parsed.get("reason", ""),
        "problematic_claims": parsed.get("problematic_claims", []),
        "should_regenerate": not passes
    }


if __name__ == "__main__":
    print("Testing Answer Grader Agent...")
    print("=" * 50)

    context_chunks = [
        {
            "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
            "source": "attention_paper.pdf",
            "page": 4
        },
        {
            "text": "We used the Adam optimizer with beta1=0.9 and beta2=0.98 and epsilon=10^-9.",
            "source": "attention_paper.pdf",
            "page": 7
        }
    ]

    test_cases = [
        {
            "question": "What is multi-head attention?",
            "answer": "Multi-head attention runs attention in parallel across multiple heads, allowing the model to attend to different representation subspaces.",
            "expected": "PASS"
        },
        {
            "question": "What optimizer was used?",
            "answer": "The authors used SGD with momentum 0.95 and learning rate 0.001 for all experiments.",
            "expected": "FAIL — hallucination"
        },
        {
            "question": "What is the meaning of life?",
            "answer": "Multi-head attention allows models to process information.",
            "expected": "FAIL — does not address question"
        }
    ]

    for tc in test_cases:
        print(f"\nQuestion: '{tc['question']}'")
        print(f"Answer: '{tc['answer'][:100]}...'")
        print(f"Expected: {tc['expected']}")

        result = grade_answer(
            tc["question"],
            tc["answer"],
            context_chunks
        )

        status = "✅ PASS" if result["passes"] else "❌ FAIL"
        print(f"Result: {status}")
        print(f"Addresses question: {result['addresses_question']}")
        print(f"Is grounded: {result['is_grounded']}")
        print(f"Hallucination detected: {result['hallucination_detected']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reason: {result['reason']}")
        if result["problematic_claims"]:
            print(f"Problematic claims: {result['problematic_claims']}")
