import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client() -> Groq:
    """
    Returns singleton Groq client.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env file")
        _client = Groq(api_key=api_key)
    return _client


def call_agent_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512
) -> str:
    """
    Calls Groq LLM for an agent decision.
    Temperature 0.0 for deterministic decisions.
    Returns raw response text.
    """
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def parse_json_response(text: str) -> dict:
    """
    Safely parses JSON from LLM response.
    Handles cases where LLM wraps JSON in markdown code blocks.
    """
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Return safe fallback
        return {"error": "Failed to parse JSON", "raw": text}


def parse_json_list_response(text: str) -> list:
    """
    Safely parses JSON array from LLM response.
    """
    text = text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        # Try to extract JSON array from text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


if __name__ == "__main__":
    # Quick test
    response = call_agent_llm(
        system_prompt="You are a helpful assistant. Reply in JSON.",
        user_prompt='Return this JSON: {"status": "working"}'
    )
    print(f"Raw response: {response}")
    parsed = parse_json_response(response)
    print(f"Parsed: {parsed}")
