class ConversationTurn:
    def __init__(self, question: str, answer: str, citations: list[dict] = None):
        self.question = question
        self.answer = answer
        self.citations = citations or []


class ConversationMemory:
    """
    Sliding window conversation buffer.
    Keeps the last max_turns Q&A pairs.
    Passed into the pipeline so the agent understands follow-up questions.
    """

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self.turns: list[ConversationTurn] = []

    def add_turn(self, question: str, answer: str, citations: list[dict] = None):
        self.turns.append(ConversationTurn(question, answer, citations))
        # Keep only the last max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format_history_for_prompt(self) -> str:
        """
        Formats conversation history as a string for injection into prompts.
        """
        if not self.turns:
            return ""
        lines = []
        for i, turn in enumerate(self.turns):
            lines.append(f"Q{i+1}: {turn.question}")
            lines.append(f"A{i+1}: {turn.answer[:300]}")
        return "\n".join(lines)

    def is_followup_question(self, question: str) -> bool:
        """
        Simple heuristic: question is likely a follow-up if it contains
        pronouns referring back to previous context.
        """
        if not self.turns:
            return False
        followup_signals = [
            "it", "they", "them", "this", "that", "these", "those",
            "the same", "also", "another", "more about", "what about",
            "how about", "and", "but", "so", "why", "when", "where"
        ]
        question_lower = question.lower()
        return any(
            question_lower.startswith(signal) or f" {signal} " in question_lower
            for signal in followup_signals
        )

    def clear(self):
        self.turns = []

    def __len__(self):
        return len(self.turns)


if __name__ == "__main__":
    memory = ConversationMemory(max_turns=4)

    memory.add_turn(
        "What is multi-head attention?",
        "Multi-head attention allows the model to attend to different representation subspaces.",
        citations=[{"source": "paper.pdf", "page": 4}]
    )
    memory.add_turn(
        "How many heads does it use?",
        "The original transformer uses 8 attention heads."
    )

    print("Conversation history:")
    print(memory.format_history_for_prompt())
    print(f"\nIs follow-up ('Why does it use 8?'): {memory.is_followup_question('Why does it use 8?')}")
    print(f"Is follow-up ('What is BERT?'): {memory.is_followup_question('What is BERT?')}")
    print(f"Turns in memory: {len(memory)}")

    # Test sliding window
    for i in range(5):
        memory.add_turn(f"Question {i}", f"Answer {i}")
    print(f"\nAfter adding 5 more turns, memory has: {len(memory)} turns (max 4)")
    print("memory.py OK")
