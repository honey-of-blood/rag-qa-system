import time
from src.memory import ConversationMemory


class SessionManager:
    """
    Manages per-user conversation memory and query trace history.
    Each session_id gets its own ConversationMemory instance.
    Sessions expire after max_age_seconds of inactivity (default 1 hour).
    """

    def __init__(self, max_turns: int = 4, max_age_seconds: int = 3600):
        self.max_turns = max_turns
        self.max_age_seconds = max_age_seconds
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """
        Returns existing memory for session_id, or creates a new one.
        Updates the last_accessed timestamp on every call.
        """
        self._expire_old_sessions()

        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "memory": ConversationMemory(max_turns=self.max_turns),
                "last_accessed": time.time(),
                "last_trace": []
            }

        self._sessions[session_id]["last_accessed"] = time.time()
        return self._sessions[session_id]["memory"]

    def store_trace(self, session_id: str, trace: list[dict]):
        """Stores the last agent trace for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["last_trace"] = trace

    def get_trace(self, session_id: str) -> list[dict]:
        """Returns the last agent trace for a session."""
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].get("last_trace", [])

    def delete(self, session_id: str) -> bool:
        """Deletes a session and its memory. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def active_session_count(self) -> int:
        self._expire_old_sessions()
        return len(self._sessions)

    def _expire_old_sessions(self):
        """Removes sessions that haven't been accessed within max_age_seconds."""
        now = time.time()
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["last_accessed"] > self.max_age_seconds
        ]
        for sid in expired:
            del self._sessions[sid]


if __name__ == "__main__":
    manager = SessionManager(max_turns=4)

    # Create two sessions
    mem1 = manager.get_or_create("user_alice")
    mem1.add_turn("What is attention?", "Attention allows models to focus on relevant parts.")

    mem2 = manager.get_or_create("user_bob")
    mem2.add_turn("What is RAG?", "RAG combines retrieval with generation.")

    assert manager.active_session_count() == 2
    print(f"✅ Active sessions: {manager.active_session_count()}")

    # Store and retrieve trace
    manager.store_trace("user_alice", [{"node": "Query Planner", "event": "test"}])
    trace = manager.get_trace("user_alice")
    assert len(trace) == 1
    print(f"✅ Trace stored and retrieved: {trace[0]['node']}")

    # Delete session
    deleted = manager.delete("user_alice")
    assert deleted == True
    assert manager.active_session_count() == 1
    print(f"✅ Session deleted, remaining: {manager.active_session_count()}")

    print("session_manager.py OK")
