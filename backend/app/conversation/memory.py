"""
Conversational Intelligence layer: per-session memory + clarification logic.

Requirement from the brief: "Handle multi-turn conversations with memory",
"Ask clarifying questions when inputs are incomplete", "Avoid repeatedly
asking for information already provided."

This is an in-memory store (dict keyed by session_id) which is fine for a
hackathon demo / single-process deployment. docs/ARCHITECTURE.md documents
the swap to Redis for a multi-instance production deployment - noted as a
P2 item, not implemented, so we don't overstate what's here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List


@dataclass
class SessionState:
    session_id: str
    known_variables: Dict[str, Any] = field(default_factory=dict)
    turns: List[Dict[str, str]] = field(default_factory=list)
    asked_questions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ConversationMemory:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def update_variables(self, session_id: str, new_vars: Dict[str, Any]) -> SessionState:
        state = self.get_or_create(session_id)
        with self._lock:
            state.known_variables.update({k: v for k, v in new_vars.items() if v is not None})
            state.updated_at = time.time()
        return state

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        state = self.get_or_create(session_id)
        with self._lock:
            state.turns.append({"role": role, "content": content})
            state.updated_at = time.time()

    def record_asked_questions(self, session_id: str, questions: List[str]) -> None:
        state = self.get_or_create(session_id)
        with self._lock:
            for q in questions:
                if q not in state.asked_questions:
                    state.asked_questions.append(q)

    def filter_unasked(self, session_id: str, questions: List[str]) -> List[str]:
        state = self.get_or_create(session_id)
        return [q for q in questions if q not in state.asked_questions]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# Module-level singleton used by the FastAPI app. For a multi-process
# deployment (e.g. multiple uvicorn workers), replace this with a Redis- or
# Postgres-backed store keyed the same way.
memory_store = ConversationMemory()
