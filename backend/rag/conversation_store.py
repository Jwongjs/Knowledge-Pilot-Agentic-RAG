from __future__ import annotations
import json
import os
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Firestore-backed store
# ---------------------------------------------------------------------------

def _make_firestore_client():
    """Build a Firestore client from either a file path or inline JSON credentials."""
    from google.cloud import firestore
    from google.oauth2 import service_account

    project = os.environ.get("FIRESTORE_PROJECT_ID")
    inline_json = os.environ.get("FIRESTORE_CREDENTIALS_JSON")

    if inline_json:
        info = json.loads(inline_json)
        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(project=project, credentials=creds)

    # Falls back to GOOGLE_APPLICATION_CREDENTIALS file or ADC
    return firestore.Client(project=project)


class FirestoreConversationStore:
    """Persistent session store backed by Firestore. Works from any deployment target."""

    _COLLECTION = "sessions"

    def __init__(self):
        self._db = _make_firestore_client()

    def get_or_create_session(self, session_id: str) -> Session:
        turns = self._load_turns(session_id)
        return Session(session_id=session_id, turns=turns)

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        doc_ref = self._db.collection(self._COLLECTION).document(session_id)
        doc = doc_ref.get()
        turns = doc.to_dict().get("turns", []) if doc.exists else []
        turns.append({"role": role, "content": content})
        doc_ref.set({"turns": turns})

    def get_history(self, session_id: str) -> list[Turn]:
        return self._load_turns(session_id)

    def _load_turns(self, session_id: str) -> list[Turn]:
        doc = self._db.collection(self._COLLECTION).document(session_id).get()
        if not doc.exists:
            return []
        return [Turn(role=t["role"], content=t["content"]) for t in doc.to_dict().get("turns", [])]


# ---------------------------------------------------------------------------
# In-memory store (local dev / no credentials configured)
# ---------------------------------------------------------------------------

class InMemoryConversationStore:
    """Fallback store: sessions live only for the process lifetime."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        self.get_or_create_session(session_id).turns.append(Turn(role=role, content=content))

    def get_history(self, session_id: str) -> list[Turn]:
        return self._sessions.get(session_id, Session(session_id)).turns


# ---------------------------------------------------------------------------
# Factory: use Firestore when credentials are present, else fall back silently
# ---------------------------------------------------------------------------

def ConversationStore():
    has_credentials = (
        os.environ.get("FIRESTORE_CREDENTIALS_JSON")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("FIRESTORE_PROJECT_ID")
    )
    if has_credentials:
        try:
            return FirestoreConversationStore()
        except Exception:
            pass
    return InMemoryConversationStore()
