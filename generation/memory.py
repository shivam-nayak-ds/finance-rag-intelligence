"""Short-term (session) and long-term (SQLite) memory for SyllAIq."""

import json
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from config.settings import MAX_CHAT_HISTORY, ROOT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = ROOT_DIR / "data" / "syllaiq.db"


# ──────────────────────────────────────────
# SHORT-TERM MEMORY (In-session RAM)
# ──────────────────────────────────────────

@dataclass
class Turn:
    """A single conversation turn."""
    role: str       # "user" | "assistant"
    content: str


@dataclass
class Session:
    """Per-session in-memory conversation state."""
    session_id: str
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_CHAT_HISTORY * 2))
    last_intent: Optional[str] = None
    last_topic: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def as_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI-compatible message list for LLM prompt injection."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def context_summary(self) -> str:
        """Last 2 exchanges as plain text (for query rewriting context)."""
        if not self.turns:
            return ""
        recent = list(self.turns)[-4:]
        return "\n".join(f"{t.role.upper()}: {t.content[:150]}" for t in recent)


class ConversationMemory:
    """
    Short-term in-memory session store.
    Each session_id gets a rolling history of last MAX_CHAT_HISTORY exchanges.
    Cleared on server restart.
    """

    def __init__(self, max_history: int = MAX_CHAT_HISTORY) -> None:
        self._sessions: Dict[str, Session] = {}
        self.max_history = max_history

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
            logger.debug("New session created: %s", session_id)
        return self._sessions[session_id]

    def add_user_turn(self, session_id: str, query: str) -> None:
        self.get_or_create(session_id).add_turn("user", query)

    def add_assistant_turn(
        self, session_id: str, answer: str, intent: Optional[str] = None
    ) -> None:
        session = self.get_or_create(session_id)
        session.add_turn("assistant", answer)
        if intent:
            session.last_intent = intent

    def get_history_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Return full history as OpenAI message list — injected into LLM prompt."""
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].as_messages()

    def get_context_summary(self, session_id: str) -> str:
        """Plain text recent context — used by query rewriter."""
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].context_summary()

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        logger.info("Session cleared: %s", session_id)

    def active_sessions(self) -> int:
        return len(self._sessions)


# ──────────────────────────────────────────
# LONG-TERM MEMORY (SQLite Student Profile)
# ──────────────────────────────────────────

from contextlib import contextmanager


class LongTermMemory:
    """
    Persists student learning patterns across sessions in SQLite.

    Tracks:
    - Topics queried (with frequency)
    - Units studied
    - Weak areas (low confidence answers)
    - Session history (query log)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("LongTermMemory initialised at: %s", db_path)

    @contextmanager
    def _connect(self):
        """Context manager that ensures connection is properly committed and closed."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS student_queries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    query       TEXT NOT NULL,
                    intent      TEXT,
                    topic       TEXT,
                    unit        INTEGER,
                    confidence  TEXT,
                    timestamp   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS topic_frequency (
                    session_id  TEXT NOT NULL,
                    topic       TEXT NOT NULL,
                    unit        INTEGER,
                    count       INTEGER DEFAULT 1,
                    PRIMARY KEY (session_id, topic)
                );

                CREATE TABLE IF NOT EXISTS weak_areas (
                    session_id  TEXT NOT NULL,
                    topic       TEXT NOT NULL,
                    unit        INTEGER,
                    low_conf_count  INTEGER DEFAULT 1,
                    last_seen   TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (session_id, topic)
                );
            """)

    def log_query(
        self,
        session_id: str,
        query: str,
        intent: Optional[str] = None,
        topic: Optional[str] = None,
        unit: Optional[int] = None,
        confidence: Optional[str] = None,
    ) -> None:
        """Log every student query to persistent storage."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO student_queries
                       (session_id, query, intent, topic, unit, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session_id, query[:500], intent, topic, unit, confidence),
                )
                if topic:
                    conn.execute(
                        """INSERT INTO topic_frequency (session_id, topic, unit, count)
                           VALUES (?, ?, ?, 1)
                           ON CONFLICT(session_id, topic)
                           DO UPDATE SET count = count + 1""",
                        (session_id, topic, unit),
                    )
            logger.debug("Logged query for session %s: topic=%s", session_id, topic)
        except Exception as err:
            logger.warning("LongTermMemory.log_query failed: %s", err)

    def mark_weak_area(
        self,
        session_id: str,
        topic: str,
        unit: Optional[int] = None,
    ) -> None:
        """Increment weak area count when confidence is LOW."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO weak_areas (session_id, topic, unit, low_conf_count, last_seen)
                       VALUES (?, ?, ?, 1, datetime('now'))
                       ON CONFLICT(session_id, topic)
                       DO UPDATE SET
                           low_conf_count = low_conf_count + 1,
                           last_seen = datetime('now')""",
                    (session_id, topic, unit),
                )
        except Exception as err:
            logger.warning("LongTermMemory.mark_weak_area failed: %s", err)

    def get_student_profile(self, session_id: str) -> Dict:
        """
        Returns a summary of what this student has studied.

        Returns:
            Dict with:
              - total_queries: int
              - top_topics: List[{topic, count}]
              - weak_areas: List[{topic, low_conf_count}]
              - units_studied: List[int]
        """
        try:
            with self._connect() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM student_queries WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]

                top_topics = conn.execute(
                    """SELECT topic, count FROM topic_frequency
                       WHERE session_id=? ORDER BY count DESC LIMIT 5""",
                    (session_id,),
                ).fetchall()

                weak = conn.execute(
                    """SELECT topic, low_conf_count FROM weak_areas
                       WHERE session_id=? ORDER BY low_conf_count DESC LIMIT 5""",
                    (session_id,),
                ).fetchall()

                units = conn.execute(
                    """SELECT DISTINCT unit FROM student_queries
                       WHERE session_id=? AND unit IS NOT NULL ORDER BY unit""",
                    (session_id,),
                ).fetchall()

            return {
                "total_queries": total,
                "top_topics": [{"topic": r[0], "count": r[1]} for r in top_topics],
                "weak_areas": [{"topic": r[0], "low_conf_count": r[1]} for r in weak],
                "units_studied": [r[0] for r in units],
            }
        except Exception as err:
            logger.warning("get_student_profile failed: %s", err)
            return {"total_queries": 0, "top_topics": [], "weak_areas": [], "units_studied": []}

    def get_personalization_hint(self, session_id: str) -> str:
        """
        Returns a concise string injected into system prompt for personalization.

        Example:
            "Student has asked 12 questions so far. Weak areas needing extra clarity: Deadlock, Semaphores.
             Recently studied: Unit 4."
        """
        profile = self.get_student_profile(session_id)
        if profile["total_queries"] == 0:
            return ""

        parts = [f"Student has asked {profile['total_queries']} questions so far."]

        if profile["weak_areas"]:
            weak = ", ".join(w["topic"] for w in profile["weak_areas"][:3])
            parts.append(f"Weak areas needing extra clarity: {weak}.")

        if profile["units_studied"]:
            units = ", ".join(f"Unit {u}" for u in profile["units_studied"][-3:])
            parts.append(f"Recently studied: {units}.")

        return " ".join(parts)


# ──────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────

_short_term = ConversationMemory()
_long_term = LongTermMemory()


def get_memory() -> ConversationMemory:
    """Global short-term memory singleton."""
    return _short_term


def get_long_term_memory() -> LongTermMemory:
    """Global long-term memory singleton."""
    return _long_term
