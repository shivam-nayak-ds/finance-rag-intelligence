"""Unit tests for Short-term and Long-term memory."""

import tempfile
from pathlib import Path
import pytest

from generation.memory import (
    ConversationMemory,
    LongTermMemory,
    Session,
    Turn,
)


class TestShortTermMemory:
    """Test in-memory session and conversation history."""

    def test_add_turns_and_message_format(self):
        memory = ConversationMemory(max_history=5)
        session_id = "test_session_1"

        memory.add_user_turn(session_id, "What is Deadlock?")
        memory.add_assistant_turn(session_id, "Deadlock is a condition...", intent="concept")

        msgs = memory.get_history_messages(session_id)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "What is Deadlock?"}
        assert msgs[1] == {"role": "assistant", "content": "Deadlock is a condition..."}

    def test_context_summary_generation(self):
        memory = ConversationMemory(max_history=5)
        session_id = "test_session_2"

        memory.add_user_turn(session_id, "Explain paging")
        memory.add_assistant_turn(session_id, "Paging is a memory management scheme...")

        summary = memory.get_context_summary(session_id)
        assert "USER: Explain paging" in summary
        assert "ASSISTANT: Paging is a memory management scheme" in summary

    def test_clear_session(self):
        memory = ConversationMemory()
        session_id = "test_session_3"
        memory.add_user_turn(session_id, "Hello")
        assert memory.active_sessions() == 1

        memory.clear_session(session_id)
        assert memory.active_sessions() == 0
        assert memory.get_history_messages(session_id) == []


class TestLongTermMemory:
    """Test SQLite-backed student profile and cross-session tracking."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_syllaiq.db"
            ltm = LongTermMemory(db_path=db_path)
            yield ltm

    def test_log_query_and_topic_frequency(self, temp_db):
        session_id = "student_101"
        temp_db.log_query(
            session_id=session_id,
            query="Explain banker's algorithm",
            intent="concept",
            topic="Deadlock Avoidance",
            unit=4,
            confidence="high",
        )
        temp_db.log_query(
            session_id=session_id,
            query="Deadlock safe state example",
            intent="pyq",
            topic="Deadlock Avoidance",
            unit=4,
            confidence="medium",
        )

        profile = temp_db.get_student_profile(session_id)
        assert profile["total_queries"] == 2
        assert len(profile["top_topics"]) == 1
        assert profile["top_topics"][0]["topic"] == "Deadlock Avoidance"
        assert profile["top_topics"][0]["count"] == 2
        assert profile["units_studied"] == [4]

    def test_weak_area_and_personalization_hint(self, temp_db):
        session_id = "student_202"
        # Log query
        temp_db.log_query(
            session_id=session_id,
            query="What is semaphores vs monitors?",
            intent="concept",
            topic="Process Synchronization",
            unit=3,
            confidence="low",
        )
        # Mark as weak area
        temp_db.mark_weak_area(
            session_id=session_id,
            topic="Process Synchronization",
            unit=3,
        )

        profile = temp_db.get_student_profile(session_id)
        assert len(profile["weak_areas"]) == 1
        assert profile["weak_areas"][0]["topic"] == "Process Synchronization"

        hint = temp_db.get_personalization_hint(session_id)
        assert "Process Synchronization" in hint
        assert "Unit 3" in hint
