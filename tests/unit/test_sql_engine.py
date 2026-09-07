"""Unit tests for SQL Database and Text-to-SQL Query Engine."""

import tempfile
from pathlib import Path
import pytest

from database.pyq_db import PYQDatabase
from models.documents import Intent
from pipeline.rag_pipeline import RAGPipeline
from tools.sql_query_engine import SQLQueryEngine


@pytest.fixture
def temp_pyq_db():
    """Create a temporary SQLite database seeded from the real PYQ dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_pyqs.db"
        db = PYQDatabase(db_path=db_path)
        yield db


class TestPYQDatabase:
    """Test SQLite seeding and query operations."""

    def test_seeding_and_count(self, temp_pyq_db):
        count = temp_pyq_db.count()
        assert count > 0, "Database should be seeded with PYQ questions."

    def test_execute_read_query(self, temp_pyq_db):
        cols, rows = temp_pyq_db.execute_read_query("SELECT year, marks, unit FROM pyqs LIMIT 5;")
        assert cols == ["year", "marks", "unit"]
        assert len(rows) == 5
        assert all(isinstance(r[0], int) for r in rows)


class TestSQLQueryEngine:
    """Test SQL safety validation and query execution."""

    def test_safe_sql_validation(self, temp_pyq_db):
        engine = SQLQueryEngine(db=temp_pyq_db)

        # Valid queries
        safe, _ = engine.is_safe_sql("SELECT * FROM pyqs WHERE unit = 1;")
        assert safe is True

        safe, _ = engine.is_safe_sql("WITH counts AS (SELECT unit, count(*) FROM pyqs GROUP BY unit) SELECT * FROM counts;")
        assert safe is True

        # Unsafe queries
        safe, msg = engine.is_safe_sql("DROP TABLE pyqs;")
        assert safe is False
        assert "drop" in msg.lower()

        safe, msg = engine.is_safe_sql("DELETE FROM pyqs WHERE id = 1;")
        assert safe is False
        assert "delete" in msg.lower()

        safe, msg = engine.is_safe_sql("UPDATE pyqs SET marks = 100;")
        assert safe is False
        assert "update" in msg.lower()

    def test_heuristic_sql_generation(self, temp_pyq_db):
        engine = SQLQueryEngine(db=temp_pyq_db)

        # Counting query
        sql = engine._heuristic_sql("How many questions were asked from unit 3 in 2023?")
        assert "count" in sql.lower()
        assert "unit = 3" in sql.lower()
        assert "year = 2023" in sql.lower()

        # Frequency query
        sql_freq = engine._heuristic_sql("Most frequent topics in OS")
        assert "group by topic" in sql_freq.lower()
        assert "order by count desc" in sql_freq.lower()

    def test_execute_and_format(self, temp_pyq_db):
        engine = SQLQueryEngine(db=temp_pyq_db)
        result = engine.execute_and_format("How many questions are there in unit 4?")
        assert result.status == "success"
        assert result.intent == "analytics"
        assert "PYQ Database Analysis" in result.answer
        assert "Executed SQL Query" in result.answer


class TestPipelineRouting:
    """Test that analytical queries are routed to SQL and conceptual queries to RAG."""

    def test_intent_classification(self):
        pipeline = RAGPipeline()

        # Analytical intent
        assert pipeline._classify_intent("How many 7 marks questions were asked in 2023?") == Intent.PYQ_ANALYTICS
        assert pipeline._classify_intent("Give me a table of most repeated topics") == Intent.PYQ_ANALYTICS
        assert pipeline._classify_intent("Kitne questions aaye the unit 1 se?") == Intent.PYQ_ANALYTICS

        # Conceptual intent
        assert pipeline._classify_intent("What is Deadlock and explain its conditions?") == Intent.CONCEPT_EXPLANATION

        # Syllabus intent
        assert pipeline._classify_intent("What is in Unit 4 syllabus?") == Intent.SYLLABUS_LOOKUP
