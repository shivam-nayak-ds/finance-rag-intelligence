"""SQLite store and seeder for structured RGPV Previous Year Questions (PYQs)."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from config.settings import ROOT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = ROOT_DIR / "data" / "syllaiq.db"
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "raw" / "os" / "pyqs" / "rgpv_os_pyqs_dataset.json"


class PYQDatabase:
    """
    Manages structured storage and queries for RGPV Operating Systems PYQs in SQLite.

    Provides auto-seeding from rgpv_os_pyqs_dataset.json and read-only query execution.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        dataset_path: Path = DEFAULT_DATASET_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.dataset_path = Path(dataset_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        self.seed_if_empty()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager guaranteeing connection commit and immediate close."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Create pyqs table and helpful indexes if they do not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pyqs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id            TEXT NOT NULL,
                    year                INTEGER NOT NULL,
                    session             TEXT,
                    semester            INTEGER,
                    question_no         TEXT,
                    unit                INTEGER,
                    topic               TEXT,
                    marks               INTEGER,
                    question_text       TEXT NOT NULL,
                    question_text_hindi TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_pyqs_unit ON pyqs(unit);
                CREATE INDEX IF NOT EXISTS idx_pyqs_year ON pyqs(year);
                CREATE INDEX IF NOT EXISTS idx_pyqs_marks ON pyqs(marks);
                CREATE INDEX IF NOT EXISTS idx_pyqs_topic ON pyqs(topic);
            """)

    def count(self) -> int:
        """Return total number of PYQ questions in the database."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pyqs")
            return cursor.fetchone()[0]

    def seed_if_empty(self) -> int:
        """Seed the pyqs table from dataset json if currently empty."""
        current_count = self.count()
        if current_count > 0:
            logger.debug("PYQ database already seeded with %d questions.", current_count)
            return current_count

        if not self.dataset_path.exists():
            logger.warning("PYQ dataset file not found at %s. Seeding skipped.", self.dataset_path)
            return 0

        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            logger.error("Failed to load PYQ dataset JSON: %s", err)
            return 0

        inserted = 0
        with self._connect() as conn:
            for paper in data:
                paper_id = paper.get("paper_id", "")
                year = paper.get("year", 0)
                session = paper.get("session", "")
                semester = paper.get("semester", 4)
                questions = paper.get("questions", [])

                for q in questions:
                    conn.execute(
                        """INSERT INTO pyqs (
                            paper_id, year, session, semester, question_no,
                            unit, topic, marks, question_text, question_text_hindi
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            paper_id,
                            year,
                            session,
                            semester,
                            q.get("question_no"),
                            q.get("unit"),
                            q.get("topic"),
                            q.get("marks"),
                            q.get("question_text", "").strip(),
                            q.get("question_text_hindi", "").strip(),
                        ),
                    )
                    inserted += 1

        logger.info("Successfully seeded %d PYQ questions into SQLite database.", inserted)
        return inserted

    def execute_read_query(self, sql: str) -> Tuple[List[str], List[List[Any]]]:
        """
        Execute a safe read-only query and return (columns, rows).
        """
        with self._connect() as conn:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [list(row) for row in cursor.fetchall()]
            return columns, rows

    def get_schema_description(self) -> str:
        """Returns schema definition and sample rows for LLM prompt context."""
        return (
            "Table: pyqs\n"
            "Columns:\n"
            "  - id (INTEGER PRIMARY KEY)\n"
            "  - paper_id (TEXT, e.g. 'RGPV_OS_JUN_2023')\n"
            "  - year (INTEGER, e.g. 2020, 2021, 2022, 2023)\n"
            "  - session (TEXT, e.g. 'June', 'December')\n"
            "  - semester (INTEGER, e.g. 4)\n"
            "  - question_no (TEXT, e.g. '1.a', '2.b')\n"
            "  - unit (INTEGER, syllabus unit 1 to 5)\n"
            "  - topic (TEXT, e.g. 'Deadlock Avoidance', 'Process States & PCB', 'Paging')\n"
            "  - marks (INTEGER, e.g. 7, 14)\n"
            "  - question_text (TEXT, English question content)\n"
            "  - question_text_hindi (TEXT, Hindi question content)\n"
        )


_pyq_db_instance: Optional[PYQDatabase] = None


def get_pyq_database() -> PYQDatabase:
    """Singleton getter for PYQDatabase."""
    global _pyq_db_instance
    if _pyq_db_instance is None:
        _pyq_db_instance = PYQDatabase()
    return _pyq_db_instance
