"""Text-to-SQL engine for answering analytical and statistical questions over PYQs."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from config.settings import GROQ_MODEL
from database.pyq_db import PYQDatabase, get_pyq_database
from models.documents import Citation, ConfidenceLevel, Document, SourceType
from models.responses import RAGResult
from utils.logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_FORBIDDEN_SQL_WORDS = {
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "detach", "truncate", "exec", "pragma", "vacuum"
}

_SQL_PROMPT_TEMPLATE = """\
You are an expert SQLite developer for an Operating Systems university exam database.
Convert the user's natural language question into a single valid, read-only SQLite SELECT query.

Database Schema:
{schema}

Rules:
1. Write ONLY the SQL query. Do not wrap in markdown or explanation.
2. Only use SELECT statements. Never modify data.
3. For fuzzy topic matches, use LIKE '%term%' with LOWER(topic) or LOWER(question_text).
4. If the user asks for "most asked", "top", or "frequency", use GROUP BY and ORDER BY COUNT(*) DESC.
5. If the user asks for questions, select at least: year, question_no, unit, marks, question_text.
6. Keep results focused with LIMIT 20 unless a specific limit is requested.

User Question: {query}
SQLite Query:"""

_EXPLAIN_PROMPT_TEMPLATE = """\
You are an AI exam assistant. The user asked: "{query}"
The database returned the following data:
{data}

Provide a concise, helpful summary in English (mixing Hinglish is OK if natural).
Highlight key statistics and patterns. Keep it under 4-5 bullet points.
"""


class SQLQueryEngine:
    """
    Translates analytical and counting questions into SQL, runs on SQLite,
    and returns a structured RAGResult.
    """

    def __init__(
        self,
        db: Optional[PYQDatabase] = None,
        model: str = GROQ_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self.db = db or get_pyq_database()
        self.model = model
        self._groq_client = None

        key = api_key or GROQ_API_KEY
        if key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=key)
            except Exception as err:
                logger.warning("Groq init failed for SQLQueryEngine: %s", err)

    def is_safe_sql(self, sql: str) -> Tuple[bool, str]:
        """Validate that SQL statement is safe and read-only."""
        cleaned = sql.strip().rstrip(";").strip()
        tokens = re.findall(r"\b\w+\b", cleaned.lower())

        if not tokens:
            return False, "Empty query."

        # Check for forbidden keywords
        for token in tokens:
            if token in _FORBIDDEN_SQL_WORDS:
                return False, f"Forbidden keyword detected: {token}"

        # Must be a SELECT or WITH statement
        first_word = tokens[0]
        if first_word not in ("select", "with"):
            return False, "Query must start with SELECT or WITH."

        return True, "Safe."

    def generate_sql(self, query: str) -> str:
        """Convert natural language to SQLite query using LLM."""
        if self._groq_client is None:
            # Fallback heuristic query generator if Groq is unavailable
            return self._heuristic_sql(query)

        try:
            prompt = _SQL_PROMPT_TEMPLATE.format(
                schema=self.db.get_schema_description(),
                query=query.strip(),
            )
            response = self._groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw_sql = response.choices[0].message.content.strip()
            # Clean markdown formatting if present
            raw_sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql, flags=re.IGNORECASE)
            raw_sql = re.sub(r"\s*```$", "", raw_sql)
            return raw_sql.strip()
        except Exception as err:
            logger.warning("LLM Text-to-SQL generation failed: %s — using heuristic", err)
            return self._heuristic_sql(query)

    def _heuristic_sql(self, query: str) -> str:
        """Rule-based SQL generator for common query patterns when LLM is unavailable."""
        q = query.lower()

        # Check for year match
        year_match = re.search(r"\b(201\d|202\d)\b", q)
        year_clause = f"year = {year_match.group(1)}" if year_match else None

        # Check for unit match
        unit_match = re.search(r"\bunit\s*([1-5])\b", q)
        unit_clause = f"unit = {unit_match.group(1)}" if unit_match else None

        # Check for marks match
        marks_match = re.search(r"\b([0-9]|1[0-4])\s*marks?\b", q)
        marks_clause = f"marks = {marks_match.group(1)}" if marks_match else None

        conditions = [c for c in (year_clause, unit_clause, marks_clause) if c]
        where_str = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if any(w in q for w in ("how many", "count", "kitne", "kitni")):
            return f"SELECT COUNT(*) AS total_questions FROM pyqs {where_str};"
        if any(w in q for w in ("most", "top", "frequency", "repeated")):
            return f"SELECT topic, COUNT(*) AS count FROM pyqs {where_str} GROUP BY topic ORDER BY count DESC LIMIT 10;"

        return f"SELECT year, question_no, unit, marks, question_text FROM pyqs {where_str} LIMIT 20;"

    def execute_and_format(self, query: str) -> RAGResult:
        """
        End-to-end Text-to-SQL pipeline:
          1. Generate SQL
          2. Validate SQL
          3. Execute SQL
          4. Format as Markdown table and natural answer
        """
        sql = self.generate_sql(query)
        logger.info("Generated SQL for query %r: %s", query[:60], sql)

        is_safe, reason = self.is_safe_sql(sql)
        if not is_safe:
            logger.warning("Unsafe SQL generated: %s (Reason: %s)", sql, reason)
            return RAGResult(
                answer=f"Could not safely process SQL query: {reason}",
                citations=[],
                confidence=ConfidenceLevel.LOW,
                intent="analytics",
                error=reason,
                status="failed",
            )

        try:
            columns, rows = self.db.execute_read_query(sql)
        except Exception as err:
            logger.error("SQL execution error: %s (SQL: %s)", err, sql)
            return RAGResult(
                answer=f"Database query execution failed: {err}",
                citations=[],
                confidence=ConfidenceLevel.LOW,
                intent="analytics",
                error=str(err),
                status="failed",
            )

        if not rows:
            return RAGResult(
                answer=(
                    f"PYQ database mein is criteria par koi record nahi mila.\n\n"
                    f"**Executed SQL:**\n```sql\n{sql}\n```"
                ),
                citations=[],
                confidence=ConfidenceLevel.HIGH,
                intent="analytics",
                status="success",
            )

        # Build Markdown Table
        table_lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in rows[:25]:  # show up to 25 rows in table
            row_str = " | ".join(str(cell).replace("\n", " ").strip()[:100] for cell in row)
            table_lines.append(f"| {row_str} |")

        table_md = "\n".join(table_lines)
        total_records = len(rows)

        # Generate summary
        answer = (
            f"### 📊 PYQ Database Analysis ({total_records} records found)\n\n"
            f"{table_md}\n\n"
            f"**Executed SQL Query:**\n```sql\n{sql}\n```"
        )

        citations: List[Citation] = []
        for i, row in enumerate(rows[:5], start=1):
            row_dict = dict(zip(columns, row))
            doc = Document(
                chunk_id=f"sql_row_{i}",
                text=str(row_dict.get("question_text") or str(row_dict)),
                source_type=SourceType.PYQ,
                year=row_dict.get("year"),
                marks=row_dict.get("marks"),
                unit=row_dict.get("unit"),
                topic=str(row_dict.get("topic", "")),
            )
            citations.append(Citation.from_document(doc, i))

        return RAGResult(
            answer=answer,
            citations=citations,
            confidence=ConfidenceLevel.HIGH,
            intent="analytics",
            status="success",
        )
