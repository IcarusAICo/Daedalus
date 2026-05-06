"""Persistent cross-run memory for Daedalus.

Stores learned facts, strategies, and skill outcomes across multiple
task runs. The Planner reads from memory to inform its decisions;
the Evaluator and Learner write to it after each run.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Fact:
    id: int
    category: str
    content: str
    source_task_id: str
    created_at: str
    tags: list[str] = field(default_factory=list)


@dataclass
class SkillOutcome:
    skill_id: str
    task_id: str
    success: bool
    notes: str
    created_at: str


_CATEGORIES = frozenset({
    "app_behavior",
    "site_layout",
    "failure_mode",
    "strategy",
    "skill_pattern",
    "general",
})


class AgentMemory:
    """Persistent memory store across runs, backed by SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS facts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        category    TEXT NOT NULL,
        content     TEXT NOT NULL,
        source_task_id TEXT NOT NULL,
        tags        TEXT NOT NULL DEFAULT '[]',
        created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS skill_outcomes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        skill_id    TEXT NOT NULL,
        task_id     TEXT NOT NULL,
        success     INTEGER NOT NULL,
        notes       TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
    CREATE INDEX IF NOT EXISTS idx_skill_outcomes_skill ON skill_outcomes(skill_id);
    """

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add_fact(
        self,
        category: str,
        content: str,
        source_task_id: str,
        tags: list[str] | None = None,
    ) -> int:
        """Store a learned fact. Returns the fact id."""
        if category not in _CATEGORIES:
            category = "general"
        ts = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags or [])
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO facts(category, content, source_task_id, tags, created_at) "
                "VALUES(?,?,?,?,?)",
                (category, content, source_task_id, tags_json, ts),
            )
            conn.commit()
            return cur.lastrowid or 0

    def recall(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[Fact]:
        """Keyword search over facts using simple BM25-like scoring."""
        with self._lock, self._connect() as conn:
            sql = "SELECT id, category, content, source_task_id, tags, created_at FROM facts"
            params: list[Any] = []
            if category:
                sql += " WHERE category = ?"
                params.append(category)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Simple BM25-inspired scoring: tf * idf
        query_terms = _tokenize(query)
        if not query_terms:
            return [self._row_to_fact(r) for r in rows[:limit]]

        doc_count = len(rows)
        # Document frequency for each term
        df: Counter[str] = Counter()
        doc_tokens: list[list[str]] = []
        for row in rows:
            tokens = _tokenize(row[2])  # content
            doc_tokens.append(tokens)
            for t in set(tokens):
                df[t] += 1

        scored: list[tuple[float, Any]] = []
        for i, row in enumerate(rows):
            tokens = doc_tokens[i]
            if not tokens:
                continue
            tf_map = Counter(tokens)
            score = 0.0
            for qt in query_terms:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                idf = math.log((doc_count + 1) / (df.get(qt, 0) + 0.5))
                score += tf * idf
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_fact(r) for _, r in scored[:limit]]

    def add_skill_outcome(
        self,
        skill_id: str,
        task_id: str,
        success: bool,
        notes: str = "",
    ) -> None:
        """Track which skills worked/failed for which tasks."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO skill_outcomes(skill_id, task_id, success, notes, created_at) "
                "VALUES(?,?,?,?,?)",
                (skill_id, task_id, int(success), notes, ts),
            )
            conn.commit()

    def get_skill_stats(self, skill_id: str) -> dict[str, Any]:
        """Get success/failure stats for a skill."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT success, COUNT(*) FROM skill_outcomes WHERE skill_id=? GROUP BY success",
                (skill_id,),
            ).fetchall()
        stats: dict[str, int] = {"successes": 0, "failures": 0}
        for success, count in rows:
            if success:
                stats["successes"] = count
            else:
                stats["failures"] = count
        return stats

    def all_facts(self, category: str | None = None, limit: int = 100) -> list[Fact]:
        """Return all facts, optionally filtered by category."""
        with self._lock, self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT id, category, content, source_task_id, tags, created_at "
                    "FROM facts WHERE category=? ORDER BY created_at DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, category, content, source_task_id, tags, created_at "
                    "FROM facts ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    @staticmethod
    def _row_to_fact(row: Any) -> Fact:
        return Fact(
            id=row[0],
            category=row[1],
            content=row[2],
            source_task_id=row[3],
            tags=json.loads(row[4]) if row[4] else [],
            created_at=row[5],
        )


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) > 1]
