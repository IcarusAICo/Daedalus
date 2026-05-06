"""Per-run typed data store for structured data exchange between skills.

Skills use the RunStore to accumulate structured data (e.g. lists of coordinates,
detected elements, observations) that can be consumed by later skills. Unlike
the flat TaskState key/value store, RunStore supports typed tables with
append, query, and count operations.

Backed by SQLite with WAL journaling for concurrent access from daemons.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class RunStore:
    """Per-run typed data store backed by SQLite.
    
    Each table is scoped to a task_id so concurrent/historical runs don't
    collide. Tables are created on-demand with a caller-specified schema.
    """

    def __init__(self, db_path: Path, task_id: str) -> None:
        self._db_path = db_path
        self._task_id = task_id
        self._lock = threading.RLock()
        self._tables: dict[str, dict[str, str]] = {}  # name -> {col: type}
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the metadata table that tracks which tables exist for this task
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _runstore_meta ("
                "  task_id TEXT NOT NULL,"
                "  table_name TEXT NOT NULL,"
                "  schema_json TEXT NOT NULL,"
                "  PRIMARY KEY (task_id, table_name)"
                ")"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _real_table_name(self, name: str) -> str:
        """Namespace table names to avoid collisions."""
        safe = name.replace("-", "_").replace(".", "_")
        # Use a short prefix + task_id hash to keep names manageable
        tid_short = self._task_id.replace("-", "_")[:20]
        return f"rs_{tid_short}_{safe}"

    def create_table(self, name: str, schema: dict[str, str]) -> None:
        """Create a typed table.

        Args:
            name: Logical table name (e.g. "spots", "grid_edges").
            schema: Column name -> type mapping. Types: "int", "float", "str", "bool", "json".
                    An auto-increment "_rowid" column is always added.
        """
        with self._lock:
            if name in self._tables:
                return  # idempotent
            real = self._real_table_name(name)
            type_map = {"int": "INTEGER", "float": "REAL", "str": "TEXT", "bool": "INTEGER", "json": "TEXT"}
            cols = ["_rowid INTEGER PRIMARY KEY AUTOINCREMENT"]
            for col, typ in schema.items():
                sql_type = type_map.get(typ, "TEXT")
                cols.append(f'"{col}" {sql_type}')
            ddl = f'CREATE TABLE IF NOT EXISTS "{real}" ({", ".join(cols)})'
            with self._connect() as conn:
                conn.execute(ddl)
                conn.execute(
                    "INSERT OR REPLACE INTO _runstore_meta(task_id, table_name, schema_json) VALUES(?,?,?)",
                    (self._task_id, name, json.dumps(schema)),
                )
                conn.commit()
            self._tables[name] = schema

    def _ensure_table(self, name: str) -> None:
        """Auto-create if we haven't seen this table yet in this process."""
        if name not in self._tables:
            # Try loading from meta
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT schema_json FROM _runstore_meta WHERE task_id=? AND table_name=?",
                    (self._task_id, name),
                ).fetchone()
            if row:
                schema = json.loads(row[0])
                self.create_table(name, schema)
            else:
                raise KeyError(f"RunStore table {name!r} does not exist. Call create_table() first.")

    def append(self, table: str, row: dict[str, Any]) -> int:
        """Append a row, return the row id."""
        self._ensure_table(table)
        schema = self._tables[table]
        real = self._real_table_name(table)
        cols = []
        vals = []
        for col in schema:
            if col in row:
                v = row[col]
                if schema[col] == "json" and not isinstance(v, str):
                    v = json.dumps(v, default=str)
                elif schema[col] == "bool":
                    v = int(bool(v))
                cols.append(f'"{col}"')
                vals.append(v)
        placeholders = ", ".join("?" for _ in vals)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f'INSERT INTO "{real}" ({", ".join(cols)}) VALUES ({placeholders})',
                vals,
            )
            conn.commit()
            return cur.lastrowid or 0

    def query(self, table: str, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Query rows, optionally filtered by column equality."""
        self._ensure_table(table)
        schema = self._tables[table]
        real = self._real_table_name(table)
        sql = f'SELECT _rowid, * FROM "{real}"'
        params: list[Any] = []
        if where:
            clauses = []
            for k, v in where.items():
                clauses.append(f'"{k}" = ?')
                params.append(v)
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(dict(r), schema) for r in rows]

    def all_rows(self, table: str) -> list[dict[str, Any]]:
        """Return all rows from a table."""
        return self.query(table)

    def count(self, table: str) -> int:
        """Return the number of rows in a table."""
        self._ensure_table(table)
        real = self._real_table_name(table)
        with self._lock, self._connect() as conn:
            row = conn.execute(f'SELECT COUNT(*) FROM "{real}"').fetchone()
        return row[0] if row else 0

    def table_names(self) -> list[str]:
        """Return all table names for this task."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT table_name FROM _runstore_meta WHERE task_id=? ORDER BY table_name",
                (self._task_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Dump all tables as {table_name: [rows...]} for LLM context injection."""
        result: dict[str, list[dict[str, Any]]] = {}
        for name in self.table_names():
            try:
                result[name] = self.all_rows(name)
            except Exception:
                result[name] = []
        return result

    def _decode_row(self, row: dict[str, Any], schema: dict[str, str]) -> dict[str, Any]:
        """Decode JSON columns and bools back to native types."""
        out: dict[str, Any] = {}
        for k, v in row.items():
            if k == "_rowid":
                out[k] = v
                continue
            col_type = schema.get(k, "str")
            if col_type == "json" and isinstance(v, str):
                try:
                    out[k] = json.loads(v)
                except json.JSONDecodeError:
                    out[k] = v
            elif col_type == "bool":
                out[k] = bool(v)
            else:
                out[k] = v
        return out
