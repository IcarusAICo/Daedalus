"""Execution context passed to every skill invocation.

The context holds:
- a handle to the current backend (RemoteDesktop)
- the per-task TaskState (key/value store backed by SQLite)
- a TraceRecorder for structured event logging
- an LLM gateway handle (None in Phase 0 if not configured)
- an abort flag (set by the overlay hotkey)

Skills should treat the context as read-mostly: write to TaskState and the
trace, do not stash mutable globals on it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daedalus.backends.protocol import RemoteDesktop
    from daedalus.core.store import RunStore
    from daedalus.llm.gateway import LLMGateway
    from daedalus.tracing.recorder import TraceRecorder

# LLM vision models internally resize images to a grid of patches. The
# empirically-determined maximum width (in patches * 28px) for Anthropic
# Claude Opus 4.6 is 49 patches = 1372 pixels. When the screen is wider
# than this, all LLM-estimated coordinates are in this smaller space and
# need scaling back up.
_LLM_PATCH_SIZE = 28
_LLM_MAX_PATCHES_W = 49
_LLM_INTERNAL_W = _LLM_MAX_PATCHES_W * _LLM_PATCH_SIZE  # 1372


def compute_coordinate_scale(screen_width: int) -> float:
    """Return the ratio screen_width / llm_internal_width.

    When the screen is wider than what the LLM processes internally, all
    coordinates from the LLM need to be multiplied by this factor. When
    the screen is small enough that no downscaling occurs, returns 1.0.
    """
    if screen_width <= _LLM_INTERNAL_W:
        return 1.0
    return screen_width / _LLM_INTERNAL_W


def llm_image_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    """Return the (width, height) to downscale screenshots to for LLM consumption.

    Preserves aspect ratio and targets the LLM's internal processing width.
    Returns original dimensions if already small enough.
    """
    if screen_width <= _LLM_INTERNAL_W:
        return screen_width, screen_height
    scale = _LLM_INTERNAL_W / screen_width
    return _LLM_INTERNAL_W, round(screen_height * scale)


class TaskState:
    """A small thread-safe key/value store scoped to one task run.

    Backed by a SQLite table inside ``tasks.db``. Daemon skills publish updates
    here; the executor and other skills read them.

    Values are JSON-encoded so the Learner can replay a task's state evolution.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS task_state (
        task_id TEXT NOT NULL,
        key     TEXT NOT NULL,
        value   TEXT NOT NULL,
        updated TEXT NOT NULL,
        PRIMARY KEY (task_id, key)
    );
    CREATE TABLE IF NOT EXISTS task_state_history (
        task_id TEXT NOT NULL,
        key     TEXT NOT NULL,
        value   TEXT NOT NULL,
        ts      TEXT NOT NULL
    );
    """

    def __init__(self, db_path: Path, task_id: str) -> None:
        self._db_path = db_path
        self._task_id = task_id
        self._lock = threading.RLock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def set(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, default=str)
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO task_state(task_id,key,value,updated) VALUES(?,?,?,?) "
                "ON CONFLICT(task_id,key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                (self._task_id, key, encoded, ts),
            )
            conn.execute(
                "INSERT INTO task_state_history(task_id,key,value,ts) VALUES(?,?,?,?)",
                (self._task_id, key, encoded, ts),
            )
            conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM task_state WHERE task_id=? AND key=?",
                (self._task_id, key),
            ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def keys(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT key FROM task_state WHERE task_id=? ORDER BY key",
                (self._task_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def snapshot(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT key,value FROM task_state WHERE task_id=?",
                (self._task_id,),
            ).fetchall()
        return {k: json.loads(v) for k, v in rows}


@dataclass
class ExecutionContext:
    """Everything a skill needs at run time. Passed positionally as ``ctx``."""

    task_id: str
    backend: RemoteDesktop
    task_state: TaskState
    tracer: TraceRecorder
    store: RunStore | None = None
    llm: LLMGateway | None = None
    config: dict[str, Any] = field(default_factory=dict)
    abort_event: threading.Event = field(default_factory=threading.Event)
    coordinate_scale: float = field(default=1.0)

    def aborted(self) -> bool:
        return self.abort_event.is_set()
