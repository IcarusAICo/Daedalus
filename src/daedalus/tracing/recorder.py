"""Per-task trace recorder.

Layout::

    traces/<task_id>/
        meta.json          # task name, status, timing, program ref
        events.jsonl       # append-only structured events
        screens/0001.png   # screenshots in capture order

We also write a row per task to ``tasks.db`` (table ``traces``) for fast
indexed lookup by status / name / time range.

The event format is intentionally generic JSON: the Learner consumes it as
plain data, no Python imports required.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

EventLevel = Literal["debug", "info", "warn", "error"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task_id() -> str:
    return f"t_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class TraceEvent:
    kind: str
    ts: str = field(default_factory=_now_iso)
    level: EventLevel = "info"
    data: dict[str, Any] = field(default_factory=dict)


_TRACES_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    task_id    TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    status     TEXT NOT NULL,           -- pending | running | success | failed | aborted
    started    TEXT NOT NULL,
    finished   TEXT,
    program    TEXT,                    -- path or inline reference
    num_events INTEGER NOT NULL DEFAULT 0,
    notes      TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started);
CREATE INDEX IF NOT EXISTS idx_traces_status  ON traces(status);
"""


class TraceRecorder:
    """Owns a single task's trace dir; thread-safe for concurrent skills.

    When constructed with ``no_persist=True`` all writes (disk, SQLite,
    screenshots) are silently skipped.  The recorder is still fully usable
    — ``task_id``, ``emit``, lifecycle methods, etc. all work, they just
    produce no side-effects.  This is used by test mode so test runs do not
    accumulate trace directories or DB rows.
    """

    def __init__(
        self,
        traces_root: Path,
        db_path: Path,
        task_name: str,
        program_ref: str | None = None,
        task_id: str | None = None,
        max_events_bytes: int = 50 * 1024 * 1024,
        no_persist: bool = False,
    ) -> None:
        self.task_id = task_id or _new_task_id()
        self.task_name = task_name
        self._program_ref = program_ref
        self._traces_root = traces_root
        self._db_path = db_path
        self._task_dir = traces_root / self.task_id
        self._screens_dir = self._task_dir / "screens"
        self._events_path = self._task_dir / "events.jsonl"
        self._meta_path = self._task_dir / "meta.json"
        self._lock = threading.RLock()
        self._events_count = 0
        self._screen_count = 0
        self._status: str = "pending"
        self._max_events_bytes = max_events_bytes
        self._started: str = _now_iso()
        self._finished: str | None = None
        self._no_persist = no_persist

        if not no_persist:
            self._task_dir.mkdir(parents=True, exist_ok=True)
            self._screens_dir.mkdir(parents=True, exist_ok=True)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_TRACES_SCHEMA)
                conn.execute(
                    "INSERT OR REPLACE INTO traces(task_id,name,status,started,program,num_events) "
                    "VALUES(?,?,?,?,?,0)",
                    (self.task_id, self.task_name, self._status, self._started, program_ref),
                )
                conn.commit()
            self._write_meta()

    # -- DB helpers ----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _write_meta(self) -> None:
        if self._no_persist:
            return
        meta = {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "program_ref": self._program_ref,
            "status": self._status,
            "started": self._started,
            "finished": self._finished,
            "events": self._events_count,
            "screenshots": self._screen_count,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))

    # -- Lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self._status = "running"
            if not self._no_persist:
                self._sync_db_status()
                self._write_meta()

    def finish(self, status: Literal["success", "failed", "aborted"], notes: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._finished = _now_iso()
            if self._no_persist:
                return
            with self._connect() as conn:
                conn.execute(
                    "UPDATE traces SET status=?, finished=?, num_events=?, notes=? WHERE task_id=?",
                    (status, self._finished, self._events_count, notes, self.task_id),
                )
                conn.commit()
            self._write_meta()

    def _sync_db_status(self) -> None:
        if self._no_persist:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE traces SET status=?, num_events=? WHERE task_id=?",
                (self._status, self._events_count, self.task_id),
            )
            conn.commit()

    # -- Events --------------------------------------------------------------------

    def _rotate_events_if_needed(self) -> None:
        """Rotate events.jsonl when it exceeds the size limit. Must hold self._lock."""
        try:
            size = self._events_path.stat().st_size
        except FileNotFoundError:
            return
        if size <= self._max_events_bytes:
            return
        n = 1
        while (self._task_dir / f"events.{n}.jsonl").exists():
            n += 1
        self._events_path.rename(self._task_dir / f"events.{n}.jsonl")

    def emit(
        self,
        kind: str,
        data: dict[str, Any] | None = None,
        level: EventLevel = "info",
    ) -> None:
        evt = TraceEvent(kind=kind, level=level, data=data or {})
        with self._lock:
            self._events_count += 1
            if self._no_persist:
                return
            self._rotate_events_if_needed()
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(evt), default=str) + "\n")

    def attach_screenshot(self, png_bytes: bytes, *, width: int, height: int) -> Path:
        with self._lock:
            self._screen_count += 1
            if self._no_persist:
                return self._screens_dir / f"{self._screen_count:04d}.png"
            name = f"{self._screen_count:04d}.png"
            path = self._screens_dir / name
            path.write_bytes(png_bytes)
            self.emit(
                "screenshot",
                {"path": str(path.relative_to(self._traces_root)), "width": width, "height": height},
            )
            return path

    # -- Convenience wrappers ------------------------------------------------------

    def skill_started(
        self,
        skill_id: str,
        version: str,
        step_idx: int,
        inputs: dict[str, Any],
        content_hash: str = "",
    ) -> None:
        data: dict[str, Any] = {
            "skill_id": skill_id,
            "version": version,
            "step_idx": step_idx,
            "inputs": inputs,
        }
        if content_hash:
            data["content_hash"] = content_hash
        self.emit("skill_started", data)

    def skill_finished(
        self,
        skill_id: str,
        step_idx: int,
        outputs: dict[str, Any],
        duration_ms: float,
    ) -> None:
        self.emit(
            "skill_finished",
            {
                "skill_id": skill_id,
                "step_idx": step_idx,
                "outputs": outputs,
                "duration_ms": round(duration_ms, 3),
            },
        )

    def skill_error(
        self,
        skill_id: str,
        step_idx: int,
        error_type: str,
        message: str,
        duration_ms: float,
    ) -> None:
        self.emit(
            "skill_error",
            {
                "skill_id": skill_id,
                "step_idx": step_idx,
                "error_type": error_type,
                "message": message,
                "duration_ms": round(duration_ms, 3),
            },
            level="error",
        )

    # -- Read-side helpers ---------------------------------------------------------

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    def _read_events_from(self, path: Path) -> list[TraceEvent]:
        if not path.exists():
            return []
        out: list[TraceEvent] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(
                    TraceEvent(kind=d["kind"], ts=d["ts"], level=d.get("level", "info"), data=d.get("data", {}))
                )
        return out

    def iter_events(self) -> Iterable[TraceEvent]:
        rotated: list[tuple[int, Path]] = []
        for p in self._task_dir.glob("events.*.jsonl"):
            stem_parts = p.name.split(".")
            try:
                idx = int(stem_parts[1])
            except (IndexError, ValueError):
                continue
            rotated.append((idx, p))
        rotated.sort(key=lambda t: t[0])

        out: list[TraceEvent] = []
        for _idx, p in rotated:
            out.extend(self._read_events_from(p))
        out.extend(self._read_events_from(self._events_path))
        return out


def list_traces(db_path: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT task_id,name,status,started,finished,num_events FROM traces "
            "ORDER BY started DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "task_id": r[0],
            "name": r[1],
            "status": r[2],
            "started": r[3],
            "finished": r[4],
            "num_events": r[5],
        }
        for r in rows
    ]
