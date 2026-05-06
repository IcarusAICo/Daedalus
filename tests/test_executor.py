"""End-to-end Phase 0 loop: parse program -> run on MockBackend -> read trace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daedalus.backends.mock import MockBackend
from daedalus.core.errors import DaedalusError
from daedalus.executor.dsl import parse_program
from daedalus.executor.runner import SequentialExecutor
from daedalus.tracing.recorder import list_traces


def _smoke_program():
    return parse_program(
        """
name: smoke
steps:
  - skill: view_screen
    inputs: {}
  - skill: click_mouse
    inputs: {x: 100, y: 200}
  - skill: type_text
    inputs: {text: "hello"}
  - skill: wait
    inputs: {ms: 1}
"""
    )


def test_full_loop_against_mock(tmp_workspace: Path):
    be = MockBackend()
    ex = SequentialExecutor(
        backend=be,
        traces_root=tmp_workspace / "traces",
        tasks_db=tmp_workspace / "tasks.db",
    )
    p = _smoke_program()
    result = ex.run(p)
    assert result.status == "success"
    assert len(result.steps) == 4
    assert all(s.status == "success" for s in result.steps)


def test_trace_dir_and_db_written(tmp_workspace: Path):
    be = MockBackend()
    ex = SequentialExecutor(
        backend=be,
        traces_root=tmp_workspace / "traces",
        tasks_db=tmp_workspace / "tasks.db",
    )
    result = ex.run(_smoke_program())
    task_dir = tmp_workspace / "traces" / result.task_id
    assert task_dir.is_dir()
    assert (task_dir / "events.jsonl").exists()
    assert (task_dir / "meta.json").exists()
    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["status"] == "success"
    assert meta["events"] >= 4

    rows = list_traces(tmp_workspace / "tasks.db")
    assert len(rows) == 1
    assert rows[0]["task_id"] == result.task_id


def test_executor_propagates_step_failure(tmp_workspace: Path):
    """A step whose Pydantic validation fails fast at validate_program_against_registry."""
    bad = parse_program(
        """
name: bad
steps:
  - skill: wait
    inputs: {ms: -1}
"""
    )
    be = MockBackend()
    ex = SequentialExecutor(
        backend=be,
        traces_root=tmp_workspace / "traces",
        tasks_db=tmp_workspace / "tasks.db",
    )
    with pytest.raises(DaedalusError):
        ex.run(bad)


def test_save_as_writes_to_task_state(tmp_workspace: Path):
    p = parse_program(
        """
name: persist
steps:
  - skill: click_mouse
    inputs: {x: 5, y: 6}
    save_as: last_click
"""
    )
    be = MockBackend()
    ex = SequentialExecutor(
        backend=be,
        traces_root=tmp_workspace / "traces",
        tasks_db=tmp_workspace / "tasks.db",
    )
    result = ex.run(p)
    # The TaskState is keyed per task; reconstruct one to peek at it.
    from daedalus.core.context import TaskState

    state = TaskState(tmp_workspace / "tasks.db", result.task_id)
    saved = state.get("last_click")
    assert saved is not None
    assert saved["clicked_at"] == [5, 6]
