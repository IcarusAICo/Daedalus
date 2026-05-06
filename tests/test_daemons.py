"""Daemon lifecycle: start, publish to task_state, stop cleanly."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from daedalus.backends.mock import MockBackend
from daedalus.core.context import ExecutionContext, TaskState
from daedalus.executor.daemons import DaemonSpec, start_daemons, stop_daemons
from daedalus.executor.dsl import parse_program
from daedalus.executor.runner import SequentialExecutor
from daedalus.tracing.recorder import TraceRecorder


def _ctx(tmp_path: Path):
    be = MockBackend()
    be.connect()
    db = tmp_path / "tasks.db"
    tracer = TraceRecorder(traces_root=tmp_path / "traces", db_path=db, task_name="d")
    return ExecutionContext(
        task_id=tracer.task_id,
        backend=be,
        task_state=TaskState(db, tracer.task_id),
        tracer=tracer,
    )


def test_tick_counter_publishes_to_task_state(tmp_workspace):
    ctx = _ctx(tmp_workspace)
    handles = start_daemons(
        [DaemonSpec(skill="tick_counter", inputs={"interval_ms": 30, "key": "ping"})],
        ctx,
    )
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            snap = ctx.task_state.get("ping")
            if snap and snap["count"] >= 3:
                break
            time.sleep(0.05)
    finally:
        stop_daemons(handles, ctx)
    snap = ctx.task_state.get("ping")
    assert snap is not None
    assert snap["count"] >= 3


def test_daemon_stops_on_abort(tmp_workspace):
    ctx = _ctx(tmp_workspace)
    handles = start_daemons(
        [DaemonSpec(skill="tick_counter", inputs={"interval_ms": 50})],
        ctx,
    )
    time.sleep(0.15)
    ctx.abort_event.set()
    stop_daemons(handles, ctx)
    # After stop, no new updates should appear.
    final = handles[0].update_count
    time.sleep(0.2)
    assert handles[0].update_count == final


def test_max_ticks_terminates_daemon(tmp_workspace):
    ctx = _ctx(tmp_workspace)
    handles = start_daemons(
        [DaemonSpec(skill="tick_counter", inputs={"interval_ms": 20, "max_ticks": 4})],
        ctx,
    )
    time.sleep(0.5)
    stop_daemons(handles, ctx)
    assert handles[0].update_count == 4


def test_daemon_kind_validated_at_validation():
    p = parse_program(
        """
name: bad
daemons:
  - skill: click_mouse   # atomic, not daemon
    inputs: {x: 10, y: 10}
steps:
  - skill: wait
    inputs: {ms: 1}
"""
    )
    from daedalus.core.errors import ProgramValidationError
    from daedalus.executor.dsl import validate_program_against_registry

    with pytest.raises(ProgramValidationError):
        validate_program_against_registry(p)


def test_executor_runs_daemons_and_steps(tmp_workspace):
    p = parse_program(
        """
name: daemon-and-steps
daemons:
  - skill: tick_counter
    inputs: {interval_ms: 30, key: hb}
steps:
  - skill: wait
    inputs: {ms: 200}
  - skill: view_screen
    inputs: {}
"""
    )
    be = MockBackend()
    ex = SequentialExecutor(
        backend=be,
        traces_root=tmp_workspace / "traces",
        tasks_db=tmp_workspace / "tasks.db",
    )
    result = ex.run(p)
    assert result.status == "success"

    state = TaskState(tmp_workspace / "tasks.db", result.task_id)
    hb = state.get("hb")
    assert hb is not None
    assert hb["count"] >= 2  # we waited 200ms with 30ms ticks


def test_program_with_only_daemons_is_invalid(tmp_workspace):
    """We require at least one step; daemons alone don't count."""
    from daedalus.core.errors import ProgramValidationError

    with pytest.raises(ProgramValidationError):
        parse_program(
            """
name: bad
daemons:
  - skill: tick_counter
    inputs: {interval_ms: 50}
steps: []
"""
        )
