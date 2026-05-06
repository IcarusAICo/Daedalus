"""Shared pytest fixtures.

Most tests load the on-disk skill library once per session. Tests that mutate
the registry should request the ``isolated_registry`` fixture which restores
the global registry state on teardown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daedalus.core.registry import get_registry
from daedalus.library import load_library

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"


@pytest.fixture(scope="session", autouse=True)
def _load_library_once():
    if len(get_registry()) == 0:
        load_library(SKILLS_DIR)
    yield


@pytest.fixture
def skills_dir() -> Path:
    return SKILLS_DIR


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    (tmp_path / "traces").mkdir()
    return tmp_path


@pytest.fixture
def mock_backend():
    from daedalus.backends.mock import MockBackend

    be = MockBackend()
    be.connect()
    yield be
    be.disconnect()


@pytest.fixture
def execution_context(tmp_workspace: Path, mock_backend):
    from daedalus.core.context import ExecutionContext, TaskState
    from daedalus.tracing.recorder import TraceRecorder

    db = tmp_workspace / "tasks.db"
    tracer = TraceRecorder(
        traces_root=tmp_workspace / "traces",
        db_path=db,
        task_name="unit-test",
    )
    state = TaskState(db, tracer.task_id)
    ctx = ExecutionContext(
        task_id=tracer.task_id,
        backend=mock_backend,
        task_state=state,
        tracer=tracer,
    )
    return ctx
