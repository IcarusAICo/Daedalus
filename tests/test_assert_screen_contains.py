"""assert_screen_contains: VLM call mocked via FakeGateway."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daedalus.core.context import ExecutionContext, TaskState
from daedalus.core.registry import get_registry
from daedalus.llm.fakes import FakeGateway
from daedalus.tracing.recorder import TraceRecorder


def _ctx(tmp_path: Path, mock_backend, gateway):
    db = tmp_path / "tasks.db"
    tracer = TraceRecorder(traces_root=tmp_path / "traces", db_path=db, task_name="t")
    return ExecutionContext(
        task_id=tracer.task_id,
        backend=mock_backend,
        task_state=TaskState(db, tracer.task_id),
        tracer=tracer,
        llm=gateway,
    )


def test_assert_returns_true_when_vlm_says_so(tmp_workspace, mock_backend):
    gw = FakeGateway(responses=[
        json.dumps({"verdict": True, "confidence": "high", "explanation": "Notepad with 'hello' is visible"})
    ])
    ctx = _ctx(tmp_workspace, mock_backend, gw)
    cls = get_registry().get("assert_screen_contains").cls
    out = cls().run(cls.Inputs(claim="Notepad is open with 'hello'"), ctx)
    assert out.verdict is True
    assert out.confidence == "high"
    assert "Notepad" in out.explanation
    # Vision call goes to the configured role
    assert gw.calls[0].role == "vision"


def test_assert_returns_false_when_vlm_disagrees(tmp_workspace, mock_backend):
    gw = FakeGateway(responses=[
        json.dumps({"verdict": False, "confidence": "low", "explanation": "screen is blank"})
    ])
    ctx = _ctx(tmp_workspace, mock_backend, gw)
    cls = get_registry().get("assert_screen_contains").cls
    out = cls().run(cls.Inputs(claim="something is on screen"), ctx)
    assert out.verdict is False
    assert out.confidence == "low"


def test_assert_strips_codefence(tmp_workspace, mock_backend):
    gw = FakeGateway(responses=[
        "```json\n" + json.dumps({"verdict": True, "confidence": "medium", "explanation": "ok"}) + "\n```"
    ])
    ctx = _ctx(tmp_workspace, mock_backend, gw)
    cls = get_registry().get("assert_screen_contains").cls
    out = cls().run(cls.Inputs(claim="anything is on screen"), ctx)
    assert out.verdict is True


def test_assert_writes_to_task_state(tmp_workspace, mock_backend):
    gw = FakeGateway(responses=[
        json.dumps({"verdict": True, "confidence": "high", "explanation": "yes"})
    ])
    ctx = _ctx(tmp_workspace, mock_backend, gw)
    cls = get_registry().get("assert_screen_contains").cls
    cls().run(cls.Inputs(claim="any visible text"), ctx)
    snap = ctx.task_state.get("last_assertion")
    assert snap is not None
    assert snap["verdict"] is True


def test_assert_requires_llm(tmp_workspace, mock_backend):
    ctx = _ctx(tmp_workspace, mock_backend, gateway=None)
    cls = get_registry().get("assert_screen_contains").cls
    with pytest.raises(RuntimeError):
        cls().run(cls.Inputs(claim="any visible text"), ctx)


def test_assert_accepts_region(tmp_workspace, mock_backend):
    gw = FakeGateway(responses=[
        json.dumps({"verdict": True, "confidence": "medium", "explanation": "fine"})
    ])
    ctx = _ctx(tmp_workspace, mock_backend, gw)
    cls = get_registry().get("assert_screen_contains").cls
    inp = cls.Inputs(
        claim="login form is visible",
        region={"x": 100, "y": 100, "width": 200, "height": 200},
    )
    out = cls().run(inp, ctx)
    assert out.verdict is True
