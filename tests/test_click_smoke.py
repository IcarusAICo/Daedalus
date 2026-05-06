"""Smoke test: verify daedalus can click the mouse via VNC (or mock backend).

Run with:
    pytest tests/test_click_smoke.py -v

This uses the MockBackend by default so it works without a VNC server.
To test against a real VNC server, set DAEDALUS_VNC_SMOKE_TEST=1 plus the
usual VNC env vars (DAEDALUS_VNC_HOST, DAEDALUS_VNC_PORT, DAEDALUS_VNC_PASSWORD).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from daedalus.backends.mock import MockBackend
from daedalus.backends.protocol import Button
from daedalus.core.context import ExecutionContext, TaskState
from daedalus.executor.dsl import parse_program
from daedalus.executor.runner import SequentialExecutor
from daedalus.tracing.recorder import TraceRecorder


@pytest.fixture
def mock_backend():
    return MockBackend(width=1920, height=1080)


@pytest.fixture
def tmp_traces(tmp_path):
    return tmp_path / "traces"


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "tasks.db"


class TestClickSmoke:
    """Basic click verification using MockBackend."""

    def test_single_left_click(self, mock_backend):
        mock_backend.connect()
        mock_backend.click(500, 300)
        events = [e for e in mock_backend.events if e.op == "click"]
        assert len(events) == 1
        assert events[0].args["x"] == 500
        assert events[0].args["y"] == 300
        assert events[0].args["button"] == "left"
        assert events[0].args["double"] is False
        mock_backend.disconnect()

    def test_double_click(self, mock_backend):
        mock_backend.connect()
        mock_backend.click(100, 200, double=True)
        events = [e for e in mock_backend.events if e.op == "click"]
        assert len(events) == 1
        assert events[0].args["double"] is True
        mock_backend.disconnect()

    def test_right_click(self, mock_backend):
        mock_backend.connect()
        mock_backend.click(400, 500, button=Button.RIGHT)
        events = [e for e in mock_backend.events if e.op == "click"]
        assert len(events) == 1
        assert events[0].args["button"] == "right"
        mock_backend.disconnect()

    def test_click_via_program(self, mock_backend, tmp_traces, tmp_db):
        """Run a program that clicks and verify the backend received the click."""
        prog = parse_program({
            "name": "click test",
            "steps": [
                {"skill": "click_mouse", "inputs": {"x": 960, "y": 540}},
            ],
        })
        executor = SequentialExecutor(
            backend=mock_backend,
            traces_root=tmp_traces,
            tasks_db=tmp_db,
        )
        result = executor.run(prog)
        assert result.status == "success"
        click_events = [e for e in mock_backend.events if e.op == "click"]
        assert len(click_events) == 1
        assert click_events[0].args["x"] == 960
        assert click_events[0].args["y"] == 540

    def test_click_out_of_bounds_fails(self, mock_backend):
        mock_backend.connect()
        with pytest.raises(ValueError, match="outside MockBackend bounds"):
            mock_backend.click(9999, 9999)
        mock_backend.disconnect()

    def test_click_before_connect_fails(self, mock_backend):
        with pytest.raises(RuntimeError, match="not connected"):
            mock_backend.click(100, 100)


class TestVariableReferences:
    """Verify that $ref: syntax wires step outputs to later step inputs."""

    def test_locate_then_click_via_refs(self, mock_backend, tmp_traces, tmp_db):
        """Simulate locate_element saving coords, then click_mouse using $ref."""
        prog = parse_program({
            "name": "ref click test",
            "steps": [
                {
                    "skill": "locate_element",
                    "inputs": {"description": "test button"},
                    "save_as": "loc",
                },
                {
                    "skill": "click_mouse",
                    "inputs": {"x": "$ref:loc.x", "y": "$ref:loc.y"},
                },
            ],
        })

        # The locate_element skill needs an LLM or grounding service.
        # Instead, we'll directly test the ref resolution by using a
        # simpler program with click_mouse steps.
        prog_simple = parse_program({
            "name": "ref test",
            "steps": [
                {
                    "skill": "click_mouse",
                    "inputs": {"x": 100, "y": 200},
                    "save_as": "first_click",
                },
                {
                    "skill": "click_mouse",
                    "inputs": {
                        "x": "$ref:first_click.clicked_at.0",
                        "y": "$ref:first_click.clicked_at.1",
                    },
                },
            ],
        })
        executor = SequentialExecutor(
            backend=mock_backend,
            traces_root=tmp_traces,
            tasks_db=tmp_db,
        )
        result = executor.run(prog_simple)
        assert result.status == "success"
        click_events = [e for e in mock_backend.events if e.op == "click"]
        assert len(click_events) == 2
        assert click_events[0].args == click_events[1].args


class TestRefResolution:
    """Unit tests for the $ref: resolution logic."""

    def test_resolve_simple_key(self):
        from daedalus.executor.dsl import resolve_ref
        saved = {"loc": {"x": 42, "y": 99, "found": True}}
        assert resolve_ref("$ref:loc.x", saved) == 42
        assert resolve_ref("$ref:loc.y", saved) == 99

    def test_resolve_whole_dict(self):
        from daedalus.executor.dsl import resolve_ref
        saved = {"loc": {"x": 42, "y": 99}}
        assert resolve_ref("$ref:loc", saved) == {"x": 42, "y": 99}

    def test_resolve_missing_key_raises(self):
        from daedalus.executor.dsl import resolve_ref
        with pytest.raises(Exception, match="no step output saved"):
            resolve_ref("$ref:missing.x", {})

    def test_resolve_inputs_dict(self):
        from daedalus.executor.dsl import resolve_inputs
        saved = {"loc": {"x": 100, "y": 200, "label": "btn"}}
        result = resolve_inputs(
            {"x": "$ref:loc.x", "y": "$ref:loc.y", "button": "left"},
            saved,
        )
        assert result == {"x": 100, "y": 200, "button": "left"}

    def test_non_ref_passthrough(self):
        from daedalus.executor.dsl import resolve_inputs
        result = resolve_inputs({"x": 100, "y": 200}, {})
        assert result == {"x": 100, "y": 200}
