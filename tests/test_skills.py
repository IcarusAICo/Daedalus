"""Round-trip every base skill against the MockBackend.

This is the on-disk fixtures + Pydantic validation in one. We use the same
fixture format as ``daedalus skills test``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daedalus.core.registry import get_registry


def _iter_fixtures(skills_dir: Path):
    registry = get_registry()
    for entry in sorted(registry, key=lambda e: e.id):
        tests_dir = skills_dir / entry.id / "tests"
        if not tests_dir.is_dir():
            continue
        for fix in sorted(tests_dir.glob("*.json")):
            yield entry.cls, fix


def test_each_skill_runs_its_fixtures(skills_dir: Path, execution_context, mock_backend):
    fixtures = list(_iter_fixtures(skills_dir))
    assert fixtures, "no fixtures found; library failed to load?"
    for skill_cls, fixture_path in fixtures:
        data = json.loads(fixture_path.read_text())
        mock_backend.reset()
        # Re-run since fixtures share the backend; reset preserves connection.
        mock_backend.connect()
        inp = skill_cls.Inputs.model_validate(data["inputs"])
        instance = skill_cls()
        out = instance.run(inp, execution_context)

        out_dict = out.model_dump(mode="json")
        ignore = set(data.get("ignore_output_keys", []))
        actual = {k: v for k, v in out_dict.items() if k not in ignore}
        expected = {k: v for k, v in data.get("expected_output", {}).items() if k not in ignore}
        assert actual == expected, (
            f"{skill_cls.__name__} fixture {fixture_path.name}: "
            f"output mismatch {actual} != {expected}"
        )

        for want in data.get("expected_events", []):
            want_op = want["op"]
            want_args = want.get("args", {})
            assert any(
                e.op == want_op and all(e.args.get(k) == v for k, v in want_args.items())
                for e in mock_backend.events
            ), f"event {want} not seen in {[e.op for e in mock_backend.events]}"


def test_view_screen_writes_to_task_state(execution_context, mock_backend):
    registry = get_registry()
    cls = registry.get("view_screen").cls
    out = cls().run(cls.Inputs(), execution_context)
    assert out.width == 1920
    assert out.height == 1080
    assert out.image_path
    snap = execution_context.task_state.get("last_screenshot")
    assert snap is not None
    assert snap["width"] == 1920
    assert snap["image_path"] == out.image_path


def test_wait_respects_abort(execution_context):
    registry = get_registry()
    cls = registry.get("wait").cls
    execution_context.abort_event.set()
    out = cls().run(cls.Inputs(ms=5000), execution_context)
    assert out.waited_ms < 1000  # abort short-circuits


def test_click_input_bounds():
    registry = get_registry()
    cls = registry.get("click_mouse").cls
    with pytest.raises(Exception):
        cls.Inputs.model_validate({"x": -1, "y": 0})
