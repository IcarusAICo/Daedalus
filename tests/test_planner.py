"""Planner end-to-end against a scripted fake LLM."""

from __future__ import annotations

import json

import pytest

from daedalus.llm.fakes import FakeGateway
from daedalus.planner import Planner, PlannerError


def _scripted(payload: dict) -> str:
    return json.dumps(payload)


def test_planner_returns_valid_program_for_known_skills():
    gw = FakeGateway(responses=[
        _scripted(
            {
                "program": {
                    "name": "type hello",
                    "description": "type hello into the focused window",
                    "steps": [
                        {"skill": "type_text", "inputs": {"text": "hello"}},
                        {"skill": "view_screen", "inputs": {}},
                    ],
                },
                "missing_skills": [],
                "notes": "",
            }
        )
    ])
    planner = Planner(gateway=gw, screen_size=(1920, 1080))
    result = planner.plan("type hello into the focused window")
    assert result.program is not None
    assert result.program.step_count == 2
    assert [s.skill for s in result.program.steps] == ["type_text", "view_screen"]
    assert result.is_actionable


def test_planner_propagates_missing_skill_proposals():
    gw = FakeGateway(responses=[
        _scripted(
            {
                "program": None,
                "missing_skills": [
                    {
                        "proposed_id": "open_url",
                        "description": "Open a URL in the default browser.",
                        "inputs_hint": {"url": "string"},
                        "outputs_hint": {"opened": "bool"},
                        "rationale": "no skill exists for browser navigation",
                    }
                ],
                "notes": "blocked on missing skill",
            }
        )
    ])
    planner = Planner(gateway=gw, screen_size=(1920, 1080))
    result = planner.plan("open https://example.com in the browser")
    assert result.program is None
    assert len(result.missing_skills) == 1
    assert result.missing_skills[0].proposed_id == "open_url"
    assert not result.is_actionable


def test_planner_retries_on_invalid_program():
    bad = _scripted(
        {
            "program": {
                "name": "x",
                "steps": [
                    {"skill": "click_mouse", "inputs": {"x": -1, "y": 0}},
                ],
            },
            "missing_skills": [],
        }
    )
    good = _scripted(
        {
            "program": {
                "name": "x",
                "steps": [
                    {"skill": "click_mouse", "inputs": {"x": 100, "y": 200}},
                ],
            },
            "missing_skills": [],
        }
    )
    gw = FakeGateway(responses=[bad, good])
    result = Planner(gateway=gw, screen_size=(1920, 1080)).plan("click somewhere")
    assert result.program is not None
    assert result.program.steps[0].inputs == {"x": 100, "y": 200}
    assert len(gw.calls) == 2  # initial + repair


def test_planner_returns_none_program_when_repair_also_invalid():
    bad = _scripted(
        {
            "program": {
                "name": "x",
                "steps": [{"skill": "no_such_skill", "inputs": {}}],
            },
            "missing_skills": [],
        }
    )
    gw = FakeGateway(responses=[bad, bad, bad])
    result = Planner(gateway=gw, screen_size=(1920, 1080)).plan("do something impossible")
    assert result.program is None
    assert "failed after" in result.notes
    assert "raw:" in result.notes


def test_planner_strips_codefence_wrapping():
    fenced = "```json\n" + _scripted({
        "program": {
            "name": "y",
            "steps": [{"skill": "wait", "inputs": {"ms": 10}}],
        },
        "missing_skills": [],
    }) + "\n```"
    gw = FakeGateway(responses=[fenced])
    result = Planner(gateway=gw, screen_size=(1920, 1080)).plan("pause briefly")
    assert result.program is not None
    assert result.program.steps[0].skill == "wait"


def test_planner_rejects_non_object_response():
    gw = FakeGateway(responses=["[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]"])
    result = Planner(gateway=gw, screen_size=(1920, 1080)).plan("anything")
    assert result.program is None
    assert "failed after" in result.notes


def test_planner_includes_skill_cards_in_prompt():
    gw = FakeGateway(responses=[_scripted(
        {"program": None, "missing_skills": [], "notes": ""}
    )])
    Planner(gateway=gw, screen_size=(1920, 1080)).plan("anything")
    user_msg = gw.calls[0].messages[-1]["content"]
    # All five base skills should appear in the inlined library when count <= threshold.
    for sid in ("click_mouse", "type_text", "type_shortcut", "view_screen", "wait"):
        assert sid in user_msg


def test_planner_repairs_json_parse_failure():
    """If the first LLM response isn't valid JSON, the planner retries."""
    bad_json = "not-json-at-all"
    good = _scripted({
        "program": {
            "name": "y",
            "steps": [{"skill": "wait", "inputs": {"ms": 10}}],
        },
        "missing_skills": [],
    })
    gw = FakeGateway(responses=[bad_json, good])
    result = Planner(gateway=gw, screen_size=(1920, 1080)).plan("pause briefly")
    assert result.program is not None
    assert result.program.steps[0].skill == "wait"
    assert len(gw.calls) == 2


def test_planner_respects_max_repair_attempts_parameter():
    bad = _scripted({
        "program": {
            "name": "x",
            "steps": [{"skill": "no_such_skill", "inputs": {}}],
        },
        "missing_skills": [],
    })
    gw = FakeGateway(responses=[bad, bad])
    result = Planner(gateway=gw, screen_size=(1920, 1080), max_repair_attempts=1).plan("do something")
    assert result.program is None
    assert "failed after 1 repair attempt(s)" in result.notes
    assert len(gw.calls) == 2


def test_planner_repair_succeeds_on_second_attempt():
    bad = _scripted({
        "program": {
            "name": "x",
            "steps": [{"skill": "no_such_skill", "inputs": {}}],
        },
        "missing_skills": [],
    })
    good = _scripted({
        "program": {
            "name": "x",
            "steps": [{"skill": "wait", "inputs": {"ms": 10}}],
        },
        "missing_skills": [],
    })
    gw = FakeGateway(responses=[bad, bad, good])
    result = Planner(gateway=gw, screen_size=(1920, 1080), max_repair_attempts=2).plan("do something")
    assert result.program is not None
    assert result.program.steps[0].skill == "wait"
    assert len(gw.calls) == 3
