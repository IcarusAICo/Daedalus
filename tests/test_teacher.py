"""Learner: heuristic analysis + LLM-backed report."""

from __future__ import annotations

import json
from pathlib import Path

from daedalus.backends.mock import MockBackend
from daedalus.executor.dsl import parse_program
from daedalus.executor.runner import SequentialExecutor
from daedalus.llm.fakes import FakeGateway
from daedalus.learner import (
    HeuristicFindings,
    Learner,
    LearnerReport,
    analyze_trace,
    analyze_traces,
)
from daedalus.learner.analysis import find_repeated_subsequences

# ---------------------------------------------------------------------------
# Heuristic-only tests (no LLM)
# ---------------------------------------------------------------------------


def _run_one(program_yaml: str, ws: Path) -> str:
    p = parse_program(program_yaml)
    be = MockBackend()
    ex = SequentialExecutor(
        backend=be, traces_root=ws / "traces", tasks_db=ws / "tasks.db"
    )
    res = ex.run(p)
    return res.task_id


def test_analyze_trace_extracts_skill_sequence(tmp_workspace):
    tid = _run_one(
        """
name: t
steps:
  - skill: click_mouse
    inputs: {x: 1, y: 1}
  - skill: type_text
    inputs: {text: "hi"}
  - skill: wait
    inputs: {ms: 1}
""",
        tmp_workspace,
    )
    summary = analyze_trace(tmp_workspace / "traces" / tid)
    assert summary.skill_sequence == ["click_mouse", "type_text", "wait"]
    assert summary.status == "success"
    assert "click_mouse" in summary.timings
    assert summary.timings["click_mouse"].calls == 1


def test_repeated_subsequences_detected(tmp_workspace):
    prog = """
name: t
steps:
  - skill: click_mouse
    inputs: {x: 0, y: 0}
  - skill: type_text
    inputs: {text: "x"}
  - skill: click_mouse
    inputs: {x: 0, y: 0}
  - skill: type_text
    inputs: {text: "y"}
  - skill: click_mouse
    inputs: {x: 0, y: 0}
  - skill: type_text
    inputs: {text: "z"}
"""
    tid = _run_one(prog, tmp_workspace)
    summary = analyze_trace(tmp_workspace / "traces" / tid)
    repeats = find_repeated_subsequences([summary], min_occurrences=2)
    assert any(ng.skills == ("click_mouse", "type_text") for ng in repeats)


def test_analyze_traces_aggregates_status(tmp_workspace):
    tid1 = _run_one(
        """
name: a
steps:
  - skill: wait
    inputs: {ms: 1}
""",
        tmp_workspace,
    )
    tid2 = _run_one(
        """
name: b
steps:
  - skill: wait
    inputs: {ms: 1}
""",
        tmp_workspace,
    )
    findings = analyze_traces(
        [tmp_workspace / "traces" / tid1, tmp_workspace / "traces" / tid2]
    )
    assert findings.traces_analyzed == 2
    assert findings.overall_status_counts["success"] == 2


# ---------------------------------------------------------------------------
# LLM-backed Learner tests
# ---------------------------------------------------------------------------


def _good_report() -> str:
    return json.dumps(
        {
            "summary": "Two traces, click_mouse->type_text occurs three times in the trace.",
            "efficiency_wins": [
                {
                    "description": "Replace ad-hoc click+type with a compound skill.",
                    "affected_skills": ["click_mouse", "type_text"],
                    "estimated_savings_ms": 15,
                    "recommendation": "Define click_then_type and use it instead.",
                }
            ],
            "new_skill_candidates": [
                {
                    "proposed_id": "click_then_type",
                    "description": "Click at (x,y) and then type the given text.",
                    "component_skills": ["click_mouse", "type_text"],
                    "occurrences": 3,
                    "inputs_hint": {"x": "int", "y": "int", "text": "str"},
                    "outputs_hint": {"clicked_at": "list", "chars_typed": "int"},
                    "rationale": "Recurring sub-sequence.",
                }
            ],
            "failure_proposals": [],
        }
    )


def test_learner_returns_structured_report(tmp_workspace):
    findings = HeuristicFindings(
        traces_analyzed=1,
        overall_status_counts={},
    )
    gw = FakeGateway(responses=[_good_report()])
    report = Learner(gateway=gw).learn_from_findings(findings)
    assert isinstance(report, LearnerReport)
    assert report.new_skill_candidates[0].proposed_id == "click_then_type"
    assert report.efficiency_wins[0].estimated_savings_ms == 15


def test_learner_candidate_yields_implementor_request(tmp_workspace):
    gw = FakeGateway(responses=[_good_report()])
    report = Learner(gateway=gw).learn_from_findings(HeuristicFindings(traces_analyzed=1, overall_status_counts={}))
    cand = report.new_skill_candidates[0]
    req = cand.as_implementor_request()
    assert req.proposed_id == "click_then_type"
    assert "click_mouse" in (req.extra_context or "")


def test_learner_strips_codefence():
    fenced = "```json\n" + _good_report() + "\n```"
    gw = FakeGateway(responses=[fenced])
    report = Learner(gateway=gw).learn_from_findings(HeuristicFindings(traces_analyzed=1, overall_status_counts={}))
    assert report.new_skill_candidates


def test_learner_round_trip_from_real_traces(tmp_workspace):
    tid1 = _run_one(
        """
name: cycle1
steps:
  - skill: click_mouse
    inputs: {x: 1, y: 1}
  - skill: type_text
    inputs: {text: "a"}
  - skill: click_mouse
    inputs: {x: 1, y: 1}
  - skill: type_text
    inputs: {text: "b"}
""",
        tmp_workspace,
    )
    tid2 = _run_one(
        """
name: cycle2
steps:
  - skill: click_mouse
    inputs: {x: 1, y: 1}
  - skill: type_text
    inputs: {text: "c"}
""",
        tmp_workspace,
    )
    gw = FakeGateway(responses=[_good_report()])
    findings, report = Learner(gateway=gw).learn_from_dirs(
        [tmp_workspace / "traces" / tid1, tmp_workspace / "traces" / tid2]
    )
    assert findings.traces_analyzed == 2
    assert report.new_skill_candidates
