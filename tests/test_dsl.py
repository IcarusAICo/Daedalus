"""DSL parsing and validation."""

from __future__ import annotations

import pytest

from daedalus.core.errors import ProgramValidationError
from daedalus.executor.dsl import parse_program, summarize, validate_program_against_registry

VALID_PROGRAM = """
name: smoke
steps:
  - skill: click_mouse
    inputs: {x: 10, y: 20}
  - skill: type_text
    inputs: {text: "hi"}
  - skill: wait
    inputs: {ms: 100}
"""


def test_parse_minimal_program():
    p = parse_program(VALID_PROGRAM)
    assert p.name == "smoke"
    assert p.step_count == 3
    assert p.referenced_skill_ids() == ["click_mouse", "type_text", "wait"]


def test_parse_rejects_extra_top_level_key():
    bad = "name: x\nsteps: []\nrogue: yes\n"
    with pytest.raises(ProgramValidationError):
        parse_program(bad)


def test_parse_requires_at_least_one_step():
    with pytest.raises(ProgramValidationError):
        parse_program("name: x\nsteps: []\n")


def test_validate_rejects_unknown_skill():
    p = parse_program("name: x\nsteps:\n  - skill: not_a_real_skill\n")
    with pytest.raises(ProgramValidationError):
        validate_program_against_registry(p)


def test_validate_rejects_bad_inputs():
    p = parse_program(
        "name: x\nsteps:\n  - skill: click_mouse\n    inputs: {x: -10, y: 0}\n"
    )
    with pytest.raises(ProgramValidationError):
        validate_program_against_registry(p)


def test_summarize_rolls_up_side_effects():
    p = parse_program(VALID_PROGRAM)
    s = summarize(p)
    assert s.step_count == 3
    assert "screen_input" in s.side_effects
    assert s.daemon_steps == 0
    assert any("click_mouse@" in entry for entry in s.skills)


def test_version_constraint_in_step():
    p = parse_program(
        "name: x\nsteps:\n  - skill: click_mouse\n    version: ^0.1.0\n    inputs: {x: 0, y: 0}\n"
    )
    validate_program_against_registry(p)


def test_unsatisfied_version_constraint_rejected():
    p = parse_program(
        "name: x\nsteps:\n  - skill: click_mouse\n    version: ^9.9.0\n    inputs: {x: 0, y: 0}\n"
    )
    with pytest.raises(ProgramValidationError):
        validate_program_against_registry(p)
