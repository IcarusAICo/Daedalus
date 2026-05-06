"""Data models for goal-level success criteria and evaluation verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SuccessCriterion(BaseModel):
    """A single criterion that must hold for the goal to be considered achieved."""

    description: str = Field(description="Human-readable description of what this criterion checks.")
    kind: Literal["visual", "trace", "state"] = Field(
        description="How this criterion is evaluated."
    )

    # "visual": capture screenshot, ask vision LLM if the claim is true.
    visual_claim: str | None = Field(
        default=None,
        description="Natural-language claim to verify against a screenshot.",
    )

    # "trace": pattern match on the recorded events.
    trace_pattern: str | None = Field(
        default=None,
        description=(
            "A simple pattern expression evaluated against the event trace. "
            "Format: 'skill_id:operator:value', e.g. 'mouse:count_gte:5' "
            "or 'assert_screen_contains:has_verdict_true'."
        ),
    )

    # "state": check a task_state key.
    state_key: str | None = Field(
        default=None,
        description="Key in task_state to inspect.",
    )
    state_condition: str | None = Field(
        default=None,
        description="Condition the state value must satisfy, e.g. 'is_truthy', 'equals:done'.",
    )


class SuccessCriteria(BaseModel):
    """Complete set of criteria the planner generates for a goal."""

    goal_summary: str = Field(description="One-sentence restatement of the user's goal.")
    criteria: list[SuccessCriterion] = Field(
        min_length=1,
        description="Criteria to evaluate after execution.",
    )
    must_pass_all: bool = Field(
        default=True,
        description="If True, ALL criteria must pass (AND). If False, ANY passing is sufficient (OR).",
    )


class CriterionResult(BaseModel):
    """Result of evaluating a single criterion."""

    criterion: SuccessCriterion
    passed: bool
    explanation: str


class GoalVerdict(BaseModel):
    """Aggregate result of evaluating all criteria for a goal."""

    achieved: bool
    results: list[CriterionResult]
    summary: str
