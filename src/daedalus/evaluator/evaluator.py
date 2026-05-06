"""Goal-level evaluation engine.

After the executor finishes all steps, the evaluator runs each
SuccessCriterion and produces a GoalVerdict that reflects whether the
user's goal was actually achieved -- not just whether all steps ran.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from daedalus.core.context import ExecutionContext
from daedalus.evaluator.criteria import (
    CriterionResult,
    GoalVerdict,
    SuccessCriteria,
    SuccessCriterion,
)
from daedalus.llm.gateway import LLMCall, LLMGateway
from daedalus.tracing.recorder import TraceRecorder

log = logging.getLogger(__name__)


_VISUAL_JUDGE_PROMPT = """\
You are a visual assertion judge for a computer-control agent named Daedalus.

Given a screenshot and a CLAIM in natural language, decide whether the claim is
true on screen RIGHT NOW. Respond with EXACTLY one JSON object on a single
line, no prose:

  {"verdict": true|false, "confidence": "low|medium|high", "explanation": "<one sentence>"}

Be conservative: if you're not sure, answer false with low confidence and say
why in one short sentence.
"""


class Evaluator:
    """Evaluates SuccessCriteria against the post-execution state."""

    def __init__(self, llm: LLMGateway | None = None, vision_role: str = "vision") -> None:
        self._llm = llm
        self._vision_role = vision_role

    def evaluate(
        self,
        criteria: SuccessCriteria,
        ctx: ExecutionContext,
        tracer: TraceRecorder,
    ) -> GoalVerdict:
        results: list[CriterionResult] = []
        for criterion in criteria.criteria:
            result = self._evaluate_one(criterion, ctx, tracer)
            results.append(result)

        if criteria.must_pass_all:
            achieved = all(r.passed for r in results)
        else:
            achieved = any(r.passed for r in results)

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        mode = "all" if criteria.must_pass_all else "any"
        summary = (
            f"{passed_count}/{total} criteria passed (mode={mode}). "
            f"Goal {'achieved' if achieved else 'NOT achieved'}."
        )

        verdict = GoalVerdict(achieved=achieved, results=results, summary=summary)

        tracer.emit(
            "goal_evaluation",
            {
                "achieved": verdict.achieved,
                "summary": verdict.summary,
                "criteria_count": total,
                "criteria_passed": passed_count,
                "results": [
                    {
                        "description": r.criterion.description,
                        "kind": r.criterion.kind,
                        "passed": r.passed,
                        "explanation": r.explanation,
                    }
                    for r in results
                ],
            },
        )

        return verdict

    def _evaluate_one(
        self,
        criterion: SuccessCriterion,
        ctx: ExecutionContext,
        tracer: TraceRecorder,
    ) -> CriterionResult:
        try:
            if criterion.kind == "visual":
                return self._evaluate_visual(criterion, ctx, tracer)
            elif criterion.kind == "trace":
                return self._evaluate_trace(criterion, tracer)
            elif criterion.kind == "state":
                return self._evaluate_state(criterion, ctx)
            else:
                return CriterionResult(
                    criterion=criterion,
                    passed=False,
                    explanation=f"Unknown criterion kind: {criterion.kind}",
                )
        except Exception as exc:
            log.warning("criterion evaluation failed: %s", exc)
            return CriterionResult(
                criterion=criterion,
                passed=False,
                explanation=f"Evaluation error: {exc}",
            )

    # ------------------------------------------------------------------
    # Visual criteria
    # ------------------------------------------------------------------

    def _evaluate_visual(
        self,
        criterion: SuccessCriterion,
        ctx: ExecutionContext,
        tracer: TraceRecorder,
    ) -> CriterionResult:
        if self._llm is None:
            return CriterionResult(
                criterion=criterion,
                passed=False,
                explanation="No LLM gateway available for visual evaluation.",
            )
        if not criterion.visual_claim:
            return CriterionResult(
                criterion=criterion,
                passed=False,
                explanation="Visual criterion has no claim to evaluate.",
            )

        shot = ctx.backend.screenshot()
        buf = io.BytesIO()
        shot.image.convert("RGB").save(buf, format="PNG")
        png = buf.getvalue()
        image_b64 = base64.b64encode(png).decode("ascii")

        tracer.attach_screenshot(png, width=shot.width, height=shot.height)

        messages = [
            {"role": "system", "content": _VISUAL_JUDGE_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"CLAIM: {criterion.visual_claim}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]

        resp = self._llm.complete(
            LLMCall(
                role=self._vision_role,
                messages=messages,
                response_format="json_object",
                temperature=0.0,
            )
        )

        try:
            data = json.loads(resp.content.strip())
        except json.JSONDecodeError:
            text = resp.content.strip()
            a, b = text.find("{"), text.rfind("}")
            if a != -1 and b > a:
                data = json.loads(text[a : b + 1])
            else:
                return CriterionResult(
                    criterion=criterion,
                    passed=False,
                    explanation=f"Vision model returned unparseable response: {resp.content[:200]}",
                )

        verdict = bool(data.get("verdict", False))
        explanation = data.get("explanation", "")
        confidence = data.get("confidence", "low")

        return CriterionResult(
            criterion=criterion,
            passed=verdict,
            explanation=f"[{confidence}] {explanation}",
        )

    # ------------------------------------------------------------------
    # Trace criteria
    # ------------------------------------------------------------------

    def _evaluate_trace(
        self,
        criterion: SuccessCriterion,
        tracer: TraceRecorder,
    ) -> CriterionResult:
        if not criterion.trace_pattern:
            return CriterionResult(
                criterion=criterion,
                passed=False,
                explanation="Trace criterion has no pattern.",
            )

        events = list(tracer.iter_events())
        pattern = criterion.trace_pattern

        # Pattern: "skill_id:count_gte:N"
        if ":count_gte:" in pattern:
            parts = pattern.split(":")
            skill_id = parts[0]
            threshold = int(parts[2])
            count = sum(
                1 for e in events
                if e.kind == "skill_finished"
                and e.data.get("skill_id") == skill_id
            )
            passed = count >= threshold
            return CriterionResult(
                criterion=criterion,
                passed=passed,
                explanation=f"Skill '{skill_id}' finished {count} time(s) (threshold: {threshold}).",
            )

        # Pattern: "skill_id:count_lte:N"
        if ":count_lte:" in pattern:
            parts = pattern.split(":")
            skill_id = parts[0]
            threshold = int(parts[2])
            count = sum(
                1 for e in events
                if e.kind == "skill_finished"
                and e.data.get("skill_id") == skill_id
            )
            passed = count <= threshold
            return CriterionResult(
                criterion=criterion,
                passed=passed,
                explanation=f"Skill '{skill_id}' finished {count} time(s) (max: {threshold}).",
            )

        # Pattern: "assert_screen_contains:has_verdict_true"
        if ":has_verdict_true" in pattern:
            skill_id = pattern.split(":")[0]
            has_true = any(
                e.kind == "visual_assertion" and e.data.get("verdict") is True
                for e in events
            )
            return CriterionResult(
                criterion=criterion,
                passed=has_true,
                explanation=(
                    f"Found a passing visual assertion."
                    if has_true
                    else f"No visual assertion with verdict=true found in trace."
                ),
            )

        # Pattern: "no_skill_errors"
        if pattern == "no_skill_errors":
            errors = [e for e in events if e.kind == "skill_error"]
            passed = len(errors) == 0
            return CriterionResult(
                criterion=criterion,
                passed=passed,
                explanation=(
                    "No skill errors in trace."
                    if passed
                    else f"{len(errors)} skill error(s) found."
                ),
            )

        return CriterionResult(
            criterion=criterion,
            passed=False,
            explanation=f"Unknown trace pattern: {pattern}",
        )

    # ------------------------------------------------------------------
    # State criteria
    # ------------------------------------------------------------------

    def _evaluate_state(
        self,
        criterion: SuccessCriterion,
        ctx: ExecutionContext,
    ) -> CriterionResult:
        if not criterion.state_key:
            return CriterionResult(
                criterion=criterion,
                passed=False,
                explanation="State criterion has no key.",
            )

        value = ctx.task_state.get(criterion.state_key)
        condition = criterion.state_condition or "is_truthy"

        if condition == "is_truthy":
            passed = bool(value)
            return CriterionResult(
                criterion=criterion,
                passed=passed,
                explanation=f"task_state['{criterion.state_key}'] = {value!r} (truthy={passed}).",
            )

        if condition.startswith("equals:"):
            expected = condition[len("equals:"):]
            passed = str(value) == expected
            return CriterionResult(
                criterion=criterion,
                passed=passed,
                explanation=f"task_state['{criterion.state_key}'] = {value!r} (expected '{expected}').",
            )

        if condition == "is_not_none":
            passed = value is not None
            return CriterionResult(
                criterion=criterion,
                passed=passed,
                explanation=f"task_state['{criterion.state_key}'] is {'set' if passed else 'None/missing'}.",
            )

        return CriterionResult(
            criterion=criterion,
            passed=False,
            explanation=f"Unknown state condition: {condition}",
        )
