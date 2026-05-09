"""LLM-backed Learner: analyzes traces and turns findings into actionable proposals.

The learner operates as an iterative tool-calling agent (like the explorer).
It receives a trace directory and uses custom tools to investigate events,
screenshots, and timing data at its own pace, then signals completion with
structured feedback.

Operating modes:

1. **Failure analysis** (default): given a failed trace, investigate what went
   wrong and produce suggestions + revised plan hints.
2. **Success optimization**: given a successful trace, look for inefficiencies.
3. **Batch heuristic analysis**: aggregated cross-trace statistics (no tool loop).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from daedalus.core.errors import DaedalusError
from daedalus.executor.dsl import Program
from daedalus.implementor import ImplementorRequest
from daedalus.llm.context import prune_old_images, summarize_and_compact
from daedalus.llm.gateway import LLMCall, LLMGateway
from daedalus.learner.analysis import (
    HeuristicFindings,
    analyze_traces,
)
from daedalus.learner.tools import ALL_TOOLS, TraceContext, dispatch_tool
from daedalus.shared.skill_caller import (
    TOOL_IMPLEMENT_SKILL,
    TOOL_REVISE_SKILL,
    handle_implement_skill,
    handle_revise_skill,
    handle_skill_call,
    skill_to_tool_def,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 15


def _now_compact() -> str:
    """Return a compact timestamp string for unique IDs (YYYYMMDD_HHMMSS)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class LearnerError(DaedalusError):
    pass


# ---------------------------------------------------------------------------
# Report shapes
# ---------------------------------------------------------------------------


class EfficiencyWin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    affected_skills: list[str] = Field(default_factory=list)
    estimated_savings_ms: int | None = None
    recommendation: str


class NewSkillCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed_id: str
    description: str
    component_skills: list[str] = Field(default_factory=list)
    occurrences: int = 0
    inputs_hint: dict[str, Any] = Field(default_factory=dict)
    outputs_hint: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    def as_implementor_request(self) -> ImplementorRequest:
        return ImplementorRequest(
            proposed_id=self.proposed_id,
            description=self.description,
            rationale=self.rationale or "Learner-proposed compound skill",
            inputs_hint=self.inputs_hint,
            outputs_hint=self.outputs_hint,
            extra_context=(
                "This skill should compose the following existing skills in order: "
                + ", ".join(self.component_skills)
            ),
        )


class FailureProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    affected_skill: str
    failure_pattern: str
    proposal: str
    ready_for_implementor: bool = False


class LearnerReport(BaseModel):
    """Backwards-compatible report from the batch-heuristic analysis path."""
    model_config = ConfigDict(extra="forbid")

    summary: str
    efficiency_wins: list[EfficiencyWin] = Field(default_factory=list)
    new_skill_candidates: list[NewSkillCandidate] = Field(default_factory=list)
    failure_proposals: list[FailureProposal] = Field(default_factory=list)


class LearnerSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    affected_step_idx: int | None = None
    category: str = "other"


class SkillAmendment(BaseModel):
    """Proposal to amend an existing skill based on trace evidence."""
    model_config = ConfigDict(extra="forbid")

    skill_id: str
    issue_description: str
    proposed_change: str
    evidence: str = ""


class LearnerFeedback(BaseModel):
    """Rich feedback from single-trace analysis (failure or success)."""
    model_config = ConfigDict(extra="forbid")

    summary: str
    failure_point: str | None = None
    suggestions: list[LearnerSuggestion] = Field(default_factory=list)
    new_skill_candidates: list[NewSkillCandidate] = Field(default_factory=list)
    skill_amendments: list[SkillAmendment] = Field(default_factory=list)
    revised_plan_hints: str = ""
    debug_session_results: list[dict[str, Any]] | None = None
    implemented_skills: list[str] = Field(default_factory=list)
    skill_revisions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SKILL_DISCIPLINE = """\
SKILL PROPOSAL DISCIPLINE
--------------------------
Do NOT propose new skills when:
- The issue can be fixed by adjusting timing (add/increase wait steps)
- The issue can be fixed by changing a click_element description
- The issue can be fixed by reordering existing steps
- The fix is achievable with Python control flow in the plan (loops, conditionals)
- The fix is a simple parameter change to existing skills

Only propose new skills when:
- The task requires genuinely reusable capability not covered by existing skills
- A complex multi-step pattern repeats across DIFFERENT tasks (not just this one)
- Domain-specific logic would benefit from encapsulation as a reusable unit

Prefer plan-level fixes and skill amendments over new skills whenever possible.

SKILL NAMING
------------
Skill names must describe the GENERAL capability, not the specific task.
- BAD:  reset_and_retry_puzzle, click_puzzle_edges_from_vision
- GOOD: retry_with_reset, click_coordinates_from_analysis

The name should make sense if used for a completely different application.
Strip domain-specific words (puzzle, sudoku, game, etc.) and describe the
abstract action pattern.
"""

_LEARNER_SYSTEM_PROMPT = """\
You are the Learner for Daedalus, a computer-control agent. You are analyzing
an execution trace to understand what happened and produce actionable feedback.

You have tools to investigate the trace interactively. Use them to build
understanding incrementally rather than trying to diagnose everything at once.

WORKFLOW
--------
1. Start with get_trace_summary to orient yourself (status, timings, failures).
2. Use get_events to examine specific regions of interest (around failures,
   slow steps, or unexpected behavior).
3. Use get_event_field to drill into large events without loading everything.
4. Use view_screenshot to see what was on screen at key moments.
5. Use get_program to see the plan that was executed.
6. If you identify a need for a new skill, call propose_new_skill.
7. If an existing skill has a bug or needs improvement, call propose_skill_amendment.
8. When you have enough evidence, call learning_done with your diagnosis.

ANALYSIS GUIDELINES
-------------------
- Be specific: reference event line numbers and screenshot indices as evidence.
- For timing issues, suggest concrete wait durations.
- For targeting issues (element not found, wrong click), describe what you see
  in the screenshot and how the plan should adapt.
- If the plan used hardcoded pixel coordinates (mouse with fixed x/y),
  flag this as fragile and recommend click_element or locate_element instead.
- Note: all screenshots are downscaled to the LLM's internal processing
  resolution, and mouse coordinates are automatically scaled back up. All
  coordinates across view_screen, locate_element, click_element, and mouse
  are in the same consistent space. If clicks still land in the wrong spot,
  the issue is likely element misidentification, not coordinate scaling.
- If a failure involved incorrect puzzle solving or computation, suggest
  extracting problem state and using a deterministic algorithmic solver.

{skill_discipline}

SKILL AMENDMENTS
----------------
Use propose_skill_amendment when an existing NON-CORE skill:
- Has a bug revealed by the trace (e.g. incorrect selector, wrong timeout)
- Needs better defaults (e.g. longer default wait, different retry behavior)
- Is missing error handling for a case you observed
- Has incorrect assumptions about the environment

Core skills (built-in primitives like click_element, view_screen, type_text,
wait, etc.) CANNOT be amended. If a core skill is involved in a failure,
suggest plan-level workarounds in revised_plan_hints instead, or propose a
new wrapper skill.

Provide evidence from the trace (event line numbers, screenshots) to support
the amendment.
"""

_LEARNER_LIVE_SYSTEM_PROMPT = """\
You are the Learner for Daedalus, a computer-control agent. A plan just FAILED \
and you have a LIVE connection to the VM. Your goal is to diagnose WHY it \
failed and produce TESTED, CONCRETE fixes.

DO NOT just passively read the trace and describe what went wrong. You must \
actively TEST your hypotheses on the live VM, implement or revise skills to \
harden them, and report what concretely works.

WORKFLOW
--------
1. Review the failure context (trace summary, events around the failure point) \
to form a hypothesis about what went wrong.
2. Call view_screen() to see the CURRENT state of the VM.
3. REPRODUCE the failure: try the action that failed (click_element, \
locate_element, etc.) to confirm the root cause.
4. TEST fixes interactively: try alternative approaches (dismiss dialogs, \
use different selectors, try keyboard shortcuts) until you find what works.
5. If a skill is fragile, call implement_skill or revise_skill to create a \
hardened version that handles the failure condition. Then TEST IT LIVE.
6. If you need to reproduce failure step-by-step, use debug_plan with \
breakpoints at the suspected failure step.
7. Call learning_done with your diagnosis BACKED BY LIVE EVIDENCE of what works.

CRITICAL RULES
--------------
- Your suggestions MUST be backed by live testing. Don't just say "try clicking X" \
-- actually click X and report what happened.
- If you implement or revise a skill, you MUST test it on the live VM before \
recommending it.
- Use the trace tools (get_trace_summary, get_events, view_screenshot, \
get_event_field, get_program) to understand context, but spend the MAJORITY \
of your iterations on live VM interaction.
- If you identify a fragile skill, harden it: add error handling for popups, \
focus loss, blocking dialogs, etc. Test the hardened version.
- When you call learning_done, include specific evidence: "I tested X and it \
worked/failed because Y."

AVAILABLE TOOLS
---------------
Live VM interaction:
- view_screen, click_element, type_text, type_shortcut, scroll, mouse, wait, \
locate_element, locate_elements, vision_query, extract_text, assert_screen_contains
- implement_skill: create a new reusable skill and test it
- revise_skill: fix a skill you implemented this session

Debugging:
- debug_plan: re-execute the plan with breakpoints for step-by-step debugging
- debug_continue, debug_step, debug_stop: control the debug session

Trace inspection (for context):
- get_trace_summary, get_events, get_event_field, view_screenshot, \
list_screenshots, get_program

Proposals (use after testing):
- propose_new_skill, propose_skill_amendment, learning_done

SKILL HARDENING
---------------
If a skill fails due to environmental conditions (popups, focus loss, overlays, \
unexpected state), you should:
1. Identify the condition that caused failure (test it live)
2. Call revise_skill or implement_skill to handle that condition
3. Test the hardened version on the live VM
4. Include the hardened skill in your feedback

{skill_discipline}

LLM COORDINATE SCALING
-----------------------
Screenshots are downscaled to the LLM's internal processing resolution. All \
coordinates across skills (view_screen, locate_element, click_element, mouse) \
use the same consistent downscaled coordinate space. The mouse skill \
automatically scales coordinates back up. Just use coordinates as you see them.
"""

_LEARNER_LIVE_FAILURE_USER_MSG = """\
The plan just FAILED. Here is the execution context:

Task ID: {task_id}
Task Name: {task_name}
Status: {status}
Duration: {duration_ms:.0f}ms
Events: {event_count}
Screenshots: {screenshot_count}

Your job: diagnose the failure, reproduce it on the live VM, test fixes, \
harden any fragile skills, and report what concretely works. Start by \
reviewing the trace context (get_trace_summary), then immediately move to \
live VM interaction.
"""

_LEARNER_LIVE_SUCCESS_USER_MSG = """\
The plan SUCCEEDED but may have inefficiencies. Here is the execution context:

Task ID: {task_id}
Task Name: {task_name}
Status: {status}
Duration: {duration_ms:.0f}ms
Events: {event_count}
Screenshots: {screenshot_count}

Your job: identify optimization opportunities and test whether skills can be \
improved or consolidated. Use the live VM to verify your hypotheses about \
unnecessary steps or slow interactions.
"""

_FAILURE_USER_MSG = """\
The trace below is from a FAILED execution. Your goal is to diagnose what
went wrong and provide suggestions to fix it for the next attempt.

Task ID: {task_id}
Task Name: {task_name}
Status: {status}
Duration: {duration_ms:.0f}ms
Events: {event_count}
Screenshots: {screenshot_count}

Investigate the trace using your tools and call learning_done when ready.
"""

_SUCCESS_USER_MSG = """\
The trace below is from a SUCCESSFUL execution. Your goal is to find
optimization opportunities -- unnecessary waits, redundant steps, or patterns
that could be improved.

Task ID: {task_id}
Task Name: {task_name}
Status: {status}
Duration: {duration_ms:.0f}ms
Events: {event_count}
Screenshots: {screenshot_count}

Investigate the trace using your tools and call learning_done when ready.
"""

_BATCH_SYSTEM_PROMPT = """\
You are the Learner for Daedalus, a computer-control agent. You receive HEURISTIC
FINDINGS from a deterministic analysis of recent task traces. Your job is to
turn those findings into ACTIONABLE proposals the user (and the Implementor)
can execute against.

Respond with EXACTLY one JSON object on a single line, no prose:

  {{
    "summary": "<2-3 sentences>",
    "efficiency_wins": [{{description, affected_skills, estimated_savings_ms?, recommendation}}, ...],
    "new_skill_candidates": [
      {{proposed_id, description, component_skills, occurrences, inputs_hint, outputs_hint, rationale}}, ...
    ],
    "failure_proposals": [{{affected_skill, failure_pattern, proposal, ready_for_implementor}}, ...]
  }}

RULES
-----
- Be concrete. Tie every proposal to specific skills or sequences from the findings.
- Only mark `ready_for_implementor: true` when the proposal already specifies a
  spec the Implementor can build (clear inputs/outputs, scope, and side effects).
- Cap the number of proposals at 5 per category.
- If there is nothing to recommend, return empty lists.

{skill_discipline}
"""



# ---------------------------------------------------------------------------
# Trace context preparation helpers
# ---------------------------------------------------------------------------


def _program_to_summary(program: Program | Any) -> str:
    from daedalus.executor.dsl import PythonProgram

    lines = [f"Program: {program.name}"]
    if program.description:
        lines.append(f"Description: {program.description}")
    if isinstance(program, PythonProgram):
        lines.append("Plan code (Python):")
        for i, line in enumerate(program.code.splitlines(), 1):
            lines.append(f"  {i:3d}: {line}")
    else:
        for i, step in enumerate(program.steps):
            inputs_str = ", ".join(f"{k}={v!r}" for k, v in step.inputs.items())
            lines.append(f"  Step {i}: {step.skill}({inputs_str})")
            if step.description:
                lines.append(f"    Note: {step.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing (for batch path)
# ---------------------------------------------------------------------------


def _findings_to_prompt(findings: HeuristicFindings) -> str:
    payload: dict[str, Any] = {
        "traces_analyzed": findings.traces_analyzed,
        "status_counts": dict(findings.overall_status_counts),
        "timings": {
            sid: {
                "calls": t.calls,
                "mean_ms": round(t.mean_ms, 2),
                "p95_ms": round(t.p95_ms, 2),
                "total_ms": round(t.total_ms, 2),
            }
            for sid, t in findings.timings.items()
        },
        "failures": {
            sid: asdict(f) for sid, f in findings.failures.items()
        },
        "repeated_subsequences": [
            {"skills": list(ng.skills), "occurrences": ng.occurrences, "in_traces": ng.in_traces}
            for ng in findings.repeated_subsequences[:20]
        ],
        "notes": findings.notes,
    }
    return "HEURISTIC FINDINGS:\n" + json.dumps(payload, indent=2) + "\n\nReturn the JSON now."


def _strip_codefence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _parse_json(content: str) -> dict[str, Any]:
    text = _strip_codefence(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b <= a:
            raise LearnerError(f"non-JSON response: {exc}") from exc
        return json.loads(text[a: b + 1])


def _parse_report(content: str) -> LearnerReport:
    data = _parse_json(content)
    try:
        return LearnerReport.model_validate(data)
    except ValidationError as exc:
        raise LearnerError(f"report does not match schema: {exc}") from exc


# ---------------------------------------------------------------------------
# Learner (iterative tool-calling agent)
# ---------------------------------------------------------------------------


class Learner:
    """Iterative tool-calling learner that investigates traces interactively.

    When live_mode=True, the learner also has access to the VM via skills,
    can implement/revise temp skills, and can debug plans step-by-step.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        role: str = "learner",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = False,
        # Live mode dependencies (all optional -- None means offline-only)
        backend: Any | None = None,
        registry: Any | None = None,
        librarian: Any | None = None,
        implementor: Any | None = None,
        skills_dir: Path | None = None,
        traces_root: Path | None = None,
        tasks_db: Path | None = None,
    ) -> None:
        self._gateway = gateway
        self._role = role
        self._max_iterations = max_iterations
        self._verbose = verbose
        # Live mode
        self._backend = backend
        self._registry = registry
        self._librarian = librarian
        self._implementor = implementor
        self._skills_dir = skills_dir
        self._traces_root = traces_root or Path("traces")
        self._tasks_db = tasks_db
        self._temp_skills: list[str] = []
        self._new_skills: list[str] = []
        self._debugger: Any | None = None

    @property
    def live_mode(self) -> bool:
        """True if the learner has live VM access."""
        return self._backend is not None and self._registry is not None

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build the full tool list based on mode (trace-only or live)."""
        tools = list(ALL_TOOLS)
        if self.live_mode:
            for entry in self._registry:
                if entry.cls.SPEC.kind == "daemon":
                    continue
                tools.append(skill_to_tool_def(entry))
            tools.append(TOOL_IMPLEMENT_SKILL)
            tools.append(TOOL_REVISE_SKILL)
            from daedalus.learner.debugger import TOOL_DEBUG_PLAN, TOOL_DEBUG_CONTINUE, TOOL_DEBUG_STEP, TOOL_DEBUG_STOP
            tools.append(TOOL_DEBUG_PLAN)
            tools.append(TOOL_DEBUG_CONTINUE)
            tools.append(TOOL_DEBUG_STEP)
            tools.append(TOOL_DEBUG_STOP)
        return tools

    def _get_execution_context(self, tracer: Any) -> Any:
        """Build an ExecutionContext for live skill calls using the provided tracer."""
        from daedalus.core.context import ExecutionContext, TaskState, compute_coordinate_scale
        from daedalus.core.store import RunStore

        screen_w, screen_h = self._backend.size if hasattr(self._backend, "size") else (1728, 1117)
        task_id = tracer.task_id
        db_path = self._tasks_db or self._traces_root / "tasks.db"
        state = TaskState(db_path, task_id)
        store = RunStore(db_path, task_id)

        return ExecutionContext(
            task_id=task_id,
            backend=self._backend,
            task_state=state,
            tracer=tracer,
            store=store,
            llm=self._gateway,
            abort_event=threading.Event(),
            coordinate_scale=compute_coordinate_scale(screen_w),
        )

    def _dispatch_live_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        exec_ctx: Any,
        trace_ctx: TraceContext,
    ) -> tuple[str | list[dict[str, Any]], bool]:
        """Dispatch a tool call -- tries trace tools first, then live tools."""
        from daedalus.learner.tools import _HANDLERS

        # Trace-inspection tools
        if tool_name in _HANDLERS:
            return dispatch_tool(tool_name, arguments, trace_ctx)

        # Debugger tools
        if tool_name in ("debug_plan", "debug_continue", "debug_step", "debug_stop"):
            return self._dispatch_debug_tool(tool_name, arguments, exec_ctx, trace_ctx)

        # Implement/revise skill
        if tool_name == "implement_skill":
            result = handle_implement_skill(
                arguments,
                registry=self._registry,
                implementor=self._implementor,
                librarian=self._librarian,
                temp_skills=self._temp_skills,
                new_skills=self._new_skills,
            )
            return result, False

        if tool_name == "revise_skill":
            result = handle_revise_skill(
                arguments,
                registry=self._registry,
                implementor=self._implementor,
                librarian=self._librarian,
                temp_skills=self._temp_skills,
            )
            return result, False

        # All other tool names are skill IDs
        result = handle_skill_call(tool_name, arguments, exec_ctx, self._registry)
        return result, False

    def _dispatch_debug_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        exec_ctx: Any,
        trace_ctx: TraceContext,
    ) -> tuple[str, bool]:
        """Handle debug_plan/continue/step/stop tool calls."""
        from daedalus.learner.debugger import PlanDebugger

        if tool_name == "debug_plan":
            if self._debugger is not None:
                return json.dumps({"error": "A debug session is already active. Call debug_stop first."}), False
            plan_source = arguments.get("plan_source", "latest")
            breakpoints = arguments.get("breakpoints", [])
            break_on_error = arguments.get("break_on_error", True)

            program_code = trace_ctx.program_text
            if not program_code:
                return json.dumps({"error": "No program found in the trace to debug."}), False

            self._debugger = PlanDebugger(
                program_code=program_code,
                backend=self._backend,
                registry=self._registry,
                gateway=self._gateway,
                traces_root=self._traces_root,
                tasks_db=self._tasks_db,
                breakpoints=breakpoints,
                break_on_error=break_on_error,
            )
            result = self._debugger.start()
            return json.dumps(result, default=str), False

        if tool_name == "debug_continue":
            if self._debugger is None:
                return json.dumps({"error": "No active debug session. Call debug_plan first."}), False
            result = self._debugger.continue_execution()
            return json.dumps(result, default=str), False

        if tool_name == "debug_step":
            if self._debugger is None:
                return json.dumps({"error": "No active debug session. Call debug_plan first."}), False
            result = self._debugger.step()
            return json.dumps(result, default=str), False

        if tool_name == "debug_stop":
            if self._debugger is None:
                return json.dumps({"error": "No active debug session."}), False
            result = self._debugger.stop()
            self._debugger = None
            return json.dumps(result, default=str), False

        return json.dumps({"error": f"Unknown debug tool: {tool_name}"}), False

    # -- Batch heuristic analysis (unchanged, no tool loop) ----------------

    def learn_from_dirs(self, task_dirs: list[Path]) -> tuple[HeuristicFindings, LearnerReport]:
        findings = analyze_traces(task_dirs)
        report = self.learn_from_findings(findings)
        return findings, report

    def learn_from_findings(self, findings: HeuristicFindings) -> LearnerReport:
        prompt = _BATCH_SYSTEM_PROMPT.format(skill_discipline=_SKILL_DISCIPLINE)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": _findings_to_prompt(findings)},
        ]
        try:
            resp = self._gateway.complete(
                LLMCall(
                    role=self._role,
                    messages=messages,
                    response_format="json_object",
                )
            )
        except Exception as exc:
            raise LearnerError(f"LLM call failed: {exc}") from exc
        return _parse_report(resp.content)

    # -- Single-trace analysis (tool-calling loop) -------------------------

    def analyze_failure(
        self,
        task_dir: Path,
        program: Program | None = None,
        stream_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, str, dict, str | None, str | None], None] | None = None,
        explorer_context: str | None = None,
        context_usage_callback: Callable[[int, int], None] | None = None,
        task_id: str | None = None,
    ) -> LearnerFeedback:
        """Analyze a failed trace using an iterative tool-calling loop."""
        return self._analyze_trace(task_dir, program, mode="failure", stream_callback=stream_callback, tool_callback=tool_callback, explorer_context=explorer_context, context_usage_callback=context_usage_callback, task_id=task_id)

    def analyze_success(
        self,
        task_dir: Path,
        program: Program | None = None,
        stream_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, str, dict, str | None, str | None], None] | None = None,
        explorer_context: str | None = None,
        context_usage_callback: Callable[[int, int], None] | None = None,
        task_id: str | None = None,
    ) -> LearnerFeedback:
        """Analyze a successful trace for optimization opportunities."""
        return self._analyze_trace(task_dir, program, mode="success", stream_callback=stream_callback, tool_callback=tool_callback, explorer_context=explorer_context, context_usage_callback=context_usage_callback, task_id=task_id)

    def _analyze_trace(
        self,
        task_dir: Path,
        program: Program | None,
        mode: str,
        stream_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, str, dict, str | None, str | None], None] | None = None,
        explorer_context: str | None = None,
        context_usage_callback: Callable[[int, int], None] | None = None,
        task_id: str | None = None,
    ) -> LearnerFeedback:
        """Core tool-calling loop for single-trace analysis."""
        program_text = _program_to_summary(program) if program else None
        trace_ctx = TraceContext(task_dir, program_text=program_text)

        # Set up live execution context if in live mode
        exec_ctx = None
        tracer = None
        if self.live_mode:
            if not self._backend.is_connected:
                self._backend.connect()
            if self._implementor:
                self._implementor.cleanup_temp()
            self._temp_skills.clear()
            self._new_skills.clear()

            # Create the TraceRecorder ONCE with a unique task_id
            from daedalus.tracing.recorder import TraceRecorder
            learner_task_id = task_id or f"learner_{_now_compact()}"
            db_path = self._tasks_db or self._traces_root / "tasks.db"
            tracer = TraceRecorder(
                traces_root=self._traces_root,
                db_path=db_path,
                task_name="learner",
                task_id=learner_task_id,
            )
            tracer.start()
            exec_ctx = self._get_execution_context(tracer)

        summary = trace_ctx.summary
        if self.live_mode:
            # Live mode: use the explorer-style active prompt
            if mode == "failure":
                user_msg = _LEARNER_LIVE_FAILURE_USER_MSG.format(
                    task_id=summary.task_id,
                    task_name=summary.name,
                    status=summary.status,
                    duration_ms=summary.total_duration_ms,
                    event_count=summary.event_count,
                    screenshot_count=summary.screenshot_count,
                )
            else:
                user_msg = _LEARNER_LIVE_SUCCESS_USER_MSG.format(
                    task_id=summary.task_id,
                    task_name=summary.name,
                    status=summary.status,
                    duration_ms=summary.total_duration_ms,
                    event_count=summary.event_count,
                    screenshot_count=summary.screenshot_count,
                )
            system_prompt = _LEARNER_LIVE_SYSTEM_PROMPT.format(
                skill_discipline=_SKILL_DISCIPLINE,
            )
        else:
            # Offline mode: trace-analysis focused prompt
            if mode == "failure":
                user_msg = _FAILURE_USER_MSG.format(
                    task_id=summary.task_id,
                    task_name=summary.name,
                    status=summary.status,
                    duration_ms=summary.total_duration_ms,
                    event_count=summary.event_count,
                    screenshot_count=summary.screenshot_count,
                )
            else:
                user_msg = _SUCCESS_USER_MSG.format(
                    task_id=summary.task_id,
                    task_name=summary.name,
                    status=summary.status,
                    duration_ms=summary.total_duration_ms,
                    event_count=summary.event_count,
                    screenshot_count=summary.screenshot_count,
                )
            system_prompt = _LEARNER_SYSTEM_PROMPT.format(
                skill_discipline=_SKILL_DISCIPLINE,
            )

        if explorer_context:
            user_msg += (
                "\n\n## Explorer Observations (pre-execution context)\n"
                "The following observations were gathered by the Explorer agent before planning.\n"
                "These describe the environment, available controls, and interaction patterns:\n\n"
                + explorer_context
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        tools = self._build_tools()
        new_skill_candidates: list[NewSkillCandidate] = []
        skill_amendments: list[SkillAmendment] = []
        feedback: LearnerFeedback | None = None
        total_tool_calls = 0

        for iteration in range(self._max_iterations):
            if self._verbose:
                log.info("learner iteration %d/%d", iteration + 1, self._max_iterations)

            summarize_and_compact(messages, self._gateway)
            prune_old_images(messages)

            if context_usage_callback:
                from daedalus.llm.context import estimate_token_count, get_context_config
                used = estimate_token_count(messages)
                max_tokens = get_context_config().max_context_tokens
                context_usage_callback(used, max_tokens)

            try:
                response = self._gateway.complete(
                    LLMCall(
                        role=self._role,
                        messages=messages,
                        tools=tools,
                    )
                )
            except Exception as exc:
                raise LearnerError(f"LLM call failed on iteration {iteration + 1}: {exc}") from exc

            if self._verbose and response.content:
                log.info("learner thinking: %s", response.content[:300])

            if stream_callback and response.content:
                stream_callback(response.content)

            if not response.tool_calls:
                if self._verbose:
                    log.info("learner returned text without tool calls — treating as implicit done")
                feedback = LearnerFeedback(
                    summary=response.content.strip() or "Analysis complete (no tool calls).",
                    failure_point=None,
                    suggestions=[],
                    new_skill_candidates=new_skill_candidates,
                    skill_amendments=skill_amendments,
                    revised_plan_hints="",
                    implemented_skills=list(self._new_skills),
                )
                break

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
            messages.append(assistant_msg)

            done = False
            for tc in response.tool_calls:
                total_tool_calls += 1

                if self._verbose:
                    args_preview = json.dumps(tc.arguments, default=str)
                    if len(args_preview) > 200:
                        args_preview = args_preview[:200] + "..."
                    log.info("learner tool_call: %s(%s)", tc.name, args_preview)

                # Emit tool_call event to tracer
                if tracer:
                    tracer.emit("tool_call", {
                        "iteration": iteration + 1,
                        "call_index": total_tool_calls,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                    })

                if tool_callback:
                    tool_callback(tc.name, tc.id, tc.arguments, None, None)

                # Dispatch: live mode uses combined dispatcher, otherwise trace-only
                if self.live_mode:
                    result_content, is_done = self._dispatch_live_tool(
                        tc.name, tc.arguments, exec_ctx, trace_ctx
                    )
                else:
                    result_content, is_done = dispatch_tool(tc.name, tc.arguments, trace_ctx)

                result_str = result_content if isinstance(result_content, str) else "[image content]"

                # Emit tool_result event to tracer
                if tracer:
                    trace_result = result_str if isinstance(result_str, str) else "[multimodal content with image]"
                    tracer.emit("tool_result", {
                        "iteration": iteration + 1,
                        "call_index": total_tool_calls,
                        "tool": tc.name,
                        "result": trace_result[:500] if len(trace_result) > 500 else trace_result,
                        "done": is_done,
                    })

                if tool_callback:
                    image_path = None
                    if isinstance(result_content, list):
                        for part in result_content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_val = part.get("text", "")
                                try:
                                    data = json.loads(text_val)
                                    if isinstance(data, dict) and "image_path" in data:
                                        image_path = data["image_path"]
                                except (json.JSONDecodeError, TypeError):
                                    if "(" in text_val and text_val.rstrip(")").rsplit("(", 1)[-1].startswith("/"):
                                        image_path = text_val.rstrip(")").rsplit("(", 1)[-1]
                    tool_callback(tc.name, tc.id, tc.arguments, result_str[:200] if isinstance(result_str, str) else result_str, image_path)

                # Collect proposals from proposal tools
                if tc.name == "propose_new_skill":
                    try:
                        candidate = NewSkillCandidate(
                            proposed_id=tc.arguments.get("proposed_id", ""),
                            description=tc.arguments.get("description", ""),
                            component_skills=tc.arguments.get("component_skills", []),
                            inputs_hint=tc.arguments.get("inputs_hint", {}),
                            outputs_hint=tc.arguments.get("outputs_hint", {}),
                            rationale=tc.arguments.get("rationale", ""),
                        )
                        new_skill_candidates.append(candidate)
                    except Exception:
                        pass

                if tc.name == "propose_skill_amendment":
                    try:
                        sid = tc.arguments.get("skill_id", "")
                        from daedalus.learner.tools import _get_core_skills
                        if sid and sid not in _get_core_skills():
                            amendment = SkillAmendment(
                                skill_id=sid,
                                issue_description=tc.arguments.get("issue_description", ""),
                                proposed_change=tc.arguments.get("proposed_change", ""),
                                evidence=tc.arguments.get("evidence", ""),
                            )
                            skill_amendments.append(amendment)
                    except Exception:
                        pass

                if tc.name == "learning_done":
                    args = tc.arguments
                    suggestions = []
                    for s in args.get("suggestions", []):
                        suggestions.append(LearnerSuggestion(
                            description=s.get("description", ""),
                            affected_step_idx=s.get("affected_step_idx"),
                            category=s.get("category", "other"),
                        ))
                    feedback = LearnerFeedback(
                        summary=args.get("summary", ""),
                        failure_point=args.get("failure_point"),
                        suggestions=suggestions,
                        new_skill_candidates=new_skill_candidates,
                        skill_amendments=skill_amendments,
                        revised_plan_hints=args.get("revised_plan_hints", ""),
                        implemented_skills=list(self._new_skills),
                    )
                    done = True

                # Rebuild tools if new skills were implemented
                if tc.name == "implement_skill" and self.live_mode:
                    tools = self._build_tools()

                # Build the tool result message
                if isinstance(result_content, list):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })

                if done:
                    break

            if done:
                break
        else:
            if self._verbose:
                log.info("learner reached max iterations, forcing summary")
            feedback = self._force_summary(messages, new_skill_candidates, skill_amendments, tools)

        if feedback is None:
            feedback = LearnerFeedback(
                summary="Learner completed without producing structured feedback.",
                new_skill_candidates=new_skill_candidates,
                skill_amendments=skill_amendments,
                implemented_skills=list(self._new_skills),
            )

        # Clean up debugger if still active
        if self._debugger is not None:
            try:
                self._debugger.stop()
            except Exception:
                pass
            self._debugger = None

        # Finalize the tracer
        if tracer:
            status = "success" if feedback and feedback.summary else "failed"
            tracer.finish(status)

        return feedback

    def _force_summary(
        self,
        messages: list[dict[str, Any]],
        new_skill_candidates: list[NewSkillCandidate],
        skill_amendments: list[SkillAmendment],
        tools: list[dict[str, Any]] | None = None,
    ) -> LearnerFeedback:
        """Send a final message asking the learner to summarize findings."""
        if tools is None:
            tools = list(ALL_TOOLS)
        if messages and messages[-1].get("role") in ("tool",):
            messages.append({
                "role": "assistant",
                "content": "I've completed my analysis. Let me summarize my findings.",
            })
        messages.append({
            "role": "user",
            "content": (
                "You've reached the iteration limit. Please call learning_done now "
                "with your best assessment based on what you've seen so far."
            ),
        })
        try:
            summarize_and_compact(messages, self._gateway)
            prune_old_images(messages)
            response = self._gateway.complete(
                LLMCall(
                    role=self._role,
                    messages=messages,
                    tools=tools,
                )
            )
        except Exception:
            return LearnerFeedback(
                summary="Learner reached iteration limit and failed to summarize.",
                new_skill_candidates=new_skill_candidates,
                skill_amendments=skill_amendments,
                implemented_skills=list(self._new_skills),
            )

        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "learning_done":
                    args = tc.arguments
                    suggestions = []
                    for s in args.get("suggestions", []):
                        suggestions.append(LearnerSuggestion(
                            description=s.get("description", ""),
                            affected_step_idx=s.get("affected_step_idx"),
                            category=s.get("category", "other"),
                        ))
                    return LearnerFeedback(
                        summary=args.get("summary", ""),
                        failure_point=args.get("failure_point"),
                        suggestions=suggestions,
                        new_skill_candidates=new_skill_candidates,
                        skill_amendments=skill_amendments,
                        revised_plan_hints=args.get("revised_plan_hints", ""),
                    )

        return LearnerFeedback(
            summary=response.content.strip() or "Learner reached iteration limit.",
            new_skill_candidates=new_skill_candidates,
            skill_amendments=skill_amendments,
            implemented_skills=list(self._new_skills),
        )

