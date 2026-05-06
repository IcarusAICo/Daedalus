"""LLM-backed planner.

Inputs
------
- a free-form user goal
- the screen size of the target host (defaults 1920x1080)
- (optional) extra context the user wants to inject

Outputs
-------
- a :class:`daedalus.executor.dsl.Program`, validated against the registry
- a list of :class:`MissingSkillSpec` for skills the planner thought it needed
  but couldn't find in the library; these are surfaced to the user and (in
  Phase 2) handed to the Implementor.

Design notes
------------
- We constrain the LLM hard: it must respond with a single JSON object that
  matches our schema. Anything else is a planning failure (we retry once).
- The Planner never invents skill ids; we hand it the available ids and
  reject any program that references unknown ones.
- Retrieval (Librarian) selects which skill cards land in the prompt. For
  Phase 1 we give the LLM the *full* card set if it's small (<=20), and the
  top-k otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from daedalus.core.errors import DaedalusError, ProgramValidationError
from daedalus.evaluator.criteria import SuccessCriteria, SuccessCriterion
from daedalus.executor.dsl import (
    AnyProgram,
    Program,
    PythonProgram,
    parse_any_program,
    parse_program,
    parse_python_program,
    validate_program_against_registry,
)
from daedalus.executor.program_executor import lint_plan_code
from daedalus.library import Librarian, SkillCard
from daedalus.llm.gateway import LLMCall, LLMGateway

log = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_K = 12
INLINE_ALL_THRESHOLD = 20


class PlannerError(DaedalusError):
    pass


class MissingSkillSpec(BaseModel):
    """A skill the planner says it needs but the library doesn't have."""

    proposed_id: str = Field(description="Snake_case proposed id.")
    description: str
    inputs_hint: dict[str, Any] = Field(
        default_factory=dict,
        description="Loose JSON-Schema-ish description of intended inputs.",
    )
    outputs_hint: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(description="Why this skill is needed for the goal.")


@dataclass
class PlanResult:
    program: AnyProgram | None
    missing_skills: list[MissingSkillSpec] = field(default_factory=list)
    notes: str = ""
    raw_response: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.program is not None and not self.missing_skills


@dataclass
class StrategyResult:
    """Output of the strategy planning phase (Phase A)."""
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    composite_skills: list[MissingSkillSpec] = field(default_factory=list)
    uses_store: bool = False
    store_tables: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    raw_response: str = ""

    @property
    def needs_new_skills(self) -> bool:
        return len(self.composite_skills) > 0


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are the Planner for a computer-control agent named Daedalus. Given a user goal
and a list of available skills, produce a Python program that calls skills as
functions to accomplish the goal.

TARGET HOST
-----------
- Operating system: __HOST_OS__
- Screen resolution: __SCREEN_W__x__SCREEN_H__ pixels. (0,0) is top-left.
__OS_HINTS__

OUTPUT FORMAT
-------------
Respond with EXACTLY one JSON object on a single line, no prose, no markdown.
The structure is:

  {
    "program": {
      "name": "<short_snake_case_name>",
      "dsl_version": 2,
      "description": "<one-line description>",
      "code": "<python function body>"
    },
    "missing_skills": [ { "proposed_id", "description", "inputs_hint", "outputs_hint", "rationale" } ],
    "notes": "<optional string>"
  }

The "code" field is the BODY of a Python function `def run(ctx):`. Skills are
called as `ctx.<skill_id>(**kwargs)` and return their output model. You can use
variables, if/else, for/while loops, list/dict operations, and define helper
functions.

If you cannot accomplish the goal with the available skills, set program to null
and populate missing_skills.

PYTHON PLAN RULES
-----------------
- Call skills as: result = ctx.<skill_id>(param1=val1, param2=val2)
- The return value is the skill's output (a Pydantic model with attributes).
- You may use: variables, if/elif/else, for/while, try/except, list/dict
  comprehensions, helper functions (def), math, re, json, collections, itertools.
- The following modules are PRE-IMPORTED in the execution sandbox and available
  as global names without any import statement: math, re, json, itertools,
  functools, collections, string, textwrap, copy, operator, heapq, bisect,
  statistics, random, time.
- DO NOT write import statements. They are unnecessary (modules are pre-imported)
  and will cause the plan to fail. Just use module names directly (e.g. re.findall,
  json.loads, collections.defaultdict).
- You may NOT use: eval, exec, compile, open, subprocess, os, sys, network calls.
- Use `time.sleep(seconds)` for waits instead of ctx.wait when more convenient,
  but ctx.wait(ms=N) is preferred for tracing visibility.

EXAMPLE PLAN
------------
```python
# Goal: open Safari and navigate to example.com
ctx.type_shortcut(keys=["super", "space"])
ctx.wait(ms=800)
ctx.type_text(text="Safari")
ctx.type_shortcut(keys=["Return"])
ctx.wait(ms=1500)
ctx.type_shortcut(keys=["super", "l"])
ctx.wait(ms=500)
ctx.type_text(text="https://example.com")
ctx.type_shortcut(keys=["Return"])
ctx.wait(ms=2000)
ctx.view_screen()
```

CRITICAL SKILL USAGE RULES
---------------------------

TIMING AND WAIT RULES
~~~~~~~~~~~~~~~~~~~~~
- The target machine's UI takes time to react. You MUST insert ctx.wait(ms=...)
  after any action that triggers a UI change:
  * After type_shortcut that opens a dialog (e.g. Spotlight, address bar)
  * After type_shortcut with Enter/Return that launches an app or navigates
  * After click_element that opens a menu/dialog/page
- On macOS, Spotlight (Cmd+Space) needs at least 800ms before it's ready
  for text input. ALWAYS add ctx.wait(ms=800) between opening Spotlight and
  typing the app name.

1. **click_element finds and clicks UI elements by description.** It takes a
   natural-language description (e.g. "submit button"), finds the element on
   screen via the grounding service, and clicks it. Use for buttons and links
   when no keyboard shortcut exists.

2. **vision_query is READ-ONLY.** It captures the screen and returns a text
   answer. It NEVER clicks, types, scrolls, or interacts with the screen.
   Use it only to observe or gather information.
   Do NOT ask it to "solve" something — it cannot act on its answers.

3. **locate_element finds UI element coordinates WITHOUT clicking.** Use it
   only when you need coordinates for a purpose other than clicking.

4. **mouse moves, clicks, or drags at pixel coordinates.** Only use when you
   already know exact coordinates (e.g. from a previous locate_element result).
   Supports action="move", action="click" (left/right/double), and action="drag".
   Coordinates are automatically scaled to match the actual screen — just use
   the values from locate_element or your visual estimates directly.

5. **Variable references**: Since this is Python, just use variables:
   loc = ctx.locate_element(description="search box")
   ctx.mouse(action="click", x=loc.x, y=loc.y)

RUNSTORE AND BATCH OPERATIONS
------------------------------
The agent has a per-run data store (ctx.store) for structured data:
- ctx.store_query(question="...", store_table="...", schema_hint={...})
- ctx.locate_elements(description="...", store_table="...")
- ctx.click_all(store_table="...")

Or manipulate the store directly in Python:
  rows = ctx.store.all_rows("edges")
  for row in rows:
      ctx.mouse(action="click", x=row["x"], y=row["y"])
      ctx.wait(ms=200)

ALGORITHMIC SOLVING
-------------------
When the task involves a puzzle, game, or problem with a known algorithmic
solution (Sudoku, pathfinding, sorting, math, logic puzzles, etc.):

1. Use vision_query or store_query to EXTRACT the current state from the screen
2. Write a deterministic solver IN YOUR PYTHON CODE (not an LLM call)
3. Apply the solution by clicking/typing the answers

NEVER ask the vision LLM to "solve" a puzzle — it is unreliable for
computation. The LLM is only for READING the screen, not REASONING about
solutions. Write real Python code for any logic that has a deterministic answer.

Example pattern for a Sudoku puzzle:
  board_info = ctx.vision_query(question="Read the Sudoku grid row by row...")
  board = parse_board(board_info.answer)
  solution = solve_sudoku(board)
  for r, c, val in solution:
      ctx.mouse(action="click", x=grid_x(c), y=grid_y(r))
      ctx.type_text(text=str(val))

SKILL CREATION RULES
--------------------
You SHOULD propose new skills (via missing_skills) ONLY when:
- The task requires genuinely reusable capability not covered by existing skills
- A complex multi-step pattern repeats across different tasks
- Domain-specific logic would benefit from encapsulation as a reusable unit

Do NOT propose new skills when:
- Python control flow (loops, conditionals) in the plan itself is sufficient
- The logic is specific to this one task and unlikely to be reused

SKILL NAMING: Skill names must describe the GENERAL capability, not a specific
task. BAD: reset_and_retry_puzzle, click_puzzle_edges. GOOD: retry_with_reset,
click_coordinates_from_analysis. Strip domain-specific words and describe the
abstract action pattern.

GENERAL RULES
-----
- ONLY use skill ids from the AVAILABLE SKILLS list. Do NOT invent ids.
- Inputs MUST satisfy the given JSON schema. Pay attention to required fields.
- Prefer the smallest correct program.
- Always include a final ctx.view_screen() (or ctx.assert_screen_contains())
  when the goal involves a visible end state.
- Coordinates must be inside the screen bounds.

INTERACTION STRATEGY
--------------------
- USE the controls, hotkeys, and navigation methods discovered by the Explorer.
  The explorer observations will tell you what works — rely on them.
- PREFER keyboard shortcuts and hotkeys over mouse clicks. Keyboard input is
  precise and reliable; mouse clicks depend on coordinates that may shift.
- Use type_shortcut for navigation, menu access, and UI actions whenever a
  keyboard shortcut exists (e.g. Tab to move between fields, arrow keys to
  navigate, Enter to confirm).
- Only fall back to click_element or mouse when no keyboard alternative
  exists or when you must target a specific visual element.
- AVOID hardcoded pixel coordinates. Coordinates estimated from screenshots or
  reported by vision_query are UNRELIABLE — they shift with window position,
  zoom level, and dynamic content. Use click_element (which finds elements by
  description) or locate_element (which returns accurate coordinates at
  runtime) instead. Only use mouse with coordinates obtained from
  locate_element in the same run.
"""

_SUCCESS_CRITERIA_PROMPT = """\
You are the Success Criteria Planner for Daedalus, a computer-control agent.
Given a user goal, define a set of success criteria that can be evaluated AFTER
the agent attempts the goal to determine whether it actually succeeded.

OUTPUT FORMAT
-------------
Respond with EXACTLY one JSON object on a single line, no prose, no markdown.
The structure must be:

  {
    "goal_summary": "<one-sentence restatement of the goal>",
    "criteria": [
      {
        "description": "<human-readable description>",
        "kind": "visual" | "trace" | "state",
        "visual_claim": "<claim to verify against a screenshot, for kind=visual>",
        "trace_pattern": "<pattern for kind=trace>",
        "state_key": "<task_state key for kind=state>",
        "state_condition": "<condition for kind=state>"
      }
    ],
    "must_pass_all": true
  }

CRITERION KINDS
---------------
1. **visual**: The evaluator captures a screenshot and asks a vision LLM
   whether the claim is true. Use this for end-state verification like
   "the puzzle shows a score", "a success message is visible", etc.
   Set the visual_claim field to a natural-language statement about what
   should be visible on screen.

2. **trace**: The evaluator checks the recorded event trace for patterns.
   Available patterns:
   - "skill_id:count_gte:N" -- skill finished at least N times
   - "skill_id:count_lte:N" -- skill finished at most N times
   - "assert_screen_contains:has_verdict_true" -- at least one passing assertion
   - "no_skill_errors" -- no skill errors in the trace
   Set the trace_pattern field.

3. **state**: The evaluator checks a value in the task_state key-value store.
   Conditions: "is_truthy", "is_not_none", "equals:<value>".
   Set state_key and state_condition.

RULES
-----
- Always include at least one **visual** criterion that describes what the
  screen should look like if the goal was achieved.
- Be specific in visual claims. Instead of "the task is done", say
  "the puzzle grid shows placed walls and a score greater than zero".
- Think about what would distinguish a truly successful outcome from the
  agent just running through steps without effect.
- Keep criteria count between 1 and 5. Focus on the most important signals.
- Set must_pass_all to true unless the goal has multiple acceptable outcomes.
"""

_OS_HINTS: dict[str, str] = {
    "macos": (
        "- On macOS, the 'super' key maps to Cmd (⌘).\n"
        "- Open apps via Spotlight: type_shortcut keys=['super','space'], then type the app name and press Enter.\n"
        "- Cmd+Space opens Spotlight. Cmd+L focuses the browser address bar.\n"
        "- The Dock is at the bottom; the menu bar is at the top."
    ),
    "windows": (
        "- On Windows, the 'super' key maps to the Windows key.\n"
        "- Win+R opens the Run dialog. Win+E opens File Explorer.\n"
        "- Open apps via Start: press 'super', type the app name, press Enter.\n"
        "- Ctrl+L focuses the browser address bar. The taskbar is at the bottom."
    ),
    "linux": (
        "- On Linux, the 'super' key typically opens the Activities/App launcher.\n"
        "- Open apps: press 'super', type the app name, press Enter.\n"
        "- Ctrl+L focuses the browser address bar."
    ),
}


def _system_prompt(screen_w: int, screen_h: int, host_os: str = "unknown") -> str:
    os_key = host_os.lower()
    hints = _OS_HINTS.get(os_key, f"- Host OS is '{host_os}'. Use platform-appropriate shortcuts.")

    from daedalus.core.context import llm_image_size
    llm_w, llm_h = llm_image_size(screen_w, screen_h)

    return (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("__SCREEN_W__", str(llm_w))
        .replace("__SCREEN_H__", str(llm_h))
        .replace("__HOST_OS__", host_os)
        .replace("__OS_HINTS__", hints)
    )


def _format_skill_cards(cards: list[SkillCard]) -> str:
    lines: list[str] = []
    for c in cards:
        lines.append(f"### ctx.{c.id}  (v{c.version}, {c.kind})")
        lines.append(c.description)
        lines.append(f"side_effects: {', '.join(c.side_effects) or 'none'}")
        lines.append("inputs: " + json.dumps(c.inputs_schema, separators=(",", ":")))
        lines.append("outputs: " + json.dumps(c.outputs_schema, separators=(",", ":")))
        if c.examples:
            ex_lines = [json.dumps(e, separators=(",", ":")) for e in c.examples[:2]]
            lines.append("examples: " + " | ".join(ex_lines))
        lines.append("")
    return "\n".join(lines)


def _build_user_message(goal: str, cards: list[SkillCard], extra_context: str | None) -> str:
    parts = ["GOAL:", goal.strip(), "", "AVAILABLE SKILLS:", _format_skill_cards(cards)]
    if extra_context:
        parts.extend(["", "EXTRA CONTEXT:", extra_context.strip()])
    parts.extend(["", "Return the JSON object now."])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _strip_codefence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # Drop opening fence (```json or ```)
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def _parse_response(content: str) -> dict[str, Any]:
    raw = _strip_codefence(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Some models still wrap or trail. Try to find the first {...} block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc2:
                raise PlannerError(f"LLM response not valid JSON: {exc2}") from exc
        else:
            raise PlannerError(f"LLM response not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlannerError("LLM response root must be a JSON object")
    return data


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    def __init__(
        self,
        gateway: LLMGateway,
        librarian: Librarian | None = None,
        *,
        role: str = "planner",
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        screen_size: tuple[int, int],
        host_os: str = "unknown",
        max_repair_attempts: int = 2,
    ) -> None:
        self._gateway = gateway
        self._librarian = librarian if librarian is not None else Librarian()
        if self._librarian._indexed_ids == set():  # type: ignore[attr-defined]
            self._librarian.reindex()
        self._role = role
        self._retrieval_k = retrieval_k
        self._screen_size = screen_size
        self._host_os = host_os
        self._max_repair_attempts = max_repair_attempts

    def plan(self, goal: str, *, extra_context: str | None = None, memory_context: str | None = None, learner_feedback: str | None = None) -> PlanResult:
        cards = self._collect_cards(goal)
        combined_extra = ""
        if memory_context:
            combined_extra += f"AGENT MEMORY (facts from past runs):\n{memory_context}\n\n"
        if learner_feedback:
            combined_extra += f"LEARNER FEEDBACK (from analysis of a previous failed attempt):\n{learner_feedback}\n\n"
        if extra_context:
            combined_extra += extra_context
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": _system_prompt(self._screen_size[0], self._screen_size[1], self._host_os),
            },
            {
                "role": "user",
                "content": _build_user_message(goal, cards, combined_extra or None),
            },
        ]
        call = LLMCall(
            role=self._role,
            messages=messages,
            response_format="json_object",
            max_tokens=4096,
        )

        content, raw = self._call_llm(call)
        last_error: str | None = None

        for attempt in range(1 + self._max_repair_attempts):
            # --- parse ---
            try:
                result = self._build_result(content, raw)
            except PlannerError as exc:
                last_error = str(exc)
                log.warning("plan parse failed (attempt %d): %s", attempt + 1, last_error)
                if attempt >= self._max_repair_attempts:
                    break
                content, raw = self._repair(messages, content, f"JSON parse error: {exc}")
                continue

            # --- validate ---
            if result.program is None:
                return result

            if isinstance(result.program, Program):
                try:
                    validate_program_against_registry(result.program)
                except ProgramValidationError as exc:
                    last_error = str(exc)
                    log.warning("plan validation failed (attempt %d): %s", attempt + 1, last_error)
                    if attempt >= self._max_repair_attempts:
                        break
                    content, raw = self._repair(messages, content, f"Program validation error: {exc}")
                    continue

            if isinstance(result.program, PythonProgram):
                lint_errors = lint_plan_code(result.program.code)
                if lint_errors:
                    last_error = "DSL lint errors:\n" + "\n".join(f"  - {e}" for e in lint_errors)
                    log.warning("plan DSL lint failed (attempt %d): %s", attempt + 1, last_error)
                    if attempt >= self._max_repair_attempts:
                        break
                    content, raw = self._repair(
                        messages,
                        content,
                        f"Plan code failed DSL safety lint:\n"
                        + "\n".join(f"  - {e}" for e in lint_errors)
                        + "\n\nThe following modules are PRE-IMPORTED in the sandbox "
                        "and available without import statements: math, re, json, "
                        "itertools, functools, collections, string, textwrap, copy, "
                        "operator, heapq, bisect, statistics, random, time.\n"
                        "Remove all import statements and use these modules directly.",
                    )
                    continue

            return result

        return PlanResult(
            program=None,
            notes=(
                f"failed after {self._max_repair_attempts} repair attempt(s): "
                f"{last_error}; raw: {raw[:500]}"
            ),
            raw_response=raw,
        )

    def _repair(
        self,
        messages: list[dict[str, str]],
        assistant_content: str,
        feedback: str,
    ) -> tuple[str, str]:
        fix_messages = [
            *messages,
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": (
                    f"Your previous response had an error:\n{feedback}\n"
                    "Return a corrected JSON object using the same format. "
                    "Do NOT invent skill ids."
                ),
            },
        ]
        return self._call_llm(LLMCall(
            role=self._role,
            messages=fix_messages,
            response_format="json_object",
        ))

    # ------------------------------------------------------------------

    def _collect_cards(self, goal: str) -> list[SkillCard]:
        if len(self._librarian) <= INLINE_ALL_THRESHOLD:
            return self._librarian.all_cards()
        return self._librarian.search(goal, k=self._retrieval_k)

    def _call_llm(self, call: LLMCall) -> tuple[str, str]:
        try:
            resp = self._gateway.complete(call)
        except Exception as exc:
            raise PlannerError(f"LLM call failed: {exc}") from exc
        return resp.content, resp.content

    def _build_result(self, content: str, raw: str) -> PlanResult:
        data = _parse_response(content)
        program: AnyProgram | None = None
        if data.get("program") is not None:
            try:
                program = parse_any_program(data["program"])
            except Exception as exc:
                raise PlannerError(f"planner returned a malformed program: {exc}") from exc
        missing: list[MissingSkillSpec] = []
        for entry in data.get("missing_skills") or []:
            try:
                missing.append(MissingSkillSpec.model_validate(entry))
            except ValidationError as exc:
                log.warning("dropping malformed missing_skill entry: %s", exc)
        return PlanResult(
            program=program,
            missing_skills=missing,
            notes=str(data.get("notes") or ""),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Success criteria planning
    # ------------------------------------------------------------------

    def plan_success_criteria(self, goal: str) -> SuccessCriteria:
        """Ask the LLM to generate success criteria for *goal*."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SUCCESS_CRITERIA_PROMPT},
            {"role": "user", "content": f"GOAL: {goal.strip()}\n\nReturn the JSON now."},
        ]
        call = LLMCall(
            role=self._role,
            messages=messages,
            response_format="json_object",
        )
        content, _raw = self._call_llm(call)
        data = _parse_response(content)
        try:
            return SuccessCriteria.model_validate(data)
        except ValidationError as exc:
            raise PlannerError(f"LLM returned invalid success criteria: {exc}") from exc

    def plan_strategy(self, goal: str, *, extra_context: str | None = None) -> StrategyResult:
        """Phase A: ask the LLM for a high-level strategy before generating a program.

        Returns a list of subtasks and any composite skills the planner
        recommends synthesizing before the program phase.
        """
        cards = self._collect_cards(goal)
        system = (
            "You are the Strategy Planner for Daedalus, a computer-control agent.\n"
            "Given a goal and available skills, produce a HIGH-LEVEL STRATEGY — not a program.\n\n"
            "OUTPUT FORMAT: Respond with EXACTLY one JSON object:\n"
            "{\n"
            '  "subtasks": [{"description": "...", "complexity": "low|medium|high"}],\n'
            '  "composite_skills_needed": [\n'
            '    {"proposed_id": "...", "description": "...", "rationale": "...",\n'
            '     "inputs_hint": {...}, "outputs_hint": {...}}\n'
            "  ],\n"
            '  "uses_store": true|false,\n'
            '  "store_tables": [{"name": "...", "columns": {"col": "type"}, "purpose": "..."}],\n'
            '  "strategy_notes": "..."\n'
            "}\n\n"
            "RULES:\n"
            "- Propose composite_skills_needed when a subtask would require >8 repetitive\n"
            "  steps, involves batch operations on collected data, or needs domain logic.\n"
            "- Set uses_store=true if the task benefits from accumulating data across steps.\n"
            "- For store_tables, suggest what tables to create and their schemas.\n"
            "- Be concise. Focus on what NEW capabilities are needed beyond the available skills."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": _build_user_message(goal, cards, extra_context)},
        ]
        content, raw = self._call_llm(LLMCall(
            role=self._role,
            messages=messages,
            response_format="json_object",
        ))
        try:
            data = _parse_response(content)
        except PlannerError:
            return StrategyResult(subtasks=[], raw_response=raw)

        subtasks = data.get("subtasks") or []
        missing: list[MissingSkillSpec] = []
        for entry in data.get("composite_skills_needed") or []:
            try:
                missing.append(MissingSkillSpec.model_validate(entry))
            except ValidationError:
                pass
        store_tables = data.get("store_tables") or []
        return StrategyResult(
            subtasks=subtasks,
            composite_skills=missing,
            uses_store=bool(data.get("uses_store")),
            store_tables=store_tables,
            notes=str(data.get("strategy_notes") or ""),
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# CLI helper: write a plan to disk for human inspection
# ---------------------------------------------------------------------------


def write_plan(result: PlanResult, out_path: Path) -> None:
    """Serialize a PlanResult next to a YAML program for inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing_skills": [m.model_dump() for m in result.missing_skills],
        "notes": result.notes,
    }
    if result.program is not None:
        payload["program"] = result.program.model_dump(exclude_none=True)
    out_path.write_text(json.dumps(payload, indent=2))
