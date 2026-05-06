"""Learner tool definitions and dispatch handlers.

The learner uses a tool-calling loop (like the explorer) to interactively
investigate traces. This module defines the tool schemas (OpenAI function-
calling format) and the dispatch logic that executes each tool against a
loaded trace.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from daedalus.learner.analysis import TraceSummary, analyze_trace, load_trace
from daedalus.library.loader import load_core_skills

log = logging.getLogger(__name__)

_MAX_PNG_BYTES_FOR_LLM = 3_500_000
_MAX_EVENTS_PER_REQUEST = 80

_CORE_SKILLS: frozenset[str] | None = None


def _get_core_skills() -> frozenset[str]:
    global _CORE_SKILLS
    if _CORE_SKILLS is None:
        _CORE_SKILLS = load_core_skills()
    return _CORE_SKILLS


def _encode_image_for_llm(path: Path) -> tuple[str, str]:
    """Encode an image for LLM consumption, converting large PNGs to JPEG."""
    raw = path.read_bytes()
    if len(raw) <= _MAX_PNG_BYTES_FOR_LLM:
        return base64.b64encode(raw).decode("ascii"), "image/png"

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling JSON schemas)
# ---------------------------------------------------------------------------

TOOL_GET_TRACE_SUMMARY = {
    "type": "function",
    "function": {
        "name": "get_trace_summary",
        "description": (
            "Get a high-level heuristic summary of the trace: per-skill timings, "
            "failures, skill execution sequence, event/screenshot counts, and "
            "total duration. Call this first to orient yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_GET_EVENTS = {
    "type": "function",
    "function": {
        "name": "get_events",
        "description": (
            "Retrieve events from the trace by line range and/or kind filter. "
            "Events are returned as JSON objects with a 'line' field prepended "
            "so you can reference specific events in follow-up calls. "
            "Use kind_filter to narrow to specific event types (e.g. 'skill_error', "
            "'tool_call', 'llm_call', 'screenshot')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_line": {
                    "type": "integer",
                    "description": "First event line to return (0-indexed). Defaults to 0.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last event line (exclusive). Defaults to start_line + limit.",
                },
                "kind_filter": {
                    "type": "string",
                    "description": "If set, only return events with this 'kind' value.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of events to return. Defaults to 30.",
                },
            },
        },
    },
}

TOOL_GET_EVENT_FIELD = {
    "type": "function",
    "function": {
        "name": "get_event_field",
        "description": (
            "Extract a specific field from an event at a given line number. "
            "Use dot-notation for nested fields (e.g. 'data.arguments', "
            "'data.tool', 'data.response_tool_calls'). Useful for inspecting "
            "large events without loading the entire object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "line": {
                    "type": "integer",
                    "description": "The 0-indexed line number of the event.",
                },
                "field_path": {
                    "type": "string",
                    "description": "Dot-separated path to the field (e.g. 'data.arguments.description').",
                },
            },
            "required": ["line", "field_path"],
        },
    },
}

TOOL_VIEW_SCREENSHOT = {
    "type": "function",
    "function": {
        "name": "view_screenshot",
        "description": (
            "View a screenshot by its 1-based index (matching the screens/NNNN.png "
            "naming). Returns the image for visual inspection along with metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "1-based screenshot index (e.g. 1 for screens/0001.png).",
                },
            },
            "required": ["index"],
        },
    },
}

TOOL_LIST_SCREENSHOTS = {
    "type": "function",
    "function": {
        "name": "list_screenshots",
        "description": (
            "List all available screenshots with their index, filename, and "
            "corresponding timestamp from the trace events. Useful for choosing "
            "which screenshots to inspect."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_GET_PROGRAM = {
    "type": "function",
    "function": {
        "name": "get_program",
        "description": (
            "Get the program/plan that was executed in this trace. Returns the "
            "program code or step list if available, or an error if no program "
            "was associated with this trace."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_PROPOSE_NEW_SKILL = {
    "type": "function",
    "function": {
        "name": "propose_new_skill",
        "description": (
            "Propose a new skill to be implemented. Only use when the issue "
            "cannot be fixed with existing skills + better planning. You may "
            "call this multiple times before calling learning_done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "proposed_id": {
                    "type": "string",
                    "description": "Snake_case id for the skill (general name, not task-specific).",
                },
                "description": {
                    "type": "string",
                    "description": "What the skill does.",
                },
                "component_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Existing skill IDs this would compose.",
                },
                "inputs_hint": {
                    "type": "object",
                    "description": "Sketch of expected input fields.",
                },
                "outputs_hint": {
                    "type": "object",
                    "description": "Sketch of expected output fields.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this skill is needed.",
                },
            },
            "required": ["proposed_id", "description", "rationale"],
        },
    },
}

TOOL_PROPOSE_SKILL_AMENDMENT = {
    "type": "function",
    "function": {
        "name": "propose_skill_amendment",
        "description": (
            "Propose an amendment to an existing non-core skill. Use when a "
            "learned/synthesized skill's implementation has a bug, an "
            "insufficient timeout, incorrect defaults, or other issues revealed "
            "by the trace. Core skills (built-in primitives like click_element, "
            "view_screen, etc.) cannot be amended -- suggest plan-level "
            "workarounds or new wrapper skills instead. You may call this "
            "multiple times before calling learning_done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "The id of the existing skill to amend.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "What is wrong with the skill.",
                },
                "proposed_change": {
                    "type": "string",
                    "description": "How the skill should be changed.",
                },
                "evidence": {
                    "type": "string",
                    "description": "What in the trace indicates the problem (reference event lines or screenshots).",
                },
            },
            "required": ["skill_id", "issue_description", "proposed_change"],
        },
    },
}

TOOL_LEARNING_DONE = {
    "type": "function",
    "function": {
        "name": "learning_done",
        "description": (
            "Signal that your analysis is complete. Provide your final "
            "diagnosis summary, failure point (if applicable), suggestions for "
            "the planner, and revised plan hints. Any skills/amendments proposed "
            "via propose_new_skill or propose_skill_amendment are automatically "
            "included in the output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2-3 sentence diagnosis or assessment.",
                },
                "failure_point": {
                    "type": "string",
                    "description": "Which step failed and what happened (null for success traces).",
                },
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "affected_step_idx": {"type": "integer"},
                            "category": {
                                "type": "string",
                                "enum": ["timing", "targeting", "missing_skill", "precondition", "other"],
                            },
                        },
                        "required": ["description", "category"],
                    },
                    "description": "Concrete suggestions for the planner (max 5).",
                },
                "revised_plan_hints": {
                    "type": "string",
                    "description": "Free-text guidance for the planner's next attempt.",
                },
            },
            "required": ["summary", "suggestions", "revised_plan_hints"],
        },
    },
}


ALL_TOOLS = [
    TOOL_GET_TRACE_SUMMARY,
    TOOL_GET_EVENTS,
    TOOL_GET_EVENT_FIELD,
    TOOL_VIEW_SCREENSHOT,
    TOOL_LIST_SCREENSHOTS,
    TOOL_GET_PROGRAM,
    TOOL_PROPOSE_NEW_SKILL,
    TOOL_PROPOSE_SKILL_AMENDMENT,
    TOOL_LEARNING_DONE,
]


# ---------------------------------------------------------------------------
# Dispatch context: holds loaded trace state for tool handlers
# ---------------------------------------------------------------------------


class TraceContext:
    """Holds the loaded trace state so tool handlers can access it."""

    def __init__(self, task_dir: Path, program_text: str | None = None) -> None:
        self.task_dir = task_dir
        self.meta, self.events = load_trace(task_dir)
        self._summary: TraceSummary | None = None
        self.program_text = program_text
        self.screens_dir = task_dir / "screens"

    @property
    def summary(self) -> TraceSummary:
        if self._summary is None:
            self._summary = analyze_trace(self.task_dir)
        return self._summary

    def screenshot_paths(self) -> list[Path]:
        if not self.screens_dir.is_dir():
            return []
        return sorted(self.screens_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Tool dispatch handlers
# ---------------------------------------------------------------------------


def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ctx: TraceContext,
) -> tuple[str | list[dict[str, Any]], bool]:
    """Dispatch a learner tool call.

    Returns (result_content, is_done). result_content is either a plain string
    or a list of multimodal content blocks (for image responses).
    """
    try:
        handler = _HANDLERS.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"unknown tool: {tool_name!r}"}), False
        return handler(arguments, ctx)
    except Exception as exc:
        log.warning("learner tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), False


def _handle_get_trace_summary(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    s = ctx.summary
    payload = {
        "task_id": s.task_id,
        "name": s.name,
        "status": s.status,
        "started": s.started,
        "finished": s.finished,
        "step_count": s.step_count,
        "event_count": s.event_count,
        "screenshot_count": s.screenshot_count,
        "total_duration_ms": round(s.total_duration_ms, 1),
        "skill_sequence": s.skill_sequence,
        "timings": {
            sid: {
                "calls": t.calls,
                "mean_ms": round(t.mean_ms, 1),
                "p95_ms": round(t.p95_ms, 1),
                "total_ms": round(t.total_ms, 1),
            }
            for sid, t in s.timings.items()
        },
        "failures": {
            sid: asdict(f) for sid, f in s.failures.items()
        },
    }
    return json.dumps(payload, indent=2), False


def _handle_get_events(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    limit = min(args.get("limit", 30), _MAX_EVENTS_PER_REQUEST)
    kind_filter = args.get("kind_filter")
    start_line = args.get("start_line", 0)
    end_line = args.get("end_line")

    if end_line is None:
        end_line = start_line + limit

    results = []
    for i, event in enumerate(ctx.events):
        if i < start_line:
            continue
        if i >= end_line:
            break
        if kind_filter and event.get("kind") != kind_filter:
            continue
        results.append({"line": i, **event})
        if len(results) >= limit:
            break

    total = len(ctx.events)
    header = f"Showing {len(results)} events (total {total} in trace, lines {start_line}-{end_line})"
    if kind_filter:
        header += f", filtered by kind={kind_filter!r}"

    lines = [header, "---"]
    for evt in results:
        lines.append(json.dumps(evt, default=str))

    return "\n".join(lines), False


def _handle_get_event_field(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    line = args["line"]
    field_path = args["field_path"]

    if line < 0 or line >= len(ctx.events):
        return json.dumps({"error": f"line {line} out of range (0-{len(ctx.events)-1})"}), False

    event = ctx.events[line]
    parts = field_path.split(".")
    current: Any = event
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return json.dumps({"error": f"field '{field_path}' not found at '{part}'", "available_keys": list(current.keys())}), False
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return json.dumps({"error": f"cannot index list with '{part}'"}), False
        else:
            return json.dumps({"error": f"cannot traverse into {type(current).__name__} at '{part}'"}), False

    result = json.dumps(current, default=str)
    if len(result) > 10000:
        result = result[:10000] + f"\n... [truncated, {len(result)} chars total]"
    return f"event[{line}].{field_path} =\n{result}", False


def _handle_view_screenshot(args: dict[str, Any], ctx: TraceContext) -> tuple[str | list[dict[str, Any]], bool]:
    index = args["index"]
    paths = ctx.screenshot_paths()

    if not paths:
        return json.dumps({"error": "no screenshots available in this trace"}), False

    if index < 1 or index > len(paths):
        return json.dumps({"error": f"index {index} out of range (1-{len(paths)})"}), False

    img_path = paths[index - 1]
    b64_data, mime = _encode_image_for_llm(img_path)

    content_parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"Screenshot {index}/{len(paths)}: {img_path.name} ({img_path})"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64_data}"},
        },
    ]
    return content_parts, False


def _handle_list_screenshots(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    paths = ctx.screenshot_paths()
    if not paths:
        return "No screenshots available in this trace.", False

    # Match screenshots to their trace events for timestamps
    screenshot_events = [
        e for e in ctx.events if e.get("kind") == "screenshot"
    ]
    ts_map: dict[str, str] = {}
    for evt in screenshot_events:
        data = evt.get("data", {})
        p = data.get("path", "")
        filename = p.split("/")[-1] if "/" in p else p
        ts_map[filename] = evt.get("ts", "")

    lines = [f"Available screenshots: {len(paths)}", "---"]
    for i, p in enumerate(paths, 1):
        ts = ts_map.get(p.name, "unknown")
        lines.append(f"  {i:3d}. {p.name}  ts={ts}")

    return "\n".join(lines), False


def _handle_get_program(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    if ctx.program_text:
        return ctx.program_text, False
    # Try to find program info from trace events
    for evt in ctx.events:
        if evt.get("kind") == "program_started":
            data = evt.get("data", {})
            parts = [f"Program: {data.get('name', 'unknown')}"]
            if data.get("description"):
                parts.append(f"Description: {data['description']}")
            parts.append(f"Steps: {data.get('step_count', '?')}")
            if data.get("skills"):
                parts.append(f"Skills used: {', '.join(data['skills'])}")
            return "\n".join(parts), False
    return json.dumps({"error": "no program information available for this trace"}), False


def _handle_learning_done(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    return "Analysis complete.", True


def _handle_propose_new_skill(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    proposed_id = args.get("proposed_id", "")
    if not proposed_id:
        return json.dumps({"error": "proposed_id is required"}), False
    return f"Skill proposal '{proposed_id}' recorded.", False


def _handle_propose_skill_amendment(args: dict[str, Any], ctx: TraceContext) -> tuple[str, bool]:
    skill_id = args.get("skill_id", "")
    if not skill_id:
        return json.dumps({"error": "skill_id is required"}), False
    core = _get_core_skills()
    if skill_id in core:
        return json.dumps({
            "error": f"'{skill_id}' is a core skill and cannot be amended. "
            "Core skills are built-in primitives. Instead, suggest a "
            "plan-level workaround or propose a new wrapper skill.",
        }), False
    return f"Amendment proposal for '{skill_id}' recorded.", False


_HANDLERS: dict[str, Any] = {
    "get_trace_summary": _handle_get_trace_summary,
    "get_events": _handle_get_events,
    "get_event_field": _handle_get_event_field,
    "view_screenshot": _handle_view_screenshot,
    "list_screenshots": _handle_list_screenshots,
    "get_program": _handle_get_program,
    "learning_done": _handle_learning_done,
    "propose_new_skill": _handle_propose_new_skill,
    "propose_skill_amendment": _handle_propose_skill_amendment,
}
