"""Resetter: LLM-based environment reset before plan execution.

Generates a short Python program that returns the environment to a clean state
(e.g., closing apps that the goal will reopen). Generated once per run and
executed before each attempt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from daedalus.executor.dsl import PythonProgram, parse_python_program
from daedalus.llm.gateway import LLMCall, LLMGateway

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Reset Planner for a computer-control agent. Given a user goal,
produce a SHORT program that resets the environment to a clean state BEFORE
the goal is attempted.

TARGET HOST
-----------
- Operating system: __HOST_OS__
- Screen resolution: __SCREEN_W__x__SCREEN_H__ pixels.
__OS_HINTS__

PURPOSE
-------
The reset program runs before each attempt so the environment is predictable.
Examples:
- If the goal mentions opening Firefox → close all Firefox windows first.
- If the goal mentions opening an app → close that app first.
- If the goal involves a fresh browser tab → close existing browser windows.

Keep it MINIMAL — only close/reset things directly relevant to the goal.
If no reset is needed (e.g. the goal is purely observational), return null.

AVAILABLE SKILLS
----------------
- ctx.type_shortcut(keys=["ctrl","w"]) — press keyboard shortcuts
- ctx.wait(ms=500) — pause for milliseconds
- ctx.view_screen() — take a screenshot (read-only)

OUTPUT FORMAT
-------------
Respond with EXACTLY one JSON object on a single line, no prose, no markdown:

  {"program": {"name": "reset_env", "dsl_version": 2, "code": "<python body>"}}

Or if no reset is needed:

  {"program": null}

PYTHON RULES
------------
- Call skills as: ctx.<skill_id>(param1=val1, param2=val2)
- Modules pre-imported: time, re, json, math, collections, itertools.
- DO NOT write import statements. They are FORBIDDEN and will crash.
- DO NOT use subprocess, os, sys, open, or any system calls.
- Keep it under 10 lines. Only close/kill relevant apps using keyboard shortcuts.
- Use the correct OS shortcuts:
  - macOS: Cmd+Q to quit app (["super","q"]), Cmd+W to close window
  - Linux: Alt+F4 to close window (["alt","F4"])
  - Windows: Alt+F4 to close window (["alt","F4"])
"""

_OS_HINTS = {
    "macos": "- macOS: Use Cmd (super) key. Cmd+Q quits apps, Cmd+W closes windows.",
    "linux": "- Linux: Use Alt+F4 to close windows. Desktop is GNOME/KDE.",
    "windows": "- Windows: Use Alt+F4 to close windows.",
}


class Resetter:
    """Generates a reset program from the goal, cached for reuse across attempts."""

    def __init__(self, gateway: LLMGateway, *, role: str = "cheap") -> None:
        self._gateway = gateway
        self._role = role

    def generate(
        self,
        goal: str,
        host_os: str = "unknown",
        screen_w: int = 1920,
        screen_h: int = 1080,
    ) -> PythonProgram | None:
        """Generate a reset program. Returns None if no reset is needed."""
        os_hints = _OS_HINTS.get(host_os.lower(), "")
        system = (
            _SYSTEM_PROMPT
            .replace("__HOST_OS__", host_os)
            .replace("__SCREEN_W__", str(screen_w))
            .replace("__SCREEN_H__", str(screen_h))
            .replace("__OS_HINTS__", os_hints)
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {goal}"},
        ]

        call = LLMCall(
            role=self._role,
            messages=messages,
            temperature=0.0,
        )

        try:
            resp = self._gateway.complete(call)
        except Exception as exc:
            log.warning("resetter LLM call failed: %s", exc)
            return None

        text = resp.content.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object from prose-wrapped response
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError as exc2:
                    log.warning("resetter: could not parse LLM response as JSON: %s", exc2)
                    return None
            else:
                log.warning("resetter: no JSON object found in LLM response")
                return None

        program_data = data.get("program")
        if program_data is None:
            log.info("resetter: LLM determined no reset needed")
            return None

        try:
            program = parse_python_program(program_data)
        except Exception as exc:
            log.warning("resetter: could not parse program: %s", exc)
            return None

        # Validate the generated code passes safety lint before returning.
        from daedalus.executor.program_executor import lint_plan_code
        errors = lint_plan_code(program.code)
        if errors:
            log.warning("resetter: generated code failed lint: %s", errors)
            return None

        return program
