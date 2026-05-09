"""PDB-style plan debugger for the Learner.

Executes a plan step-by-step with breakpoints, pausing control back to the
learner agent at breakpoints or errors so it can investigate interactively.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from daedalus.core.context import ExecutionContext, TaskState, compute_coordinate_scale
from daedalus.core.errors import SkillNotFoundError, UserAbortError
from daedalus.core.registry import Registry, get_registry
from daedalus.core.store import RunStore
from daedalus.executor.program_executor import SkillProxy, lint_plan_code
from daedalus.executor.runner import StepResult
from daedalus.tracing.recorder import TraceRecorder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions for the debugger
# ---------------------------------------------------------------------------

TOOL_DEBUG_PLAN: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "debug_plan",
        "description": (
            "Start a debug session on the most recent plan. The plan will execute "
            "step-by-step, pausing at breakpoints or errors. You receive a state "
            "snapshot at each pause and can use other tools to investigate, then "
            "call debug_continue, debug_step, or debug_stop to control execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "breakpoints": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "List of step indices (0-based) where execution should pause. "
                        "The debugger pauses BEFORE executing the step at each index."
                    ),
                },
                "break_on_error": {
                    "type": "boolean",
                    "description": "If true (default), pause on any skill error instead of crashing.",
                },
                "plan_source": {
                    "type": "string",
                    "description": "Which plan to debug. Default 'latest' uses the plan from the trace.",
                },
            },
        },
    },
}

TOOL_DEBUG_CONTINUE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "debug_continue",
        "description": (
            "Resume execution of the debugged plan until the next breakpoint "
            "or error is hit, or the plan finishes."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_DEBUG_STEP: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "debug_step",
        "description": (
            "Execute exactly one more skill call in the debugged plan, then "
            "pause again. Returns the result of the executed step."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_DEBUG_STOP: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "debug_stop",
        "description": (
            "Abort the current debug session. Execution is terminated and "
            "a summary of completed steps is returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Debuggable Skill Proxy
# ---------------------------------------------------------------------------


class DebuggableSkillProxy(SkillProxy):
    """SkillProxy subclass that supports breakpoints and pause/resume.

    Before each skill invocation, checks if the current step index is in the
    breakpoint set or if single-step mode is active. If so, captures state
    and blocks until the controller signals to proceed.
    """

    def __init__(
        self,
        execution_ctx: ExecutionContext,
        registry: Registry,
        tracer: TraceRecorder,
        abort_event: threading.Event,
        *,
        breakpoints: set[int] | None = None,
        break_on_error: bool = True,
        step_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(
            execution_ctx=execution_ctx,
            registry=registry,
            tracer=tracer,
            abort_event=abort_event,
            step_timeout_s=step_timeout_s,
        )
        self._breakpoints = breakpoints or set()
        self._break_on_error = break_on_error
        self._single_step = False

        # Synchronization primitives
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()  # Start unpaused

        # State visible to the controller
        self._paused = False
        self._pause_reason: str | None = None
        self._last_error: str | None = None
        self._finished = False
        self._plan_error: str | None = None

    def __getattr__(self, name: str) -> Any:
        try:
            entry = self._registry.get(name)
        except SkillNotFoundError:
            raise AttributeError(f"no skill named {name!r}") from None

        if entry.cls.SPEC.kind == "service":
            return self._make_service_starter(entry)

        def _invoke(**kwargs: Any) -> Any:
            if self._abort_event.is_set():
                raise UserAbortError("aborted by debugger")

            idx = self._step_idx

            # Check breakpoint BEFORE execution
            if idx in self._breakpoints or self._single_step:
                self._pause_reason = f"breakpoint at step {idx}" if idx in self._breakpoints else f"single-step at step {idx}"
                self._paused = True
                self._pause_event.set()
                self._resume_event.clear()
                self._resume_event.wait()
                self._paused = False
                self._single_step = False

                if self._abort_event.is_set():
                    raise UserAbortError("aborted by debugger")

            self._step_idx += 1

            inputs_model = entry.cls.Inputs.model_validate(kwargs)
            self._tracer.skill_started(
                skill_id=entry.id,
                version=entry.version.raw,
                step_idx=idx,
                inputs=kwargs,
                content_hash=entry.content_hash,
            )

            t0 = time.perf_counter()
            try:
                instance = entry.cls()
                output = instance.run(inputs_model, self._ctx)
            except Exception as exc:
                dur = (time.perf_counter() - t0) * 1000
                error_msg = f"{type(exc).__name__}: {exc}"
                self._tracer.skill_error(
                    skill_id=entry.id,
                    step_idx=idx,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    duration_ms=dur,
                )
                self._results.append(StepResult(
                    skill_id=entry.id, step_idx=idx, status="failed",
                    duration_ms=dur, error=error_msg,
                ))
                self._last_error = error_msg

                if self._break_on_error:
                    self._pause_reason = f"error at step {idx}: {error_msg}"
                    self._paused = True
                    self._pause_event.set()
                    self._resume_event.clear()
                    self._resume_event.wait()
                    self._paused = False

                    if self._abort_event.is_set():
                        raise UserAbortError("aborted by debugger")
                    return None
                raise

            dur = (time.perf_counter() - t0) * 1000
            out_dict = output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output)
            self._tracer.skill_finished(
                skill_id=entry.id,
                step_idx=idx,
                outputs=out_dict,
                duration_ms=dur,
            )
            self._results.append(StepResult(
                skill_id=entry.id, step_idx=idx, status="success",
                duration_ms=dur, output=out_dict,
            ))
            self._last_error = None
            return output

        return _invoke

    def get_state_snapshot(self) -> dict[str, Any]:
        """Capture current debugger state for the learner."""
        return {
            "step_idx": self._step_idx,
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "last_error": self._last_error,
            "steps_completed": len(self._results),
            "finished": self._finished,
            "results_summary": [
                {"step": r.step_idx, "skill": r.skill_id, "status": r.status, "duration_ms": round(r.duration_ms, 1)}
                for r in self._results[-5:]  # last 5 results
            ],
        }


# ---------------------------------------------------------------------------
# Plan Debugger
# ---------------------------------------------------------------------------


_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "hasattr", "int", "isinstance",
    "len", "list", "map", "max", "min", "pow", "print", "range",
    "repr", "reversed", "round", "set", "sorted", "str", "sum",
    "tuple", "type", "zip",
}


class PlanDebugger:
    """Executes a plan with breakpoints, providing step-by-step control.

    The plan runs in a background thread. The debugger uses threading events
    to pause/resume execution and communicate state back to the learner.
    """

    def __init__(
        self,
        program_code: str,
        backend: Any,
        registry: Registry | None = None,
        gateway: Any | None = None,
        traces_root: Path | None = None,
        tasks_db: Path | None = None,
        breakpoints: list[int] | None = None,
        break_on_error: bool = True,
    ) -> None:
        self._program_code = self._extract_code(program_code)
        self._backend = backend
        self._registry = registry or get_registry()
        self._gateway = gateway
        self._traces_root = traces_root or Path("traces")
        self._tasks_db = tasks_db
        self._breakpoints = set(breakpoints or [])
        self._break_on_error = break_on_error

        self._proxy: DebuggableSkillProxy | None = None
        self._thread: threading.Thread | None = None
        self._abort_event = threading.Event()
        self._started = False

    def _extract_code(self, program_text: str) -> str:
        """Extract raw Python code from the program text summary."""
        lines = program_text.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip() == "Plan code (Python):":
                in_code = True
                continue
            if in_code:
                if line.startswith("  ") and len(line) > 4 and line[2:5].strip().isdigit():
                    # Strip line numbers like "  1: code_here"
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        code_lines.append(parts[1][1:] if parts[1].startswith(" ") else parts[1])
                    else:
                        code_lines.append(line)
                elif line.startswith("    "):
                    code_lines.append(line[4:])
                elif not line.strip():
                    code_lines.append("")
                else:
                    break
        if code_lines:
            return "\n".join(code_lines)
        return program_text

    def start(self) -> dict[str, Any]:
        """Start executing the plan in a background thread."""
        if self._started:
            return {"error": "Debug session already started."}

        lint_errors = lint_plan_code(self._program_code)
        if lint_errors:
            return {"error": "Plan code has lint errors", "lint_errors": lint_errors}

        self._started = True

        db_path = self._tasks_db or self._traces_root / "tasks.db"
        task_id = "debug_session"

        tracer = TraceRecorder(
            traces_root=self._traces_root,
            db_path=db_path,
            task_name="debug_session",
            task_id=task_id,
        )

        screen_w = self._backend.size[0] if hasattr(self._backend, "size") else 1728
        state = TaskState(db_path, task_id)
        store = RunStore(db_path, task_id)

        ctx = ExecutionContext(
            task_id=task_id,
            backend=self._backend,
            task_state=state,
            tracer=tracer,
            store=store,
            llm=self._gateway,
            abort_event=self._abort_event,
            coordinate_scale=compute_coordinate_scale(screen_w),
        )

        self._proxy = DebuggableSkillProxy(
            execution_ctx=ctx,
            registry=self._registry,
            tracer=tracer,
            abort_event=self._abort_event,
            breakpoints=self._breakpoints,
            break_on_error=self._break_on_error,
        )

        self._thread = threading.Thread(
            target=self._run_plan,
            daemon=True,
        )
        self._thread.start()

        # Wait for either a breakpoint pause or completion
        return self._wait_for_pause_or_finish()

    def _run_plan(self) -> None:
        """Execute the plan in the background thread."""
        import builtins as _builtins
        import collections
        import copy
        import functools
        import heapq
        import itertools
        import math
        import operator
        import random
        import re
        import statistics
        import string
        import textwrap

        safe_builtins = {
            name: getattr(_builtins, name)
            for name in _ALLOWED_BUILTINS
            if hasattr(_builtins, name)
        }
        safe_builtins["True"] = True
        safe_builtins["False"] = False
        safe_builtins["None"] = None

        sandbox: dict[str, Any] = {"__builtins__": safe_builtins}
        sandbox["math"] = math
        sandbox["re"] = re
        sandbox["json"] = json
        sandbox["itertools"] = itertools
        sandbox["functools"] = functools
        sandbox["collections"] = collections
        sandbox["string"] = string
        sandbox["textwrap"] = textwrap
        sandbox["copy"] = copy
        sandbox["operator"] = operator
        sandbox["heapq"] = heapq
        sandbox["time"] = time
        sandbox["random"] = random
        sandbox["statistics"] = statistics

        func_code = "def __plan__(ctx):\n"
        for line in self._program_code.splitlines():
            func_code += f"    {line}\n"

        try:
            exec(compile(func_code, "<debug_plan>", "exec"), sandbox)  # noqa: S102
            plan_fn = sandbox["__plan__"]
            plan_fn(self._proxy)
            self._proxy._finished = True
        except UserAbortError:
            self._proxy._finished = True
            self._proxy._plan_error = "Aborted by user/debugger"
        except Exception as exc:
            self._proxy._finished = True
            self._proxy._plan_error = f"{type(exc).__name__}: {exc}"
        finally:
            # Signal pause event so controller isn't stuck waiting
            self._proxy._pause_event.set()

    def _wait_for_pause_or_finish(self, timeout: float = 300.0) -> dict[str, Any]:
        """Block until the plan pauses at a breakpoint or finishes."""
        self._proxy._pause_event.wait(timeout=timeout)
        self._proxy._pause_event.clear()

        state = self._proxy.get_state_snapshot()
        if self._proxy._finished:
            state["status"] = "finished"
            if self._proxy._plan_error:
                state["plan_error"] = self._proxy._plan_error
        else:
            state["status"] = "paused"
        return state

    def continue_execution(self) -> dict[str, Any]:
        """Resume until next breakpoint or finish."""
        if self._proxy is None or self._proxy._finished:
            return {"error": "Plan has already finished.", "status": "finished"}

        self._proxy._single_step = False
        self._proxy._resume_event.set()
        return self._wait_for_pause_or_finish()

    def step(self) -> dict[str, Any]:
        """Execute one more step then pause."""
        if self._proxy is None or self._proxy._finished:
            return {"error": "Plan has already finished.", "status": "finished"}

        self._proxy._single_step = True
        self._proxy._resume_event.set()
        return self._wait_for_pause_or_finish()

    def stop(self) -> dict[str, Any]:
        """Abort the debug session."""
        if self._proxy is None:
            return {"status": "not_started"}

        self._abort_event.set()
        self._proxy._resume_event.set()  # Unblock if paused

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        state = self._proxy.get_state_snapshot()
        state["status"] = "stopped"
        return state
