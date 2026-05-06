"""Daemon skill lifecycle.

A :class:`daedalus.core.DaemonSkill` is a long-lived async generator: it yields
state updates that the executor forwards into ``task_state`` under a key
declared in the spec (``publishes_state_key``). The :class:`DaemonHandle`
class owns one daemon's lifetime: a background thread runs an asyncio loop
that drives the generator, and we expose ``stop()`` to cancel cooperatively.

Design notes
------------
- Daemons run on their own ``asyncio`` loop in their own thread, which keeps
  the synchronous executor unchanged and lets us add as many daemons as the
  user wants without async-coloring the main code path.
- Daemons must check ``ctx.aborted()`` (or honor cancellation) to exit
  promptly. The lifecycle gives them up to ``stop_timeout_s`` after
  cancellation before joining the thread.
- We capture exceptions and surface them via :attr:`DaemonHandle.error`; the
  executor inspects that at shutdown and emits a trace event.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daedalus.core.context import ExecutionContext
from daedalus.core.registry import Registry, get_registry

if TYPE_CHECKING:
    from daedalus.core.skill import DaemonSkill

log = logging.getLogger(__name__)


@dataclass
class DaemonSpec:
    """A reference + inputs that the executor will spin up before running steps."""

    skill: str
    inputs: dict[str, Any] = field(default_factory=dict)
    version: str | None = None


class DaemonHandle:
    """Owns one running daemon: thread + asyncio loop + stop signal."""

    def __init__(
        self,
        skill_cls: type["DaemonSkill"],
        inputs_obj: Any,
        ctx: ExecutionContext,
        stop_timeout_s: float = 2.0,
    ) -> None:
        self._skill_cls = skill_cls
        self._inputs = inputs_obj
        self._ctx = ctx
        self._stop_timeout_s = stop_timeout_s
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_evt = threading.Event()
        self._started_evt = threading.Event()
        self._update_count = 0
        self.error: BaseException | None = None

    @property
    def skill_id(self) -> str:
        return self._skill_cls.SPEC.id

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"daedalus-daemon:{self.skill_id}",
            daemon=True,
        )
        self._thread.start()
        # Wait briefly for the loop to come up.
        self._started_evt.wait(timeout=1.0)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._task is not None and self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread is not None:
            self._thread.join(timeout=self._stop_timeout_s)

    @property
    def update_count(self) -> int:
        return self._update_count

    # -----------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._task = self._loop.create_task(self._drive())
            self._started_evt.set()
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self.error = exc
            log.warning("daemon %s crashed: %s", self.skill_id, exc)
        finally:
            self._started_evt.set()
            if self._loop is not None:
                with contextlib.suppress(Exception):
                    self._loop.close()

    async def _drive(self) -> None:
        skill = self._skill_cls()
        publishes = self._skill_cls.SPEC.publishes_state_key
        gen = skill.loop(self._inputs, self._ctx)  # type: ignore[attr-defined]
        try:
            async for value in gen:
                if self._stop_evt.is_set() or self._ctx.aborted():
                    break
                # Validate against the skill's Outputs.
                validated = self._skill_cls.Outputs.model_validate(
                    value.model_dump(mode="json") if hasattr(value, "model_dump") else value
                )
                payload = validated.model_dump(mode="json") if hasattr(validated, "model_dump") else validated
                self._update_count += 1
                if publishes and not self._skill_cls.SPEC.self_publishes:
                    self._ctx.task_state.set(publishes, payload)
                self._ctx.tracer.emit(
                    "daemon_update",
                    {
                        "skill_id": self.skill_id,
                        "update_idx": self._update_count,
                        "publishes_key": publishes,
                    },
                )
        finally:
            # Make sure async generator gets a chance to clean up.
            with contextlib.suppress(Exception):
                await gen.aclose()


# ---------------------------------------------------------------------------
# Lifecycle helpers used by the executor
# ---------------------------------------------------------------------------


def start_daemons(
    daemon_specs: list[DaemonSpec],
    ctx: ExecutionContext,
    *,
    registry: Registry | None = None,
) -> list[DaemonHandle]:
    """Resolve, validate, and start all declared daemons. Returns handles."""
    registry = registry if registry is not None else get_registry()
    handles: list[DaemonHandle] = []
    for ds in daemon_specs:
        entry = registry.get(ds.skill, version_constraint=ds.version)
        if entry.cls.SPEC.kind != "daemon":
            raise RuntimeError(
                f"{ds.skill} is registered as kind={entry.cls.SPEC.kind!r}, not daemon"
            )
        inputs = entry.cls.Inputs.model_validate(ds.inputs)
        ctx.tracer.emit(
            "daemon_started",
            {
                "skill_id": entry.id,
                "version": entry.version.raw,
                "publishes_key": entry.cls.SPEC.publishes_state_key,
                "inputs": inputs.model_dump(mode="json") if hasattr(inputs, "model_dump") else inputs,
            },
        )
        h = DaemonHandle(entry.cls, inputs, ctx)  # type: ignore[arg-type]
        h.start()
        handles.append(h)
    return handles


def stop_daemons(handles: list[DaemonHandle], ctx: ExecutionContext) -> None:
    for h in handles:
        h.stop()
        ctx.tracer.emit(
            "daemon_stopped",
            {
                "skill_id": h.skill_id,
                "updates": h.update_count,
                "error": str(h.error) if h.error else None,
            },
            level="error" if h.error else "info",
        )
