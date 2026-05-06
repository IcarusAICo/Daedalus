"""tick_counter: reference daemon skill. Emits a tick + counter on a cadence."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import DaemonSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class TickInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(default="tick", min_length=1, max_length=64)
    interval_ms: int = Field(default=200, ge=10, le=600_000)
    max_ticks: int | None = Field(
        default=None,
        ge=1,
        description="Stop after this many ticks. None means run forever (until cancelled).",
    )


class TickOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    elapsed_ms: int


@register
class TickCounter(DaemonSkill):
    SPEC = SkillSpec(
        id="tick_counter",
        version=SkillVersion(raw="0.1.0"),
        kind="daemon",
        description=(
            "Long-running daemon that increments a counter at a fixed cadence and "
            "publishes the current value (and elapsed time) to task_state[key]."
        ),
        side_effects=["clock", "task_state_write"],
        preconditions=["interval_ms > 0"],
        postconditions=["task_state[key].count >= 1 once daemon has been running for >= interval_ms"],
        publishes_state_key="tick",
        self_publishes=True,
        examples=[
            SkillExample(inputs={"key": "tick", "interval_ms": 100}),
        ],
        tests=[],
        tags=["daemon", "heartbeat", "core"],
    )
    Inputs = TickInput
    Outputs = TickOutput

    async def loop(self, inputs: TickInput, ctx: ExecutionContext) -> AsyncIterator[TickOutput]:  # type: ignore[override]
        interval_s = inputs.interval_ms / 1000.0
        started = time.perf_counter()
        count = 0
        # Allow per-instance override of the publishes_state_key by accepting
        # an explicit `key`. The daemons.py forwarder uses SPEC.publishes_state_key
        # as the channel; we honor that *and* mirror to the requested key for
        # convenience.
        try:
            while True:
                if ctx.aborted():
                    return
                count += 1
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                out = TickOutput(count=count, elapsed_ms=elapsed_ms)
                if inputs.key:
                    ctx.task_state.set(
                        inputs.key,
                        out.model_dump(mode="json"),
                    )
                yield out
                if inputs.max_ticks is not None and count >= inputs.max_ticks:
                    return
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
