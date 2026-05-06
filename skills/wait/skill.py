"""wait: explicit pause between steps."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class WaitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ms: int = Field(ge=0, le=600_000, description="Milliseconds to sleep (max 10 min).")


class WaitOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    waited_ms: int


@register
class Wait(AtomicSkill):
    SPEC = SkillSpec(
        id="wait",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description="Sleep for the given number of milliseconds.",
        side_effects=["clock"],
        preconditions=["ms >= 0"],
        examples=[SkillExample(inputs={"ms": 250}, expected={"waited_ms": 250})],
        tests=["basic.json"],
        tags=["timing", "core"],
    )
    Inputs = WaitInput
    Outputs = WaitOutput

    def run(self, inputs: WaitInput, ctx: ExecutionContext) -> WaitOutput:  # type: ignore[override]
        # Honour aborts: poll in 50ms slices instead of one long sleep.
        slice_s = 0.05
        remaining = inputs.ms / 1000.0
        slept = 0.0
        while remaining > 0 and not ctx.aborted():
            chunk = min(slice_s, remaining)
            time.sleep(chunk)
            slept += chunk
            remaining -= chunk
        return WaitOutput(waited_ms=int(slept * 1000))
