"""monitor_text_region: VLM-backed text watcher (daemon)."""

from __future__ import annotations

import asyncio
import base64
import io
import time
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from daedalus.backends.protocol import Rect
from daedalus.core import DaemonSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion
from daedalus.llm.gateway import LLMCall


class _Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def to_rect(self) -> Rect:
        return Rect(x=self.x, y=self.y, width=self.width, height=self.height)


class MonitorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64)
    region: _Region
    interval_ms: int = Field(ge=200, le=600_000, default=1000)
    role: str = Field(default="vision")


class MonitorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    captured_at: float


_PROMPT = (
    "You are a text reader. Look at the supplied screenshot of a small region "
    "and return ONLY the text you can read, verbatim, with no extra commentary. "
    "If you cannot read any text, return an empty string."
)


@register
class MonitorTextRegion(DaemonSkill):
    SPEC = SkillSpec(
        id="monitor_text_region",
        version=SkillVersion(raw="0.1.0"),
        kind="daemon",
        description=(
            "Periodically captures a screen region and asks the vision LLM what "
            "text it contains. Publishes {text, captured_at} to task_state[key]."
        ),
        side_effects=["screen_capture", "llm_call", "task_state_write"],
        preconditions=[
            "backend.connected",
            "ctx.llm is configured with a vision role",
            "interval_ms >= 200",
        ],
        publishes_state_key="monitored_text",
        self_publishes=True,
        examples=[
            SkillExample(
                inputs={
                    "key": "current_score",
                    "region": {"x": 1200, "y": 60, "width": 200, "height": 50},
                    "interval_ms": 1000,
                }
            ),
        ],
        tests=[],
        requires=["view_screen"],
        tags=["daemon", "vision", "monitoring"],
    )
    Inputs = MonitorInput
    Outputs = MonitorOutput

    async def loop(self, inputs: MonitorInput, ctx: ExecutionContext) -> AsyncIterator[MonitorOutput]:  # type: ignore[override]
        if ctx.llm is None:
            raise RuntimeError("monitor_text_region requires an LLM gateway")

        interval_s = inputs.interval_ms / 1000.0
        rect = inputs.region.to_rect()
        try:
            while True:
                if ctx.aborted():
                    return
                shot = ctx.backend.screenshot(region=rect)
                buf = io.BytesIO()
                shot.image.convert("RGB").save(buf, format="PNG")
                png = buf.getvalue()
                b64 = base64.b64encode(png).decode("ascii")
                messages = [
                    {"role": "system", "content": _PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Read the text in this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ]
                resp = ctx.llm.complete(LLMCall(role=inputs.role, messages=messages, temperature=0.0))
                out = MonitorOutput(text=resp.content.strip(), captured_at=time.time())
                ctx.task_state.set(inputs.key, out.model_dump(mode="json"))
                yield out
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
