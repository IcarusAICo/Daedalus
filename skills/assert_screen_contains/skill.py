"""assert_screen_contains: visual claim verification via the vision LLM."""

from __future__ import annotations

import base64
import io
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from daedalus.backends.protocol import Rect
from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
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


class AssertScreenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=3, max_length=2000)
    region: _Region | None = None
    role: str = Field(
        default="vision",
        description="LLM role to invoke for the visual judgement.",
    )


class AssertScreenOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: bool
    confidence: Literal["low", "medium", "high"] = "medium"
    explanation: str = Field(default="")


_PROMPT = """\
You are a visual assertion judge for a computer-control agent.

Given a screenshot and a CLAIM in natural language, decide whether the claim is
true on screen RIGHT NOW. Respond with EXACTLY one JSON object on a single
line, no prose:

  {"verdict": true|false, "confidence": "low|medium|high", "explanation": "<one sentence>"}

Be conservative: if you're not sure, answer false with low confidence and say
why in one short sentence. Never guess about content you cannot see clearly.
"""


def _parse(content: str) -> AssertScreenOutput:
    text = content.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Try to find a {...} block
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b <= a:
            raise ValueError(f"vision model returned non-JSON: {text!r}") from exc
        data = json.loads(text[a : b + 1])
    return AssertScreenOutput.model_validate(data)


@register
class AssertScreenContains(AtomicSkill):
    SPEC = SkillSpec(
        id="assert_screen_contains",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Vision-LM-backed visual assertion: captures the screen and asks the "
            "vision model whether `claim` is true. Returns verdict + confidence."
        ),
        side_effects=["screen_capture", "llm_call", "task_state_write"],
        preconditions=["backend.connected", "ctx.llm is configured with a vision role"],
        postconditions=["verdict in {true, false}"],
        examples=[
            SkillExample(
                inputs={"claim": "Notepad is open with the text 'hello' visible"},
                note="Verify Notepad opened and we typed correctly.",
            ),
        ],
        tests=[],
        requires=["view_screen"],
        tags=["vision", "verification", "assertion", "core"],
    )
    Inputs = AssertScreenInput
    Outputs = AssertScreenOutput

    def run(self, inputs: AssertScreenInput, ctx: ExecutionContext) -> AssertScreenOutput:  # type: ignore[override]
        if ctx.llm is None:
            raise RuntimeError(
                "assert_screen_contains requires an LLM gateway in the execution context"
            )

        rect = inputs.region.to_rect() if inputs.region else None
        shot = ctx.backend.screenshot(region=rect)
        buf = io.BytesIO()
        shot.image.convert("RGB").save(buf, format="PNG")
        png = buf.getvalue()
        image_b64 = base64.b64encode(png).decode("ascii")

        # Persist the frame the assertion saw, so the Teacher can replay it.
        ctx.tracer.attach_screenshot(png, width=shot.width, height=shot.height)

        # Anthropic / OpenAI multimodal message shape (LiteLLM normalises both).
        messages = [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"CLAIM: {inputs.claim}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        call = LLMCall(
            role=inputs.role,
            messages=messages,
            response_format="json_object",
            temperature=0.0,
        )
        resp = ctx.llm.complete(call)
        try:
            verdict = _parse(resp.content)
        except Exception as exc:
            raise RuntimeError(f"vision model returned unparseable response: {exc}") from exc

        ctx.task_state.set(
            "last_assertion",
            {"claim": inputs.claim, **verdict.model_dump()},
        )
        ctx.tracer.emit(
            "visual_assertion",
            {
                "claim": inputs.claim,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "explanation": verdict.explanation,
                "screen_region": rect.__dict__ if rect else None,
            },
        )
        return verdict
