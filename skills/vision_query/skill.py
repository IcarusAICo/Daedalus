"""vision_query: ask a vision LLM a question about an image.

READ-ONLY observation skill — returns text only, never clicks or interacts.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion
from daedalus.llm.gateway import LLMCall, UnknownRoleError

log = logging.getLogger(__name__)


class VisionReasonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000, description="Question or instruction for the vision model.")
    image_path: str | None = Field(
        default=None,
        description=(
            "Path to a PNG image to analyze. If not provided, uses the last "
            "screenshot taken by view_screen."
        ),
    )
    role: str = Field(default="vision", description="LLM role to invoke.")


class VisionReasonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(description="The vision model's answer/reasoning about the image content.")


@register
class VisionQuery(AtomicSkill):
    SPEC = SkillSpec(
        id="vision_query",
        version=SkillVersion(raw="0.3.0"),
        kind="atomic",
        description=(
            "Ask a question about an image. By default uses the last screenshot "
            "from view_screen, or pass an explicit image_path. "
            "This skill is READ-ONLY — it never clicks, types, or interacts."
        ),
        side_effects=["llm_call"],
        preconditions=["ctx.llm is configured with a vision role"],
        postconditions=["len(response) > 0"],
        examples=[
            SkillExample(
                inputs={"prompt": "What numbers are visible in the puzzle grid?"},
                note="Ask about the last screenshot taken by view_screen.",
            ),
            SkillExample(
                inputs={"prompt": "What text is on the button?", "image_path": "/path/to/screenshot.png"},
                note="Ask about a specific image file.",
            ),
        ],
        tests=["basic.json"],
        tags=["vision", "reasoning", "llm", "observation", "core"],
    )
    Inputs = VisionReasonInput
    Outputs = VisionReasonOutput

    def run(self, inputs: VisionReasonInput, ctx: ExecutionContext) -> VisionReasonOutput:  # type: ignore[override]
        if ctx.llm is None:
            return VisionReasonOutput(response="[no llm configured]")

        image_path = inputs.image_path
        if image_path is None:
            last = ctx.task_state.get("last_screenshot")
            if last and isinstance(last, dict) and "image_path" in last:
                image_path = last["image_path"]

        if image_path is None:
            return VisionReasonOutput(response="[no image available — call view_screen first]")

        p = Path(image_path)
        if not p.exists():
            return VisionReasonOutput(response=f"[image not found: {image_path}]")

        raw = p.read_bytes()
        mime = "image/png"
        if len(raw) > 3_500_000:
            import io as _io
            from PIL import Image as _Image
            img = _Image.open(_io.BytesIO(raw))
            buf = _io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            raw = buf.getvalue()
            mime = "image/jpeg"
        image_b64 = base64.b64encode(raw).decode("ascii")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": inputs.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            },
        ]
        call = LLMCall(
            role=inputs.role,
            messages=messages,
            temperature=0.0,
        )
        try:
            resp = ctx.llm.complete(call)
        except UnknownRoleError:
            if inputs.role != "vision":
                log.warning(
                    "LLM role %r not configured, falling back to 'vision'",
                    inputs.role,
                )
                call = LLMCall(role="vision", messages=messages, temperature=0.0)
                resp = ctx.llm.complete(call)
            else:
                raise
        return VisionReasonOutput(response=resp.content)
