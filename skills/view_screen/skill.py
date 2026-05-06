"""view_screen: capture the current screen and save it to disk."""

from __future__ import annotations

import io

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.context import llm_image_size
from daedalus.core.spec import SkillExample, SkillVersion


class ViewScreenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ViewScreenOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    captured_at: float
    image_path: str = Field(description="Absolute path to the saved PNG screenshot on disk.")


@register
class ViewScreen(AtomicSkill):
    SPEC = SkillSpec(
        id="view_screen",
        version=SkillVersion(raw="0.3.0"),
        kind="atomic",
        description=(
            "Capture the current screen and save it as a PNG file. "
            "Returns the file path, width, and height. The image is "
            "downscaled to the LLM's internal processing resolution so "
            "that all coordinates are consistent across vision skills."
        ),
        side_effects=["screen_capture", "filesystem_write", "task_state_write"],
        preconditions=["backend.connected"],
        postconditions=["image_path exists on disk"],
        examples=[
            SkillExample(inputs={}, note="Capture the entire screen."),
        ],
        tests=["basic.json"],
        tags=["screen", "capture", "core"],
    )
    Inputs = ViewScreenInput
    Outputs = ViewScreenOutput

    def run(self, inputs: ViewScreenInput, ctx: ExecutionContext) -> ViewScreenOutput:  # type: ignore[override]
        shot = ctx.backend.screenshot()

        llm_w, llm_h = llm_image_size(shot.width, shot.height)

        img = shot.image
        if (llm_w, llm_h) != (shot.width, shot.height):
            from PIL import Image as _Image
            img = img.resize((llm_w, llm_h), _Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        path = ctx.tracer.attach_screenshot(png, width=llm_w, height=llm_h)

        ctx.task_state.set(
            "last_screenshot",
            {
                "image_path": str(path),
                "width": llm_w,
                "height": llm_h,
                "captured_at": shot.captured_at,
            },
        )

        return ViewScreenOutput(
            width=llm_w,
            height=llm_h,
            captured_at=shot.captured_at,
            image_path=str(path),
        )
