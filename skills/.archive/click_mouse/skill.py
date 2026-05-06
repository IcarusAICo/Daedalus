"""click_mouse: click at an absolute screen pixel."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from daedalus.backends.protocol import Button
from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class ClickInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0, le=4095, description="Screen x in pixels.")
    y: int = Field(ge=0, le=4095, description="Screen y in pixels.")
    button: Button = Field(default=Button.LEFT)
    double: bool = Field(default=False, description="If true, perform a double-click.")


class ClickOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clicked_at: tuple[int, int]
    button: Button
    double: bool


@register
class ClickMouse(AtomicSkill):
    SPEC = SkillSpec(
        id="click_mouse",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description="Click the mouse at an absolute screen pixel.",
        side_effects=["screen_input"],
        preconditions=["backend.connected", "0 <= x < screen.width", "0 <= y < screen.height"],
        postconditions=["cursor.position == (x, y)"],
        examples=[
            SkillExample(
                inputs={"x": 100, "y": 200},
                expected={"clicked_at": [100, 200], "button": "left", "double": False},
            ),
        ],
        tests=["basic.json"],
        tags=["mouse", "input", "core"],
    )
    Inputs = ClickInput
    Outputs = ClickOutput

    def run(self, inputs: ClickInput, ctx: ExecutionContext) -> ClickOutput:  # type: ignore[override]
        ctx.backend.click(inputs.x, inputs.y, button=inputs.button, double=inputs.double)
        return ClickOutput(
            clicked_at=(inputs.x, inputs.y),
            button=inputs.button,
            double=inputs.double,
        )
