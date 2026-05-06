"""scroll: scroll the viewport up, down, left, or right."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion

# macOS VNC scroll events are very fine-grained (1 pixel per tick).
# This multiplier makes each logical "amount" unit feel like a normal scroll.
_TICKS_PER_UNIT = 50


class ScrollInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(description="X coordinate to position the mouse before scrolling.")
    y: int = Field(description="Y coordinate to position the mouse before scrolling.")
    direction: Literal["up", "down", "left", "right"] = Field(
        description="Direction to scroll.",
    )
    amount: int = Field(
        default=12,
        ge=1,
        le=100,
        description=(
            "Scroll intensity. Recommended values: 5=a few lines, "
            "15=half a page, 30=one full page, 60=two pages. "
            "Default 12 (roughly a third of a page). Values below 5 are barely perceptible."
        ),
    )


class ScrollOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: str
    amount: int
    ticks: int = Field(description="Actual VNC scroll ticks sent.")
    dx: int = Field(description="Horizontal scroll delta applied.")
    dy: int = Field(description="Vertical scroll delta applied.")


@register
class Scroll(AtomicSkill):
    SPEC = SkillSpec(
        id="scroll",
        version=SkillVersion(raw="0.3.0"),
        kind="atomic",
        description=(
            "Scroll at a specific screen location. Moves the mouse to (x, y) "
            "then scrolls in the given direction. Use amount=15 for half a page, "
            "amount=30 for a full page. Values below 5 are barely visible."
        ),
        side_effects=["screen_input"],
        preconditions=["backend.connected"],
        postconditions=["viewport shifted by the requested amount"],
        examples=[
            SkillExample(
                inputs={"x": 500, "y": 400, "direction": "down", "amount": 15},
                note="Scroll down half a page.",
            ),
            SkillExample(
                inputs={"x": 500, "y": 400, "direction": "up", "amount": 30},
                note="Scroll up one full page.",
            ),
            SkillExample(
                inputs={"x": 500, "y": 400, "direction": "down", "amount": 60},
                note="Scroll down two full pages.",
            ),
        ],
        tests=[],
        tags=["mouse", "input", "scroll", "core"],
    )
    Inputs = ScrollInput
    Outputs = ScrollOutput

    def run(self, inputs: ScrollInput, ctx: ExecutionContext) -> ScrollOutput:  # type: ignore[override]
        ctx.backend.move(inputs.x, inputs.y)

        ticks = inputs.amount * _TICKS_PER_UNIT
        dx = 0
        dy = 0

        if inputs.direction == "down":
            dy = ticks
        elif inputs.direction == "up":
            dy = -ticks
        elif inputs.direction == "right":
            dx = ticks
        elif inputs.direction == "left":
            dx = -ticks

        ctx.backend.scroll(dx, dy)

        return ScrollOutput(
            direction=inputs.direction,
            amount=inputs.amount,
            ticks=ticks,
            dx=dx,
            dy=dy,
        )
