"""click_all: click every coordinate row in a RunStore table."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from daedalus.backends.protocol import Button
from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class ClickAllInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_table: str = Field(
        min_length=1,
        description="RunStore table name containing rows with 'x' and 'y' columns.",
    )
    button: Button = Field(default=Button.LEFT)
    double: bool = Field(default=False)
    delay_ms: int = Field(
        default=200,
        ge=0,
        le=5000,
        description="Milliseconds to wait between clicks.",
    )


class ClickAllOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clicked: int = Field(description="Number of clicks performed.")
    store_table: str


@register
class ClickAll(AtomicSkill):
    SPEC = SkillSpec(
        id="click_all",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Click every coordinate in a RunStore table. The table must have "
            "'x' and 'y' integer columns. Clicks are performed sequentially "
            "with a configurable delay between them."
        ),
        side_effects=["screen_input"],
        preconditions=["backend.connected"],
        postconditions=["all rows clicked"],
        examples=[
            SkillExample(
                inputs={"store_table": "spots", "delay_ms": 100},
                note="Click all spots with 100ms delay between clicks.",
            ),
        ],
        tests=[],
        tags=["mouse", "input", "store", "core"],
    )
    Inputs = ClickAllInput
    Outputs = ClickAllOutput

    def run(self, inputs: ClickAllInput, ctx: ExecutionContext) -> ClickAllOutput:  # type: ignore[override]
        if ctx.store is None:
            raise RuntimeError("RunStore not available on ExecutionContext")

        rows = ctx.store.all_rows(inputs.store_table)
        clicked = 0
        scale = ctx.coordinate_scale
        for row in rows:
            x = int(int(row["x"]) * scale)
            y = int(int(row["y"]) * scale)
            ctx.backend.click(x, y, button=inputs.button, double=inputs.double)
            clicked += 1
            if inputs.delay_ms > 0 and clicked < len(rows):
                time.sleep(inputs.delay_ms / 1000.0)

        return ClickAllOutput(clicked=clicked, store_table=inputs.store_table)
