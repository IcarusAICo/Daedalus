"""type_text: type a literal text string."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class TypeTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=0, max_length=10_000)


class TypeTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chars_typed: int


@register
class TypeText(AtomicSkill):
    SPEC = SkillSpec(
        id="type_text",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description="Type a literal text string at the current focus.",
        side_effects=["screen_input"],
        preconditions=["backend.connected"],
        examples=[
            SkillExample(
                inputs={"text": "hello world"},
                expected={"chars_typed": 11},
            ),
        ],
        tests=["basic.json"],
        tags=["keyboard", "input", "core"],
    )
    Inputs = TypeTextInput
    Outputs = TypeTextOutput

    def run(self, inputs: TypeTextInput, ctx: ExecutionContext) -> TypeTextOutput:  # type: ignore[override]
        ctx.backend.write(inputs.text)
        return TypeTextOutput(chars_typed=len(inputs.text))
