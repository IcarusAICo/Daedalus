"""type_shortcut: press one or more keyboard shortcuts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class TypeShortcutInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keys: list[str] | list[list[str]] = Field(
        description=(
            "A single shortcut as a list of key names (e.g. ['ctrl','c']), "
            "or multiple shortcuts as a list of lists (e.g. [['ctrl','a'],['ctrl','c']])."
        ),
    )

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "TypeShortcutInput":
        if not self.keys:
            raise ValueError("keys must contain at least one shortcut")
        if len(self.keys) > 100:
            raise ValueError("keys must not exceed 100 shortcuts")
        return self

    @property
    def shortcuts(self) -> list[list[str]]:
        """Normalize to a list of shortcuts regardless of input form."""
        if not self.keys:
            return []
        if isinstance(self.keys[0], str):
            return [list(self.keys)]  # type: ignore[arg-type]
        return list(self.keys)  # type: ignore[arg-type]


class TypeShortcutOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    shortcuts_pressed: list[list[str]]


@register
class TypeShortcut(AtomicSkill):
    SPEC = SkillSpec(
        id="type_shortcut",
        version=SkillVersion(raw="0.2.0"),
        kind="atomic",
        description=(
            "Press one or more keyboard shortcuts. Accepts a single shortcut "
            "(list of keys) or multiple shortcuts (list of lists)."
        ),
        side_effects=["screen_input"],
        preconditions=["backend.connected", "len(keys) >= 1"],
        examples=[
            SkillExample(
                inputs={"keys": ["ctrl", "c"]},
                expected={"count": 1, "shortcuts_pressed": [["ctrl", "c"]]},
            ),
            SkillExample(
                inputs={"keys": [["ctrl", "a"], ["ctrl", "c"]]},
                expected={"count": 2, "shortcuts_pressed": [["ctrl", "a"], ["ctrl", "c"]]},
            ),
            SkillExample(
                inputs={"keys": [["Left"], ["Left"], ["Left"]]},
                expected={"count": 3, "shortcuts_pressed": [["Left"], ["Left"], ["Left"]]},
            ),
        ],
        tests=["basic.json"],
        tags=["keyboard", "shortcut", "core"],
    )
    Inputs = TypeShortcutInput
    Outputs = TypeShortcutOutput

    def run(self, inputs: TypeShortcutInput, ctx: ExecutionContext) -> TypeShortcutOutput:  # type: ignore[override]
        shortcuts = inputs.shortcuts
        for keys in shortcuts:
            ctx.backend.press(*keys)
        return TypeShortcutOutput(
            count=len(shortcuts),
            shortcuts_pressed=shortcuts,
        )
