"""type_text_secret: type text from an environment variable (never logged)."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillVersion


class TypeTextSecretInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_var: str = Field(min_length=1, max_length=256,
                         description="Name of an environment variable holding the text to type.")


class TypeTextSecretOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chars_typed: int


@register
class TypeTextSecret(AtomicSkill):
    SPEC = SkillSpec(
        id="type_text_secret",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Type text from an environment variable. The variable name is logged "
            "but the actual value never enters program YAML or trace files."
        ),
        side_effects=["screen_input"],
        preconditions=["backend.connected", "env_var is set in the process environment"],
        sensitive_inputs=["env_var"],
        tags=["keyboard", "input", "secret", "core"],
    )
    Inputs = TypeTextSecretInput
    Outputs = TypeTextSecretOutput

    def run(self, inputs: TypeTextSecretInput, ctx: ExecutionContext) -> TypeTextSecretOutput:
        value = os.environ.get(inputs.env_var)
        if value is None:
            raise RuntimeError(
                f"environment variable {inputs.env_var!r} is not set"
            )
        ctx.backend.write(value)
        return TypeTextSecretOutput(chars_typed=len(value))
