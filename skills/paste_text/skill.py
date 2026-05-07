"""paste_text: copy text to the macOS clipboard then paste with Cmd+V."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class PasteTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The exact text to place on the clipboard and paste.")


class PasteTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    text_length: int = Field(description="Number of characters pasted.")


@register
class PasteText(AtomicSkill):
    SPEC = SkillSpec(
        id="paste_text",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Paste text into the currently focused input field by writing to the macOS "
            "clipboard via NSPasteboard and triggering Cmd+V. Avoids auto-correct and IME issues."
        ),
        side_effects=["screen_input"],
        preconditions=[
            "backend.connected",
            "macOS with PyObjC available",
            "an input field is focused",
        ],
        postconditions=[
            "clipboard contains the given text",
            "text has been pasted into the focused field",
        ],
        examples=[
            SkillExample(
                inputs={"text": "Hello, World!"},
                expected={"success": True, "text_length": 13},
            ),
        ],
        tests=["basic.json"],
        tags=["clipboard", "input", "macos", "text"],
    )
    Inputs = PasteTextInput
    Outputs = PasteTextOutput

    def run(self, inputs: PasteTextInput, ctx: ExecutionContext) -> PasteTextOutput:  # type: ignore[override]
        try:
            from AppKit import NSPasteboard, NSPasteboardTypeString  # type: ignore

            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(inputs.text, NSPasteboardTypeString)
        except ImportError:
            # Fallback: use pbcopy via the backend's key/type mechanism if PyObjC
            # is unavailable (e.g. in tests or non-macOS environments).
            # We signal failure so the caller can decide what to do.
            return PasteTextOutput(success=False, text_length=len(inputs.text))

        # Paste using Cmd+V
        ctx.backend.press("command", "v")
        time.sleep(0.1)

        return PasteTextOutput(success=True, text_length=len(inputs.text))
