"""click_element: locate a UI element and click it in one atomic step.

Combines locate_element + mouse so plans never need hardcoded pixel
coordinates for UI interactions.  Captures the screen, sends it to the
grounding service (or vision LLM fallback), and clicks the best match.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from daedalus.backends.protocol import Button
from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion
from daedalus.llm.gateway import LLMCall

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:8420"


class ClickElementInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=1,
        max_length=500,
        description="Natural-language description of the element to click, e.g. 'submit button'.",
    )
    button: Button = Field(default=Button.LEFT)
    double: bool = Field(default=False, description="If true, perform a double-click.")
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept a match.",
    )


class ClickElementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    clicked: bool
    x: int | None = Field(default=None, description="X coordinate where click was performed.")
    y: int | None = Field(default=None, description="Y coordinate where click was performed.")
    confidence: float = Field(default=0.0, description="Confidence of the element match.")
    label: str = Field(default="", description="Label of the matched element.")


@register
class ClickElement(AtomicSkill):
    SPEC = SkillSpec(
        id="click_element",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Locate a UI element by natural-language description and click it. "
            "Combines grounding + click in one step so coordinates are resolved "
            "dynamically at runtime, never hardcoded."
        ),
        side_effects=["screen_capture", "screen_input", "network"],
        preconditions=["backend.connected"],
        postconditions=["found implies click was performed at element center"],
        examples=[
            SkillExample(
                inputs={"description": "submit button"},
                note="Find and left-click the submit button.",
            ),
            SkillExample(
                inputs={"description": "close X icon", "button": "right"},
                note="Right-click the close icon.",
            ),
        ],
        tests=[],
        tags=["mouse", "grounding", "vision", "input", "core"],
    )
    Inputs = ClickElementInput
    Outputs = ClickElementOutput

    def run(self, inputs: ClickElementInput, ctx: ExecutionContext) -> ClickElementOutput:  # type: ignore[override]
        from daedalus.core.context import llm_image_size

        shot = ctx.backend.screenshot()

        # Send full-resolution image to grounding for best accuracy,
        # but request coordinates in the LLM's reference frame.
        llm_w, llm_h = llm_image_size(shot.width, shot.height)

        buf = io.BytesIO()
        shot.image.convert("RGB").save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        grounding_cfg = (ctx.config or {}).get("grounding", {})
        endpoint = grounding_cfg.get("endpoint", _DEFAULT_ENDPOINT)
        timeout_s = float(grounding_cfg.get("timeout_s", 10))

        x, y, confidence, label = self._locate(
            inputs, ctx, endpoint, timeout_s, image_b64,
            llm_w, llm_h,
            target_width=llm_w,
            target_height=llm_h,
        )

        if x is None or y is None:
            return ClickElementOutput(found=False, clicked=False, label=label)

        # Scale coordinates from LLM space back to actual backend resolution
        scale = ctx.coordinate_scale
        click_x = int(x * scale)
        click_y = int(y * scale)

        ctx.backend.click(click_x, click_y, button=inputs.button, double=inputs.double)

        return ClickElementOutput(
            found=True,
            clicked=True,
            x=x,
            y=y,
            confidence=confidence,
            label=label,
        )

    def _locate(
        self,
        inputs: ClickElementInput,
        ctx: ExecutionContext,
        endpoint: str,
        timeout_s: float,
        image_b64: str,
        screen_w: int,
        screen_h: int,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> tuple[int | None, int | None, float, str]:
        """Returns (x, y, confidence, label). x/y are None if not found."""
        try:
            payload: dict = {
                "image_b64": image_b64,
                "description": inputs.description,
                "mode": "point",
                "confidence_threshold": inputs.confidence_threshold,
            }
            if target_width is not None:
                payload["target_width"] = target_width
            if target_height is not None:
                payload["target_height"] = target_height

            resp = requests.post(
                f"{endpoint}/locate",
                json=payload,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            log.warning(
                "grounding service unreachable at %s — falling back to vision LLM",
                endpoint,
            )
            return self._vision_fallback(inputs, ctx, image_b64, screen_w, screen_h)
        except Exception as exc:
            log.error("grounding service error: %s — falling back to vision LLM", exc)
            return self._vision_fallback(inputs, ctx, image_b64, screen_w, screen_h)

        matches = data.get("matches", [])
        if not matches:
            # If the grounding service returned instantly with no matches, it's
            # likely non-functional (stub endpoint). Fall back to vision LLM.
            if data.get("locate_time_ms", -1) < 1.0:
                log.warning("grounding service returned 0 matches in 0ms — likely non-functional, falling back to vision LLM")
                return self._vision_fallback(inputs, ctx, image_b64, screen_w, screen_h)
            return None, None, 0.0, "no_match"

        best = matches[0]
        return best["x"], best["y"], best["confidence"], best.get("label", "")

    def _vision_fallback(
        self,
        inputs: ClickElementInput,
        ctx: ExecutionContext,
        image_b64: str,
        screen_w: int,
        screen_h: int,
    ) -> tuple[int | None, int | None, float, str]:
        """Use the vision LLM to estimate element coordinates."""
        if ctx.llm is None:
            return None, None, 0.0, "no_llm_configured"

        prompt = (
            f"The screen is {screen_w}x{screen_h} pixels. "
            f"Find the UI element matching this description: \"{inputs.description}\"\n\n"
            "Return ONLY a JSON object on one line with these fields:\n"
            '  {"found": true, "x": <center_x_int>, "y": <center_y_int>, "label": "<short label>"}\n'
            "If the element is not visible, return:\n"
            '  {"found": false}\n'
            "Return ONLY the JSON, no other text."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        call = LLMCall(role="vision", messages=messages, temperature=0.0)

        try:
            resp = ctx.llm.complete(call)
            raw_text = resp.content.strip()
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                log.warning("vision fallback returned non-JSON: %s", raw_text[:200])
                return None, None, 0.0, "vision_fallback_parse_error"
            data = json.loads(json_match.group())
        except Exception as exc:
            log.error("vision LLM fallback failed: %s", exc)
            return None, None, 0.0, f"vision_fallback_error: {exc}"

        if not data.get("found"):
            return None, None, 0.0, "not_found_by_vision"

        x = max(0, min(int(data.get("x", 0)), screen_w - 1))
        y = max(0, min(int(data.get("y", 0)), screen_h - 1))
        label = str(data.get("label", inputs.description))

        return x, y, 0.5, label
