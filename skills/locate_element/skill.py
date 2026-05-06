"""locate_element: find UI elements by natural-language description.

Captures the current screen and sends it to the grounding microservice,
which uses OmniParser V2 (or a fallback model) to detect and locate
elements matching the description.

If the grounding service is unavailable, falls back to a vision LLM
to estimate coordinates from the screenshot.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion
from daedalus.llm.gateway import LLMCall

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:8420"


class LocateElementInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=1,
        max_length=500,
        description="Natural-language description of the element to find, e.g. 'submit button'.",
    )
    mode: Literal["point", "box", "all"] = Field(
        default="point",
        description="'point' returns center coords, 'box' returns bounding box, 'all' returns all matches.",
    )
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to consider a match.",
    )


class LocateElementMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    x: int
    y: int
    box: list[int] | None = None
    confidence: float


class LocateElementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    x: int | None = Field(default=None, description="Center X of the best match.")
    y: int | None = Field(default=None, description="Center Y of the best match.")
    box: list[int] | None = Field(
        default=None, description="Bounding box [x1,y1,x2,y2] of the best match."
    )
    matches: list[LocateElementMatch] = Field(
        default_factory=list, description="All matches above threshold."
    )
    confidence: float = Field(default=0.0, description="Confidence of the best match.")
    label: str = Field(default="", description="Label of the best match.")


@register
class LocateElement(AtomicSkill):
    SPEC = SkillSpec(
        id="locate_element",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Locate a UI element by natural-language description. Captures the screen, "
            "sends it to the grounding service, and returns coordinates/bounding box."
        ),
        side_effects=["screen_capture", "network"],
        preconditions=["backend.connected"],
        postconditions=["found implies x and y are set"],
        examples=[
            SkillExample(
                inputs={"description": "submit button"},
                note="Find the submit button and return its center coordinates.",
            ),
            SkillExample(
                inputs={"description": "close X icon", "mode": "box"},
                note="Find the close button and return its bounding box.",
            ),
        ],
        tests=[],
        tags=["grounding", "vision", "coordinates", "core"],
    )
    Inputs = LocateElementInput
    Outputs = LocateElementOutput

    def run(self, inputs: LocateElementInput, ctx: ExecutionContext) -> LocateElementOutput:  # type: ignore[override]
        from daedalus.core.context import llm_image_size

        shot = ctx.backend.screenshot()

        # Downscale to LLM resolution so returned coordinates match view_screen's space
        llm_w, llm_h = llm_image_size(shot.width, shot.height)
        img = shot.image
        if (llm_w, llm_h) != (shot.width, shot.height):
            from PIL import Image as _Image
            img = img.resize((llm_w, llm_h), _Image.LANCZOS)

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        grounding_cfg = (ctx.config or {}).get("grounding", {})
        endpoint = grounding_cfg.get("endpoint", _DEFAULT_ENDPOINT)
        timeout_s = float(grounding_cfg.get("timeout_s", 10))

        try:
            resp = requests.post(
                f"{endpoint}/locate",
                json={
                    "image_b64": image_b64,
                    "description": inputs.description,
                    "mode": inputs.mode,
                    "confidence_threshold": inputs.confidence_threshold,
                },
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            log.warning("grounding service unreachable at %s — falling back to vision LLM", endpoint)
            return self._vision_fallback(inputs, ctx, image_b64, llm_w, llm_h)
        except Exception as exc:
            log.error("grounding service error: %s — falling back to vision LLM", exc)
            return self._vision_fallback(inputs, ctx, image_b64, llm_w, llm_h)

        matches = data.get("matches", [])
        if not matches:
            # If the grounding service returned instantly with no matches, it's
            # likely non-functional (stub endpoint). Fall back to vision LLM.
            if data.get("locate_time_ms", -1) < 1.0:
                log.warning("grounding service returned 0 matches in 0ms — likely non-functional, falling back to vision LLM")
                return self._vision_fallback(inputs, ctx, image_b64, llm_w, llm_h)
            return LocateElementOutput(found=False, label="no_match")

        parsed_matches = [
            LocateElementMatch(
                label=m["label"],
                x=m["x"],
                y=m["y"],
                box=list(m["box"]) if m.get("box") else None,
                confidence=m["confidence"],
            )
            for m in matches
        ]

        if not parsed_matches:
            return LocateElementOutput(found=False, label="no_match")

        best = parsed_matches[0]
        return LocateElementOutput(
            found=True,
            x=best.x,
            y=best.y,
            box=best.box,
            matches=parsed_matches if inputs.mode == "all" else [best],
            confidence=best.confidence,
            label=best.label,
        )

    def _vision_fallback(
        self,
        inputs: LocateElementInput,
        ctx: ExecutionContext,
        image_b64: str,
        screen_w: int,
        screen_h: int,
    ) -> LocateElementOutput:
        """Use the vision LLM to estimate element coordinates when the grounding service is down."""
        if ctx.llm is None:
            return LocateElementOutput(found=False, label="no_llm_configured")

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
            # Extract JSON from possible markdown fencing
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                log.warning("vision fallback returned non-JSON: %s", raw_text[:200])
                return LocateElementOutput(found=False, label="vision_fallback_parse_error")
            data = json.loads(json_match.group())
        except Exception as exc:
            log.error("vision LLM fallback failed: %s", exc)
            return LocateElementOutput(found=False, label=f"vision_fallback_error: {exc}")

        if not data.get("found"):
            return LocateElementOutput(found=False, label="not_found_by_vision")

        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        label = str(data.get("label", inputs.description))

        x = max(0, min(x, screen_w - 1))
        y = max(0, min(y, screen_h - 1))

        match = LocateElementMatch(label=label, x=x, y=y, box=None, confidence=0.5)
        return LocateElementOutput(
            found=True,
            x=x,
            y=y,
            matches=[match],
            confidence=0.5,
            label=label,
        )
