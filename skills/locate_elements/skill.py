"""locate_elements: find ALL matching UI elements and store them in RunStore.

Captures the screen, sends it to the grounding service (or vision LLM
fallback), and appends every match to a RunStore table with x, y, label,
and confidence columns. This is the batch version of locate_element.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re

import requests
from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion
from daedalus.llm.gateway import LLMCall

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:8420"


class LocateElementsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=1,
        max_length=500,
        description="Natural-language description of the elements to find.",
    )
    store_table: str = Field(
        min_length=1,
        max_length=100,
        description="RunStore table name to append results to (created if needed).",
    )
    confidence_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to include a match.",
    )


class LocateElementsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="Number of elements found and stored.")
    store_table: str = Field(description="Table name where results were stored.")


@register
class LocateElements(AtomicSkill):
    SPEC = SkillSpec(
        id="locate_elements",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Find ALL UI elements matching a description and store them in the "
            "RunStore. Each match gets a row with x, y, label, and confidence."
        ),
        side_effects=["screen_capture", "network", "task_state_write"],
        preconditions=["backend.connected"],
        postconditions=["store table populated with matching elements"],
        examples=[
            SkillExample(
                inputs={"description": "clickable grid edge", "store_table": "edges"},
                note="Find all clickable grid edges and store in 'edges' table.",
            ),
        ],
        tests=[],
        tags=["grounding", "vision", "coordinates", "store", "core"],
    )
    Inputs = LocateElementsInput
    Outputs = LocateElementsOutput

    def run(self, inputs: LocateElementsInput, ctx: ExecutionContext) -> LocateElementsOutput:  # type: ignore[override]
        if ctx.store is None:
            raise RuntimeError("RunStore not available on ExecutionContext")

        from daedalus.core.context import llm_image_size

        shot = ctx.backend.screenshot()

        # Send full-resolution image for best grounding accuracy,
        # request coordinates in the LLM's reference frame.
        llm_w, llm_h = llm_image_size(shot.width, shot.height)

        buf = io.BytesIO()
        shot.image.convert("RGB").save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        grounding_cfg = (ctx.config or {}).get("grounding", {})
        endpoint = grounding_cfg.get("endpoint", _DEFAULT_ENDPOINT)
        timeout_s = float(grounding_cfg.get("timeout_s", 10))

        matches = self._locate_all(
            inputs, ctx, endpoint, timeout_s, image_b64,
            llm_w, llm_h,
        )

        ctx.store.create_table(inputs.store_table, {
            "x": "int",
            "y": "int",
            "label": "str",
            "confidence": "float",
        })

        for m in matches:
            ctx.store.append(inputs.store_table, m)

        return LocateElementsOutput(
            count=len(matches),
            store_table=inputs.store_table,
        )

    def _locate_all(
        self, inputs, ctx, endpoint, timeout_s, image_b64, screen_w, screen_h,
    ) -> list[dict]:
        try:
            resp = requests.post(
                f"{endpoint}/locate",
                json={
                    "image_b64": image_b64,
                    "description": inputs.description,
                    "mode": "all",
                    "confidence_threshold": inputs.confidence_threshold,
                    "target_width": screen_w,
                    "target_height": screen_h,
                },
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            log.warning("grounding service unreachable — falling back to vision LLM")
            return self._vision_fallback(inputs, ctx, image_b64, screen_w, screen_h)
        except Exception as exc:
            log.error("grounding service error: %s — falling back to vision LLM", exc)
            return self._vision_fallback(inputs, ctx, image_b64, screen_w, screen_h)

        return [
            {"x": m["x"], "y": m["y"], "label": m.get("label", ""), "confidence": m["confidence"]}
            for m in data.get("matches", [])
        ]

    def _vision_fallback(self, inputs, ctx, image_b64, screen_w, screen_h) -> list[dict]:
        if ctx.llm is None:
            return []

        prompt = (
            f"The screen is {screen_w}x{screen_h} pixels. "
            f"Find ALL UI elements matching: \"{inputs.description}\"\n\n"
            "Return ONLY a JSON array of objects, each with:\n"
            '  {"x": <int>, "y": <int>, "label": "<short label>"}\n'
            "If none are found, return an empty array: []\n"
            "Return ONLY the JSON array, no other text."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }]
        call = LLMCall(role="vision", messages=messages, temperature=0.0)

        try:
            resp = ctx.llm.complete(call)
            raw = resp.content.strip()
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                return []
            items = json.loads(json_match.group())
        except Exception as exc:
            log.error("vision fallback failed: %s", exc)
            return []

        results = []
        for item in items:
            if isinstance(item, dict) and "x" in item and "y" in item:
                x = max(0, min(int(item["x"]), screen_w - 1))
                y = max(0, min(int(item["y"]), screen_h - 1))
                results.append({
                    "x": x, "y": y,
                    "label": str(item.get("label", "")),
                    "confidence": 0.5,
                })
        return results
