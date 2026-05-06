"""store_query: use a vision LLM to extract structured data from the screen into a RunStore table."""

from __future__ import annotations

import base64
import io
import json
import logging
import re

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion
from daedalus.llm.gateway import LLMCall

log = logging.getLogger(__name__)


class StoreQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "What to extract from the screen. Describe the data and its structure. "
            "The LLM should return a JSON array of objects with consistent keys."
        ),
    )
    store_table: str = Field(
        min_length=1,
        max_length=100,
        description="RunStore table name to store results (created automatically from first row keys).",
    )
    schema_hint: dict[str, str] = Field(
        default_factory=dict,
        description="Optional column type hints, e.g. {'x': 'int', 'y': 'int', 'label': 'str'}.",
    )


class StoreQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="Number of rows extracted and stored.")
    store_table: str
    columns: list[str] = Field(default_factory=list, description="Column names in the table.")


@register
class StoreQuery(AtomicSkill):
    SPEC = SkillSpec(
        id="store_query",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Ask a vision LLM to extract structured data from the current screen "
            "and store it in a RunStore table. The LLM returns a JSON array of "
            "objects; each object becomes a row."
        ),
        side_effects=["screen_capture", "network", "task_state_write"],
        preconditions=["backend.connected"],
        postconditions=["store table populated with extracted data"],
        examples=[
            SkillExample(
                inputs={
                    "question": "Identify every clickable grid edge. Return each as {x, y, orientation}.",
                    "store_table": "grid_edges",
                    "schema_hint": {"x": "int", "y": "int", "orientation": "str"},
                },
                note="Extract grid edge positions from a puzzle screenshot.",
            ),
        ],
        tests=[],
        tags=["vision", "store", "extraction", "core"],
    )
    Inputs = StoreQueryInput
    Outputs = StoreQueryOutput

    def run(self, inputs: StoreQueryInput, ctx: ExecutionContext) -> StoreQueryOutput:  # type: ignore[override]
        if ctx.store is None:
            raise RuntimeError("RunStore not available on ExecutionContext")
        if ctx.llm is None:
            raise RuntimeError("No LLM gateway configured for store_query")

        shot = ctx.backend.screenshot()
        buf = io.BytesIO()
        shot.image.convert("RGB").save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        schema_desc = ""
        if inputs.schema_hint:
            schema_desc = (
                "\n\nEach object MUST have exactly these keys with these types:\n"
                + json.dumps(inputs.schema_hint)
            )

        prompt = (
            f"The screen is {shot.width}x{shot.height} pixels.\n\n"
            f"TASK: {inputs.question}{schema_desc}\n\n"
            "Return ONLY a JSON array of objects. Each object is one data row.\n"
            "If nothing matches, return an empty array: []\n"
            "Return ONLY the JSON array, no other text."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }]

        resp = ctx.llm.complete(LLMCall(role="vision", messages=messages, temperature=0.0))
        raw = resp.content.strip()

        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            log.warning("store_query got non-array response: %s", raw[:200])
            return StoreQueryOutput(count=0, store_table=inputs.store_table)

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            log.warning("store_query JSON parse error: %s", exc)
            return StoreQueryOutput(count=0, store_table=inputs.store_table)

        if not items or not isinstance(items, list):
            return StoreQueryOutput(count=0, store_table=inputs.store_table)

        # Infer schema from hint or first row
        if inputs.schema_hint:
            schema = dict(inputs.schema_hint)
        else:
            first = items[0] if isinstance(items[0], dict) else {}
            schema = {}
            for k, v in first.items():
                if isinstance(v, int):
                    schema[k] = "int"
                elif isinstance(v, float):
                    schema[k] = "float"
                elif isinstance(v, bool):
                    schema[k] = "bool"
                else:
                    schema[k] = "str"

        ctx.store.create_table(inputs.store_table, schema)

        count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            row = {}
            for k in schema:
                if k in item:
                    row[k] = item[k]
            ctx.store.append(inputs.store_table, row)
            count += 1

        return StoreQueryOutput(
            count=count,
            store_table=inputs.store_table,
            columns=list(schema.keys()),
        )
