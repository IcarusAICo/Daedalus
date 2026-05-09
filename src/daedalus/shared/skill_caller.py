"""Shared skill invocation and management utilities.

Used by both the Explorer and Learner agents to invoke skills against the VM,
implement new temp skills, and revise existing ones.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

from daedalus.core.context import ExecutionContext
from daedalus.core.errors import SkillNotFoundError
from daedalus.core.registry import Registry
from daedalus.implementor.implementor import ImplementorRequest, SyntheticSkillImplementor
from daedalus.library.librarian import Librarian

log = logging.getLogger(__name__)

_MAX_PNG_BYTES_FOR_LLM = 3_500_000


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def encode_image_for_llm(path: Path) -> tuple[str, str]:
    """Encode an image for LLM consumption, converting large PNGs to JPEG.

    Returns (base64_str, mime_type).
    On-disk file is never modified -- conversion is in-memory only.
    """
    raw = path.read_bytes()
    if len(raw) <= _MAX_PNG_BYTES_FOR_LLM:
        return base64.b64encode(raw).decode("ascii"), "image/png"

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()
    log.debug(
        "image %s converted PNG(%dKB)->JPEG(%dKB) for LLM",
        path.name, len(raw) // 1024, len(jpeg_bytes) // 1024,
    )
    return base64.b64encode(jpeg_bytes).decode("ascii"), "image/jpeg"


# ---------------------------------------------------------------------------
# Skill-to-tool conversion
# ---------------------------------------------------------------------------


def skill_to_tool_def(entry: Any) -> dict[str, Any]:
    """Convert a registered skill into an OpenAI function-calling tool definition."""
    spec = entry.cls.SPEC
    input_schema = entry.cls.Inputs.model_json_schema()

    params: dict[str, Any] = {"type": "object"}
    if "properties" in input_schema:
        params["properties"] = input_schema["properties"]
    else:
        params["properties"] = {}
    if "required" in input_schema:
        params["required"] = input_schema["required"]
    if "$defs" in input_schema:
        params["$defs"] = input_schema["$defs"]

    return {
        "type": "function",
        "function": {
            "name": entry.id,
            "description": spec.description,
            "parameters": params,
        },
    }


# ---------------------------------------------------------------------------
# Tool definitions for implement_skill and revise_skill
# ---------------------------------------------------------------------------


TOOL_IMPLEMENT_SKILL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "implement_skill",
        "description": (
            "Request implementation of a new skill. The system will use an LLM to "
            "generate, test, and publish the skill. Returns the tool signature "
            "(input/output schema) on success, or error details on failure. "
            "After a successful implementation, the new skill becomes available "
            "as a tool call in subsequent turns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Snake_case name for the new skill (e.g. 'extract_grid_state').",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of what the skill should do, including inputs/outputs.",
                },
            },
            "required": ["skill_name", "description"],
        },
    },
}

TOOL_REVISE_SKILL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "revise_skill",
        "description": (
            "Revise an existing temp skill that was implemented earlier in this "
            "session. Use this when you tested a skill and found a bug or "
            "limitation -- describe what is wrong and what the fix should be. "
            "The implementor will re-synthesize the skill incorporating your "
            "feedback and the current source code. The revised skill replaces "
            "the original in the registry. Only skills created via "
            "implement_skill in this session can be revised."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The skill_id to revise (must be a temp skill from this session).",
                },
                "feedback": {
                    "type": "string",
                    "description": (
                        "What is wrong with the current implementation and what "
                        "should change. Be specific -- include the observed failure, "
                        "the root cause if known, and the desired behaviour."
                    ),
                },
            },
            "required": ["skill_name", "feedback"],
        },
    },
}


# ---------------------------------------------------------------------------
# Skill invocation handler
# ---------------------------------------------------------------------------


def handle_skill_call(
    skill_id: str,
    kwargs: dict[str, Any],
    ctx: ExecutionContext,
    registry: Registry,
) -> str | list[dict[str, Any]]:
    """Execute a skill directly by ID with the provided arguments.

    Returns either a plain string or a multimodal content list (with image
    blocks) when the skill output contains base64 image data.
    """
    try:
        entry = registry.get(skill_id)
    except SkillNotFoundError:
        return json.dumps({"error": f"skill {skill_id!r} not found"})

    try:
        inputs_model = entry.cls.Inputs.model_validate(kwargs)
        instance = entry.cls()
        output = instance.run(inputs_model, ctx)
        out_dict = output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output)

        image_path = out_dict.get("image_path")
        if image_path and isinstance(image_path, str):
            img_file = Path(image_path)
            if img_file.exists():
                image_b64, mime = encode_image_for_llm(img_file)
                metadata = json.dumps(
                    {k: v for k, v in out_dict.items() if k != "image_path"},
                    default=str,
                )
                content_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": json.dumps({"image_path": image_path, **{k: v for k, v in out_dict.items() if k != "image_path"}}, default=str)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ]
                return content_parts

        image_b64_val = out_dict.pop("image_b64", None)
        if image_b64_val and isinstance(image_b64_val, str) and len(image_b64_val) > 100:
            metadata = json.dumps(out_dict, default=str)
            content_parts = [
                {"type": "text", "text": metadata},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64_val}"},
                },
            ]
            return content_parts

        result_str = json.dumps(out_dict, default=str)
        if len(result_str) > 8000:
            for key in list(out_dict.keys()):
                val = out_dict[key]
                if isinstance(val, str) and len(val) > 2000:
                    out_dict[key] = val[:200] + f"... [truncated, {len(val)} chars total]"
            result_str = json.dumps(out_dict, default=str)

        return result_str
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Implement / revise skill handlers
# ---------------------------------------------------------------------------


def handle_implement_skill(
    args: dict[str, Any],
    *,
    registry: Registry,
    implementor: SyntheticSkillImplementor,
    librarian: Librarian,
    temp_skills: list[str],
    new_skills: list[str],
) -> str:
    """Handle an implement_skill tool call. Shared by Explorer and Learner."""
    skill_name = args.get("skill_name", "")
    description = args.get("description", "")

    if not skill_name or not description:
        return json.dumps({"error": "skill_name and description are both required"})

    if skill_name in registry:
        entry = registry.get(skill_name)
        return json.dumps({
            "status": "already_exists",
            "inputs": entry.cls.Inputs.model_json_schema(),
            "outputs": entry.cls.Outputs.model_json_schema(),
        })

    request = ImplementorRequest(
        proposed_id=skill_name,
        description=description,
        rationale="Requested during exploration/learning phase",
        side_effects=["screen_capture", "screen_input", "llm_call"],
    )

    try:
        result = implementor.synthesize(request)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": f"Implementor error: {exc}"})

    if result.ok and result.bundle is not None:
        try:
            implementor.publish_temp(result.bundle)
            librarian.reindex()
            temp_skills.append(skill_name)
            new_skills.append(skill_name)

            entry = registry.get(skill_name)
            return json.dumps({
                "status": "success",
                "skill_id": skill_name,
                "inputs": entry.cls.Inputs.model_json_schema(),
                "outputs": entry.cls.Outputs.model_json_schema(),
                "note": "Skill is available as a temp skill. You can test and revise it.",
            })
        except Exception as exc:
            implementor.cleanup_temp(skill_name)
            return json.dumps({"status": "failed", "error": f"Publish error: {exc}"})
    else:
        errors = result.test_failures + [str(v) for v in result.violations]
        return json.dumps({
            "status": "failed",
            "errors": errors,
            "notes": result.notes,
        })


def handle_revise_skill(
    args: dict[str, Any],
    *,
    registry: Registry,
    implementor: SyntheticSkillImplementor,
    librarian: Librarian,
    temp_skills: list[str],
) -> str:
    """Handle a revise_skill tool call. Shared by Explorer and Learner."""
    skill_name = args.get("skill_name", "")
    feedback = args.get("feedback", "")

    if not skill_name or not feedback:
        return json.dumps({"error": "skill_name and feedback are both required"})

    if skill_name not in temp_skills:
        return json.dumps({
            "error": f"{skill_name!r} is not a temp skill from this session. "
            f"Only skills implemented in this session can be revised. "
            f"Available temp skills: {temp_skills}"
        })

    try:
        result = implementor.revise(skill_name, feedback)
    except Exception as exc:
        return json.dumps({"status": "failed", "error": f"Implementor error: {exc}"})

    if result.ok and result.bundle is not None:
        try:
            implementor.publish_temp(result.bundle)
            librarian.reindex()
            entry = registry.get(skill_name)
            return json.dumps({
                "status": "revised",
                "skill_id": skill_name,
                "inputs": entry.cls.Inputs.model_json_schema(),
                "outputs": entry.cls.Outputs.model_json_schema(),
                "note": "Skill has been revised and reloaded. Test it again.",
            })
        except Exception as exc:
            return json.dumps({"status": "failed", "error": f"Publish error: {exc}"})
    else:
        errors = result.test_failures + [str(v) for v in result.violations]
        return json.dumps({
            "status": "failed",
            "errors": errors,
            "notes": result.notes,
        })


def get_image_path_from_result(result_content: str | list[dict[str, Any]]) -> str | None:
    """Extract the image path from a skill call result if present."""
    if isinstance(result_content, list):
        for part in result_content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_val = part.get("text", "")
                if "image_path" in text_val:
                    try:
                        data = json.loads(text_val)
                        return data.get("image_path")
                    except (json.JSONDecodeError, AttributeError):
                        pass
    return None
