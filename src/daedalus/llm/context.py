"""Context management utilities for LLM message lists.

Provides functions to keep the conversation context within token budgets:
- Image pruning: removes old screenshots, keeping only recent ones
- Token estimation: heuristic-based token counting
- Summarize-and-compact: condenses old messages into a summary when near capacity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Tunables for context management, loadable from the 'context' YAML section."""

    max_images_in_context: int = 5
    max_context_tokens: int = 200_000
    compact_threshold_pct: float = 0.85
    keep_recent_messages: int = 6

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ContextConfig:
        if not d:
            return cls()
        return cls(
            max_images_in_context=int(d.get("max_images_in_context", 5)),
            max_context_tokens=int(d.get("max_context_tokens", 200_000)),
            compact_threshold_pct=float(d.get("compact_threshold_pct", 0.85)),
            keep_recent_messages=int(d.get("keep_recent_messages", 6)),
        )


DEFAULT_CONFIG = ContextConfig()

_active_config: ContextConfig = DEFAULT_CONFIG


def set_context_config(config: ContextConfig) -> None:
    """Set the active context config (call once at startup from CLI)."""
    global _active_config
    _active_config = config


def get_context_config() -> ContextConfig:
    """Return the active context config."""
    return _active_config


_TOKENS_PER_IMAGE = 1600
_CHARS_PER_TOKEN = 4


def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens in a message list using heuristics.

    Uses ~4 chars/token for text and ~1600 tokens per image as rough estimates.
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content) // _CHARS_PER_TOKEN
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    total += _TOKENS_PER_IMAGE
                elif part.get("type") == "text":
                    total += len(part.get("text", "")) // _CHARS_PER_TOKEN
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total += len(func.get("arguments", "")) // _CHARS_PER_TOKEN
            total += len(func.get("name", "")) // _CHARS_PER_TOKEN
    return total


def prune_old_images(
    messages: list[dict[str, Any]], max_images: int | None = None, config: ContextConfig | None = None,
) -> int:
    """Remove old images from messages in-place, keeping only the most recent.

    Returns the number of images removed.
    """
    cfg = config or _active_config
    limit = max_images if max_images is not None else cfg.max_images_in_context

    image_positions: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for j, part in enumerate(content):
                if part.get("type") == "image_url":
                    image_positions.append((i, j))

    to_remove = len(image_positions) - limit
    if to_remove <= 0:
        return 0

    for msg_idx, part_idx in image_positions[:to_remove]:
        messages[msg_idx]["content"][part_idx] = {
            "type": "text",
            "text": "[screenshot removed to save context]",
        }

    log.debug("pruned %d old images from context (kept %d)", to_remove, limit)
    return to_remove


def summarize_and_compact(
    messages: list[dict[str, Any]],
    gateway: Any,
    max_context_tokens: int | None = None,
    threshold_pct: float | None = None,
    keep_recent: int | None = None,
    config: ContextConfig | None = None,
) -> bool:
    """If context exceeds threshold, summarize old messages and compact.

    Returns True if compaction occurred.
    """
    cfg = config or _active_config
    _max_tokens = max_context_tokens if max_context_tokens is not None else cfg.max_context_tokens
    _threshold = threshold_pct if threshold_pct is not None else cfg.compact_threshold_pct
    _keep = keep_recent if keep_recent is not None else cfg.keep_recent_messages

    estimated = estimate_token_count(messages)
    threshold_tokens = int(_max_tokens * _threshold)
    if estimated < threshold_tokens:
        return False

    log.info(
        "context at ~%d tokens (threshold %d) — triggering compaction",
        estimated,
        threshold_tokens,
    )

    if len(messages) <= 2 + _keep:
        return False

    to_summarize = messages[2:-_keep]
    if not to_summarize:
        return False

    from daedalus.llm.gateway import LLMCall, LLMRole

    summary_text = _format_messages_for_summary(to_summarize)
    summary_response = gateway.complete(LLMCall(
        role=LLMRole.CHEAP,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation history into a concise structured summary. "
                    "Preserve all key observations, coordinates, element locations, UI state, "
                    "and action results. Be thorough but concise."
                ),
            },
            {"role": "user", "content": summary_text},
        ],
        max_tokens=2000,
    ))

    summary_msg = {
        "role": "user",
        "content": (
            f"[Context compacted — summary of prior {len(to_summarize)} messages]\n\n"
            f"{summary_response.content}"
        ),
    }
    messages[2:-_keep] = [summary_msg]

    # Ensure we don't leave orphaned tool messages at the boundary.
    # If the first kept message is a "tool" role, it needs its preceding
    # "assistant" message (with tool_calls). Walk backwards from the summary
    # insertion point and verify the message sequence is valid.
    insert_end = 3  # index after summary_msg (system, user, summary)
    while insert_end < len(messages) and messages[insert_end].get("role") == "tool":
        # Orphaned tool message — convert to user message
        tool_msg = messages[insert_end]
        tool_content = tool_msg.get("content", "")
        if isinstance(tool_content, list):
            text_parts = [p.get("text", "") for p in tool_content if isinstance(p, dict) and p.get("type") == "text"]
            tool_content = "\n".join(text_parts) or "[tool result]"
        messages[insert_end] = {
            "role": "user",
            "content": f"[Previous tool result]: {str(tool_content)[:500]}",
        }
        insert_end += 1

    new_estimated = estimate_token_count(messages)
    log.info("compaction complete: %d -> %d estimated tokens", estimated, new_estimated)
    return True


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Format a slice of messages into a text representation for summarization."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    text_parts.append("[image]")
            text = "\n".join(text_parts)
        elif content is None:
            text = ""
        else:
            text = str(content)

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_strs = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "")
                if len(args) > 200:
                    args = args[:200] + "..."
                tc_strs.append(f"  {name}({args})")
            text += "\nTool calls:\n" + "\n".join(tc_strs)

        if text.strip():
            parts.append(f"[{role}]: {text.strip()}")

    return "\n\n".join(parts)
