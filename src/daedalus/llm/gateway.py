"""Thin LLM gateway built on top of LiteLLM.

The gateway is the only file in the codebase that imports ``litellm``; every
other component talks to :class:`LLMGateway` instances. This makes it cheap
to:

- swap providers,
- mock in tests (the test gateway returns canned responses),
- enforce a security boundary (we re-check ``litellm.__version__`` at startup
  and refuse to run with a compromised release).

Roles
-----
Each callsite identifies itself by a role string (``"planner"``, ``"vision"``,
``"cheap"``, ...). The gateway maps roles to concrete LiteLLM model strings
via configuration, so the role-to-model wiring is data, not code.
"""

from __future__ import annotations

import enum
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daedalus.tracing.recorder import TraceRecorder

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

# Hard block list: never run with these.
BANNED_LITELLM_VERSIONS = frozenset({"1.82.7", "1.82.8"})


class UnknownRoleError(KeyError):
    """The configured role-to-model mapping has no entry for the requested role."""


class LLMRole(enum.StrEnum):
    PLANNER = "planner"
    IMPLEMENTOR = "implementor"
    TEACHER = "teacher"
    VISION = "vision"
    CHEAP = "cheap"
    EXPLORER = "explorer"


class LLMConfig(BaseModel):
    """Role -> model mapping plus provider-level options."""

    model_config = ConfigDict(extra="forbid")

    roles: dict[str, str] = Field(default_factory=dict)
    aws_region: str | None = None
    request_timeout_s: float = 120.0
    max_retries: int = 2
    creative_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    analytical_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    _CREATIVE_ROLES: frozenset[str] = frozenset({
        LLMRole.EXPLORER, LLMRole.PLANNER, LLMRole.IMPLEMENTOR, LLMRole.TEACHER,
        "learner",
    })

    def model_for(self, role: str) -> str:
        if role not in self.roles:
            if role == LLMRole.EXPLORER and LLMRole.PLANNER in self.roles:
                return self.roles[LLMRole.PLANNER]
            raise UnknownRoleError(f"no model configured for role {role!r}")
        return self.roles[role]

    def temperature_for(self, role: str) -> float:
        """Return the configured temperature for a given role.

        Creative roles (explorer, planner, implementor, teacher) use
        creative_temperature. Analytical roles (vision, cheap) use
        analytical_temperature.
        """
        if role in self._CREATIVE_ROLES:
            return self.creative_temperature
        return self.analytical_temperature


@dataclass
class LLMCall:
    """A single completion request. We accept a list of messages and optional images."""

    role: str
    messages: list[dict[str, Any]]
    response_format: str = "text"  # "text" | "json_object"
    temperature: float | None = None  # None means "use config default for role"
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A single tool call parsed from an LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    role: str
    model: str
    content: str
    raw: dict[str, Any]
    duration_s: float
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMGateway:
    """Concrete gateway. In Phase 0 callers are optional; the executor accepts
    ``llm=None``. The Planner / Implementor / Learner in Phase 1+ require it.
    """

    def __init__(self, config: LLMConfig, tracer: TraceRecorder | None = None) -> None:
        self._config = config
        self._tracer = tracer
        self._verify_litellm_version()

    @staticmethod
    def _verify_litellm_version() -> None:
        try:
            import importlib.metadata as md

            ver = md.version("litellm")
        except Exception:
            log.warning("litellm not installed; gateway will fail on first call")
            return
        if ver in BANNED_LITELLM_VERSIONS:
            raise RuntimeError(
                f"litellm=={ver} is in the BANNED list (March 2026 supply-chain attack). "
                "Refusing to start. Run scripts/verify_litellm.sh and rotate credentials "
                "if this version was ever installed."
            )

    def complete(self, call: LLMCall, *, stream_callback: Any | None = None) -> LLMResponse:
        """Execute a completion. If stream_callback is provided, streams content tokens."""
        import litellm  # local import: keeps daedalus importable without litellm in dev

        # Apply provider env defaults at call time so config swaps work.
        if self._config.aws_region:
            os.environ.setdefault("AWS_REGION", self._config.aws_region)
            os.environ.setdefault("AWS_DEFAULT_REGION", self._config.aws_region)

        model = self._config.model_for(call.role)
        temperature = call.temperature if call.temperature is not None else self._config.temperature_for(call.role)
        timeout_s = self._config.request_timeout_s
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": call.messages,
            "temperature": temperature,
            "timeout": timeout_s,
            "num_retries": self._config.max_retries,
        }
        if call.max_tokens:
            kwargs["max_tokens"] = call.max_tokens
        if call.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if call.tools:
            kwargs["tools"] = call.tools
        kwargs.update(call.extra)

        _inject_cache_control(kwargs["messages"])

        t0 = time.perf_counter()

        if stream_callback:
            # Streaming mode — streams content tokens, accumulates tool calls
            kwargs["stream"] = True
            try:
                stream = litellm.completion(**kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"LLM call ({call.role} -> {model}) failed "
                    f"(timeout={timeout_s}s): {exc}"
                ) from exc

            import json as _json
            content_parts: list[str] = []
            # Tool call accumulator: {index: {id, name, arguments_str}}
            tc_acc: dict[int, dict[str, str]] = {}
            usage: dict[str, int] = {}

            for chunk in stream:
                if not chunk.choices:
                    # Final chunk may contain usage
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                        if hasattr(usage_obj, "model_dump"):
                            usage = usage_obj.model_dump()
                        elif isinstance(usage_obj, dict):
                            usage = dict(usage_obj)
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    stream_callback(delta.content)
                # Accumulate streaming tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if hasattr(tc_delta, "index") else 0
                        if idx not in tc_acc:
                            tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_acc[idx]["id"] = tc_delta.id
                        if hasattr(tc_delta, "function") and tc_delta.function:
                            if tc_delta.function.name:
                                tc_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_acc[idx]["arguments"] += tc_delta.function.arguments
                # Check for usage in each chunk (some providers send it in the final chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage
                    if hasattr(usage_obj, "model_dump"):
                        usage = usage_obj.model_dump()
                    elif isinstance(usage_obj, dict):
                        usage = dict(usage_obj)

            dur = time.perf_counter() - t0
            content = "".join(content_parts)

            parsed_tool_calls: list[ToolCall] = []
            for _idx in sorted(tc_acc.keys()):
                tc_data = tc_acc[_idx]
                try:
                    args = _json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except (ValueError, _json.JSONDecodeError):
                    args = {}
                parsed_tool_calls.append(
                    ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args)
                )
        else:
            # Non-streaming mode (default, required for tool calls)
            try:
                raw = litellm.completion(**kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"LLM call ({call.role} -> {model}) failed "
                    f"(timeout={timeout_s}s): {exc}"
                ) from exc
            dur = time.perf_counter() - t0

            try:
                content = raw.choices[0].message.content or ""
            except Exception:
                content = ""

            parsed_tool_calls = []
            try:
                raw_tcs = raw.choices[0].message.tool_calls
                if raw_tcs:
                    import json as _json

                    for tc in raw_tcs:
                        args = tc.function.arguments
                        if isinstance(args, str):
                            args = _json.loads(args)
                        parsed_tool_calls.append(
                            ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                        )
            except (AttributeError, IndexError, TypeError):
                pass

            usage_obj = getattr(raw, "usage", None) or {}
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage = dict(usage_obj)
            else:
                usage = {}

        resp = LLMResponse(
            role=call.role,
            model=model,
            content=content,
            raw={},
            duration_s=dur,
            usage=usage,
            tool_calls=parsed_tool_calls,
        )

        if self._tracer is not None:
            self._tracer.emit(
                "llm_call",
                {
                    "role": call.role,
                    "model": model,
                    "temperature": temperature,
                    "messages": _summarize_messages(call.messages),
                    "tools": [t.get("function", {}).get("name", "?") for t in (call.tools or [])],
                    "response_content": content,
                    "response_tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in parsed_tool_calls
                    ],
                    "duration_s": round(dur, 3),
                    "usage": usage,
                },
            )

        return resp

    def set_tracer(self, tracer: TraceRecorder | None) -> None:
        self._tracer = tracer


def _inject_cache_control(messages: list[dict[str, Any]]) -> None:
    """Add cache_control markers to the system message for Bedrock prompt caching.

    LiteLLM auto-translates these to Bedrock's native cachePoint format.
    """
    if not messages or messages[0].get("role") != "system":
        return

    sys_msg = messages[0]
    content = sys_msg.get("content")

    if isinstance(content, str):
        sys_msg["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
        ]
    elif isinstance(content, list) and content:
        last_block = content[-1]
        if isinstance(last_block, dict) and "cache_control" not in last_block:
            last_block["cache_control"] = {"type": "ephemeral"}


def _summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a trace-friendly summary of messages, replacing large base64 images
    with metadata placeholders."""
    summarized = []
    for msg in messages:
        content = msg.get("content")
        if content is None or isinstance(content, str):
            summarized.append(msg)
            continue

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    if isinstance(url, dict):
                        url_str = url.get("url", "")
                    else:
                        url_str = str(url)
                    if url_str.startswith("data:"):
                        # Replace base64 content with size metadata
                        b64_start = url_str.find(",")
                        b64_len = len(url_str) - b64_start - 1 if b64_start > 0 else 0
                        mime = url_str[:b64_start] if b64_start > 0 else "unknown"
                        parts.append({
                            "type": "image_url",
                            "image_url": f"[base64 image: {mime}, ~{b64_len // 1024}KB encoded]",
                        })
                    else:
                        parts.append(part)
                else:
                    parts.append(part)
            summarized.append({**msg, "content": parts})
        else:
            summarized.append(msg)
    return summarized


def make_gateway(config: LLMConfig | None) -> LLMGateway | None:
    """Construct a gateway from optional config. Returns ``None`` if no config."""
    if config is None or not config.roles:
        return None
    return LLMGateway(config)
