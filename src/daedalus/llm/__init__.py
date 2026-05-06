"""LLM gateway. Single entry point for every LLM call in the agent.

All LLM traffic flows through :class:`LLMGateway`, so we can:

- swap providers (LiteLLM today, possibly direct SDKs later) without touching
  callers,
- enforce per-role model selection from config,
- log cost / token usage into the trace,
- redact secrets and screenshots before persisting prompts.

CRITICAL SECURITY NOTE: LiteLLM versions 1.82.7 and 1.82.8 are compromised
(March 2026 supply-chain attack). pyproject.toml pins to >= 1.83.0. Run
``scripts/verify_litellm.sh`` to confirm.
"""

from daedalus.llm.gateway import (
    LLMCall,
    LLMConfig,
    LLMGateway,
    LLMResponse,
    LLMRole,
    UnknownRoleError,
    make_gateway,
)

__all__ = [
    "LLMCall",
    "LLMConfig",
    "LLMGateway",
    "LLMResponse",
    "LLMRole",
    "UnknownRoleError",
    "make_gateway",
]
