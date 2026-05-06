"""Skill spec model.

A `SkillSpec` is the descriptive contract of a skill: identity, version,
description, side effects, pre/post conditions (textual for now), examples,
test fixtures, and required other skills.

The *inputs/outputs* JSON schema is derived from the skill's Pydantic
`Inputs`/`Outputs` classes at registration time, not duplicated in the YAML.
This keeps a single source of truth for typing and avoids drift.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SkillKindLiteral = Literal["atomic", "daemon"]

# Side effects a skill is allowed to declare. Used by the static safety lint
# in the Implementor (Phase 2) and by the executor for transparency in the
# user-confirmation UI.
KNOWN_SIDE_EFFECTS = frozenset(
    {
        "screen_input",      # mouse / keyboard
        "screen_capture",    # reads pixels
        "filesystem_read",
        "filesystem_write",
        "network",           # outgoing HTTP / RPC
        "llm_call",          # invokes an LLM via the gateway
        "task_state_write",  # writes to the per-task DB (daemons)
        "task_state_read",
        "clock",             # sleeps / waits
    }
)

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class SkillVersion(BaseModel):
    """Tiny semver wrapper. We only use the (major, minor, patch) tuple for
    compatibility checks; pre-release/build metadata is preserved as a string.
    """

    model_config = ConfigDict(frozen=True)

    raw: str

    @field_validator("raw")
    @classmethod
    def _validate(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(f"invalid semver: {v!r}")
        return v

    @property
    def tuple(self) -> tuple[int, int, int]:
        core = self.raw.split("-", 1)[0].split("+", 1)[0]
        a, b, c = (int(x) for x in core.split("."))
        return (a, b, c)

    def is_compatible_with(self, requested: str) -> bool:
        """Caret-range compatibility: ^X.Y.Z accepts >=X.Y.Z, <X+1.0.0.

        Plain version strings require exact major.minor.patch match.
        """
        if requested.startswith("^"):
            req = SkillVersion(raw=requested[1:])
            ra, rb, rc = req.tuple
            ma, mb, mc = self.tuple
            if ma != ra:
                return False
            return (mb, mc) >= (rb, rc)
        return SkillVersion(raw=requested).tuple == self.tuple


class SkillExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, Any]
    expected: dict[str, Any] | None = None
    note: str | None = None


class SkillSpec(BaseModel):
    """Descriptive metadata for a skill. Stored on disk as ``spec.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable, lowercase, snake_case id.")
    version: SkillVersion
    kind: SkillKindLiteral = "atomic"
    description: str = Field(min_length=1, description="One- or two-line summary.")
    side_effects: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    examples: list[SkillExample] = Field(default_factory=list)
    tests: list[str] = Field(
        default_factory=list,
        description="Filenames inside the skill's tests/ folder.",
    )
    requires: list[str] = Field(
        default_factory=list,
        description="Other skill ids this skill depends on.",
    )
    # Optional, for daemon skills: which key in TaskState the loop publishes to.
    publishes_state_key: str | None = None
    # When True, the daemon framework skips auto-writing yielded values to
    # task_state (the skill handles it internally).
    self_publishes: bool = False
    # Free-form notes the librarian indexes for retrieval.
    tags: list[str] = Field(default_factory=list)
    # Input field names whose values should be redacted in traces.
    sensitive_inputs: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not _SKILL_ID_RE.match(v):
            raise ValueError(
                f"invalid skill id {v!r}: must match {_SKILL_ID_RE.pattern}"
            )
        return v

    @field_validator("side_effects")
    @classmethod
    def _validate_side_effects(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if s not in KNOWN_SIDE_EFFECTS]
        if bad:
            raise ValueError(
                f"unknown side_effects {bad}; allowed: {sorted(KNOWN_SIDE_EFFECTS)}"
            )
        return v

    def to_yaml_dict(self) -> dict[str, Any]:
        """Produce a YAML-friendly dict from this spec (for generating spec.yaml)."""
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version.raw,
            "kind": self.kind,
            "description": self.description,
        }
        if self.side_effects:
            d["side_effects"] = list(self.side_effects)
        if self.preconditions:
            d["preconditions"] = list(self.preconditions)
        if self.postconditions:
            d["postconditions"] = list(self.postconditions)
        if self.examples:
            d["examples"] = [ex.model_dump(exclude_none=True) for ex in self.examples]
        if self.tests:
            d["tests"] = list(self.tests)
        if self.requires:
            d["requires"] = list(self.requires)
        if self.publishes_state_key:
            d["publishes_state_key"] = self.publishes_state_key
        if self.self_publishes:
            d["self_publishes"] = True
        if self.tags:
            d["tags"] = list(self.tags)
        if self.sensitive_inputs:
            d["sensitive_inputs"] = list(self.sensitive_inputs)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillSpec:
        # Allow `version: "0.1.0"` as a plain string in YAML.
        if isinstance(data.get("version"), str):
            data = {**data, "version": {"raw": data["version"]}}
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Precondition predicate registry
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from daedalus.core.context import ExecutionContext

_PRECONDITION_PREDICATES: dict[str, Callable[[ExecutionContext], bool]] = {}


def _register_precondition(name: str, fn: Callable[[ExecutionContext], bool]) -> None:
    _PRECONDITION_PREDICATES[name] = fn


def check_preconditions(preconditions: list[str], ctx: ExecutionContext) -> str | None:
    """Check all known preconditions. Returns the first failing predicate name, or None."""
    for p in preconditions:
        fn = _PRECONDITION_PREDICATES.get(p)
        if fn is not None and not fn(ctx):
            return p
    return None


_register_precondition("backend.connected", lambda ctx: ctx.backend.is_connected)
_register_precondition("ctx.llm is configured", lambda ctx: ctx.llm is not None)
_register_precondition("ctx.llm is configured with a vision role", lambda ctx: ctx.llm is not None)
