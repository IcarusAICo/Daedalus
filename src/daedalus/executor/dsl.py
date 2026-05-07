"""Program DSL.

Programs are data, not code. A program is a small YAML/JSON document that the
Planner emits and the Executor runs. Phase 0 supports sequential steps only;
Phase 2 will add ``parallel``, ``daemon``, ``until``, and ``if``.

The DSL is deliberately tiny so that:
    - The Planner cannot smuggle arbitrary Python in.
    - The user-facing confirm UI can render the whole plan compactly.
    - Future structural validations (cost estimates, side-effect summaries) are
      a single AST walk away.

Variable references
-------------------
Step inputs can reference outputs from earlier steps via ``$ref:`` syntax.
When a step uses ``save_as: "loc"``, later steps can reference its outputs:

    - ``"$ref:loc.x"``  → the ``x`` field from the step saved as ``loc``
    - ``"$ref:loc"``    → the entire output dict

References are resolved at run time by the executor, not at parse time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from daedalus.core.errors import ProgramValidationError, SkillNotFoundError, SkillValidationError
from daedalus.core.registry import Registry, get_registry

DSL_VERSION = 1
DSL_VERSION_PYTHON = 2

_REF_PATTERN = re.compile(r"^\$ref:([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z0-9_][a-zA-Z0-9_.]*))?\s*$")


class ProgramStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(description="Registered skill id.")
    version: str | None = Field(
        default=None,
        description="Optional caret-range version constraint, e.g. '^0.1.0'.",
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    save_as: str | None = Field(
        default=None,
        description="If set, store this step's output under task_state[save_as].",
    )
    max_duration_ms: int | None = Field(
        default=None, ge=100, description="Per-step timeout override in ms."
    )


class ProgramDaemon(BaseModel):
    """A daemon-kind skill that runs for the duration of the whole program."""

    model_config = ConfigDict(extra="forbid")

    skill: str
    version: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class Program(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dsl_version: int = Field(default=DSL_VERSION, ge=1, le=DSL_VERSION)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[ProgramStep] = Field(min_length=1, max_length=1000)
    daemons: list[ProgramDaemon] = Field(default_factory=list, max_length=16)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def referenced_skill_ids(self) -> list[str]:
        # preserve order, deduplicate
        seen: dict[str, None] = {}
        for d in self.daemons:
            seen.setdefault(d.skill, None)
        for s in self.steps:
            seen.setdefault(s.skill, None)
        return list(seen.keys())


class PythonProgram(BaseModel):
    """A v2 program where the plan is expressed as Python code calling skills."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dsl_version: int = Field(default=DSL_VERSION_PYTHON, ge=2, le=DSL_VERSION_PYTHON)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    code: str = Field(min_length=1, description="Python function body invoking skills via ctx.*")
    daemons: list[ProgramDaemon] = Field(default_factory=list, max_length=16)

    @property
    def step_count(self) -> int:
        return self.code.count("ctx.")

    def referenced_skill_ids(self) -> list[str]:
        import re as _re

        ids = _re.findall(r"ctx\.([a-z][a-z0-9_]*)\s*\(", self.code)
        seen: dict[str, None] = {}
        for d in self.daemons:
            seen.setdefault(d.skill, None)
        for sid in ids:
            seen.setdefault(sid, None)
        return list(seen.keys())


AnyProgram = Program | PythonProgram


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------


def parse_program(data: dict[str, Any] | str) -> Program:
    """Parse a Program from a dict or a YAML/JSON string."""
    if isinstance(data, str):
        loaded = yaml.safe_load(data)
        if not isinstance(loaded, dict):
            raise ProgramValidationError("program YAML must be a mapping")
        data = loaded
    try:
        return Program.model_validate(data)
    except Exception as exc:
        raise ProgramValidationError(str(exc)) from exc


def parse_python_program(data: dict[str, Any] | str) -> PythonProgram:
    """Parse a PythonProgram (v2) from a dict or a YAML/JSON string."""
    if isinstance(data, str):
        loaded = yaml.safe_load(data)
        if not isinstance(loaded, dict):
            raise ProgramValidationError("program YAML must be a mapping")
        data = loaded
    try:
        return PythonProgram.model_validate(data)
    except Exception as exc:
        raise ProgramValidationError(str(exc)) from exc


def parse_any_program(data: dict[str, Any] | str) -> AnyProgram:
    """Parse either a v1 Program or v2 PythonProgram based on dsl_version."""
    if isinstance(data, str):
        loaded = yaml.safe_load(data)
        if not isinstance(loaded, dict):
            raise ProgramValidationError("program YAML must be a mapping")
        data = loaded
    version = data.get("dsl_version", 1)
    if version >= 2:
        return parse_python_program(data)
    return parse_program(data)


def load_program(path: Path) -> AnyProgram:
    text = path.read_text()
    prog = parse_any_program(text)
    return prog


def validate_program_against_registry(
    program: AnyProgram, registry: Registry | None = None
) -> None:
    """Ensure every step references a known skill and inputs satisfy its schema.

    This is the second validation pass (the first is structural via Pydantic).
    Catches:
        - typos in skill id
        - version constraints no installed skill satisfies
        - inputs that fail the skill's Pydantic ``Inputs`` model

    For PythonProgram (v2), validation is skipped since skills are called
    dynamically from Python code.
    """
    if isinstance(program, PythonProgram):
        return

    registry = registry if registry is not None else get_registry()
    errors: list[str] = []

    def _check(prefix: str, skill_id: str, version: str | None, inputs: dict, expected_kind: str) -> None:
        try:
            entry = registry.get(skill_id, version_constraint=version)
        except SkillNotFoundError:
            errors.append(f"{prefix}: unknown skill id {skill_id!r}")
            return
        except SkillValidationError as exc:
            errors.append(f"{prefix}: {exc}")
            return
        actual_kind = entry.cls.SPEC.kind
        if actual_kind != expected_kind and not (expected_kind == "atomic" and actual_kind == "service"):
            errors.append(
                f"{prefix}: skill {skill_id!r} is kind={actual_kind!r}, "
                f"expected {expected_kind!r}"
            )
            return
        if _has_dynamic_refs(inputs):
            return
        try:
            entry.cls.Inputs.model_validate(inputs)
        except Exception as exc:
            errors.append(f"{prefix} ({skill_id}): invalid inputs: {exc}")

    for i, step in enumerate(program.steps):
        _check(f"step {i}", step.skill, step.version, step.inputs, expected_kind="atomic")
    for i, daemon in enumerate(program.daemons):
        _check(f"daemon {i}", daemon.skill, daemon.version, daemon.inputs, expected_kind="daemon")

    if errors:
        raise ProgramValidationError("; ".join(errors))


# ---------------------------------------------------------------------------
# Variable reference resolution
# ---------------------------------------------------------------------------


def is_ref(value: Any) -> bool:
    """Return True if *value* is a ``$ref:...`` string."""
    return isinstance(value, str) and value.startswith("$ref:")


def is_store_ref(value: Any) -> bool:
    """Return True if *value* is a ``$store:...`` string."""
    return isinstance(value, str) and value.startswith("$store:")


def _has_dynamic_refs(inputs: dict[str, Any]) -> bool:
    """Return True if any value in *inputs* (recursively) is a ``$ref:`` or ``$store:``."""
    for v in inputs.values():
        if is_ref(v) or is_store_ref(v):
            return True
        if isinstance(v, dict) and _has_dynamic_refs(v):
            return True
        if isinstance(v, list) and any(is_ref(item) or is_store_ref(item) for item in v):
            return True
    return False


def resolve_ref(ref_str: str, saved_outputs: dict[str, dict[str, Any]]) -> Any:
    """Resolve a ``$ref:key`` or ``$ref:key.field.subfield`` against saved outputs.

    Raises ``ProgramValidationError`` if the reference cannot be resolved.
    """
    m = _REF_PATTERN.match(ref_str)
    if not m:
        raise ProgramValidationError(f"malformed reference: {ref_str!r}")
    key = m.group(1)
    path = m.group(2)

    if key not in saved_outputs:
        raise ProgramValidationError(
            f"reference {ref_str!r}: no step output saved as {key!r}. "
            f"Available: {sorted(saved_outputs.keys())}"
        )
    value = saved_outputs[key]

    if path is None:
        return value

    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdigit():
            idx = int(part)
            if idx < len(value):
                value = value[idx]
            else:
                raise ProgramValidationError(
                    f"reference {ref_str!r}: index {idx} out of range (length {len(value)})"
                )
        else:
            raise ProgramValidationError(
                f"reference {ref_str!r}: cannot resolve '.{part}' in {type(value).__name__}"
            )
    return value


_STORE_PATTERN = re.compile(r"^\$store:([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z0-9_]+))?\s*$")


def resolve_store_ref(ref_str: str, store: Any) -> Any:
    """Resolve ``$store:table`` or ``$store:table.count`` against a RunStore.

    Raises ``ProgramValidationError`` if the store is unavailable or the table
    doesn't exist.
    """
    if store is None:
        raise ProgramValidationError(f"$store: reference {ref_str!r} but no RunStore available")
    m = _STORE_PATTERN.match(ref_str)
    if not m:
        raise ProgramValidationError(f"malformed store reference: {ref_str!r}")
    table = m.group(1)
    prop = m.group(2)

    if prop == "count":
        return store.count(table)
    elif prop is None:
        return store.all_rows(table)
    else:
        raise ProgramValidationError(
            f"store reference {ref_str!r}: unknown property '.{prop}' (only '.count' is supported)"
        )


def resolve_inputs(
    inputs: dict[str, Any],
    saved_outputs: dict[str, dict[str, Any]],
    store: Any = None,
) -> dict[str, Any]:
    """Deep-resolve all ``$ref:`` and ``$store:`` strings in an inputs dict."""
    resolved: dict[str, Any] = {}
    for k, v in inputs.items():
        resolved[k] = _resolve_value(v, saved_outputs, store)
    return resolved


def _resolve_value(v: Any, saved_outputs: dict[str, dict[str, Any]], store: Any = None) -> Any:
    if is_ref(v):
        return resolve_ref(v, saved_outputs)
    if is_store_ref(v):
        return resolve_store_ref(v, store)
    if isinstance(v, dict):
        return {dk: _resolve_value(dv, saved_outputs, store) for dk, dv in v.items()}
    if isinstance(v, list):
        return [_resolve_value(item, saved_outputs, store) for item in v]
    return v


# ---------------------------------------------------------------------------
# Summary helpers (for the user-confirmation UI)
# ---------------------------------------------------------------------------


class ProgramSummary(BaseModel):
    name: str
    description: str | None
    step_count: int
    skills: list[str]
    side_effects: list[str]
    daemon_steps: int


def summarize(program: Program, registry: Registry | None = None) -> ProgramSummary:
    registry = registry if registry is not None else get_registry()
    skills_used: list[str] = []
    side_effects: set[str] = set()
    daemons = len(program.daemons)
    seen: set[str] = set()
    for ref in (*program.steps, *program.daemons):
        sid = ref.skill
        try:
            entry = registry.get(sid)
        except SkillNotFoundError:
            if sid not in seen:
                skills_used.append(f"{sid} [MISSING]")
                seen.add(sid)
            continue
        if sid in seen:
            continue
        seen.add(sid)
        skills_used.append(f"{entry.id}@{entry.version.raw}")
        side_effects.update(entry.cls.SPEC.side_effects)
    return ProgramSummary(
        name=program.name,
        description=program.description,
        step_count=program.step_count,
        skills=skills_used,
        side_effects=sorted(side_effects),
        daemon_steps=daemons,
    )


def summarize_any(program: AnyProgram, registry: Registry | None = None) -> ProgramSummary:
    """Summarize either a v1 Program or v2 PythonProgram."""
    if isinstance(program, Program):
        return summarize(program, registry)
    registry = registry if registry is not None else get_registry()
    skill_ids = program.referenced_skill_ids()
    skills_used: list[str] = []
    side_effects: set[str] = set()
    for sid in skill_ids:
        try:
            entry = registry.get(sid)
            skills_used.append(f"{entry.id}@{entry.version.raw}")
            side_effects.update(entry.cls.SPEC.side_effects)
        except SkillNotFoundError:
            skills_used.append(f"{sid} [MISSING]")
    return ProgramSummary(
        name=program.name,
        description=program.description,
        step_count=program.step_count,
        skills=skills_used,
        side_effects=sorted(side_effects),
        daemon_steps=len(program.daemons),
    )


# Phase 1 helper. Detects skills referenced by name that aren't registered,
# letting the planner show a "missing skills" panel.
ProgramReferenceStatus = Literal["registered", "missing", "incompatible_version"]


def reference_statuses(
    program: Program, registry: Registry | None = None
) -> list[tuple[str, ProgramReferenceStatus]]:
    registry = registry if registry is not None else get_registry()
    out: list[tuple[str, ProgramReferenceStatus]] = []
    for sid in program.referenced_skill_ids():
        if sid not in registry:
            out.append((sid, "missing"))
            continue
        # If any step pinning this id has an unsatisfied version, mark it.
        bad_pin = False
        for step in program.steps:
            if step.skill != sid or step.version is None:
                continue
            try:
                registry.get(sid, version_constraint=step.version)
            except SkillValidationError:
                bad_pin = True
                break
        out.append((sid, "incompatible_version" if bad_pin else "registered"))
    return out
