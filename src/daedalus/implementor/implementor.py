"""LLM-driven implementor: spec -> sandbox -> safety -> tests -> publish."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from daedalus.core.errors import DaedalusError
from daedalus.core.registry import Registry, get_registry
from daedalus.implementor.safety import SafetyViolation, lint_skill_source
from daedalus.library.loader import load_skill
from daedalus.llm.gateway import LLMCall, LLMGateway

log = logging.getLogger(__name__)


class ImplementorError(DaedalusError):
    pass


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


class ImplementorRequest(BaseModel):
    """What we ask the Implementor to build.

    Usually constructed from a Planner ``MissingSkillSpec`` or a Learner
    refactor proposal.
    """

    proposed_id: str = Field(description="Snake_case id for the new skill.")
    description: str
    rationale: str = ""
    inputs_hint: dict[str, Any] = Field(default_factory=dict)
    outputs_hint: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)
    extra_context: str | None = None


@dataclass
class SkillBundle:
    """Files produced by the Implementor in a sandbox folder."""

    skill_id: str
    sandbox_dir: Path
    spec_path: Path
    skill_py_path: Path
    test_paths: list[Path]


@dataclass
class ImplementorResult:
    bundle: SkillBundle | None
    violations: list[SafetyViolation] = field(default_factory=list)
    test_failures: list[str] = field(default_factory=list)
    notes: str = ""
    raw_response: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.bundle is not None
            and not self.violations
            and not self.test_failures
        )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Implementor for Daedalus, a computer-control agent. Given a spec,
produce a complete on-disk skill: a spec.yaml, a skill.py, and at least one
JSON test fixture. The skill plugs into the existing daedalus framework.

OUTPUT FORMAT
-------------
Respond with EXACTLY one JSON object on a single line, no prose, no markdown:

  {
    "skill_id": "<snake_case>",
    "spec_yaml": "<full file content>",
    "skill_py":  "<full file content>",
    "tests": [
      {"name": "basic", "content": { ...JSON fixture... }}
    ],
    "notes": "<optional>"
  }

CONVENTIONS
-----------
- The spec_yaml MUST be valid YAML. Quote any string values that contain
  special YAML characters (: -> { } [ ] , # | > @ ` ! % * & ?). Use
  single or double quotes. The description field should always be quoted.
  IMPORTANT: Tags must be strings. Always quote numeric tag values (e.g.
  write "2048" not 2048) or YAML will parse them as integers.
- The skill_py file MUST:
  * import: from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
  * import: from daedalus.core.spec import SkillVersion (and SkillExample if used)
  * define Pydantic Inputs and Outputs models with model_config=ConfigDict(extra='forbid')
  * declare a class decorated with @register that subclasses AtomicSkill
  * define class attributes SPEC, Inputs, Outputs and a `run(self, inputs, ctx)` method
  * NEVER use subprocess, os.system, eval, exec, or compile.
  * If it touches the network or filesystem, declare the matching side_effect
    in spec.yaml (network / filesystem_read / filesystem_write). The agent
    will reject undeclared side effects.
- The spec.yaml MUST agree with the SPEC inside skill.py on id/version/kind.
- Each test fixture has shape: {"name": "...", "inputs": {...}, "expected_output": {...}, "expected_events": [...]}
  Optional fields: "validation_mode" ("exact"|"any_valid"|"schema_only", default "exact"),
  "valid_outputs" (list of acceptable outputs for "any_valid" mode),
  "ignore_output_keys" (list of keys to skip in comparison).
  Tests run against an in-memory MockBackend. Keep them self-contained.

ALGORITHMIC SOLVING
-------------------
When implementing skills that involve puzzles, games, or deterministic problems:
- Use vision (ctx.llm + screenshots) ONLY to extract the problem state from the screen
- Implement the actual solving logic as pure Python (algorithms, not LLM calls)
- Then act on the solution by clicking/typing
- NEVER rely on the LLM to compute solutions — LLMs are unreliable for
  deterministic computation. Write real solvers.

TEST FIXTURES FOR HEURISTIC/SEARCH ALGORITHMS
----------------------------------------------
When the skill uses heuristic search (expectimax, minimax, A*, etc.) or any
algorithm where the "best" output depends on tuning weights and search depth:
- Do NOT write fixtures that assert a specific "best" output value. The
  algorithm may legitimately return different valid answers.
- Instead, use "validation_mode": "any_valid" in the fixture and provide a
  "valid_outputs" list of all acceptable results, OR use "validation_mode":
  "schema_only" to just verify the output has the correct keys and types.
- You can also use "ignore_output_keys": ["<key>"] to skip checking volatile fields.
- Always include at least one fixture that checks the skill runs without error.

Example fixture for a heuristic solver:
  {"name": "basic", "inputs": {"grid": [[0,0,2,4],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
   "expected_output": {"move": "left"},
   "validation_mode": "any_valid",
   "valid_outputs": [{"move": "left"}, {"move": "down"}, {"move": "right"}, {"move": "up"}]}

For deterministic algorithms (sorting, exact solvers, etc.), use the default
"exact" validation mode as normal.

SKILL NAMING
------------
Skill names must describe the GENERAL capability, not the specific task.
- BAD:  reset_and_retry_puzzle, click_puzzle_edges_from_vision, solve_sudoku_game
- GOOD: retry_with_reset, click_coordinates_from_analysis, extract_and_solve_grid

The name should make sense if this skill were used for a completely different
application. Strip domain-specific words (puzzle, sudoku, game, specific app
names, etc.) and describe the abstract action pattern.

EXAMPLE SKILL FOR REFERENCE
---------------------------
__EXAMPLE__

Now implement the requested skill.
"""


def _example_skill_text(skills_dir: Path) -> str:
    """Inline example skills so the LLM has concrete shapes to mimic."""
    parts: list[str] = []
    for skill_id in ("mouse", "view_screen", "assert_screen_contains"):
        base = skills_dir / skill_id
        if not base.exists():
            continue
        spec = (base / "spec.yaml").read_text()
        py = (base / "skill.py").read_text()
        parts.append(f"### {skill_id}/spec.yaml\n{spec}\n### {skill_id}/skill.py\n{py}\n")
    if not parts:
        return "(no example skills available)"

    # Include key framework type info so the LLM uses the API correctly.
    parts.append("""### Key framework types (from daedalus.backends.protocol)
- Screenshot: has fields .image (PIL.Image.Image), .width (int), .height (int), .captured_at (float)
- ctx.backend.screenshot(region=None) -> Screenshot
- ctx.backend.click(x, y, button=Button.LEFT, double=False) -> None
- ctx.backend.write(text) -> None
- ctx.backend.press(*keys) -> None
- ctx.llm.complete(LLMCall(...)) -> LLMResponse  (LLMResponse has .content: str)
- To convert a Screenshot to base64 PNG: import io, base64; buf = io.BytesIO(); shot.image.save(buf, format='PNG'); b64 = base64.b64encode(buf.getvalue()).decode()

### RunStore (from daedalus.core.store) — per-run data store
Skills can read/write structured data via ctx.store (a RunStore instance):
- ctx.store.create_table(name, schema)  -- e.g. create_table("spots", {"x": "int", "y": "int", "label": "str"})
- ctx.store.append(table, row_dict) -> int  -- append a row, returns row id
- ctx.store.all_rows(table) -> list[dict]   -- get all rows
- ctx.store.query(table, where={"col": val}) -> list[dict]  -- filtered query
- ctx.store.count(table) -> int             -- row count
- ctx.store.table_names() -> list[str]      -- list all tables

### Example: composite skill that reads from store and clicks
```python
class ClickFilteredLocations(AtomicSkill):
    class Inputs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        table: str = Field(description="RunStore table with x,y columns")
        max_clicks: int = Field(default=10)

    class Outputs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        clicked: int

    def run(self, inputs, ctx):
        rows = ctx.store.all_rows(inputs.table)
        clicked = 0
        for row in rows[:inputs.max_clicks]:
            ctx.backend.click(int(row["x"]), int(row["y"]))
            clicked += 1
            import time; time.sleep(0.2)
        return self.Outputs(clicked=clicked)
```
""")
    return "\n".join(parts)


def _user_message(req: ImplementorRequest) -> str:
    body = {
        "proposed_id": req.proposed_id,
        "description": req.description,
        "rationale": req.rationale,
        "inputs_hint": req.inputs_hint,
        "outputs_hint": req.outputs_hint,
        "side_effects": req.side_effects,
        "examples": req.examples,
    }
    parts = ["SPEC TO IMPLEMENT:", json.dumps(body, indent=2)]
    if req.extra_context:
        parts.extend(["", "EXTRA CONTEXT:", req.extra_context.strip()])
    parts.extend(["", "Return the JSON object now."])
    return "\n".join(parts)


def _strip_codefence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


# ---------------------------------------------------------------------------
# Test fixture runner (mirrors cli._run_skill_fixture but works on a fresh registry)
# ---------------------------------------------------------------------------


def _run_fixture_against_mock(skill_cls, fixture: dict, tmp_root: Path) -> tuple[bool, str]:  # type: ignore[no-untyped-def]
    from daedalus.backends.mock import MockBackend
    from daedalus.core.context import ExecutionContext, TaskState
    from daedalus.core.store import RunStore
    from daedalus.tracing.recorder import TraceRecorder

    inputs = fixture["inputs"]
    expected_output = fixture.get("expected_output", {})
    expected_events = fixture.get("expected_events", [])
    ignore_keys = set(fixture.get("ignore_output_keys", []))
    validation_mode = fixture.get("validation_mode", "exact")
    valid_outputs = fixture.get("valid_outputs", [])

    backend = MockBackend()
    backend.connect()
    tracer = TraceRecorder(traces_root=tmp_root, db_path=tmp_root / "tasks.db", task_name="impl")
    state = TaskState(tmp_root / "tasks.db", tracer.task_id)
    store = RunStore(tmp_root / "tasks.db", tracer.task_id)
    ctx = ExecutionContext(
        task_id=tracer.task_id,
        backend=backend,
        task_state=state,
        tracer=tracer,
        store=store,
    )
    try:
        inp = skill_cls.Inputs.model_validate(inputs)
        out = skill_cls().run(inp, ctx)
        out_dict = out.model_dump(mode="json") if hasattr(out, "model_dump") else dict(out)
    except Exception as exc:
        tracer.finish("failed")
        return False, f"raised {type(exc).__name__}: {exc}"
    tracer.finish("success")

    actual = {k: v for k, v in out_dict.items() if k not in ignore_keys}
    expected = {k: v for k, v in expected_output.items() if k not in ignore_keys}

    if validation_mode == "any_valid":
        if valid_outputs and actual not in valid_outputs:
            return False, f"output {actual} not in valid_outputs"
        return True, "ok"
    elif validation_mode == "schema_only":
        for key, val in expected.items():
            if key not in actual:
                return False, f"missing key {key!r} in output"
            if type(actual[key]) is not type(val):
                return False, (
                    f"key {key!r}: expected type {type(val).__name__}, "
                    f"got {type(actual[key]).__name__}"
                )
        return True, "ok"

    if actual != expected:
        return False, f"output {actual} != expected {expected}"
    search_start = 0
    for want in expected_events:
        op = want["op"]
        args = want.get("args", {})
        matched = False
        for j, e in enumerate(backend.events[search_start:], start=search_start):
            if e.op != op:
                continue
            if all(e.args.get(k) == v for k, v in args.items()):
                matched = True
                search_start = j + 1
                break
        if not matched:
            return False, f"missing expected event {want} (searched from index {search_start})"
    return True, "ok"


# ---------------------------------------------------------------------------
# Implementor
# ---------------------------------------------------------------------------


class SyntheticSkillImplementor:
    """Default implementor: LLM -> sandbox -> safety -> tests -> publish."""

    def __init__(
        self,
        gateway: LLMGateway,
        skills_dir: Path,
        *,
        role: str = "implementor",
        sandbox_root: Path | None = None,
        max_repair_attempts: int = 2,
    ) -> None:
        self._gateway = gateway
        self._skills_dir = skills_dir
        self._role = role
        self._sandbox_root = sandbox_root or (skills_dir.parent / ".daedalus" / "implementor_sandbox")
        self._max_repair_attempts = max_repair_attempts

    # ---- Public ----------------------------------------------------------

    def synthesize(self, request: ImplementorRequest) -> ImplementorResult:
        attempts = 0
        feedback: str | None = None
        last_response: str = ""
        while True:
            attempts += 1
            response = self._call_llm(request, repair_feedback=feedback)
            last_response = response
            try:
                payload = self._parse_response(response)
            except ImplementorError as exc:
                if attempts > self._max_repair_attempts:
                    return ImplementorResult(
                        bundle=None, notes=f"unparseable LLM output: {exc}", raw_response=response
                    )
                feedback = f"Your last response was not valid JSON: {exc}"
                continue
            try:
                bundle = self._materialize_sandbox(payload)
            except ImplementorError as exc:
                if attempts > self._max_repair_attempts:
                    return ImplementorResult(
                        bundle=None, notes=f"sandbox materialization failed: {exc}", raw_response=response
                    )
                feedback = f"Error materializing the skill: {exc}"
                continue
            violations = lint_skill_source(
                bundle.skill_py_path.read_text(),
                declared_side_effects=self._declared_side_effects(bundle),
            )
            if violations:
                if attempts > self._max_repair_attempts:
                    return ImplementorResult(
                        bundle=bundle,
                        violations=violations,
                        notes="safety violations after repair attempt",
                        raw_response=last_response,
                    )
                feedback = "Safety lint violations:\n" + "\n".join(
                    f"  - line {v.lineno}: {v.rule}: {v.detail}" for v in violations
                ) + "\nFix and re-emit the JSON object."
                continue

            sandbox_result = self._sandbox_load_and_test(bundle)
            if not sandbox_result["ok"]:
                error_msg = sandbox_result.get("error") or ""
                test_failures = [
                    f"{tr['name']}: {tr['message']}"
                    for tr in sandbox_result.get("test_results", [])
                    if not tr.get("ok", False)
                ]
                if not test_failures and error_msg:
                    if attempts > self._max_repair_attempts:
                        return ImplementorResult(
                            bundle=bundle,
                            notes=f"sandbox failed: {error_msg}",
                            raw_response=last_response,
                        )
                    feedback = f"Loading/testing the skill failed: {error_msg}"
                    continue
                if test_failures:
                    if attempts > self._max_repair_attempts:
                        return ImplementorResult(
                            bundle=bundle,
                            test_failures=test_failures,
                            raw_response=last_response,
                        )
                    feedback = "Some fixtures failed:\n" + "\n".join(test_failures)
                    continue

            return ImplementorResult(
                bundle=bundle,
                notes=payload.get("notes") or "",
                raw_response=last_response,
            )

    def publish(self, bundle: SkillBundle, *, registry: Registry | None = None) -> str:
        """Move a green sandbox bundle into ``skills_dir/<id>/`` and load it
        into the global (or supplied) registry. Returns the skill id."""
        target = self._skills_dir / bundle.skill_id
        if target.exists():
            raise ImplementorError(f"skills/{bundle.skill_id}/ already exists; not overwriting")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle.sandbox_dir, target)
        load_skill(target, registry=registry if registry is not None else get_registry())
        return bundle.skill_id

    def revise(self, skill_id: str, feedback: str) -> ImplementorResult:
        """Re-synthesize an existing temp skill given feedback about what is wrong.

        Reads the current ``skills/_temp/<skill_id>/skill.py`` and
        ``spec.yaml``, embeds them in the prompt as context, and asks the
        implementor to produce a corrected version.  The revised bundle is
        written back to the same temp path, replacing the old files.

        Returns an ``ImplementorResult`` — check ``.ok`` before publishing.
        """
        temp_path = self._skills_dir / "_temp" / skill_id
        if not temp_path.exists():
            return ImplementorResult(
                bundle=None,
                notes=f"no temp skill {skill_id!r} found; implement it first",
            )

        current_skill_py = (temp_path / "skill.py").read_text()
        current_spec_yaml = (temp_path / "spec.yaml").read_text()

        extra_context = (
            "REVISION REQUEST\n"
            "----------------\n"
            f"The skill '{skill_id}' already exists. Below is its current "
            "implementation. Your job is to fix it according to the feedback.\n\n"
            f"FEEDBACK FROM TESTER:\n{feedback}\n\n"
            f"CURRENT spec.yaml:\n```yaml\n{current_spec_yaml}\n```\n\n"
            f"CURRENT skill.py:\n```python\n{current_skill_py}\n```\n\n"
            "Produce a corrected JSON object with the same skill_id. "
            "Keep all parts that are not related to the feedback unchanged."
        )

        request = ImplementorRequest(
            proposed_id=skill_id,
            description=f"Revised version of '{skill_id}' — see feedback in extra_context.",
            rationale="Revision requested by explorer after live testing",
            side_effects=["screen_capture", "screen_input", "llm_call"],
            extra_context=extra_context,
        )
        return self.synthesize(request)

    def publish_temp(self, bundle: SkillBundle, *, registry: Registry | None = None) -> Path:
        """Publish a bundle into ``skills_dir/_temp/<id>/`` and load it into
        the registry. Returns the path to the temp skill directory.

        Unlike ``publish``, this is a staging area — the skill is usable but
        not yet permanently registered. Call ``promote_temp`` to move it to
        the main skills directory.
        """
        temp_dir = self._skills_dir / "_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        target = temp_dir / bundle.skill_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundle.sandbox_dir, target)
        load_skill(target, registry=registry if registry is not None else get_registry())
        return target

    def promote_temp(self, skill_id: str, *, registry: Registry | None = None) -> str:
        """Move a skill from ``skills_dir/_temp/<id>/`` to ``skills_dir/<id>/``.
        Returns the skill id on success."""
        temp_path = self._skills_dir / "_temp" / skill_id
        if not temp_path.exists():
            raise ImplementorError(f"no temp skill {skill_id!r} to promote")
        target = self._skills_dir / skill_id
        if target.exists():
            raise ImplementorError(f"skills/{skill_id}/ already exists; not overwriting")
        shutil.move(str(temp_path), str(target))
        return skill_id

    def cleanup_temp(self, skill_id: str | None = None) -> None:
        """Remove temp skill(s). If skill_id is None, remove the entire _temp dir."""
        temp_dir = self._skills_dir / "_temp"
        if not temp_dir.exists():
            return
        if skill_id:
            target = temp_dir / skill_id
            if target.exists():
                shutil.rmtree(target)
        else:
            shutil.rmtree(temp_dir)

    # ---- Internal --------------------------------------------------------

    def _declared_side_effects(self, bundle: SkillBundle) -> set[str]:
        try:
            data = yaml.safe_load(bundle.spec_path.read_text()) or {}
            return set(data.get("side_effects") or [])
        except Exception:
            return set()

    def _call_llm(self, req: ImplementorRequest, *, repair_feedback: str | None) -> str:
        system = _SYSTEM_PROMPT.replace("__EXAMPLE__", _example_skill_text(self._skills_dir))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _user_message(req)},
        ]
        if repair_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        repair_feedback
                        + "\n\nReturn a corrected JSON object using the same format. Only the JSON, no prose."
                    ),
                }
            )
        try:
            resp = self._gateway.complete(
                LLMCall(
                    role=self._role,
                    messages=messages,
                    response_format="json_object",
                )
            )
        except Exception as exc:
            raise ImplementorError(f"LLM call failed: {exc}") from exc
        return resp.content

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = _strip_codefence(content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            a, b = text.find("{"), text.rfind("}")
            if a == -1 or b <= a:
                raise ImplementorError(f"could not extract JSON: {exc}") from exc
            data = json.loads(text[a : b + 1])
        if not isinstance(data, dict):
            raise ImplementorError("response root must be a JSON object")
        for required in ("skill_id", "spec_yaml", "skill_py"):
            if required not in data or not isinstance(data[required], str):
                raise ImplementorError(f"missing or non-string field {required!r}")
        return data

    def _materialize_sandbox(self, payload: dict[str, Any]) -> SkillBundle:
        sid = payload["skill_id"]
        sandbox = self._sandbox_root / sid
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True, exist_ok=True)
        spec_path = sandbox / "spec.yaml"
        py_path = sandbox / "skill.py"

        # Validate spec_yaml is parseable before writing it.
        spec_text = payload["spec_yaml"]
        try:
            spec_data = yaml.safe_load(spec_text)
        except yaml.YAMLError as exc:
            raise ImplementorError(f"LLM produced invalid spec.yaml: {exc}") from exc

        # Coerce tags to strings — YAML parses bare numbers (e.g. 2048) as int.
        if isinstance(spec_data, dict) and "tags" in spec_data:
            spec_data["tags"] = [str(t) for t in spec_data["tags"]]
            spec_text = yaml.dump(spec_data, default_flow_style=False, sort_keys=False)

        spec_path.write_text(spec_text)
        py_path.write_text(payload["skill_py"])
        tests_dir = sandbox / "tests"
        tests_dir.mkdir(exist_ok=True)
        test_paths: list[Path] = []
        for fix in payload.get("tests") or []:
            if not isinstance(fix, dict):
                continue
            name = str(fix.get("name") or "fixture")
            content = fix.get("content")
            if content is None:
                continue
            p = tests_dir / f"{name}.json"
            p.write_text(json.dumps(content, indent=2))
            test_paths.append(p)
        return SkillBundle(
            skill_id=sid,
            sandbox_dir=sandbox,
            spec_path=spec_path,
            skill_py_path=py_path,
            test_paths=test_paths,
        )

    def _sandbox_load_and_test(self, bundle: SkillBundle) -> dict[str, Any]:
        """Load the skill and run fixtures in an isolated subprocess."""
        import site
        import subprocess as sp

        fixtures = []
        for fp in bundle.test_paths:
            try:
                fx = json.loads(fp.read_text())
                fx["name"] = fp.stem
                fixtures.append(fx)
            except Exception:
                pass

        request = {
            "skill_dir": str(bundle.sandbox_dir),
            "skill_id": bundle.skill_id,
            "fixtures": fixtures,
        }

        src_dir = str(Path(__file__).resolve().parent.parent.parent)
        site_dirs = site.getsitepackages() + [site.getusersitepackages()]
        python_path = os.pathsep.join([src_dir, *site_dirs])

        env: dict[str, str] = {"PYTHONPATH": python_path}
        for var in ("PATH", "SYSTEMROOT", "TEMP", "TMP"):
            if var in os.environ:
                env[var] = os.environ[var]

        try:
            proc = sp.run(
                [sys.executable, "-I", "-B", "-m", "daedalus.implementor.sandbox_runner"],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        except sp.TimeoutExpired:
            return {"ok": False, "error": "subprocess timed out after 30s", "test_results": []}

        if proc.returncode != 0:
            return {
                "ok": False,
                "error": f"subprocess exited {proc.returncode}: {proc.stderr[:500]}",
                "test_results": [],
            }

        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"bad JSON from subprocess: {proc.stdout[:500]}", "test_results": []}

    def _legacy_sandbox_load(self, bundle: SkillBundle) -> dict[str, Any]:
        from daedalus.core.registry import Registry as _Registry
        from daedalus.core.registry import use_registry

        sandbox_registry = _Registry()
        with use_registry(sandbox_registry):
            sid = load_skill(bundle.sandbox_dir, registry=sandbox_registry)
        if sid != bundle.skill_id:
            raise ImplementorError(
                f"loaded id {sid!r} != bundle id {bundle.skill_id!r}"
            )
        return {"cls": sandbox_registry.get(sid).cls, "registry": sandbox_registry}

    def _legacy_run_fixtures(self, skill_cls, bundle: SkillBundle) -> list[str]:  # type: ignore[no-untyped-def]
        failures: list[str] = []
        if not bundle.test_paths:
            return ["no test fixtures provided"]
        tmp_root = bundle.sandbox_dir / "_runs"
        tmp_root.mkdir(exist_ok=True)
        for fixture_path in bundle.test_paths:
            try:
                fixture = json.loads(fixture_path.read_text())
            except json.JSONDecodeError as exc:
                failures.append(f"{fixture_path.name}: not valid JSON: {exc}")
                continue
            ok, msg = _run_fixture_against_mock(skill_cls, fixture, tmp_root)
            if not ok:
                failures.append(f"{fixture_path.name}: {msg}")
        return failures
