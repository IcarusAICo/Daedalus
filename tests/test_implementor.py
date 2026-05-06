"""Implementor: end-to-end synthesis with a scripted FakeGateway."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from daedalus.implementor import (
    ImplementorRequest,
    SyntheticSkillImplementor,
    lint_skill_source,
)
from daedalus.llm.fakes import FakeGateway

# ---------------------------------------------------------------------------
# Safety lint
# ---------------------------------------------------------------------------


def test_safety_blocks_subprocess():
    src = "import subprocess\nsubprocess.run(['ls'])"
    v = lint_skill_source(src, declared_side_effects=set())
    assert any(viol.rule.startswith("hard_banned") for viol in v)


def test_safety_blocks_os_system():
    src = "import os\nos.system('rm -rf /')"
    v = lint_skill_source(src, declared_side_effects=set())
    assert any(viol.rule == "hard_banned_func" for viol in v)


def test_safety_blocks_network_without_declaration():
    src = "import requests\nrequests.get('http://x')"
    v = lint_skill_source(src, declared_side_effects=set())
    assert any(viol.rule == "undeclared_network" for viol in v)


def test_safety_allows_network_when_declared():
    src = "import requests\nrequests.get('http://x')"
    v = lint_skill_source(src, declared_side_effects={"network"})
    assert v == []


def test_safety_blocks_eval_exec():
    src = "eval('1+1')"
    v = lint_skill_source(src, declared_side_effects=set())
    assert any(viol.rule == "hard_banned_builtin" for viol in v)


def test_safety_passes_clean_skill():
    src = textwrap.dedent(
        """
        from pydantic import BaseModel
        class IO(BaseModel):
            n: int = 0
        def run(i, ctx):
            return IO(n=i.n + 1)
        """
    ).strip()
    assert lint_skill_source(src, declared_side_effects=set()) == []


# ---------------------------------------------------------------------------
# Implementor end-to-end
# ---------------------------------------------------------------------------


_VALID_SPEC_YAML = textwrap.dedent(
    """
    id: noop
    version: 0.1.0
    kind: atomic
    description: Returns the same number it was given.
    side_effects: []
    preconditions: []
    postconditions: []
    examples: []
    tests:
      - basic.json
    requires: []
    tags: [util]
    """
).strip()

_VALID_SKILL_PY = textwrap.dedent(
    """
    from __future__ import annotations
    from pydantic import BaseModel, ConfigDict, Field
    from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
    from daedalus.core.spec import SkillVersion


    class NoopInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        n: int = Field(default=0)


    class NoopOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        n: int


    @register
    class Noop(AtomicSkill):
        SPEC = SkillSpec(
            id="noop",
            version=SkillVersion(raw="0.1.0"),
            kind="atomic",
            description="Returns the same number it was given.",
            side_effects=[],
        )
        Inputs = NoopInput
        Outputs = NoopOutput

        def run(self, inputs, ctx):
            return NoopOutput(n=inputs.n)
    """
).strip()

_VALID_FIXTURE = {
    "name": "basic",
    "inputs": {"n": 7},
    "expected_output": {"n": 7},
    "expected_events": [],
}


def _good_payload() -> str:
    return json.dumps(
        {
            "skill_id": "noop",
            "spec_yaml": _VALID_SPEC_YAML,
            "skill_py": _VALID_SKILL_PY,
            "tests": [{"name": "basic", "content": _VALID_FIXTURE}],
            "notes": "ok",
        }
    )


def test_implementor_synthesizes_and_publishes_a_skill(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    # Provide an example skill on disk so the prompt can include it.
    src_root = Path(__file__).resolve().parent.parent / "skills" / "click_mouse"
    (skills_dir / "click_mouse").mkdir(parents=True)
    (skills_dir / "click_mouse" / "spec.yaml").write_text((src_root / "spec.yaml").read_text())
    (skills_dir / "click_mouse" / "skill.py").write_text((src_root / "skill.py").read_text())

    gw = FakeGateway(responses=[_good_payload()])
    impl = SyntheticSkillImplementor(
        gateway=gw,
        skills_dir=skills_dir,
        sandbox_root=tmp_path / "sandbox",
    )
    req = ImplementorRequest(
        proposed_id="noop",
        description="Returns the same number it was given.",
        rationale="needed for tests",
        inputs_hint={"n": "int"},
        outputs_hint={"n": "int"},
    )
    result = impl.synthesize(req)
    assert result.ok, f"synthesis failed: violations={result.violations} failures={result.test_failures}"
    assert result.bundle is not None
    assert result.bundle.skill_id == "noop"
    assert (result.bundle.sandbox_dir / "tests" / "basic.json").exists()


def test_implementor_rejects_unsafe_code_after_repair(tmp_path: Path):
    bad_py = _VALID_SKILL_PY + "\nimport subprocess\n"
    bad_payload = json.dumps(
        {
            "skill_id": "noop",
            "spec_yaml": _VALID_SPEC_YAML,
            "skill_py": bad_py,
            "tests": [{"name": "basic", "content": _VALID_FIXTURE}],
        }
    )
    gw = FakeGateway(responses=[bad_payload, bad_payload, bad_payload])  # repair also bad
    impl = SyntheticSkillImplementor(
        gateway=gw,
        skills_dir=tmp_path / "skills",
        sandbox_root=tmp_path / "sandbox",
    )
    (tmp_path / "skills").mkdir()
    result = impl.synthesize(
        ImplementorRequest(proposed_id="noop", description="x")
    )
    assert not result.ok
    assert result.violations  # surfaced after second attempt


def test_implementor_repairs_on_first_failure(tmp_path: Path):
    bad_py = _VALID_SKILL_PY + "\nimport subprocess\n"
    bad_payload = json.dumps(
        {
            "skill_id": "noop",
            "spec_yaml": _VALID_SPEC_YAML,
            "skill_py": bad_py,
            "tests": [{"name": "basic", "content": _VALID_FIXTURE}],
        }
    )
    gw = FakeGateway(responses=[bad_payload, _good_payload()])
    impl = SyntheticSkillImplementor(
        gateway=gw,
        skills_dir=tmp_path / "skills",
        sandbox_root=tmp_path / "sandbox",
    )
    (tmp_path / "skills").mkdir()
    result = impl.synthesize(ImplementorRequest(proposed_id="noop", description="x"))
    assert result.ok, f"violations={result.violations} failures={result.test_failures}"
    assert len(gw.calls) == 2  # one bad, one good


def test_publish_writes_into_skills_dir(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    gw = FakeGateway(responses=[_good_payload()])
    impl = SyntheticSkillImplementor(
        gateway=gw, skills_dir=skills_dir, sandbox_root=tmp_path / "sandbox"
    )
    result = impl.synthesize(ImplementorRequest(proposed_id="noop", description="x"))
    assert result.ok
    sid = impl.publish(result.bundle)
    assert sid == "noop"
    assert (skills_dir / "noop" / "spec.yaml").exists()
    assert (skills_dir / "noop" / "skill.py").exists()


def test_implementor_fails_when_fixtures_dont_pass(tmp_path: Path):
    bad_fixture = {
        "name": "basic",
        "inputs": {"n": 7},
        "expected_output": {"n": 999},  # wrong
        "expected_events": [],
    }
    payload = json.dumps(
        {
            "skill_id": "noop",
            "spec_yaml": _VALID_SPEC_YAML,
            "skill_py": _VALID_SKILL_PY,
            "tests": [{"name": "basic", "content": bad_fixture}],
        }
    )
    gw = FakeGateway(responses=[payload, payload, payload])
    impl = SyntheticSkillImplementor(
        gateway=gw, skills_dir=tmp_path / "skills", sandbox_root=tmp_path / "sandbox"
    )
    (tmp_path / "skills").mkdir()
    result = impl.synthesize(ImplementorRequest(proposed_id="noop", description="x"))
    assert not result.ok
    assert result.test_failures
