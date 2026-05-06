"""Library loader cross-checks spec.yaml against the registered SPEC."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from daedalus.core.registry import Registry
from daedalus.library.loader import LoaderError, load_skill

SKILL_PY = '''
from pydantic import BaseModel, ConfigDict
from daedalus.core import AtomicSkill, SkillSpec, register
from daedalus.core.spec import SkillVersion


class _IO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = 0


@register
class Mini(AtomicSkill):
    SPEC = SkillSpec(
        id="{sid}",
        version=SkillVersion(raw="{ver}"),
        kind="atomic",
        description="x",
    )
    Inputs = _IO
    Outputs = _IO

    def run(self, inputs, ctx):
        return _IO(n=inputs.n)
'''


def _write_skill(root: Path, sid: str, ver: str = "0.1.0", spec_id: str | None = None) -> Path:
    skill_dir = root / sid
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.py").write_text(textwrap.dedent(SKILL_PY.format(sid=sid, ver=ver)))
    (skill_dir / "spec.yaml").write_text(
        textwrap.dedent(
            f"""
            id: {spec_id or sid}
            version: {ver}
            kind: atomic
            description: x
            """
        )
    )
    return skill_dir


def test_loader_registers_skill(tmp_path: Path):
    reg = Registry()
    sd = _write_skill(tmp_path, "mini_a")
    sid = load_skill(sd, registry=reg)
    assert sid == "mini_a"
    assert "mini_a" in reg


def test_loader_detects_id_drift(tmp_path: Path):
    reg = Registry()
    sd = _write_skill(tmp_path, "mini_b", spec_id="something_else")
    with pytest.raises(LoaderError):
        load_skill(sd, registry=reg)


def test_loader_missing_spec_yaml_raises(tmp_path: Path):
    reg = Registry()
    skill_dir = tmp_path / "mini_c"
    skill_dir.mkdir()
    (skill_dir / "skill.py").write_text(textwrap.dedent(SKILL_PY.format(sid="mini_c", ver="0.1.0")))
    with pytest.raises(LoaderError):
        load_skill(skill_dir, registry=reg)
