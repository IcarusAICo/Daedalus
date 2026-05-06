"""Registry behaviour: lookup, version constraints, duplicate guard."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from daedalus.core import AtomicSkill, SkillSpec, SkillValidationError
from daedalus.core.registry import Registry
from daedalus.core.spec import SkillVersion


class _IO(BaseModel):
    n: int = 0


def _spec(sid: str, ver: str = "0.1.0") -> SkillSpec:
    return SkillSpec(
        id=sid,
        version=SkillVersion(raw=ver),
        kind="atomic",
        description="t",
    )


def _make_skill(sid: str, ver: str = "0.1.0"):
    cls = type(
        f"S_{sid}_{ver.replace('.', '_')}",
        (AtomicSkill,),
        {
            "SPEC": _spec(sid, ver),
            "Inputs": _IO,
            "Outputs": _IO,
            "run": lambda self, i, ctx: _IO(n=i.n),
        },
    )
    return cls


def test_register_and_get():
    reg = Registry()
    cls = _make_skill("foo")
    reg.register(cls)
    entry = reg.get("foo")
    assert entry.cls is cls
    assert entry.version.raw == "0.1.0"


def test_duplicate_id_with_different_class_rejected():
    reg = Registry()
    reg.register(_make_skill("foo"))
    with pytest.raises(SkillValidationError):
        reg.register(_make_skill("foo", "0.2.0"))


def test_re_register_same_class_is_idempotent():
    reg = Registry()
    cls = _make_skill("foo")
    reg.register(cls)
    reg.register(cls)
    assert len(reg) == 1


def test_get_unknown_skill_raises():
    reg = Registry()
    from daedalus.core.errors import SkillNotFoundError

    with pytest.raises(SkillNotFoundError):
        reg.get("nope")


def test_version_constraint_enforced():
    reg = Registry()
    reg.register(_make_skill("foo", "0.3.5"))
    reg.get("foo", "^0.3.0")
    with pytest.raises(SkillValidationError):
        reg.get("foo", "^1.0.0")


def test_registered_class_must_have_pydantic_io():
    reg = Registry()

    class BadInputs:  # not a Pydantic model
        pass

    cls = type(
        "BadSkill",
        (AtomicSkill,),
        {
            "SPEC": _spec("bad"),
            "Inputs": BadInputs,
            "Outputs": _IO,
            "run": lambda self, i, ctx: _IO(),
        },
    )
    with pytest.raises(SkillValidationError):
        reg.register(cls)
