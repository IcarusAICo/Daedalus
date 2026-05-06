"""Global skill registry.

Skills register themselves via the :func:`register` decorator (or are added by
the on-disk loader). The registry enforces:

- unique ids
- no two registrations of the same id at different versions in the same process
  (you can have one version active at a time per id)
- ``Skill.validate_class()`` invariants

The planner and executor only ever see registered ids, so an LLM cannot
hallucinate a skill into existence.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from daedalus.core.errors import SkillNotFoundError, SkillValidationError
from daedalus.core.skill import Skill
from daedalus.core.spec import SkillVersion


@dataclass(frozen=True)
class RegisteredSkill:
    cls: type[Skill]
    content_hash: str = ""

    @property
    def id(self) -> str:
        return self.cls.SPEC.id

    @property
    def version(self) -> SkillVersion:
        return self.cls.SPEC.version


class Registry:
    """Container of registered skills. One per process by default."""

    def __init__(self) -> None:
        self._by_id: dict[str, RegisteredSkill] = {}
        self._lock = threading.RLock()

    def register(self, cls: type[Skill]) -> type[Skill]:
        try:
            cls.validate_class()
        except TypeError as exc:
            raise SkillValidationError(str(exc)) from exc

        sid = cls.SPEC.id
        with self._lock:
            existing = self._by_id.get(sid)
            if existing is not None and existing.cls is not cls:
                raise SkillValidationError(
                    f"skill id {sid!r} already registered by {existing.cls!r}; "
                    f"cannot replace with {cls!r}"
                )
            self._by_id[sid] = RegisteredSkill(cls=cls)
        return cls

    def get(self, skill_id: str, version_constraint: str | None = None) -> RegisteredSkill:
        with self._lock:
            entry = self._by_id.get(skill_id)
        if entry is None:
            raise SkillNotFoundError(skill_id)
        if version_constraint is not None and not entry.version.is_compatible_with(
            version_constraint
        ):
            raise SkillValidationError(
                f"skill {skill_id} v{entry.version.raw} does not satisfy {version_constraint}"
            )
        return entry

    def __contains__(self, skill_id: object) -> bool:
        if not isinstance(skill_id, str):
            return False
        with self._lock:
            return skill_id in self._by_id

    def __iter__(self) -> Iterator[RegisteredSkill]:
        with self._lock:
            return iter(list(self._by_id.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def clear(self) -> None:
        """Test helper. Drops all registered skills."""
        with self._lock:
            self._by_id.clear()


_GLOBAL_REGISTRY = Registry()
_THREAD_LOCAL = threading.local()


def _get_active_registry() -> Registry:
    return getattr(_THREAD_LOCAL, "registry", _GLOBAL_REGISTRY)


def get_registry() -> Registry:
    return _GLOBAL_REGISTRY


def get_active_registry() -> Registry:
    return _get_active_registry()


@contextmanager
def use_registry(registry: Registry):
    """Temporarily route ``@register`` writes to ``registry``.

    Uses thread-local storage so concurrent loader threads don't
    cross-contaminate.
    """
    old = _get_active_registry()
    _THREAD_LOCAL.registry = registry
    try:
        yield registry
    finally:
        _THREAD_LOCAL.registry = old


def register(cls: type[Skill]) -> type[Skill]:
    """Decorator: register ``cls`` in the active registry (global by default).

    Usage::

        @register
        class ClickMouse(AtomicSkill):
            ...
    """
    return _get_active_registry().register(cls)
