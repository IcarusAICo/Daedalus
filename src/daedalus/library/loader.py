"""Load on-disk skills into the global registry.

Each ``skills/<skill_id>/skill.py`` module is loaded in isolation and is
expected to register exactly one :class:`Skill` subclass via the
:func:`daedalus.core.register` decorator. The loader cross-checks the in-code
``SPEC`` against ``spec.yaml`` so neither side can drift unnoticed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from daedalus.core.errors import SkillValidationError
from daedalus.core.registry import RegisteredSkill, Registry, get_registry, use_registry
from daedalus.core.spec import SkillSpec

log = logging.getLogger(__name__)


class LoaderError(SkillValidationError):
    """Raised by the library loader for filesystem-level problems."""


def _compute_skill_hash(skill_dir: Path) -> str:
    """SHA256 over spec.yaml + skill.py + sorted test fixtures."""
    h = hashlib.sha256()
    for name in ["spec.yaml", "skill.py"]:
        p = skill_dir / name
        if p.exists():
            h.update(p.read_bytes())
    tests_dir = skill_dir / "tests"
    if tests_dir.is_dir():
        for tf in sorted(tests_dir.glob("*.json")):
            h.update(tf.read_bytes())
    return h.hexdigest()


def _read_spec_yaml(path: Path) -> SkillSpec:
    if not path.exists():
        raise LoaderError(f"missing spec.yaml at {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise LoaderError(f"spec.yaml at {path} must be a mapping, got {type(raw).__name__}")
    try:
        return SkillSpec.from_dict(raw)
    except Exception as exc:
        raise LoaderError(f"invalid spec.yaml at {path}: {exc}") from exc


def _import_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise LoaderError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise LoaderError(f"importing {path} failed: {exc}") from exc
    return mod


def load_skill(skill_dir: Path, registry: Registry | None = None) -> str:
    """Load a single skill folder. Returns the registered skill id."""
    if registry is None:
        registry = get_registry()
    skill_dir = skill_dir.resolve()
    spec = _read_spec_yaml(skill_dir / "spec.yaml")

    module_name = f"daedalus._skills.{skill_dir.name}"
    before = set(registry._by_id.keys())  # type: ignore[attr-defined]
    # Route the @register decorator into the target registry while we import.
    with use_registry(registry):
        _import_module_from_path(module_name, skill_dir / "skill.py")
    after = set(registry._by_id.keys())  # type: ignore[attr-defined]
    new = after - before

    # The module should have registered exactly one skill, and that skill's id
    # must match the folder's spec.yaml.
    if not new:
        raise LoaderError(
            f"{skill_dir / 'skill.py'} did not register any skill (missing @register?)"
        )
    if len(new) > 1:
        raise LoaderError(
            f"{skill_dir / 'skill.py'} registered multiple skills: {sorted(new)}"
        )
    registered_id = next(iter(new))
    if registered_id != spec.id:
        raise LoaderError(
            f"{skill_dir.name}: spec.yaml id={spec.id!r} but skill.py registered {registered_id!r}"
        )

    # Also cross-check version & kind: the SPEC declared inside skill.py is
    # already what the registry stores; we ensure it matches the YAML.
    cls = registry.get(registered_id).cls
    yaml_v = spec.version.raw
    code_v = cls.SPEC.version.raw
    if yaml_v != code_v:
        raise LoaderError(
            f"{skill_dir.name}: spec.yaml version={yaml_v!r} != skill.py SPEC.version={code_v!r}"
        )
    if spec.kind != cls.SPEC.kind:
        raise LoaderError(
            f"{skill_dir.name}: spec.yaml kind={spec.kind!r} != skill.py SPEC.kind={cls.SPEC.kind!r}"
        )

    # Regenerate spec.yaml from the canonical Python SPEC.
    generated = cls.SPEC.to_yaml_dict()
    (skill_dir / "spec.yaml").write_text(
        yaml.safe_dump(generated, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )

    # Validate examples (warn, don't error — examples may be partial).
    for i, ex in enumerate(cls.SPEC.examples):
        if ex.expected is not None:
            try:
                cls.Outputs.model_validate(ex.expected)
            except Exception as exc:
                log.warning(
                    "%s: example[%d].expected failed Outputs validation: %s",
                    spec.id, i, exc,
                )

    content_hash = _compute_skill_hash(skill_dir)
    with registry._lock:
        old_entry = registry._by_id[registered_id]
        registry._by_id[registered_id] = RegisteredSkill(cls=old_entry.cls, content_hash=content_hash)

    log.debug("loaded skill %s v%s from %s", spec.id, spec.version.raw, skill_dir)
    return spec.id


def load_library(skills_root: Path, registry: Registry | None = None) -> list[str]:
    """Load every ``<skill_id>/`` subdirectory under ``skills_root``.

    Returns the list of registered skill ids in load order. Folders without a
    ``spec.yaml`` are skipped silently so the directory can also hold
    documentation, examples, etc.
    """
    if registry is None:
        registry = get_registry()
    skills_root = skills_root.resolve()
    if not skills_root.is_dir():
        raise LoaderError(f"skills directory not found: {skills_root}")

    ids: list[str] = []
    for child in sorted(skills_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "spec.yaml").exists():
            continue
        ids.append(load_skill(child, registry=registry))
    return ids


def load_core_skills(skills_root: Path | None = None) -> frozenset[str]:
    """Load the set of core skill IDs from CORE.yaml.

    Core skills are built-in primitives that cannot be archived or amended
    by the learner. If skills_root is None, we look relative to the project
    root (two levels up from this source file, then into ``skills/``).
    """
    if skills_root is None:
        skills_root = Path(__file__).resolve().parents[3] / "skills"

    core_yaml = skills_root / "CORE.yaml"
    if not core_yaml.exists():
        log.warning("CORE.yaml not found at %s; returning empty core set", core_yaml)
        return frozenset()

    data = yaml.safe_load(core_yaml.read_text())
    ids = data.get("core_skills", [])
    return frozenset(ids)
