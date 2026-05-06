"""Implementor: synthesizes new skills from specs.

Workflow
--------
1. Take a :class:`ImplementorRequest` (typically a ``MissingSkillSpec`` from
   the Planner, or a refactor proposal from the Learner).
2. Ask the LLM (role=``implementor``) to produce a ``spec.yaml``, a
   ``skill.py``, and at least one JSON test fixture.
3. Write everything to a sandbox folder.
4. Lint the generated code with a small AST safety-walker (no ``subprocess``,
   no ``os.system``, no undeclared network/filesystem access).
5. Load the sandbox skill into a fresh :class:`Registry` using
   :func:`daedalus.core.use_registry` and replay the included test fixtures
   against a :class:`MockBackend`.
6. Only on green does ``publish()`` copy the bundle into the on-disk skill
   library and register it in the global registry.
"""

from daedalus.implementor.implementor import (
    ImplementorError,
    ImplementorRequest,
    ImplementorResult,
    SkillBundle,
    SyntheticSkillImplementor,
)
from daedalus.implementor.safety import SafetyViolation, SafetyVisitor, lint_skill_source

__all__ = [
    "ImplementorError",
    "ImplementorRequest",
    "ImplementorResult",
    "SafetyViolation",
    "SafetyVisitor",
    "SkillBundle",
    "SyntheticSkillImplementor",
    "lint_skill_source",
]
