"""Error hierarchy. Keep narrow and stable; the executor and UI key off them."""

from __future__ import annotations


class DaedalusError(Exception):
    """Root exception for everything raised by the agent."""


class SkillNotFoundError(DaedalusError):
    """Raised when a program references a skill id that is not in the registry."""


class SkillValidationError(DaedalusError):
    """Raised when a skill's spec.yaml and its Python class disagree, or when
    inputs/outputs fail Pydantic validation."""


class PreconditionError(DaedalusError):
    """A declared precondition for a skill was not satisfied at call time."""


class PostconditionError(DaedalusError):
    """A skill returned but a declared postcondition was not satisfied."""


class BackendError(DaedalusError):
    """The remote-desktop backend failed (connection lost, refused, etc.)."""


class ProgramValidationError(DaedalusError):
    """The DSL program failed structural validation."""


class UserAbortError(DaedalusError):
    """The user aborted execution (overlay hotkey, Ctrl-C in the confirm prompt, etc.)."""


class TimeoutError(DaedalusError):
    """A skill or step exceeded its configured timeout."""
