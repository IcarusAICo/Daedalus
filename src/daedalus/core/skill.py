"""Skill base classes.

A skill is a typed unit of work the executor can call. There are two kinds:

- :class:`AtomicSkill` runs once with validated inputs and returns validated outputs.
- :class:`DaemonSkill` runs a long-lived async loop that publishes updates to
  the per-task ``TaskState`` until cancelled.

Subclasses MUST declare ``Inputs`` and ``Outputs`` (Pydantic models) and provide
a ``SPEC`` :class:`SkillSpec` instance describing the skill metadata. The
:func:`register` decorator wires the class into the global registry.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from daedalus.core.spec import SkillSpec

if TYPE_CHECKING:
    from daedalus.core.context import ExecutionContext


class SkillKind(enum.StrEnum):
    ATOMIC = "atomic"
    DAEMON = "daemon"


class Skill(ABC):
    """Base for all skills. Do not subclass directly; use Atomic or Daemon."""

    SPEC: ClassVar[SkillSpec]
    Inputs: ClassVar[type[BaseModel]]
    Outputs: ClassVar[type[BaseModel]]
    KIND: ClassVar[SkillKind]

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        return cls.Inputs.model_json_schema()

    @classmethod
    def output_schema(cls) -> dict[str, Any]:
        return cls.Outputs.model_json_schema()

    @classmethod
    def validate_class(cls) -> None:
        """Runtime invariants the registry checks at load time."""
        for required in ("SPEC", "Inputs", "Outputs", "KIND"):
            if not hasattr(cls, required):
                raise TypeError(
                    f"{cls.__name__} is missing required class attribute {required!r}"
                )
        if not isinstance(cls.SPEC, SkillSpec):
            raise TypeError(f"{cls.__name__}.SPEC must be a SkillSpec")
        if cls.SPEC.kind != cls.KIND.value:
            raise TypeError(
                f"{cls.__name__}.SPEC.kind={cls.SPEC.kind!r} disagrees with KIND={cls.KIND.value!r}"
            )
        if not (isinstance(cls.Inputs, type) and issubclass(cls.Inputs, BaseModel)):
            raise TypeError(f"{cls.__name__}.Inputs must be a Pydantic BaseModel subclass")
        if not (isinstance(cls.Outputs, type) and issubclass(cls.Outputs, BaseModel)):
            raise TypeError(f"{cls.__name__}.Outputs must be a Pydantic BaseModel subclass")


class AtomicSkill(Skill):
    """A skill that runs synchronously and returns once. The common case."""

    KIND: ClassVar[SkillKind] = SkillKind.ATOMIC

    @abstractmethod
    def run(self, inputs: BaseModel, ctx: ExecutionContext) -> BaseModel:
        """Execute the skill. ``inputs`` is already a validated ``self.Inputs``;
        the return value will be validated against ``self.Outputs``."""


class DaemonSkill(Skill):
    """A long-running skill that yields state updates until cancelled.

    The executor starts the loop on a worker task, forwards each yielded value
    into ``ctx.task_state[SPEC.publishes_state_key]``, and cancels the task
    when the program-step that owns the daemon finishes.
    """

    KIND: ClassVar[SkillKind] = SkillKind.DAEMON

    @abstractmethod
    async def loop(
        self, inputs: BaseModel, ctx: ExecutionContext
    ) -> AsyncIterator[BaseModel]:
        """Asynchronously yield ``self.Outputs`` instances. Should respect
        ``ctx.aborted()`` and clean up on cancellation."""
        if False:  # pragma: no cover - keeps signature an async generator
            yield  # type: ignore[unreachable]
