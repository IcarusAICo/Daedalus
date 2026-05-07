"""Service skill lifecycle.

A :class:`daedalus.core.ServiceSkill` has an expensive startup phase (loading
model weights, establishing connections) and stays warm for many queries. The
:class:`ServiceHandle` class owns one service's lifetime:

    1. ``start()`` — load resources
    2. ``query()`` — process input(s) against the warm service
    3. ``stop()`` — release resources

Unlike daemons (which push updates on their own schedule), services are
pull-based: nothing happens between queries.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daedalus.core.context import ExecutionContext
from daedalus.core.registry import Registry, get_registry

if TYPE_CHECKING:
    from daedalus.core.skill import ServiceSkill

log = logging.getLogger(__name__)


@dataclass
class ServiceSpec:
    """Reference + start inputs that the executor will spin up."""

    skill: str
    inputs: dict[str, Any] = field(default_factory=dict)
    version: str | None = None


class ServiceHandle:
    """Owns one running service instance."""

    def __init__(
        self,
        skill_cls: type["ServiceSkill"],
        inputs_obj: Any,
        ctx: ExecutionContext,
    ) -> None:
        self._skill_cls = skill_cls
        self._instance: "ServiceSkill" = skill_cls()
        self._inputs = inputs_obj
        self._ctx = ctx
        self._lock = threading.Lock()
        self._started = False
        self._query_count = 0
        self.error: BaseException | None = None

    @property
    def skill_id(self) -> str:
        return self._skill_cls.SPEC.id

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def query_count(self) -> int:
        return self._query_count

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            try:
                self._instance.start(self._inputs, self._ctx)
                self._started = True
                self._ctx.tracer.emit(
                    "service_started",
                    {"skill_id": self.skill_id},
                )
            except BaseException as exc:
                self.error = exc
                log.error("service %s failed to start: %s", self.skill_id, exc)
                raise

    def query(self, query_inputs: Any) -> Any:
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    f"service {self.skill_id} is not started; call start() first"
                )
        validated = self._skill_cls.QueryInputs.model_validate(
            query_inputs if isinstance(query_inputs, dict) else query_inputs.model_dump(mode="json")
        )
        result = self._instance.query(validated, self._ctx)
        self._query_count += 1
        self._ctx.tracer.emit(
            "service_query",
            {"skill_id": self.skill_id, "query_idx": self._query_count},
        )
        return result

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._instance.stop(self._ctx)
            except BaseException as exc:
                self.error = exc
                log.warning("service %s stop error: %s", self.skill_id, exc)
            finally:
                self._started = False
                self._ctx.tracer.emit(
                    "service_stopped",
                    {
                        "skill_id": self.skill_id,
                        "queries": self._query_count,
                        "error": str(self.error) if self.error else None,
                    },
                    level="error" if self.error else "info",
                )


# ---------------------------------------------------------------------------
# Lifecycle helpers used by the executor
# ---------------------------------------------------------------------------


def start_service(
    service_spec: ServiceSpec,
    ctx: ExecutionContext,
    *,
    registry: Registry | None = None,
) -> ServiceHandle:
    """Resolve, validate, and start a service. Returns a handle."""
    registry = registry if registry is not None else get_registry()
    entry = registry.get(service_spec.skill, version_constraint=service_spec.version)
    if entry.cls.SPEC.kind != "service":
        raise RuntimeError(
            f"{service_spec.skill} is registered as kind={entry.cls.SPEC.kind!r}, not service"
        )
    inputs = entry.cls.Inputs.model_validate(service_spec.inputs)
    handle = ServiceHandle(entry.cls, inputs, ctx)  # type: ignore[arg-type]
    handle.start()
    return handle


def stop_service(handle: ServiceHandle) -> None:
    """Stop a running service and release its resources."""
    handle.stop()
